//! # Adg
//!
//! $$
//! dX_{i,t}=(k(t)-\theta(t)X_{i,t})dt+\sigma_i dW_{i,t},\quad
//! r_i(t)=\phi(t)+b(t)X_{i,t}+c(t)X_{i,t}^2
//! $$
//!
//! `xn` independent latent factors, each an affine/quadratic-Gaussian-style
//! diffusion with a time-dependent additive drift target `k(t)` and a
//! time-dependent mean-reversion speed `θ(t)`, observed through a
//! quadratic output map `phi(t) + b(t)X + c(t)X²`.
//!

use ndarray::Array1;
use ndarray::Array2;
#[cfg(feature = "python")]
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::Fn1D;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

pub struct Adg<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Time-dependent additive drift target function k(t) — the level each
  /// latent factor is pulled toward, analogous to Hull-White's θ(t).
  pub k: Fn1D<T>,
  /// Time-dependent mean-reversion speed function θ(t), multiplying the
  /// pull-back term `−θ(t)·X_t`. Despite the name, this is a speed, not
  /// a level — see `k` for the level.
  pub theta: Fn1D<T>,
  /// Per-factor diffusion scale vector σ_i, one entry per latent factor
  /// (length `xn`).
  pub sigma: Array1<T>,
  /// Time-dependent intercept φ(t) of the output observation equation
  /// `r = φ(t) + b(t)·X + c(t)·X²`.
  pub phi: Fn1D<T>,
  /// Time-dependent linear loading b(t) on the latent factor in the
  /// observation equation.
  pub b: Fn1D<T>,
  /// Time-dependent quadratic loading c(t) on the squared latent factor
  /// in the observation equation (the "quadratic" in quadratic-Gaussian).
  pub c: Fn1D<T>,
  /// Number of time steps per latent-factor path.
  pub n: usize,
  /// Number of independent latent factors (rows of the output matrix);
  /// each evolves as its own `dX_i = (k(t) − θ(t)X_i)dt + σ_i dW_i`.
  pub xn: usize,
  /// Initial values `X_i(0)` for each latent factor, length `xn`.
  pub x0: Array1<T>,
  /// Simulation horizon [0, t] shared by all `xn` latent-factor paths
  /// (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Adg<T, S> {
  pub fn new(
    k: impl Into<Fn1D<T>>,
    theta: impl Into<Fn1D<T>>,
    sigma: Array1<T>,
    phi: impl Into<Fn1D<T>>,
    b: impl Into<Fn1D<T>>,
    c: impl Into<Fn1D<T>>,
    n: usize,
    xn: usize,
    x0: Array1<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert_eq!(
      sigma.len(),
      xn,
      "sigma length ({}) must match xn ({})",
      sigma.len(),
      xn
    );
    assert_eq!(
      x0.len(),
      xn,
      "x0 length ({}) must match xn ({})",
      x0.len(),
      xn
    );
    Self {
      backend: Cpu,
      k: k.into(),
      theta: theta.into(),
      sigma,
      phi: phi.into(),
      b: b.into(),
      c: c.into(),
      n,
      xn,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Adg<T, S, B> {
  /// The grid spacing: `n - 1` increments over the horizon, zero when the
  /// grid has no increment to take.
  fn dt(&self) -> T {
    if self.n > 1 {
      self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
    } else {
      T::zero()
    }
  }

  /// One factor's row of the observed rate. The tail is pre-filled with
  /// `N(0, dt)` draws, the state `x` walks its mean reversion under `k(t)`
  /// and `theta(t)` consuming them, and each point is written back as the
  /// quadratic observation `phi(t) + b(t) x + c(t) x^2` at the same time.
  /// Shared by the matrix sampler and the row view so the two cannot drift.
  fn fill_row<S2: SeedExt>(&self, row: &mut [T], factor: usize, seed: &S2) {
    if row.is_empty() {
      return;
    }
    let dt = self.dt();
    let observe = |t: T, x: T| self.phi.call(t) + self.b.call(t) * x + self.c.call(t) * x * x;
    let mut x = self.x0[factor];
    row[0] = observe(T::zero(), x);
    if row.len() == 1 {
      return;
    }
    let normal = SimdNormal::<T>::new(T::zero(), dt.sqrt(), seed);
    normal.fill_slice(&mut row[1..]);
    for j in 1..row.len() {
      let t = T::from_usize_(j) * dt;
      x = x + (self.k.call(t) - self.theta.call(t) * x) * dt + self.sigma[factor] * row[j];
      row[j] = observe(t, x);
    }
  }
}

/// One factor of the model, stepped on its own. `Adg` reports every factor
/// as a row of one matrix and the engine speaks in single paths, so this view
/// is what carries a launch: factor `i` under its own diffusion scale, started
/// at its own level, with the five time-varying coefficients tabulated once
/// for all of them. It borrows rather than owns, so the seed it advances is
/// the process's own.
#[doc(hidden)]
pub struct AdgRow<'a, T: FloatExt, S: SeedExt, B>(&'a Adg<T, S, B>, usize);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for AdgRow<'_, T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = AdgRowSampler<'s, T, S, B>
  where
    Self: 's;

  fn sampler(&self) -> AdgRowSampler<'_, T, S, B> {
    AdgRowSampler {
      adg: self.0,
      factor: self.1,
      seed: self.0.seed.derive(),
    }
  }
}

/// [`AdgRow`]'s sampler: one factor's recursion from a seed derived once.
#[doc(hidden)]
pub struct AdgRowSampler<'a, T: FloatExt, S: SeedExt, B> {
  adg: &'a Adg<T, S, B>,
  factor: usize,
  seed: S,
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> PathSampler<T>
  for AdgRowSampler<'_, T, S, B>
{
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let row = out
      .as_slice_mut()
      .expect("Adg row must be contiguous in memory");
    self.adg.fill_row(row, self.factor, &self.seed);
  }

  fn sample(&mut self) -> Array1<T> {
    let mut out = Array1::<T>::zeros(self.adg.n);
    self.sample_into(&mut out);
    out
  }
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for AdgRow<'_, T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::AffineDiffusionGaussian {
      sigma: self.0.sigma[self.1],
    }
  }

  fn initial_value(&self) -> T {
    self.0.x0[self.1]
  }

  fn grid_points(&self) -> usize {
    self.0.n
  }

  fn horizon(&self) -> T {
    self.0.t.unwrap_or(T::one())
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.0.seed)
  }

  /// `k`, `theta`, `phi`, `b` and `c` at each grid point, in the order the
  /// step names them as `ct` through `ct4`. Step `j` and the observation
  /// written at it both read `j · dt`, which is what the host evaluates.
  fn curves(&self) -> Option<Vec<Vec<T>>> {
    let adg = self.0;
    let dt = adg.dt();
    let grid = |f: &dyn Fn(T) -> T| (0..adg.n).map(|j| f(T::from_usize_(j) * dt)).collect();
    Some(vec![
      grid(&|t| adg.k.call(t)),
      grid(&|t| adg.theta.call(t)),
      grid(&|t| adg.phi.call(t)),
      grid(&|t| adg.b.call(t)),
      grid(&|t| adg.c.call(t)),
    ])
  }

  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Adg<T, S> { k, theta, sigma, phi, b, c, n, xn, x0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Adg<T, S, B> {
  type Output = Array2<T>;
  type Sampler<'s>
    = AdgSampler<'s, T, S, B>
  where
    Self: 's;

  /// Derives a seed once, at construction, for [`AdgSampler`] to own.
  /// Deriving (not cloning) is what decorrelates chunks: the derived value
  /// is `self.seed`'s *mixed* next tick, not a raw snapshot, so chunk `i`'s
  /// basis and chunk `i+1`'s basis are hash-scrambled relative to each
  /// other rather than one raw stride apart. The drift, level and
  /// observation maps are user-supplied [`Fn1D`] callables (not clonable,
  /// since the Python variant holds a `pyo3::Py`) so there is nothing else
  /// reusable to hoist across calls beyond the borrowed process itself.
  fn sampler(&self) -> AdgSampler<'_, T, S, B> {
    AdgSampler {
      adg: self,
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine, one launch per factor: the factors are
  /// independent, so a device steps each one's whole batch in one kernel
  /// under the tabulated coefficients and the matrix is assembled here. On
  /// the host devices each factor runs the same row recursion the matrix
  /// sampler does.
  fn sample(&self) -> Array2<T> {
    let mut out = Array2::<T>::zeros((self.xn, self.n));
    for i in 0..self.xn {
      out
        .row_mut(i)
        .assign(&self.backend.euler_sample(&AdgRow(self, i)));
    }
    out
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array2<T>) -> R + Sync) -> Vec<R> {
    self.sample_par(m).iter().map(f).collect()
  }

  fn sample_par(&self, m: usize) -> Vec<Array2<T>> {
    let mut out: Vec<Array2<T>> = (0..m).map(|_| Array2::zeros((self.xn, self.n))).collect();
    for i in 0..self.xn {
      let rows = self.backend.euler_paths(&AdgRow(self, i), m);
      for (matrix, row) in out.iter_mut().zip(rows) {
        matrix.row_mut(i).assign(&row);
      }
    }
    out
  }

  fn try_sample(&self) -> Result<Array2<T>, crate::device::DeviceError> {
    let mut out = Array2::<T>::zeros((self.xn, self.n));
    for i in 0..self.xn {
      out
        .row_mut(i)
        .assign(&self.backend.try_sample(&AdgRow(self, i))?);
    }
    Ok(out)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array2<T>>, crate::device::DeviceError> {
    let mut out: Vec<Array2<T>> = (0..m).map(|_| Array2::zeros((self.xn, self.n))).collect();
    for i in 0..self.xn {
      let rows = self.backend.try_euler_paths(&AdgRow(self, i), m)?;
      for (matrix, row) in out.iter_mut().zip(rows) {
        matrix.row_mut(i).assign(&row);
      }
    }
    Ok(out)
  }
}

/// Reusable [`Adg`] sampler: borrows the process and owns a seed derived
/// once at construction. Each row's Gaussian increments are generated
/// inside the row recursion from that owned seed.
#[doc(hidden)]
pub struct AdgSampler<'a, T: FloatExt, S: SeedExt, B> {
  adg: &'a Adg<T, S, B>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> PathSampler<T>
  for AdgSampler<'_, T, S, B>
{
  type Output = Array2<T>;

  fn sample_into(&mut self, out: &mut Array2<T>) {
    for i in 0..self.adg.xn {
      let mut row = out.row_mut(i);
      let row = row
        .as_slice_mut()
        .expect("Adg state row must be contiguous in memory");
      self.adg.fill_row(row, i, &self.seed);
    }
  }

  fn sample(&mut self) -> Array2<T> {
    let mut out = Array2::<T>::zeros((self.adg.xn, self.adg.n));
    self.sample_into(&mut out);
    out
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyAdg {
  inner: Option<Adg<f64>>,
  seeded: Option<Adg<f64, crate::simd_rng::Deterministic>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyAdg {
  #[new]
  #[pyo3(signature = (k, theta, sigma, phi, b, c, n, xn, x0, t=None, seed=None))]
  fn new(
    k: pyo3::Py<pyo3::PyAny>,
    theta: pyo3::Py<pyo3::PyAny>,
    sigma: Vec<f64>,
    phi: pyo3::Py<pyo3::PyAny>,
    b: pyo3::Py<pyo3::PyAny>,
    c: pyo3::Py<pyo3::PyAny>,
    n: usize,
    xn: usize,
    x0: Vec<f64>,
    t: Option<f64>,
    seed: Option<u64>,
  ) -> Self {
    match seed {
      Some(s) => Self {
        inner: None,
        seeded: Some(Adg::new(
          Fn1D::Py(k),
          Fn1D::Py(theta),
          ndarray::Array1::from_vec(sigma),
          Fn1D::Py(phi),
          Fn1D::Py(b),
          Fn1D::Py(c),
          n,
          xn,
          ndarray::Array1::from_vec(x0),
          t,
          Deterministic::new(s),
        )),
      },
      None => Self {
        inner: Some(Adg::new(
          Fn1D::Py(k),
          Fn1D::Py(theta),
          ndarray::Array1::from_vec(sigma),
          Fn1D::Py(phi),
          Fn1D::Py(b),
          Fn1D::Py(c),
          n,
          xn,
          ndarray::Array1::from_vec(x0),
          t,
          Unseeded,
        )),
        seeded: None,
      },
    }
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_dispatch_f64!(self, |inner| inner
      .sample()
      .into_pyarray(py)
      .into_py_any(py)
      .unwrap())
  }

  /// `m` independent paths stacked into an `(m, factors, n)` array. The
  /// GIL is released while the paths are generated; the Python callables
  /// re-acquire it per evaluation.
  fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize) -> pyo3::Py<pyo3::PyAny> {
    use ndarray::Array3;
    use ndarray::Axis;
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_dispatch_f64!(self, |inner| {
      let paths = py.detach(|| inner.sample_par(m));
      let (rows, cols) = paths.first().map_or((0, 0), |p| p.dim());
      let mut result = Array3::zeros((m, rows, cols));
      for (i, path) in paths.iter().enumerate() {
        result.index_axis_mut(Axis(0), i).assign(path);
      }
      result.into_pyarray(py).into_py_any(py).unwrap()
    })
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;

  use super::*;

  fn const_one(_t: f64) -> f64 {
    1.0
  }

  fn const_zero(_t: f64) -> f64 {
    0.0
  }

  #[test]
  fn sample_runs() {
    let xn = 2;
    let sigma = Array1::<f64>::from_vec(vec![0.01, 0.01]);
    let x0 = Array1::<f64>::from_vec(vec![0.05, 0.05]);
    let adg = Adg::<f64>::new(
      const_one as fn(f64) -> f64,
      const_one as fn(f64) -> f64,
      sigma,
      const_zero as fn(f64) -> f64,
      const_one as fn(f64) -> f64,
      const_zero as fn(f64) -> f64,
      32,
      xn,
      x0,
      Some(1.0),
      Unseeded,
    );
    let path = adg.sample();
    assert_eq!(path.nrows(), xn);
    assert_eq!(path.ncols(), 32);
  }
}

#[cfg(test)]
mod tabulation {
  use ndarray::array;
  use stochastic_rs_core::simd_rng::Unseeded;

  use super::Adg;
  use super::AdgRow;

  /// The five coefficients are tabulated at `j · dt`, the time both the step
  /// and the observation written at it read on the host.
  #[test]
  fn the_curves_are_the_host_coefficients_at_the_host_times() {
    fn k(t: f64) -> f64 {
      0.01 + 0.02 * t
    }
    fn theta(t: f64) -> f64 {
      0.5 + 0.2 * t
    }
    fn phi(t: f64) -> f64 {
      0.002 + 0.01 * t
    }
    fn b(t: f64) -> f64 {
      0.8 + 0.4 * t
    }
    fn c(t: f64) -> f64 {
      2.0 + t
    }
    let adg = Adg::<f64>::new(
      k as fn(f64) -> f64,
      theta as fn(f64) -> f64,
      array![0.01, 0.02],
      phi as fn(f64) -> f64,
      b as fn(f64) -> f64,
      c as fn(f64) -> f64,
      33,
      2,
      array![0.02, 0.03],
      Some(1.0),
      Unseeded,
    );
    let row = AdgRow(&adg, 1);
    let curves = crate::euler::EulerCoefficients::curves(&row).expect("five curves");
    assert_eq!(curves.len(), 5);
    let dt = 1.0 / 32.0;
    for j in 0..33 {
      let t = j as f64 * dt;
      assert_eq!(curves[0][j], k(t), "k at step {j}");
      assert_eq!(curves[1][j], theta(t), "theta at step {j}");
      assert_eq!(curves[2][j], phi(t), "phi at step {j}");
      assert_eq!(curves[3][j], b(t), "b at step {j}");
      assert_eq!(curves[4][j], c(t), "c at step {j}");
    }
  }
}

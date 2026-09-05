//! # Wu Zhang
//!
//! $$
//! dv_{i,t}=\beta_i(\alpha_i-v_{i,t})dt+\nu_i\sqrt{v_{i,t}}\,dW_{i,t}^v,\quad
//! dF_{i,t}=\lambda_i F_{i,t}\sqrt{v_{i,t}}\,dW_{i,t}^F
//! $$
//!
//! `xn` independent CEV/SABR-style forward-volatility pairs: each
//! dimension's volatility `v_i` is a CIR-style square-root diffusion, and
//! its forward `F_i` is driven multiplicatively by `v_i`'s own path
//! (no cross-dimension coupling, no `dW^v`/`dW^F` correlation).
//!

use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;
#[cfg(feature = "python")]
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

pub struct WuZhangD<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Mean reversion level for each dimension's volatility.
  pub alpha: Array1<T>,
  /// Mean reversion speed for each dimension's volatility.
  pub beta: Array1<T>,
  /// Volatility of volatility for each dimension.
  pub nu: Array1<T>,
  /// Parameter controlling the impact of volatility on the forward rate.
  pub lambda: Array1<T>,
  /// Initial forward rates for each dimension.
  pub x0: Array1<T>,
  /// Initial volatilities for each dimension.
  pub v0: Array1<T>,
  /// Number of (rate, vol) pairs.
  pub xn: usize,
  /// Total time horizon.
  pub t: Option<T>,
  /// Number of time steps in the simulation.
  pub n: usize,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> WuZhangD<T, S> {
  pub fn new(
    alpha: Array1<T>,
    beta: Array1<T>,
    nu: Array1<T>,
    lambda: Array1<T>,
    x0: Array1<T>,
    v0: Array1<T>,
    xn: usize,
    t: Option<T>,
    n: usize,
    seed: S,
  ) -> Self {
    assert_eq!(
      alpha.len(),
      xn,
      "alpha length ({}) must match xn ({})",
      alpha.len(),
      xn
    );
    assert_eq!(
      beta.len(),
      xn,
      "beta length ({}) must match xn ({})",
      beta.len(),
      xn
    );
    assert_eq!(
      nu.len(),
      xn,
      "nu length ({}) must match xn ({})",
      nu.len(),
      xn
    );
    assert_eq!(
      lambda.len(),
      xn,
      "lambda length ({}) must match xn ({})",
      lambda.len(),
      xn
    );
    assert_eq!(
      x0.len(),
      xn,
      "x0 length ({}) must match xn ({})",
      x0.len(),
      xn
    );
    assert_eq!(
      v0.len(),
      xn,
      "v0 length ({}) must match xn ({})",
      v0.len(),
      xn
    );
    assert!(
      alpha.iter().all(|&x| x >= T::zero()),
      "alpha entries must be non-negative"
    );
    assert!(
      beta.iter().all(|&x| x >= T::zero()),
      "beta entries must be non-negative"
    );
    assert!(
      nu.iter().all(|&x| x >= T::zero()),
      "nu entries must be non-negative"
    );
    assert!(
      v0.iter().all(|&x| x >= T::zero()),
      "v0 entries must be non-negative"
    );
    Self {
      backend: Cpu,
      alpha,
      beta,
      nu,
      lambda,
      x0,
      v0,
      xn,
      t,
      n,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> WuZhangD<T, S, B> {
  /// The grid spacing: `n - 1` increments over the horizon, zero when the
  /// grid has no increment to take.
  fn dt(&self) -> T {
    if self.n > 1 {
      self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
    } else {
      T::zero()
    }
  }
}

/// One pair's recursion. Both tails are pre-filled with `N(0, dt)` draws —
/// the forward rate's first, then the variance's — and each step floors the
/// variance, mean-reverts it under its square-root diffusion, floors it
/// again, and moves the floored rate by `lambda` times the square root of
/// the variance just stepped. Shared by the matrix sampler and the pair view
/// so the two cannot drift.
fn fill_wu_zhang_pair<T: FloatExt, S: SeedExt>(
  f: &mut [T],
  v: &mut [T],
  x0: T,
  v0: T,
  alpha: T,
  beta: T,
  nu: T,
  lambda: T,
  dt: T,
  seed: &S,
) {
  if f.is_empty() {
    return;
  }
  f[0] = x0;
  v[0] = v0;
  if f.len() == 1 {
    return;
  }
  let sqrt_dt = dt.sqrt();
  SimdNormal::<T>::new(T::zero(), sqrt_dt, seed).fill_slice(&mut f[1..]);
  SimdNormal::<T>::new(T::zero(), sqrt_dt, seed).fill_slice(&mut v[1..]);
  for j in 1..f.len() {
    let v_old = v[j - 1].max(T::zero());
    let f_old = f[j - 1].max(T::zero());
    let (d_w_v, d_w_f) = (v[j], f[j]);
    let dv = beta * (alpha - v_old) * dt + nu * v_old.sqrt() * d_w_v;
    let v_new = (v_old + dv).max(T::zero());
    v[j] = v_new;
    f[j] = f_old + f_old * lambda * v_new.sqrt() * d_w_f;
  }
}

/// One forward-rate / variance pair of the model, stepped on its own.
/// `WuZhangD` reports every pair as two rows of one matrix — the rates first,
/// then the variances — and the engine speaks in `[Array1<T>; 2]`, so this
/// view is what carries a launch: pair `i` under its own four scalars,
/// started at its own levels. It borrows rather than owns, so the seed it
/// advances is the process's own.
#[doc(hidden)]
pub struct WuZhangPair<'a, T: FloatExt, S: SeedExt, B>(&'a WuZhangD<T, S, B>, usize);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for WuZhangPair<'_, T, S, B>
{
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = WuZhangPairSampler<'s, T, S, B>
  where
    Self: 's;

  fn sampler(&self) -> WuZhangPairSampler<'_, T, S, B> {
    WuZhangPairSampler {
      model: self.0,
      pair: self.1,
      seed: self.0.seed.derive(),
    }
  }
}

/// [`WuZhangPair`]'s sampler: one pair's recursion from a seed derived once.
#[doc(hidden)]
pub struct WuZhangPairSampler<'a, T: FloatExt, S: SeedExt, B> {
  model: &'a WuZhangD<T, S, B>,
  pair: usize,
  seed: S,
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> PathSampler<T>
  for WuZhangPairSampler<'_, T, S, B>
{
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let m = self.model;
    let i = self.pair;
    let [f, v] = out;
    fill_wu_zhang_pair(
      f.as_slice_mut()
        .expect("WuZhang forward row must be contiguous in memory"),
      v.as_slice_mut()
        .expect("WuZhang volatility row must be contiguous in memory"),
      m.x0[i],
      m.v0[i],
      m.alpha[i],
      m.beta[i],
      m.nu[i],
      m.lambda[i],
      m.dt(),
      &self.seed,
    );
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    let mut out = [
      Array1::<T>::zeros(self.model.n),
      Array1::<T>::zeros(self.model.n),
    ];
    self.sample_into(&mut out);
    out
  }
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 2>
  for WuZhangPair<'_, T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let (m, i) = (self.0, self.1);
    crate::euler::EulerSpec::WuZhang {
      alpha: m.alpha[i],
      beta: m.beta[i],
      nu: m.nu[i],
      lambda: m.lambda[i],
    }
  }

  fn initial_state(&self) -> [T; 4] {
    [self.0.x0[self.1], self.0.v0[self.1], T::zero(), T::zero()]
  }

  fn grid_points(&self) -> usize {
    self.0.n
  }

  fn horizon(&self) -> T {
    self.0.t.unwrap_or(T::one())
  }

  fn time_step(&self) -> T {
    self.0.dt()
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.0.seed)
  }

  fn host_sample(&self) -> [Array1<T>; 2] {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] WuZhangD<T, S> { alpha, beta, nu, lambda, x0, v0, xn, t, n, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for WuZhangD<T, S, B>
{
  type Output = Array2<T>;
  type Sampler<'s>
    = WuZhangDSampler<T, S>
  where
    Self: 's;

  /// Derives (not clones) `self.seed` into the returned sampler: the
  /// derived value is `self.seed`'s *mixed* next tick, not a raw snapshot,
  /// so chunk `i`'s basis and chunk `i+1`'s basis are hash-scrambled
  /// relative to each other rather than one raw stride apart.
  fn sampler(&self) -> WuZhangDSampler<T, S> {
    WuZhangDSampler {
      alpha: self.alpha.clone(),
      beta: self.beta.clone(),
      nu: self.nu.clone(),
      lambda: self.lambda.clone(),
      x0: self.x0.clone(),
      v0: self.v0.clone(),
      xn: self.xn,
      n: self.n,
      t: self.t,
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine, one launch per pair: the pairs are
  /// independent, so a device steps each one's whole batch in one kernel and
  /// the matrix — rates in the first `xn` rows, variances in the next — is
  /// assembled here. On the host devices each pair runs the same recursion
  /// the matrix sampler does.
  fn sample(&self) -> Array2<T> {
    let mut out = Array2::<T>::zeros((2 * self.xn, self.n));
    for i in 0..self.xn {
      let [f, v] = self.backend.system_sample(&WuZhangPair(self, i));
      out.row_mut(i).assign(&f);
      out.row_mut(self.xn + i).assign(&v);
    }
    out
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array2<T>) -> R + Sync) -> Vec<R> {
    self.sample_par(m).iter().map(f).collect()
  }

  fn sample_par(&self, m: usize) -> Vec<Array2<T>> {
    let mut out: Vec<Array2<T>> = (0..m)
      .map(|_| Array2::zeros((2 * self.xn, self.n)))
      .collect();
    for i in 0..self.xn {
      let pairs = self.backend.system_paths(&WuZhangPair(self, i), m);
      for (matrix, [f, v]) in out.iter_mut().zip(pairs) {
        matrix.row_mut(i).assign(&f);
        matrix.row_mut(self.xn + i).assign(&v);
      }
    }
    out
  }

  fn try_sample(&self) -> Result<Array2<T>, crate::device::DeviceError> {
    let mut out = Array2::<T>::zeros((2 * self.xn, self.n));
    for i in 0..self.xn {
      let [f, v] = self.backend.try_system_sample(&WuZhangPair(self, i))?;
      out.row_mut(i).assign(&f);
      out.row_mut(self.xn + i).assign(&v);
    }
    Ok(out)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array2<T>>, crate::device::DeviceError> {
    let mut out: Vec<Array2<T>> = (0..m)
      .map(|_| Array2::zeros((2 * self.xn, self.n)))
      .collect();
    for i in 0..self.xn {
      let pairs = self.backend.try_system_paths(&WuZhangPair(self, i), m)?;
      for (matrix, [f, v]) in out.iter_mut().zip(pairs) {
        matrix.row_mut(i).assign(&f);
        matrix.row_mut(self.xn + i).assign(&v);
      }
    }
    Ok(out)
  }
}

/// Reusable [`WuZhangD`] sampling state: owns the per-pair parameter vectors,
/// the initial curves and the seed source so a Monte-Carlo loop reuses the
/// output matrix.
#[doc(hidden)]
pub struct WuZhangDSampler<T: FloatExt, S: SeedExt> {
  alpha: Array1<T>,
  beta: Array1<T>,
  nu: Array1<T>,
  lambda: Array1<T>,
  x0: Array1<T>,
  v0: Array1<T>,
  xn: usize,
  n: usize,
  t: Option<T>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> WuZhangDSampler<T, S> {
  fn fill_matrix(&mut self, fv: &mut Array2<T>) {
    let dt = if self.n > 1 {
      self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
    } else {
      T::zero()
    };
    let (mut f_rows, mut v_rows) = fv.view_mut().split_at(Axis(0), self.xn);
    for i in 0..self.xn {
      let mut f_row = f_rows.row_mut(i);
      let mut v_row = v_rows.row_mut(i);
      fill_wu_zhang_pair(
        f_row
          .as_slice_mut()
          .expect("WuZhang forward row must be contiguous in memory"),
        v_row
          .as_slice_mut()
          .expect("WuZhang volatility row must be contiguous in memory"),
        self.x0[i],
        self.v0[i],
        self.alpha[i],
        self.beta[i],
        self.nu[i],
        self.lambda[i],
        dt,
        &self.seed,
      );
    }
  }
}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for WuZhangDSampler<T, S> {
  type Output = Array2<T>;

  fn sample_into(&mut self, out: &mut Array2<T>) {
    self.fill_matrix(out);
  }

  fn sample(&mut self) -> Array2<T> {
    let mut fv = Array2::<T>::zeros((2 * self.xn, self.n));
    self.fill_matrix(&mut fv);
    fv
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyWuZhangD {
  inner_f32: Option<WuZhangD<f32>>,
  inner_f64: Option<WuZhangD<f64>>,
  seeded_f32: Option<WuZhangD<f32, crate::simd_rng::Deterministic>>,
  seeded_f64: Option<WuZhangD<f64, crate::simd_rng::Deterministic>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyWuZhangD {
  #[new]
  #[pyo3(signature = (alpha, beta, nu, lambda_, x0, v0, xn, n, t=None, seed=None, dtype=None))]
  fn new(
    alpha: Vec<f64>,
    beta: Vec<f64>,
    nu: Vec<f64>,
    lambda_: Vec<f64>,
    x0: Vec<f64>,
    v0: Vec<f64>,
    xn: usize,
    n: usize,
    t: Option<f64>,
    seed: Option<u64>,
    dtype: Option<&str>,
  ) -> Self {
    match (seed, dtype.unwrap_or("f64")) {
      (Some(s), "f32") => {
        let to_f32_arr =
          |v: Vec<f64>| ndarray::Array1::from_vec(v.iter().map(|&x| x as f32).collect());
        Self {
          inner_f32: None,
          inner_f64: None,
          seeded_f32: Some(WuZhangD::new(
            to_f32_arr(alpha),
            to_f32_arr(beta),
            to_f32_arr(nu),
            to_f32_arr(lambda_),
            to_f32_arr(x0),
            to_f32_arr(v0),
            xn,
            t.map(|v| v as f32),
            n,
            Deterministic::new(s),
          )),
          seeded_f64: None,
        }
      }
      (Some(s), _) => {
        let to_arr = |v: Vec<f64>| ndarray::Array1::from_vec(v);
        Self {
          inner_f32: None,
          inner_f64: None,
          seeded_f32: None,
          seeded_f64: Some(WuZhangD::new(
            to_arr(alpha),
            to_arr(beta),
            to_arr(nu),
            to_arr(lambda_),
            to_arr(x0),
            to_arr(v0),
            xn,
            t,
            n,
            Deterministic::new(s),
          )),
        }
      }
      (None, "f32") => {
        let to_f32_arr =
          |v: Vec<f64>| ndarray::Array1::from_vec(v.iter().map(|&x| x as f32).collect());
        Self {
          inner_f32: Some(WuZhangD::new(
            to_f32_arr(alpha),
            to_f32_arr(beta),
            to_f32_arr(nu),
            to_f32_arr(lambda_),
            to_f32_arr(x0),
            to_f32_arr(v0),
            xn,
            t.map(|v| v as f32),
            n,
            Unseeded,
          )),
          inner_f64: None,
          seeded_f32: None,
          seeded_f64: None,
        }
      }
      (None, _) => {
        let to_arr = |v: Vec<f64>| ndarray::Array1::from_vec(v);
        Self {
          inner_f32: None,
          inner_f64: Some(WuZhangD::new(
            to_arr(alpha),
            to_arr(beta),
            to_arr(nu),
            to_arr(lambda_),
            to_arr(x0),
            to_arr(v0),
            xn,
            t,
            n,
            Unseeded,
          )),
          seeded_f32: None,
          seeded_f64: None,
        }
      }
    }
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_dispatch!(self, |inner| inner
      .sample()
      .into_pyarray(py)
      .into_py_any(py)
      .unwrap())
  }

  fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_dispatch!(self, |inner| {
      let samples = inner.sample_par(m);
      pyo3::types::PyList::new(
        py,
        samples
          .iter()
          .map(|s| s.clone().into_pyarray(py).into_py_any(py).unwrap()),
      )
      .unwrap()
      .into_py_any(py)
      .unwrap()
    })
  }
}

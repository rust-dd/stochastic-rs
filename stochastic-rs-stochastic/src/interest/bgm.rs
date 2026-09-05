//! # Bgm
//!
//! $$
//! L_i(t+dt) = L_i(t)\,\bigl(1 + \lambda_i\,\Delta W_t^{(i)}\bigr),\qquad
//! \Delta W_t^{(i)}\sim \mathcal N(0,\,\Delta t),\quad
//! W^{(i)}\perp W^{(j)}\ \text{for}\ i\ne j
//! $$
//!
//! **⚠️ Scope warning — this is NOT a BGM / LIBOR Market Model in the standard
//! sense, and the per-step recurrence is NOT exact log-normal evolution.**
//! Despite the type name, the implementation samples `xn` **independent**
//! per-rate paths by **forward-Euler discretization** of the formal SDE
//! `dL_i = λ_i L_i dW^{(i)}`:
//!
//! - Each rate `L_i` is driven by its **own independent** Brownian motion
//!   `W^{(i)}` (separate `SimdNormal::from_seed_source` call per row inside
//!   the [`ProcessExt::sample`](crate::traits::ProcessExt) impl).
//! - There is **no tenor / accrual-period structure** (no `δ_j`, no payment
//!   dates), so concepts like "the j-th forward measure" or
//!   "spot-LIBOR measure" do not even apply here.
//! - The drift coupling `µ_i = −σ_i Σ_{j>i} (τ_j δ_j σ_j L_j)/(1+δ_j L_j)`
//!   that defines BGM/LMM under a common measure is **not** present; nor is
//!   it conceptually meaningful for the current type, since there is no
//!   tenor structure to derive it from.
//! - The recurrence `L(t+dt) = L(t)(1 + λ·ΔW)` is a **discrete-time
//!   martingale by construction** (`E[L(t+dt)|L(t)] = L(t)` since
//!   `E[ΔW] = 0`), but it is **not** a log-normal sample — the marginal
//!   distribution of `L(t)` is not log-normal at any finite `dt`. The exact
//!   log-normal evolution `L(t+dt) = L(t)·exp(−½ λ² dt + λ ΔW)` is **not**
//!   used. In particular, paths can become **negative** when
//!   `λ·ΔW < −1` (a non-trivial event whenever `λ √dt` is large compared to
//!   one), so the impl differs qualitatively from a geometric Brownian
//!   motion. Only in the limit `dt → 0` does the law converge to log-normal.
//!
//! Suitable for:
//!
//! - **Single-path** Monte-Carlo where you only care about the (Euler-biased)
//!   marginal of one `L_i` and `λ_i √dt ≪ 1`;
//! - **Sanity / smoke testing** that consumes a matrix-shaped output of
//!   `(xn, n)` rate-like paths;
//! - **Demoware / teaching examples** illustrating Euler-Maruyama on a
//!   driftless multiplicative SDE.
//!
//! **NOT suitable for:**
//!
//! - Caplet / floorlet calibration (no tenor structure, no measure framework,
//!   Euler-vs-exact-lognormal bias);
//! - Swaption / Bermudan swaption pricing (requires the joint distribution
//!   of multiple correlated rates under a common measure);
//! - Any product whose payoff depends on cross-rate dependence, since the
//!   rates here are statistically **independent**;
//! - Any path-sensitive product where negative `L` values would be ill-defined
//!   (e.g., direct exponentiation, Black-formula payoffs on the path).
//!
//! For a proper drift-coupled, factor-correlated LIBOR Market Model with
//! tenor structure, change-of-numéraire drifts, correlation matrix, and
//! log-Euler positivity-preserving stepping, see
//! [`crate::interest::lmm::Lmm`] (added 2026-05-08; spot-LIBOR measure,
//! Glasserman 2003 §3.7).
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
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// **NOT a BGM / LIBOR Market Model** despite the name — see the module
/// header for the precise scope. Samples `xn` **independent** discrete-time
/// martingale paths `L_i` via the forward-Euler recurrence
/// `L_i(t+dt) = L_i(t)·(1 + λ_i·ΔW_t^{(i)})` for the formal SDE
/// `dL_i = λ_i L_i dW^{(i)}`. The discrete law is **not log-normal** at
/// finite `dt` (paths may go negative when `λ √dt` is not small); only the
/// continuous-time limit is log-normal. No tenor structure, no measure
/// choice, no cross-forward drift coupling.
pub struct Bgm<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Per-rate noise scale `λ_i` in the Euler step
  /// `L_i(t+dt) = L_i(t)·(1 + λ_i·ΔW)`. **Not** a Black/log-normal vol —
  /// the discrete recurrence is an Euler approximation, not exact log-normal
  /// evolution (see module doc).
  pub lambda: Array1<T>,
  /// Initial values `L_i(0)` (one entry per simulated rate).
  pub x0: Array1<T>,
  /// Number of independent rate paths to simulate (one per matrix row).
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

impl<T: FloatExt, S: SeedExt> Bgm<T, S> {
  pub fn new(lambda: Array1<T>, x0: Array1<T>, xn: usize, t: Option<T>, n: usize, seed: S) -> Self {
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
    Self {
      backend: Cpu,
      lambda,
      x0,
      xn,
      t,
      n,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Bgm<T, S, B> {
  /// The square root of the grid spacing, which is the standard deviation
  /// of one increment; zero when the grid has no increment to take.
  fn sqrt_dt(&self) -> T {
    if self.n > 1 {
      (self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)).sqrt()
    } else {
      T::zero()
    }
  }
}

/// One row's recursion: the tail is pre-filled with `N(0, dt)` draws and each
/// rate then multiplies its own increment, `f_j = f_{j-1} (1 + lambda z_j)`.
/// Shared by the matrix sampler and the row view so the two cannot drift.
fn fill_bgm_row<T: FloatExt, S: SeedExt>(row: &mut [T], x0: T, lambda: T, sqrt_dt: T, seed: &S) {
  if row.is_empty() {
    return;
  }
  row[0] = x0;
  if row.len() == 1 {
    return;
  }
  let normal = SimdNormal::<T>::new(T::zero(), sqrt_dt, seed);
  normal.fill_slice(&mut row[1..]);
  for j in 1..row.len() {
    let f_old = row[j - 1];
    row[j] = f_old + f_old * lambda * row[j];
  }
}

/// One forward rate of the batch, stepped on its own. `Bgm` reports every
/// rate as a row of one matrix and the engine speaks in single paths, so this
/// view is what carries a launch: row `i` is the linear SDE with no drift and
/// the proportional diffusion `lambda[i]`, started at `x0[i]`. It borrows
/// rather than owns, so the seed it advances is the process's own.
#[doc(hidden)]
pub struct BgmRow<'a, T: FloatExt, S: SeedExt, B>(&'a Bgm<T, S, B>, usize);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for BgmRow<'_, T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = BgmRowSampler<T, S>
  where
    Self: 's;

  fn sampler(&self) -> BgmRowSampler<T, S> {
    BgmRowSampler {
      x0: self.0.x0[self.1],
      lambda: self.0.lambda[self.1],
      n: self.0.n,
      sqrt_dt: self.0.sqrt_dt(),
      seed: self.0.seed.derive(),
    }
  }
}

/// [`BgmRow`]'s sampler: one rate's recursion from a seed derived once.
#[doc(hidden)]
pub struct BgmRowSampler<T: FloatExt, S: SeedExt> {
  x0: T,
  lambda: T,
  n: usize,
  sqrt_dt: T,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for BgmRowSampler<T, S> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let row = out
      .as_slice_mut()
      .expect("Bgm row must be contiguous in memory");
    fill_bgm_row(row, self.x0, self.lambda, self.sqrt_dt, &self.seed);
  }

  fn sample(&mut self) -> Array1<T> {
    let mut out = Array1::<T>::zeros(self.n);
    self.sample_into(&mut out);
    out
  }
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for BgmRow<'_, T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::LinearSde {
      a: T::zero(),
      b: T::zero(),
      c: self.0.lambda[self.1],
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

  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Bgm<T, S> { lambda, x0, xn, t, n, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Bgm<T, S, B> {
  type Output = Array2<T>;
  type Sampler<'s>
    = BgmSampler<T, S>
  where
    Self: 's;

  /// Derives (not clones) `self.seed` into the returned sampler: the
  /// derived value is `self.seed`'s *mixed* next tick, not a raw snapshot,
  /// so chunk `i`'s basis and chunk `i+1`'s basis are hash-scrambled
  /// relative to each other rather than one raw stride apart.
  fn sampler(&self) -> BgmSampler<T, S> {
    BgmSampler {
      lambda: self.lambda.clone(),
      x0: self.x0.clone(),
      xn: self.xn,
      n: self.n,
      t: self.t,
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine, one launch per rate: the rows are independent
  /// linear SDEs, so a device steps each row's whole batch in one kernel and
  /// the matrix is assembled here. On the host devices each row runs its
  /// own recursion, which is the same arithmetic the matrix sampler uses.
  fn sample(&self) -> Array2<T> {
    let mut out = Array2::<T>::zeros((self.xn, self.n));
    for i in 0..self.xn {
      out
        .row_mut(i)
        .assign(&self.backend.euler_sample(&BgmRow(self, i)));
    }
    out
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array2<T>) -> R + Sync) -> Vec<R> {
    self.sample_par(m).iter().map(f).collect()
  }

  fn sample_par(&self, m: usize) -> Vec<Array2<T>> {
    let mut out: Vec<Array2<T>> = (0..m).map(|_| Array2::zeros((self.xn, self.n))).collect();
    for i in 0..self.xn {
      let rows = self.backend.euler_paths(&BgmRow(self, i), m);
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
        .assign(&self.backend.try_sample(&BgmRow(self, i))?);
    }
    Ok(out)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array2<T>>, crate::device::DeviceError> {
    let mut out: Vec<Array2<T>> = (0..m).map(|_| Array2::zeros((self.xn, self.n))).collect();
    for i in 0..self.xn {
      let rows = self.backend.try_euler_paths(&BgmRow(self, i), m)?;
      for (matrix, row) in out.iter_mut().zip(rows) {
        matrix.row_mut(i).assign(&row);
      }
    }
    Ok(out)
  }
}

/// Reusable [`Bgm`] sampling state: owns the per-rate scales, the initial
/// curve and the seed source so a Monte-Carlo loop reuses the output matrix.
#[doc(hidden)]
pub struct BgmSampler<T: FloatExt, S: SeedExt> {
  lambda: Array1<T>,
  x0: Array1<T>,
  xn: usize,
  n: usize,
  t: Option<T>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> BgmSampler<T, S> {
  fn fill_matrix(&mut self, fwd: &mut Array2<T>) {
    let sqrt_dt = if self.n > 1 {
      (self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)).sqrt()
    } else {
      T::zero()
    };
    for i in 0..self.xn {
      let mut row = fwd.row_mut(i);
      let row_slice = row
        .as_slice_mut()
        .expect("Bgm row must be contiguous in memory");
      fill_bgm_row(row_slice, self.x0[i], self.lambda[i], sqrt_dt, &self.seed);
    }
  }
}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for BgmSampler<T, S> {
  type Output = Array2<T>;

  fn sample_into(&mut self, out: &mut Array2<T>) {
    self.fill_matrix(out);
  }

  fn sample(&mut self) -> Array2<T> {
    let mut fwd = Array2::<T>::zeros((self.xn, self.n));
    self.fill_matrix(&mut fwd);
    fwd
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyBgm {
  inner_f32: Option<Bgm<f32>>,
  inner_f64: Option<Bgm<f64>>,
  seeded_f32: Option<Bgm<f32, crate::simd_rng::Deterministic>>,
  seeded_f64: Option<Bgm<f64, crate::simd_rng::Deterministic>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyBgm {
  #[new]
  #[pyo3(signature = (lambda_, x0, xn, n, t=None, seed=None, dtype=None))]
  fn new(
    lambda_: Vec<f64>,
    x0: Vec<f64>,
    xn: usize,
    n: usize,
    t: Option<f64>,
    seed: Option<u64>,
    dtype: Option<&str>,
  ) -> Self {
    match (seed, dtype.unwrap_or("f64")) {
      (Some(s), "f32") => {
        let lambda_f32 = ndarray::Array1::from_vec(lambda_.iter().map(|&v| v as f32).collect());
        let x0_f32 = ndarray::Array1::from_vec(x0.iter().map(|&v| v as f32).collect());
        Self {
          inner_f32: None,
          inner_f64: None,
          seeded_f32: Some(Bgm::new(
            lambda_f32,
            x0_f32,
            xn,
            t.map(|v| v as f32),
            n,
            Deterministic::new(s),
          )),
          seeded_f64: None,
        }
      }
      (Some(s), _) => {
        let lambda_arr = ndarray::Array1::from_vec(lambda_);
        let x0_arr = ndarray::Array1::from_vec(x0);
        Self {
          inner_f32: None,
          inner_f64: None,
          seeded_f32: None,
          seeded_f64: Some(Bgm::new(
            lambda_arr,
            x0_arr,
            xn,
            t,
            n,
            Deterministic::new(s),
          )),
        }
      }
      (None, "f32") => {
        let lambda_f32 = ndarray::Array1::from_vec(lambda_.iter().map(|&v| v as f32).collect());
        let x0_f32 = ndarray::Array1::from_vec(x0.iter().map(|&v| v as f32).collect());
        Self {
          inner_f32: Some(Bgm::new(
            lambda_f32,
            x0_f32,
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
        let lambda_arr = ndarray::Array1::from_vec(lambda_);
        let x0_arr = ndarray::Array1::from_vec(x0);
        Self {
          inner_f32: None,
          inner_f64: Some(Bgm::new(lambda_arr, x0_arr, xn, t, n, Unseeded)),
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

#[cfg(test)]
mod tests {
  use ndarray::Array1;

  use super::*;

  #[test]
  fn bgm_sample_runs() {
    let lambda = Array1::<f64>::from_vec(vec![0.2, 0.2, 0.2]);
    let x0 = Array1::<f64>::from_vec(vec![0.03, 0.035, 0.04]);
    let bgm = Bgm::<f64>::new(lambda, x0, 3, Some(1.0), 50, Unseeded);
    let path = bgm.sample();
    // Bgm produces a 2D matrix (n_rates × n_steps)
    assert_eq!(path.nrows(), 3);
    assert_eq!(path.ncols(), 50);
  }
}

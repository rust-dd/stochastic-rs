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
//! A proper drift-coupled, factor-correlated LMM (with tenor / accrual
//! structure, change-of-numéraire drifts, low-rank correlation matrix, and
//! exact log-normal stepping) is planned for a 2.x `interest::lmm` module.
//!
use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// **NOT a BGM / LIBOR Market Model** despite the name — see the module
/// header for the precise scope. Samples `xn` **independent** discrete-time
/// martingale paths `L_i` via the forward-Euler recurrence
/// `L_i(t+dt) = L_i(t)·(1 + λ_i·ΔW_t^{(i)})` for the formal SDE
/// `dL_i = λ_i L_i dW^{(i)}`. The discrete law is **not log-normal** at
/// finite `dt` (paths may go negative when `λ √dt` is not small); only the
/// continuous-time limit is log-normal. No tenor structure, no measure
/// choice, no cross-forward drift coupling.
pub struct Bgm<T: FloatExt, S: SeedExt = Unseeded> {
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
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt> Bgm<T> {
  pub fn new(lambda: Array1<T>, x0: Array1<T>, xn: usize, t: Option<T>, n: usize) -> Self {
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
      lambda,
      x0,
      xn,
      t,
      n,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> Bgm<T, Deterministic> {
  pub fn seeded(
    lambda: Array1<T>,
    x0: Array1<T>,
    xn: usize,
    t: Option<T>,
    n: usize,
    seed: u64,
  ) -> Self {
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
      lambda,
      x0,
      xn,
      t,
      n,
      seed: Deterministic::new(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Bgm<T, S> {
  type Output = Array2<T>;

  fn sample(&self) -> Self::Output {
    let mut fwd = Array2::<T>::zeros((self.xn, self.n));
    if self.n == 0 {
      return fwd;
    }

    for i in 0..self.xn {
      fwd[(i, 0)] = self.x0[i];
    }

    if self.n == 1 {
      return fwd;
    }

    let n_increments = self.n - 1;
    let sqrt_dt = (self.t.unwrap_or(T::one()) / T::from_usize_(n_increments)).sqrt();

    for i in 0..self.xn {
      let mut row = fwd.row_mut(i);
      let row_slice = row
        .as_slice_mut()
        .expect("Bgm row must be contiguous in memory");
      let tail = &mut row_slice[1..];
      let normal = SimdNormal::<T>::from_seed_source(T::zero(), sqrt_dt, &self.seed);
      normal.fill_slice_fast(tail);

      for j in 1..self.n {
        let f_old = row_slice[j - 1];
        row_slice[j] = f_old + f_old * self.lambda[i] * row_slice[j];
      }
    }

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
          seeded_f32: Some(Bgm::seeded(
            lambda_f32,
            x0_f32,
            xn,
            t.map(|v| v as f32),
            n,
            s,
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
          seeded_f64: Some(Bgm::seeded(lambda_arr, x0_arr, xn, t, n, s)),
        }
      }
      (None, "f32") => {
        let lambda_f32 = ndarray::Array1::from_vec(lambda_.iter().map(|&v| v as f32).collect());
        let x0_f32 = ndarray::Array1::from_vec(x0.iter().map(|&v| v as f32).collect());
        Self {
          inner_f32: Some(Bgm::new(lambda_f32, x0_f32, xn, t.map(|v| v as f32), n)),
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
          inner_f64: Some(Bgm::new(lambda_arr, x0_arr, xn, t, n)),
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
    let bgm = Bgm::<f64>::new(lambda, x0, 3, Some(1.0), 50);
    let path = bgm.sample();
    // Bgm produces a 2D matrix (n_rates × n_steps)
    assert_eq!(path.nrows(), 3);
    assert_eq!(path.ncols(), 50);
  }
}

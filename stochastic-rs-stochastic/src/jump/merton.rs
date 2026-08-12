//! # Merton
//!
//! $$
//! \frac{dS_t}{S_{t^-}}=(\mu-\lambda\kappa)dt+\sigma dW_t+(Y-1)dN_t
//! $$
//!
//! ## Generic distribution parameter `D`
//!
//! `Merton<T, D, S>` is generic over the jump-size distribution `D`, which
//! must implement [`rand_distr::Distribution<T>`]. Common choices:
//!
//! - [`SimdNormal<T>`](stochastic_rs_distributions::normal::SimdNormal) for
//!   the classical lognormal-jump Merton (1976) model
//! - [`SimdLaplace<T>`] / a custom asymmetric double-exponential for Kou-style
//!   jumps
//! - any user-defined `Distribution<T>`
//!
//! Python bindings (under the `python` feature) need a monomorphic type
//! signature, so the `PyMerton` wrapper fixes `D = SimdNormal<f64>`. If
//! you need a different jump distribution from Python, prefer the SVJ /
//! Bates calibrators, or compose your own wrapper struct on the Rust side
//! and re-bind via PyO3.
//!
use ndarray::Array1;
use rand_distr::Distribution;
#[cfg(feature = "python")]
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::scalar::ScalarNormal;

use crate::buffer::array1_from_fill;
use crate::process::cpoisson::CompoundPoisson;
use crate::process::poisson::Poisson;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Clone)]
pub struct Merton<T, D, S: SeedExt = Unseeded>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  /// Drift rate μ of the log-price (this field is named `alpha`, not the
  /// module header's compensator κ).
  pub alpha: T,
  /// Diffusion scale σ of the continuous (Brownian) component.
  pub sigma: T,
  /// Jump (Poisson) intensity λ — arrival rate of the log-normal jumps.
  pub lambda: T,
  /// Jump-size compensator κ (E\[Y−1\]-like term, matching the module
  /// header's own λκ), subtracted from the drift scaled by `lambda` —
  /// not a mean-reversion level; Merton's jump-diffusion has no mean
  /// reversion.
  pub theta: T,
  /// Number of points sampled along the Merton path.
  pub n: usize,
  /// Initial value X₀ of the Merton path.
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Compound-Poisson jump driver generating the log-normal jump sizes.
  /// **Partial exception to [`ProcessExt`]'s reproducibility guarantee:**
  /// hard-wired to `Unseeded` (default `S`) by pre-existing design, so the
  /// jump arrivals/sizes are never seed-reproducible even though the
  /// diffusion component below is. See MIGRATION.md and [`ProcessExt`]'s
  /// trait-level reproducibility section.
  pub cpoisson: CompoundPoisson<T, D>,
  /// Seed strategy (compile-time: `Unseeded` or `Deterministic`). Consulted
  /// by the diffusion component only — see `cpoisson`'s doc above.
  pub seed: S,
}

impl<T, D, S: SeedExt> Merton<T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub fn new(
    alpha: T,
    sigma: T,
    lambda: T,
    theta: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    cpoisson: CompoundPoisson<T, D>,
    seed: S,
  ) -> Self {
    Self {
      alpha,
      sigma,
      lambda,
      theta,
      n,
      x0,
      t,
      cpoisson,
      seed,
    }
  }

  /// Every field has a matching `with_*` builder setter, e.g.
  /// `Merton::default().with_alpha(0.06)`. No persisted cache:
  /// `sampler()` builds its Gaussian diffusion source fresh from `self`
  /// every call, correctly threading `self.seed` (unlike `Bates1996`'s
  /// documented `cgns` quirk — see that type's own doc).
  /// Replace `alpha`, all else unchanged.
  pub fn with_alpha(mut self, alpha: T) -> Self {
    self.alpha = alpha;
    self
  }

  /// Replace `sigma`, all else unchanged.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    self.sigma = sigma;
    self
  }

  /// Replace `lambda`, all else unchanged.
  pub fn with_lambda(mut self, lambda: T) -> Self {
    self.lambda = lambda;
    self
  }

  /// Replace `theta`, all else unchanged.
  pub fn with_theta(mut self, theta: T) -> Self {
    self.theta = theta;
    self
  }

  /// Replace `x0`, all else unchanged.
  pub fn with_x0(mut self, x0: Option<T>) -> Self {
    self.x0 = x0;
    self
  }

  /// Replace the compound-Poisson jump driver, all else unchanged.
  pub fn with_cpoisson(mut self, cpoisson: CompoundPoisson<T, D>) -> Self {
    self.cpoisson = cpoisson;
    self
  }

  /// Replace the number of simulation steps `n`, all else unchanged.
  pub fn with_steps(mut self, n: usize) -> Self {
    self.n = n;
    self
  }

  /// Replace the simulation horizon `t`, all else unchanged.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    self.t = t;
    self
  }

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

/// α=0.03, σ=0.2, λ=1.0, θ=0.0, x₀=0 — matches the crate's Merton
/// visualization-gallery fixture
/// (`stochastic-rs-viz/src/tests/categories/jump.rs`'s `normal_cpoisson(1.0,
/// n, 0.1)`, which itself runs at n=96, not the n=252 below); `D =
/// ScalarNormal<T>` per this crate's jump-size-distribution convention
/// (`Sync`-safe, drives the shared RNG — see
/// `stochastic-rs-distributions::scalar`). The log-jump (not the jump
/// factor `Y` itself) is Gaussian, `N(0, 0.1)` — the classical
/// lognormal-jump Merton (1976) model this file's own top doc names. t=1,
/// n=252 — one trading year of daily steps (this crate's `Default`
/// convention, not itself drawn from that fixture).
impl<T: FloatExt> Default for Merton<T, ScalarNormal<T>, Unseeded> {
  fn default() -> Self {
    let n = 252;
    let t = Some(T::one());
    Self::new(
      T::from_f64_fast(0.03),
      T::from_f64_fast(0.2),
      T::one(),
      T::zero(),
      n,
      Some(T::zero()),
      t,
      CompoundPoisson::new(
        ScalarNormal::new(T::zero(), T::from_f64_fast(0.1)),
        Poisson::new(T::one(), Some(n), t, Unseeded),
        Unseeded,
      ),
      Unseeded,
    )
  }
}

impl<T, D, S: SeedExt> ProcessExt<T> for Merton<T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  type Output = Array1<T>;
  type Sampler<'s>
    = MertonSampler<'s, T, D>
  where
    Self: 's;

  fn sampler(&self) -> MertonSampler<'_, T, D> {
    // The diffusion source is owned and derived from `self.seed`; the
    // compound-Poisson jump driver is borrowed and re-drawn per fill exactly
    // as the legacy `sample()` did (it rebuilds its own RNG from
    // `cpoisson.seed` each call). The two seed sources are independent, so the
    // first fill reproduces the legacy stream bit-for-bit.
    let dt = if self.n > 1 {
      self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
    } else {
      T::zero()
    };
    let drift_dt = (self.alpha
      - self.sigma.powf(T::from_usize(2).unwrap()) / T::from_usize(2).unwrap()
      - self.lambda * self.theta)
      * dt;
    MertonSampler {
      n: self.n,
      sigma: self.sigma,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      drift_dt,
      cpoisson: &self.cpoisson,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }
}

/// Reusable [`Merton`] sampling state: owns the Gaussian diffusion source and
/// borrows the compound-Poisson jump driver, so a Monte-Carlo loop pays the
/// `SimdNormal` setup once.
#[doc(hidden)]
pub struct MertonSampler<'a, T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  n: usize,
  sigma: T,
  x0: T,
  dt: T,
  drift_dt: T,
  cpoisson: &'a CompoundPoisson<T, D>,
  normal: SimdNormal<T>,
}

impl<T, D> MertonSampler<'_, T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }

    let jump_increments = self.cpoisson.sample_grid_increments(out.len(), self.dt);
    let mut gn = Array1::<T>::zeros(out.len() - 1);
    if let Some(gn_slice) = gn.as_slice_mut() {
      self.normal.fill_slice(gn_slice);
    }

    out[0] = self.x0;

    for i in 1..out.len() {
      out[i] = out[i - 1] + self.drift_dt + self.sigma * gn[i - 1] + jump_increments[i];
    }
  }
}

impl<T, D> PathSampler<T> for MertonSampler<'_, T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    self.fill_path(
      out
        .as_slice_mut()
        .expect("Merton output must be contiguous"),
    );
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyMerton {
  inner_f32: Option<Merton<f32, crate::traits::CallableDist<f32>>>,
  inner_f64: Option<Merton<f64, crate::traits::CallableDist<f64>>>,
  seeded_f32: Option<Merton<f32, crate::traits::CallableDist<f32>, crate::simd_rng::Deterministic>>,
  seeded_f64: Option<Merton<f64, crate::traits::CallableDist<f64>, crate::simd_rng::Deterministic>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyMerton {
  #[new]
  #[pyo3(signature = (alpha, sigma, lambda_, theta, distribution, n, x0=None, t=None, seed=None, dtype=None))]
  fn new(
    alpha: f64,
    sigma: f64,
    lambda_: f64,
    theta: f64,
    distribution: pyo3::Py<pyo3::PyAny>,
    n: usize,
    x0: Option<f64>,
    t: Option<f64>,
    seed: Option<u64>,
    dtype: Option<&str>,
  ) -> Self {
    use crate::process::poisson::Poisson;
    let mut s = Self {
      inner_f32: None,
      inner_f64: None,
      seeded_f32: None,
      seeded_f64: None,
    };
    match dtype.unwrap_or("f64") {
      "f32" => {
        let cpoisson = CompoundPoisson::new(
          crate::traits::CallableDist::new(distribution),
          Poisson::new(lambda_ as f32, Some(n), t.map(|v| v as f32), Unseeded),
          Unseeded,
        );
        match seed {
          Some(sd) => {
            s.seeded_f32 = Some(Merton::new(
              alpha as f32,
              sigma as f32,
              lambda_ as f32,
              theta as f32,
              n,
              x0.map(|v| v as f32),
              t.map(|v| v as f32),
              cpoisson,
              Deterministic::new(sd),
            ));
          }
          None => {
            s.inner_f32 = Some(Merton::new(
              alpha as f32,
              sigma as f32,
              lambda_ as f32,
              theta as f32,
              n,
              x0.map(|v| v as f32),
              t.map(|v| v as f32),
              cpoisson,
              Unseeded,
            ));
          }
        }
      }
      _ => {
        let cpoisson = CompoundPoisson::new(
          crate::traits::CallableDist::new(distribution),
          Poisson::new(lambda_, Some(n), t, Unseeded),
          Unseeded,
        );
        match seed {
          Some(sd) => {
            s.seeded_f64 = Some(Merton::new(
              alpha,
              sigma,
              lambda_,
              theta,
              n,
              x0,
              t,
              cpoisson,
              Deterministic::new(sd),
            ));
          }
          None => {
            s.inner_f64 = Some(Merton::new(
              alpha, sigma, lambda_, theta, n, x0, t, cpoisson, Unseeded,
            ));
          }
        }
      }
    }
    s
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
    use numpy::ndarray::Array2;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_dispatch!(self, |inner| {
      let paths = inner.sample_par(m);
      let n = paths[0].len();
      let mut result = Array2::zeros((m, n));
      for (i, path) in paths.iter().enumerate() {
        result.row_mut(i).assign(path);
      }
      result.into_pyarray(py).into_py_any(py).unwrap()
    })
  }
}

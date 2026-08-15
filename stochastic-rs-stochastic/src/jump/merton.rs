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
//! - [`SimdGed<T>`](stochastic_rs_distributions::ged::SimdGed) with β=1
//!   (Laplace / double-exponential) / a custom asymmetric variant for
//!   Kou-style jumps
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
  /// Single source of truth: `sampler()` reads this field directly (not
  /// `cpoisson.poisson.lambda`) for the jump-arrival rate. Every setter
  /// that can change it (`with_lambda`, `with_cpoisson`) keeps
  /// `cpoisson.poisson.lambda` synced to match — see those methods' docs
  /// and `resync_cpoisson_poisson`. That syncing only happens through the
  /// setters: a direct `pub` field assignment (`merton.lambda = x`) is not
  /// intercepted and will desync it from `cpoisson.poisson.lambda`.
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
  /// Fully seed-reproducible: [`new`](Self::new) builds it internally from
  /// `seed` (`seed.clone().derive()` — a hash-mixed child, decorrelated
  /// from but a deterministic function of the same `seed` the diffusion
  /// component consults directly), and `sampler()` derives a fresh,
  /// chunk-local basis off `self.cpoisson.seed` for every chunk, mirroring
  /// the diffusion component's own per-chunk `self.seed`-derived basis.
  ///
  /// `sampler()` reads only `cpoisson.distribution` (the jump-size law)
  /// and `self.lambda` — **not** `cpoisson.poisson.lambda` — from this
  /// field on the sampling path; `cpoisson.poisson.{n,t_max,seed}` are
  /// inert there (`grid_increments` never consults them). That inertness
  /// is scoped to *this type's own* sampling, though: `cpoisson` is a
  /// `CompoundPoisson` in its own right, and calling `.sample()` on it
  /// directly (bypassing `Merton` entirely) drives it through
  /// `Poisson::sample_impl`, which *does* branch on `.n`/`.t_max` (fixed
  /// count vs. horizon mode) and *does* consult `.seed` — genuinely live
  /// there. Left `pub` for both reasons: a caller can inspect or directly
  /// `.sample()` the embedded compound-Poisson process as its own
  /// standalone `ProcessExt`, and can replace it wholesale via
  /// [`with_cpoisson`](Self::with_cpoisson) (which keeps `self.lambda` in
  /// sync with the replacement — see that method's doc) or direct field
  /// assignment (which does not; assign through `with_cpoisson` unless
  /// you separately update `self.lambda` to match).
  pub cpoisson: CompoundPoisson<T, D, S>,
  /// Seed strategy (compile-time: `Unseeded` or `Deterministic`). Consulted
  /// directly by the diffusion component; `cpoisson`'s own seed (set at
  /// construction from this same value — see `cpoisson`'s doc above) drives
  /// the jump component.
  pub seed: S,
}

/// Every field has a matching `with_*` builder setter, e.g.
/// `Merton::default().with_alpha(0.06)`. No persisted cache:
/// `sampler()` builds its Gaussian diffusion source fresh from `self`
/// every call, correctly threading `self.seed` (unlike `Bates1996`'s
/// documented `cgns` quirk — see that type's own doc).
impl<T, D, S: SeedExt> Merton<T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  /// Builds the compound-Poisson jump driver internally from `jump_dist`
  /// and `lambda`, seeded from `seed` (see `cpoisson`'s field doc) — the
  /// caller supplies the jump-size distribution and intensity directly
  /// instead of pre-building a `Poisson`/`CompoundPoisson` pair and
  /// threading a third, independent seed through it by hand.
  pub fn new(
    alpha: T,
    sigma: T,
    lambda: T,
    theta: T,
    jump_dist: D,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    let cpoisson = CompoundPoisson::new(
      jump_dist,
      Poisson::new(lambda, Some(n), t, Unseeded),
      seed.clone().derive(),
    );
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

  /// Replace `lambda`, all else unchanged. `sampler()` reads `self.lambda`
  /// directly for the jump-arrival intensity (see `cpoisson`'s field doc),
  /// so this alone already changes the sampled jump rate; it also
  /// re-syncs the otherwise-cosmetic mirror `cpoisson.poisson.lambda` (see
  /// `resync_cpoisson_poisson`) so a caller inspecting it does not see a
  /// stale value.
  pub fn with_lambda(mut self, lambda: T) -> Self {
    self.lambda = lambda;
    self.resync_cpoisson_poisson();
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

  /// Replace the compound-Poisson jump driver wholesale, adopting its
  /// intensity as the new `self.lambda` — `sampler()` reads `self.lambda`,
  /// not `cpoisson.poisson.lambda`, for the jump-arrival rate (see
  /// `cpoisson`'s field doc), so without this adoption the incoming
  /// driver's own intensity would be silently ignored and the *old*
  /// `self.lambda` would keep driving jumps while only the distribution
  /// changed. `cpoisson.poisson.{n,t_max}` are left exactly as the caller
  /// supplied them (not normalized to `self.{n,t}`) since, unlike
  /// `lambda`, they carry no live weight on this type's sampling path
  /// either way.
  pub fn with_cpoisson(mut self, cpoisson: CompoundPoisson<T, D, S>) -> Self {
    self.lambda = cpoisson.poisson.lambda;
    self.cpoisson = cpoisson;
    self
  }

  /// Replace the number of simulation steps `n`, all else unchanged; also
  /// re-syncs `cpoisson.poisson.n` (see `resync_cpoisson_poisson`) — dead
  /// on this type's own sampling path, but kept from silently going stale
  /// for a caller inspecting `cpoisson` directly.
  pub fn with_steps(mut self, n: usize) -> Self {
    self.n = n;
    self.resync_cpoisson_poisson();
    self
  }

  /// Replace the simulation horizon `t`, all else unchanged; also re-syncs
  /// `cpoisson.poisson.t_max` (see `resync_cpoisson_poisson`) — dead on
  /// this type's own sampling path, but kept from silently going stale for
  /// a caller inspecting `cpoisson` directly.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    self.t = t;
    self.resync_cpoisson_poisson();
    self
  }

  /// Replace the seed strategy's value, all else unchanged — including
  /// re-deriving `cpoisson`'s own seed from the new value exactly as
  /// [`new`](Self::new) does (`cpoisson`'s distribution and lambda are
  /// untouched), so the result matches a fresh construction with this
  /// seed rather than leaving the jump component keyed to the old one.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.cpoisson.seed = seed.clone().derive();
    self.seed = seed;
    self
  }

  /// Rebuilds `cpoisson.poisson` from `self.{lambda, n, t}` so a caller
  /// reading `cpoisson.poisson` directly never sees it disagree with the
  /// outer struct's own record of the same three values — most load-
  /// bearing for `lambda`, which `sampler()` actually reads off `self`
  /// (not off this mirror) for the jump-arrival rate, but applied
  /// uniformly to `n`/`t_max` too even though those two are inert on the
  /// sampling path either way (see `cpoisson`'s field doc). Called from
  /// every setter that changes `lambda`, `n`, or `t`.
  fn resync_cpoisson_poisson(&mut self) {
    self.cpoisson.poisson = Poisson::new(self.lambda, Some(self.n), self.t, Unseeded);
  }
}

/// α=0.03, σ=0.2, λ=1.0, θ=0.0, x₀=0, with a `ScalarNormal(0, 0.1)` jump
/// size — `D = ScalarNormal<T>` per this crate's jump-size-distribution
/// convention (`Sync`-safe, drives the shared RNG — see
/// `stochastic-rs-distributions::scalar`). The log-jump (not the jump
/// factor `Y` itself) is Gaussian, `N(0, 0.1)` — the classical
/// lognormal-jump Merton (1976) model this file's own top doc names. t=1,
/// n=252 — one trading year of daily steps (this crate's `Default`
/// convention).
impl<T: FloatExt> Default for Merton<T, ScalarNormal<T>, Unseeded> {
  fn default() -> Self {
    let n = 252;
    let t = Some(T::one());
    Self::new(
      T::from_f64_fast(0.03),
      T::from_f64_fast(0.2),
      T::one(),
      T::zero(),
      ScalarNormal::new(T::zero(), T::from_f64_fast(0.1)),
      n,
      Some(T::zero()),
      t,
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
    = MertonSampler<'s, T, D, S>
  where
    Self: 's;

  fn sampler(&self) -> MertonSampler<'_, T, D, S> {
    // The diffusion source is owned and derived from `self.seed`. The jump
    // driver's distribution/lambda are borrowed straight off `self.cpoisson`
    // (read-only parameters, safe to share across chunks), but its seed is
    // captured as an owned, chunk-local `self.cpoisson.seed.derive()` —
    // never a borrowed `&self.cpoisson`, which would let every chunk's
    // sampler race on the same shared atomic during the parallel region
    // (see `ProcessExt`'s "Reproducibility requirement on implementors").
    // Each path within one chunk still re-derives its own jump sub-stream
    // from that owned basis exactly as the legacy `sample()` did from the
    // old (always-`Unseeded`) field, so only the seed *source* changed.
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
      jump_distribution: &self.cpoisson.distribution,
      lambda: self.lambda,
      jump_seed: self.cpoisson.seed.derive(),
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }
}

/// Reusable [`Merton`] sampling state: owns the Gaussian diffusion source and
/// borrows the compound-Poisson jump driver, so a Monte-Carlo loop pays the
/// `SimdNormal` setup once.
#[doc(hidden)]
pub struct MertonSampler<'a, T, D, S: SeedExt>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  n: usize,
  sigma: T,
  x0: T,
  dt: T,
  drift_dt: T,
  jump_distribution: &'a D,
  lambda: T,
  jump_seed: S,
  normal: SimdNormal<T>,
}

impl<T, D, S: SeedExt> MertonSampler<'_, T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }

    let jump_increments = crate::process::cpoisson::grid_increments(
      self.jump_distribution,
      self.lambda,
      &self.jump_seed,
      out.len(),
      self.dt,
    );
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

impl<T, D, S: SeedExt> PathSampler<T> for MertonSampler<'_, T, D, S>
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
    let mut s = Self {
      inner_f32: None,
      inner_f64: None,
      seeded_f32: None,
      seeded_f64: None,
    };
    match dtype.unwrap_or("f64") {
      "f32" => {
        let jump_dist = crate::traits::CallableDist::new(distribution);
        match seed {
          Some(sd) => {
            s.seeded_f32 = Some(Merton::new(
              alpha as f32,
              sigma as f32,
              lambda_ as f32,
              theta as f32,
              jump_dist,
              n,
              x0.map(|v| v as f32),
              t.map(|v| v as f32),
              Deterministic::new(sd),
            ));
          }
          None => {
            s.inner_f32 = Some(Merton::new(
              alpha as f32,
              sigma as f32,
              lambda_ as f32,
              theta as f32,
              jump_dist,
              n,
              x0.map(|v| v as f32),
              t.map(|v| v as f32),
              Unseeded,
            ));
          }
        }
      }
      _ => {
        let jump_dist = crate::traits::CallableDist::new(distribution);
        match seed {
          Some(sd) => {
            s.seeded_f64 = Some(Merton::new(
              alpha,
              sigma,
              lambda_,
              theta,
              jump_dist,
              n,
              x0,
              t,
              Deterministic::new(sd),
            ));
          }
          None => {
            s.inner_f64 = Some(Merton::new(
              alpha, sigma, lambda_, theta, jump_dist, n, x0, t, Unseeded,
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

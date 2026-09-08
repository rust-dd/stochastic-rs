//! # Vasicek
//!
//! $$
//! dr_t=a(b-r_t)dt+\sigma dW_t
//! $$
//!
//! ## Same SDE as [`Ou`]
//!
//! Vasicek's contribution was the term-structure theory built on top of this
//! equation, not the equation: `dr = a(b − r)dt + σ dW` is the
//! Ornstein-Uhlenbeck process with the short rate as its state. The
//! dictionary is `a = ` [`theta`](Vasicek::theta) `= ` [`Ou::theta`]
//! (reversion speed), `b = ` [`mu`](Vasicek::mu) `= ` [`Ou::mu`] (the level
//! reverted to), `σ = ` [`sigma`](Vasicek::sigma) `= ` [`Ou::sigma`], and
//! `r = X`.
//!
//! So this type does not re-derive anything. It holds an [`Ou`], its sampler
//! *is* [`OuSampler`], and the Euler family it reports is the one that `Ou`
//! reports — one recursion and one device kernel serve both names. The
//! separate identity buys the short-rate application: bond prices, yield
//! curves and the calibration surface in `stochastic-rs-quant` are stated
//! against `Vasicek`, and none of them are meaningful for a spread or a
//! log-price sampled as an `Ou`.
//!
//! The one thing the two do not share is a random stream. [`Vasicek::new`]
//! derives a child seed for the embedded `Ou`, so an `Ou` and a `Vasicek`
//! built from the same [`Deterministic`](stochastic_rs_core::simd_rng::Deterministic)
//! seed are two independent draws of the same law, not the same path.
//!
//! References:
//! - Vasicek O. (1977) — *An Equilibrium Characterization of the Term
//!   Structure*, Journal of Financial Economics 5(2), 177–188,
//!   DOI: 10.1016/0304-405X(77)90016-2.
//! - Uhlenbeck G. E., Ornstein L. S. (1930) — *On the Theory of the
//!   Brownian Motion*, Physical Review 36(5), 823–841,
//!   DOI: 10.1103/PhysRev.36.823 — the underlying mean-reverting
//!   diffusion ([`Ou`]) this file wraps under
//!   short-rate parameter names.
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::device::Cpu;
use crate::diffusion::ou::Ou;
use crate::diffusion::ou::OuSampler;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Vasicek short-rate model — the [`Ou`] process under short-rate
/// vocabulary, and built on it: in the SDE `dr = a(b − r) dt + σ dW` (file
/// header) the Rust field [`theta`](Self::theta) is `a` (mean-reversion
/// speed) and [`mu`](Self::mu) is `b` (long-run mean level).
///
/// The embedded `Ou` is what actually samples and what names the Euler
/// family, so the two names cannot come apart. It is kept in step with the
/// mirrored fields by [`new`](Self::new) and by every `with_*` setter;
/// assigning to a public field directly changes the mirror only, so build
/// with those rather than by mutating in place.
#[derive(Clone)]
pub struct Vasicek<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Mean-reversion speed (`a` in the SDE). Controls how fast `r` is pulled
  /// back toward [`mu`](Self::mu).
  pub theta: T,
  /// Long-run mean level (`b` in the SDE). The value `r` reverts to as
  /// `t → ∞`.
  pub mu: T,
  /// Diffusion scale σ multiplying `dW_t` (`σ` in the SDE).
  pub sigma: T,
  /// Number of points sampled along the Vasicek path.
  pub n: usize,
  /// Initial short rate r₀.
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  ou: Ou<T, S>,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

/// Every field has a matching `with_*` builder setter, e.g.
/// `Vasicek::default().with_theta(1.0).with_sigma(0.05)`.
impl<T: FloatExt, S: SeedExt> Vasicek<T, S> {
  pub fn new(theta: T, mu: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    Self {
      backend: Cpu,
      mu,
      sigma,
      theta,
      n,
      x0,
      t,
      ou: Ou::new(theta, mu, sigma, n, x0, t, seed.derive()),
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Vasicek<T, S, B> {
  /// Rebuilds the embedded [`Ou`] from the mirrored fields, keeping the
  /// child seed it already holds.
  ///
  /// Keeping that seed rather than deriving a new one is what makes a
  /// setter agree with a fresh [`new`](Self::new). `new` derives its child
  /// from an *unadvanced* outer seed; deriving again here would take a
  /// second, different child off a seed the first derive already advanced,
  /// so `Vasicek::new(a, ..).with_mu(b)` and `Vasicek::new(a, b, ..)` would
  /// draw different paths from the same seed. `Ou::seed` is `pub`, so the
  /// fixed child is simply cloned across. Only [`with_seed`](Self::with_seed)
  /// derives, because it is the one setter that mirrors `new`'s own
  /// construction order.
  fn resync(mut self) -> Self {
    let ou_seed = self.ou.seed.clone();
    self.ou = Ou::new(
      self.theta,
      self.mu,
      self.sigma,
      self.n,
      self.x0,
      self.t,
      ou_seed,
    );
    self
  }

  /// Replace `theta`, all else unchanged.
  pub fn with_theta(mut self, theta: T) -> Self {
    self.theta = theta;
    self.resync()
  }

  /// Replace `mu`, all else unchanged.
  pub fn with_mu(mut self, mu: T) -> Self {
    self.mu = mu;
    self.resync()
  }

  /// Replace `sigma`, all else unchanged.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    self.sigma = sigma;
    self.resync()
  }

  /// Replace `x0`, all else unchanged.
  pub fn with_x0(mut self, x0: Option<T>) -> Self {
    self.x0 = x0;
    self.resync()
  }

  /// Replace the number of simulation steps `n`, all else unchanged.
  pub fn with_steps(mut self, n: usize) -> Self {
    self.n = n;
    self.resync()
  }

  /// Replace the simulation horizon `t`, all else unchanged.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    self.t = t;
    self.resync()
  }

  /// Replace the seed strategy's value; re-derives the embedded `Ou`'s
  /// seed from the *new* outer seed, exactly mirroring `new()`'s own
  /// construction order (derive before moving `seed` into `self.seed`).
  pub fn with_seed(mut self, seed: S) -> Self {
    self.ou = Ou::new(
      self.theta,
      self.mu,
      self.sigma,
      self.n,
      self.x0,
      self.t,
      seed.derive(),
    );
    self.seed = seed;
    self
  }
}

/// a=3.0, b=0.03, σ=0.02, r₀=0.03 — a textbook Vasicek parameterization.
/// t=1, n=252 — one trading year of daily steps (this crate's `Default`
/// convention).
impl<T: FloatExt> Default for Vasicek<T, Unseeded> {
  fn default() -> Self {
    Self::new(
      T::from_f64_fast(3.0),
      T::from_f64_fast(0.03),
      T::from_f64_fast(0.02),
      252,
      Some(T::from_f64_fast(0.03)),
      Some(T::one()),
      Unseeded,
    )
  }
}

/// Vasicek is Ornstein-Uhlenbeck under another name, so it reaches every
/// device through the family a Gaussian [`Ou`] already declares — no kernel
/// and no declaration of its own. The embedded process answers for the
/// family, the state and the grid rather than this file restating them,
/// which is what keeps the device path and the host path describing one
/// model.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for Vasicek<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerCoefficients::euler_spec(&self.ou)
  }

  fn initial_value(&self) -> T {
    crate::euler::EulerCoefficients::initial_value(&self.ou)
  }

  fn grid_points(&self) -> usize {
    crate::euler::EulerCoefficients::grid_points(&self.ou)
  }

  fn horizon(&self) -> T {
    crate::euler::EulerCoefficients::horizon(&self.ou)
  }

  /// The one thing not taken from the embedded `Ou`: its seed is a child of
  /// this one, so reading it here would give a `Vasicek` the stream of the
  /// `Ou` it holds rather than its own.
  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Vasicek<T, S> { theta, mu, sigma, n, x0, t, seed, ou } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Vasicek<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = VasicekSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> VasicekSampler<T> {
    self.ou.sampler()
  }

  /// Through the Euler engine: on a device the recursion runs in the kernel,
  /// on the host devices it is this process's own sampler, chunked exactly as
  /// `ProcessExt` chunks.
  fn sample(&self) -> Array1<T> {
    self.backend.euler_sample(self)
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    self.backend.euler_paths_map(self, m, f)
  }

  fn sample_map_view<R: Send>(
    &self,
    m: usize,
    f: impl Fn(ndarray::ArrayView1<T>) -> R + Sync,
  ) -> Vec<R> {
    self.backend.euler_paths_map_view(self, m, f)
  }

  fn sample_reduce(&self, m: usize, reduce: crate::euler::Reduce) -> Vec<T> {
    crate::euler::EulerBackend::try_euler_reduce(&self.backend, self, m, reduce)
      .unwrap_or_else(crate::device::device_panic)
  }

  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    self.backend.euler_paths(self, m)
  }

  fn try_sample(&self) -> Result<Array1<T>, crate::device::DeviceError> {
    self.backend.try_sample(self)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, crate::device::DeviceError> {
    self.backend.try_euler_paths(self, m)
  }
}

/// Reusable [`Vasicek`] sampling state, which is [`OuSampler`] itself: the
/// Vasicek path is the OU path, so there is one recursion rather than a
/// forwarding shell around it.
#[doc(hidden)]
pub type VasicekSampler<T> = OuSampler<T>;

py_process_1d!(PyVasicek, Vasicek,
  sig: (theta, mu, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (theta: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>),
  device
);

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  /// Both names run the same recursion, with the same parameters.
  ///
  /// This is what makes the consolidation a refactor rather than a rewrite:
  /// `Vasicek` samples through the `Ou` it holds. The two *paths* differ,
  /// because the embedded process gets a derived child seed — so what is
  /// checked is the recursion itself. Inverting the Euler step,
  /// `(x[i] − x[i−1] − θ(μ − x[i−1])Δ) / (σ√Δ)` is the standard normal the
  /// step drew, and it is standard only if all three parameters entered
  /// where they claim to. A swapped `theta`/`mu`, a missing `dt` or a
  /// re-derived recursion moves the residual mean or its spread well past
  /// the band, whatever the noise was.
  #[test]
  fn both_names_run_the_same_recursion() {
    for (theta, mu, sigma, x0) in [
      (0.5_f64, 0.04, 0.01, 0.05),
      (2.0, 0.0, 0.3, -1.0),
      (0.1, 1.5, 0.02, 1.5),
    ] {
      const N: usize = 512;
      const M: usize = 64;
      let vasicek =
        Vasicek::<f64, _>::new(theta, mu, sigma, N, Some(x0), Some(1.0), Deterministic::new(7));
      let ou = crate::diffusion::ou::Ou::<f64, _>::new(
        theta,
        mu,
        sigma,
        N,
        Some(x0),
        Some(1.0),
        Deterministic::new(7),
      );
      for (name, paths) in [
        ("Vasicek", vasicek.sample_par(M)),
        ("Ou", ou.sample_par(M)),
      ] {
        let dt = 1.0 / (N - 1) as f64;
        let mut residuals = Vec::with_capacity(M * (N - 1));
        for path in &paths {
          assert_eq!(path[0], x0, "{name}: the path does not start at x0");
          for i in 1..path.len() {
            let drift = theta * (mu - path[i - 1]) * dt;
            residuals.push((path[i] - path[i - 1] - drift) / (sigma * dt.sqrt()));
          }
        }
        let n = residuals.len() as f64;
        let mean = residuals.iter().sum::<f64>() / n;
        let var = residuals.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        // Five standard errors of each estimate: 1/sqrt(n) for the mean of a
        // standard normal, sqrt(2/n) for its variance.
        assert!(
          mean.abs() < 5.0 / n.sqrt(),
          "{name} theta = {theta}, mu = {mu}: residual mean {mean}, not 0 — \
           the drift is not theta * (mu - x) * dt"
        );
        assert!(
          (var - 1.0).abs() < 5.0 * (2.0 / n).sqrt(),
          "{name} theta = {theta}, mu = {mu}: residual variance {var}, not 1 — \
           the diffusion is not sigma * sqrt(dt)"
        );
      }
    }
  }

  #[test]
  fn sample_length_matches_n() {
    let v = Vasicek::<f64>::new(0.5, 0.04, 0.01, 100, Some(0.05), Some(1.0), Unseeded);
    let path = v.sample();
    assert_eq!(path.len(), 100);
  }

  #[test]
  fn sample_starts_at_x0() {
    let x0 = 0.05;
    let v = Vasicek::<f64>::new(0.5, 0.04, 0.01, 100, Some(x0), Some(1.0), Unseeded);
    let path = v.sample();
    assert!((path[0] - x0).abs() < 1e-12);
  }
}

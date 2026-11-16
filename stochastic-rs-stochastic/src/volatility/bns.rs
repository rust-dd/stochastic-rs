//! # Barndorff-Nielsen-Shephard (BNS) stochastic volatility
//!
//! $$
//! \begin{aligned}
//! \frac{dS_t}{S_t} &= \mu\,dt + \sigma_t\,dW_t,\\
//! d\sigma^2_t &= -\lambda\,\sigma^2_t\,dt + dZ_{\lambda t},
//! \end{aligned}
//! $$
//!
//! where $Z = (Z_t)_{t \ge 0}$ is a **non-negative Lévy subordinator**
//! (Barndorff-Nielsen & Shephard 2001, *Journal of the Royal Statistical
//! Society* Series B, §2). The volatility process is the
//! [`non-Gaussian Ornstein-Uhlenbeck process`](https://doi.org/10.1111/1467-9868.00282)
//! with mean-reversion rate $\lambda$; the jumps of $Z$ provide the positive shocks that
//! keep $\sigma^2_t \ge 0$ at all times.
//!
//! ## Subordinator choice — compound Poisson + Gamma jumps
//!
//! We use the standard "BNS-Gamma" approximation: $Z$ is a compound
//! Poisson process with Poisson intensity $\nu$ (jumps per unit time on
//! the $\lambda$-rescaled clock) and i.i.d. Gamma jump sizes
//! $Y_j \sim \mathrm{Gamma}(\omega, 1)$. The discretised dynamics on a
//! grid of step $\Delta t$ are:
//!
//! 1. $N_i \sim \mathrm{Poisson}(\nu \cdot \lambda \cdot \Delta t)$ — number
//!    of jumps in $[t_i, t_{i+1}]$.
//! 2. $\Delta Z_i = \sum_{j=1}^{N_i} Y_j$, $Y_j \sim \mathrm{Gamma}(\omega, 1)$.
//! 3. $\sigma^2_{t_{i+1}} = e^{-\lambda \Delta t} \sigma^2_{t_i} + \Delta Z_i$.
//! 4. $S_{t_{i+1}} = S_{t_i} \exp\!\bigl((\mu - \tfrac{1}{2}\sigma^2_{t_i}) \Delta t + \sigma_{t_i} \sqrt{\Delta t}\,\varepsilon_i\bigr)$ with $\varepsilon_i \sim \mathcal{N}(0, 1)$.
//!
//! Pure-jump leverage ($\rho \ne 0$ between $W$ and the jumps of $Z$) is
//! not yet implemented; the Barndorff-Nielsen-Shephard 2001b
//! shifted-Brownian construction needs careful pre-allocation of two
//! correlated noise streams.
//!
//! ## References
//!
//! - Barndorff-Nielsen, O.E., Shephard, N. (2001), "Non-Gaussian
//!   Ornstein-Uhlenbeck-based models and some of their uses in financial
//!   economics", *Journal of the Royal Statistical Society* Series B,
//!   63(2), 167-241, DOI 10.1111/1467-9868.00282.
//! - Arai, T. (2015), "Local risk-minimization for Barndorff-Nielsen and
//!   Shephard models with volatility risk premium", arXiv:1506.01477.
//! - Roberts, G.O., Papaspiliopoulos, O., Dellaportas, P. (2004),
//!   "Bayesian inference for non-Gaussian Ornstein-Uhlenbeck stochastic
//!   volatility processes", *JRSS* B 66(2), 369-393.

use ndarray::Array1;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::FloatExt;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::gamma::SimdGamma;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::poisson::SimdPoisson;

use crate::device::Cpu;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// BNS-Gamma stochastic volatility model. See module documentation for
/// the SDE form and the discretisation.
pub struct Bns<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Initial spot.
  pub s0: T,
  /// Initial variance $\sigma^2_0 > 0$.
  pub sigma2_0: T,
  /// Mean-reversion rate $\lambda > 0$.
  pub lambda: T,
  /// Drift $\mu$ (risk-neutral $r - q$ when pricing).
  pub mu: T,
  /// Jump intensity $\nu \ge 0$ — expected number of jumps per unit time
  /// **on the $\lambda$-rescaled clock**.
  pub nu: T,
  /// Gamma shape parameter $\omega > 0$ for the i.i.d. jump sizes.
  pub jump_shape: T,
  /// Number of time steps.
  pub n: usize,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Seed strategy.
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Bns<T, S> {
  pub fn new(
    s0: T,
    sigma2_0: T,
    lambda: T,
    mu: T,
    nu: T,
    jump_shape: T,
    n: usize,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert!(
      s0.is_finite(),
      "s0 must satisfy `s0.is_finite()`, got s0 = {s0:?}"
    );
    assert!(
      sigma2_0.is_finite(),
      "sigma2_0 must satisfy `sigma2_0.is_finite()`, got sigma2_0 = {sigma2_0:?}"
    );
    assert!(
      lambda.is_finite(),
      "lambda must satisfy `lambda.is_finite()`, got lambda = {lambda:?}"
    );
    assert!(
      mu.is_finite(),
      "mu must satisfy `mu.is_finite()`, got mu = {mu:?}"
    );
    assert!(
      nu.is_finite(),
      "nu must satisfy `nu.is_finite()`, got nu = {nu:?}"
    );
    assert!(
      jump_shape.is_finite(),
      "jump_shape must satisfy `jump_shape.is_finite()`, got jump_shape = {jump_shape:?}"
    );
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    assert!(
      sigma2_0 > T::zero(),
      "sigma2_0 must satisfy `sigma2_0 > T::zero()`, got sigma2_0 = {sigma2_0:?}"
    );
    assert!(
      lambda > T::zero(),
      "lambda must satisfy `lambda > T::zero()`, got lambda = {lambda:?}"
    );
    assert!(
      nu >= T::zero(),
      "nu must satisfy `nu >= T::zero()`, got nu = {nu:?}"
    );
    assert!(
      jump_shape > T::zero(),
      "jump_shape must satisfy `jump_shape > T::zero()`, got jump_shape = {jump_shape:?}"
    );
    Self {
      backend: Cpu,
      s0,
      sigma2_0,
      lambda,
      mu,
      nu,
      jump_shape,
      n,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Bns<T, S, B> {
  /// `nu` as host and device read it; a write past `new` that breaks its rule fails here.
  fn checked_nu(&self) -> T {
    let nu = self.nu;
    assert!(
      nu.is_finite() && nu >= T::zero(),
      "nu must satisfy `nu.is_finite() && nu >= T::zero()`, got nu = {nu:?}"
    );
    nu
  }
}

/// The Euler engine's view of the Barndorff-Nielsen-Shephard model. The
/// variance decays exactly over the step, as the host takes it, and the
/// step's jump sum is one gamma draw whose shape is the jump count times a
/// single jump's — which is what the sum of that many gamma jumps is.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 2>
  for Bns<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let lam_dt = self.lambda * self.time_step();
    crate::euler::EulerSpec::BarndorffNielsenShephard {
      decay: (-lam_dt).exp(),
      mu: self.mu,
    }
  }

  fn initial_state(&self) -> [T; 4] {
    [self.s0, self.sigma2_0, T::zero(), T::zero()]
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  /// The subordinator's jump rate per unit time; none at `nu = 0`, where a launch skips the jumps.
  fn jump_intensity(&self) -> Option<T> {
    let nu = self.checked_nu();
    (nu != T::zero()).then(|| nu * self.lambda)
  }

  /// One gamma draw whose shape is the step's own jump count times the shape
  /// of a single jump.
  fn gamma_draws(&self) -> Option<crate::euler::GammaDraws<T>> {
    Some(crate::euler::GammaDraws {
      first: (T::zero(), T::one(), self.jump_shape),
      second: None,
    })
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> [Array1<T>; 2] {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Bns<T, S> { s0, sigma2_0, lambda, mu, nu, jump_shape, n, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Bns<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Bns<T, S, B> {
  /// `(log-stock path, variance σ² path)`.
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = BnsSampler<T, S>
  where
    Self: 's;

  /// Derives (not clones) `self.seed` into the returned sampler: the
  /// derived value is `self.seed`'s *mixed* next tick, not a raw snapshot,
  /// so chunk `i`'s basis and chunk `i+1`'s basis are hash-scrambled
  /// relative to each other rather than one raw stride apart.
  fn sampler(&self) -> BnsSampler<T, S> {
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1);
    let lam_dt = self.lambda * dt;
    let jump_rate_per_step = (self.checked_nu() * lam_dt).to_f64().unwrap();
    BnsSampler {
      n: self.n,
      s0: self.s0,
      sigma2_0: self.sigma2_0,
      mu: self.mu,
      jump_shape: self.jump_shape,
      dt,
      decay: (-lam_dt).exp(), // e^{-λ·Δt}
      counts: (jump_rate_per_step != 0.0).then(|| SimdPoisson::new(jump_rate_per_step)),
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine: on a device the variance's jump sum and the
  /// asset step together in the kernel, on the host devices it is this
  /// process's own sampler, chunked exactly as `ProcessExt` chunks.
  fn sample(&self) -> [Array1<T>; 2] {
    self.backend.system_sample(self)
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&[Array1<T>; 2]) -> R + Sync) -> Vec<R> {
    self.backend.system_paths_map(self, m, f)
  }

  fn sample_par(&self, m: usize) -> Vec<[Array1<T>; 2]> {
    self.backend.system_paths(self, m)
  }

  fn try_sample(&self) -> Result<[Array1<T>; 2], crate::device::DeviceError> {
    self.backend.try_system_sample(self)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<[Array1<T>; 2]>, crate::device::DeviceError> {
    self.backend.try_system_paths(self, m)
  }
}

/// Reusable [`Bns`] sampling state: the step's constants and its jump-count law, with the gamma,
/// normal and count streams rebuilt from the seed in one fixed order per fill.
#[doc(hidden)]
pub struct BnsSampler<T: FloatExt, S: SeedExt> {
  n: usize,
  s0: T,
  sigma2_0: T,
  mu: T,
  jump_shape: T,
  dt: T,
  decay: T,
  /// The per-step count `Poisson(ν·λ·Δt)`; `None` at a zero rate, where no step jumps.
  counts: Option<SimdPoisson<u64>>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> BnsSampler<T, S> {
  fn fill_paths(&mut self, s: &mut [T], sigma2: &mut [T]) {
    if self.n == 0 {
      return;
    }
    let dt = self.dt;
    let decay = self.decay;

    s[0] = self.s0;
    sigma2[0] = self.sigma2_0;

    // Each stream draws from `self.seed` in a fixed order, so a `Deterministic` seed reproduces the path.
    let mut jump_dist = SimdGamma::<T>::new(self.jump_shape, T::one()).seeded(&self.seed);
    let mut normal_dist = SimdNormal::<T>::new(T::zero(), T::one()).seeded(&self.seed);
    let mut rng = self.seed.rng();
    for i in 1..self.n {
      // 1. Number of jumps in [t_{i-1}, t_i] from the compound-Poisson
      //    subordinator.
      let n_jumps = self.counts.as_ref().map_or(0, |law| law.sample(&mut rng));

      // 2. Sum N Gamma(ω, 1) jump sizes to form ΔZ.
      let mut dz = T::zero();
      for _ in 0..n_jumps {
        dz += jump_dist.sample();
      }

      // 3. Variance update: σ²_t = e^{-λΔt} σ²_{t-Δt} + ΔZ.
      sigma2[i] = decay * sigma2[i - 1] + dz;

      // 4. Asset update under risk-neutral log-Euler.
      let v_prev = sigma2[i - 1];
      let eps = normal_dist.sample();
      let log_inc =
        (self.mu - v_prev * T::from_f64_fast(0.5)) * dt + v_prev.sqrt() * dt.sqrt() * eps;
      s[i] = s[i - 1] * log_inc.exp();
    }
  }
}

impl<T: FloatExt, S: SeedExt> crate::traits::Sealed for BnsSampler<T, S> {}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for BnsSampler<T, S> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [s, sigma2] = out;
    self.fill_paths(
      s.as_slice_mut().expect("Bns output must be contiguous"),
      sigma2
        .as_slice_mut()
        .expect("Bns output must be contiguous"),
    );
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    let mut s = Array1::<T>::zeros(self.n);
    let mut sigma2 = Array1::<T>::zeros(self.n);
    self.fill_paths(
      s.as_slice_mut().expect("contiguous"),
      sigma2.as_slice_mut().expect("contiguous"),
    );
    [s, sigma2]
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  /// Variance path stays strictly positive.
  #[test]
  fn bns_variance_non_negative() {
    let p = Bns::<f64>::new(100.0, 0.04, 2.0, 0.0, 5.0, 2.0, 512, Some(1.0), Unseeded);
    let [_s, v] = p.sample();
    assert!(v.iter().all(|x| *x >= 0.0));
    assert!(v.iter().any(|x| *x > 0.0));
  }

  /// Stock stays strictly positive (log-Euler structure).
  #[test]
  fn bns_stock_strictly_positive() {
    let p = Bns::<f64>::new(100.0, 0.04, 1.5, 0.05, 3.0, 2.0, 256, Some(1.0), Unseeded);
    let [s, _v] = p.sample();
    assert!(s.iter().all(|x| *x > 0.0));
  }

  /// Stationary mean of σ² under the BNS-Gamma model is
  /// $\nu\,\omega\cdot\lambda\Delta t / (1 - e^{-\lambda\Delta t}) \to \nu\,\omega$
  /// as $\Delta t \to 0$ (jump rate × mean jump size). A single 40-unit path
  /// with $\lambda = 1$ holds only ~15 effective samples, so its time-average
  /// has an ~11% standard error — far too wide for a 20% band to pass
  /// reliably. We average 128 independently seeded paths, shrinking the
  /// estimator's standard error to ~1% and keeping the check reproducible.
  #[test]
  fn bns_stationary_variance_matches_jump_intensity() {
    let nu = 4.0_f64;
    let omega = 1.5_f64;
    let expected = nu * omega;
    let paths = 128usize;
    let burn_in = 1024usize;
    let mean: f64 = (0..paths)
      .map(|i| {
        let p = Bns::<f64, Deterministic>::new(
          100.0,
          omega * nu,
          1.0,
          0.0,
          nu,
          omega,
          4096,
          Some(40.0),
          Deterministic::new(i as u64),
        );
        let [_s, v] = p.sample();
        v.iter().skip(burn_in).copied().sum::<f64>() / (v.len() - burn_in) as f64
      })
      .sum::<f64>()
      / paths as f64;
    assert!(
      (mean - expected).abs() / expected < 0.06,
      "BNS-Gamma stationary E[σ²] = {mean}, expected ≈ {expected}"
    );
  }

  /// A `Deterministic` seed makes the whole BNS path reproducible — the seed
  /// must reach the jump, Gamma and normal streams inside `sample`.
  #[test]
  fn bns_seeded_is_deterministic() {
    let mk = || {
      Bns::<f64, Deterministic>::new(
        100.0,
        0.04,
        2.0,
        0.0,
        5.0,
        2.0,
        256,
        Some(1.0),
        Deterministic::new(42),
      )
    };
    let [s1, v1] = mk().sample();
    let [s2, v2] = mk().sample();
    for i in 0..s1.len() {
      assert!((s1[i] - s2[i]).abs() < 1e-12);
      assert!((v1[i] - v2[i]).abs() < 1e-12);
    }
  }

  #[test]
  fn a_zero_jump_rate_decays_the_variance_exactly() {
    let p = Bns::<f64, Deterministic>::new(
      100.0,
      0.04,
      1.5,
      0.05,
      0.0,
      2.0,
      65,
      Some(1.0),
      Deterministic::new(4),
    );
    let [_, sigma2] = p.sample();
    let decay = (-1.5 / 64.0_f64).exp();
    for i in 1..65 {
      assert_eq!(
        sigma2[i],
        sigma2[i - 1] * decay,
        "step {i}: no jumps, so the OU decay is the whole step"
      );
    }
  }

  #[test]
  #[should_panic(expected = "nu must satisfy `nu.is_finite() && nu >= T::zero()`, got nu = NaN")]
  fn a_nan_nu_written_past_new_is_refused() {
    let mut p = Bns::<f64>::new(100.0, 0.04, 1.0, 0.0, 1.0, 1.0, 16, Some(1.0), Unseeded);
    p.nu = f64::NAN;
    let _ = p.sample();
  }

  #[test]
  fn jump_counts_follow_the_crate_poisson_law() {
    let p = Bns::<f64, Deterministic>::new(
      100.0,
      0.04,
      1.0,
      0.0,
      40.0,
      1.0,
      2,
      Some(1.0),
      Deterministic::new(11),
    );
    let jumps = p
      .sample_par(4000)
      .iter()
      .map(|[_, v]| v[1] - v[0] * (-1.0_f64).exp())
      .collect::<Vec<_>>();
    let mean = jumps.iter().sum::<f64>() / 4000.0;
    let var = jumps.iter().map(|z| (z - mean).powi(2)).sum::<f64>() / 3999.0;
    // Compound Poisson(40) of Exp(1) sizes, κ_k = 40·k!: ±5σ is ±0.71 (mean) and ±9.3 (variance).
    assert!(
      (mean - 40.0).abs() < 5.0 * (80.0_f64 / 4000.0).sqrt(),
      "E[ΔZ] = ν·λ·Δ·E[Y] = 40, got {mean}"
    );
    assert!(
      (var - 80.0).abs() < 5.0 * ((960.0 + 2.0 * 6400.0) / 4000.0_f64).sqrt(),
      "Var ΔZ = ν·λ·Δ·E[Y²] = 80, got {var}"
    );
  }
}

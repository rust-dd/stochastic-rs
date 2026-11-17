//! # Heston Model with Stochastic Correlation (HSCM)
//!
//! Teng, Ehrhardt & Günther (2016), "On the Heston model with stochastic
//! correlation", Int. J. Theor. Appl. Finance 19(6).
//!
//! $$
//! dS = rS\,dt + \sqrt{v}\,S\,dW^S \\
//! dv = \kappa_v(\mu_v - v)\,dt + \sigma_v\sqrt{v}\,dW^v \\
//! d\rho = \kappa_r(\mu_r - \tanh(X))\,dt + \sigma_r\,dW^\rho,\quad \rho=\tanh(X)
//! $$
//!
//! With correlations (Eq. 2.4, ρ₁ = 0 assumption):
//!   dW^S·dW^v = ρ_t dt,    dW^v·dW^ρ = ρ₂ dt

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Single-asset Heston model with stochastic price-vol correlation.
///
/// The correlation ρ_t between the asset log-return and variance
/// innovations follows a modified Ou process (Eq. 2.10):
///
/// dρ_t = (1−ρ²)\[κ_r(μ_r−ρ_t)−σ_r²ρ_t\]dt + (1−ρ²)σ_r dW^ρ
///
/// The log-price Cholesky decomposition (Eq. 2.7 with ρ₁=0):
///
/// dx = (r−½v)dt + ρ_t√v dW̃^ν + ρ₂√v dW̃^ρ + √(1−ρ_t²−ρ₂²)√v dW̃^x
///
/// Every field has a matching `with_*` builder setter, e.g.
/// `HestonStochCorr::new(..).with_kappa_v(2.5).with_rho2(-0.4)`.
#[derive(Debug, Clone)]
pub struct HestonStochCorr<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  // Market
  /// Risk-free rate.
  pub r: T,
  /// Initial spot price.
  pub s0: T,

  // Cir variance: dv = κ_v(μ_v − v)dt + σ_v√v dW^v
  /// Initial variance.
  pub v0: T,
  /// Mean-reversion speed of variance.
  pub kappa_v: T,
  /// Long-run variance level.
  pub mu_v: T,
  /// Vol-of-vol.
  pub sigma_v: T,

  // Stochastic correlation (modified Ou, tanh): ρ = tanh(X)
  /// Initial correlation (ρ₀ ∈ (−1, 1)).
  pub rho0: T,
  /// Mean-reversion speed of correlation.
  pub kappa_r: T,
  /// Long-run correlation level (μ_r ∈ (−1, 1)).
  pub mu_r: T,
  /// Volatility of correlation process.
  pub sigma_r: T,

  /// Constant correlation between dW^v and dW^ρ.
  pub rho2: T,

  /// Number of points sampled along each of the three paths.
  pub n: usize,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Seed strategy.
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

/// The three paths of a Heston model with stochastic correlation.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct HestonStochCorrPaths<T> {
  pub s: Array1<T>,
  pub v: Array1<T>,
  pub rho: Array1<T>,
}

impl<T> From<[Array1<T>; 3]> for HestonStochCorrPaths<T> {
  fn from([s, v, rho]: [Array1<T>; 3]) -> Self {
    Self { s, v, rho }
  }
}

impl<T: FloatExt, S: SeedExt> HestonStochCorr<T, S> {
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    r: T,
    s0: T,
    v0: T,
    kappa_v: T,
    mu_v: T,
    sigma_v: T,
    rho0: T,
    kappa_r: T,
    mu_r: T,
    sigma_r: T,
    rho2: T,
    n: usize,
    t: Option<T>,
    seed: S,
  ) -> Self {
    Self::check_r(r);
    Self::check_s0(s0);
    Self::check_v0(v0);
    Self::check_kappa_v(kappa_v);
    Self::check_mu_v(mu_v);
    Self::check_sigma_v(sigma_v);
    Self::check_rho0(rho0);
    Self::check_kappa_r(kappa_r);
    Self::check_mu_r(mu_r);
    Self::check_sigma_r(sigma_r);
    Self::check_rho2(rho2);
    Self::check_n(n);
    Self::check_t(t);
    Self {
      backend: Cpu,
      r,
      s0,
      v0,
      kappa_v,
      mu_v,
      sigma_v,
      rho0,
      kappa_r,
      mu_r,
      sigma_r,
      rho2,
      n,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> HestonStochCorr<T, S, B> {
  fn check_r(r: T) {
    require!(r.is_finite(), r);
  }

  fn check_s0(s0: T) {
    require!(s0.is_finite(), s0);
  }

  fn check_v0(v0: T) {
    require!(v0.is_finite(), v0);
  }

  fn check_kappa_v(kappa_v: T) {
    require!(kappa_v.is_finite(), kappa_v);
  }

  fn check_mu_v(mu_v: T) {
    require!(mu_v.is_finite(), mu_v);
  }

  fn check_sigma_v(sigma_v: T) {
    require!(sigma_v.is_finite(), sigma_v);
  }

  fn check_rho0(rho0: T) {
    require!(rho0.is_finite(), rho0);
  }

  fn check_kappa_r(kappa_r: T) {
    require!(kappa_r.is_finite(), kappa_r);
  }

  fn check_mu_r(mu_r: T) {
    require!(mu_r.is_finite(), mu_r);
  }

  fn check_sigma_r(sigma_r: T) {
    require!(sigma_r.is_finite(), sigma_r);
  }

  fn check_rho2(rho2: T) {
    require!(rho2.is_finite(), rho2);
  }

  fn check_n(n: usize) {
    require!(n >= 2, n);
  }

  fn check_t(t: Option<T>) {
    require!(t.is_none_or(|t| t > T::zero() && t.is_finite()), t);
  }

  /// Replace `r`, all else unchanged.
  pub fn with_r(mut self, r: T) -> Self {
    Self::check_r(r);
    self.r = r;
    self
  }

  /// Replace `s0`, all else unchanged.
  pub fn with_s0(mut self, s0: T) -> Self {
    Self::check_s0(s0);
    self.s0 = s0;
    self
  }

  /// Replace `v0`, all else unchanged.
  pub fn with_v0(mut self, v0: T) -> Self {
    Self::check_v0(v0);
    self.v0 = v0;
    self
  }

  /// Replace `kappa_v`, all else unchanged.
  pub fn with_kappa_v(mut self, kappa_v: T) -> Self {
    Self::check_kappa_v(kappa_v);
    self.kappa_v = kappa_v;
    self
  }

  /// Replace `mu_v`, all else unchanged.
  pub fn with_mu_v(mut self, mu_v: T) -> Self {
    Self::check_mu_v(mu_v);
    self.mu_v = mu_v;
    self
  }

  /// Replace `sigma_v`, all else unchanged.
  pub fn with_sigma_v(mut self, sigma_v: T) -> Self {
    Self::check_sigma_v(sigma_v);
    self.sigma_v = sigma_v;
    self
  }

  /// Replace `rho0`, all else unchanged.
  pub fn with_rho0(mut self, rho0: T) -> Self {
    Self::check_rho0(rho0);
    self.rho0 = rho0;
    self
  }

  /// Replace `kappa_r`, all else unchanged.
  pub fn with_kappa_r(mut self, kappa_r: T) -> Self {
    Self::check_kappa_r(kappa_r);
    self.kappa_r = kappa_r;
    self
  }

  /// Replace `mu_r`, all else unchanged.
  pub fn with_mu_r(mut self, mu_r: T) -> Self {
    Self::check_mu_r(mu_r);
    self.mu_r = mu_r;
    self
  }

  /// Replace `sigma_r`, all else unchanged.
  pub fn with_sigma_r(mut self, sigma_r: T) -> Self {
    Self::check_sigma_r(sigma_r);
    self.sigma_r = sigma_r;
    self
  }

  /// Replace `rho2`, all else unchanged.
  pub fn with_rho2(mut self, rho2: T) -> Self {
    Self::check_rho2(rho2);
    self.rho2 = rho2;
    self
  }

  /// Replace `n`, the grid points along the path, all else unchanged.
  pub fn with_points(mut self, n: usize) -> Self {
    Self::check_n(n);
    self.n = n;
    self
  }

  /// Replace the simulation horizon `t`, all else unchanged.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    Self::check_t(t);
    self.t = t;
    self
  }

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

/// The Euler engine's view of the stochastic-correlation Heston model. The
/// correlation is stepped on the unbounded variable it lives on, so `ρ₀`
/// enters as its inverse transform and the reported value is transformed
/// back.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 3>
  for HestonStochCorr<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::StochasticCorrelationHeston {
      kappa_r: self.kappa_r,
      mu_r: self.mu_r,
      sigma_r: self.sigma_r,
      kappa_v: self.kappa_v,
      mu_v: self.mu_v,
      sigma_v: self.sigma_v,
      r: self.r,
      rho2: self.rho2,
    }
  }

  fn initial_state(&self) -> [T; 4] {
    [
      self.s0,
      self.v0,
      self
        .rho0
        .clamp(T::from_f64_fast(-0.999), T::from_f64_fast(0.999))
        .atanh(),
      T::zero(),
    ]
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> HestonStochCorrPaths<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] HestonStochCorr<T, S> { r, s0, v0, kappa_v, mu_v, sigma_v, rho0, kappa_r, mu_r, sigma_r, rho2, n, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for HestonStochCorr<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for HestonStochCorr<T, S, B>
{
  type Output = HestonStochCorrPaths<T>;
  type Sampler<'s>
    = HestonStochCorrSampler<T, S>
  where
    Self: 's;

  /// Derives (not clones) `self.seed` into the returned sampler: the
  /// derived value is `self.seed`'s *mixed* next tick, not a raw snapshot,
  /// so chunk `i`'s basis and chunk `i+1`'s basis are hash-scrambled
  /// relative to each other rather than one raw stride apart.
  fn sampler(&self) -> HestonStochCorrSampler<T, S> {
    HestonStochCorrSampler {
      r: self.r,
      s0: self.s0,
      v0: self.v0,
      kappa_v: self.kappa_v,
      mu_v: self.mu_v,
      sigma_v: self.sigma_v,
      rho0: self.rho0,
      kappa_r: self.kappa_r,
      mu_r: self.mu_r,
      sigma_r: self.sigma_r,
      rho2: self.rho2,
      n: self.n,
      t: self.t,
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine: on a device every component steps in the
  /// kernel, on the host devices it is this process's own sampler, chunked
  /// exactly as `ProcessExt` chunks.
  fn sample(&self) -> HestonStochCorrPaths<T> {
    self.backend.system_sample(self)
  }

  fn sample_map<R: Send>(
    &self,
    m: usize,
    f: impl Fn(&HestonStochCorrPaths<T>) -> R + Sync,
  ) -> Vec<R> {
    self.backend.system_paths_map(self, m, f)
  }

  fn sample_par(&self, m: usize) -> Vec<HestonStochCorrPaths<T>> {
    self.backend.system_paths(self, m)
  }

  fn try_sample(&self) -> Result<HestonStochCorrPaths<T>, crate::device::DeviceError> {
    self.backend.try_system_sample(self)
  }

  fn try_sample_par(
    &self,
    m: usize,
  ) -> Result<Vec<HestonStochCorrPaths<T>>, crate::device::DeviceError> {
    self.backend.try_system_paths(self, m)
  }
}

/// Reusable [`HestonStochCorr`] sampling state: owns the seed source so a
/// Monte-Carlo loop reuses all three output buffers. The three Gaussian streams
/// are rebuilt per call from the (derived) seed, exactly as the legacy `sample`
/// body did.
#[doc(hidden)]
#[derive(Debug)]
pub struct HestonStochCorrSampler<T: FloatExt, S: SeedExt> {
  r: T,
  s0: T,
  v0: T,
  kappa_v: T,
  mu_v: T,
  sigma_v: T,
  rho0: T,
  kappa_r: T,
  mu_r: T,
  sigma_r: T,
  rho2: T,
  n: usize,
  t: Option<T>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> HestonStochCorrSampler<T, S> {
  fn fill_paths(&mut self, s_path: &mut [T], v_path: &mut [T], rho_path: &mut [T]) {
    let n_steps = self.n.saturating_sub(1);
    let dt = if n_steps > 0 {
      self.t.unwrap_or(T::one()) / T::from_usize_(n_steps)
    } else {
      T::zero()
    };
    let sqrt_dt = dt.sqrt();
    let half = T::from_f64_fast(0.5);
    let zero = T::zero();

    // 3 independent noise streams: dW̃^ν, dW̃^ρ, dW̃^x
    let gen_noise = |seed: &S| -> Array1<T> {
      let mut gn = Array1::<T>::zeros(n_steps);
      if let Some(slice) = gn.as_slice_mut() {
        let mut normal = SimdNormal::<T>::new(zero, sqrt_dt).seeded(seed);
        normal.fill_slice(slice);
      }
      gn
    };

    let dw_v = gen_noise(&self.seed); // variance Bm
    let dw_rho = gen_noise(&self.seed); // correlation Bm
    let dw_x = gen_noise(&self.seed); // independent price Bm

    if self.n == 0 {
      return;
    }

    s_path[0] = self.s0;
    v_path[0] = self.v0;

    // Correlation: work in X-space, X₀ = atanh(ρ₀)
    let mut x_corr = self
      .rho0
      .clamp(T::from_f64_fast(-0.999), T::from_f64_fast(0.999))
      .atanh();
    rho_path[0] = x_corr.tanh();

    for i in 1..self.n {
      // Correlation dynamics (modified Ou in X-space)
      let corr_drift = self.kappa_r * (self.mu_r - x_corr.tanh());
      x_corr = x_corr + corr_drift * dt + self.sigma_r * dw_rho[i - 1];
      let rho_t = x_corr.tanh();
      rho_path[i] = rho_t;

      // Variance dynamics (Cir)
      let v_prev = v_path[i - 1].max(zero);
      v_path[i] = (v_prev
        + self.kappa_v * (self.mu_v - v_prev) * dt
        + self.sigma_v * v_prev.sqrt() * dw_v[i - 1])
        .max(zero);

      // Log-price dynamics (Cholesky, Eq. 2.7 with ρ₁=0)
      // dx = (r − ½v)dt + ρ_t√v dW̃^ν + ρ₂√v dW̃^ρ + √(1−ρ_t²−ρ₂²)√v dW̃^x
      let rho2_sq = self.rho2 * self.rho2;
      let indep_coeff = (T::one() - rho_t * rho_t - rho2_sq).max(zero).sqrt();
      let sqrt_v = v_prev.sqrt();

      let log_inc = (self.r - half * v_prev) * dt
        + rho_t * sqrt_v * dw_v[i - 1]
        + self.rho2 * sqrt_v * dw_rho[i - 1]
        + indep_coeff * sqrt_v * dw_x[i - 1];

      s_path[i] = s_path[i - 1] * log_inc.exp();
    }
  }
}

impl<T: FloatExt, S: SeedExt> crate::traits::Sealed for HestonStochCorrSampler<T, S> {}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for HestonStochCorrSampler<T, S> {
  type Output = HestonStochCorrPaths<T>;

  fn sample_into(&mut self, out: &mut HestonStochCorrPaths<T>) {
    let HestonStochCorrPaths { s, v, rho } = out;
    self.fill_paths(
      s.as_slice_mut()
        .expect("HestonStochCorr output must be contiguous"),
      v.as_slice_mut()
        .expect("HestonStochCorr output must be contiguous"),
      rho
        .as_slice_mut()
        .expect("HestonStochCorr output must be contiguous"),
    );
  }

  fn sample(&mut self) -> HestonStochCorrPaths<T> {
    let mut s = Array1::<T>::zeros(self.n);
    let mut v = Array1::<T>::zeros(self.n);
    let mut rho = Array1::<T>::zeros(self.n);
    self.fill_paths(
      s.as_slice_mut().expect("contiguous"),
      v.as_slice_mut().expect("contiguous"),
      rho.as_slice_mut().expect("contiguous"),
    );
    HestonStochCorrPaths { s, v, rho }
  }
}

#[cfg(test)]
mod tests;

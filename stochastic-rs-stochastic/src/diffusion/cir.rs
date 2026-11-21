//! Cox–Ingersoll–Ross square-root diffusion; its `VarianceBoundary` treats a step below zero.
//!
//! $$
//! dX_t=\kappa(\theta-X_t)\,dt+\sigma\sqrt{X_t}\,dW_t
//! $$
//!
//! Reference: Cox, Ingersoll & Ross (1985), Econometrica 53(2), 385–407, DOI 10.2307/1911242
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::device::DeviceError;
use crate::euler::EulerBackend;
use crate::params::VarianceBoundary;
use crate::precondition::check_n;
use crate::precondition::check_t;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

mod transition;

/// CIR in this crate's names: [`theta`](Self::theta) is the speed κ and [`mu`](Self::mu) the
/// long-run level θ of `dX = κ(θ − X) dt + σ √X dW`.
#[derive(Debug, Clone)]
pub struct Cir<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Mean-reversion speed (κ in the SDE). Controls how fast `X` is pulled
  /// back toward [`mu`](Self::mu).
  pub theta: T,
  /// Long-run mean level (θ in the SDE). The value `X` reverts to as
  /// `t → ∞`.
  pub mu: T,
  /// Diffusion scale σ multiplying `√X_t dW_t` (σ in the SDE).
  pub sigma: T,
  /// Number of points sampled along the CIR path.
  pub n: usize,
  /// Initial value; `None` starts at zero.
  pub x0: Option<T>,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Treatment of a negative square-root state after a step; the Feller condition is not required.
  pub boundary: VarianceBoundary,
  /// Seed: [`Unseeded`] or a [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

/// Every field has a `with_*` setter (`Cir::default().with_theta(3.0).with_sigma(0.3)`); nothing
/// is cached, since `sampler()` builds its Gaussian stream from `self` on every call.
impl<T: FloatExt, S: SeedExt> Cir<T, S> {
  /// A sub-Feller `2·theta·mu < sigma²` is accepted; it warns unless reflected.
  /// Panics unless `n >= 2`, a given `t` is positive and finite, and every parameter is finite.
  pub fn new(
    theta: T,
    mu: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    boundary: VarianceBoundary,
    seed: S,
  ) -> Self {
    Self::check_theta(theta);
    Self::check_mu(mu);
    Self::check_sigma(sigma);
    check_n(n);
    Self::check_x0(x0);
    check_t(t);
    if T::from_usize_(2) * theta * mu < sigma.powi(2) && boundary != VarianceBoundary::Reflect {
      log::warn!(
        "Cir::new: Feller condition violated (2*theta*mu < sigma^2) \
         under VarianceBoundary::{boundary:?}; the path sits at zero on every \
         boundary hit instead of reflecting"
      );
    }

    Self {
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      boundary,
      seed,
      backend: Cpu,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Cir<T, S, B> {
  fn check_theta(theta: T) {
    require!(theta.is_finite(), theta);
  }

  fn check_mu(mu: T) {
    require!(mu.is_finite(), mu);
  }

  fn check_sigma(sigma: T) {
    require!(sigma.is_finite(), sigma);
  }

  fn check_x0(x0: Option<T>) {
    require!(x0.is_none_or(|v| v.is_finite()), x0);
  }

  /// Replace `theta`; panics unless `theta` is finite.
  pub fn with_theta(mut self, theta: T) -> Self {
    Self::check_theta(theta);
    self.theta = theta;
    self
  }

  /// Replace `mu`; panics unless `mu` is finite.
  pub fn with_mu(mut self, mu: T) -> Self {
    Self::check_mu(mu);
    self.mu = mu;
    self
  }

  /// Replace `sigma`; panics unless `sigma` is finite.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    Self::check_sigma(sigma);
    self.sigma = sigma;
    self
  }

  /// Replace `x0`; panics unless `x0` is finite.
  pub fn with_x0(mut self, x0: Option<T>) -> Self {
    Self::check_x0(x0);
    self.x0 = x0;
    self
  }

  /// Replace `boundary`.
  pub fn with_boundary(mut self, boundary: VarianceBoundary) -> Self {
    self.boundary = boundary;
    self
  }

  /// Replace `n`, the grid points along the path; panics unless `n >= 2`.
  pub fn with_points(mut self, n: usize) -> Self {
    check_n(n);
    self.n = n;
    self
  }

  /// Replace the simulation horizon `t`; panics unless `t`, when given, is positive and finite.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    check_t(t);
    self.t = t;
    self
  }

  /// Replace the seed strategy's value.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }

  /// Exact transition density to `r_next` after `t` from `r_t`; NaN at `t = 0`, for a negative
  /// state or outside the domain.
  pub fn transition_pdf(&self, t: T, r_t: T, r_next: T) -> T {
    let [theta, mu, sigma, t, r_t, r_next] =
      [self.theta, self.mu, self.sigma, t, r_t, r_next].map(|v| v.to_f64().unwrap());
    T::from_f64_fast(transition::transition_pdf(theta, mu, sigma, t, r_t, r_next))
  }

  /// Logarithm of [`transition_pdf`](Self::transition_pdf), finite where that underflows to `0`;
  /// NaN where it is NaN, `−∞` into `r_next = 0` when `2θμ > σ²`.
  pub fn transition_ln_pdf(&self, t: T, r_t: T, r_next: T) -> T {
    let [theta, mu, sigma, t, r_t, r_next] =
      [self.theta, self.mu, self.sigma, t, r_t, r_next].map(|v| v.to_f64().unwrap());
    T::from_f64_fast(transition::transition_ln_pdf(
      theta, mu, sigma, t, r_t, r_next,
    ))
  }

  /// One exact transition draw after `t` from `r_t`; the conditional mean once the χ² parameters
  /// overflow.
  pub fn sample_transition<S2: SeedExt>(&self, t: T, r_t: T, seed: &S2) -> T {
    let [theta, mu, sigma, t, r_t] =
      [self.theta, self.mu, self.sigma, t, r_t].map(|v| v.to_f64().unwrap());
    T::from_f64_fast(transition::sample_transition(
      theta, mu, sigma, t, r_t, seed,
    ))
  }

  /// Stationary density at `r`: Gamma with shape `2θμ/σ²` and rate `2θ/σ²`; NaN for a negative
  /// state or outside the domain.
  pub fn stationary_pdf(&self, r: T) -> T {
    let [theta, mu, sigma, r] = [self.theta, self.mu, self.sigma, r].map(|v| v.to_f64().unwrap());
    T::from_f64_fast(transition::stationary_pdf(theta, mu, sigma, r))
  }
}

/// θ=2.5, μ=0.04, σ=0.2, x₀=0.04, fully truncated, a textbook Cir with `2θμ = 0.2 ≥ σ² = 0.04`;
/// t=1, 252 points, a trading year of daily closes.
impl<T: FloatExt> Default for Cir<T, Unseeded> {
  fn default() -> Self {
    Self::new(
      T::from_f64_fast(2.5),
      T::from_f64_fast(0.04),
      T::from_f64_fast(0.2),
      252,
      Some(T::from_f64_fast(0.04)),
      Some(T::one()),
      VarianceBoundary::default(),
      Unseeded,
    )
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Cir<T, S> { theta, mu, sigma, n, x0, t, boundary, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: EulerBackend<T>> crate::traits::Sealed for Cir<T, S, B> {}

impl<T: FloatExt, S: SeedExt, B: EulerBackend<T>> ProcessExt<T> for Cir<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = CirSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> CirSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    CirSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      theta: self.theta,
      mu: self.mu,
      diff_scale: self.sigma,
      boundary: self.boundary,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt()).seeded(&self.seed),
    }
  }

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

  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    self.backend.euler_paths(self, m)
  }

  fn try_sample(&self) -> Result<Array1<T>, DeviceError> {
    self.backend.try_sample(self)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, DeviceError> {
    self.backend.try_euler_paths(self, m)
  }
}

/// Reusable [`Cir`] sampling state.
#[doc(hidden)]
#[derive(Debug)]
pub struct CirSampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  theta: T,
  mu: T,
  diff_scale: T,
  boundary: VarianceBoundary,
  normal: Seeded<SimdNormal<T>>,
}

impl<T: FloatExt> CirSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.boundary.observed(self.x0);
    if out.len() == 1 {
      return;
    }
    let tail = &mut out[1..];
    self.normal.fill_slice(tail);
    let mut prev = self.boundary.carried(self.x0);
    for z in tail.iter_mut() {
      let x = self.boundary.observed(prev);
      let dcir = self.theta * (self.mu - x) * self.dt + self.diff_scale * x.sqrt() * *z;
      prev = self.boundary.carried(prev + dcir);
      *z = self.boundary.observed(prev);
    }
  }
}

impl<T: FloatExt> crate::traits::Sealed for CirSampler<T> {}

impl<T: FloatExt> PathSampler<T> for CirSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("Cir output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyCir, Cir,
  sig: (theta, mu, sigma, n, x0=None, t=None, boundary=None, seed=None, dtype=None),
  params: (theta: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>, boundary: Option<crate::python_params::PyVarianceBoundary> => crate::python_params::boundary),
  device
);

#[cfg(test)]
mod tests;

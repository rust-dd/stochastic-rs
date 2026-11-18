//! Shifted lognormal for negative rates: $Y_t = S_t + \beta$ is a GBM (second display). The
//! sampler steps `diffusion::gbm::Gbm`'s own recursion on $Y$, so `beta = 0` is `Gbm` bit for bit.
//!
//! $$
//! dS_t=\mu(S_t+\beta)\,dt+\sigma(S_t+\beta)\,dW_t
//! $$
//!
//! $$
//! S_t+\beta=(S_0+\beta)\exp\!\left(\left(\mu-\tfrac{\sigma^2}{2}\right)t+\sigma W_t\right)
//! $$

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::precondition::check_n;
use crate::precondition::check_t;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Displaced (shifted-lognormal) diffusion; the module doc has the closed form and the
/// bit-for-bit relationship to [`Gbm`](crate::diffusion::gbm::Gbm) at `beta = 0`.
#[derive(Debug, Clone)]
pub struct DisplacedDiffusion<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Constant proportional drift rate μ of the shifted variable `S_t + beta`; no mean reversion,
  /// as [`Gbm::mu`](crate::diffusion::gbm::Gbm::mu).
  pub mu: T,
  /// Diffusion scale σ multiplying `(S_t + beta) dW_t`.
  pub sigma: T,
  /// Shift β applied to the state before the GBM dynamics act; `beta = 0` is
  /// [`Gbm`](crate::diffusion::gbm::Gbm), and `beta > 0` lets `S_t` go negative.
  pub beta: T,
  /// Number of points sampled along the displaced-diffusion path.
  pub n: usize,
  /// Initial value S₀.
  pub x0: T,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or
  /// the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

/// Every field has a `with_*` setter (`DisplacedDiffusion::default().with_beta(10.0)`); nothing
/// is cached, since `sampler()` builds its Gaussian source from `self` on every call.
impl<T: FloatExt, S: SeedExt> DisplacedDiffusion<T, S> {
  /// Panics unless `n >= 2`, a given `t` is positive and finite, and every parameter is finite.
  pub fn new(mu: T, sigma: T, beta: T, n: usize, x0: T, t: Option<T>, seed: S) -> Self {
    Self::check_mu(mu);
    Self::check_sigma(sigma);
    Self::check_beta(beta);
    check_n(n);
    Self::check_x0(x0);
    check_t(t);
    Self {
      backend: Cpu,
      mu,
      sigma,
      beta,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> DisplacedDiffusion<T, S, B> {
  fn check_mu(mu: T) {
    require!(mu.is_finite(), mu);
  }

  fn check_sigma(sigma: T) {
    require!(sigma.is_finite(), sigma);
  }

  fn check_beta(beta: T) {
    require!(beta.is_finite(), beta);
  }

  fn check_x0(x0: T) {
    require!(x0.is_finite(), x0);
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

  /// Replace `beta`; panics unless `beta` is finite.
  pub fn with_beta(mut self, beta: T) -> Self {
    Self::check_beta(beta);
    self.beta = beta;
    self
  }

  /// Replace `x0`; panics unless `x0` is finite.
  pub fn with_x0(mut self, x0: T) -> Self {
    Self::check_x0(x0);
    self.x0 = x0;
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

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

/// [`Gbm`](crate::diffusion::gbm::Gbm)'s `Default` (μ=0.05, σ=0.2, x₀=100, t=1) shifted by β=30,
/// the closed-form mean test's shift; 252 points, a trading year of daily closes.
impl<T: FloatExt> Default for DisplacedDiffusion<T, Unseeded> {
  fn default() -> Self {
    Self::new(
      T::from_f64_fast(0.05),
      T::from_f64_fast(0.2),
      T::from_f64_fast(30.0),
      252,
      T::from_f64_fast(100.0),
      Some(T::one()),
      Unseeded,
    )
  }
}

/// The Euler engine's view: the state is the shifted `Y = S + β`, which starts at `X₀ + β` and is
/// reported with the shift taken back out.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for DisplacedDiffusion<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::Displaced {
      mu: self.mu,
      sigma: self.sigma,
      beta: self.beta,
    }
  }

  fn initial_value(&self) -> T {
    self.x0 + self.beta
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

  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] DisplacedDiffusion<T, S> { mu, sigma, beta, n, x0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for DisplacedDiffusion<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for DisplacedDiffusion<T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = DisplacedDiffusionSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> DisplacedDiffusionSampler<T> {
    // Same `n_increments` / `dt` derivation as `Gbm::sampler` — required for
    // the `beta = 0` path to consume the Gaussian stream identically.
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    DisplacedDiffusionSampler {
      n: self.n,
      x0: self.x0,
      beta: self.beta,
      drift_scale: self.mu * dt,
      diff_scale: self.sigma,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt()).seeded(&self.seed),
    }
  }

  /// Through the Euler engine: a device runs the recursion in its kernel, the host runs this
  /// process's own sampler, chunked as `ProcessExt` chunks.
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

  fn try_sample(&self) -> Result<Array1<T>, crate::device::DeviceError> {
    self.backend.try_sample(self)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, crate::device::DeviceError> {
    self.backend.try_euler_paths(self, m)
  }
}

/// Reusable [`DisplacedDiffusion`] sampling state: the Euler scales and the owned Gaussian
/// source, as [`GbmSampler`](crate::diffusion::gbm::GbmSampler) with the `beta` shift.
#[doc(hidden)]
#[derive(Debug)]
pub struct DisplacedDiffusionSampler<T: FloatExt> {
  n: usize,
  x0: T,
  beta: T,
  drift_scale: T,
  diff_scale: T,
  normal: Seeded<SimdNormal<T>>,
}

impl<T: FloatExt> DisplacedDiffusionSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }
    let tail = &mut out[1..];
    self.normal.fill_slice(tail);
    // `GbmSampler::fill_path`'s recursion on the shifted `Y = S + beta`, reported as `Y - beta`;
    // at `beta = 0` it is that recursion term for term.
    let mut prev_shifted = self.x0 + self.beta;
    for z in tail.iter_mut() {
      let next_shifted =
        prev_shifted + self.drift_scale * prev_shifted + self.diff_scale * prev_shifted * *z;
      *z = next_shifted - self.beta;
      prev_shifted = next_shifted;
    }
  }
}

impl<T: FloatExt> crate::traits::Sealed for DisplacedDiffusionSampler<T> {}

impl<T: FloatExt> PathSampler<T> for DisplacedDiffusionSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("DisplacedDiffusion output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyDisplacedDiffusion, DisplacedDiffusion,
  sig: (mu, sigma, beta, n, x0, t=None, seed=None, dtype=None),
  params: (mu: f64, sigma: f64, beta: f64, n: usize, x0: f64, t: Option<f64>),
  device
);

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;
  use crate::diffusion::gbm::Gbm;

  /// β=0 must reproduce Gbm exactly: same parameters, same seed ⇒
  /// bit-identical path.
  #[test]
  fn displaced_diffusion_beta_zero_equals_gbm() {
    let mu = 0.07;
    let sigma = 0.25;
    let x0 = 50.0;
    let n = 256;
    let t = 2.0;
    let seed = 2718u64;

    let gbm = Gbm::<f64, _>::new(mu, sigma, n, x0, Some(t), Deterministic::new(seed));
    let dd =
      DisplacedDiffusion::<f64, _>::new(mu, sigma, 0.0, n, x0, Some(t), Deterministic::new(seed));

    let gbm_path = gbm.sample();
    let dd_path = dd.sample();

    assert_eq!(gbm_path.len(), dd_path.len());
    for (g, d) in gbm_path.iter().zip(dd_path.iter()) {
      assert_eq!(g.to_bits(), d.to_bits(), "gbm={g} displaced={d}");
    }
  }

  /// E[S_t] = (S_0+β)e^{μt} − β for the displaced lognormal.
  #[test]
  fn displaced_diffusion_mean_matches_closed_form() {
    let mu = 0.05_f64;
    let sigma = 0.2;
    let beta = 30.0;
    let x0 = 100.0;
    let t = 1.0;
    let n = 200;
    let paths = 20_000;
    let expected = (x0 + beta) * (mu * t).exp() - beta;

    let best_rel_err = [2718u64, 999, 42]
      .into_iter()
      .map(|seed| {
        let dd = DisplacedDiffusion::<f64, _>::new(
          mu,
          sigma,
          beta,
          n,
          x0,
          Some(t),
          Deterministic::new(seed),
        );
        let mean = dd
          .sample_par(paths)
          .iter()
          .map(|path| *path.last().unwrap())
          .sum::<f64>()
          / paths as f64;
        (mean - expected).abs() / expected.abs()
      })
      .fold(f64::INFINITY, f64::min);

    assert!(
      best_rel_err <= 2e-2,
      "best-of-3 relative error {best_rel_err} exceeds 2e-2 (expected {expected})"
    );
  }

  /// Same seed twice must be bit-identical.
  #[test]
  fn displaced_diffusion_is_deterministic() {
    let dd1 =
      DisplacedDiffusion::<f64, _>::new(0.05, 0.2, 0.1, 50, 1.0, Some(1.0), Deterministic::new(42))
        .sample();
    let dd2 =
      DisplacedDiffusion::<f64, _>::new(0.05, 0.2, 0.1, 50, 1.0, Some(1.0), Deterministic::new(42))
        .sample();
    assert_eq!(dd1, dd2);
  }
}

//! # Rough Heston
//!
//! $$
//! \begin{aligned} dS_t &= \mu S_t\,dt + \sqrt{V_t}\,S_t\,dW^s_t,\qquad d\langle W^s, W^v\rangle_t = \rho\,dt \\ V_t &= V_0 + \frac{1}{\Gamma(H+1/2)}\int_0^t (t-s)^{H-1/2} \kappa(\theta - V_s)\,ds + \frac{1}{\Gamma(H+1/2)}\int_0^t (t-s)^{H-1/2} \nu\sqrt{V_s}\,dW^v_s \end{aligned}
//! $$
//!
//! The variance, a Riemann–Liouville Volterra–CIR, steps by one Markov lift with drift
//! $\kappa(\theta-V)$ and diffusion $\nu\sqrt{V^+}$; the price steps by Euler on $W^s$.
//!
//! Reference: Bilokon & Wong (2026) §5.5; El Euch & Rosenbaum (2019), Math. Finance 29, 3–38.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use super::kernel::RlKernel;
use super::markov_lift::MarkovLift;
use crate::device::Cpu;
use crate::noise::cgns::Cgns;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Rough Heston model with Volterra-Cir variance and correlated Gbm asset.
#[derive(Clone)]
pub struct RlHeston<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Hurst exponent $H \in (0, 1/2)$ of the variance kernel.
  pub hurst: T,
  /// Initial spot.
  pub s0: T,
  /// Initial variance.
  pub v0: T,
  /// Variance mean-reversion speed $\kappa$.
  pub kappa: T,
  /// Long-run variance $\theta$.
  pub theta: T,
  /// Volatility of variance $\nu$.
  pub nu: T,
  /// Correlation between $W^s$ and $W^v$.
  pub rho: T,
  /// Asset drift $\mu$ (risk-free rate under $\mathbb{Q}$).
  pub mu: T,
  /// Number of simulation points.
  pub n: usize,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Quadrature degree passed through to the kernel.
  pub degree: Option<usize>,
  /// Seed strategy.
  pub seed: S,
  cgns: Cgns<T>,
  markov: MarkovLift<T>,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

fn build_markov<T: FloatExt>(
  hurst: T,
  n: usize,
  t: Option<T>,
  degree: Option<usize>,
) -> MarkovLift<T> {
  let dt = t.unwrap_or(T::one()) / T::from_usize_(n - 1);
  let deg = degree.unwrap_or_else(|| RlKernel::<T>::default_degree(n));
  let kernel = RlKernel::<T>::new(hurst, deg);
  MarkovLift::new(kernel, dt)
}

impl<T: FloatExt, S: SeedExt> RlHeston<T, S> {
  #[must_use]
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    hurst: T,
    s0: T,
    v0: T,
    kappa: T,
    theta: T,
    nu: T,
    rho: T,
    mu: T,
    n: usize,
    t: Option<T>,
    degree: Option<usize>,
    seed: S,
  ) -> Self {
    assert!(
      hurst.is_finite(),
      "hurst must satisfy `hurst.is_finite()`, got hurst = {hurst:?}"
    );
    assert!(
      s0.is_finite(),
      "s0 must satisfy `s0.is_finite()`, got s0 = {s0:?}"
    );
    assert!(
      v0.is_finite(),
      "v0 must satisfy `v0.is_finite()`, got v0 = {v0:?}"
    );
    assert!(
      kappa.is_finite(),
      "kappa must satisfy `kappa.is_finite()`, got kappa = {kappa:?}"
    );
    assert!(
      theta.is_finite(),
      "theta must satisfy `theta.is_finite()`, got theta = {theta:?}"
    );
    assert!(
      nu.is_finite(),
      "nu must satisfy `nu.is_finite()`, got nu = {nu:?}"
    );
    assert!(
      rho.is_finite(),
      "rho must satisfy `rho.is_finite()`, got rho = {rho:?}"
    );
    assert!(
      mu.is_finite(),
      "mu must satisfy `mu.is_finite()`, got mu = {mu:?}"
    );
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    assert!(
      kappa >= T::zero(),
      "kappa must satisfy `kappa >= T::zero()`, got kappa = {kappa:?}"
    );
    assert!(
      theta >= T::zero(),
      "theta must satisfy `theta >= T::zero()`, got theta = {theta:?}"
    );
    assert!(
      nu >= T::zero(),
      "nu must satisfy `nu >= T::zero()`, got nu = {nu:?}"
    );
    assert!(
      v0 >= T::zero(),
      "v0 must satisfy `v0 >= T::zero()`, got v0 = {v0:?}"
    );
    Self {
      backend: Cpu,
      hurst,
      s0,
      v0,
      kappa,
      theta,
      nu,
      rho,
      mu,
      n,
      t,
      degree,
      seed,
      cgns: Cgns::new(rho, n - 1, t, Unseeded),
      markov: build_markov(hurst, n, t, degree),
    }
  }
}

/// The Euler engine's view: the variance's lift goes to the device as its node
/// constants and boundary terms — started at `v0`, floored as the host floors
/// it — and the family steps the spot by Euler beside it, the two shocks
/// correlated in the step exactly as the host's correlated pair is.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 2>
  for RlHeston<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::RiemannLiouvilleHeston {
      mu: self.mu,
      kappa: self.kappa,
      theta: self.theta,
      nu: self.nu,
      rho: self.rho,
    }
  }

  fn initial_state(&self) -> [T; 4] {
    [self.s0, self.v0.max(T::zero()), T::zero(), T::zero()]
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  fn time_step(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn lift_spec(&self) -> Option<crate::euler::LiftSpec<'_, T>> {
    let lift = self.markov.lift();
    Some(crate::euler::LiftSpec {
      decay: lift.exp_neg_x_dt.as_slice().expect("contiguous"),
      weight: lift.we.as_slice().expect("contiguous"),
      drift_scale: lift.one_minus_e_over_x.as_slice().expect("contiguous"),
      drift_boundary: lift.drift_boundary,
      diffusion_boundary: lift.diffusion_boundary,
      x0: self.v0.max(T::zero()),
    })
  }

  fn host_sample(&self) -> [Array1<T>; 2] {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] RlHeston<T, S> { hurst, s0, v0, kappa, theta, nu, rho, mu, n, t, degree, seed, cgns, markov } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for RlHeston<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for RlHeston<T, S, B>
{
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = RlHestonSampler<'s, T, S>
  where
    Self: 's;

  /// Derives (not clones) `self.seed` into the returned sampler: the
  /// derived value is `self.seed`'s *mixed* next tick, not a raw snapshot,
  /// so chunk `i`'s basis and chunk `i+1`'s basis are hash-scrambled
  /// relative to each other rather than one raw stride apart. `fill_paths`
  /// then uses this owned seed directly (no further derive) — exactly one
  /// derive from `self.seed` per chunk, matching what the legacy per-call
  /// `derive()` consumed.
  fn sampler(&self) -> RlHestonSampler<'_, T, S> {
    RlHestonSampler {
      n: self.n,
      s0: self.s0,
      v0: self.v0.max(T::zero()),
      kappa: self.kappa,
      theta: self.theta,
      nu: self.nu,
      mu: self.mu,
      dt: self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1),
      cgns: self.cgns,
      markov: &self.markov,
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine: on a device the lift and both components run
  /// in the kernel, on the host devices it is this process's own sampler,
  /// chunked exactly as [`ProcessExt`] chunks.
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

/// Reusable [`RlHeston`] sampling state: owns the correlated-Gaussian generator
/// and the seed source, and borrows the prebuilt Markov-lift so a Monte-Carlo
/// loop reuses both output buffers without rebuilding the kernel.
#[doc(hidden)]
pub struct RlHestonSampler<'a, T: FloatExt, S: SeedExt> {
  n: usize,
  s0: T,
  v0: T,
  kappa: T,
  theta: T,
  nu: T,
  mu: T,
  dt: T,
  cgns: Cgns<T>,
  markov: &'a MarkovLift<T>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> RlHestonSampler<'_, T, S> {
  fn fill_paths(&mut self, s: &mut [T], v_out: &mut [T]) {
    if self.n == 0 {
      return;
    }
    let dt = self.dt;
    let [dw_s, dw_v] = self.cgns.sample_impl(&self.seed);

    let kappa = self.kappa;
    let theta = self.theta;
    let nu = self.nu;
    let v = self.markov.simulate(
      self.v0,
      |vv| kappa * (theta - vv.max(T::zero())),
      |vv| nu * vv.max(T::zero()).sqrt(),
      dw_v.as_slice().expect("dw_v must be contiguous"),
    );

    s[0] = self.s0;
    for i in 1..self.n {
      let v_prev = v[i - 1].max(T::zero());
      s[i] = s[i - 1] + self.mu * s[i - 1] * dt + s[i - 1] * v_prev.sqrt() * dw_s[i - 1];
    }

    for i in 0..self.n {
      v_out[i] = v[i].max(T::zero());
    }
  }
}

impl<T: FloatExt, S: SeedExt> crate::traits::Sealed for RlHestonSampler<'_, T, S> {}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for RlHestonSampler<'_, T, S> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [s, v] = out;
    self.fill_paths(
      s.as_slice_mut()
        .expect("RlHeston output must be contiguous"),
      v.as_slice_mut()
        .expect("RlHeston output must be contiguous"),
    );
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    let mut s = Array1::<T>::zeros(self.n);
    let mut v = Array1::<T>::zeros(self.n);
    self.fill_paths(
      s.as_slice_mut().expect("contiguous"),
      v.as_slice_mut().expect("contiguous"),
    );
    [s, v]
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_core::simd_rng::Unseeded;

  use super::RlHeston;
  use crate::traits::ProcessExt;

  #[test]
  #[should_panic(expected = "n must satisfy `n >= 2`, got n = 1")]
  fn rejects_too_short_grid() {
    let _ = RlHeston::<f64>::new(
      0.3,
      100.0,
      0.04,
      1.0,
      0.04,
      0.5,
      -0.5,
      0.0,
      1,
      Some(1.0),
      None,
      Unseeded,
    );
  }

  #[test]
  fn variance_stays_non_negative() {
    let p = RlHeston::new(
      0.12_f64,
      100.0,
      0.04,
      0.1,
      0.3156,
      0.0331,
      -0.681,
      0.0,
      256,
      Some(1.0),
      None,
      Deterministic::new(11),
    );
    let [s, v] = p.sample();
    assert!(v.iter().all(|x| *x >= 0.0));
    assert!(s.iter().all(|x| x.is_finite()));
  }

  #[test]
  fn paths_have_the_grid_length_and_a_non_negative_variance() {
    let p = RlHeston::new(
      0.12_f64,
      100.0,
      0.04,
      0.1,
      0.3156,
      0.0331,
      -0.681,
      0.0,
      64,
      Some(1.0),
      Some(25),
      Deterministic::new(17),
    );
    let paths = p.sample_par(13);
    assert_eq!(paths.len(), 13);
    for [spot, variance] in &paths {
      assert_eq!((spot.len(), variance.len()), (64, 64));
      assert!(variance.iter().all(|x| *x >= 0.0));
      assert!(spot.iter().all(|x| x.is_finite()));
    }
  }

  #[test]
  fn heston_limit_h_half_asymptotic_variance_is_finite() {
    let p = RlHeston::new(
      0.49_f64,
      100.0,
      0.04,
      1.0,
      0.04,
      0.3,
      -0.5,
      0.0,
      512,
      Some(1.0),
      None,
      Deterministic::new(3),
    );
    let [_s, v] = p.sample();
    let mean: f64 = v.iter().sum::<f64>() / v.len() as f64;
    assert!(
      mean > 0.0 && mean < 1.0,
      "mean variance out of sanity range: {mean}"
    );
  }

  /// Guards the field-vs-doc contradiction fixed in A1-b: the module doc
  /// calls this quantity ν, so the field is `nu`. Larger ν must widen the
  /// simulated variance dispersion.
  #[test]
  fn rl_heston_nu_drives_variance_dispersion() {
    let small = RlHeston::new(
      0.12_f64,
      100.0,
      0.04,
      0.5,
      0.04,
      0.05,
      -0.3,
      0.0,
      256,
      Some(1.0),
      None,
      Deterministic::new(11),
    );
    let large = RlHeston::new(
      0.12_f64,
      100.0,
      0.04,
      0.5,
      0.04,
      0.5,
      -0.3,
      0.0,
      256,
      Some(1.0),
      None,
      Deterministic::new(11),
    );
    assert_eq!(small.nu, 0.05);
    assert_eq!(large.nu, 0.5);

    let variance = |x: &Array1<f64>| {
      let mean = x.iter().sum::<f64>() / x.len() as f64;
      x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64
    };

    let [_s_small, v_small] = small.sample();
    let [_s_large, v_large] = large.sample();
    assert!(
      variance(&v_large) > variance(&v_small),
      "larger nu must widen variance dispersion"
    );
  }
}

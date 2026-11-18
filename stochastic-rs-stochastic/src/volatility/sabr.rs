//! # Sabr
//!
//! $$
//! dF_t=\alpha_t F_t^\beta dW_t^1,\quad
//! d\alpha_t=\nu\alpha_t dW_t^2,\ d\langle W^1,W^2\rangle_t=\rho dt
//! $$
//!
//! $\alpha_t$ starts at the field `alpha0`; $\nu$, $\beta$ and $\rho$ are `nu`, `beta` and `rho`.
//!
//! Reference: Hagan, Kumar, Lesniewski & Woodward (2002), Wilmott Magazine, Sept., 84–108, no DOI
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::device::Cpu;
use crate::malliavin::Malliavin;
use crate::noise::cgns::Cgns;
use crate::precondition::check_n;
use crate::precondition::check_t;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Debug, Clone)]
pub struct Sabr<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Vol-of-vol $\nu$: diffusion coefficient of the volatility state
  /// $\alpha_t$ (see module docs for the SDE).
  pub nu: T,
  /// CEV exponent β ∈ [0, 1] — elasticity of the forward's own volatility
  /// (β=1 is lognormal SABR, β=0 is normal SABR).
  pub beta: T,
  /// Instantaneous correlation ρ between the forward's and volatility's
  /// driving Brownian motions.
  pub rho: T,
  /// Number of points sampled along the SABR path.
  pub n: usize,
  /// Initial forward-rate level.
  pub f0: Option<T>,
  /// Initial volatility level $\alpha_0$.
  pub alpha0: Option<T>,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// [`Unseeded`] or a [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic).
  pub seed: S,
  cgns: Cgns<T>,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

/// Every field has a matching `with_*` builder setter, e.g.
/// `Sabr::default().with_beta(0.5).with_rho(-0.6)`.
impl<T: FloatExt, S: SeedExt> Sabr<T, S> {
  /// Panics unless `nu >= 0`, `0 <= beta <= 1`, `-1 <= rho <= 1`, `n >= 2`, a given `alpha0` is
  /// non-negative, a given `t` is positive and finite, and every parameter is finite.
  pub fn new(
    nu: T,
    beta: T,
    rho: T,
    n: usize,
    f0: Option<T>,
    alpha0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    Self::check_nu(nu);
    Self::check_beta(beta);
    Self::check_rho(rho);
    check_n(n);
    Self::check_f0(f0);
    Self::check_alpha0(alpha0);
    check_t(t);

    Self {
      backend: Cpu,
      nu,
      beta,
      rho,
      n,
      f0,
      alpha0,
      t,
      seed,
      cgns: Cgns::new(rho, n - 1, t, Unseeded),
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Sabr<T, S, B> {
  fn check_nu(nu: T) {
    require!(nu.is_finite(), nu);
    require!(nu >= T::zero(), nu);
  }

  fn check_beta(beta: T) {
    require!(beta.is_finite(), beta);
    require!(beta >= T::zero() && beta <= T::one(), beta);
  }

  fn check_rho(rho: T) {
    require!(rho.is_finite(), rho);
  }

  fn check_f0(f0: Option<T>) {
    require!(f0.is_none_or(|v| v.is_finite()), f0);
  }

  fn check_alpha0(alpha0: Option<T>) {
    require!(alpha0.is_none_or(|v| v.is_finite()), alpha0);
    require!(alpha0.is_none_or(|a| a >= T::zero()), alpha0);
  }

  /// Replace `nu`; panics unless `nu >= 0` and `nu` is finite.
  pub fn with_nu(mut self, nu: T) -> Self {
    Self::check_nu(nu);
    self.nu = nu;
    self
  }

  /// Replace `beta`; panics unless `0 <= beta <= 1`.
  pub fn with_beta(mut self, beta: T) -> Self {
    Self::check_beta(beta);
    self.beta = beta;
    self
  }

  /// Replace `rho` and rebuild the cached correlated-Gaussian generator `cgns`, which the sampler
  /// reads; panics unless `-1 <= rho <= 1`.
  pub fn with_rho(mut self, rho: T) -> Self {
    Self::check_rho(rho);
    self.rho = rho;
    self.cgns = Cgns::new(rho, self.n - 1, self.t, Unseeded);
    self
  }

  /// Replace `f0`; panics unless `f0` is finite.
  pub fn with_f0(mut self, f0: Option<T>) -> Self {
    Self::check_f0(f0);
    self.f0 = f0;
    self
  }

  /// Replace `alpha0`; panics unless a given `alpha0` is non-negative and finite.
  pub fn with_alpha0(mut self, alpha0: Option<T>) -> Self {
    Self::check_alpha0(alpha0);
    self.alpha0 = alpha0;
    self
  }

  /// Replace `n`, the grid points along the path; rebuilds the cached correlated-Gaussian
  /// generator, whose length and step size derive from `n`; panics unless `n >= 2`.
  pub fn with_points(mut self, n: usize) -> Self {
    check_n(n);
    self.n = n;
    self.cgns = Cgns::new(self.rho, n - 1, self.t, Unseeded);
    self
  }

  /// Replace the simulation horizon `t`; rebuilds the cached correlated-Gaussian generator's step
  /// size, which derives from `t`; panics unless `t`, when given, is positive and finite.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    check_t(t);
    self.t = t;
    self.cgns = Cgns::new(self.rho, self.n - 1, t, Unseeded);
    self
  }

  /// Replace the seed strategy's value.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

/// ν=0.4, β=0.7, ρ=-0.3, f₀=1, α₀=0.3, a textbook Sabr; t=1, 252 points, a trading year of daily
/// closes.
impl<T: FloatExt> Default for Sabr<T, Unseeded> {
  fn default() -> Self {
    Self::new(
      T::from_f64_fast(0.4),
      T::from_f64_fast(0.7),
      T::from_f64_fast(-0.3),
      252,
      Some(T::one()),
      Some(T::from_f64_fast(0.3)),
      Some(T::one()),
      Unseeded,
    )
  }
}

/// The Euler engine's view of SABR. `½ν²` depends on no state, so it is
/// folded here rather than recomputed on every step of every path.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 2>
  for Sabr<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::Sabr {
      beta: self.beta,
      rho: self.rho,
      nu: self.nu,
      half_nu_sq: T::from_f64_fast(0.5) * self.nu * self.nu,
    }
  }

  fn initial_state(&self) -> [T; 4] {
    [
      self.f0.unwrap_or(T::zero()),
      self.alpha0.unwrap_or(T::zero()).max(T::zero()),
      T::zero(),
      T::zero(),
    ]
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  /// The correlated-noise source divides the horizon by the number of points,
  /// not by the number of steps, so the device steps by that same amount.
  fn time_step(&self) -> T {
    self.cgns.dt()
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

backend_switch!([T: FloatExt, S: SeedExt] Sabr<T, S> { nu, beta, rho, n, f0, alpha0, t, seed, cgns } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Sabr<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Sabr<T, S, B> {
  /// `[F path, α path]`: index 0 is the forward `F`, index 1 is the
  /// stochastic-volatility state `α` (see module docs for the SDE).
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = SabrSampler<T, S>
  where
    Self: 's;

  /// Derives `self.seed` into the sampler, never clones it: the mixed next tick hash-scrambles
  /// chunk bases. `fill_paths` uses the owned seed directly, one derive from `self.seed` per chunk.
  fn sampler(&self) -> SabrSampler<T, S> {
    SabrSampler {
      n: self.n,
      f0: self.f0.unwrap_or(T::zero()),
      alpha0: self.alpha0.unwrap_or(T::zero()).max(T::zero()),
      nu: self.nu,
      beta: self.beta,
      dt: self.cgns.dt(),
      cgns: self.cgns,
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine: a device steps every component in its kernel, the host runs this
  /// process's own sampler, chunked as `ProcessExt` chunks.
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

/// Reusable [`Sabr`] sampling state: owns the correlated-Gaussian generator
/// and the seed source so a Monte-Carlo loop reuses both output buffers.
#[doc(hidden)]
#[derive(Debug)]
pub struct SabrSampler<T: FloatExt, S: SeedExt> {
  n: usize,
  f0: T,
  alpha0: T,
  nu: T,
  beta: T,
  dt: T,
  cgns: Cgns<T>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> SabrSampler<T, S> {
  fn fill_paths(&mut self, f_: &mut [T], v: &mut [T]) {
    if self.n == 0 {
      return;
    }
    let [cgn1, cgn2] = &self.cgns.sample_impl(&self.seed);

    f_[0] = self.f0;
    v[0] = self.alpha0;

    for i in 1..self.n {
      let f_prev = f_[i - 1].max(T::zero());
      let v_prev = v[i - 1].max(T::zero());
      f_[i] = f_[i - 1] + v_prev * f_prev.powf(self.beta) * cgn1[i - 1];
      // Exact step for dα = ν α dW preserves non-negativity.
      v[i] =
        v_prev * (self.nu * cgn2[i - 1] - T::from_f64_fast(0.5) * self.nu.powi(2) * self.dt).exp();
    }
  }
}

impl<T: FloatExt, S: SeedExt> crate::traits::Sealed for SabrSampler<T, S> {}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for SabrSampler<T, S> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [f_arr, v_arr] = out;
    let f_ = f_arr
      .as_slice_mut()
      .expect("Sabr output must be contiguous");
    let v = v_arr
      .as_slice_mut()
      .expect("Sabr output must be contiguous");
    self.fill_paths(f_, v);
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    let mut f_ = Array1::<T>::zeros(self.n);
    let mut v = Array1::<T>::zeros(self.n);
    self.fill_paths(
      f_.as_slice_mut().expect("contiguous"),
      v.as_slice_mut().expect("contiguous"),
    );
    [f_, v]
  }
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> Sabr<T, S, B> {
  /// The path and `D_{t_i} α_T = ν α_T` at every grid point, since the volatility is lognormal.
  pub fn malliavin(&self) -> Malliavin<T, [Array1<T>; 2]> {
    let [f, v] = self.sample();

    let mut malliavin = Array1::<T>::zeros(self.n);

    for i in 0..self.n {
      malliavin[i] = self.nu * *v.last().unwrap();
    }

    Malliavin {
      paths: [f, v],
      derivative: malliavin,
    }
  }
}

// The Python parameter names stay `alpha`/`v0`; they forward positionally into `Sabr::new`'s
// `nu`/`alpha0`.
py_process_2x1d!(PySabr, Sabr,
  sig: (alpha, beta, rho, n, f0=None, v0=None, t=None, seed=None, dtype=None),
  params: (alpha: f64, beta: f64, rho: f64, n: usize, f0: Option<f64>, v0: Option<f64>, t: Option<f64>),
  device
);

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;
  use crate::traits::ProcessExt;

  #[test]
  fn the_volatility_derivative_is_nu_times_the_terminal_volatility() {
    let make = || {
      Sabr::<f64, _>::new(
        0.4,
        0.7,
        -0.3,
        17,
        Some(1.0),
        Some(0.3),
        Some(1.0),
        Deterministic::new(11),
      )
    };
    let Malliavin {
      paths: [f, v],
      derivative,
      ..
    } = make().malliavin();
    assert_eq!([f, v.clone()], make().sample());
    assert!(derivative.iter().all(|d| *d == 0.4 * v[16]));
  }

  #[test]
  fn volatility_stays_non_negative() {
    let p = Sabr::new(
      0.4_f64,
      0.5,
      -0.3,
      256,
      Some(1.0),
      Some(0.2),
      Some(1.0),
      Unseeded,
    );
    let [_f, v] = p.sample();
    assert!(v.iter().all(|x| *x >= 0.0));
  }

  /// The module doc's `dα = ν α dW₂` makes `nu` the vol-of-vol and `alpha0` the initial
  /// volatility state; the vol-of-vol must drive the volatility path's dispersion.
  #[test]
  fn sabr_nu_drives_volatility_dispersion() {
    let make = |nu: f64| {
      Sabr::new(
        nu,
        0.5,
        -0.3,
        512,
        Some(1.0),
        Some(0.2),
        Some(1.0),
        Deterministic::new(7),
      )
    };

    let small = make(0.05);
    let large = make(0.9);
    // Field-name guard: `nu` is vol-of-vol, `alpha0` is the initial state.
    assert_eq!(small.nu, 0.05);
    assert_eq!(large.alpha0, Some(0.2));

    let [_f_small, v_small] = small.sample();
    let [_f_large, v_large] = large.sample();

    let variance = |x: &Array1<f64>| {
      let mean = x.iter().copied().sum::<f64>() / x.len() as f64;
      x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64
    };

    assert!(
      variance(&v_large) > variance(&v_small),
      "large-nu volatility path must disperse more than small-nu: var(small)={}, var(large)={}",
      variance(&v_small),
      variance(&v_large)
    );
  }
}

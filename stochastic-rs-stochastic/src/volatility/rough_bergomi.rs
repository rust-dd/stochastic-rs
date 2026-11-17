//! # Rbergomi
//!
//! $$
//! dS_t = r S_t\,dt + S_t\sqrt{v_t}\,dW^1_t,\qquad
//! v_t = v_0^2\exp\!\Bigl(\nu W^H_t - \tfrac12\nu^2 t^{2H}\Bigr),\qquad
//! W^H_t = \sqrt{2H}\int_0^t (t-s)^{H-1/2}\,dW^2_s
//! $$
//!
//! Rough Bergomi of Bayer, Friz & Gatheral (2016) with a flat forward
//! variance curve $\xi_0(t) = v_0^2$. The variance driver $W^H$ is the
//! Riemann–Liouville (Volterra) fractional Brownian motion, simulated with
//! the hybrid scheme of Bennedsen, Lunde & Pakkanen (2017) at $\kappa = 1$:
//! on the grid $t_i = i\Delta$ the Volterra process is
//!
//! $$
//! X_{t_i} = \int_{t_{i-1}}^{t_i} (t_i-s)^{\alpha}\,dW^2_s
//!         + \sum_{k=2}^{i} \bigl(b_k^*\Delta\bigr)^{\alpha}\,
//!           \bigl(W^2_{t_{i-k+1}} - W^2_{t_{i-k}}\bigr),\qquad
//! \alpha = H - \tfrac12,\quad
//! b_k^* = \Bigl(\frac{k^{\alpha+1}-(k-1)^{\alpha+1}}{\alpha+1}\Bigr)^{1/\alpha},
//! $$
//!
//! and $W^H_{t_i} = \sqrt{2H}\,X_{t_i}$. The first term is drawn exactly:
//! it is jointly Gaussian with the increment $\Delta W^2_i$, with
//! $\operatorname{Cov} = \Delta^{\alpha+1}/(\alpha+1)$ and variance
//! $\Delta^{2\alpha+1}/(2\alpha+1)$, so it is the increment's regression
//! plus an independent residual. The remaining sum uses the kernel at the
//! optimal discretisation points $b_k^*$ (their Proposition 2.8), which
//! makes the scheme's covariance error second order in $\Delta$. The
//! scheme reproduces $\operatorname{Var}(W^H_t) = t^{2H}$, hence the
//! $\tfrac12\nu^2 t^{2H}$ compensator keeps $\mathbb E(v_t) = v_0^2$, and
//! reduces to the plain Brownian recipe at $H = \tfrac12$. The price uses
//! Euler steps on $S$ with $dW^1 = \rho\,dW^2 + \sqrt{1-\rho^2}\,dW^\perp$.
//!
//! The convolution costs $O(n^2)$ per path, the same order as the
//! Markov-lift alternatives at small $n$; for long grids see
//! [`crate::rough::MarkovLift`] (exponential-sum kernel, $O(nN')$) and
//! [`crate::rough::rl_heston::RlHeston`].
//!
//! References: Bayer, C., Friz, P. & Gatheral, J. (2016), *Pricing under
//! rough volatility*, Quantitative Finance 16(6), 887–904; Bennedsen, M.,
//! Lunde, A. & Pakkanen, M. S. (2017), *Hybrid scheme for Brownian
//! semistationary processes*, Finance and Stochastics 21, 931–965, DOI 10.1007/s00780-017-0335-5.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;

use crate::device::Cpu;
use crate::noise::cgns::Cgns;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Debug, Clone)]
pub struct RoughBergomi<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Hurst exponent `H ∈ (0, ½]` of the Volterra fractional Brownian
  /// driver, simulated with the hybrid scheme (see the module doc); `H = ½`
  /// is the Brownian case.
  pub hurst: T,
  /// Vol-of-vol ν scaling the log-variance driver's dispersion.
  pub nu: T,
  /// Initial volatility; the variance starts at `v0²`.
  pub v0: T,
  /// Initial spot.
  pub s0: T,
  /// Constant proportional drift rate of the asset (a GBM-style drift, not
  /// a correlation — that role belongs to `rho`).
  pub r: T,
  /// Correlation ρ between the asset's Brownian shock and the
  /// log-variance driver's innovations.
  pub rho: T,
  /// Number of points sampled along the rBergomi path.
  pub n: usize,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  cgns: Cgns<T>,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

/// Every field has a matching `with_*` builder setter, e.g.
/// `RoughBergomi::default().with_hurst(0.3).with_rho(-0.2)`.
impl<T: FloatExt, S: SeedExt> RoughBergomi<T, S> {
  pub fn new(hurst: T, nu: T, v0: T, s0: T, r: T, rho: T, n: usize, t: Option<T>, seed: S) -> Self {
    Self::check_hurst(hurst);
    Self::check_nu(nu);
    Self::check_v0(v0);
    Self::check_s0(s0);
    Self::check_r(r);
    Self::check_rho(rho);
    Self::check_n(n);
    Self::check_t(t);
    RoughBergomi {
      backend: Cpu,
      hurst,
      nu,
      v0,
      s0,
      r,
      rho,
      n,
      t,
      seed,
      cgns: Cgns::new(rho, n - 1, t, Unseeded),
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> RoughBergomi<T, S, B> {
  fn check_hurst(hurst: T) {
    require!(hurst.is_finite(), hurst);
  }

  fn check_nu(nu: T) {
    require!(nu.is_finite(), nu);
  }

  fn check_v0(v0: T) {
    require!(v0.is_finite(), v0);
  }

  fn check_s0(s0: T) {
    require!(s0.is_finite(), s0);
  }

  fn check_r(r: T) {
    require!(r.is_finite(), r);
  }

  fn check_rho(rho: T) {
    require!(rho.is_finite(), rho);
  }

  fn check_n(n: usize) {
    require!(n >= 2, n);
  }

  fn check_t(t: Option<T>) {
    require!(t.is_none_or(|t| t > T::zero() && t.is_finite()), t);
  }

  /// Replace `hurst`, all else unchanged.
  pub fn with_hurst(mut self, hurst: T) -> Self {
    Self::check_hurst(hurst);
    self.hurst = hurst;
    self
  }

  /// Replace `nu`, all else unchanged.
  pub fn with_nu(mut self, nu: T) -> Self {
    Self::check_nu(nu);
    self.nu = nu;
    self
  }

  /// Replace `v0`, all else unchanged.
  pub fn with_v0(mut self, v0: T) -> Self {
    Self::check_v0(v0);
    self.v0 = v0;
    self
  }

  /// Replace `s0`, all else unchanged.
  pub fn with_s0(mut self, s0: T) -> Self {
    Self::check_s0(s0);
    self.s0 = s0;
    self
  }

  /// Replace `r`, all else unchanged.
  pub fn with_r(mut self, r: T) -> Self {
    Self::check_r(r);
    self.r = r;
    self
  }

  /// Replace `rho`; rebuilds the cached correlated-Gaussian generator
  /// (`cgns`) so the new correlation actually reaches the sampler instead
  /// of a stale one computed from the old `rho`.
  pub fn with_rho(mut self, rho: T) -> Self {
    Self::check_rho(rho);
    self.rho = rho;
    self.cgns = Cgns::new(rho, self.n - 1, self.t, Unseeded);
    self
  }

  /// Replace `n`, the grid points along the path; rebuilds the cached correlated-Gaussian
  /// generator, whose length and step size derive from `n`.
  pub fn with_points(mut self, n: usize) -> Self {
    Self::check_n(n);
    self.n = n;
    self.cgns = Cgns::new(self.rho, n - 1, self.t, Unseeded);
    self
  }

  /// Replace the simulation horizon `t`; rebuilds the cached
  /// correlated-Gaussian generator's step size, which derives from `t`.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    Self::check_t(t);
    self.t = t;
    self.cgns = Cgns::new(self.rho, self.n - 1, t, Unseeded);
    self
  }

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

/// H=0.1, ν=0.4, v₀=0.2, s₀=100, r=0.01, ρ=-0.6, a textbook rough Bergomi; t=1, 252 points, a
/// trading year of daily closes.
impl<T: FloatExt> Default for RoughBergomi<T, Unseeded> {
  fn default() -> Self {
    Self::new(
      T::from_f64_fast(0.1),
      T::from_f64_fast(0.4),
      T::from_f64_fast(0.2),
      T::from_f64_fast(100.0),
      T::from_f64_fast(0.01),
      T::from_f64_fast(-0.6),
      252,
      Some(T::one()),
      Unseeded,
    )
  }
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 2>
  for RoughBergomi<T, S, B>
{
  /// The hybrid scheme's constants, from the same tabulation the host sampler
  /// steps with.
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let (_, _, residual_sd) =
      RoughBergomiSampler::<T, S>::hybrid_weights(self.hurst, self.cgns.dt(), self.n);
    crate::euler::EulerSpec::RoughBergomiMemory {
      r: self.r,
      a: self.nu * (T::from_usize_(2) * self.hurst).sqrt(),
      v0sq: self.v0.powi(2),
      residual_sd,
      rho: self.rho,
    }
  }

  fn initial_state(&self) -> [T; 4] {
    [self.s0, self.v0.powi(2), T::zero(), T::zero()]
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  fn time_step(&self) -> T {
    self.cgns.dt()
  }

  /// Two curves: the scheme's weights by lag — the last interval's exact
  /// kernel integral first, then the kernel at the optimal points — and the
  /// deterministic exponent `ν² t^{2H} / 2` at each grid point.
  fn curves(&self) -> Option<Vec<Vec<T>>> {
    let dt = self.cgns.dt();
    let (kernel, on_increment, _) =
      RoughBergomiSampler::<T, S>::hybrid_weights(self.hurst, dt, self.n);
    let weights = (0..self.n)
      .map(|m| {
        if m == 0 {
          on_increment
        } else {
          kernel.get(m + 1).copied().unwrap_or(T::zero())
        }
      })
      .collect();
    let half = T::from_f64_fast(0.5);
    let two_h = T::from_usize_(2) * self.hurst;
    let exponent = (0..self.n)
      .map(|i| half * self.nu * self.nu * (T::from_usize_(i) * dt).powf(two_h))
      .collect();
    Some(vec![weights, exponent])
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

backend_switch!([T: FloatExt, S: SeedExt] RoughBergomi<T, S> { hurst, nu, v0, s0, r, rho, n, t, seed, cgns } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for RoughBergomi<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for RoughBergomi<T, S, B>
{
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = RoughBergomiSampler<T, S>
  where
    Self: 's;

  /// Derives (not clones) `self.seed` into the returned sampler: the
  /// derived value is `self.seed`'s *mixed* next tick, not a raw snapshot,
  /// so chunk `i`'s basis and chunk `i+1`'s basis are hash-scrambled
  /// relative to each other rather than one raw stride apart.
  fn sampler(&self) -> RoughBergomiSampler<T, S> {
    let dt = self.cgns.dt();
    let (kernel, integral_on_increment, integral_residual_sd) =
      RoughBergomiSampler::<T, S>::hybrid_weights(self.hurst, dt, self.n);
    RoughBergomiSampler {
      n: self.n,
      hurst: self.hurst,
      nu: self.nu,
      v0_sq: self.v0.powi(2),
      s0: self.s0,
      r: self.r,
      dt,
      kernel,
      integral_on_increment,
      integral_residual_sd,
      cgns: self.cgns,
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine when the grid fits the kernels'
  /// per-path history; a longer grid keeps the process on the host,
  /// chunked exactly as [`ProcessExt`] chunks.
  fn sample(&self) -> [Array1<T>; 2] {
    if self.device_ready() {
      self.backend.system_sample(self)
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&[Array1<T>; 2]) -> R + Sync) -> Vec<R> {
    if self.device_ready() {
      self.backend.system_paths_map(self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, f)
    }
  }

  fn sample_par(&self, m: usize) -> Vec<[Array1<T>; 2]> {
    if self.device_ready() {
      self.backend.system_paths(self, m)
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<[Array1<T>; 2], crate::device::DeviceError> {
    if self.device_ready() {
      self.backend.try_system_sample(self)
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<[Array1<T>; 2]>, crate::device::DeviceError> {
    if self.device_ready() {
      self.backend.try_system_paths(self, m)
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }

  /// What a device runs here: the grid fits the kernels'
  /// per-path history. A longer grid samples on the host.
  fn device_fallback(&self) -> Option<&'static str> {
    (!(self.n <= crate::euler::HISTORY_SLOTS))
      .then_some("a grid longer than the kernels' per-path history")
  }
}

/// Reusable [`RoughBergomi`] sampling state: owns the correlated-Gaussian
/// generator, the hybrid-scheme kernel weights and the seed source so a
/// Monte-Carlo loop reuses both output buffers and the noise setup.
#[doc(hidden)]
#[derive(Debug)]
pub struct RoughBergomiSampler<T: FloatExt, S: SeedExt> {
  n: usize,
  hurst: T,
  nu: T,
  v0_sq: T,
  s0: T,
  r: T,
  dt: T,
  /// `g(b_k^* Δ)` for `k = 2..n`, stored at index `k`, index 0 and 1 unused.
  kernel: Vec<T>,
  /// Regression coefficient of the exact last-interval integral on the
  /// Brownian increment, `Δ^α / (α + 1)`.
  integral_on_increment: T,
  /// Conditional standard deviation of that integral,
  /// `Δ^{α + 1/2} √(1/(2α+1) − 1/(α+1)²)`.
  integral_residual_sd: T,
  cgns: Cgns<T>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> RoughBergomiSampler<T, S> {
  /// Bennedsen–Lunde–Pakkanen hybrid scheme with `κ = 1` (2017, eq. (3.6)):
  /// the last interval's kernel integral is drawn exactly, jointly Gaussian
  /// with the Brownian increment; earlier intervals use the kernel at the
  /// optimal discretisation points `b_k^*` (Proposition 2.8).
  pub(crate) fn hybrid_weights(hurst: T, dt: T, n: usize) -> (Vec<T>, T, T) {
    let alpha = hurst.to_f64().unwrap() - 0.5;
    let dt = dt.to_f64().unwrap();
    let mut kernel = vec![T::zero(); n.max(2)];
    let dt_alpha = dt.powf(alpha);
    for (k, w) in kernel.iter_mut().enumerate().skip(2) {
      let k_t = k as f64;
      *w = T::from_f64_fast(
        dt_alpha * (k_t.powf(alpha + 1.0) - (k_t - 1.0).powf(alpha + 1.0)) / (alpha + 1.0),
      );
    }
    let on_increment = T::from_f64_fast(dt_alpha / (alpha + 1.0));
    let residual_var = 1.0 / (2.0 * alpha + 1.0) - 1.0 / (alpha + 1.0).powi(2);
    let residual_sd = T::from_f64_fast(dt.powf(alpha + 0.5) * residual_var.max(0.0).sqrt());
    (kernel, on_increment, residual_sd)
  }

  fn fill_paths(&mut self, s: &mut [T], v2: &mut [T]) {
    if self.n == 0 {
      return;
    }
    let dt = self.dt;
    let [cgn1, z] = &self.cgns.sample_impl(&self.seed);
    let steps = self.n - 1;
    let mut eps = vec![T::zero(); steps];
    if steps > 0 {
      stochastic_rs_distributions::normal::SimdNormal::<T>::new(T::zero(), T::one())
        .seeded(&self.seed)
        .fill_slice(&mut eps);
    }
    let sqrt_2h = (T::from_usize_(2) * self.hurst).sqrt();
    let two_h = T::from_usize_(2) * self.hurst;

    s[0] = self.s0;
    v2[0] = self.v0_sq;

    for i in 1..self.n {
      s[i] = s[i - 1] + self.r * s[i - 1] * dt + v2[i - 1].sqrt() * s[i - 1] * cgn1[i - 1];

      // Volterra process X_{t_i} = Σ_k kernel-weighted increments; the k = 1
      // term is the exact integral over the last interval.
      let mut x = self.integral_on_increment * z[i - 1] + self.integral_residual_sd * eps[i - 1];
      for k in 2..=i {
        x += self.kernel[k] * z[i - k];
      }
      let t = T::from_usize_(i) * dt;
      v2[i] = self.v0_sq
        * (self.nu * sqrt_2h * x - T::from_f64_fast(0.5) * self.nu.powi(2) * t.powf(two_h)).exp();
    }
  }
}

impl<T: FloatExt, S: SeedExt> crate::traits::Sealed for RoughBergomiSampler<T, S> {}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for RoughBergomiSampler<T, S> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [s, v2] = out;
    self.fill_paths(
      s.as_slice_mut()
        .expect("RoughBergomi output must be contiguous"),
      v2.as_slice_mut()
        .expect("RoughBergomi output must be contiguous"),
    );
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    let mut s = Array1::<T>::zeros(self.n);
    let mut v2 = Array1::<T>::zeros(self.n);
    self.fill_paths(
      s.as_slice_mut().expect("contiguous"),
      v2.as_slice_mut().expect("contiguous"),
    );
    [s, v2]
  }
}

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
#[doc(hidden)]
pub use python::PyRoughBergomi;

#[cfg(test)]
mod tests;

//! # BilateralGamma
//!
//! $$
//! X_t=\Gamma^+_t-\Gamma^-_t,\quad
//! \Gamma^+\sim\Gamma(\alpha_p t,\lambda_p^{-1}),\ \Gamma^-\sim\Gamma(\alpha_m t,\lambda_m^{-1})
//! $$
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::gamma::SimdGamma;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// The difference of two independent gamma processes, with characteristic exponent
/// $$
/// \psi(\xi)=\alpha_p\ln\!\frac{\lambda_p}{\lambda_p-i\xi}
/// +\alpha_m\ln\!\frac{\lambda_m}{\lambda_m+i\xi}
/// $$
#[derive(Debug, Clone)]
pub struct BilateralGamma<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Shape parameter for positive jumps.
  pub alpha_p: T,
  /// Rate parameter for positive jumps.
  pub lambda_p: T,
  /// Shape parameter for negative jumps.
  pub alpha_m: T,
  /// Rate parameter for negative jumps.
  pub lambda_m: T,
  /// Number of points sampled along the bilateral-gamma path.
  pub n: usize,
  /// Initial value; `None` starts at zero.
  pub x0: Option<T>,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Seed strategy (compile-time: `Unseeded` or `Deterministic`).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> BilateralGamma<T, S> {
  /// Panics unless `n >= 2`, a given `t` is positive and finite, `alpha_p > 0`, `lambda_p > 0`,
  /// `alpha_m > 0`, `lambda_m > 0`, and every parameter is finite.
  pub fn new(
    alpha_p: T,
    lambda_p: T,
    alpha_m: T,
    lambda_m: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert!(
      alpha_p.is_finite(),
      "alpha_p must satisfy `alpha_p.is_finite()`, got alpha_p = {alpha_p:?}"
    );
    assert!(
      lambda_p.is_finite(),
      "lambda_p must satisfy `lambda_p.is_finite()`, got lambda_p = {lambda_p:?}"
    );
    assert!(
      alpha_m.is_finite(),
      "alpha_m must satisfy `alpha_m.is_finite()`, got alpha_m = {alpha_m:?}"
    );
    assert!(
      lambda_m.is_finite(),
      "lambda_m must satisfy `lambda_m.is_finite()`, got lambda_m = {lambda_m:?}"
    );
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    assert!(
      x0.is_none_or(|v| v.is_finite()),
      "x0 must satisfy `x0.is_none_or(|v| v.is_finite())`, got x0 = {x0:?}"
    );
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    assert!(
      alpha_p > T::zero(),
      "alpha_p must satisfy `alpha_p > T::zero()`, got alpha_p = {alpha_p:?}"
    );
    assert!(
      lambda_p > T::zero(),
      "lambda_p must satisfy `lambda_p > T::zero()`, got lambda_p = {lambda_p:?}"
    );
    assert!(
      alpha_m > T::zero(),
      "alpha_m must satisfy `alpha_m > T::zero()`, got alpha_m = {alpha_m:?}"
    );
    assert!(
      lambda_m > T::zero(),
      "lambda_m must satisfy `lambda_m > T::zero()`, got lambda_m = {lambda_m:?}"
    );
    Self {
      backend: Cpu,
      alpha_p,
      lambda_p,
      alpha_m,
      lambda_m,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> BilateralGamma<T, S, B> {
  #[inline]
  fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n.saturating_sub(1).max(1))
  }
}

/// The Euler engine's view of the bilateral gamma process: two draws, one per
/// side, differenced in the step.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for BilateralGamma<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::BilateralGamma
  }

  fn initial_value(&self) -> T {
    self.x0.unwrap_or(T::zero())
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  fn time_step(&self) -> T {
    self.dt()
  }

  /// The two sides' own shapes and rates.
  fn gamma_draws(&self) -> Option<crate::euler::GammaDraws<T>> {
    let dt = self.dt();
    Some(crate::euler::GammaDraws {
      first: (self.alpha_p * dt, T::one() / self.lambda_p, T::zero()),
      second: Some((self.alpha_m * dt, T::one() / self.lambda_m, T::zero())),
    })
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

backend_switch!([T: FloatExt, S: SeedExt] BilateralGamma<T, S> { alpha_p, lambda_p, alpha_m, lambda_m, n, x0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for BilateralGamma<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for BilateralGamma<T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = BilateralGammaSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> BilateralGammaSampler<T> {
    // The positive, then the negative gamma source are derived from `self.seed`, an order the
    // pinned streams depend on; reuse advances both for independent paths.
    let dt = self.dt();
    BilateralGammaSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      gamma_p: SimdGamma::<T>::new(self.alpha_p * dt, T::one() / self.lambda_p).seeded(&self.seed),
      gamma_m: SimdGamma::<T>::new(self.alpha_m * dt, T::one() / self.lambda_m).seeded(&self.seed),
    }
  }

  /// Through the Euler engine: a device draws the increment in its kernel, the host runs this
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

/// Reusable [`BilateralGamma`] sampling state: owns the positive- and
/// negative-side gamma sources so a Monte-Carlo loop pays their setup once.
#[doc(hidden)]
#[derive(Debug)]
pub struct BilateralGammaSampler<T: FloatExt> {
  n: usize,
  x0: T,
  gamma_p: Seeded<SimdGamma<T>>,
  gamma_m: Seeded<SimdGamma<T>>,
}

impl<T: FloatExt> BilateralGammaSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }

    let mut gp = Array1::<T>::zeros(out.len() - 1);
    self.gamma_p.fill_slice(gp.as_slice_mut().unwrap());
    let mut gm = Array1::<T>::zeros(out.len() - 1);
    self.gamma_m.fill_slice(gm.as_slice_mut().unwrap());

    for i in 1..out.len() {
      out[i] = out[i - 1] + gp[i - 1] - gm[i - 1];
    }
  }
}

impl<T: FloatExt> crate::traits::Sealed for BilateralGammaSampler<T> {}

impl<T: FloatExt> PathSampler<T> for BilateralGammaSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    self.fill_path(
      out
        .as_slice_mut()
        .expect("BilateralGamma output must be contiguous"),
    );
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

/// Bilateral Gamma Motion: Brownian component plus Bilateral Gamma jumps.
///
/// $$
/// X_t=\sigma W_t+\Gamma^+_t-\Gamma^-_t
/// $$
#[derive(Debug, Clone)]
pub struct BilateralGammaMotion<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Diffusion coefficient of the Brownian component.
  pub sigma: T,
  /// Shape parameter for positive jumps.
  pub alpha_p: T,
  /// Rate parameter for positive jumps.
  pub lambda_p: T,
  /// Shape parameter for negative jumps.
  pub alpha_m: T,
  /// Rate parameter for negative jumps.
  pub lambda_m: T,
  /// Number of points sampled along the bilateral-gamma-motion path.
  pub n: usize,
  /// Initial value; `None` starts at zero.
  pub x0: Option<T>,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Seed strategy (compile-time: `Unseeded` or `Deterministic`).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> BilateralGammaMotion<T, S> {
  /// Panics unless `n >= 2`, a given `t` is positive and finite, `alpha_p > 0`, `lambda_p > 0`,
  /// `alpha_m > 0`, `lambda_m > 0`, and every parameter is finite.
  pub fn new(
    sigma: T,
    alpha_p: T,
    lambda_p: T,
    alpha_m: T,
    lambda_m: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert!(
      sigma.is_finite(),
      "sigma must satisfy `sigma.is_finite()`, got sigma = {sigma:?}"
    );
    assert!(
      alpha_p.is_finite(),
      "alpha_p must satisfy `alpha_p.is_finite()`, got alpha_p = {alpha_p:?}"
    );
    assert!(
      lambda_p.is_finite(),
      "lambda_p must satisfy `lambda_p.is_finite()`, got lambda_p = {lambda_p:?}"
    );
    assert!(
      alpha_m.is_finite(),
      "alpha_m must satisfy `alpha_m.is_finite()`, got alpha_m = {alpha_m:?}"
    );
    assert!(
      lambda_m.is_finite(),
      "lambda_m must satisfy `lambda_m.is_finite()`, got lambda_m = {lambda_m:?}"
    );
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    assert!(
      x0.is_none_or(|v| v.is_finite()),
      "x0 must satisfy `x0.is_none_or(|v| v.is_finite())`, got x0 = {x0:?}"
    );
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    assert!(
      alpha_p > T::zero(),
      "alpha_p must satisfy `alpha_p > T::zero()`, got alpha_p = {alpha_p:?}"
    );
    assert!(
      lambda_p > T::zero(),
      "lambda_p must satisfy `lambda_p > T::zero()`, got lambda_p = {lambda_p:?}"
    );
    assert!(
      alpha_m > T::zero(),
      "alpha_m must satisfy `alpha_m > T::zero()`, got alpha_m = {alpha_m:?}"
    );
    assert!(
      lambda_m > T::zero(),
      "lambda_m must satisfy `lambda_m > T::zero()`, got lambda_m = {lambda_m:?}"
    );
    Self {
      backend: Cpu,
      sigma,
      alpha_p,
      lambda_p,
      alpha_m,
      lambda_m,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> BilateralGammaMotion<T, S, B> {
  #[inline]
  fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n.saturating_sub(1).max(1))
  }
}

/// The Euler engine's view of the bilateral gamma process with a Brownian
/// part.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for BilateralGammaMotion<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::BilateralGammaMotion { sigma: self.sigma }
  }

  fn initial_value(&self) -> T {
    self.x0.unwrap_or(T::zero())
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  fn time_step(&self) -> T {
    self.dt()
  }

  /// The two sides' own shapes and rates.
  fn gamma_draws(&self) -> Option<crate::euler::GammaDraws<T>> {
    let dt = self.dt();
    Some(crate::euler::GammaDraws {
      first: (self.alpha_p * dt, T::one() / self.lambda_p, T::zero()),
      second: Some((self.alpha_m * dt, T::one() / self.lambda_m, T::zero())),
    })
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

backend_switch!([T: FloatExt, S: SeedExt] BilateralGammaMotion<T, S> { sigma, alpha_p, lambda_p, alpha_m, lambda_m, n, x0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for BilateralGammaMotion<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for BilateralGammaMotion<T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = BilateralGammaMotionSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> BilateralGammaMotionSampler<T> {
    // The gamma sources, then the Gaussian one are derived from `self.seed`, an order the pinned
    // streams depend on; reuse advances all of them for independent paths.
    let dt = self.dt();
    BilateralGammaMotionSampler {
      n: self.n,
      sigma: self.sigma,
      x0: self.x0.unwrap_or(T::zero()),
      sqrt_dt: dt.sqrt(),
      gamma_p: SimdGamma::<T>::new(self.alpha_p * dt, T::one() / self.lambda_p).seeded(&self.seed),
      gamma_m: SimdGamma::<T>::new(self.alpha_m * dt, T::one() / self.lambda_m).seeded(&self.seed),
      normal: SimdNormal::<T>::new(T::zero(), T::one()).seeded(&self.seed),
    }
  }

  /// Through the Euler engine: a device draws the increment in its kernel, the host runs this
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

/// Reusable [`BilateralGammaMotion`] sampling state: owns both gamma sources
/// and the Gaussian source so a Monte-Carlo loop pays their setup once.
#[doc(hidden)]
#[derive(Debug)]
pub struct BilateralGammaMotionSampler<T: FloatExt> {
  n: usize,
  sigma: T,
  x0: T,
  sqrt_dt: T,
  gamma_p: Seeded<SimdGamma<T>>,
  gamma_m: Seeded<SimdGamma<T>>,
  normal: Seeded<SimdNormal<T>>,
}

impl<T: FloatExt> BilateralGammaMotionSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }

    let mut gp = Array1::<T>::zeros(out.len() - 1);
    self.gamma_p.fill_slice(gp.as_slice_mut().unwrap());
    let mut gm = Array1::<T>::zeros(out.len() - 1);
    self.gamma_m.fill_slice(gm.as_slice_mut().unwrap());
    let mut z = Array1::<T>::zeros(out.len() - 1);
    self.normal.fill_slice(z.as_slice_mut().unwrap());

    for i in 1..out.len() {
      out[i] = out[i - 1] + self.sigma * self.sqrt_dt * z[i - 1] + gp[i - 1] - gm[i - 1];
    }
  }
}

impl<T: FloatExt> crate::traits::Sealed for BilateralGammaMotionSampler<T> {}

impl<T: FloatExt> PathSampler<T> for BilateralGammaMotionSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    self.fill_path(
      out
        .as_slice_mut()
        .expect("BilateralGammaMotion output must be contiguous"),
    );
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

#[cfg(test)]
mod tests;

py_process_1d!(PyBilateralGamma, BilateralGamma,
  sig: (alpha_p, lambda_p, alpha_m, lambda_m, n, x0=None, t=None, seed=None, dtype=None),
  params: (alpha_p: f64, lambda_p: f64, alpha_m: f64, lambda_m: f64, n: usize, x0: Option<f64>, t: Option<f64>),
  device
);

py_process_1d!(PyBilateralGammaMotion, BilateralGammaMotion,
  sig: (sigma, alpha_p, lambda_p, alpha_m, lambda_m, n, x0=None, t=None, seed=None, dtype=None),
  params: (sigma: f64, alpha_p: f64, lambda_p: f64, alpha_m: f64, lambda_m: f64, n: usize, x0: Option<f64>, t: Option<f64>),
  device
);

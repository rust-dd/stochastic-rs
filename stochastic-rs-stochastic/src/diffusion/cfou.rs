//! Complex fractional Ornstein–Uhlenbeck: decay `lambda`, rotation `omega`, on a complex fBM.
//!
//! $$
//! dZ_t=-(\lambda-i\omega)Z_t\,dt+\sqrt{a}\,d\zeta_t,\qquad
//! \zeta_t=\frac{B_t^{(1)}+iB_t^{(2)}}{\sqrt{2}}
//! $$
//!
//! The real-imaginary form this implementation steps:
//! $$
//! \begin{aligned}
//! dX_1(t)&=-(\lambda X_1(t)+\omega X_2(t))\,dt+\sqrt{a/2}\,dB_t^{(1)},\\
//! dX_2(t)&=(\omega X_1(t)-\lambda X_2(t))\,dt+\sqrt{a/2}\,dB_t^{(2)}.
//! \end{aligned}
//! $$
//!
//! Reference: Alazemi, Alsenafi, Chen & Zhou (2024), arXiv:2406.18004
use ndarray::Array1;
use num_complex::Complex;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::device::Cpu;
use crate::device::DeviceError;
use crate::device::FgnBackend;
use crate::noise::fgn::Fgn;
use crate::precondition::check_n;
use crate::precondition::check_t;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Complex fractional Ornstein-Uhlenbeck process. Fields are private, since the fGN driver caches
/// a spectrum of `hurst`, `n` and `t`: read them through getters, change them through `with_*`.
///
/// Reference: Alazemi, Alsenafi, Chen & Zhou (2024), arXiv:2406.18004
///
/// ```compile_fail,E0616
/// use stochastic_rs_core::simd_rng::Unseeded;
/// use stochastic_rs_stochastic::diffusion::cfou::Cfou;
/// let mut p = Cfou::<f64>::new(0.7, 1.0, 0.5, 0.2, 10, None, None, None, Unseeded);
/// p.n = 1000;
/// ```
#[derive(Debug, Clone)]
pub struct Cfou<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  hurst: T,
  lambda: T,
  omega: T,
  a: T,
  n: usize,
  x1_0: Option<T>,
  x2_0: Option<T>,
  t: Option<T>,
  seed: S,
  fgn: Fgn<T, Unseeded, B>,
}

impl<T: FloatExt, S: SeedExt> Cfou<T, S, Cpu> {
  fn check_hurst(hurst: T) {
    require!(hurst.is_finite(), hurst);
  }

  /// Panics unless `0 <= hurst <= 1`, `lambda > 0`, `a > 0`, `n >= 2`, a given `t` is positive and
  /// finite, and every parameter is finite.
  #[must_use]
  pub fn new(
    hurst: T,
    lambda: T,
    omega: T,
    a: T,
    n: usize,
    x1_0: Option<T>,
    x2_0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    Self::check_hurst(hurst);
    Self::check_lambda(lambda);
    Self::check_omega(omega);
    Self::check_a(a);
    Self::check_x1_0(x1_0);
    Self::check_x2_0(x2_0);
    check_t(t);
    check_n(n);

    Self {
      hurst,
      lambda,
      omega,
      a,
      n,
      x1_0,
      x2_0,
      t,
      seed,
      fgn: Fgn::for_grid(hurst, n, t),
    }
  }

  /// Replace `hurst` and rebuild the cache, so it is defined on `Cpu` (move with `on` afterwards);
  /// panics unless `0 <= hurst <= 1`.
  pub fn with_hurst(mut self, hurst: T) -> Self {
    Self::check_hurst(hurst);
    self.hurst = hurst;
    self.fgn = Fgn::for_grid(hurst, self.n, self.t);
    self
  }

  /// Replace `n`; panics if `n < 2`.
  /// Rebuilds the cache, so it is defined on `Cpu`; move with `on` afterwards.
  pub fn with_points(mut self, n: usize) -> Self {
    check_n(n);
    self.n = n;
    self.fgn = Fgn::for_grid(self.hurst, n, self.t);
    self
  }

  /// Replace `t` and rebuild the cache, so it is defined on `Cpu` (move with `on` afterwards);
  /// panics unless `t`, when given, is positive and finite.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    check_t(t);
    self.t = t;
    self.fgn = Fgn::for_grid(self.hurst, self.n, t);
    self
  }
}

impl<T: FloatExt, S: SeedExt, B> Cfou<T, S, B> {
  fn check_lambda(lambda: T) {
    require!(lambda.is_finite(), lambda);
    require!(lambda > T::zero(), lambda);
  }

  fn check_omega(omega: T) {
    require!(omega.is_finite(), omega);
  }

  fn check_a(a: T) {
    require!(a.is_finite(), a);
    require!(a > T::zero(), a);
  }

  fn check_x1_0(x1_0: Option<T>) {
    require!(x1_0.is_none_or(|v| v.is_finite()), x1_0);
  }

  fn check_x2_0(x2_0: Option<T>) {
    require!(x2_0.is_none_or(|v| v.is_finite()), x2_0);
  }

  /// Replace `lambda`; panics unless `lambda > 0` and finite.
  pub fn with_lambda(mut self, lambda: T) -> Self {
    Self::check_lambda(lambda);
    self.lambda = lambda;
    self
  }

  /// Replace `omega`; panics unless `omega` is finite.
  pub fn with_omega(mut self, omega: T) -> Self {
    Self::check_omega(omega);
    self.omega = omega;
    self
  }

  /// Replace `a`; panics unless `a > 0` and finite.
  pub fn with_a(mut self, a: T) -> Self {
    Self::check_a(a);
    self.a = a;
    self
  }

  /// Replace `x1_0`; panics unless `x1_0` is finite.
  pub fn with_x1_0(mut self, x1_0: Option<T>) -> Self {
    Self::check_x1_0(x1_0);
    self.x1_0 = x1_0;
    self
  }

  /// Replace `x2_0`; panics unless `x2_0` is finite.
  pub fn with_x2_0(mut self, x2_0: Option<T>) -> Self {
    Self::check_x2_0(x2_0);
    self.x2_0 = x2_0;
    self
  }

  /// Replace the seed strategy's value, all else unchanged. `fgn`'s own
  /// seed is a never-read dummy, so this does not touch it.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }

  /// Hurst exponent of the driving fractional Brownian motion.
  pub fn hurst(&self) -> T {
    self.hurst
  }

  /// Real part of the complex mean-reversion coefficient (`lambda > 0`).
  pub fn lambda(&self) -> T {
    self.lambda
  }

  /// Imaginary-frequency part of the complex mean-reversion coefficient.
  pub fn omega(&self) -> T {
    self.omega
  }

  /// Noise intensity parameter in `sqrt(a) d\zeta_t` (`a > 0`).
  pub fn a(&self) -> T {
    self.a
  }

  /// Number of points sampled along the complex fOU path.
  pub fn n(&self) -> usize {
    self.n
  }

  /// Initial value of the real part `X_1(0)`.
  pub fn x1_0(&self) -> Option<T> {
    self.x1_0
  }

  /// Initial value of the imaginary part `X_2(0)`.
  pub fn x2_0(&self) -> Option<T> {
    self.x2_0
  }

  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub fn t(&self) -> Option<T> {
    self.t
  }

  /// Seed: [`Unseeded`] or a [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic).
  pub fn seed(&self) -> &S {
    &self.seed
  }
}

/// The complex path's two real rows as a process of their own: the engine speaks in
/// `[Array1<T>; 2]` and [`Cfou`] rejoins the planes. It borrows, so it advances the process's seed.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct CfouParts<'a, T: FloatExt, S: SeedExt, B>(&'a Cfou<T, S, B>);

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>>
  crate::traits::Sealed for CfouParts<'_, T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>> ProcessExt<T>
  for CfouParts<'_, T, S, B>
{
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = CfouPartsSampler<'s, T, S, B>
  where
    Self: 's;

  fn sampler(&self) -> CfouPartsSampler<'_, T, S, B> {
    CfouPartsSampler(<Cfou<T, S, B> as ProcessExt<T>>::sampler(self.0))
  }

  fn advance_chunk_seed(&self) {
    <Cfou<T, S, B> as ProcessExt<T>>::advance_chunk_seed(self.0)
  }
}

/// [`CfouParts`]'s sampler: the process's own complex sampler, split into the
/// two real rows as each path comes off it.
#[doc(hidden)]
#[derive(Debug)]
pub struct CfouPartsSampler<'a, T: FloatExt, S: SeedExt, B>(CfouSampler<'a, T, S, B>);

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> crate::traits::Sealed
  for CfouPartsSampler<'_, T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> PathSampler<T> for CfouPartsSampler<'_, T, S, B> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [x1, x2] = self.sample();
    out[0] = x1;
    out[1] = x2;
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    split_complex(&self.0.sample())
  }

  fn try_sample(&mut self) -> Result<[Array1<T>; 2], DeviceError> {
    self.0.try_sample().map(|z| split_complex(&z))
  }
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>>
  crate::euler::EulerSystem<T, 2> for CfouParts<'_, T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::ComplexFractionalOu {
      lambda: self.0.lambda,
      omega: self.0.omega,
      scale: (self.0.a * T::from_f64_fast(0.5)).sqrt(),
    }
  }

  fn initial_state(&self) -> [T; 4] {
    [
      self.0.x1_0.unwrap_or(T::zero()),
      self.0.x2_0.unwrap_or(T::zero()),
      T::zero(),
      T::zero(),
    ]
  }

  fn grid_points(&self) -> usize {
    self.0.n
  }

  fn horizon(&self) -> T {
    self.0.t.unwrap_or(T::one())
  }

  fn time_step(&self) -> T {
    self.0.fgn.dt()
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.0.seed)
  }

  /// The real and imaginary noises are the two halves of one complex fractional increment: one
  /// embedding draws `2 · m` paths, and the step reads its second stream `paths` rows further on.
  fn fgn_spec(&self) -> Option<crate::euler::FgnSpec<'_, T>> {
    Some(self.0.fgn.fgn_spec(2))
  }

  fn host_sample(&self) -> [Array1<T>; 2] {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

/// The real and imaginary rows of a complex path, the shape the engine's
/// two-component launch speaks in.
fn split_complex<T: FloatExt>(z: &Array1<Complex<T>>) -> [Array1<T>; 2] {
  let mut x1 = Array1::<T>::zeros(z.len());
  let mut x2 = Array1::<T>::zeros(z.len());
  for (i, v) in z.iter().enumerate() {
    x1[i] = v.re;
    x2[i] = v.im;
  }
  [x1, x2]
}

/// The inverse of [`split_complex`]: the engine reports two real planes and
/// the process reports one complex path.
fn join_complex<T: FloatExt>([x1, x2]: [Array1<T>; 2]) -> Array1<Complex<T>> {
  Array1::from_iter(
    x1.iter()
      .zip(x2.iter())
      .map(|(re, im)| Complex::new(*re, *im)),
  )
}

backend_switch!([T: FloatExt, S: SeedExt] Cfou<T, S> { hurst, lambda, omega, a, n, x1_0, x2_0, t, seed } via fgn euler);

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>>
  crate::traits::Sealed for Cfou<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>> ProcessExt<T>
  for Cfou<T, S, B>
{
  type Output = Array1<Complex<T>>;
  type Sampler<'s>
    = CfouSampler<'s, T, S, B>
  where
    Self: 's;

  /// A CPU sampler that borrows the process for its inner [`Fgn`] and owns a seed derived once:
  /// one derive from `self.seed` per chunk, and each call advances the owned seed to a new path.
  fn sampler(&self) -> CfouSampler<'_, T, S, B> {
    CfouSampler {
      cfou: self,
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine: a device runs the complex recursion in its kernel on the fGN
  /// increments written there, and the planes are rejoined here; the host runs its own sampler.
  fn sample(&self) -> Array1<Complex<T>> {
    join_complex(crate::euler::EulerBackend::system_sample(
      &self.fgn.backend,
      &CfouParts(self),
    ))
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<Complex<T>>) -> R + Sync) -> Vec<R> {
    crate::euler::EulerBackend::system_paths_map(&self.fgn.backend, &CfouParts(self), m, |parts| {
      f(&join_complex([parts[0].clone(), parts[1].clone()]))
    })
  }

  fn sample_par(&self, m: usize) -> Vec<Array1<Complex<T>>> {
    crate::euler::EulerBackend::system_paths(&self.fgn.backend, &CfouParts(self), m)
      .into_iter()
      .map(join_complex)
      .collect()
  }

  fn try_sample(&self) -> Result<Array1<Complex<T>>, DeviceError> {
    crate::euler::EulerBackend::try_system_sample(&self.fgn.backend, &CfouParts(self))
      .map(join_complex)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<Complex<T>>>, DeviceError> {
    Ok(
      crate::euler::EulerBackend::try_system_paths(&self.fgn.backend, &CfouParts(self), m)?
        .into_iter()
        .map(join_complex)
        .collect(),
    )
  }
}

/// Reusable [`Cfou`] sampling state on the process's [`Fgn`] and a seed derived once. The step is
/// `Z_{k+1} = Z_k − (λ − iω) Z_k dt + √a Δζ_k`, `Δζ_k = (ΔB_k^{(1)} + i ΔB_k^{(2)}) / √2`.
///
/// Reference: Alazemi, Alsenafi, Chen & Zhou (2024), arXiv:2406.18004
#[doc(hidden)]
#[derive(Debug)]
pub struct CfouSampler<'a, T: FloatExt, S: SeedExt, B> {
  cfou: &'a Cfou<T, S, B>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> CfouSampler<'_, T, S, B> {
  fn try_fill_path(&mut self, out: &mut Array1<Complex<T>>) -> Result<(), DeviceError> {
    if out.is_empty() {
      return Ok(());
    }
    let p = self.cfou;
    let dt = p.fgn.dt();
    let (noise_1, noise_2) = p.fgn.try_noise_pair(&self.seed)?;
    let gamma = Complex::new(p.lambda, -p.omega);
    let dt_c = Complex::new(dt, T::zero());
    let noise_scale = (p.a * T::from_f64_fast(0.5)).sqrt();

    out[0] = Complex::new(p.x1_0.unwrap_or(T::zero()), p.x2_0.unwrap_or(T::zero()));

    for i in 1..p.n {
      let z_prev = out[i - 1];
      let drift = -gamma * z_prev;
      let d_zeta = Complex::new(noise_1[i - 1], noise_2[i - 1]);
      out[i] = z_prev + drift * dt_c + d_zeta * noise_scale;
    }
    Ok(())
  }

  fn fill_path(&mut self, out: &mut Array1<Complex<T>>) {
    self
      .try_fill_path(out)
      .unwrap_or_else(crate::device::device_panic)
  }
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> crate::traits::Sealed for CfouSampler<'_, T, S, B> {}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> PathSampler<T> for CfouSampler<'_, T, S, B> {
  type Output = Array1<Complex<T>>;

  fn sample_into(&mut self, out: &mut Array1<Complex<T>>) {
    self.fill_path(out);
  }

  fn sample(&mut self) -> Array1<Complex<T>> {
    let mut out = Array1::<Complex<T>>::from_elem(self.cfou.n, Complex::new(T::zero(), T::zero()));
    self.fill_path(&mut out);
    out
  }

  fn try_sample(&mut self) -> Result<Array1<Complex<T>>, DeviceError> {
    let mut out = Array1::<Complex<T>>::from_elem(self.cfou.n, Complex::new(T::zero(), T::zero()));
    self.try_fill_path(&mut out)?;
    Ok(out)
  }
}

impl<T: FloatExt, S: SeedExt> Cfou<T, S> {
  /// Samples the process and returns explicit real/imaginary components.
  #[must_use]
  pub fn sample_components(&self) -> [Array1<T>; 2] {
    let z = <Self as ProcessExt<T>>::sample(self);
    let mut x1 = Array1::<T>::zeros(self.n);
    let mut x2 = Array1::<T>::zeros(self.n);
    for i in 0..self.n {
      x1[i] = z[i].re;
      x2[i] = z[i].im;
    }
    [x1, x2]
  }
}

py_process_2d!(PyCfou, Cfou,
  sig: (hurst, lambda, omega, a, n, x1_0=None, x2_0=None, t=None, seed=None, dtype=None),
  params: (hurst: f64, lambda: f64, omega: f64, a: f64, n: usize, x1_0: Option<f64>, x2_0: Option<f64>, t: Option<f64>),
  device
);

#[cfg(test)]
mod tests;

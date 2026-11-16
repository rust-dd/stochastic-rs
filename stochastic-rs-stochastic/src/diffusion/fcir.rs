//! # fCIR
//!
//! $$
//! dX_t=\kappa(\theta-X_t)dt+\sigma\sqrt{X_t}\,dB_t^H
//! $$
//!
//! Reference: Mishura Y., Yurchenko-Tytarenko A. (2018) — *Fractional
//! Cox-Ingersoll-Ross Process with Non-Zero "Mean"*, Modern Stochastics:
//! Theory and Applications 5(1), 99–111, DOI: 10.15559/18-vmsta97 — the
//! non-zero-mean-reverting fCIR process this file discretises by Euler
//! scheme, floored or reflected at zero like
//! [`Cir`](crate::diffusion::cir::Cir).
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::device::DeviceError;
use crate::device::FgnBackend;
use crate::noise::fgn::Fgn;
use crate::params::VarianceBoundary;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Fractional Cox-Ingersoll-Ross (Fcir) process.
/// dX(t) = theta(mu - X(t))dt + sigma * sqrt(X(t))dW^H(t)
/// where X(t) is the Fcir process.
///
/// Fields are private: the fGN driver caches a spectrum derived from `hurst`, `n` and `t`, so
/// parameters are read through getters and changed through the cache-rebuilding `with_*` setters.
///
/// ```compile_fail,E0616
/// use stochastic_rs_core::simd_rng::Unseeded;
/// use stochastic_rs_stochastic::diffusion::fcir::Fcir;
/// use stochastic_rs_stochastic::params::VarianceBoundary;
/// let boundary = VarianceBoundary::Floor;
/// let mut p = Fcir::<f64>::new(0.7, 1.0, 0.04, 0.1, 10, None, None, boundary, Unseeded);
/// p.n = 1000;
/// ```
#[derive(Clone)]
pub struct Fcir<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  hurst: T,
  theta: T,
  mu: T,
  sigma: T,
  n: usize,
  x0: Option<T>,
  t: Option<T>,
  boundary: VarianceBoundary,
  seed: S,
  fgn: Fgn<T, Unseeded, B>,
}

impl<T: FloatExt, S: SeedExt> Fcir<T, S, Cpu> {
  /// [`Cir::new`](crate::diffusion::cir::Cir::new)'s Feller contract: a violation under
  /// [`VarianceBoundary::Floor`] emits a `log::warn!`, which the `with_*` setters do not repeat.
  #[must_use]
  pub fn new(
    hurst: T,
    theta: T,
    mu: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    boundary: VarianceBoundary,
    seed: S,
  ) -> Self {
    assert!(
      hurst.is_finite(),
      "hurst must satisfy `hurst.is_finite()`, got hurst = {hurst:?}"
    );
    assert!(
      theta.is_finite(),
      "theta must satisfy `theta.is_finite()`, got theta = {theta:?}"
    );
    assert!(
      mu.is_finite(),
      "mu must satisfy `mu.is_finite()`, got mu = {mu:?}"
    );
    assert!(
      sigma.is_finite(),
      "sigma must satisfy `sigma.is_finite()`, got sigma = {sigma:?}"
    );
    assert!(
      x0.is_none_or(|v| v.is_finite()),
      "x0 must satisfy `x0.is_none_or(|v| v.is_finite())`, got x0 = {x0:?}"
    );
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    if T::from_usize_(2) * theta * mu < sigma.powi(2) && boundary != VarianceBoundary::Reflect {
      log::warn!(
        "Fcir::new: Feller condition violated (2*theta*mu < sigma^2) \
         under VarianceBoundary::Floor; the path floors at zero on every \
         boundary hit instead of reflecting — pass VarianceBoundary::Reflect \
         for the standard sub-Feller mitigation"
      );
    }

    Self {
      hurst,
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      boundary,
      seed,
      fgn: Fgn::for_grid(hurst, n, t),
    }
  }

  /// Replace `hurst`; rebuilds the cache, so it is defined on `Cpu`; move with `on` afterwards.
  pub fn with_hurst(mut self, hurst: T) -> Self {
    assert!(
      hurst.is_finite(),
      "hurst must satisfy `hurst.is_finite()`, got hurst = {hurst:?}"
    );
    self.hurst = hurst;
    self.fgn = Fgn::for_grid(hurst, self.n, self.t);
    self
  }

  /// Replace `n`; panics if `n < 2`.
  /// Rebuilds the cache, so it is defined on `Cpu`; move with `on` afterwards.
  pub fn with_points(mut self, n: usize) -> Self {
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    self.n = n;
    self.fgn = Fgn::for_grid(self.hurst, n, self.t);
    self
  }

  /// Replace `t`; rebuilds the cache, so it is defined on `Cpu`; move with `on` afterwards.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    self.t = t;
    self.fgn = Fgn::for_grid(self.hurst, self.n, t);
    self
  }
}

impl<T: FloatExt, S: SeedExt, B> Fcir<T, S, B> {
  /// Replace `theta`, all else unchanged.
  pub fn with_theta(mut self, theta: T) -> Self {
    assert!(
      theta.is_finite(),
      "theta must satisfy `theta.is_finite()`, got theta = {theta:?}"
    );
    self.theta = theta;
    self
  }

  /// Replace `mu`, all else unchanged.
  pub fn with_mu(mut self, mu: T) -> Self {
    assert!(
      mu.is_finite(),
      "mu must satisfy `mu.is_finite()`, got mu = {mu:?}"
    );
    self.mu = mu;
    self
  }

  /// Replace `sigma`, all else unchanged.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    assert!(
      sigma.is_finite(),
      "sigma must satisfy `sigma.is_finite()`, got sigma = {sigma:?}"
    );
    self.sigma = sigma;
    self
  }

  /// Replace `x0`, all else unchanged.
  pub fn with_x0(mut self, x0: Option<T>) -> Self {
    assert!(
      x0.is_none_or(|v| v.is_finite()),
      "x0 must satisfy `x0.is_none_or(|v| v.is_finite())`, got x0 = {x0:?}"
    );
    self.x0 = x0;
    self
  }

  /// Replace `boundary`, all else unchanged.
  pub fn with_boundary(mut self, boundary: VarianceBoundary) -> Self {
    self.boundary = boundary;
    self
  }

  /// Replace the seed strategy's value, all else unchanged. `fgn`'s own
  /// seed is a never-read dummy, so this does not touch it.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }

  /// Hurst exponent controlling roughness and long-memory.
  pub fn hurst(&self) -> T {
    self.hurst
  }

  /// Mean-reversion speed (κ in the module header). Multiplies
  /// `(mu - X_t)`, despite the field's own name.
  pub fn theta(&self) -> T {
    self.theta
  }

  /// Long-run mean level (θ in the module header). The level `X`
  /// reverts to between fractional-noise shocks.
  pub fn mu(&self) -> T {
    self.mu
  }

  /// Diffusion scale σ multiplying `√X_t dB_t^H`.
  pub fn sigma(&self) -> T {
    self.sigma
  }

  /// Number of points sampled along the fCIR path.
  pub fn n(&self) -> usize {
    self.n
  }

  /// Initial value X₀ of the fCIR path.
  pub fn x0(&self) -> Option<T> {
    self.x0
  }

  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub fn t(&self) -> Option<T> {
    self.t
  }

  /// Treatment of a negative square-root state after a step; the Feller condition is not required.
  pub fn boundary(&self) -> VarianceBoundary {
    self.boundary
  }

  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub fn seed(&self) -> &S {
    &self.seed
  }
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>>
  crate::traits::Sealed for Fcir<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>> ProcessExt<T>
  for Fcir<T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = FcirSampler<'s, T, S, B>
  where
    Self: 's;

  /// A CPU sampler borrowing the process for its inner [`Fgn`] (`Arc`-shared
  /// FFT plan + eigenvalues) and owning a seed derived once at construction.
  /// Deriving (not cloning) is what decorrelates chunks: the derived value
  /// is `self.seed`'s *mixed* next tick, not a raw snapshot, so chunk `i`'s
  /// basis and chunk `i+1`'s basis are hash-scrambled relative to each
  /// other rather than one raw stride apart. `fill_path` then uses this
  /// owned seed *directly* (no further derive) — exactly one derive from
  /// `self.seed` per chunk, matching what the legacy per-call `derive()`
  /// consumed, so the first path reproduces the legacy stream bit-for-bit.
  /// Repeat calls on one sampler advance the same owned seed further, for
  /// an independent path each time.
  fn sampler(&self) -> FcirSampler<'_, T, S, B> {
    FcirSampler {
      fcir: self,
      seed: self.seed.derive(),
    }
  }

  /// The same launch as [`sample_par`](Self::sample_par), mapped. Without
  /// these two the trait's defaults would send a mapped batch through the
  /// host sampler while `sample_par` ran on the device — the same law, but
  /// a different stream and none of the device's speed.
  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    crate::euler::EulerBackend::euler_paths_map(&self.fgn.backend, self, m, f)
  }

  fn sample_map_view<R: Send>(
    &self,
    m: usize,
    f: impl Fn(ndarray::ArrayView1<T>) -> R + Sync,
  ) -> Vec<R> {
    crate::euler::EulerBackend::euler_paths_map_view(&self.fgn.backend, self, m, f)
  }

  /// `m` paths through the Euler engine: on a device the whole recursion runs
  /// in the kernel from fGN increments, on the host devices it is this
  /// process's own sampler chunked exactly as `ProcessExt` chunks.
  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    crate::euler::EulerBackend::euler_paths(&self.fgn.backend, self, m)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, crate::device::DeviceError> {
    crate::euler::EulerBackend::try_euler_paths(&self.fgn.backend, self, m)
  }
}

/// Reusable [`Fcir`] sampling state: borrows the process for its inner [`Fgn`] and owns a seed
/// derived once at construction.
#[doc(hidden)]
pub struct FcirSampler<'a, T: FloatExt, S: SeedExt, B> {
  fcir: &'a Fcir<T, S, B>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> FcirSampler<'_, T, S, B> {
  fn try_fill_path(&mut self, out: &mut [T]) -> Result<(), DeviceError> {
    if out.is_empty() {
      return Ok(());
    }
    let p = self.fcir;
    let dt = p.fgn.dt();
    let fgn = p.fgn.try_noise(&self.seed)?;

    out[0] = p.x0.unwrap_or(T::zero());
    let mut prev = out[0];
    for (dst, inc) in out[1..].iter_mut().zip(fgn.iter()) {
      let dfcir = p.theta * (p.mu - prev) * dt + p.sigma * prev.abs().sqrt() * *inc;
      let next = match p.boundary {
        VarianceBoundary::Floor => (prev + dfcir).max(T::zero()),
        VarianceBoundary::Reflect => (prev + dfcir).abs(),
      };
      *dst = next;
      prev = next;
    }
    Ok(())
  }

  fn fill_path(&mut self, out: &mut [T]) {
    self
      .try_fill_path(out)
      .unwrap_or_else(crate::device::device_panic)
  }
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> crate::traits::Sealed for FcirSampler<'_, T, S, B> {}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> PathSampler<T> for FcirSampler<'_, T, S, B> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("Fcir output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.fcir.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }

  fn try_sample(&mut self) -> Result<Array1<T>, DeviceError> {
    let mut out = Array1::<T>::zeros(self.fcir.n);
    self.try_fill_path(out.as_slice_mut().expect("Fcir output must be contiguous"))?;
    Ok(out)
  }
}

/// The Euler engine's view of fractional CIR, in the floored or the mirrored form by its boundary.
impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>>
  crate::euler::EulerCoefficients<T> for Fcir<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    match self.boundary {
      VarianceBoundary::Floor => crate::euler::EulerSpec::ReflectedSquareRoot {
        theta: self.theta,
        mu: self.mu,
        sigma: self.sigma,
      },
      VarianceBoundary::Reflect => crate::euler::EulerSpec::MirroredSquareRoot {
        theta: self.theta,
        mu: self.mu,
        sigma: self.sigma,
      },
    }
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

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }

  /// The pipeline that produces this process's increments: the device runs it
  /// and keeps the result in its own buffer.
  fn fgn_spec(&self) -> Option<crate::euler::FgnSpec<'_, T>> {
    Some(self.fgn.fgn_spec(1))
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Fcir<T, S> { hurst, theta, mu, sigma, n, x0, t, boundary, seed } via fgn euler);

py_process_1d!(PyFcir, Fcir,
  sig: (hurst, theta, mu, sigma, n, x0=None, t=None, boundary=None, seed=None, dtype=None),
  params: (hurst: f64, theta: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>, boundary: Option<crate::python_params::PyVarianceBoundary> => crate::python_params::boundary),
  device
);

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  /// `2·theta·mu = 0.1 < sigma² = 1.0` violates Feller; the reflected path must still build and stay finite.
  #[test]
  fn fcir_accepts_sub_feller_under_reflection() {
    let fcir = Fcir::<f64, _>::new(
      0.7,
      0.5,
      0.1,
      1.0,
      256,
      Some(0.1),
      Some(1.0),
      VarianceBoundary::Reflect,
      Deterministic::new(7),
    );
    let path = fcir.sample();
    assert_eq!(path.len(), 256);
    assert!(
      path.iter().all(|x| x.is_finite()),
      "sub-Feller Fcir path must stay finite under VarianceBoundary::Reflect"
    );
  }

  /// The floored step accepts sub-Feller parameters too; only the diagnostic warning differs.
  #[test]
  fn fcir_accepts_sub_feller_under_the_floor() {
    let fcir = Fcir::<f64, _>::new(
      0.7,
      0.5,
      0.1,
      1.0,
      256,
      Some(0.1),
      Some(1.0),
      VarianceBoundary::Floor,
      Deterministic::new(7),
    );
    let path = fcir.sample();
    assert!(path.iter().all(|x| x.is_finite() && *x >= 0.0));
  }
}

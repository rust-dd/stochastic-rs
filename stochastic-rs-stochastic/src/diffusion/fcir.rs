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
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Fractional Cox-Ingersoll-Ross (Fcir) process.
/// dX(t) = theta(mu - X(t))dt + sigma * sqrt(X(t))dW^H(t)
/// where X(t) is the Fcir process.
///
/// The fields are private because the embedded fGN driver caches a spectrum
/// derived from `hurst`, `n` and `t`: read the parameters through the getters
/// and change them through the `with_*` setters, which rebuild it. Assigning
/// to a field does not compile:
///
/// ```compile_fail,E0616
/// use stochastic_rs_core::simd_rng::Unseeded;
/// use stochastic_rs_stochastic::diffusion::fcir::Fcir;
/// let mut p = Fcir::<f64>::new(0.7, 1.0, 0.04, 0.1, 10, None, None, None, Unseeded);
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
  use_sym: Option<bool>,
  seed: S,
  fgn: Fgn<T, Unseeded, B>,
}

impl<T: FloatExt, S: SeedExt> Fcir<T, S, Cpu> {
  /// Create a new Fcir process.
  ///
  /// Same Feller-condition contract as [`crate::diffusion::cir::Cir`]:
  /// `2·theta·mu ≥ sigma²` keeps the continuous-time process strictly
  /// positive, but sub-Feller parameters are accepted rather than
  /// rejected, since the discretised step already keeps every sample
  /// non-negative — floored at zero by default, or reflected when
  /// [`use_sym`](Self::use_sym) is `true`. A violation not paired with
  /// `use_sym = Some(true)` unconditionally prints a one-line diagnostic
  /// to stderr — including in release builds; it never panics. The
  /// `with_*` setters, like [`Cir`](crate::diffusion::cir::Cir)'s, do not
  /// repeat that diagnostic.
  #[must_use]
  pub fn new(
    hurst: T,
    theta: T,
    mu: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    use_sym: Option<bool>,
    seed: S,
  ) -> Self {
    assert!(n >= 2, "n must be at least 2");
    if T::from_usize_(2) * theta * mu < sigma.powi(2) && use_sym != Some(true) {
      eprintln!(
        "warning: Fcir::new: Feller condition violated (2*theta*mu < sigma^2) \
         without use_sym = Some(true); the path floors at zero on every \
         boundary hit instead of reflecting — pass use_sym = Some(true) for \
         the standard sub-Feller mitigation"
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
      use_sym,
      seed,
      fgn: Self::fgn_for(hurst, n, t),
    }
  }

  /// The fGN driver of a path with `n` points (one increment per step),
  /// shared by `new()` and every `with_*` setter that feeds it so they
  /// can never drift apart.
  fn fgn_for(hurst: T, n: usize, t: Option<T>) -> Fgn<T, Unseeded, Cpu> {
    Fgn::new(hurst, n - 1, t, Unseeded)
  }

  /// Replace `hurst`; rebuilds the embedded `fgn`.
  pub fn with_hurst(mut self, hurst: T) -> Self {
    self.hurst = hurst;
    self.fgn = Self::fgn_for(hurst, self.n, self.t);
    self
  }

  /// Replace `theta`, all else unchanged.
  pub fn with_theta(mut self, theta: T) -> Self {
    self.theta = theta;
    self
  }

  /// Replace `mu`, all else unchanged.
  pub fn with_mu(mut self, mu: T) -> Self {
    self.mu = mu;
    self
  }

  /// Replace `sigma`, all else unchanged.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    self.sigma = sigma;
    self
  }

  /// Replace the number of simulation steps `n`; rebuilds the embedded
  /// `fgn`. Panics if `n < 2`, matching `new()`'s own assertion.
  pub fn with_steps(mut self, n: usize) -> Self {
    assert!(n >= 2, "n must be at least 2");
    self.n = n;
    self.fgn = Self::fgn_for(self.hurst, n, self.t);
    self
  }

  /// Replace `x0`, all else unchanged.
  pub fn with_x0(mut self, x0: Option<T>) -> Self {
    self.x0 = x0;
    self
  }

  /// Replace the simulation horizon `t`; rebuilds the embedded `fgn`.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    self.t = t;
    self.fgn = Self::fgn_for(self.hurst, self.n, t);
    self
  }

  /// Replace `use_sym`, all else unchanged.
  pub fn with_use_sym(mut self, use_sym: Option<bool>) -> Self {
    self.use_sym = use_sym;
    self
  }

  /// Replace the seed strategy's value, all else unchanged. `fgn`'s own
  /// seed is a never-read dummy, so this does not touch it.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

impl<T: FloatExt, S: SeedExt, B> Fcir<T, S, B> {
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

  /// Enables symmetric/truncated update variant when true.
  pub fn use_sym(&self) -> Option<bool> {
    self.use_sym
  }

  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub fn seed(&self) -> &S {
    &self.seed
  }
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

/// Reusable [`Fcir`] sampling state: borrows the process for its inner [`Fgn`]
/// and owns a seed derived once at construction. The path is an Euler
/// discretisation of `dX = theta(mu - X) dt + sigma sqrt(X) dB^H`, clamped at
/// zero (or reflected when `use_sym`) so the variance stays non-negative.
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
    let use_sym = p.use_sym.unwrap_or(false);

    out[0] = p.x0.unwrap_or(T::zero());
    let mut prev = out[0];
    for (dst, inc) in out[1..].iter_mut().zip(fgn.iter()) {
      let dfcir = p.theta * (p.mu - prev) * dt + p.sigma * prev.abs().sqrt() * *inc;
      let next = match use_sym {
        true => (prev + dfcir).abs(),
        false => (prev + dfcir).max(T::zero()),
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

/// The Euler engine's view of fractional CIR, in the reflected or the
/// mirrored form depending on `use_sym`.
impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>>
  crate::euler::EulerCoefficients<T> for Fcir<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    if self.use_sym.unwrap_or(false) {
      crate::euler::EulerSpec::MirroredSquareRoot {
        theta: self.theta,
        mu: self.mu,
        sigma: self.sigma,
      }
    } else {
      crate::euler::EulerSpec::ReflectedSquareRoot {
        theta: self.theta,
        mu: self.mu,
        sigma: self.sigma,
      }
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
    Some(crate::euler::FgnSpec {
      sqrt_eigenvalues: self.fgn.sqrt_eigenvalues.as_slice().expect("contiguous"),
      n: self.fgn.n,
      offset: self.fgn.offset,
      hurst: self.fgn.hurst.to_f64().unwrap_or(0.5),
      t: self.fgn.t.unwrap_or(T::one()).to_f64().unwrap_or(1.0),
      streams: 1,
    })
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Fcir<T, S> { hurst, theta, mu, sigma, n, x0, t, use_sym, seed } via fgn euler);

py_process_1d!(PyFcir, Fcir,
  sig: (hurst, theta, mu, sigma, n, x0=None, t=None, use_sym=None, seed=None, dtype=None),
  params: (hurst: f64, theta: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>, use_sym: Option<bool>),
  device
);

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  /// `2*theta*mu = 2*0.5*0.1 = 0.1 < sigma^2 = 1.0` — Feller condition
  /// violated, mirroring the equivalent `Cir` test. `use_sym = Some(true)`
  /// must build and sample without panicking.
  #[test]
  fn fcir_accepts_sub_feller_with_use_sym() {
    let fcir = Fcir::<f64, _>::new(
      0.7,
      0.5,
      0.1,
      1.0,
      256,
      Some(0.1),
      Some(1.0),
      Some(true),
      Deterministic::new(7),
    );
    let path = fcir.sample();
    assert_eq!(path.len(), 256);
    assert!(
      path.iter().all(|x| x.is_finite()),
      "sub-Feller Fcir path must stay finite under use_sym = Some(true)"
    );
  }

  /// The default (floor-at-zero) scheme must also accept sub-Feller
  /// parameters without panicking — only the diagnostic warning differs.
  #[test]
  fn fcir_accepts_sub_feller_without_use_sym() {
    let fcir = Fcir::<f64, _>::new(
      0.7,
      0.5,
      0.1,
      1.0,
      256,
      Some(0.1),
      Some(1.0),
      None,
      Deterministic::new(7),
    );
    let path = fcir.sample();
    assert!(path.iter().all(|x| x.is_finite() && *x >= 0.0));
  }
}

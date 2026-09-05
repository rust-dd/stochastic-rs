//! # Fvasicek
//!
//! $$
//! dr_t=a(b-r_t)dt+\sigma dB_t^H
//! $$
//!
//! References:
//! - Vasicek O. (1977) — *An Equilibrium Characterization of the Term
//!   Structure*, Journal of Financial Economics 5(2), 177–188,
//!   DOI: 10.1016/0304-405X(77)90016-2 — the (non-fractional) short-rate
//!   model this generalises.
//! - Cheridito P., Kawaguchi H., Maejima M. (2003) — *Fractional
//!   Ornstein-Uhlenbeck Processes*, Electronic Journal of Probability 8,
//!   paper 3, 1–14, DOI: 10.1214/EJP.v8-125 — the fractional-noise
//!   driver ([`Fou`]) this file wraps under
//!   short-rate parameter names.
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::device::Cpu;
use crate::diffusion::fou::Fou;
use crate::diffusion::fou::FouSampler;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Clone)]
pub struct FVasicek<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Hurst exponent controlling roughness and long-memory.
  pub hurst: T,
  /// Mean-reversion speed (`a` in the module header's `a(b-r_t)dt`) —
  /// despite the name, this is the wrapped [`Fou::theta`] (speed), fed
  /// straight through to it.
  pub theta: T,
  /// Long-run mean level (`b` in the module header) — the wrapped
  /// [`Fou::mu`], the level `r` reverts to.
  pub mu: T,
  /// Diffusion scale σ multiplying `dB_t^H`.
  pub sigma: T,
  /// Number of points sampled along the fractional-Vasicek path.
  pub n: usize,
  /// Initial value X₀ of the fractional-Vasicek path.
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)), kept
  /// for API symmetry with every other process in the crate. Sampling
  /// itself never reads this field directly — [`FVasicek::new`] derives a
  /// child seed from it once, at construction, to seed [`fou`](Self::fou),
  /// which is what all sampling actually consults — including chunk
  /// decorrelation, via `fou`'s own `sampler()`.
  pub seed: S,
  /// Wrapped fractional-OU process carrying the actual sampling state;
  /// see module header — `FVasicek` is `Fou` under short-rate-model
  /// parameter names.
  pub fou: Fou<T, S>,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on) or [`on_device`](Self::on_device).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> FVasicek<T, S> {
  pub fn new(
    hurst: T,
    theta: T,
    mu: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    Self {
      backend: Cpu,
      hurst,
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      fou: Fou::new(hurst, theta, mu, sigma, n, x0, t, seed.derive()),
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> FVasicek<T, S, B> {}

/// The Euler engine's view of the fractional Vasicek model. It is the wrapped
/// [`Fou`] under short-rate names, so both the family and the increment
/// pipeline come from that process rather than being restated here.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>>
  crate::euler::EulerCoefficients<T> for FVasicek<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::OrnsteinUhlenbeck {
      theta: self.theta,
      mu: self.mu,
      sigma: self.sigma,
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
    rand::Rng::random(&mut self.seed.rng())
  }

  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }

  /// The wrapped process owns the increment pipeline, so this defers to it
  /// rather than restating the spectrum, the offset and the horizon.
  fn fgn_spec(&self) -> Option<crate::euler::FgnSpec<'_, T>> {
    crate::euler::EulerCoefficients::fgn_spec(&self.fou)
  }
}

backend_switch!([T: FloatExt, S: SeedExt] FVasicek<T, S> { hurst, theta, mu, sigma, n, x0, t, seed, fou } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for FVasicek<T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = FVasicekSampler<'s, T, S>
  where
    Self: 's;

  /// Builds and owns the wrapped [`Fou`]'s own sampler *once*, rather than
  /// calling [`Fou::sample`] (which would call `Fou::sampler()` fresh, and
  /// hence derive a new seed basis, on every path). Owning it here is what
  /// makes `sample_par`/`sample_map`'s chunked fan-out deterministic: each
  /// chunk's `FVasicekSampler` is built sequentially, so each owns a
  /// distinct `FouSampler` basis; repeat calls on one sampler reuse and
  /// advance that same inner sampler, exactly as a standalone `Fou` would.
  fn sampler(&self) -> FVasicekSampler<'_, T, S> {
    FVasicekSampler {
      fou: self.fou.sampler(),
    }
  }

  /// Through the Euler engine: on a device the fractional increments and the
  /// mean-reverting recursion run in one launch, on the host devices it is
  /// the wrapped process's own sampler.
  fn sample(&self) -> Array1<T> {
    self.backend.euler_sample(self)
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    self.backend.euler_paths_map(self, m, f)
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

/// Reusable [`FVasicek`] sampling state: owns the wrapped [`Fou`]'s sampler
/// (borrowed FFT plan, owned per-chunk seed), built once.
#[doc(hidden)]
pub struct FVasicekSampler<'a, T: FloatExt, S: SeedExt> {
  fou: FouSampler<'a, T, S, Cpu>,
}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for FVasicekSampler<'_, T, S> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    self.fou.sample_into(out);
  }

  fn sample(&mut self) -> Array1<T> {
    self.fou.sample()
  }
}

py_process_1d!(PyFVasicek, FVasicek,
  sig: (hurst, theta, mu, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (hurst: f64, theta: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>),
  device
);

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn sample_length_matches_n() {
    let v = FVasicek::<f64>::new(0.7, 0.5, 0.04, 0.01, 64, Some(0.05), Some(1.0), Unseeded);
    let path = v.sample();
    assert_eq!(path.len(), 64);
  }
}

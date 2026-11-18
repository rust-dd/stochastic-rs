//! The fractional OU with the short rate as its state: this type holds a [`Fou`] (`a` = `theta`,
//! `b` = `mu`) and samples through it, on a child seed that [`Fvasicek::new`] derives.
//!
//! $$
//! dr_t=a(b-r_t)dt+\sigma dB_t^H
//! $$
//!
//! Reference: Vasicek (1977), J. Financ. Econ. 5(2), 177–188, DOI 10.1016/0304-405X(77)90016-2
//! Reference: Cheridito, Kawaguchi & Maejima (2003), EJP 8, paper 3, 1–14, DOI 10.1214/EJP.v8-125
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

#[derive(Debug, Clone)]
pub struct Fvasicek<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Hurst exponent controlling roughness and long-memory.
  pub hurst: T,
  /// Mean-reversion speed, `a` in the module's `a(b − r_t)dt`: the wrapped [`Fou::theta`].
  pub theta: T,
  /// Long-run mean level (`b` in the module header) — the wrapped
  /// [`Fou::mu`], the level `r` reverts to.
  pub mu: T,
  /// Diffusion scale σ multiplying `dB_t^H`.
  pub sigma: T,
  /// Number of points sampled along the fractional-Vasicek path.
  pub n: usize,
  /// Initial value; `None` starts at zero.
  pub x0: Option<T>,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Seed: [`Unseeded`] or a [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic).
  /// Sampling reads [`fou`](Self::fou)'s child seed, derived from this one in [`Fvasicek::new`].
  pub seed: S,
  /// The wrapped fractional OU, which carries the sampling state.
  pub fou: Fou<T, S>,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Fvasicek<T, S> {
  /// Panics unless a given `t` is positive and finite and every parameter is finite.
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

/// The Euler engine's view: the wrapped [`Fou`] under short-rate names, so the family, the state,
/// the grid and the increment pipeline all come from that process.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for Fvasicek<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerCoefficients::euler_spec(&self.fou)
  }

  fn initial_value(&self) -> T {
    crate::euler::EulerCoefficients::initial_value(&self.fou)
  }

  fn grid_points(&self) -> usize {
    crate::euler::EulerCoefficients::grid_points(&self.fou)
  }

  fn horizon(&self) -> T {
    crate::euler::EulerCoefficients::horizon(&self.fou)
  }

  /// Not taken from the embedded `Fou`, whose seed is a child of this one: reading it here would
  /// give an `Fvasicek` the stream of the `Fou` it holds.
  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
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

backend_switch!([T: FloatExt, S: SeedExt] Fvasicek<T, S> { hurst, theta, mu, sigma, n, x0, t, seed, fou } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Fvasicek<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for Fvasicek<T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = FvasicekSampler<'s, T, S>
  where
    Self: 's;

  /// Owns the wrapped [`Fou`]'s sampler, built once; [`Fou::sample`] would derive a new basis on
  /// every path. Chunks are built in sequence, so each owns a distinct `FouSampler` basis.
  fn sampler(&self) -> FvasicekSampler<'_, T, S> {
    FvasicekSampler {
      fou: self.fou.sampler(),
    }
  }

  /// Through the Euler engine: a device runs the fractional increments and the recursion in one
  /// launch, the host runs the wrapped process's own sampler.
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

/// Reusable [`Fvasicek`] sampling state: owns the wrapped [`Fou`]'s sampler
/// (borrowed FFT plan, owned per-chunk seed), built once.
#[doc(hidden)]
#[derive(Debug)]
pub struct FvasicekSampler<'a, T: FloatExt, S: SeedExt> {
  fou: FouSampler<'a, T, S, Cpu>,
}

impl<T: FloatExt, S: SeedExt> crate::traits::Sealed for FvasicekSampler<'_, T, S> {}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for FvasicekSampler<'_, T, S> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    self.fou.sample_into(out);
  }

  fn sample(&mut self) -> Array1<T> {
    self.fou.sample()
  }
}

py_process_1d!(PyFVasicek, Fvasicek,
  sig: (hurst, theta, mu, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (hurst: f64, theta: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>),
  device
);

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn sample_length_matches_n() {
    let v = Fvasicek::<f64>::new(0.7, 0.5, 0.04, 0.01, 64, Some(0.05), Some(1.0), Unseeded);
    let path = v.sample();
    assert_eq!(path.len(), 64);
  }
}

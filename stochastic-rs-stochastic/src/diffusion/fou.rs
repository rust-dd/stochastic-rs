//! # fOU
//!
//! $$
//! dX_t=\theta(\mu-X_t)\,dt+\sigma\,dB_t^H
//! $$
//!
//! Reference: Cheridito P., Kawaguchi H., Maejima M. (2003) —
//! *Fractional Ornstein-Uhlenbeck Processes*, Electronic Journal of
//! Probability 8, paper 3, 1–14, DOI: 10.1214/EJP.v8-125.
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

#[derive(Clone)]
pub struct Fou<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Hurst exponent controlling roughness and long-memory.
  pub hurst: T,
  /// Mean-reversion speed.
  pub theta: T,
  /// Long-run mean level.
  pub mu: T,
  /// Diffusion scale σ multiplying `dB_t^H`.
  pub sigma: T,
  /// Number of points sampled along the fOU path.
  pub n: usize,
  /// Initial value X₀ of the fOU path.
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  fgn: Fgn<T, Unseeded, B>,
}

impl<T: FloatExt, S: SeedExt> Fou<T, S, Cpu> {
  #[must_use]
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
    assert!(n >= 2, "n must be at least 2");

    Self {
      hurst,
      theta,
      mu,
      sigma,
      n,
      x0,
      t,
      seed,
      fgn: Fgn::new(hurst, n - 1, t, Unseeded),
    }
  }
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>> ProcessExt<T>
  for Fou<T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = FouSampler<'s, T, S, B>
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
  fn sampler(&self) -> FouSampler<'_, T, S, B> {
    FouSampler {
      fou: self,
      seed: self.seed.derive(),
    }
  }

  /// `m` paths through the Euler engine, which on a device runs the whole
  /// recursion in the kernel from fGN increments and on the host devices is
  /// the process's own sampler, chunked exactly as `ProcessExt` chunks — so
  /// the CPU stream is the one it always was.
  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    crate::euler::EulerBackend::euler_paths(&self.fgn.backend, self, m)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, crate::device::DeviceError> {
    crate::euler::EulerBackend::try_euler_paths(&self.fgn.backend, self, m)
  }
}

/// Reusable [`Fou`] sampling state: borrows the process for its inner [`Fgn`]
/// and owns a seed derived once at construction. The path is an Euler
/// discretisation of `dX = theta(mu - X) dt + sigma dB^H` started at `x0`.
#[doc(hidden)]
pub struct FouSampler<'a, T: FloatExt, S: SeedExt, B> {
  fou: &'a Fou<T, S, B>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> FouSampler<'_, T, S, B> {
  fn try_fill_path(&mut self, out: &mut [T]) -> Result<(), DeviceError> {
    if out.is_empty() {
      return Ok(());
    }
    let p = self.fou;
    let dt = p.fgn.dt();
    let fgn = p.fgn.try_noise(&self.seed)?;

    out[0] = p.x0.unwrap_or(T::zero());
    let mut prev = out[0];
    for (dst, inc) in out[1..].iter_mut().zip(fgn.iter()) {
      let next = prev + p.theta * (p.mu - prev) * dt + p.sigma * *inc;
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

impl<T: FloatExt, S: SeedExt, B: FgnBackend<T>> PathSampler<T> for FouSampler<'_, T, S, B> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("Fou output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.fou.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }

  fn try_sample(&mut self) -> Result<Array1<T>, DeviceError> {
    let mut out = Array1::<T>::zeros(self.fou.n);
    self.try_fill_path(out.as_slice_mut().expect("Fou output must be contiguous"))?;
    Ok(out)
  }
}

/// The Euler engine's view of fOU: the same OU family a Gaussian process
/// uses, with the fractional increments supplied instead of hashed ones.
/// Nothing about the recursion differs, which is the point — a step that
/// multiplies by `dz` does not care where `dz` came from.
impl<T: FloatExt, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>>
  crate::euler::EulerCoefficients<T> for Fou<T, S, B>
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
    })
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Fou<T, S> { hurst, theta, mu, sigma, n, x0, t, seed } via fgn euler);

py_process_1d!(PyFou, Fou,
  sig: (hurst, theta, mu, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (hurst: f64, theta: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>),
  device
);

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Unseeded;

  use super::Fou;
  use crate::traits::ProcessExt;

  #[test]
  #[should_panic(expected = "n must be at least 2")]
  fn fou_requires_at_least_two_points() {
    let _ = Fou::<f64>::new(0.7, 1.0, 0.0, 0.2, 1, Some(0.0), Some(1.0), Unseeded);
  }

  #[test]
  fn fou_sigma_zero_matches_deterministic_euler() {
    let theta = 1.3_f64;
    let mu = 0.8_f64;
    let n = 129_usize;
    let x0 = 0.2_f64;
    let t = 1.0_f64;

    let p = Fou::<f64>::new(0.7, theta, mu, 0.0, n, Some(x0), Some(t), Unseeded);
    let x = p.sample();

    let dt = t / (n as f64 - 1.0);
    let mut expected = x0;
    for i in 1..n {
      expected = expected + theta * (mu - expected) * dt;
      assert!((x[i] - expected).abs() < 1e-12, "mismatch at index {i}");
    }
  }

  #[test]
  fn fou_dt_alignment_holds_for_multiple_grid_sizes() {
    let theta = 0.9_f64;
    let mu = -0.1_f64;
    let x0 = 0.35_f64;
    let hs = [0.55_f64, 0.9_f64];
    let ns = [3_usize, 17, 129, 1000];
    let ts = [0.7_f64, 2.0_f64];

    for &h in &hs {
      for &n in &ns {
        for &t in &ts {
          let p = Fou::<f64>::new(h, theta, mu, 0.0, n, Some(x0), Some(t), Unseeded);
          let x = p.sample();

          let dt = t / (n as f64 - 1.0);
          let mut expected = x0;
          for i in 1..n {
            expected = expected + theta * (mu - expected) * dt;
            assert!(
              (x[i] - expected).abs() < 1e-12,
              "mismatch at i={i}, n={n}, t={t}, h={h}"
            );
          }
        }
      }
    }
  }

  #[test]
  fn fou_sample_is_finite() {
    let p = Fou::<f64>::new(0.65, 1.0, 0.0, 0.5, 256, Some(0.1), Some(1.0), Unseeded);
    let x = p.sample();
    assert_eq!(x.len(), 256);
    assert!(x.iter().all(|v| v.is_finite()));
  }
}

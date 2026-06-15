//! # fCIR
//!
//! $$
//! dX_t=\kappa(\theta-X_t)dt+\sigma\sqrt{X_t}\,dB_t^H
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::buffer::array1_from_fill;
use crate::device::Backend;
use crate::device::Cpu;
use crate::noise::fgn::Fgn;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Fractional Cox-Ingersoll-Ross (Fcir) process.
/// dX(t) = theta(mu - X(t))dt + sigma * sqrt(X(t))dW^H(t)
/// where X(t) is the Fcir process.
pub struct Fcir<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Hurst exponent controlling roughness and long-memory.
  pub hurst: T,
  /// Long-run target level / model location parameter.
  pub theta: T,
  /// Drift / long-run mean-level parameter.
  pub mu: T,
  /// Diffusion / noise scale parameter.
  pub sigma: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Initial value of the primary state variable.
  pub x0: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Enables symmetric/truncated update variant when true.
  pub use_sym: Option<bool>,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
  fgn: Fgn<T, Unseeded, B>,
}

impl<T: FloatExt, S: SeedExt> Fcir<T, S, Cpu> {
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
    assert!(
      T::from_usize_(2) * theta * mu >= sigma.powi(2),
      "2 * theta * mu < sigma^2"
    );

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
      fgn: Fgn::new(hurst, n - 1, t, Unseeded),
    }
  }
}

impl<T: FloatExt, S: SeedExt, B: Backend> ProcessExt<T> for Fcir<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = FcirSampler<'s, T, S, B>
  where
    Self: 's;

  /// A CPU sampler borrowing the process for its inner [`Fgn`] (`Arc`-shared
  /// FFT plan + eigenvalues) and seed source. The first `sample` derives the
  /// same child seed the legacy `sample()` did — bit-identical — and each
  /// subsequent call advances the seed for an independent path.
  fn sampler(&self) -> FcirSampler<'_, T, S, B> {
    FcirSampler { fcir: self }
  }
}

/// Reusable [`Fcir`] sampling state: borrows the process for its inner [`Fgn`]
/// and seed source. The path is an Euler discretisation of
/// `dX = theta(mu - X) dt + sigma sqrt(X) dB^H`, clamped at zero (or reflected
/// when `use_sym`) so the variance stays non-negative.
#[doc(hidden)]
pub struct FcirSampler<'a, T: FloatExt, S: SeedExt, B> {
  fcir: &'a Fcir<T, S, B>,
}

impl<T: FloatExt, S: SeedExt, B: Backend> FcirSampler<'_, T, S, B> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    let p = self.fcir;
    let dt = p.fgn.dt();
    let fgn = p.fgn.noise(&p.seed.derive());
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
  }
}

impl<T: FloatExt, S: SeedExt, B: Backend> PathSampler<T> for FcirSampler<'_, T, S, B> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("Fcir output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.fcir.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Fcir<T, S> { hurst, theta, mu, sigma, n, x0, t, use_sym, seed } via fgn);

py_process_1d!(PyFcir, Fcir,
  sig: (hurst, theta, mu, sigma, n, x0=None, t=None, use_sym=None, seed=None, dtype=None),
  params: (hurst: f64, theta: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>, use_sym: Option<bool>)
);

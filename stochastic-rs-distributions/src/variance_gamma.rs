//! # Variance Gamma
//!
//! Madan, Carr and Chang's VG law at unit time — a Brownian motion with
//! drift $\theta$ and volatility $\sigma$ evaluated at a gamma time with
//! unit mean and variance $\nu$, shifted by $\mu$:
//!
//! $$
//! X = \mu + \theta G + \sigma\sqrt{G}\,Z,\qquad G \sim \mathrm{Gamma}(1/\nu,\ \nu),\ Z \sim \mathcal N(0,1),
//! $$
//!
//! $$
//! f(x) = \frac{2\,e^{\theta(x-\mu)/\sigma^2}}{\sigma\sqrt{2\pi}\,\nu^{1/\nu}\,\Gamma(1/\nu)}
//! \left(\frac{|x-\mu|}{\sqrt{2\sigma^2/\nu + \theta^2}}\right)^{\frac1\nu - \frac12}
//! K_{\frac1\nu - \frac12}\!\left(\frac{|x-\mu|\sqrt{2\sigma^2/\nu + \theta^2}}{\sigma^2}\right),
//! \qquad
//! \varphi(u) = e^{iu\mu}\bigl(1 - i\theta\nu u + \tfrac12\sigma^2\nu u^2\bigr)^{-1/\nu}
//! $$
//!
//! (Madan–Carr–Chang 1998, eq. 23 and eq. 2 at $t = 1$). The mean is
//! $\mu + \theta$, the variance $\sigma^2 + \nu\theta^2$; skewness and excess
//! kurtosis follow from the gamma mixture's cumulants, reducing to $0$ and
//! $3\nu$ when $\theta = 0$. At $x = \mu$ the density is finite only for
//! $\nu < 2$. There is no closed-form CDF or quantile.
//!
//! Reference: Madan, D.B., Carr, P.P., Chang, E.C. (1998), "The Variance
//! Gamma Process and Option Pricing", *European Finance Review* 2(1),
//! 79-105. DOI: 10.1023/A:1009703431535

use std::cell::Cell;
use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::Unseeded;

use super::SimdFloatExt;
use super::gamma::SimdGamma;
use super::normal::SimdNormal;
use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;
use crate::special::bessel_k::bessel_ke;
use crate::special::ln_gamma;

const SMALL_VG_THRESHOLD: usize = 16;

/// Variance-gamma distribution with volatility `σ > 0`, variance rate
/// `ν > 0` of the gamma clock, drift `θ` and location `μ`.
pub struct SimdVarianceGamma<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  sigma: T,
  nu: T,
  theta: T,
  mu: T,
  gamma: SimdGamma<T, R>,
  normal: SimdNormal<T, 64, R>,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  stream_seed: Cell<u64>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdVarianceGamma<T, R> {
  /// Construct a VG$(\sigma, \nu, \theta, \mu)$ over an internal gamma
  /// subordinator with shape $1/\nu$ and scale $\nu$.
  pub fn new<S: crate::simd_rng::SeedExt>(sigma: T, nu: T, theta: T, mu: T, seed: &S) -> Self {
    assert!(sigma > T::zero(), "VG: sigma must be positive");
    assert!(nu > T::zero(), "VG: nu must be positive");
    let gamma = SimdGamma::<T, R>::new(T::one() / nu, nu, seed);
    let normal = SimdNormal::<T, 64, R>::new(T::zero(), T::one(), seed);
    let stream_seed = seed.seed_value();
    Self {
      sigma,
      nu,
      theta,
      mu,
      gamma,
      normal,
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
      stream_seed: Cell::new(stream_seed),
    }
  }

  /// Builds an independent worker stream for the `stream_idx`-th chunk of a
  /// parallel `sample_matrix` fan-out; see
  /// [`DistributionSampler::fork`](crate::traits::DistributionSampler::fork).
  #[doc(hidden)]
  pub fn fork(&self, stream_idx: u64) -> Self {
    let mut basis = self.stream_seed.get();
    let call_basis = crate::simd_rng::derive_seed(&mut basis);
    self.stream_seed.set(basis);
    let child_seed = crate::simd_rng::derive_fork_seed(call_basis, stream_idx);
    Self::new(
      self.sigma,
      self.nu,
      self.theta,
      self.mu,
      &crate::simd_rng::Deterministic::new(child_seed),
    )
  }

  /// Returns a single sample using the internal SIMD RNG.
  #[inline]
  pub fn sample_fast(&self) -> T {
    let index = unsafe { &mut *self.index.get() };
    if *index >= 16 {
      self.refill_buffer();
    }
    let buf = unsafe { &mut *self.buffer.get() };
    let z = buf[*index];
    *index += 1;
    z
  }

  /// Fills `out` from the gamma-time Brownian mixture on the internal SIMD
  /// streams; the gamma clock and the normal each refill their own buffer.
  pub fn fill_slice(&self, out: &mut [T]) {
    if out.len() < SMALL_VG_THRESHOLD {
      for x in out.iter_mut() {
        let g = self.gamma.sample_fast();
        let z = self.normal.sample_fast();
        *x = self.mu + self.theta * g + self.sigma * g.sqrt() * z;
      }
      return;
    }
    let mu = T::splat(self.mu);
    let theta = T::splat(self.theta);
    let sigma = T::splat(self.sigma);
    let mut gbuf = [T::zero(); 64];
    let mut zbuf = [T::zero(); 64];
    let (chunks, rem) = out.as_chunks_mut::<64>();
    for chunk in chunks {
      self.gamma.fill_slice(&mut gbuf);
      self.normal.fill_standard_fast(&mut zbuf);
      for (sub, (g8, z8)) in chunk.as_chunks_mut::<8>().0.iter_mut().zip(
        gbuf
          .as_chunks::<8>()
          .0
          .iter()
          .zip(zbuf.as_chunks::<8>().0.iter()),
      ) {
        let g = T::simd_from_array(*g8);
        let z = T::simd_from_array(*z8);
        let x = mu + theta * g + sigma * T::simd_sqrt(g) * z;
        *sub = T::simd_to_array(x);
      }
    }
    if !rem.is_empty() {
      let n = rem.len();
      self.gamma.fill_slice(&mut gbuf[..n]);
      self.normal.fill_standard_fast(&mut zbuf[..n]);
      for i in 0..n {
        rem[i] = self.mu + self.theta * gbuf[i] + self.sigma * gbuf[i].sqrt() * zbuf[i];
      }
    }
  }

  fn refill_buffer(&self) {
    let buf = unsafe { &mut *self.buffer.get() };
    self.fill_slice(buf);
    unsafe {
      *self.index.get() = 0;
    }
  }

  fn params(&self) -> (f64, f64, f64, f64) {
    (
      self.sigma.to_f64().unwrap(),
      self.nu.to_f64().unwrap(),
      self.theta.to_f64().unwrap(),
      self.mu.to_f64().unwrap(),
    )
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdVarianceGamma<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.sigma, self.nu, self.theta, self.mu, &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Distribution<T> for SimdVarianceGamma<T, R> {
  /// The `rng` argument is intentionally unused — this type draws from its
  /// own internal SIMD streams seeded at construction.
  fn sample<Rr: Rng + ?Sized>(&self, _rng: &mut Rr) -> T {
    let idx = unsafe { &mut *self.index.get() };
    if *idx >= 16 {
      self.refill_buffer();
    }
    let val = unsafe { (*self.buffer.get())[*idx] };
    *idx += 1;
    val
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> crate::traits::DistributionExt for SimdVarianceGamma<T, R> {
  /// Madan–Carr–Chang eq. 23, evaluated in log space through the scaled
  /// Bessel function; at `x = μ` the $y^a K_a(y)$ limit $2^{a-1}\Gamma(a)$
  /// for $a = 1/\nu - 1/2 > 0$, `+∞` otherwise.
  fn pdf(&self, x: f64) -> f64 {
    let (sigma, nu, theta, mu) = self.params();
    let a = 1.0 / nu - 0.5;
    let root = (2.0 * sigma * sigma / nu + theta * theta).sqrt();
    let log_norm = std::f64::consts::LN_2
      - sigma.ln()
      - 0.5 * (2.0 * std::f64::consts::PI).ln()
      - nu.ln() / nu
      - ln_gamma(1.0 / nu);
    let d = x - mu;
    if d == 0.0 {
      if a <= 0.0 {
        return f64::INFINITY;
      }
      let scale = sigma * sigma / (root * root);
      return (log_norm + a * scale.ln() + (a - 1.0) * std::f64::consts::LN_2 + ln_gamma(a)).exp();
    }
    let y = d.abs() * root / (sigma * sigma);
    let log_kernel =
      theta * d / (sigma * sigma) + a * (d.abs() / root).ln() + bessel_ke(a, y).ln() - y;
    (log_norm + log_kernel).exp()
  }

  fn cdf(&self, _x: f64) -> f64 {
    unimplemented!("DistributionExt::cdf for SimdVarianceGamma has no closed form")
  }

  fn inv_cdf(&self, _p: f64) -> f64 {
    unimplemented!("DistributionExt::inv_cdf for SimdVarianceGamma has no closed form")
  }

  fn mean(&self) -> f64 {
    let (_, _, theta, mu) = self.params();
    mu + theta
  }

  fn variance(&self) -> f64 {
    let (sigma, nu, theta, _) = self.params();
    sigma * sigma + nu * theta * theta
  }

  /// $(2\theta^3\nu^2 + 3\theta\sigma^2\nu)/(\sigma^2 + \nu\theta^2)^{3/2}$.
  fn skewness(&self) -> f64 {
    let (sigma, nu, theta, _) = self.params();
    let var = sigma * sigma + nu * theta * theta;
    (2.0 * theta.powi(3) * nu * nu + 3.0 * theta * sigma * sigma * nu) / var.powf(1.5)
  }

  /// Excess kurtosis from the fourth central moment
  /// $\theta^4(3\nu^2 + 6\nu^3) + 6\theta^2\sigma^2(\nu + 2\nu^2) + 3\sigma^4(1 + \nu)$.
  fn kurtosis(&self) -> f64 {
    let (sigma, nu, theta, _) = self.params();
    let var = sigma * sigma + nu * theta * theta;
    let m4 = theta.powi(4) * (3.0 * nu * nu + 6.0 * nu.powi(3))
      + 6.0 * theta * theta * sigma * sigma * (nu + 2.0 * nu * nu)
      + 3.0 * sigma.powi(4) * (1.0 + nu);
    m4 / (var * var) - 3.0
  }

  /// $e^{\mu t}(1 - \theta\nu t - \tfrac12\sigma^2\nu t^2)^{-1/\nu}$ where
  /// the base is positive, `NaN` outside that domain.
  fn moment_generating_function(&self, t: f64) -> f64 {
    let (sigma, nu, theta, mu) = self.params();
    let base = 1.0 - theta * nu * t - 0.5 * sigma * sigma * nu * t * t;
    if base <= 0.0 {
      f64::NAN
    } else {
      (mu * t).exp() * base.powf(-1.0 / nu)
    }
  }

  /// $e^{iu\mu}(1 - i\theta\nu u + \tfrac12\sigma^2\nu u^2)^{-1/\nu}$.
  fn characteristic_function(&self, u: f64) -> num_complex::Complex64 {
    use num_complex::Complex64;
    let (sigma, nu, theta, mu) = self.params();
    let base = Complex64::new(1.0 + 0.5 * sigma * sigma * nu * u * u, -theta * nu * u);
    Complex64::new(0.0, mu * u).exp() * base.powf(-1.0 / nu)
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;
  use crate::traits::DistributionExt;

  fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * b.abs().max(1e-300)
  }

  /// The Bessel-form density agrees with a numerical integration of the
  /// gamma-time normal mixture (`scipy.integrate.quad` over
  /// `norm.pdf(x; μ+θg, σ√g)·gamma.pdf(g; 1/ν, ν)`), including the finite
  /// `x = μ` limit for ν = 1/2.
  #[test]
  fn pdf_matches_the_mixture_integral() {
    let d = SimdVarianceGamma::<f64>::new(0.2, 0.5, -0.1, 0.05, &Unseeded);
    let grid = [
      (-1.0, 0.007_426_216_254_428_035),
      (-0.3, 0.684_113_783_523_804_4),
      (0.0, 2.341_138_247_888_838_7),
      (0.05, 2.282_688_235_636_076),
      (0.5, 0.040_416_261_951_548_835),
      (2.0, 6.857_510_508_067_127e-10),
    ];
    for (x, want) in grid {
      assert!(
        close(d.pdf(x), want, 1e-9),
        "pdf({x}) = {} vs {want}",
        d.pdf(x)
      );
    }
    let heavy = SimdVarianceGamma::<f64>::new(1.0, 2.0, 0.3, 0.0, &Unseeded);
    let grid = [
      (-1.0, 0.093_258_741_713_085_1),
      (-0.3, 0.387_808_764_238_471_16),
      (0.05, 0.992_415_312_923_996_3),
      (0.5, 0.328_754_432_477_837_9),
      (2.0, 0.059_310_979_016_730_006),
    ];
    for (x, want) in grid {
      assert!(
        close(heavy.pdf(x), want, 1e-9),
        "pdf({x}) = {} vs {want}",
        heavy.pdf(x)
      );
    }
    // ν = 2 puts the order of K at zero: the density diverges at μ.
    assert_eq!(heavy.pdf(0.0), f64::INFINITY);
  }

  /// Closed-form moments against the same reference (numerically
  /// integrated central moments of the mixture).
  #[test]
  fn moments_and_transforms_match_the_reference() {
    let d = SimdVarianceGamma::<f64>::new(0.2, 0.5, -0.1, 0.05, &Unseeded);
    assert!(close(d.mean(), -0.05, 1e-15));
    assert!(close(d.variance(), 0.045, 1e-15));
    assert!(close(d.skewness() * 0.045_f64.powf(1.5), -0.0065, 1e-12));
    assert!(close((d.kurtosis() + 3.0) * 0.045 * 0.045, 0.00975, 1e-12));
    let cf = d.characteristic_function(0.7);
    assert!(
      close(cf.re, 0.988_478_712_093_533_1, 1e-12) && close(cf.im, -0.034_245_228_379_174_4, 1e-12)
    );
    let heavy = SimdVarianceGamma::<f64>::new(1.0, 2.0, 0.3, 0.0, &Unseeded);
    assert!(close(heavy.variance(), 1.18, 1e-15));
    assert!(close(heavy.skewness() * 1.18_f64.powf(1.5), 2.016, 1e-12));
    assert!(close((heavy.kurtosis() + 3.0) * 1.18 * 1.18, 14.886, 1e-12));
    let symmetric = SimdVarianceGamma::<f64>::new(0.3, 0.7, 0.0, 0.0, &Unseeded);
    assert_eq!(symmetric.skewness(), 0.0);
    assert!(close(symmetric.kurtosis(), 3.0 * 0.7, 1e-12));
    assert!(d.moment_generating_function(50.0).is_nan());
    assert!(close(d.moment_generating_function(0.0), 1.0, 1e-15));
  }

  #[test]
  fn pdf_integrates_to_one() {
    let d = SimdVarianceGamma::<f64>::new(0.2, 0.5, -0.1, 0.05, &Unseeded);
    let (lo, hi, n) = (-6.0_f64, 6.0_f64, 600_000usize);
    let h = (hi - lo) / n as f64;
    let s: f64 = (0..n).map(|k| d.pdf(lo + (k as f64 + 0.5) * h) * h).sum();
    assert!((s - 1.0).abs() < 1e-6, "integral = {s}");
  }

  /// Sample mean, variance and skewness from the subordinated sampler
  /// match the closed forms.
  #[test]
  fn sample_moments_match_closed_forms() {
    let d = SimdVarianceGamma::<f64>::new(0.2, 0.5, -0.1, 0.05, &Deterministic::new(11));
    let n = 400_000;
    let mut xs = vec![0.0; n];
    d.fill_slice(&mut xs);
    let mean = xs.iter().sum::<f64>() / n as f64;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let m3 = xs.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n as f64;
    assert!((mean - d.mean()).abs() < 2e-3, "mean {mean}");
    assert!(
      (var - d.variance()).abs() / d.variance() < 0.02,
      "var {var}"
    );
    assert!(
      (m3 / var.powf(1.5) - d.skewness()).abs() < 0.05,
      "skew {}",
      m3 / var.powf(1.5)
    );
  }

  #[test]
  fn deterministic_seed_reproduces_stream() {
    let a = SimdVarianceGamma::<f64>::new(0.2, 0.5, -0.1, 0.05, &Deterministic::new(7));
    let b = SimdVarianceGamma::<f64>::new(0.2, 0.5, -0.1, 0.05, &Deterministic::new(7));
    for _ in 0..256 {
      assert_eq!(a.sample_fast(), b.sample_fast());
    }
  }
}

py_distribution!(PyVarianceGamma, SimdVarianceGamma,
  sig: (sigma, nu, theta, mu, seed=None, dtype=None),
  params: (sigma: f64, nu: f64, theta: f64, mu: f64)
);

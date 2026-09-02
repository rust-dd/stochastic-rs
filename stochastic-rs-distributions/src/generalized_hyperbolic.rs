//! # Generalized Hyperbolic (GH)
//!
//! Barndorff-Nielsen's normal mean-variance mixture over a generalized
//! inverse Gaussian clock,
//!
//! $$
//! X = \mu + \beta W + \sqrt{W}\,Z,\qquad W \sim \mathrm{GIG}(\lambda, \delta^2, \alpha^2 - \beta^2),\ Z \sim \mathcal N(0,1),
//! $$
//!
//! $$
//! f(x) = \frac{(\alpha^2-\beta^2)^{\lambda/2}}{\sqrt{2\pi}\,\alpha^{\lambda-1/2}\,\delta^\lambda\,K_\lambda\bigl(\delta\sqrt{\alpha^2-\beta^2}\bigr)}
//! \bigl(\delta^2 + (x-\mu)^2\bigr)^{\frac{\lambda - 1/2}{2}}
//! K_{\lambda-1/2}\!\Bigl(\alpha\sqrt{\delta^2 + (x-\mu)^2}\Bigr)\,e^{\beta(x-\mu)},
//! $$
//!
//! with $\lambda \in \mathbb R$, $\alpha > |\beta|$, $\delta > 0$, $\mu \in
//! \mathbb R$ (the sampler needs the interior of the parameter space). The
//! normal-inverse-Gaussian is $\lambda = -1/2$, the hyperbolic $\lambda = 1$,
//! and the variance gamma the $\delta \to 0$ limit for $\lambda > 0$. Moments
//! follow from the mixture: the mean is $\mu + \beta\,\mathbb E W$, the
//! variance $\mathbb E W + \beta^2\mathrm{Var}\,W$, and the third and fourth
//! central moments $\beta^3\mu_3(W) + 3\beta\mathrm{Var}\,W$ and
//! $\beta^4\mu_4(W) + 6\beta^2(\mu_3(W) + \mathbb E W\,\mathrm{Var}\,W) + 3\,\mathbb E W^2$.
//! There is no closed-form CDF or quantile.
//!
//! References:
//! - Barndorff-Nielsen, O. (1977), "Exponentially decreasing distributions
//!   for the logarithm of particle size", *Proceedings of the Royal Society
//!   A* 353(1674), 401-419. DOI: 10.1098/rspa.1977.0041
//! - Eberlein, E., Keller, U. (1995), "Hyperbolic Distributions in Finance",
//!   *Bernoulli* 1(3), 281-299. DOI: 10.2307/3318481
//! - Prause, K. (1999), *The Generalized Hyperbolic Model: Estimation,
//!   Financial Derivatives, and Risk Measures*, PhD thesis, University of
//!   Freiburg, ch. 1.

use std::cell::Cell;
use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::Unseeded;

use super::SimdFloatExt;
use super::generalized_inverse_gauss::SimdGig;
use super::normal::SimdNormal;
use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;
use crate::special::bessel_k::bessel_ke;

const SMALL_GH_THRESHOLD: usize = 16;

/// Generalized hyperbolic distribution GH$(\lambda, \alpha, \beta, \delta, \mu)$.
pub struct SimdGeneralizedHyperbolic<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  lambda: T,
  alpha: T,
  beta: T,
  delta: T,
  mu: T,
  gig: SimdGig<T, R>,
  normal: SimdNormal<T, 64, R>,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  stream_seed: Cell<u64>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdGeneralizedHyperbolic<T, R> {
  /// Construct a GH$(\lambda, \alpha, \beta, \delta, \mu)$ over an internal
  /// GIG$(\lambda, \delta^2, \alpha^2 - \beta^2)$ clock.
  pub fn new<S: crate::simd_rng::SeedExt>(
    lambda: T,
    alpha: T,
    beta: T,
    delta: T,
    mu: T,
    seed: &S,
  ) -> Self {
    assert!(delta > T::zero(), "GH: delta must be positive");
    assert!(
      alpha > T::zero() && alpha > beta.abs(),
      "GH: alpha must exceed |beta|"
    );
    let gig = SimdGig::<T, R>::new(lambda, delta * delta, alpha * alpha - beta * beta, seed);
    let normal = SimdNormal::<T, 64, R>::new(T::zero(), T::one(), seed);
    let stream_seed = seed.seed_value();
    Self {
      lambda,
      alpha,
      beta,
      delta,
      mu,
      gig,
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
      self.lambda,
      self.alpha,
      self.beta,
      self.delta,
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

  /// Fills `out` from the GIG-clock normal mixture; the clock draws are
  /// scalar rejection samples, the mixing step runs 8-wide.
  pub fn fill_slice(&self, out: &mut [T]) {
    if out.len() < SMALL_GH_THRESHOLD {
      for x in out.iter_mut() {
        let w = self.gig.sample_fast();
        let z = self.normal.sample_fast();
        *x = self.mu + self.beta * w + w.sqrt() * z;
      }
      return;
    }
    let mu = T::splat(self.mu);
    let beta = T::splat(self.beta);
    let mut wbuf = [T::zero(); 64];
    let mut zbuf = [T::zero(); 64];
    let (chunks, rem) = out.as_chunks_mut::<64>();
    for chunk in chunks {
      self.gig.fill_slice(&mut wbuf);
      self.normal.fill_standard_fast(&mut zbuf);
      for (sub, (w8, z8)) in chunk.as_chunks_mut::<8>().0.iter_mut().zip(
        wbuf
          .as_chunks::<8>()
          .0
          .iter()
          .zip(zbuf.as_chunks::<8>().0.iter()),
      ) {
        let w = T::simd_from_array(*w8);
        let z = T::simd_from_array(*z8);
        *sub = T::simd_to_array(mu + beta * w + T::simd_sqrt(w) * z);
      }
    }
    if !rem.is_empty() {
      let n = rem.len();
      self.gig.fill_slice(&mut wbuf[..n]);
      self.normal.fill_standard_fast(&mut zbuf[..n]);
      for i in 0..n {
        rem[i] = self.mu + self.beta * wbuf[i] + wbuf[i].sqrt() * zbuf[i];
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

  fn params(&self) -> (f64, f64, f64, f64, f64) {
    (
      self.lambda.to_f64().unwrap(),
      self.alpha.to_f64().unwrap(),
      self.beta.to_f64().unwrap(),
      self.delta.to_f64().unwrap(),
      self.mu.to_f64().unwrap(),
    )
  }

  /// Central moments $(\mathbb E W, \mathrm{Var}\,W, \mu_3(W), \mu_4(W))$ of
  /// the GIG clock.
  fn clock_moments(&self) -> (f64, f64, f64, f64) {
    let (m1, m2, m3, m4) = (
      self.gig.raw_moment(1),
      self.gig.raw_moment(2),
      self.gig.raw_moment(3),
      self.gig.raw_moment(4),
    );
    let var = m2 - m1 * m1;
    let mu3 = m3 - 3.0 * m1 * m2 + 2.0 * m1.powi(3);
    let mu4 = m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1.powi(4);
    (m1, var, mu3, mu4)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdGeneralizedHyperbolic<T, R> {
  fn clone(&self) -> Self {
    Self::new(
      self.lambda,
      self.alpha,
      self.beta,
      self.delta,
      self.mu,
      &Unseeded,
    )
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Distribution<T> for SimdGeneralizedHyperbolic<T, R> {
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

impl<T: SimdFloatExt, R: SimdRngExt> crate::traits::DistributionExt
  for SimdGeneralizedHyperbolic<T, R>
{
  fn pdf(&self, x: f64) -> f64 {
    let (lambda, alpha, beta, delta, mu) = self.params();
    let gamma = (alpha * alpha - beta * beta).sqrt();
    let d = x - mu;
    let root = (delta * delta + d * d).sqrt();
    let order = lambda - 0.5;
    let log_norm = lambda * gamma.ln()
      - 0.5 * (2.0 * std::f64::consts::PI).ln()
      - order * alpha.ln()
      - lambda * delta.ln()
      - (bessel_ke(lambda, delta * gamma).ln() - delta * gamma);
    let y = alpha * root;
    (log_norm + order * root.ln() + bessel_ke(order, y).ln() - y + beta * d).exp()
  }

  fn cdf(&self, _x: f64) -> f64 {
    unimplemented!("DistributionExt::cdf for SimdGeneralizedHyperbolic has no closed form")
  }

  fn inv_cdf(&self, _p: f64) -> f64 {
    unimplemented!("DistributionExt::inv_cdf for SimdGeneralizedHyperbolic has no closed form")
  }

  fn mean(&self) -> f64 {
    let (_, _, beta, _, mu) = self.params();
    mu + beta * self.gig.raw_moment(1)
  }

  fn variance(&self) -> f64 {
    let (_, _, beta, _, _) = self.params();
    let (ew, var, _, _) = self.clock_moments();
    ew + beta * beta * var
  }

  fn skewness(&self) -> f64 {
    let (_, _, beta, _, _) = self.params();
    let (ew, var, mu3, _) = self.clock_moments();
    let m2 = ew + beta * beta * var;
    (beta.powi(3) * mu3 + 3.0 * beta * var) / m2.powf(1.5)
  }

  /// Excess kurtosis.
  fn kurtosis(&self) -> f64 {
    let (_, _, beta, _, _) = self.params();
    let (ew, var, mu3, mu4) = self.clock_moments();
    let m2 = ew + beta * beta * var;
    let ew2 = self.gig.raw_moment(2);
    (beta.powi(4) * mu4 + 6.0 * beta * beta * (mu3 + ew * var) + 3.0 * ew2) / (m2 * m2) - 3.0
  }

  /// $e^{\mu t}\bigl(\tfrac{\alpha^2-\beta^2}{\alpha^2-(\beta+t)^2}\bigr)^{\lambda/2}
  /// K_\lambda\bigl(\delta\sqrt{\alpha^2-(\beta+t)^2}\bigr)/K_\lambda\bigl(\delta\sqrt{\alpha^2-\beta^2}\bigr)$
  /// for $|\beta + t| < \alpha$, `NaN` beyond.
  fn moment_generating_function(&self, t: f64) -> f64 {
    let (lambda, alpha, beta, delta, mu) = self.params();
    let shifted = alpha * alpha - (beta + t).powi(2);
    if shifted <= 0.0 {
      return f64::NAN;
    }
    let gamma = (alpha * alpha - beta * beta).sqrt();
    let gt = shifted.sqrt();
    (mu * t).exp() * (gamma * gamma / shifted).powf(0.5 * lambda) * bessel_ke(lambda, delta * gt)
      / bessel_ke(lambda, delta * gamma)
      * (delta * (gamma - gt)).exp()
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;
  use crate::normal_inverse_gauss::SimdNormalInverseGauss;
  use crate::traits::DistributionExt;

  fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * b.abs().max(1e-300)
  }

  /// The Bessel-form density agrees with `scipy.integrate.quad` over the
  /// GIG-clock normal mixture (`norm.pdf(x; μ+βw, √w)·geninvgauss.pdf(w)`),
  /// and the mixture moments with `geninvgauss.stats`.
  #[test]
  fn pdf_and_moments_match_the_mixture_reference() {
    let cases: [((f64, f64, f64, f64, f64), [f64; 6], f64, f64); 3] = [
      (
        (1.0, 2.0, 0.5, 1.5, -0.2),
        [
          0.026_874_696_001_484_795,
          0.289_432_836_608_569_1,
          0.384_055_007_209_786_75,
          0.389_575_357_963_150_46,
          0.280_187_896_740_125_84,
          0.030_237_901_725_699_894,
        ],
        0.400_326_852_907_575_25,
        1.310_435_697_043_944,
      ),
      (
        (-0.5, 3.0, -1.0, 0.8, 0.1),
        [
          0.016_014_118_038_505_026,
          0.537_142_357_104_786_2,
          0.823_021_530_784_875_8,
          0.556_377_253_304_921_9,
          0.048_358_038_340_823_45,
          7.013_824_572_701_23e-6,
        ],
        -0.182_842_712_474_619_06,
        0.318_198_051_533_946_4,
      ),
      (
        (0.7, 1.5, 0.0, 2.0, 0.0),
        [
          0.084_874_755_258_301_43,
          0.292_657_191_965_061_7,
          0.323_634_547_066_262_05,
          0.312_003_562_657_869_9,
          0.220_286_315_784_668_7,
          0.024_708_268_671_536_315,
        ],
        0.0,
        1.880_316_791_495_301_4,
      ),
    ];
    for ((lambda, alpha, beta, delta, mu), pdf, mean, var) in cases {
      let d = SimdGeneralizedHyperbolic::<f64>::new(lambda, alpha, beta, delta, mu, &Unseeded);
      for (x, want) in [-2.0, -0.5, 0.0, 0.3, 1.0, 3.0].into_iter().zip(pdf) {
        assert!(
          close(d.pdf(x), want, 1e-9),
          "λ={lambda}: pdf({x}) = {} vs {want}",
          d.pdf(x)
        );
      }
      assert!(
        (d.mean() - mean).abs() < 1e-10,
        "λ={lambda}: mean {}",
        d.mean()
      );
      assert!(
        close(d.variance(), var, 1e-10),
        "λ={lambda}: variance {}",
        d.variance()
      );
      assert!(close(d.moment_generating_function(0.0), 1.0, 1e-12));
    }
  }

  /// GH at λ = −1/2 is the normal-inverse Gaussian: the densities coincide
  /// with the crate's own NIG, which uses `K₁` directly.
  #[test]
  fn half_negative_lambda_is_the_nig() {
    let gh = SimdGeneralizedHyperbolic::<f64>::new(-0.5, 2.0, 0.5, 1.0, 0.3, &Unseeded);
    let nig = SimdNormalInverseGauss::<f64>::new(2.0, 0.5, 1.0, 0.3, &Unseeded);
    for x in [-3.0, -1.0, 0.0, 0.3, 1.0, 2.5, 6.0] {
      assert!(
        close(gh.pdf(x), nig.pdf(x), 1e-11),
        "pdf({x}): {} vs {}",
        gh.pdf(x),
        nig.pdf(x)
      );
    }
    assert!(close(gh.mean(), nig.mean(), 1e-11));
    assert!(close(gh.variance(), nig.variance(), 1e-10));
    assert!(close(gh.skewness(), nig.skewness(), 1e-9));
    assert!(close(gh.kurtosis(), nig.kurtosis(), 1e-8));
  }

  #[test]
  fn pdf_integrates_to_one() {
    let d = SimdGeneralizedHyperbolic::<f64>::new(1.0, 2.0, 0.5, 1.5, -0.2, &Unseeded);
    let (lo, hi, n) = (-30.0_f64, 30.0_f64, 600_000usize);
    let h = (hi - lo) / n as f64;
    let s: f64 = (0..n).map(|k| d.pdf(lo + (k as f64 + 0.5) * h) * h).sum();
    assert!((s - 1.0).abs() < 1e-6, "integral = {s}");
  }

  #[test]
  fn sample_moments_match_closed_forms() {
    let d = SimdGeneralizedHyperbolic::<f64>::new(1.0, 2.0, 0.5, 1.5, -0.2, &Deterministic::new(3));
    let n = 400_000;
    let mut xs = vec![0.0; n];
    d.fill_slice(&mut xs);
    let mean = xs.iter().sum::<f64>() / n as f64;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let m3 = xs.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n as f64;
    assert!(
      (mean - d.mean()).abs() < 0.01,
      "mean {mean} vs {}",
      d.mean()
    );
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
    let a = SimdGeneralizedHyperbolic::<f64>::new(1.0, 2.0, 0.5, 1.5, -0.2, &Deterministic::new(7));
    let b = SimdGeneralizedHyperbolic::<f64>::new(1.0, 2.0, 0.5, 1.5, -0.2, &Deterministic::new(7));
    for _ in 0..256 {
      assert_eq!(a.sample_fast(), b.sample_fast());
    }
  }

  #[test]
  #[should_panic(expected = "alpha must exceed |beta|")]
  fn rejects_beta_outside_alpha() {
    let _ = SimdGeneralizedHyperbolic::<f64>::new(1.0, 1.0, 1.0, 1.0, 0.0, &Unseeded);
  }
}

py_distribution!(PyGeneralizedHyperbolic, SimdGeneralizedHyperbolic,
  sig: (lambda, alpha, beta, delta, mu, seed=None, dtype=None),
  params: (lambda: f64, alpha: f64, beta: f64, delta: f64, mu: f64)
);

//! # Generalized Inverse Gaussian (GIG)
//!
//! $$
//! f(x) = \frac{(\psi/\chi)^{\lambda/2}}{2K_\lambda(\sqrt{\chi\psi})}\,x^{\lambda-1}
//! \exp\!\Bigl(-\tfrac12\bigl(\tfrac{\chi}{x} + \psi x\bigr)\Bigr),\qquad x > 0,
//! $$
//!
//! with $\lambda \in \mathbb R$ and, for the sampler, $\chi > 0$, $\psi > 0$
//! (the boundary cases are the gamma and inverse-gamma laws). The mixing
//! law behind the generalized hyperbolic family; the inverse Gaussian is
//! $\lambda = -1/2$.
//!
//! ## Sampling
//!
//! Hörmann and Leydold's uniformly fast generator on the two-parameter
//! quasi-density $g(y \mid \lambda, \beta) = y^{\lambda-1}e^{-\beta(y + 1/y)/2}$
//! with $\beta = \sqrt{\chi\psi}$, rescaled by $\alpha = \sqrt{\psi/\chi}$ as
//! $X = Y/\alpha$, and $1/Y$ for negative $\lambda$: their Algorithm 1
//! (three-piece hat with rejection constant below 2.73) for $\lambda < 1$ and
//! small $\beta$, the ratio-of-uniforms without mode shift (Algorithm 2)
//! in the $T_{-1/2}$-concave middle range, and the Dagpunar–Lehner
//! ratio-of-uniforms with mode shift (Algorithm 3, Cardano roots) for large
//! $\lambda$ or $\beta$ — the regime split of their `GIGrvg` reference
//! implementation.
//!
//! Raw moments are Bessel ratios, $\mathbb E X^k = (\chi/\psi)^{k/2}
//! K_{\lambda+k}(\sqrt{\chi\psi}) / K_\lambda(\sqrt{\chi\psi})$; there is no
//! closed-form CDF or quantile.
//!
//! References:
//! - Hörmann, W., Leydold, J. (2014), "Generating generalized inverse
//!   Gaussian random variates", *Statistics and Computing* 24(4), 547-557.
//!   DOI: 10.1007/s11222-013-9387-3
//! - Dagpunar, J.S. (1989), "An easily implemented generalised inverse
//!   Gaussian generator", *Communications in Statistics — Simulation and
//!   Computation* 18(2), 703-710. DOI: 10.1080/03610918908812785
//! - Jørgensen, B. (1982), *Statistical Properties of the Generalized
//!   Inverse Gaussian Distribution*, Lecture Notes in Statistics 9,
//!   Springer. DOI: 10.1007/978-1-4612-5698-4

use std::cell::Cell;
use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::Unseeded;

use super::SimdFloatExt;
use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;
use crate::special::bessel_k::bessel_ke;

/// Which Hörmann–Leydold generator the parameters select.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Regime {
  /// Algorithm 1: three-piece hat for $\lambda < 1$ and small $\beta$.
  Hat,
  /// Algorithm 2: ratio-of-uniforms without mode shift.
  RatioOfUniforms,
  /// Algorithm 3: ratio-of-uniforms with mode shift (Dagpunar–Lehner).
  RatioOfUniformsShifted,
}

/// Precomputed generator state for the quasi-density $g(y \mid \lambda, \beta)$
/// with $\lambda \ge 0$.
#[derive(Debug, Clone, Copy)]
struct Setup {
  lambda: f64,
  beta: f64,
  regime: Regime,
  /// $\log g(m)$, the normalisation that keeps every hat in `(0, 1]`.
  log_g_mode: f64,
  m: f64,
  x0: f64,
  x_star: f64,
  k2: f64,
  k3: f64,
  a1: f64,
  a2: f64,
  a3: f64,
  u_minus: f64,
  u_plus: f64,
}

impl Setup {
  fn new(lambda: f64, beta: f64) -> Self {
    let log_g = |x: f64| (lambda - 1.0) * x.ln() - 0.5 * beta * (x + 1.0 / x);
    let regime = if lambda > 2.0 || beta > 3.0 {
      Regime::RatioOfUniformsShifted
    } else if lambda >= 1.0 - 2.25 * beta * beta || beta > 0.2 {
      Regime::RatioOfUniforms
    } else {
      Regime::Hat
    };
    let mut s = Self {
      lambda,
      beta,
      regime,
      log_g_mode: 0.0,
      m: 0.0,
      x0: 0.0,
      x_star: 0.0,
      k2: 0.0,
      k3: 0.0,
      a1: 0.0,
      a2: 0.0,
      a3: 0.0,
      u_minus: 0.0,
      u_plus: 0.0,
    };
    match regime {
      Regime::Hat => {
        s.m = beta / ((1.0 - lambda) + ((1.0 - lambda).powi(2) + beta * beta).sqrt());
        s.log_g_mode = log_g(s.m);
        s.x0 = beta / (1.0 - lambda);
        s.x_star = s.x0.max(2.0 / beta);
        s.a1 = s.x0;
        if s.x0 < 2.0 / beta {
          s.k2 = (-beta - s.log_g_mode).exp();
          s.a2 = if lambda == 0.0 {
            s.k2 * (2.0 / (beta * beta)).ln()
          } else {
            s.k2 * ((2.0 / beta).powf(lambda) - s.x0.powf(lambda)) / lambda
          };
        }
        s.k3 = ((lambda - 1.0) * s.x_star.ln() - s.log_g_mode).exp();
        s.a3 = 2.0 * s.k3 * (-s.x_star * beta / 2.0).exp() / beta;
      }
      Regime::RatioOfUniforms => {
        s.m = beta / ((1.0 - lambda) + ((1.0 - lambda).powi(2) + beta * beta).sqrt());
        s.log_g_mode = log_g(s.m);
        let x_plus = ((1.0 + lambda) + ((1.0 + lambda).powi(2) + beta * beta).sqrt()) / beta;
        s.u_plus = x_plus * (0.5 * (log_g(x_plus) - s.log_g_mode)).exp();
      }
      Regime::RatioOfUniformsShifted => {
        s.m = (((lambda - 1.0).powi(2) + beta * beta).sqrt() + (lambda - 1.0)) / beta;
        s.log_g_mode = log_g(s.m);
        let a = -2.0 * (lambda + 1.0) / beta - s.m;
        let b = 2.0 * (lambda - 1.0) * s.m / beta - 1.0;
        let c = s.m;
        let p = b - a * a / 3.0;
        let q = 2.0 * a.powi(3) / 27.0 - a * b / 3.0 + c;
        let phi = (-(q / 2.0) * (-27.0 / p.powi(3)).sqrt())
          .clamp(-1.0, 1.0)
          .acos();
        let radius = (-4.0 * p / 3.0).sqrt();
        let x_minus = radius * (phi / 3.0 + 4.0 * std::f64::consts::PI / 3.0).cos() - a / 3.0;
        let x_plus = radius * (phi / 3.0).cos() - a / 3.0;
        s.u_minus = (x_minus - s.m) * (0.5 * (log_g(x_minus) - s.log_g_mode)).exp();
        s.u_plus = (x_plus - s.m) * (0.5 * (log_g(x_plus) - s.log_g_mode)).exp();
      }
    }
    s
  }

  /// $\log g(x) - \log g(m)$.
  #[inline]
  fn log_g_normalised(&self, x: f64) -> f64 {
    (self.lambda - 1.0) * x.ln() - 0.5 * self.beta * (x + 1.0 / x) - self.log_g_mode
  }

  /// One draw from $g(\cdot \mid \lambda, \beta)$ using uniforms and (for the
  /// shifted ratio-of-uniforms) nothing else.
  fn draw<T: SimdFloatExt, R: SimdRngExt>(&self, rng: &mut R) -> f64 {
    let uniform = |rng: &mut R| T::sample_uniform_simd(rng).to_f64().unwrap();
    match self.regime {
      Regime::Hat => loop {
        let u = uniform(rng);
        let v = uniform(rng) * (self.a1 + self.a2 + self.a3);
        let (x, log_h) = if v <= self.a1 {
          (self.x0 * v / self.a1, 0.0)
        } else if v <= self.a1 + self.a2 {
          let v = v - self.a1;
          let x = if self.lambda == 0.0 {
            self.beta * (v * self.beta.exp()).exp()
          } else {
            (self.x0.powf(self.lambda) + v * self.lambda / self.k2).powf(1.0 / self.lambda)
          };
          (x, self.k2.ln() + (self.lambda - 1.0) * x.ln())
        } else {
          let v = v - (self.a1 + self.a2);
          let x = -2.0 / self.beta
            * ((-self.x_star * self.beta / 2.0).exp() - v * self.beta / (2.0 * self.k3)).ln();
          (x, self.k3.ln() - x * self.beta / 2.0)
        };
        if x > 0.0 && u.ln() + log_h <= self.log_g_normalised(x) {
          return x;
        }
      },
      Regime::RatioOfUniforms => loop {
        let u = uniform(rng) * self.u_plus;
        let v = uniform(rng);
        let x = u / v;
        if 2.0 * v.ln() <= self.log_g_normalised(x) {
          return x;
        }
      },
      Regime::RatioOfUniformsShifted => loop {
        let u = self.u_minus + uniform(rng) * (self.u_plus - self.u_minus);
        let v = uniform(rng);
        let x = u / v + self.m;
        if x > 0.0 && 2.0 * v.ln() <= self.log_g_normalised(x) {
          return x;
        }
      },
    }
  }
}

/// Generalized inverse Gaussian distribution GIG$(\lambda, \chi, \psi)$ with
/// `chi > 0` and `psi > 0`.
pub struct SimdGig<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  lambda: T,
  chi: T,
  psi: T,
  setup: Setup,
  /// $\alpha = \sqrt{\psi/\chi}$, the scale taken out of the quasi-density.
  scale_out: f64,
  invert: bool,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  simd_rng: UnsafeCell<R>,
  stream_seed: Cell<u64>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdGig<T, R> {
  /// Construct a GIG$(\lambda, \chi, \psi)$.
  pub fn new<S: crate::simd_rng::SeedExt>(lambda: T, chi: T, psi: T, seed: &S) -> Self {
    let lambda_f = lambda.to_f64().unwrap();
    let chi_f = chi.to_f64().unwrap();
    let psi_f = psi.to_f64().unwrap();
    assert!(chi_f > 0.0, "GIG: chi must be positive");
    assert!(psi_f > 0.0, "GIG: psi must be positive");
    let beta = (chi_f * psi_f).sqrt();
    let setup = Setup::new(lambda_f.abs(), beta);
    let stream_seed = seed.seed_value();
    Self {
      lambda,
      chi,
      psi,
      setup,
      scale_out: (psi_f / chi_f).sqrt(),
      invert: lambda_f < 0.0,
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
      simd_rng: UnsafeCell::new(R::from_seed(stream_seed)),
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
      self.chi,
      self.psi,
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

  /// Fills `out` with GIG draws; the rejection loops are scalar on the
  /// internal SIMD uniform stream.
  pub fn fill_slice(&self, out: &mut [T]) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    for x in out.iter_mut() {
      let y = self.setup.draw::<T, R>(rng);
      let y = if self.invert { 1.0 / y } else { y };
      *x = T::from_f64_fast(y / self.scale_out);
    }
  }

  fn refill_buffer(&self) {
    let buf = unsafe { &mut *self.buffer.get() };
    self.fill_slice(buf);
    unsafe {
      *self.index.get() = 0;
    }
  }

  fn params(&self) -> (f64, f64, f64) {
    (
      self.lambda.to_f64().unwrap(),
      self.chi.to_f64().unwrap(),
      self.psi.to_f64().unwrap(),
    )
  }

  /// $\mathbb E X^k = (\chi/\psi)^{k/2} K_{\lambda+k}(b)/K_\lambda(b)$,
  /// $b = \sqrt{\chi\psi}$.
  pub fn raw_moment(&self, k: u32) -> f64 {
    let (lambda, chi, psi) = self.params();
    let b = (chi * psi).sqrt();
    (chi / psi).powf(0.5 * k as f64) * bessel_ke(lambda + k as f64, b) / bessel_ke(lambda, b)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdGig<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.lambda, self.chi, self.psi, &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Distribution<T> for SimdGig<T, R> {
  /// The `rng` argument is intentionally unused — this type draws from its
  /// own internal SIMD stream seeded at construction.
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

impl<T: SimdFloatExt, R: SimdRngExt> crate::traits::DistributionExt for SimdGig<T, R> {
  fn pdf(&self, x: f64) -> f64 {
    if x <= 0.0 {
      return 0.0;
    }
    let (lambda, chi, psi) = self.params();
    let b = (chi * psi).sqrt();
    let log_norm =
      0.5 * lambda * (psi / chi).ln() - std::f64::consts::LN_2 - (bessel_ke(lambda, b).ln() - b);
    (log_norm + (lambda - 1.0) * x.ln() - 0.5 * (chi / x + psi * x)).exp()
  }

  fn cdf(&self, _x: f64) -> f64 {
    unimplemented!("DistributionExt::cdf for SimdGig has no closed form")
  }

  fn inv_cdf(&self, _p: f64) -> f64 {
    unimplemented!("DistributionExt::inv_cdf for SimdGig has no closed form")
  }

  fn mean(&self) -> f64 {
    self.raw_moment(1)
  }

  /// $\bigl(\lambda - 1 + \sqrt{(\lambda-1)^2 + \chi\psi}\bigr)/\psi$.
  fn mode(&self) -> f64 {
    let (lambda, chi, psi) = self.params();
    (lambda - 1.0 + ((lambda - 1.0).powi(2) + chi * psi).sqrt()) / psi
  }

  fn variance(&self) -> f64 {
    let m1 = self.raw_moment(1);
    self.raw_moment(2) - m1 * m1
  }

  fn skewness(&self) -> f64 {
    let (m1, m2, m3) = (self.raw_moment(1), self.raw_moment(2), self.raw_moment(3));
    let var = m2 - m1 * m1;
    (m3 - 3.0 * m1 * m2 + 2.0 * m1.powi(3)) / var.powf(1.5)
  }

  /// Excess kurtosis.
  fn kurtosis(&self) -> f64 {
    let (m1, m2, m3, m4) = (
      self.raw_moment(1),
      self.raw_moment(2),
      self.raw_moment(3),
      self.raw_moment(4),
    );
    let var = m2 - m1 * m1;
    (m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1.powi(4)) / (var * var) - 3.0
  }

  /// $(\psi/(\psi - 2t))^{\lambda/2}\,K_\lambda(\sqrt{\chi(\psi - 2t)})/K_\lambda(\sqrt{\chi\psi})$
  /// for $t < \psi/2$, `NaN` beyond.
  fn moment_generating_function(&self, t: f64) -> f64 {
    let (lambda, chi, psi) = self.params();
    let shifted = psi - 2.0 * t;
    if shifted <= 0.0 {
      return f64::NAN;
    }
    let b = (chi * psi).sqrt();
    let bt = (chi * shifted).sqrt();
    (psi / shifted).powf(0.5 * lambda) * bessel_ke(lambda, bt) / bessel_ke(lambda, b)
      * (b - bt).exp()
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

  /// `scipy.stats.geninvgauss(p=λ, b=√(χψ), scale=√(χ/ψ))`: pdf on a grid,
  /// `stats('mvsk')` and the closed-form mode, across all three sampler
  /// regimes, λ = 0 and λ < 0.
  #[test]
  fn matches_scipy_geninvgauss() {
    let cases: [((f64, f64, f64), [f64; 5], [f64; 5]); 6] = [
      (
        (-0.5, 1.0, 4.0),
        [
          0.514_242_212_635_176_6,
          1.128_379_167_095_512_8,
          0.241_970_724_519_143_34,
          0.014_866_286_152_953_675,
          1.083_102_992_885_464e-5,
        ],
        [
          0.5,
          0.125_000_000_000_000_06,
          2.121_320_343_559_638_4,
          7.499_999_999_999_998,
          0.25,
        ],
      ),
      (
        (0.3, 2.0, 0.5),
        [
          0.000_207_154_166_797_105_37,
          0.181_109_435_668_869_9,
          0.267_440_855_009_698,
          0.211_388_022_266_015_88,
          0.070_972_459_002_713_6,
        ],
        [
          3.510_406_673_837_61,
          9.931_159_688_231_931,
          2.266_483_108_101_507_8,
          8.099_683_845_591_92,
          1.041_311_123_146_740_7,
        ],
      ),
      (
        (1.5, 0.2, 3.0),
        [
          0.253_779_589_649_395,
          0.693_107_485_732_857_7,
          0.511_710_317_868_987,
          0.169_750_934_196_500_7,
          0.003_072_457_210_418_208_3,
        ],
        [
          1.112_701_665_379_257_9,
          0.683_064_446_160_988_5,
          1.592_679_045_829_844,
          3.836_661_502_533_310_4,
          0.473_984_815_243_096_27,
        ],
      ),
      (
        (0.0, 1.0, 1.0),
        [
          0.076_115_931_334_513_58,
          0.680_494_457_892_701_8,
          0.436_886_089_924_687_64,
          0.170_123_614_473_175_45,
          0.017_641_156_063_218_862,
        ],
        [
          1.429_625_398_260_401_7,
          1.815_422_017_169_591_4,
          2.517_766_867_950_683,
          10.168_858_438_704_254,
          0.414_213_562_373_095_15,
        ],
      ),
      (
        (-2.0, 3.0, 0.3),
        [
          0.000_819_140_933_607_579_6,
          1.004_441_689_077_8,
          0.522_040_719_315_402_3,
          0.118_902_526_154_247_41,
          0.007_609_761_673_871_839,
        ],
        [
          1.130_335_190_237_882_4,
          1.186_774_422_790_675,
          4.513_096_168_856_083,
          41.651_500_096_330_146,
          0.488_088_481_701_516_3,
        ],
      ),
      (
        (0.2, 0.01, 1.0),
        [
          1.746_805_491_733_430_1,
          0.410_753_646_911_001,
          0.184_652_538_917_075_06,
          0.064_486_644_906_236_72,
          0.006_923_528_656_657_863,
        ],
        [
          0.639_899_966_429_868_3,
          1.136_287_952_394_739,
          3.603_221_141_680_864_4,
          19.812_668_188_757_524,
          0.006_225_774_829_854_98,
        ],
      ),
    ];
    for ((lambda, chi, psi), pdf, stats) in cases {
      let d = SimdGig::<f64>::new(lambda, chi, psi, &Unseeded);
      for (x, want) in [0.1, 0.5, 1.0, 2.0, 5.0].into_iter().zip(pdf) {
        assert!(
          close(d.pdf(x), want, 1e-11),
          "λ={lambda}: pdf({x}) = {}",
          d.pdf(x)
        );
      }
      assert!(
        close(d.mean(), stats[0], 1e-11),
        "λ={lambda}: mean {}",
        d.mean()
      );
      assert!(
        close(d.variance(), stats[1], 1e-10),
        "λ={lambda}: variance {}",
        d.variance()
      );
      assert!(
        close(d.skewness(), stats[2], 1e-9),
        "λ={lambda}: skewness {}",
        d.skewness()
      );
      assert!(
        close(d.kurtosis(), stats[3], 1e-8),
        "λ={lambda}: kurtosis {}",
        d.kurtosis()
      );
      assert!(
        close(d.mode(), stats[4], 1e-12),
        "λ={lambda}: mode {}",
        d.mode()
      );
      assert!(close(d.moment_generating_function(0.0), 1.0, 1e-12));
    }
  }

  /// Each generator regime reproduces the Bessel-ratio mean and variance.
  #[test]
  fn sample_moments_match_closed_forms_in_every_regime() {
    let cases = [
      (0.2, 0.01, 1.0, Regime::Hat),
      (0.3, 2.0, 0.5, Regime::RatioOfUniforms),
      (-2.0, 3.0, 0.3, Regime::RatioOfUniforms),
      (3.0, 4.0, 4.0, Regime::RatioOfUniformsShifted),
      (0.5, 1.0, 25.0, Regime::RatioOfUniformsShifted),
    ];
    for (lambda, chi, psi, regime) in cases {
      let d = SimdGig::<f64>::new(lambda, chi, psi, &Deterministic::new(11));
      assert_eq!(d.setup.regime, regime, "λ={lambda}");
      let n = 400_000;
      let mut xs = vec![0.0; n];
      d.fill_slice(&mut xs);
      assert!(xs.iter().all(|x| *x > 0.0));
      let mean = xs.iter().sum::<f64>() / n as f64;
      let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
      assert!(
        (mean - d.mean()).abs() / d.mean() < 0.01,
        "λ={lambda}: mean {mean} vs {}",
        d.mean()
      );
      assert!(
        (var - d.variance()).abs() / d.variance() < 0.05,
        "λ={lambda}: var {var} vs {}",
        d.variance()
      );
    }
  }

  #[test]
  fn pdf_integrates_to_one() {
    let d = SimdGig::<f64>::new(0.3, 2.0, 0.5, &Unseeded);
    let (hi, n) = (80.0_f64, 400_000usize);
    let h = hi / n as f64;
    let s: f64 = (0..n).map(|k| d.pdf((k as f64 + 0.5) * h) * h).sum();
    assert!((s - 1.0).abs() < 1e-6, "integral = {s}");
  }

  #[test]
  fn deterministic_seed_reproduces_stream() {
    let a = SimdGig::<f64>::new(0.3, 2.0, 0.5, &Deterministic::new(7));
    let b = SimdGig::<f64>::new(0.3, 2.0, 0.5, &Deterministic::new(7));
    for _ in 0..256 {
      assert_eq!(a.sample_fast(), b.sample_fast());
    }
  }

  #[test]
  #[should_panic(expected = "chi must be positive")]
  fn rejects_zero_chi() {
    let _ = SimdGig::<f64>::new(1.0, 0.0, 1.0, &Unseeded);
  }
}

py_distribution!(PyGig, SimdGig,
  sig: (lambda, chi, psi, seed=None, dtype=None),
  params: (lambda: f64, chi: f64, psi: f64)
);

//! Truncated exponential: inversion of the survival function referred to the lower bound.
//!
//! Reference: Johnson, N. L., Kotz, S. & Balakrishnan, N. (1994), *Continuous Univariate Distributions*, vol. 1, 2nd ed., Wiley, ch. 19, ISBN 0-471-58495-9 (the doubly truncated exponential's moments are its renormalised integrals).

use num_complex::Complex64;
use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use super::quadrature::Cut;
use super::quadrature::Kernel;
use super::quadrature::Truncation;
use crate::source::uniform53;
use crate::traits::DistributionExt;
use crate::traits::SimdFloatExt;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

/// `e^{−λx}` on `[0, ∞)`, whose integral gives the third and fourth moments.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ExpKernel {
  lambda: f64,
}

impl Kernel for ExpKernel {
  fn ln_k(&self, x: f64) -> f64 {
    if x < 0.0 {
      f64::NEG_INFINITY
    } else {
      -self.lambda * x
    }
  }

  fn anchor(&mut self, _c: Cut) {}

  fn ln_rel(&self, _x: f64, y: f64, _branch: Option<(f64, f64)>) -> f64 {
    -self.lambda * y
  }

  fn d1(&self, _x: f64, _y: f64) -> f64 {
    -self.lambda
  }

  fn d2(&self, _x: f64, _y: f64) -> f64 {
    0.0
  }

  fn mode(&self) -> Cut {
    Cut::at(0.0)
  }

  fn branch_points(&self) -> [Option<(f64, f64)>; 2] {
    [None, None]
  }

  fn length_scale(&self) -> f64 {
    1.0 / self.lambda
  }

  fn support(&self) -> (f64, f64) {
    (0.0, f64::INFINITY)
  }

  fn base_moments(&self) -> [f64; 4] {
    [1.0 / self.lambda, 1.0 / self.lambda, 2.0, 6.0]
  }
}

/// Truncated exponential law $\mathrm{Exp}(\lambda)$ on $[\text{lower}, \text{upper}]$, $\text{lower} \ge 0$:
/// parameters only; a [`Seeded`](crate::Seeded) stream draws it.
///
/// Devroye, L. (1986), *Non-Uniform Random Variate Generation*, Springer, §II.2, DOI 10.1007/978-1-4613-8643-8.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdTruncatedExp<T> {
  lambda: T,
  lower: T,
  upper: T,
  /// $1 − e^{-\lambda(\text{upper} − \text{lower})}$ through `exp_m1`, the interval's mass relative to `lower`; exactly 1 for an open tail.
  tail_mass: f64,
}

/// A truncated-exponential stream: the engine of its inversion uniforms.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct TruncatedExpState<R: SimdRngExt> {
  rng: R,
}

impl<T: SimdFloatExt> SimdTruncatedExp<T> {
  /// The base `Exp(lambda)`, `lambda > 0`, renormalised on `[lower, upper]`, `0 ≤ lower < upper` (`upper` may be infinite).
  pub fn new(lambda: T, lower: T, upper: T) -> Self {
    assert!(
      lambda.is_finite(),
      "lambda must satisfy `lambda.is_finite()`, got lambda = {lambda:?}"
    );
    assert!(
      !lower.is_nan(),
      "lower must satisfy `!lower.is_nan()`, got lower = {lower:?}"
    );
    assert!(
      !upper.is_nan(),
      "upper must satisfy `!upper.is_nan()`, got upper = {upper:?}"
    );
    assert!(
      lambda > T::zero(),
      "lambda must satisfy `lambda > T::zero()`, got lambda = {lambda:?}"
    );
    assert!(
      lower >= T::zero(),
      "lower must satisfy `lower >= T::zero()`, got lower = {lower:?}"
    );
    assert!(
      lower < upper,
      "lower must satisfy `lower < upper`, got lower = {lower:?}, upper = {upper:?}"
    );
    let lam = lambda.to_f64().unwrap();
    let lo = lower.to_f64().unwrap();
    let up = upper.to_f64().unwrap();
    Self {
      lambda,
      lower,
      upper,
      tail_mass: -(-lam * (up - lo)).exp_m1(),
    }
  }

  /// The rate `λ` of the untruncated base.
  pub fn lambda(&self) -> T {
    self.lambda
  }

  /// The lower bound.
  pub fn lower(&self) -> T {
    self.lower
  }

  /// The upper bound.
  pub fn upper(&self) -> T {
    self.upper
  }

  /// `lower − ln(1 − u·m)/λ` with `m` the interval's mass, the cdf's inverse referred to `lower`; a subnormal `u·m` takes
  /// its first order `u·m/λ`, which the logarithm of the rounded product would lose.
  #[inline]
  fn inverse_cdf(&self, u: f64) -> f64 {
    let y = u * self.tail_mass;
    if y < f64::MIN_POSITIVE {
      self.lower.to_f64().unwrap() + u * self.mass_per_rate()
    } else {
      self.log_map(y)
    }
  }

  /// `∫ e^{−λ(x − lower)}` over the interval, `tail_mass/λ`; where `λL` underflows the mass, `L·φ(λL)` with
  /// `φ(z) = (1 − e^{−z})/z`, so the law is its uniform limit, not 0/0.
  fn mass_per_rate(&self) -> f64 {
    let lam = self.lambda.to_f64().unwrap();
    if self.tail_mass >= f64::MIN_POSITIVE {
      self.tail_mass / lam
    } else {
      let len = self.length().unwrap();
      len * one_minus_exp_over(lam * len)
    }
  }

  #[inline]
  fn log_map(&self, y: f64) -> f64 {
    self.lower.to_f64().unwrap() - (-y).ln_1p() / self.lambda.to_f64().unwrap()
  }

  /// A draw's uniform is 0 or at least `2⁻⁵³`, so its `u·m` is normal unless the mass is below `2⁵³·MIN_POSITIVE`.
  #[inline]
  fn invert(&self, u: f64) -> T {
    T::from_f64_fast(if self.tail_mass >= f64::MIN_POSITIVE * 2f64.powi(53) {
      self.log_map(u * self.tail_mass)
    } else {
      self.inverse_cdf(u)
    })
  }

  /// The shape depends on `λL` alone: in units of the shorter of `1/λ` and `L`, from the lower bound, neither the
  /// rate nor the length under- or overflows.
  fn moments(&self) -> Option<[f64; 4]> {
    let x = self.lambda.to_f64()? * self.length()?;
    let kernel = ExpKernel { lambda: x.min(1.0) };
    Some(Truncation::mass_only(kernel, 0.0, x.max(1.0)).moments(1.0))
  }

  /// Infinite for an open tail.
  fn length(&self) -> Option<f64> {
    Some(self.upper.to_f64()? - self.lower.to_f64()?)
  }

  /// An infinite `upper`, or `e^{−λL}` below the subnormals: there the bounded moments meet `∞·0` and equal the open
  /// tail's to the last bit.
  fn open_tail(&self) -> Option<bool> {
    let len = self.length()?;
    Some(len.is_infinite() || (-self.lambda.to_f64()? * len).exp() == 0.0)
  }
}

impl<T: SimdFloatExt> Sealed for SimdTruncatedExp<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdTruncatedExp<T> {
  type State<R: SimdRngExt> = TruncatedExpState<R>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (TruncatedExpState<R>, u64) {
    let stream_seed = seed.next_seed();
    (
      TruncatedExpState {
        rng: R::from_seed(stream_seed),
      },
      stream_seed,
    )
  }
}

impl<T: SimdFloatExt> SimdKernel for SimdTruncatedExp<T> {
  type Item = T;

  fn fill<R: SimdRngExt>(&self, state: &mut TruncatedExpState<R>, out: &mut [T]) {
    for x in out.iter_mut() {
      *x = self.invert(state.rng.next_f64());
    }
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut TruncatedExpState<R>) -> T {
    self.invert(state.rng.next_f64())
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdTruncatedExp<T> {
  /// The inversion of one 53-bit uniform from the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.invert(uniform53(rng.next_u64()))
  }
}

impl<T: SimdFloatExt> DistributionExt for SimdTruncatedExp<T> {
  /// `e^{it·lower} (λL/m) φ₁((λ − it)L)`, `φ₁(w) = (1 − e^{−w})/w`, `m` the mass; an open tail `e^{it·lower}/(1 − it/λ)`
  /// divided scaled, so no `λ²` or `t²` leaves the range. NaN where `t·lower` overflows: the phase keeps no digit.
  fn characteristic_function(&self, t: f64) -> Option<Complex64> {
    if t == 0.0 {
      return Some(Complex64::new(1.0, 0.0));
    }
    let (lam, lo, len) = (self.lambda.to_f64()?, self.lower.to_f64()?, self.length()?);
    let phase = Complex64::new(0.0, t * lo).exp();
    if self.open_tail()? {
      return Some(phase.fdiv(Complex64::new(1.0, -t / lam)));
    }
    Some(phase * (len / self.mass_per_rate()) * phi1(Complex64::new(lam, -t) * len))
  }

  fn pdf(&self, x: f64) -> Option<f64> {
    let lo = self.lower.to_f64().unwrap();
    let up = self.upper.to_f64().unwrap();
    if x < lo || x > up {
      return Some(0.0);
    }
    let lam = self.lambda.to_f64().unwrap();
    // Referred to `lower` for the same reason the draw is; the leading
    // `e^{-λ·lower}` cancels between the density and the interval's mass.
    Some((-lam * (x - lo)).exp() / self.mass_per_rate())
  }

  fn cdf(&self, x: f64) -> Option<f64> {
    let lo = self.lower.to_f64().unwrap();
    let up = self.upper.to_f64().unwrap();
    if x <= lo {
      return Some(0.0);
    }
    if x >= up {
      return Some(1.0);
    }
    let (lam, d) = (self.lambda.to_f64().unwrap(), x - lo);
    if self.tail_mass < f64::MIN_POSITIVE {
      return Some(d * one_minus_exp_over(lam * d) / self.mass_per_rate());
    }
    Some(-(-lam * d).exp_m1() / self.tail_mass)
  }

  fn quantile(&self, p: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&p) {
      return Some(f64::NAN);
    }
    // The bounds exactly, which the inverse map misses by a few ulps.
    if p == 0.0 {
      return self.lower.to_f64();
    }
    if p == 1.0 {
      return self.upper.to_f64();
    }
    if p <= 0.5 {
      return Some(self.inverse_cdf(p));
    }
    let (lo, lam) = (self.lower.to_f64()?, self.lambda.to_f64()?);
    let lam_len = lam * self.length()?;
    let x = if lam_len <= 1.0 {
      self.inverse_cdf(p)
    } else {
      // `1 − p·m` as `(1 − p) + p·e^{−λL}`: above ½ `1 − p` is exact and both terms positive, where the map cancels.
      lo - ((1.0 - p) + p * (-lam_len).exp()).ln() / lam
    };
    // Next to level 1 the roundings of `lower + y` and of `λL` can pass `upper`, which bounds the support exactly.
    Some(x.min(self.upper.to_f64()?))
  }

  fn mean(&self) -> Option<f64> {
    let (lam, lo) = (self.lambda.to_f64()?, self.lower.to_f64()?);
    if self.open_tail()? {
      return Some(lo + 1.0 / lam);
    }
    let len = self.length()?;
    Some(lo + len * mean_fraction(lam * len))
  }

  fn median(&self) -> Option<f64> {
    self.quantile(0.5)
  }

  fn mode(&self) -> Option<f64> {
    self.lower.to_f64()
  }

  fn variance(&self) -> Option<f64> {
    let lam = self.lambda.to_f64()?;
    if self.open_tail()? {
      return Some(1.0 / (lam * lam));
    }
    let len = self.length()?;
    Some(len * len * variance_fraction(lam * len))
  }

  fn skewness(&self) -> Option<f64> {
    Some(self.moments()?[2])
  }

  fn kurtosis(&self) -> Option<f64> {
    Some(self.moments()?[3])
  }

  fn entropy(&self) -> Option<f64> {
    let lam = self.lambda.to_f64()?;
    if self.open_tail()? {
      return Some(1.0 - lam.ln());
    }
    let x = lam * self.length()?;
    Some((-(-x).exp_m1() / lam).ln() + x * mean_fraction(x))
  }

  /// Finite for every `t` on a bounded interval (`λL e^{λ·lower}/m` at `t = λ`, `m` the mass); an open tail diverges from `t = λ` on.
  fn moment_generating_function(&self, t: f64) -> Option<f64> {
    if t.is_nan() {
      return Some(f64::NAN);
    }
    if t == 0.0 {
      return Some(1.0);
    }
    let (lam, lo, len) = (self.lambda.to_f64()?, self.lower.to_f64()?, self.length()?);
    if len.is_infinite() {
      return Some(if t < lam {
        (t * lo).exp() * lam / (lam - t)
      } else {
        f64::INFINITY
      });
    }
    let s = t - lam;
    let integral = if s == 0.0 {
      len
    } else {
      (s * len).exp_m1() / s
    };
    Some((t * lo).exp() * integral / self.mass_per_rate())
  }
}

/// `(1 − e^{−z})/z`, 1 where `z` is below the digits `e^{−z}` keeps.
fn one_minus_exp_over(z: f64) -> f64 {
  if z < 1e-8 {
    1.0 - 0.5 * z
  } else {
    -(-z).exp_m1() / z
  }
}

/// `(1 − e^{−w})/w`, by its series `Σ (−w)^n/(n+1)!` for `|w| < ½`, where the quotient's imaginary part `−Im(w)/2`
/// sits below the real part's rounding; the quotient divides scaled.
fn phi1(w: Complex64) -> Complex64 {
  if w.norm() < 0.5 {
    // 18 terms: the next is below 0.5^18/19! ≈ 3e-23.
    return (1..=18).rev().fold(Complex64::new(0.0, 0.0), |acc, n| {
      (acc * -w + 1.0) / n as f64
    });
  }
  (-exp_m1_complex(-w)).fdiv(w)
}

/// `e^{u+iv} − 1` as `(expm1(u) cos v − 2 sin²(v/2), e^u sin v)`: no `e^w − 1` subtraction.
fn exp_m1_complex(w: Complex64) -> Complex64 {
  let (u, v) = (w.re, w.im);
  Complex64::new(
    u.exp_m1() * v.cos() - 2.0 * (0.5 * v).sin().powi(2),
    u.exp() * v.sin(),
  )
}

/// The $x^{2k-1}$ coefficients $-B_{2k}/(2k)!$, $k = 1..=7$, of `mean_fraction` past its constant ½, rounded to `f64`.
const MEAN_SERIES: [f64; 7] = [
  -0.08333333333333333,
  0.001388888888888889,
  -3.306878306878307e-05,
  8.267195767195768e-07,
  -2.08767569878681e-08,
  5.284190138687493e-10,
  -1.3382536530684679e-11,
];

/// The $x^{2k-2}$ coefficients $(2k-1)B_{2k}/(2k)!$, $k = 1..=12$, of `variance_fraction`, rounded to `f64`.
const VARIANCE_SERIES: [f64; 12] = [
  0.08333333333333333,
  -0.004166666666666667,
  0.00016534391534391533,
  -5.787037037037037e-06,
  1.8789081289081288e-07,
  -5.812609152556243e-09,
  1.7397297489890083e-10,
  -5.084520444483875e-12,
  1.4596305495672335e-13,
  -4.1322505272603176e-15,
  1.1568905939556482e-16,
  -3.2095268777368804e-18,
];

/// $1/x - 1/(e^x - 1)$ at $x = \lambda L$: the mean's offset from `lower` over `L`; below `x = 0.5` the two terms
/// cancel, so it takes the Bernoulli series there.
fn mean_fraction(x: f64) -> f64 {
  if x < 0.5 {
    let x2 = x * x;
    0.5 + x * MEAN_SERIES.iter().rev().fold(0.0, |acc, &c| acc * x2 + c)
  } else {
    1.0 / x - 1.0 / x.exp_m1()
  }
}

/// $1/x^2 - 1/(4\sinh^2(x/2))$ at $x = \lambda L$: the variance over $L^2$; the series below `x = 1`, where the
/// difference amplifies `sinh`'s rounding past 12-fold.
fn variance_fraction(x: f64) -> f64 {
  if x < 1.0 {
    let x2 = x * x;
    VARIANCE_SERIES
      .iter()
      .rev()
      .fold(0.0, |acc, &c| acc * x2 + c)
  } else {
    let s = (0.5 * x).sinh();
    1.0 / (x * x) - 1.0 / (4.0 * s * s)
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;
  use crate::tests::scalar_draws;
  use crate::tests::scalar_ks_best_p;

  /// Truncated exponential: closed-form CDF must round-trip to inputs.
  #[test]
  fn truncated_exp_cdf_round_trips() {
    let te = SimdTruncatedExp::<f64>::new(2.0, 0.0, 1.5);
    for x in [0.0, 0.3, 0.7, 1.0, 1.5] {
      let f = te.cdf(x).unwrap();
      assert!((0.0..=1.0).contains(&f));
    }
    assert_eq!(te.cdf(-0.1).unwrap(), 0.0);
    assert_eq!(te.cdf(2.0).unwrap(), 1.0);
  }

  /// Draws in bounds with the mean `(1 − 1.5e^{−½})/(1 − e^{−½})` of Exp(1) on `[0, ½]`, best of three pinned seeds.
  #[test]
  fn truncated_exp_samples_mean() {
    let expected = (1.0 - 1.5 * (-0.5_f64).exp()) / (1.0 - (-0.5_f64).exp());
    let best = [2718, 999, 42]
      .into_iter()
      .map(|seed| {
        let mut te = SimdTruncatedExp::<f64>::new(1.0, 0.0, 0.5).seeded(&Deterministic::new(seed));
        let xs = (0..20_000).map(|_| te.sample()).collect::<Vec<_>>();
        assert!(xs.iter().all(|x| (0.0..=0.5).contains(x)), "seed {seed}");
        (xs.iter().sum::<f64>() / xs.len() as f64 - expected).abs()
      })
      .fold(f64::INFINITY, f64::min);
    assert!(best < 0.01, "best |mean − {expected}| = {best}");
  }

  /// Draws in bounds with mean `lower + 1/λ − (upper − lower)·r/(1 − r)`, `r = e^{−λ(upper−lower)}`, on intervals where
  /// the cdf form saturated and put every draw at 690.78, the old `-ln(1e-300)` guard.
  fn assert_far_interval_means(mut draws: impl FnMut(SimdTruncatedExp<f64>) -> Vec<f64>) {
    for (lam, lo, hi) in [(1.0_f64, 50.0_f64, 60.0_f64), (100.0, 0.5, 0.6)] {
      let xs = draws(SimdTruncatedExp::<f64>::new(lam, lo, hi));
      for x in &xs {
        assert!((lo..=hi).contains(x), "sample {x} out of [{lo}, {hi}]");
      }
      let r = (-lam * (hi - lo)).exp();
      let want = lo + 1.0 / lam - (hi - lo) * r / (1.0 - r);
      let mean = xs.iter().sum::<f64>() / xs.len() as f64;
      assert!(
        (mean - want).abs() < 5.0 / lam / (xs.len() as f64).sqrt(),
        "lambda = {lam} on [{lo}, {hi}]: mean = {mean}, expected ≈ {want}"
      );
    }
  }

  /// A far interval keeps its truncated-exponential draws inside its bounds.
  #[test]
  fn truncated_exp_far_interval_stays_in_bounds() {
    assert_far_interval_means(|d| {
      let mut s = d.seeded(&Deterministic::new(2718));
      (0..40_000).map(|_| s.sample()).collect()
    });
  }

  /// The honest draw keeps the same far-interval bounds and means on the caller's rng.
  #[test]
  fn scalar_far_interval_stays_in_bounds() {
    assert_far_interval_means(|d| scalar_draws(&d, 2718, 40_000));
  }

  /// The honest `Distribution` draws from the caller's rng and agrees with the cdf.
  #[test]
  fn scalar_sample_matches_cdf() {
    let d = SimdTruncatedExp::<f64>::new(2.0, 0.0, 1.5);
    let best = scalar_ks_best_p(&d, |x| d.cdf(x).unwrap());
    assert!(best > 0.01, "best p = {best}");
  }

  /// `1 − e^{−λL}` through `exp_m1` and `ln(1 − u·m)` through `ln_1p`, where the plain forms read `∞`, NaN and `lower`
  /// at λL = 1e-17; mpmath references.
  #[test]
  fn tiny_rate_times_length_keeps_pdf_cdf_quantile() {
    let rel = |a: f64, b: f64| ((a - b) / b).abs();
    let flat = SimdTruncatedExp::<f64>::new(1e-17, 0.0, 1.0);
    assert!(rel(flat.pdf(0.5).unwrap(), 1.0) < 1e-12);
    assert!(rel(flat.cdf(0.5).unwrap(), 0.5) < 1e-12);
    assert!(rel(flat.quantile(0.25).unwrap(), 0.25) < 1e-12);
    let short = SimdTruncatedExp::<f64>::new(1.0, 0.0, 1e-17);
    assert!(rel(short.pdf(5e-18).unwrap(), 1e17) < 1e-12);
    assert!(rel(short.quantile(0.25).unwrap(), 2.5e-18) < 1e-12);
    let nano = SimdTruncatedExp::<f64>::new(1.0, 0.0, 1e-9);
    assert!(rel(nano.quantile(0.25).unwrap(), 2.499_999_999_062_5e-10) < 1e-12);
    assert!(rel(nano.cdf(5e-10).unwrap(), 0.500_000_000_125) < 1e-12);
    assert!(rel(nano.pdf(5e-10).unwrap(), 999_999_999.999_999_9) < 1e-12);
  }

  /// The engine's mean and variance, in its units of the shorter of `1/λ` and `L` from the lower bound, against the
  /// series forms over every regime of `λL`, at `λ = 1.5`.
  #[test]
  fn engine_agrees_with_the_series_forms() {
    for lam_len in [1e-17, 1e-9, 0.3, 1.0, 5.0, 50.0, 1050.0] {
      let lo = if lam_len < 1e-8 { 0.0 } else { 0.5 };
      let law = SimdTruncatedExp::<f64>::new(1.5, lo, lo + lam_len / 1.5);
      let [mean, variance, _, _] = law.moments().unwrap();
      let len = law.length().unwrap();
      let unit = len / (1.5 * len).max(1.0);
      let rel = |a: f64, b: f64| ((a - b) / b).abs();
      assert!(
        rel(lo + unit * mean, law.mean().unwrap()) < 1e-13,
        "λL = {lam_len}: mean {mean}"
      );
      assert!(
        rel(unit * unit * variance, law.variance().unwrap()) < 1e-13,
        "λL = {lam_len}: variance {variance}"
      );
    }
  }
}

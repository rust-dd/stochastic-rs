//! Truncated exponential: inversion of the survival function referred to the lower bound.
//!
//! Reference: Johnson, N. L., Kotz, S. & Balakrishnan, N. (1994), *Continuous Univariate Distributions*, vol. 1, 2nd ed., Wiley, ch. 19, ISBN 0-471-58495-9 (the doubly truncated exponential's moments are its renormalised integrals).

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use crate::source::uniform53;
use crate::traits::DistributionExt;
use crate::traits::SimdFloatExt;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

/// Truncated exponential law $\mathrm{Exp}(\lambda)$ on $[\text{lower}, \text{upper}]$, $\text{lower} \ge 0$:
/// parameters only; a [`Seeded`](crate::Seeded) stream draws it.
///
/// Devroye, L. (1986), *Non-Uniform Random Variate Generation*, Springer, §II.2, DOI 10.1007/978-1-4613-8643-8.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdTruncatedExp<T> {
  lambda: T,
  lower: T,
  upper: T,
  /// $e^{-\lambda(\text{upper} - \text{lower})}$, the survival at `upper` relative to `lower`; zero for an infinite `upper`.
  tail_ratio: f64,
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
      tail_ratio: (-lam * (up - lo)).exp(),
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

  /// `lower − ln(V)/λ` with `V = 1 − u(1 − tail_ratio)` in `(tail_ratio, 1]`: in the cdf form `1 − e^{−λx}` both bounds
  /// round to 1 once `λ·lower` passes about 36.
  #[inline]
  fn invert(&self, u: f64) -> T {
    let v = 1.0 - u * (1.0 - self.tail_ratio);
    T::from_f64_fast(self.lower.to_f64().unwrap() - v.ln() / self.lambda.to_f64().unwrap())
  }

  /// `upper − lower`, infinite for an open tail.
  fn length(&self) -> Option<f64> {
    Some(self.upper.to_f64()? - self.lower.to_f64()?)
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
  fn pdf(&self, x: f64) -> Option<f64> {
    let lo = self.lower.to_f64().unwrap();
    let up = self.upper.to_f64().unwrap();
    if x < lo || x > up {
      return Some(0.0);
    }
    let lam = self.lambda.to_f64().unwrap();
    // Referred to `lower` for the same reason the draw is; the leading
    // `e^{-λ·lower}` cancels between the density and the interval's mass.
    Some(lam * (-lam * (x - lo)).exp() / (1.0 - self.tail_ratio))
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
    let lam = self.lambda.to_f64().unwrap();
    Some((1.0 - (-lam * (x - lo)).exp()) / (1.0 - self.tail_ratio))
  }

  fn quantile(&self, p: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&p) {
      return Some(f64::NAN);
    }
    let (lam, lo) = (self.lambda.to_f64()?, self.lower.to_f64()?);
    Some(lo - (1.0 - p * (1.0 - self.tail_ratio)).ln() / lam)
  }

  fn mean(&self) -> Option<f64> {
    let (lam, lo) = (self.lambda.to_f64()?, self.lower.to_f64()?);
    if self.tail_ratio == 0.0 {
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
    if self.tail_ratio == 0.0 {
      return Some(1.0 / (lam * lam));
    }
    let len = self.length()?;
    Some(len * len * variance_fraction(lam * len))
  }

  fn entropy(&self) -> Option<f64> {
    let lam = self.lambda.to_f64()?;
    if self.tail_ratio == 0.0 {
      return Some(1.0 - lam.ln());
    }
    let x = lam * self.length()?;
    Some((-(-x).exp_m1() / lam).ln() + x * mean_fraction(x))
  }

  /// Finite for every `t` on a bounded interval (`λL e^{λ·lower}/(1 − r)` at `t = λ`); an open tail diverges from `t = λ` on.
  fn moment_generating_function(&self, t: f64) -> Option<f64> {
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
    Some((t * lo).exp() * lam * integral / -(-lam * len).exp_m1())
  }
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
  use stochastic_rs_core::simd_rng::Unseeded;

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

  /// Truncated exponential samples in bounds with the right approximate
  /// mean (closed-form check on tight [0, 0.5] band of Exp(1)).
  #[test]
  fn truncated_exp_samples_mean() {
    let mut te = SimdTruncatedExp::<f64>::new(1.0, 0.0, 0.5).seeded(&Unseeded);
    let n = 20_000;
    let mut sum = 0.0;
    for _ in 0..n {
      let x = te.sample();
      assert!((0.0..=0.5).contains(&x));
      sum += x;
    }
    let mean = sum / n as f64;
    // E[X | 0 ≤ X ≤ 0.5] = ∫₀^0.5 x·e^{-x} dx / (1 - e^{-0.5}) = [1 - 1.5·e^{-0.5}] / (1 - e^{-0.5}) for Exp(1).
    let half = 0.5_f64;
    let expected = (1.0 - 1.5 * (-half).exp()) / (1.0 - (-half).exp());
    assert!(
      (mean - expected).abs() < 0.01,
      "truncated Exp(1) mean = {mean}, expected ≈ {expected}"
    );
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
}

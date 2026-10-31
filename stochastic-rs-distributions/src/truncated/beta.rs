//! Truncated beta: rejection from the base law while the interval holds 1e-3 of its mass, else the inverse cdf.
//!
//! Reference: Johnson, N. L., Kotz, S. & Balakrishnan, N. (1995), *Continuous Univariate Distributions*, vol. 2, 2nd ed., Wiley, ch. 25, ISBN 0-471-58494-0 (beta moments; the truncated ones renormalise the incomplete beta ratio).
//! Reference: Devroye, L. (1986), *Non-Uniform Random Variate Generation*, Springer, §II.2 and §II.3, DOI 10.1007/978-1-4613-8643-8 (inversion and rejection).

use std::fmt;
use std::sync::OnceLock;

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use super::quadrature::Cut;
use super::quadrature::Kernel;
use super::quadrature::Truncation;
use super::quadrature::ln_1p_mx;
use crate::beta::BetaState;
use crate::beta::SimdBeta;
use crate::source::AnyRng;
use crate::source::uniform53;
use crate::traits::DistributionExt;
use crate::traits::SimdFloatExt;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

/// `x^{a−1} (1 − x)^{b−1}` on `[0, 1]` about the reference point `c`; a unit shape has no log term, whose `0·ln 0`
/// would read NaN at its end.
#[derive(Clone, Copy, Debug, PartialEq)]
struct BetaKernel {
  a: f64,
  b: f64,
  c: f64,
  /// `1 − c`, exact where `c` is held beside 1.
  cc: f64,
  ln_c: f64,
  ln_cc: f64,
  /// `(a − 1)/c − (b − 1)/(1 − c)`, the slope at `c`, `None` where it overflows; zero at the mode, whose rounding moves
  /// the law by under an ulp.
  slope: Option<f64>,
}

impl BetaKernel {
  fn new(a: f64, b: f64) -> Self {
    Self {
      a,
      b,
      c: 0.5,
      cc: 0.5,
      ln_c: -std::f64::consts::LN_2,
      ln_cc: -std::f64::consts::LN_2,
      slope: None,
    }
  }

  /// `(a − 1)(ln(x/c) − y/c)`: `ln_1p_mx` beside `c`, read from `x` toward 0.
  fn left(&self, x: f64, y: f64) -> f64 {
    let z = y / self.c;
    if self.a == 1.0 {
      0.0
    } else if z < -0.5 {
      (self.a - 1.0) * ((x.ln() - self.ln_c) - z)
    } else {
      (self.a - 1.0) * ln_1p_mx(z)
    }
  }

  /// `(b − 1)(ln((1 − x)/(1 − c)) + y/(1 − c))`, `1 − x` toward 1 as the offset from 1 where `branch` holds it, else as
  /// `(1 − c) − y`: never from a rounded `x`.
  fn right(&self, y: f64, branch: Option<(f64, f64)>) -> f64 {
    let z = -y / self.cc;
    if self.b == 1.0 {
      0.0
    } else if z < -0.5 {
      let gap = match branch {
        Some((1.0, t)) => -t,
        _ => self.cc - y,
      };
      (self.b - 1.0) * ((gap.ln() - self.ln_cc) - z)
    } else {
      (self.b - 1.0) * ln_1p_mx(z)
    }
  }

  /// `(a − 1)y/c − (b − 1)y/(1 − c)`, from the slope where it is finite; where it overflows the mode is far and the two
  /// terms do not cancel.
  fn linear(&self, y: f64) -> f64 {
    match self.slope {
      Some(slope) => slope * y,
      None => self.per(self.a, y, self.c) - self.per(self.b, y, self.cc),
    }
  }

  /// `(e − 1)·y/d`, 0 for a unit exponent, whose end `d` may be 0.
  fn per(&self, e: f64, y: f64, d: f64) -> f64 {
    if e == 1.0 { 0.0 } else { (e - 1.0) * (y / d) }
  }
}

impl Kernel for BetaKernel {
  fn ln_k(&self, x: f64) -> f64 {
    if !(0.0..=1.0).contains(&x) {
      return f64::NEG_INFINITY;
    }
    let left = if self.a == 1.0 {
      0.0
    } else {
      (self.a - 1.0) * x.ln()
    };
    let right = if self.b == 1.0 {
      0.0
    } else {
      (self.b - 1.0) * (-x).ln_1p()
    };
    left + right
  }

  /// The slope from `(a − 1)(1 − c) − (b − 1)c` by an exact product, so a clamped end next to the mode keeps it; 0 at a
  /// mode whose rounding moves it past the law's width `|d2|^{−1/2}`, where the law lies within its ulp.
  fn anchor(&mut self, c: Cut) {
    self.c = c.x;
    self.cc = -c.offset(1.0);
    self.ln_c = self.c.ln();
    self.ln_cc = self.cc.ln();
    let (a1, b1) = (self.a - 1.0, self.b - 1.0);
    let slope = if a1 == 0.0 || b1 == 0.0 {
      self.per(self.a, 1.0, self.c) - self.per(self.b, 1.0, self.cc)
    } else {
      let p = b1 * self.c;
      let e = b1.mul_add(self.c, -p);
      (a1.mul_add(self.cc, -p) - e) / self.c / self.cc
    };
    let width = 1.0 / (-self.d2(self.c, 0.0)).sqrt();
    let slope = if c == self.mode() && a1 > 0.0 && b1 > 0.0 && slope.abs() * width > 1.0 {
      0.0
    } else {
      slope
    };
    self.slope = slope.is_finite().then_some(slope);
  }

  fn ln_rel(&self, x: f64, y: f64, branch: Option<(f64, f64)>) -> f64 {
    if y == 0.0 {
      return 0.0;
    }
    self.left(x, y) + self.right(y, branch) + self.linear(y)
  }

  fn ln_rel_regular(&self, x: f64, y: f64, s: f64) -> f64 {
    let (a1, b1) = (self.a - 1.0, self.b - 1.0);
    if s == 0.0 {
      -a1 * self.ln_c + self.right(y, None) - b1 * (y / self.cc)
    } else {
      self.left(x, y) + a1 * (y / self.c) - b1 * self.ln_cc
    }
  }

  fn d1(&self, x: f64, y: f64) -> f64 {
    match self.slope {
      Some(slope) => {
        slope - self.per(self.a, y, self.c) / x - self.per(self.b, y, self.cc) / (self.cc - y)
      }
      None => self.per(self.a, 1.0, x) - self.per(self.b, 1.0, self.cc - y),
    }
  }

  fn d2(&self, x: f64, y: f64) -> f64 {
    let gap = self.cc - y;
    -self.per(self.a, 1.0, x * x) - self.per(self.b, 1.0, gap * gap)
  }

  /// The interior mode when both shapes exceed 1, by halves so `a + b − 2` cannot overflow, beside 1 where it rounds
  /// onto it; else the end the density rises toward, the heavier of both (0 on a tie), and ½ for the uniform.
  fn mode(&self) -> Cut {
    let (a, b) = (self.a, self.b);
    if a > 1.0 && b > 1.0 {
      let (ha, hb) = (0.5 * (a - 1.0), 0.5 * (b - 1.0));
      let (m, gap) = (ha / (ha + hb), hb / (ha + hb));
      let floor = 0.1 * self.length_scale();
      if m <= 0.5 {
        Cut::at(m.max(floor))
      } else if 1.0 - gap < 1.0 {
        Cut::at(1.0 - gap)
      } else {
        Cut::beside(1.0, -gap.max(floor))
      }
    } else if a == 1.0 && b == 1.0 {
      Cut::at(0.5)
    } else if a < 1.0 && (1.0 <= b || a <= b) || (a == 1.0 && b > 1.0) {
      Cut::at(0.0)
    } else {
      Cut::at(1.0)
    }
  }

  fn branch_points(&self) -> [Option<(f64, f64)>; 2] {
    [
      (self.a != 1.0).then_some((0.0, self.a)),
      (self.b != 1.0).then_some((1.0, self.b)),
    ]
  }

  /// `1/(α + β)` from halves, so it never overflows to 0: beside 0 the factor `(1 − x)^{β−1}` falls an e-fold in
  /// about `1/β`, beside 1 `x^{α−1}` in `1/α`.
  fn length_scale(&self) -> f64 {
    0.5 / (0.5 * self.a + 0.5 * self.b)
  }

  fn support(&self) -> (f64, f64) {
    (0.0, 1.0)
  }

  /// The closed forms in `r = (a − b)/√(ab)` and `1/(a + b)`, so no product of the shapes over- or underflows.
  fn base_moments(&self) -> [f64; 4] {
    let (a, b) = (self.a, self.b);
    let root = a.sqrt() * b.sqrt();
    let s = a + b;
    if s < 2.0 {
      let r = (a - b) / root;
      return [
        a / s,
        root / s / (s + 1.0).sqrt(),
        -2.0 * r * (s + 1.0).sqrt() / (s + 2.0),
        6.0 * (r * r * (s + 1.0) / ((s + 2.0) * (s + 3.0)) - 1.0 / (s + 3.0)),
      ];
    }
    let inv = self.length_scale();
    let f = inv * (1.0 + inv) / ((1.0 + 2.0 * inv) * (1.0 + 3.0 * inv));
    let rf = (a - b) * f.sqrt() / root;
    [
      a * inv,
      root * inv * (inv / (1.0 + inv)).sqrt(),
      -2.0 * ((a - b) * (inv * (1.0 + inv)).sqrt() / root) / (1.0 + 2.0 * inv),
      6.0 * (rf * rf - inv / (1.0 + 3.0 * inv)),
    ]
  }
}

/// Truncated beta law on $[\text{lower}, \text{upper}] \subseteq [0, 1]$: parameters and the base mass on the
/// interval; a [`Seeded`](crate::Seeded) stream draws it.
#[derive(Clone)]
pub struct SimdTruncatedBeta<T> {
  base: SimdBeta<T>,
  lower: T,
  upper: T,
  /// The base law's mass on the interval: the sampler's regime switch.
  mass: f64,
  /// The interval's kernel integral, built by the first analytic call or inverse-cdf draw.
  analytics: OnceLock<Truncation<BetaKernel>>,
}

impl<T: fmt::Debug> fmt::Debug for SimdTruncatedBeta<T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("SimdTruncatedBeta")
      .field("base", &self.base)
      .field("lower", &self.lower)
      .field("upper", &self.upper)
      .finish_non_exhaustive()
  }
}

/// The parameters decide every cached value, so they alone are compared.
impl<T: PartialEq> PartialEq for SimdTruncatedBeta<T> {
  fn eq(&self, other: &Self) -> bool {
    self.base == other.base && self.lower == other.lower && self.upper == other.upper
  }
}

/// The base draw and the `[0, 1)` uniform a truncated-beta draw consumes, from a stream or the caller's rng.
trait TruncatedBetaSource<T: SimdFloatExt> {
  fn base(&mut self, law: &SimdBeta<T>) -> T;

  fn uniform(&mut self) -> f64;
}

impl<T: SimdFloatExt, R: SimdRngExt> TruncatedBetaSource<T> for BetaState<T, R> {
  #[inline]
  fn base(&mut self, law: &SimdBeta<T>) -> T {
    law.next(self)
  }

  #[inline]
  fn uniform(&mut self) -> f64 {
    self.g1.rng.next_f64()
  }
}

impl<T: SimdFloatExt, G: Rng + ?Sized> TruncatedBetaSource<T> for AnyRng<'_, G> {
  #[inline]
  fn base(&mut self, law: &SimdBeta<T>) -> T {
    law.draw_with(self.0)
  }

  #[inline]
  fn uniform(&mut self) -> f64 {
    uniform53(self.0.next_u64())
  }
}

impl<T: SimdFloatExt> SimdTruncatedBeta<T> {
  /// The base [`SimdBeta`] `Beta(alpha, beta)`, both > 0, renormalised on `[lower, upper] ⊆ [0, 1]`, `lower < upper`.
  pub fn new(alpha: T, beta: T, lower: T, upper: T) -> Self {
    assert!(
      alpha.is_finite(),
      "alpha must satisfy `alpha.is_finite()`, got alpha = {alpha:?}"
    );
    assert!(
      beta.is_finite(),
      "beta must satisfy `beta.is_finite()`, got beta = {beta:?}"
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
      alpha > T::zero(),
      "alpha must satisfy `alpha > T::zero()`, got alpha = {alpha:?}"
    );
    assert!(
      beta > T::zero(),
      "beta must satisfy `beta > T::zero()`, got beta = {beta:?}"
    );
    let lo = lower.to_f64().unwrap();
    let up = upper.to_f64().unwrap();
    assert!(
      (0.0..=1.0).contains(&lo),
      "lower must satisfy `0.0 <= lower <= 1.0`, got lower = {lower:?}"
    );
    assert!(
      (0.0..=1.0).contains(&up),
      "upper must satisfy `0.0 <= upper <= 1.0`, got upper = {upper:?}"
    );
    assert!(
      lower < upper,
      "lower must satisfy `lower < upper`, got lower = {lower:?}, upper = {upper:?}"
    );
    let (a, b) = (alpha.to_f64().unwrap(), beta.to_f64().unwrap());
    // Past the mean `I_b − I_a` cancels; the switch needs a few digits of the side that does not.
    let mass = if lo < a / (a + b) {
      crate::special::beta_i(a, b, up) - crate::special::beta_i(a, b, lo)
    } else {
      crate::special::beta_i(b, a, 1.0 - lo) - crate::special::beta_i(b, a, 1.0 - up)
    };
    Self {
      base: SimdBeta::new(alpha, beta),
      lower,
      upper,
      mass,
      analytics: OnceLock::new(),
    }
  }

  fn analytics(&self) -> Option<&Truncation<BetaKernel>> {
    if let Some(t) = self.analytics.get() {
      return Some(t);
    }
    let (a, b) = (self.base.alpha().to_f64()?, self.base.beta().to_f64()?);
    let (lo, up) = (self.lower.to_f64()?, self.upper.to_f64()?);
    Some(
      self
        .analytics
        .get_or_init(|| Truncation::new(BetaKernel::new(a, b), lo, up)),
    )
  }

  /// The first shape `α` of the untruncated base.
  pub fn alpha(&self) -> T {
    self.base.alpha()
  }

  /// The second shape `β` of the untruncated base.
  pub fn beta(&self) -> T {
    self.base.beta()
  }

  /// The lower bound.
  pub fn lower(&self) -> T {
    self.lower
  }

  /// The upper bound.
  pub fn upper(&self) -> T {
    self.upper
  }

  /// Rejection from the base while the interval holds 1e-3 of its mass (10⁵ tries bound the loop), else the inverse cdf.
  #[inline]
  fn draw<S: TruncatedBetaSource<T>>(&self, src: &mut S) -> T {
    if self.mass >= 1e-3 {
      for _ in 0..100_000 {
        let x = src.base(&self.base);
        if x >= self.lower && x <= self.upper {
          return x;
        }
      }
    }
    self.invert(src.uniform())
  }

  /// Out of line, so the rejection loop stays lean.
  #[inline(never)]
  /// Clamped to the support as `quantile` is: the last exp of a deep singular panel rounds past its near end.
  fn invert(&self, u: f64) -> T {
    let (lo, up) = (self.lower.to_f64().unwrap(), self.upper.to_f64().unwrap());
    T::from_f64_fast(
      self
        .analytics()
        .map_or(f64::NAN, |a| a.quantile(u).clamp(lo, up)),
    )
  }
}

impl<T: SimdFloatExt> Sealed for SimdTruncatedBeta<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdTruncatedBeta<T> {
  type State<R: SimdRngExt> = BetaState<T, R>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (BetaState<T, R>, u64) {
    self.base.init::<R, S>(seed)
  }
}

impl<T: SimdFloatExt> SimdKernel for SimdTruncatedBeta<T> {
  type Item = T;

  fn fill<R: SimdRngExt>(&self, state: &mut BetaState<T, R>, out: &mut [T]) {
    for x in out.iter_mut() {
      *x = self.draw(state);
    }
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut BetaState<T, R>) -> T {
    self.draw(state)
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdTruncatedBeta<T> {
  /// The same rejection or inverse-cdf draw, on scalar beta draws and uniforms from the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.draw(&mut AnyRng(rng))
  }
}

impl<T: SimdFloatExt> DistributionExt for SimdTruncatedBeta<T> {
  fn pdf(&self, x: f64) -> Option<f64> {
    if x < self.lower.to_f64()? || x > self.upper.to_f64()? {
      return Some(0.0);
    }
    Some(self.analytics()?.density(x, 0.0))
  }

  fn cdf(&self, x: f64) -> Option<f64> {
    if x <= self.lower.to_f64()? {
      return Some(0.0);
    }
    if x >= self.upper.to_f64()? {
      return Some(1.0);
    }
    Some(self.analytics()?.cdf(x))
  }

  fn quantile(&self, p: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&p) {
      return Some(f64::NAN);
    }
    let (lo, up) = (self.lower.to_f64()?, self.upper.to_f64()?);
    if p == 0.0 {
      return Some(lo);
    }
    if p == 1.0 {
      return Some(up);
    }
    Some(self.analytics()?.quantile(p).clamp(lo, up))
  }

  fn mean(&self) -> Option<f64> {
    Some(self.analytics()?.moments(1.0)[0])
  }

  fn median(&self) -> Option<f64> {
    self.quantile(0.5)
  }

  /// Off the interior mode monotone toward the shape above 1, or U-shaped: one side of the antimode `(1 − α)/(2 − α − β)`
  /// falls or rises, across it the higher end wins, and a tie (the uniform's too) is NaN, no mode being unique.
  fn mode(&self) -> Option<f64> {
    let (a, b) = (self.base.alpha().to_f64()?, self.base.beta().to_f64()?);
    let (lo, up) = (self.lower.to_f64()?, self.upper.to_f64()?);
    if a > 1.0 && b > 1.0 {
      return Some(((a - 1.0) / (a + b - 2.0)).clamp(lo, up));
    }
    if a >= 1.0 && b <= 1.0 && (a, b) != (1.0, 1.0) {
      return Some(up);
    }
    if a <= 1.0 && b >= 1.0 && (a, b) != (1.0, 1.0) {
      return Some(lo);
    }
    if a < 1.0 && b < 1.0 {
      let antimode = (1.0 - a) / (2.0 - a - b);
      if up <= antimode {
        return Some(lo);
      }
      if lo >= antimode {
        return Some(up);
      }
    }
    let term = |e: f64, y: f64| if e == 0.0 { 0.0 } else { e * y.ln() };
    let log_kernel = |x: f64| term(a - 1.0, x) + term(b - 1.0, 1.0 - x);
    let (at_lo, at_up) = (log_kernel(lo), log_kernel(up));
    Some(if at_lo > at_up {
      lo
    } else if at_up > at_lo {
      up
    } else {
      f64::NAN
    })
  }

  fn variance(&self) -> Option<f64> {
    Some(self.analytics()?.moments(1.0)[1])
  }

  /// NaN, with kurtosis and entropy, for a law narrower than any offset from its reference point resolves.
  fn skewness(&self) -> Option<f64> {
    Some(self.analytics()?.moments(1.0)[2])
  }

  fn kurtosis(&self) -> Option<f64> {
    Some(self.analytics()?.moments(1.0)[3])
  }

  fn entropy(&self) -> Option<f64> {
    Some(self.analytics()?.entropy())
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;
  use crate::tests::scalar_draws;
  use crate::tests::scalar_ks_best_p;

  #[test]
  fn truncated_beta_samples_within_bounds() {
    for seed in [2718, 999, 42] {
      let law = SimdTruncatedBeta::<f64>::new(2.0, 2.0, 0.2, 0.8);
      let mut tb = law.seeded(&Deterministic::new(seed));
      for _ in 0..3_000 {
        let x = tb.sample();
        assert!((0.2..=0.8).contains(&x), "seed {seed}: sample {x}");
      }
    }
  }

  /// Rejection draws leave the analytics unbuilt; equality and `Debug` read the parameters, not the cache.
  #[test]
  fn construction_and_rejection_draws_skip_the_quadrature() {
    let law = SimdTruncatedBeta::<f64>::new(2.0, 2.0, 0.2, 0.8);
    let (twin, text) = (law.clone(), format!("{law:?}"));
    let draws = scalar_draws(&law, 7, 1_000);
    assert!(draws.iter().all(|x| (0.2..=0.8).contains(x)) && law.analytics.get().is_none());
    law.mean();
    assert!(law.analytics.get().is_some() && law == twin && format!("{law:?}") == text);
  }

  /// A bad `beta` shape is reported as `beta`, not as `alpha`.
  #[test]
  #[should_panic(expected = "beta must satisfy `beta > T::zero()`, got beta = -1.0")]
  fn truncated_beta_names_a_bad_beta() {
    SimdTruncatedBeta::<f64>::new(2.0, -1.0, 0.2, 0.8);
  }

  /// Both regimes of the honest `Distribution` draw from the caller's rng and agree with the cdf: rejection on
  /// `[0.2, 0.8]`, inversion on `[0.99, 0.999]`, which holds 4e-6 of the base mass.
  #[test]
  fn scalar_sample_matches_cdf() {
    for (a, b, lo, up) in [(2.0, 2.0, 0.2, 0.8), (2.0, 3.0, 0.99, 0.999)] {
      let law = SimdTruncatedBeta::<f64>::new(a, b, lo, up);
      let best = scalar_ks_best_p(&law, |x| law.cdf(x).unwrap());
      assert!(best > 0.01, "[{lo}, {up}]: best p = {best}");
    }
  }
}

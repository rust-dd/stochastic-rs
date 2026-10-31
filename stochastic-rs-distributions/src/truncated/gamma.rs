//! Truncated gamma: rejection from the base law while the interval holds 1e-3 of its mass, else the inverse cdf.
//!
//! Reference: Coffey, C. S. & Muller, K. E. (2000), "Properties of doubly-truncated gamma variables", *Communications in Statistics - Theory and Methods* 29(4), 851-857, DOI 10.1080/03610920008832519 (truncated gamma moments).
//! Reference: Devroye, L. (1986), *Non-Uniform Random Variate Generation*, Springer, §II.2 and §II.3, DOI 10.1007/978-1-4613-8643-8 (inversion and rejection).

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use super::quadrature::Kernel;
use super::quadrature::Truncation;
use crate::gamma::GammaState;
use crate::gamma::SimdGamma;
use crate::source::AnyRng;
use crate::source::uniform53;
use crate::traits::DistributionExt;
use crate::traits::SimdFloatExt;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

/// `x^{k−1} e^{−x/θ}` on `[0, ∞)`; a unit shape has no log term, whose `0·ln 0` would read NaN at 0.
#[derive(Clone, Copy, Debug, PartialEq)]
struct GammaKernel {
  k: f64,
  theta: f64,
}

impl Kernel for GammaKernel {
  fn ln_k(&self, x: f64) -> f64 {
    if x < 0.0 {
      return f64::NEG_INFINITY;
    }
    let lead = if self.k == 1.0 {
      0.0
    } else {
      (self.k - 1.0) * x.ln()
    };
    lead - x / self.theta
  }

  /// `ln x − ln c` read from `x` beside 0, where `ln_1p(y/c)` would not be.
  fn ln_rel(&self, x: f64, y: f64, c: f64, _branch: Option<(f64, f64)>) -> f64 {
    let lead = if self.k == 1.0 {
      0.0
    } else if y.abs() > 0.5 * c {
      (self.k - 1.0) * (x.ln() - c.ln())
    } else {
      (self.k - 1.0) * (y / c).ln_1p()
    };
    lead - y / self.theta
  }

  fn d1(&self, x: f64) -> f64 {
    let lead = if self.k == 1.0 {
      0.0
    } else {
      (self.k - 1.0) / x
    };
    lead - 1.0 / self.theta
  }

  fn d2(&self, x: f64) -> f64 {
    if self.k == 1.0 {
      0.0
    } else {
      -(self.k - 1.0) / (x * x)
    }
  }

  fn mode(&self) -> f64 {
    ((self.k - 1.0) * self.theta).max(0.0)
  }

  fn branch_points(&self) -> [Option<(f64, f64)>; 2] {
    [(self.k != 1.0).then_some((0.0, self.k - 1.0)), None]
  }

  fn length_scale(&self) -> f64 {
    self.theta
  }
}

/// Truncated $\mathrm{Gamma}(k, \theta)$ on $[\text{lower}, \text{upper}]$, $\text{lower} \ge 0$: parameters and the
/// interval's kernel integral; a [`Seeded`](crate::Seeded) stream draws it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdTruncatedGamma<T> {
  base: SimdGamma<T>,
  lower: T,
  upper: T,
  /// The base law's mass on the interval, `exp(ln ∫k − ln Z)`: the sampler's regime switch.
  mass: f64,
  /// The interval's kernel integral, which the analytics and the inverse-cdf draw read.
  analytics: Truncation<GammaKernel>,
}

/// The base draw and the `[0, 1)` uniform a truncated-gamma draw consumes, from a stream or the caller's rng.
trait TruncatedGammaSource<T: SimdFloatExt> {
  fn base(&mut self, law: &SimdGamma<T>) -> T;

  fn uniform(&mut self) -> f64;
}

impl<T: SimdFloatExt, R: SimdRngExt> TruncatedGammaSource<T> for GammaState<T, R> {
  #[inline]
  fn base(&mut self, law: &SimdGamma<T>) -> T {
    law.next(self)
  }

  #[inline]
  fn uniform(&mut self) -> f64 {
    self.rng.next_f64()
  }
}

impl<T: SimdFloatExt, G: Rng + ?Sized> TruncatedGammaSource<T> for AnyRng<'_, G> {
  #[inline]
  fn base(&mut self, law: &SimdGamma<T>) -> T {
    law.draw_with(self.0)
  }

  #[inline]
  fn uniform(&mut self) -> f64 {
    uniform53(self.0.next_u64())
  }
}

impl<T: SimdFloatExt> SimdTruncatedGamma<T> {
  /// The base [`SimdGamma`] `Gamma(shape, scale)` (`shape` is its `alpha`), both > 0, renormalised on `[lower, upper]`,
  /// `0 ≤ lower < upper`.
  pub fn new(shape: T, scale: T, lower: T, upper: T) -> Self {
    assert!(
      shape.is_finite(),
      "shape must satisfy `shape.is_finite()`, got shape = {shape:?}"
    );
    assert!(
      scale.is_finite(),
      "scale must satisfy `scale.is_finite()`, got scale = {scale:?}"
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
      shape > T::zero(),
      "shape must satisfy `shape > T::zero()`, got shape = {shape:?}"
    );
    assert!(
      scale > T::zero(),
      "scale must satisfy `scale > T::zero()`, got scale = {scale:?}"
    );
    assert!(
      lower >= T::zero(),
      "lower must satisfy `lower >= T::zero()`, got lower = {lower:?}"
    );
    assert!(
      lower < upper,
      "lower must satisfy `lower < upper`, got lower = {lower:?}, upper = {upper:?}"
    );
    let (k, theta) = (shape.to_f64().unwrap(), scale.to_f64().unwrap());
    let kernel = GammaKernel { k, theta };
    let analytics = Truncation::new(kernel, lower.to_f64().unwrap(), upper.to_f64().unwrap());
    let ln_z = crate::special::ln_gamma(k) + k * theta.ln();
    Self {
      base: SimdGamma::new(shape, scale),
      lower,
      upper,
      mass: (analytics.ln_mass() - ln_z).exp(),
      analytics,
    }
  }

  /// The shape `k` of the untruncated base.
  pub fn shape(&self) -> T {
    self.base.alpha()
  }

  /// The scale `θ` of the untruncated base.
  pub fn scale(&self) -> T {
    self.base.scale()
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
  fn draw<S: TruncatedGammaSource<T>>(&self, src: &mut S) -> T {
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
  fn invert(&self, u: f64) -> T {
    T::from_f64_fast(self.analytics.quantile(u))
  }
}

impl<T: SimdFloatExt> Sealed for SimdTruncatedGamma<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdTruncatedGamma<T> {
  type State<R: SimdRngExt> = GammaState<T, R>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (GammaState<T, R>, u64) {
    self.base.init::<R, S>(seed)
  }
}

impl<T: SimdFloatExt> SimdKernel for SimdTruncatedGamma<T> {
  type Item = T;

  fn fill<R: SimdRngExt>(&self, state: &mut GammaState<T, R>, out: &mut [T]) {
    for x in out.iter_mut() {
      *x = self.draw(state);
    }
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut GammaState<T, R>) -> T {
    self.draw(state)
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdTruncatedGamma<T> {
  /// The same rejection or inverse-cdf draw, on scalar gamma draws and uniforms from the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.draw(&mut AnyRng(rng))
  }
}

impl<T: SimdFloatExt> DistributionExt for SimdTruncatedGamma<T> {
  fn pdf(&self, x: f64) -> Option<f64> {
    if x < self.lower.to_f64()? || x > self.upper.to_f64()? {
      return Some(0.0);
    }
    Some(self.analytics.density(x))
  }

  fn cdf(&self, x: f64) -> Option<f64> {
    if x <= self.lower.to_f64()? {
      return Some(0.0);
    }
    if x >= self.upper.to_f64()? {
      return Some(1.0);
    }
    Some(self.analytics.cdf(x))
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
    Some(self.analytics.quantile(p).clamp(lo, up))
  }

  fn mean(&self) -> Option<f64> {
    Some(self.analytics.moments()[0])
  }

  fn median(&self) -> Option<f64> {
    self.quantile(0.5)
  }

  fn mode(&self) -> Option<f64> {
    let (k, theta) = (self.base.alpha().to_f64()?, self.base.scale().to_f64()?);
    let (lo, up) = (self.lower.to_f64()?, self.upper.to_f64()?);
    Some(if k >= 1.0 {
      ((k - 1.0) * theta).clamp(lo, up)
    } else {
      lo
    })
  }

  fn variance(&self) -> Option<f64> {
    Some(self.analytics.moments()[1])
  }

  fn skewness(&self) -> Option<f64> {
    Some(self.analytics.moments()[2])
  }

  fn kurtosis(&self) -> Option<f64> {
    Some(self.analytics.moments()[3])
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;
  use crate::tests::scalar_ks_best_p;

  /// Truncated gamma draws stay in bounds.
  #[test]
  fn truncated_gamma_samples_within_bounds() {
    for seed in [2718, 999, 42] {
      let law = SimdTruncatedGamma::<f64>::new(2.0, 1.0, 1.0, 5.0);
      let mut tg = law.seeded(&Deterministic::new(seed));
      for _ in 0..3_000 {
        let x = tg.sample();
        assert!((1.0..=5.0).contains(&x), "seed {seed}: sample {x}");
      }
    }
  }

  /// Both regimes of the honest `Distribution` draw from the caller's rng and agree with the cdf: rejection on
  /// `[1, 5]`, inversion on `[45, 60]`, which holds 1.3e-18 of the base mass.
  #[test]
  fn scalar_sample_matches_cdf() {
    for (lo, up) in [(1.0, 5.0), (45.0, 60.0)] {
      let law = SimdTruncatedGamma::<f64>::new(2.0, 1.0, lo, up);
      let best = scalar_ks_best_p(&law, |x| law.cdf(x).unwrap());
      assert!(best > 0.01, "[{lo}, {up}]: best p = {best}");
    }
  }
}

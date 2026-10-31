//! Truncated gamma: rejection from the base law while the interval holds 1e-3 of its mass, else the inverse cdf.
//!
//! Reference: Coffey, C. S. & Muller, K. E. (2000), "Properties of doubly-truncated gamma variables", *Communications in Statistics - Theory and Methods* 29(4), 851-857, DOI 10.1080/03610920008832519 (truncated gamma moments).
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
use crate::gamma::GammaState;
use crate::gamma::SimdGamma;
use crate::source::AnyRng;
use crate::source::uniform53;
use crate::traits::DistributionExt;
use crate::traits::SimdFloatExt;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

/// `u^{k−1} e^{−u/s}` on `[0, ∞)` about the reference point `c`; a unit shape has no log term, whose `0·ln 0` would
/// read NaN at 0.
#[derive(Clone, Copy, Debug, PartialEq)]
struct GammaKernel {
  k: f64,
  /// The scale in the kernel's own units: 1 in units of θ, θ in the law's.
  s: f64,
  c: f64,
  ln_c: f64,
  /// `(k − 1)/c − 1/s`, the slope at `c`, `None` where it overflows: the linear part, so `ln_rel` sums no two large
  /// terms that cancel.
  slope: Option<f64>,
}

impl GammaKernel {
  fn new(k: f64, s: f64) -> Self {
    Self {
      k,
      s,
      c: 0.0,
      ln_c: f64::NEG_INFINITY,
      slope: None,
    }
  }
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
    lead - x / self.s
  }

  /// Above a unit shape the slope from `(k − 1)s − c` by an exact product, 0 at a mode whose rounding moves it past the
  /// law's width `c/√(k − 1)`, where the law lies within its ulp; below one the two terms share a sign.
  fn anchor(&mut self, c: Cut) {
    self.c = c.x;
    self.ln_c = c.x.ln();
    let slope = if self.k == 1.0 {
      -1.0 / self.s
    } else if self.k < 1.0 {
      (self.k - 1.0) / c.x - 1.0 / self.s
    } else {
      let slope = (self.k - 1.0).mul_add(self.s, -c.x) / (c.x * self.s);
      if c == self.mode() && slope.abs() * c.x > (self.k - 1.0).sqrt() {
        0.0
      } else {
        slope
      }
    };
    self.slope = slope.is_finite().then_some(slope);
  }

  /// `(k − 1)(ln(x/c) − y/c) + slope·y`, the first part `ln_1p_mx` beside `c` and read from `x` toward 0; where the
  /// slope overflows the mode is far, and `(k − 1) ln(x/c) − y/s` cancels nothing.
  fn ln_rel(&self, x: f64, y: f64, _branch: Option<(f64, f64)>) -> f64 {
    let z = y / self.c;
    let ln_ratio = || {
      if z < -0.5 {
        x.ln() - self.ln_c
      } else {
        z.ln_1p()
      }
    };
    match self.slope {
      _ if self.k == 1.0 => -y / self.s,
      Some(slope) if z < -0.5 => (self.k - 1.0) * (ln_ratio() - z) + slope * y,
      Some(slope) => (self.k - 1.0) * ln_1p_mx(z) + slope * y,
      None => (self.k - 1.0) * ln_ratio() - y / self.s,
    }
  }

  fn ln_rel_regular(&self, _x: f64, y: f64, _s: f64) -> f64 {
    -(self.k - 1.0) * self.ln_c - y / self.s
  }

  fn d1(&self, x: f64, y: f64) -> f64 {
    match self.slope {
      _ if self.k == 1.0 => -1.0 / self.s,
      Some(slope) => slope - (self.k - 1.0) * (y / self.c) / x,
      None => (self.k - 1.0) / x - 1.0 / self.s,
    }
  }

  fn d2(&self, x: f64, _y: f64) -> f64 {
    if self.k == 1.0 {
      0.0
    } else {
      -(self.k - 1.0) / (x * x)
    }
  }

  fn mode(&self) -> Cut {
    Cut::at(((self.k - 1.0) * self.s).max(0.0))
  }

  fn branch_points(&self) -> [Option<(f64, f64)>; 2] {
    [(self.k != 1.0).then_some((0.0, self.k)), None]
  }

  fn length_scale(&self) -> f64 {
    self.s
  }

  fn support(&self) -> (f64, f64) {
    (0.0, f64::INFINITY)
  }

  fn base_moments(&self) -> [f64; 4] {
    let root = self.k.sqrt();
    [self.k * self.s, root * self.s, 2.0 / root, 6.0 / self.k]
  }
}

/// Truncated $\mathrm{Gamma}(k, \theta)$ on $[\text{lower}, \text{upper}]$, $\text{lower} \ge 0$: parameters and the
/// base mass on the interval; a [`Seeded`](crate::Seeded) stream draws it.
#[derive(Clone)]
pub struct SimdTruncatedGamma<T> {
  base: SimdGamma<T>,
  lower: T,
  upper: T,
  /// The base law's mass on the interval: the sampler's regime switch.
  mass: f64,
  /// The kernel's unit, a power of two: see `unit_of`.
  unit: f64,
  /// The interval's kernel integral, built by the first analytic call or inverse-cdf draw.
  analytics: OnceLock<Truncation<GammaKernel>>,
}

impl<T: fmt::Debug> fmt::Debug for SimdTruncatedGamma<T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("SimdTruncatedGamma")
      .field("base", &self.base)
      .field("lower", &self.lower)
      .field("upper", &self.upper)
      .finish_non_exhaustive()
  }
}

/// The parameters decide every cached value, so they alone are compared.
impl<T: PartialEq> PartialEq for SimdTruncatedGamma<T> {
  fn eq(&self, other: &Self) -> bool {
    self.base == other.base && self.lower == other.lower && self.upper == other.upper
  }
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
    let (x_lo, x_up) = (
      lower.to_f64().unwrap() / theta,
      upper.to_f64().unwrap() / theta,
    );
    // Past the mean `P(k, b) − P(k, a)` cancels; the switch needs a few digits of the side that does not.
    let mass = if x_lo < k {
      crate::special::gamma_p(k, x_up) - crate::special::gamma_p(k, x_lo)
    } else {
      crate::special::gamma_q(k, x_lo) - crate::special::gamma_q(k, x_up)
    };
    Self {
      base: SimdGamma::new(shape, scale),
      lower,
      upper,
      mass,
      unit: unit_of(k, theta),
      analytics: OnceLock::new(),
    }
  }

  fn analytics(&self) -> Option<&Truncation<GammaKernel>> {
    if let Some(t) = self.analytics.get() {
      return Some(t);
    }
    let (k, theta) = (self.base.alpha().to_f64()?, self.base.scale().to_f64()?);
    let unit = self.unit()?;
    let (lo, up) = (self.lower.to_f64()? / unit, self.upper.to_f64()? / unit);
    Some(
      self
        .analytics
        .get_or_init(|| Truncation::new(GammaKernel::new(k, theta / unit), lo, up)),
    )
  }

  fn unit(&self) -> Option<f64> {
    Some(self.unit)
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
  /// Clamped to the support as `quantile` is: the last exp of a deep singular panel rounds past its near end.
  fn invert(&self, u: f64) -> T {
    let (lo, up) = (self.lower.to_f64().unwrap(), self.upper.to_f64().unwrap());
    let x = self
      .analytics()
      .zip(self.unit())
      .map_or(f64::NAN, |(a, unit)| (a.quantile(u) * unit).clamp(lo, up));
    T::from_f64_fast(x)
  }
}

/// The power of two at or above θ: `x/unit` is exact and the mode `(k − 1)θ/unit` representable; a shape below 1 with
/// θ above 1 stays in `x`, its singular end holding mass under `unit·2⁻¹⁰⁷⁴` (40 % of TG(0.001, 1e300) below 1e-100).
fn unit_of(k: f64, theta: f64) -> f64 {
  if k < 1.0 && theta > 1.0 && theta <= f64::MAX / 128.0 {
    return 1.0;
  }
  let floor = f64::from_bits(theta.to_bits() & (0x7ff << 52));
  if floor == 0.0 {
    f64::MIN_POSITIVE
  } else if floor == theta || floor >= 2f64.powi(1023) {
    floor
  } else {
    2.0 * floor
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
  /// At a point whose `x/θ` overflows the limit 0, where the kernel's `(k − 1) ln u − u` reads `∞ − ∞`.
  fn pdf(&self, x: f64) -> Option<f64> {
    if x < self.lower.to_f64()? || x > self.upper.to_f64()? {
      return Some(0.0);
    }
    let unit = self.unit()?;
    let u = x / unit;
    if u == f64::INFINITY {
      return Some(0.0);
    }
    // A power of two: its logarithm is its exponent's multiple of ln 2.
    let ln_unit = f64::from(((unit.to_bits() >> 52) as i32) - 1023) * std::f64::consts::LN_2;
    Some(self.analytics()?.density(u, ln_unit))
  }

  fn cdf(&self, x: f64) -> Option<f64> {
    if x <= self.lower.to_f64()? {
      return Some(0.0);
    }
    if x >= self.upper.to_f64()? {
      return Some(1.0);
    }
    Some(self.analytics()?.cdf(x / self.unit()?))
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
    Some((self.analytics()?.quantile(p) * self.unit()?).clamp(lo, up))
  }

  fn mean(&self) -> Option<f64> {
    let unit = self.unit()?;
    Some(self.analytics()?.moments(unit)[0] * unit)
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
    Some(self.analytics()?.moments(self.unit()?)[1])
  }

  /// NaN, with kurtosis and entropy, for a law narrower than any offset from its reference point resolves.
  fn skewness(&self) -> Option<f64> {
    Some(self.analytics()?.moments(1.0)[2])
  }

  fn kurtosis(&self) -> Option<f64> {
    Some(self.analytics()?.moments(1.0)[3])
  }

  fn entropy(&self) -> Option<f64> {
    Some(self.analytics()?.entropy() + self.unit()?.ln())
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;
  use crate::tests::scalar_draws;
  use crate::tests::scalar_ks_best_p;

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

  /// Rejection draws leave the analytics unbuilt; equality and `Debug` read the parameters, not the cache.
  #[test]
  fn construction_and_rejection_draws_skip_the_quadrature() {
    let law = SimdTruncatedGamma::<f64>::new(2.0, 1.0, 1.0, 5.0);
    let (twin, text) = (law.clone(), format!("{law:?}"));
    let draws = scalar_draws(&law, 7, 1_000);
    assert!(draws.iter().all(|x| (1.0..=5.0).contains(x)) && law.analytics.get().is_none());
    law.mean();
    assert!(law.analytics.get().is_some() && law == twin && format!("{law:?}") == text);
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

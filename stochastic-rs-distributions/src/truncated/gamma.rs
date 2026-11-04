//! Truncated gamma: rejection from the base law while the interval holds 1e-3 of its mass, else the inverse cdf. Past
//! `θ(2k + 260) = MAX·2^{e+1022}`, `2^e ≤ lower`, no unit holds both, and beside `lower > 0` the law reads untruncated.
//!
//! Reference: Coffey, C. S. & Muller, K. E. (2000), "Properties of doubly-truncated gamma variables", *Communications in Statistics - Theory and Methods* 29(4), 851-857, DOI 10.1080/03610920008832519 (truncated gamma moments).
//! Reference: Devroye, L. (1986), *Non-Uniform Random Variate Generation*, Springer, §II.2 and §II.3, DOI 10.1007/978-1-4613-8643-8 (inversion and rejection).

use std::fmt;
use std::sync::OnceLock;

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use super::local::Local;
use super::local::Model;
use super::local::Signed;
use super::local::ln_cubic;
use super::quadrature::Cut;
use super::quadrature::Offset;
use super::quadrature::Point;
use super::quadrature::ldexp;
use crate::gamma::GammaState;
use crate::gamma::SimdGamma;
use crate::source::AnyRng;
use crate::source::uniform53;
use crate::special::split_pow2;
use crate::traits::DistributionExt;
use crate::traits::SimdFloatExt;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;
pub(super) use kernel::GammaKernel;

mod kernel;

/// Truncated $\mathrm{Gamma}(k, \theta)$ on $[\text{lower}, \text{upper}]$, $\text{lower} \ge 0$: parameters and the
/// base mass on the interval; a [`Seeded`](crate::Seeded) stream draws it.
#[derive(Clone)]
pub struct SimdTruncatedGamma<T> {
  base: SimdGamma<T>,
  lower: T,
  upper: T,
  /// The base law's mass on the interval: the sampler's regime switch.
  mass: f64,
  /// The kernel's unit `2^p`: see `unit_exponent`.
  p: i32,
  /// `2^{−p}`, exact even at `p = 1024` (a subnormal power of two), and `ln 2^p`: a coordinate costs one product.
  inv_unit: f64,
  ln_unit: f64,
  /// The interval's kernel integral, built by the first analytic call or inverse-cdf draw.
  analytics: OnceLock<Model<GammaKernel>>,
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
    let (lo, up) = (lower.to_f64().unwrap(), upper.to_f64().unwrap());
    let p = unit_exponent(theta.min(up));
    // At most 2^1022 times `lower`'s exponent, so `lower/2^p` is normal and offsets beside it exact, wherever the kernel's
    // reach past its mode (under 260 scales) fits that unit; past it `lower/2^p` underflows and the law reads untruncated.
    let floor = split_pow2(lo).0 + 1022;
    let reach = (ldexp(theta, -floor) * (2.0 * k + 260.0)).min(ldexp(up, -floor));
    let p = if lo > 0.0 && floor < p && reach.is_finite() {
      floor
    } else {
      p
    };
    Self {
      base: SimdGamma::new(shape, scale),
      lower,
      upper,
      mass,
      p,
      inv_unit: ldexp(1.0, -p),
      ln_unit: f64::from(p) * std::f64::consts::LN_2,
      analytics: OnceLock::new(),
    }
  }

  #[inline]
  fn analytics(&self) -> Option<&Model<GammaKernel>> {
    match self.analytics.get() {
      Some(m) => Some(m),
      None => self.fill(),
    }
  }

  /// The first analytic call's integral, out of the hot path.
  #[cold]
  #[inline(never)]
  fn fill(&self) -> Option<&Model<GammaKernel>> {
    let (k, theta) = (self.base.alpha().to_f64()?, self.base.scale().to_f64()?);
    let (lo, up) = (self.lower.to_f64()?, self.upper.to_f64()?);
    let (lo_u, up_u) = (self.units(lo), self.units(up));
    Some(self.analytics.get_or_init(|| {
      // Past θ·f64::MAX the law sits at `lo`, its mode `(k − 1)θ` below: an exponential of scale θ there.
      if lo_u == f64::INFINITY
        && let Some(l) = self.local(lo)
      {
        return Model::Local(l);
      }
      Model::new(GammaKernel::new(k, self.units(theta)), lo_u, up_u, |c| {
        self.local(if c == lo_u {
          lo
        } else if c == up_u {
          up
        } else {
          ldexp(c, self.p)
        })
      })
    }))
  }

  /// The local model at `r`: slope `(k − 1)/r − 1/θ` and curvature `−(k − 1)/r²`, in logs.
  fn local(&self, r: f64) -> Option<Local> {
    let (k, theta) = (self.base.alpha().to_f64()?, self.base.scale().to_f64()?);
    let (lo, up) = (self.lower.to_f64()?, self.upper.to_f64()?);
    let ln_r = r.ln();
    let slope = Signed::over(k - 1.0, ln_r).plus(Signed::new(-1.0, -theta.ln()));
    Local::new(
      r,
      (lo - r, up - r),
      slope,
      Signed::over(1.0 - k, 2.0 * ln_r),
      |y| ln_cubic(k - 1.0, y / r),
    )
  }

  /// The clipped upper end of the mass, in the law's coordinate: the interval's end for a law its local model holds.
  pub(super) fn clipped_end(&self) -> Option<f64> {
    Some(match self.analytics()? {
      Model::Full(t) => ldexp(t.clipped().1, self.p),
      Model::Local(_) => self.upper.to_f64()?,
    })
  }

  /// `x` in the kernel's unit, exact unless it leaves the range.
  fn units(&self, x: f64) -> f64 {
    x * self.inv_unit
  }

  fn ln_unit(&self) -> f64 {
    self.ln_unit
  }

  /// `x/2^p` below the normal doubles beside 0, given by its log: where the lower end is 0 the engine reads it there.
  fn beside_zero(&self, x: f64, u: f64) -> Option<f64> {
    (x > 0.0 && u < f64::MIN_POSITIVE && self.lower.to_f64()? == 0.0)
      .then(|| x.ln() - self.ln_unit())
  }

  /// A position the engine reports, back in the law's coordinate: beside a point its log carries the unit.
  fn located(&self, point: Point) -> f64 {
    match point {
      Point::Beside { s, sign, ln_t } => ldexp(s, self.p) + sign * (ln_t + self.ln_unit()).exp(),
      _ => ldexp(point.value(), self.p),
    }
  }

  /// The pdf off the normal doubles in the kernel's unit, out of line: the limit 0 where `x/2^p` overflows, where the
  /// kernel's `(k − 1) ln u − u` reads `∞ − ∞`, and beside 0 through its log.
  #[cold]
  #[inline(never)]
  fn pdf_rare(&self, model: &Model<GammaKernel>, x: f64, u: f64) -> Option<f64> {
    let t = match model {
      Model::Full(t) => t,
      Model::Local(l) => return Some(l.pdf(x, 0.0)),
    };
    if u == f64::INFINITY {
      return Some(0.0);
    }
    Some(match self.beside_zero(x, u) {
      Some(ln_t) => t.density_beside(0.0, self.base.alpha().to_f64()?, ln_t, self.ln_unit),
      None => t.density(u, self.ln_unit),
    })
  }

  /// `(mean, variance, skewness, excess kurtosis)` in the law's coordinate.
  pub(super) fn moments(&self) -> Option<[f64; 4]> {
    Some(match self.analytics()? {
      Model::Full(t) => {
        let [mean, variance, skewness, kurtosis] = t.moments(1.0, self.p);
        [ldexp(mean, self.p), variance, skewness, kurtosis]
      }
      Model::Local(l) => l.moments(),
    })
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

  /// Out of line, so the rejection loop stays lean; clamped to the support as `quantile` is, since the last exp of a
  /// deep singular panel rounds past its near end.
  #[inline(never)]
  fn invert(&self, u: f64) -> T {
    let (lo, up) = (self.lower.to_f64().unwrap(), self.upper.to_f64().unwrap());
    let x = match self.analytics() {
      Some(Model::Full(t)) => self.located(t.quantile(u)),
      Some(Model::Local(l)) => l.quantile(u),
      None => f64::NAN,
    };
    T::from_f64_fast(x.clamp(lo, up))
  }
}

/// The least power of two at or above the law's length, the shorter of θ and `upper` (2⁻¹⁰²² for a subnormal one):
/// `x/2^p` is exact, an interval far below θ keeps its digits, and past 2¹⁰²³ the unit 2¹⁰²⁴ lives in its exponent.
pub(super) fn unit_exponent(length: f64) -> i32 {
  let biased = ((length.to_bits() >> 52) & 0x7ff) as i32;
  if biased == 0 {
    return -1022;
  }
  let exact = length.to_bits() & ((1 << 52) - 1) == 0;
  biased - 1023 + i32::from(!exact)
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
  #[inline]
  fn pdf(&self, x: f64) -> Option<f64> {
    if x < self.lower.to_f64()? || x > self.upper.to_f64()? {
      return Some(0.0);
    }
    let u = self.units(x);
    match self.analytics()? {
      Model::Full(t) if (f64::MIN_POSITIVE..f64::INFINITY).contains(&u) => {
        Some(t.density(u, self.ln_unit))
      }
      model => self.pdf_rare(model, x, u),
    }
  }

  fn cdf(&self, x: f64) -> Option<f64> {
    if x <= self.lower.to_f64()? {
      return Some(0.0);
    }
    if x >= self.upper.to_f64()? {
      return Some(1.0);
    }
    let t = match self.analytics()? {
      Model::Full(t) => t,
      Model::Local(l) => return Some(l.cdf(x)),
    };
    let u = self.units(x);
    Some(match self.beside_zero(x, u) {
      Some(_) => {
        let Offset { m, q } = Offset::of(x);
        t.cdf_beside(
          0.0,
          self.base.alpha().to_f64()?,
          Offset { m, q: q - self.p },
        )
      }
      None => t.cdf(Cut::at(u)),
    })
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
    let x = match self.analytics()? {
      Model::Full(t) => self.located(t.quantile(p)),
      Model::Local(l) => l.quantile(p),
    };
    Some(x.clamp(lo, up))
  }

  fn mean(&self) -> Option<f64> {
    Some(self.moments()?[0])
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
    Some(self.moments()?[1])
  }

  fn skewness(&self) -> Option<f64> {
    Some(self.moments()?[2])
  }

  fn kurtosis(&self) -> Option<f64> {
    Some(self.moments()?[3])
  }

  fn entropy(&self) -> Option<f64> {
    Some(match self.analytics()? {
      Model::Full(t) => t.entropy() + self.ln_unit(),
      Model::Local(l) => l.entropy(),
    })
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

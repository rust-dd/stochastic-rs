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
use fractions::mean_fraction;
use fractions::one_minus_exp_over;
use fractions::variance_fraction;

mod cf;
mod fractions;
mod mgf;
mod turns;

/// 2⁶⁴: a subnormal times it is a normal double, and every value the frame scales stays below `MAX` unless past `2⁹⁶⁰`.
const SCALE: f64 = 18_446_744_073_709_551_616.0;

/// `2⁻⁹⁶⁰`: from here up `t` and its mass per rate, at least `0.63t`, are normal doubles of full precision.
const TINY: f64 = f64::from_bits(63 << 52);

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

  fn anchor(&mut self, _c: Cut, _at_mode: bool) {}

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
      self.lower.to_f64().unwrap() + self.over_rate(u)
    } else {
      self.log_map(y)
    }
  }

  /// `v·m/λ`; where `m/λ` alone passes the doubles (a subnormal `λ`) from factors scaled by 2⁶⁴, its value of order `v/λ`.
  fn over_rate(&self, v: f64) -> f64 {
    let per_rate = self.mass_per_rate();
    if per_rate.is_finite() {
      return v * per_rate;
    }
    (v * SCALE) * (self.tail_mass / (self.lambda.to_f64().unwrap() * SCALE))
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

  /// `mass_per_rate`·2⁶⁴, from `tail_mass·2⁶⁴` or `L·2⁶⁴` as it forms them.
  #[cold]
  #[inline(never)]
  fn scaled_mass_per_rate(&self) -> f64 {
    let lam = self.lambda.to_f64().unwrap();
    if self.tail_mass >= f64::MIN_POSITIVE {
      (self.tail_mass * SCALE) / lam
    } else {
      let len = self.length().unwrap();
      (len * SCALE) * one_minus_exp_over(lam * len)
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
    Some(Truncation::mass_only(kernel, 0.0, x.max(1.0)).moments(1.0, 0))
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
  fn characteristic_function(&self, t: f64) -> Option<Complex64> {
    self.cf(t)
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
    let decay = (-lam * (x - lo)).exp();
    let per_rate = self.mass_per_rate();
    // A subnormal `λ` puts `m/λ` past the doubles: `λ·(e/m)` rounds the density once.
    Some(if per_rate.is_finite() {
      decay / per_rate
    } else {
      lam * (decay / self.tail_mass)
    })
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
    let (lam, d, w) = (self.lambda.to_f64().unwrap(), x - lo, up - x);
    // Below `λL = 1` as `t·φ(λt)` over the mass, `t` under 2⁻⁹⁶⁰ in `over_rate`'s frame scaled by 2⁶⁴; above it a
    // subnormal `λt` is `t·λ/m`, which `expm1` would take rounded on the subnormals' grid.
    let flat = lam * (up - lo) <= 1.0;
    let mass = |t: f64| {
      let rate_t = lam * t;
      if !flat {
        if rate_t < f64::MIN_POSITIVE {
          t * (lam / self.tail_mass)
        } else {
          -(-rate_t).exp_m1() / self.tail_mass
        }
      } else if t < TINY {
        (t * SCALE) * one_minus_exp_over(rate_t) / self.scaled_mass_per_rate()
      } else {
        t * one_minus_exp_over(rate_t) / self.mass_per_rate()
      }
    };
    // Past the median one less the mass above `x`, which keeps the digits a level next to 1 has.
    let above = (-lam * d).exp() * mass(w);
    Some(if above < 0.5 { 1.0 - above } else { mass(d) })
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
    let (lo, up, lam) = (
      self.lower.to_f64()?,
      self.upper.to_f64()?,
      self.lambda.to_f64()?,
    );
    let len = self.length()?;
    let lam_len = lam * len;
    // Nearer a finite `upper` its offset `w`, from `e^{λw} − 1 = (1 − p)(e^{λL} − 1)`: a level next to 1 keeps its
    // digits.
    if up.is_finite() && lam_len < 700.0 {
      let w = if lam_len < 1e-8 {
        (1.0 - p) * len * (1.0 + 0.5 * p * lam_len)
      } else {
        ((1.0 - p) * lam_len.exp_m1()).ln_1p() / lam
      };
      if w <= 0.5 * len {
        return Some((up - w).max(lo));
      }
    }
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

  /// Past `λL = 50` the cut's share is below ε: `(1/λ)²` and `L·(L·f)`, neither `λ²` nor `L²` formed alone.
  fn variance(&self) -> Option<f64> {
    let (lam, len) = (self.lambda.to_f64()?, self.length()?);
    if self.open_tail()? || lam * len > 50.0 {
      let r = 1.0 / lam;
      return Some(r * r);
    }
    Some(len * (len * variance_fraction(lam * len)))
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
    Some(self.mass_per_rate().ln() + x * mean_fraction(x))
  }

  /// Finite for every `t` on a bounded interval (`λL e^{λ·lower}/m` at `t = λ`, `m` the mass); an open tail diverges from `t = λ` on.
  /// At an infinite `t` the limit of `e^{tx}` on `x ≥ 0`: +∞ above, 0 below.
  fn moment_generating_function(&self, t: f64) -> Option<f64> {
    self.mgf(t)
  }
}

#[cfg(test)]
mod tests;

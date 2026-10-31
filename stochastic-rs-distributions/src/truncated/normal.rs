//! Truncated normal: rejection on the base normal, Robert's one-sided tail proposals, or the inverse cdf.
//!
//! Reference: Johnson, N. L., Kotz, S. & Balakrishnan, N. (1994), *Continuous Univariate Distributions*, vol. 1, 2nd ed., Wiley, ch. 13 §10.1, ISBN 0-471-58495-9 (truncated normal moments).
//! Reference: Kan, R. & Robotti, C. (2017), "On Moments of Folded and Truncated Multivariate Normal Distributions", *Journal of Computational and Graphical Statistics* 26(4), 930-934, DOI 10.1080/10618600.2017.1322092 (the mgf as the shifted interval's mass).

use std::fmt;
use std::sync::OnceLock;

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use super::quadrature::Kernel;
use super::quadrature::Truncation;
use crate::normal::SimdNormal;
use crate::seeded::StreamState;
use crate::source::AnyRng;
use crate::source::uniform53;
use crate::traits::DistributionExt;
use crate::traits::SimdFloatExt;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

/// `e^{−z²/2}` in standardised units.
#[derive(Clone, Copy, Debug, PartialEq)]
struct NormalKernel;

impl Kernel for NormalKernel {
  fn ln_k(&self, z: f64) -> f64 {
    -0.5 * z * z
  }

  fn ln_rel(&self, _z: f64, y: f64, c: f64, _branch: Option<(f64, f64)>) -> f64 {
    -0.5 * y * (y + 2.0 * c)
  }

  fn d1(&self, z: f64) -> f64 {
    -z
  }

  fn d2(&self, _z: f64) -> f64 {
    -1.0
  }

  fn mode(&self) -> f64 {
    0.0
  }

  fn branch_points(&self) -> [Option<(f64, f64)>; 2] {
    [None, None]
  }

  fn length_scale(&self) -> f64 {
    1.0
  }
}

/// Which of Robert (1995)'s proposals a one-sided standardised interval $[a, b]$ with $a \ge 0$ takes.
#[derive(Clone, Copy, Debug, PartialEq)]
enum TailProposal {
  /// The translated exponential of rate $\alpha^\*$ (Robert 1995, §2.1, Proposition 2.3).
  Exponential { alpha_star: f64 },
  /// The uniform on $[a, b]$ reweighted by $e^{(a^2 - z^2)/2}$ (Robert 1995, §2.2).
  Uniform,
}

/// The standardised one-sided interval $0 \le a < b \le \infty$ of a tail draw, negated back when `mirrored`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct TailSetup {
  a: f64,
  b: f64,
  mirrored: bool,
  proposal: TailProposal,
}

impl TailSetup {
  fn new(a: f64, b: f64, mirrored: bool) -> Self {
    let root = (a * a + 4.0).sqrt();
    let alpha_star = 0.5 * (a + root);
    // Robert (1995), (2.1): the exponential proposal accepts more often from this width on.
    let exponential_from =
      2.0 * std::f64::consts::E.sqrt() / (a + root) * (0.25 * (a * a - a * root)).exp();
    let proposal = if b - a > exponential_from {
      TailProposal::Exponential { alpha_star }
    } else {
      TailProposal::Uniform
    };
    Self {
      a,
      b,
      mirrored,
      proposal,
    }
  }
}

/// Truncated normal law $\mathcal{N}(\mu, \sigma^2)$ on $[\text{lower}, \text{upper}]$: parameters and the interval's
/// cdf constants; a [`Seeded`](crate::Seeded) stream draws it.
///
/// Robert, C.P. (1995), "Simulation of truncated normal variables", *Statistics and Computing* 5(2), 121-125, DOI 10.1007/BF00143942.
/// Devroye, L. (1986), *Non-Uniform Random Variate Generation*, Springer, §II.3, DOI 10.1007/978-1-4613-8643-8.
#[derive(Clone)]
pub struct SimdTruncatedNormal<T> {
  base: SimdNormal<T>,
  lower: T,
  upper: T,
  /// `F(lower)`, `F(upper)` and their difference: the sampler's regime switch and its inverse-cdf map (the analytics integrate the kernel).
  f_lo: f64,
  f_up: f64,
  norm_mass: f64,
  /// Set when the interval lies wholly on one side of `μ`, where the inverse cdf loses its digits first.
  tail: Option<TailSetup>,
  /// The standardised interval's kernel integral, built by the first analytic call; construction and draws skip it.
  analytics: OnceLock<Truncation<NormalKernel>>,
  /// `1/σ`, so a density costs a product.
  inv_sigma: f64,
}

impl<T: fmt::Debug> fmt::Debug for SimdTruncatedNormal<T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("SimdTruncatedNormal")
      .field("base", &self.base)
      .field("lower", &self.lower)
      .field("upper", &self.upper)
      .finish_non_exhaustive()
  }
}

/// The parameters decide every cached value, so they alone are compared.
impl<T: PartialEq> PartialEq for SimdTruncatedNormal<T> {
  fn eq(&self, other: &Self) -> bool {
    self.base == other.base && self.lower == other.lower && self.upper == other.upper
  }
}

/// A truncated-normal stream: the base normal's sub-stream and the engine of the tail and inverse-cdf uniforms.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct TruncatedNormalState<T: SimdFloatExt, R: SimdRngExt> {
  base: StreamState<T, R, 64>,
  rng: R,
}

/// The base normal draw and the `[0, 1)` uniform a truncated-normal draw consumes, from a stream or the caller's rng.
trait TruncatedNormalSource<T: SimdFloatExt> {
  fn base(&mut self, law: &SimdNormal<T>) -> T;

  fn uniform(&mut self) -> f64;
}

impl<T: SimdFloatExt, R: SimdRngExt> TruncatedNormalSource<T> for TruncatedNormalState<T, R> {
  #[inline]
  fn base(&mut self, law: &SimdNormal<T>) -> T {
    law.next(&mut self.base)
  }

  #[inline]
  fn uniform(&mut self) -> f64 {
    self.rng.next_f64()
  }
}

impl<T: SimdFloatExt, G: Rng + ?Sized> TruncatedNormalSource<T> for AnyRng<'_, G> {
  #[inline]
  fn base(&mut self, law: &SimdNormal<T>) -> T {
    law.draw_with(self.0)
  }

  #[inline]
  fn uniform(&mut self) -> f64 {
    uniform53(self.0.next_u64())
  }
}

impl<T: SimdFloatExt> SimdTruncatedNormal<T> {
  /// The parent [`SimdNormal`] `N(mu, sigma²)`, `sigma > 0`, renormalised on `[lower, upper]`, `lower < upper`.
  pub fn new(mu: T, sigma: T, lower: T, upper: T) -> Self {
    assert!(
      mu.is_finite(),
      "mu must satisfy `mu.is_finite()`, got mu = {mu:?}"
    );
    assert!(
      sigma.is_finite(),
      "sigma must satisfy `sigma.is_finite()`, got sigma = {sigma:?}"
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
      sigma > T::zero(),
      "sigma must satisfy `sigma > T::zero()`, got sigma = {sigma:?}"
    );
    assert!(
      lower < upper,
      "lower must satisfy `lower < upper`, got lower = {lower:?}, upper = {upper:?}"
    );
    let mu_f64 = mu.to_f64().unwrap();
    let sigma_f64 = sigma.to_f64().unwrap();
    let a_std = (lower.to_f64().unwrap() - mu_f64) / sigma_f64;
    let b_std = (upper.to_f64().unwrap() - mu_f64) / sigma_f64;
    let f_lo = norm_cdf_scalar(a_std);
    let f_up = norm_cdf_scalar(b_std);
    // A left-tail interval is reflected into the right tail, so the tail sampler only sees `a >= 0`.
    let tail = if a_std >= 0.0 {
      Some(TailSetup::new(a_std, b_std, false))
    } else if b_std <= 0.0 {
      Some(TailSetup::new(-b_std, -a_std, true))
    } else {
      None
    };
    Self {
      base: SimdNormal::new(mu, sigma),
      lower,
      upper,
      f_lo,
      f_up,
      norm_mass: f_up - f_lo,
      tail,
      analytics: OnceLock::new(),
      inv_sigma: 1.0 / sigma_f64,
    }
  }

  fn analytics(&self) -> Option<&Truncation<NormalKernel>> {
    if let Some(t) = self.analytics.get() {
      return Some(t);
    }
    let (alpha, beta) = self.standardised_bounds()?;
    Some(
      self
        .analytics
        .get_or_init(|| Truncation::new(NormalKernel, alpha, beta)),
    )
  }

  /// The parent normal's location `μ`, not the truncated law's mean.
  pub fn mu(&self) -> T {
    self.base.mean()
  }

  /// The parent normal's scale `σ`, not the truncated law's standard deviation.
  pub fn sigma(&self) -> T {
    self.base.std_dev()
  }

  /// The lower bound.
  pub fn lower(&self) -> T {
    self.lower
  }

  /// The upper bound.
  pub fn upper(&self) -> T {
    self.upper
  }

  /// Rejection on the base normal while the interval holds over 5 % of its mass (1000 tries), else the exact scheme
  /// the interval's position allows.
  #[inline]
  fn draw<S: TruncatedNormalSource<T>>(&self, src: &mut S) -> T {
    if self.norm_mass > 0.05 {
      for _ in 0..1000 {
        let x = src.base(&self.base);
        if x >= self.lower && x <= self.upper {
          return x;
        }
      }
    }
    match self.tail {
      Some(setup) => self.tail_sample(setup, src),
      None => self.inverse_cdf_sample(src),
    }
  }

  /// Robert (1995) accept-reject on the standardised one-sided interval, which forms no cdf: deep in a tail
  /// `F(upper) − F(lower)` loses its digits, and past about 8.3σ both round to 1 and inversion returns `+inf`.
  fn tail_sample<S: TruncatedNormalSource<T>>(&self, setup: TailSetup, src: &mut S) -> T {
    let z = match setup.proposal {
      TailProposal::Exponential { alpha_star } => loop {
        // `1 - u` keeps the generator's `[0, 1)` off the `ln`'s zero.
        let z = setup.a - (1.0 - src.uniform()).ln() / alpha_star;
        if z > setup.b {
          continue;
        }
        let d = z - alpha_star;
        if src.uniform() <= (-0.5 * d * d).exp() {
          break z;
        }
      },
      TailProposal::Uniform => loop {
        let z = setup.a + src.uniform() * (setup.b - setup.a);
        if src.uniform() <= (0.5 * (setup.a * setup.a - z * z)).exp() {
          break z;
        }
      },
    };
    let z = if setup.mirrored { -z } else { z };
    self.affine(z)
  }

  fn inverse_cdf_sample<S: TruncatedNormalSource<T>>(&self, src: &mut S) -> T {
    T::from_f64_fast(self.inverse_cdf(src.uniform()))
  }

  /// $F^{-1}(F(\text{lower}) + u (F(\text{upper}) - F(\text{lower})))$: the sampler's inverse-cdf map.
  fn inverse_cdf(&self, u: f64) -> f64 {
    self.unstandardise(crate::special::ndtri(
      self.f_lo + u * (self.f_up - self.f_lo),
    ))
  }

  #[inline]
  fn affine(&self, z: f64) -> T {
    T::from_f64_fast(self.unstandardise(z))
  }

  #[inline]
  fn unstandardise(&self, z: f64) -> f64 {
    self.base.mean().to_f64().unwrap() + self.base.std_dev().to_f64().unwrap() * z
  }

  fn standardise(&self, x: f64) -> Option<f64> {
    Some((x - self.base.mean().to_f64()?) / self.base.std_dev().to_f64()?)
  }

  /// An infinite bound stays infinite.
  fn standardised_bounds(&self) -> Option<(f64, f64)> {
    let (mu, sigma) = (self.base.mean().to_f64()?, self.base.std_dev().to_f64()?);
    Some((
      (self.lower.to_f64()? - mu) / sigma,
      (self.upper.to_f64()? - mu) / sigma,
    ))
  }
}

impl<T: SimdFloatExt> Sealed for SimdTruncatedNormal<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdTruncatedNormal<T> {
  type State<R: SimdRngExt> = TruncatedNormalState<T, R>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (TruncatedNormalState<T, R>, u64) {
    let (base, basis) = self.base.init::<R, S>(seed);
    let rng = R::from_seed(seed.next_seed());
    (TruncatedNormalState { base, rng }, basis)
  }
}

impl<T: SimdFloatExt> SimdKernel for SimdTruncatedNormal<T> {
  type Item = T;

  fn fill<R: SimdRngExt>(&self, state: &mut TruncatedNormalState<T, R>, out: &mut [T]) {
    for x in out.iter_mut() {
      *x = self.draw(state);
    }
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut TruncatedNormalState<T, R>) -> T {
    self.draw(state)
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdTruncatedNormal<T> {
  /// The same rejection, tail or inverse-cdf draw, on scalar normals and uniforms from the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.draw(&mut AnyRng(rng))
  }
}

impl<T: SimdFloatExt> DistributionExt for SimdTruncatedNormal<T> {
  fn pdf(&self, x: f64) -> Option<f64> {
    if x < self.lower.to_f64()? || x > self.upper.to_f64()? {
      return Some(0.0);
    }
    Some(self.analytics()?.density(self.standardise(x)?) * self.inv_sigma)
  }

  fn cdf(&self, x: f64) -> Option<f64> {
    if x <= self.lower.to_f64()? {
      return Some(0.0);
    }
    if x >= self.upper.to_f64()? {
      return Some(1.0);
    }
    Some(self.analytics()?.cdf(self.standardise(x)?))
  }

  fn quantile(&self, p: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&p) {
      return Some(f64::NAN);
    }
    let (lo, up) = (self.lower.to_f64()?, self.upper.to_f64()?);
    // The bounds exactly, which the unstandardised map misses by a few ulps.
    if p == 0.0 {
      return Some(lo);
    }
    if p == 1.0 {
      return Some(up);
    }
    Some(
      self
        .unstandardise(self.analytics()?.quantile(p))
        .clamp(lo, up),
    )
  }

  fn mean(&self) -> Option<f64> {
    Some(self.unstandardise(self.analytics()?.moments()[0]))
  }

  fn median(&self) -> Option<f64> {
    self.quantile(0.5)
  }

  fn mode(&self) -> Option<f64> {
    let mu = self.base.mean().to_f64()?;
    Some(mu.clamp(self.lower.to_f64()?, self.upper.to_f64()?))
  }

  fn variance(&self) -> Option<f64> {
    let sigma = self.base.std_dev().to_f64()?;
    Some(sigma * sigma * self.analytics()?.moments()[1])
  }

  fn skewness(&self) -> Option<f64> {
    Some(self.analytics()?.moments()[2])
  }

  fn kurtosis(&self) -> Option<f64> {
    Some(self.analytics()?.moments()[3])
  }

  fn entropy(&self) -> Option<f64> {
    Some(self.base.std_dev().to_f64()?.ln() + self.analytics()?.entropy())
  }

  /// The shifted interval's mass over the interval's, both as `exp(ln ∫k)`: finite as far as `e^{μt + σ²t²/2}` is.
  fn moment_generating_function(&self, t: f64) -> Option<f64> {
    if t.is_nan() {
      return Some(f64::NAN);
    }
    let (mu, sigma) = (self.base.mean().to_f64()?, self.base.std_dev().to_f64()?);
    let (alpha, beta) = self.standardised_bounds()?;
    let shift = sigma * t;
    let (lo, up) = (alpha - shift, beta - shift);
    let exponent = if shift.is_finite() && lo < up {
      let shifted = Truncation::mass_only(NormalKernel, lo, up);
      mu * t + 0.5 * shift * shift + shifted.ln_mass_over(self.analytics()?)
    } else {
      f64::NAN
    };
    if exponent.is_nan() || exponent == f64::NEG_INFINITY {
      // A shift past ~6e8σ leaves the shifted kernel inside one ulp: `e^{t·upper}` or `e^{t·lower}` decides.
      let end = if t > 0.0 { self.upper } else { self.lower };
      return Some(if end.to_f64()? * t > 0.0 {
        f64::INFINITY
      } else {
        0.0
      });
    }
    Some(exponent.exp())
  }
}

fn norm_cdf_scalar(z: f64) -> f64 {
  0.5 * (1.0 + crate::special::erf(z / std::f64::consts::SQRT_2))
}

#[cfg(test)]
mod tests;

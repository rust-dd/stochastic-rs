//! Truncated normal: rejection on the base normal, Robert's one-sided tail proposals, or the inverse cdf.
//!
//! Reference: Johnson, N. L., Kotz, S. & Balakrishnan, N. (1994), *Continuous Univariate Distributions*, vol. 1, 2nd ed., Wiley, ch. 13 §10.1, ISBN 0-471-58495-9 (truncated normal moments).

use std::fmt;
use std::sync::OnceLock;

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use super::local::Local;
use super::local::Model;
use super::local::Signed;
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
use kernels::NormalKernel;
use located::standardised;
use located::width;
use sampler::TailSetup;

mod kernels;
mod located;
mod mgf;
mod sampler;

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
  /// The standardised bounds and, off `μ`, the width `(up − lo)/σ` (NaN about `μ` or past the doubles).
  bounds: [f64; 3],
  /// The point a density's offset is read from and its standardised double: on one side of `μ` the near bound, which
  /// the engine's reference point is held beside, else `μ`.
  origin: (f64, f64),
  /// The standardised interval's kernel integral, built by the first analytic call; construction and draws skip it.
  analytics: OnceLock<Model<NormalKernel>>,
  /// `ln σ`, so a density costs no logarithm of its own.
  ln_sigma: f64,
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
    let (lo, up) = (lower.to_f64().unwrap(), upper.to_f64().unwrap());
    let (a_std, b_std) = (
      standardised(lo, mu_f64, sigma_f64),
      standardised(up, mu_f64, sigma_f64),
    );
    let f_lo = norm_cdf_scalar(a_std);
    let f_up = norm_cdf_scalar(b_std);
    let w = width(a_std, b_std, lo, up, sigma_f64);
    // A left-tail interval is reflected into the right tail, so the tail sampler only sees `a >= 0`.
    let tail = if a_std >= 0.0 {
      Some(TailSetup::new(a_std, b_std, w, false))
    } else if b_std <= 0.0 {
      Some(TailSetup::new(-b_std, -a_std, w, true))
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
      bounds: [a_std, b_std, w],
      origin: match tail {
        Some(setup) if setup.mirrored => (up, b_std),
        Some(_) => (lo, a_std),
        None => (mu_f64, 0.0),
      },
      analytics: OnceLock::new(),
      ln_sigma: sigma_f64.ln(),
    }
  }

  #[inline]
  fn analytics(&self) -> Option<&Model<NormalKernel>> {
    match self.analytics.get() {
      Some(m) => Some(m),
      None => self.fill(),
    }
  }

  /// The first analytic call's integral, out of the hot path.
  #[cold]
  #[inline(never)]
  fn fill(&self) -> Option<&Model<NormalKernel>> {
    let (lo, up) = self.held();
    Some(self.analytics.get_or_init(|| {
      // Past `μ ± σ·f64::MAX` the standardised near bound is infinite: the law is its local model at that bound.
      if self.beyond().is_some()
        && let Some(l) = self.local()
      {
        return Model::Local(l);
      }
      Model::of(Truncation::new(NormalKernel::default(), lo, up), |_| {
        self.local()
      })
    }))
  }

  /// The local model at `r = clamp(μ)`: slope `(μ − r)/σ²` and curvature `−1/σ²`, in logs.
  fn local(&self) -> Option<Local> {
    let (mu, sigma) = (self.base.mean().to_f64()?, self.base.std_dev().to_f64()?);
    let (lo, up) = (self.lower.to_f64()?, self.upper.to_f64()?);
    let r = mu.clamp(lo, up);
    let ln_var = 2.0 * sigma.ln();
    let slope = if (mu - r).is_finite() {
      Signed::over(mu - r, ln_var)
    } else {
      Signed::over(0.5 * mu - 0.5 * r, ln_var - std::f64::consts::LN_2)
    };
    // The log-kernel is its quadratic: no remainder.
    Local::new(
      r,
      (lo - r, up - r),
      slope,
      Signed::new(-1.0, -ln_var),
      |_| f64::NEG_INFINITY,
    )
  }

  /// `(mean, variance, skewness, excess kurtosis)`.
  fn moments(&self) -> Option<[f64; 4]> {
    Some(match self.analytics()? {
      Model::Full(t) => {
        let (frame, offset, [variance, skewness, kurtosis]) =
          t.located_moments(self.base.std_dev().to_f64()?);
        [
          self.mean_from(t, frame, offset)?,
          variance,
          skewness,
          kurtosis,
        ]
      }
      Model::Local(l) => l.moments(),
    })
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
  /// In logs with `σ`, so a subnormal `σ` meets the density's 0 at an open end before `1/σ` overflows.
  #[inline]
  fn pdf(&self, x: f64) -> Option<f64> {
    if x < self.lower.to_f64()? || x > self.upper.to_f64()? {
      return Some(0.0);
    }
    Some(match self.analytics()? {
      Model::Full(t) => {
        let (from, at) = self.origin;
        let z = standardised(x, from, self.base.std_dev().to_f64()?);
        t.density_from(z, at, self.ln_sigma)
      }
      Model::Local(l) => local_pdf(l, x),
    })
  }

  fn cdf(&self, x: f64) -> Option<f64> {
    if x <= self.lower.to_f64()? {
      return Some(0.0);
    }
    if x >= self.upper.to_f64()? {
      return Some(1.0);
    }
    Some(match self.analytics()? {
      Model::Full(t) => match self.zone_offset(x).and_then(|offset| t.zone_cdf(offset)) {
        Some(f) => f,
        None => t.cdf(self.cut(x)?),
      },
      Model::Local(l) => l.cdf(x),
    })
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
    let x = match self.analytics()? {
      Model::Full(t) => {
        let x = self.located(t.quantile(p))?;
        // A point whose standardised offset the doubles hold only as a coarse subnormal: the engine's zone, in logs.
        match t.zone_quantile(p).filter(|_| self.zone_offset(x).is_some()) {
          Some(point) => self.located(point)?,
          None => x,
        }
      }
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
    let mu = self.base.mean().to_f64()?;
    Some(mu.clamp(self.lower.to_f64()?, self.upper.to_f64()?))
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
      Model::Full(t) => self.base.std_dev().to_f64()?.ln() + t.entropy(),
      Model::Local(l) => l.entropy(),
    })
  }

  fn moment_generating_function(&self, t: f64) -> Option<f64> {
    self.mgf(t)
  }
}

/// Out of line, so the common path's pdf inlines.
#[cold]
#[inline(never)]
fn local_pdf(l: &Local, x: f64) -> f64 {
  l.pdf(x, 0.0)
}

fn norm_cdf_scalar(z: f64) -> f64 {
  0.5 * (1.0 + crate::special::erf(z / std::f64::consts::SQRT_2))
}

#[cfg(test)]
mod tests;

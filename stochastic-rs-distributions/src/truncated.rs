//! # Truncated distributions
//!
//! Truncated $\mathrm{Normal}$, $\mathrm{Beta}$, $\mathrm{Gamma}$ and
//! $\mathrm{Exponential}$ distributions restricted to a closed interval
//! $[a, b]$ (with $-\infty \le a < b \le +\infty$ allowed per family).
//!
//! ## Sampling
//!
//! - **Truncated Normal:** plain rejection from the base
//!   $\mathcal{N}(\mu, \sigma^2)$ via the existing [`SimdNormal`] sampler
//!   while acceptance stays above 5 %. Tight intervals (mass $< 0.05$,
//!   where rejection would spin) split by where they sit: one lying wholly
//!   on one side of the mean takes Robert's accept-reject on the
//!   standardised interval, which never forms a CDF and so cannot lose the
//!   interval's mass to rounding; one straddling the mean takes the
//!   closed-form inverse-CDF transform, exact where the normal CDF still
//!   resolves, at one [`crate::special::ndtri`] call per draw.
//! - **Truncated Exponential:** closed-form inverse-CDF sampling on the
//!   survival function — no rejection needed.
//! - **Truncated Beta / Gamma:** plain rejection from the corresponding
//!   [`SimdBeta`] / [`SimdGamma`] sampler. For very tight intervals where
//!   acceptance falls below 1 % the rejection loop bails after 1000 tries
//!   and returns the clamped midpoint — the caller should widen the
//!   bounds in that regime (the boundary itself is hit with measure zero).
//!
//! ## Density
//!
//! Standard normalisation: $f_{\[a,b\]}(x) = f(x) / (F(b) - F(a))$ for
//! $x \in [a, b]$ and $0$ elsewhere. The CDF normalising constant is
//! cached at construction time.
//!
//! References:
//! - Devroye, L. (1986), *Non-Uniform Random Variate Generation*,
//!   Springer, §II.3 (general rejection, the wide-interval path for all
//!   four families here).
//! - Robert, C.P. (1995), "Simulation of truncated normal variables",
//!   *Statistics and Computing* 5(2), 121-125, DOI: 10.1007/BF00143942
//!   (the two one-sided proposals the tail path picks between).

use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::beta::SimdBeta;
use crate::gamma::SimdGamma;
use crate::normal::SimdNormal;
use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;
use crate::traits::DistributionExt;
use crate::traits::SimdFloatExt;

/// Which of Robert (1995)'s proposals a one-sided standardised interval
/// $[a, b]$ with $a \ge 0$ takes.
#[derive(Debug, Clone, Copy)]
enum TailProposal {
  /// Translated exponential of rate $\alpha^\*$ (Robert 1995, §2.2), for an
  /// interval at least two proposal means wide.
  Exponential { alpha_star: f64 },
  /// Uniform on $[a, b]$ reweighted by $e^{(a^2 - z^2)/2}$ (Robert 1995,
  /// §2.3), for a narrower one, where the exponential's own truncation
  /// rejection would dominate.
  Uniform,
}

/// The standardised one-sided interval a tail draw runs on.
#[derive(Debug, Clone, Copy)]
struct TailSetup {
  /// Standardised bounds with $0 \le a < b \le \infty$.
  a: f64,
  b: f64,
  /// Whether the interval was reflected about the mean to get there; the
  /// draw is negated back on the way out.
  mirrored: bool,
  proposal: TailProposal,
}

impl TailSetup {
  fn new(a: f64, b: f64, mirrored: bool) -> Self {
    let alpha_star = 0.5 * (a + (a * a + 4.0).sqrt());
    // Two proposal means of width is where the exponential's truncation
    // rejection stops costing more than the uniform's Gaussian weight:
    // below it the exponential overshoots `b` more often than not, above
    // it the uniform's worst weight has already decayed past e^-2.
    let proposal = if b - a >= 2.0 / alpha_star {
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

/// Truncated normal $\mathcal{N}(\mu, \sigma^2)$ restricted to
/// $[\text{lower}, \text{upper}]$.
pub struct SimdTruncatedNormal<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  mean: T,
  std_dev: T,
  lower: T,
  upper: T,
  base: SimdNormal<T, 64, R>,
  /// Cached CDF values $F(\text{lower})$ / $F(\text{upper})$ and their
  /// difference — the normalising constant of the density and the affine
  /// map of the inverse-CDF fallback in tight intervals.
  f_lo: f64,
  f_up: f64,
  norm_mass: f64,
  /// Set when the interval lies wholly on one side of the mean, where the
  /// CDF pair the inverse transform needs is the one that loses its
  /// significance first.
  tail: Option<TailSetup>,
  simd_rng: UnsafeCell<R>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdTruncatedNormal<T, R> {
  /// Create a truncated normal.
  ///
  /// - `mean` — location μ of the untruncated base [`SimdNormal`].
  /// - `std_dev` — scale σ > 0 of the untruncated base.
  /// - `lower`, `upper` — truncation interval bounds (`lower < upper`);
  ///   the base's own μ/σ are unchanged, only the support is restricted
  ///   and renormalised.
  pub fn new<S: SeedExt>(mean: T, std_dev: T, lower: T, upper: T, seed: &S) -> Self {
    assert!(std_dev > T::zero(), "std_dev must be positive");
    assert!(lower < upper, "lower bound must be < upper bound");
    let mean_f64 = mean.to_f64().unwrap();
    let std_f64 = std_dev.to_f64().unwrap();
    let a_std = (lower.to_f64().unwrap() - mean_f64) / std_f64;
    let b_std = (upper.to_f64().unwrap() - mean_f64) / std_f64;
    let f_lo = norm_cdf_scalar(a_std);
    let f_up = norm_cdf_scalar(b_std);
    // Reflect a left-tail interval into the right tail so the tail sampler
    // only ever has to handle `a >= 0`; one straddling the mean keeps the
    // inverse transform, which is accurate exactly where `Φ` is.
    let tail = if a_std >= 0.0 {
      Some(TailSetup::new(a_std, b_std, false))
    } else if b_std <= 0.0 {
      Some(TailSetup::new(-b_std, -a_std, true))
    } else {
      None
    };
    Self {
      mean,
      std_dev,
      lower,
      upper,
      base: SimdNormal::<T, 64, R>::new(mean, std_dev, seed),
      f_lo,
      f_up,
      norm_mass: f_up - f_lo,
      tail,
      simd_rng: UnsafeCell::new(seed.rng_ext::<R>()),
    }
  }

  /// Draw a single truncated normal sample using the internal SIMD RNG.
  ///
  /// On wide intervals (acceptance ≥ 5 %) we use plain rejection on the
  /// base normal sampler; on tight intervals we route to whichever of the
  /// two exact schemes the interval's position allows.
  #[inline]
  pub fn sample_fast(&self) -> T {
    if self.norm_mass > 0.05 {
      // Plain rejection — fast path.
      for _ in 0..1000 {
        let x = self.base.sample_fast();
        if x >= self.lower && x <= self.upper {
          return x;
        }
      }
      // Fall through if we somehow had a 1000-shot run of rejections
      // (numerical edge cases).
    }
    match self.tail {
      Some(setup) => self.tail_sample(setup),
      None => self.inverse_cdf_sample(),
    }
  }

  /// Robert (1995) accept-reject on the standardised one-sided interval.
  ///
  /// The inverse transform cannot serve this case: `F(lower)` and
  /// `F(upper)` both round to 1 once the interval sits past about four
  /// standard deviations, their difference loses every digit it had, and
  /// `ndtri` of the saturated quantile is `+inf` — 17 % of the draws on
  /// `[8σ, 8.5σ]`, all of them on `[10σ, 12σ]`. Neither proposal here ever
  /// forms a CDF, so the interval's mass never has to be representable as
  /// a difference of two numbers near 1.
  fn tail_sample(&self, setup: TailSetup) -> T {
    let rng = unsafe { &mut *self.simd_rng.get() };
    let z = match setup.proposal {
      TailProposal::Exponential { alpha_star } => loop {
        // `1 - u` keeps the generator's `[0, 1)` off the `ln`'s zero.
        let z = setup.a - (1.0 - rng.next_f64()).ln() / alpha_star;
        if z > setup.b {
          continue;
        }
        let d = z - alpha_star;
        if rng.next_f64() <= (-0.5 * d * d).exp() {
          break z;
        }
      },
      TailProposal::Uniform => loop {
        let z = setup.a + rng.next_f64() * (setup.b - setup.a);
        if rng.next_f64() <= (0.5 * (setup.a * setup.a - z * z)).exp() {
          break z;
        }
      },
    };
    let z = if setup.mirrored { -z } else { z };
    T::from_f64_fast(self.mean.to_f64().unwrap() + self.std_dev.to_f64().unwrap() * z)
  }

  /// Inverse-CDF sample: $X = F^{-1}(F(\text{lower}) + U \cdot (F(\text{upper}) - F(\text{lower})))$.
  fn inverse_cdf_sample(&self) -> T {
    let rng = unsafe { &mut *self.simd_rng.get() };
    let q = self.f_lo + rng.next_f64() * (self.f_up - self.f_lo);
    let z = crate::special::ndtri(q);
    T::from_f64_fast(self.mean.to_f64().unwrap() + self.std_dev.to_f64().unwrap() * z)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdTruncatedNormal<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.mean, self.std_dev, self.lower, self.upper, &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Distribution<T> for SimdTruncatedNormal<T, R> {
  fn sample<Rr: Rng + ?Sized>(&self, _rng: &mut Rr) -> T {
    self.sample_fast()
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> DistributionExt for SimdTruncatedNormal<T, R> {
  fn pdf(&self, x: f64) -> f64 {
    let lo = self.lower.to_f64().unwrap();
    let up = self.upper.to_f64().unwrap();
    if x < lo || x > up {
      return 0.0;
    }
    let mean = self.mean.to_f64().unwrap();
    let std = self.std_dev.to_f64().unwrap();
    let z = (x - mean) / std;
    let phi = (-0.5 * z * z).exp() / ((2.0 * std::f64::consts::PI).sqrt() * std);
    phi / self.norm_mass
  }

  fn cdf(&self, x: f64) -> f64 {
    let lo = self.lower.to_f64().unwrap();
    let up = self.upper.to_f64().unwrap();
    if x <= lo {
      return 0.0;
    }
    if x >= up {
      return 1.0;
    }
    let mean = self.mean.to_f64().unwrap();
    let std = self.std_dev.to_f64().unwrap();
    let f_x = norm_cdf_scalar((x - mean) / std);
    let f_lo = norm_cdf_scalar((lo - mean) / std);
    (f_x - f_lo) / self.norm_mass
  }
}

/// Truncated exponential $\mathrm{Exp}(\lambda)$ restricted to
/// $[\text{lower}, \text{upper}]$ ($\text{lower} \ge 0$). Closed-form
/// inverse-CDF sampling — no rejection needed.
pub struct SimdTruncatedExp<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  lambda: T,
  lower: T,
  upper: T,
  /// $e^{-\lambda(\text{upper} - \text{lower})}$ — the survival at the upper
  /// bound relative to the lower one, and the only summary of the interval
  /// the draw and the density need. Zero when `upper` is infinite.
  tail_ratio: f64,
  simd_rng: UnsafeCell<R>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdTruncatedExp<T, R> {
  /// Create a truncated exponential.
  ///
  /// - `lambda` — rate λ > 0 of the untruncated base Exp(λ).
  /// - `lower`, `upper` — truncation interval bounds (`0 ≤ lower < upper`,
  ///   `upper` may be infinite).
  pub fn new<S: SeedExt>(lambda: T, lower: T, upper: T, seed: &S) -> Self {
    assert!(lambda > T::zero(), "lambda must be positive");
    assert!(lower >= T::zero(), "lower bound must be ≥ 0");
    assert!(lower < upper, "lower < upper");
    let lam = lambda.to_f64().unwrap();
    let lo = lower.to_f64().unwrap();
    let up = upper.to_f64().unwrap();
    Self {
      lambda,
      lower,
      upper,
      tail_ratio: (-lam * (up - lo)).exp(),
      simd_rng: UnsafeCell::new(seed.rng_ext::<R>()),
    }
  }

  /// Closed-form inverse-CDF draw, written on the survival function rather
  /// than the CDF: $X = \text{lower} - \ln(V)/\lambda$ with $V$ uniform on
  /// $(e^{-\lambda(\text{upper}-\text{lower})}, 1]$.
  ///
  /// The CDF form $F(x) = 1 - e^{-\lambda x}$ saturates at 1 as soon as
  /// $\lambda \cdot \text{lower}$ passes about 36: both bounds round to the
  /// same double, the normalising mass comes out zero, and every draw
  /// collapses onto whatever the guard against $\ln 0$ happened to be —
  /// 690.78, a value outside the interval entirely. Shifting the origin to
  /// `lower` leaves a difference of positives that keeps its significance
  /// wherever the interval itself is representable.
  #[inline]
  pub fn sample_fast(&self) -> T {
    let rng = unsafe { &mut *self.simd_rng.get() };
    // `1 - u` maps the generator's `[0, 1)` onto `(0, 1]`, so `v` stays
    // strictly above the tail ratio and the log is never `-inf`.
    let v = 1.0 - rng.next_f64() * (1.0 - self.tail_ratio);
    T::from_f64_fast(self.lower.to_f64().unwrap() - v.ln() / self.lambda.to_f64().unwrap())
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdTruncatedExp<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.lambda, self.lower, self.upper, &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Distribution<T> for SimdTruncatedExp<T, R> {
  fn sample<Rr: Rng + ?Sized>(&self, _rng: &mut Rr) -> T {
    self.sample_fast()
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> DistributionExt for SimdTruncatedExp<T, R> {
  fn pdf(&self, x: f64) -> f64 {
    let lo = self.lower.to_f64().unwrap();
    let up = self.upper.to_f64().unwrap();
    if x < lo || x > up {
      return 0.0;
    }
    let lam = self.lambda.to_f64().unwrap();
    // Referred to `lower` for the same reason the draw is; the leading
    // `e^{-λ·lower}` cancels between the density and the interval's mass.
    lam * (-lam * (x - lo)).exp() / (1.0 - self.tail_ratio)
  }

  fn cdf(&self, x: f64) -> f64 {
    let lo = self.lower.to_f64().unwrap();
    let up = self.upper.to_f64().unwrap();
    if x <= lo {
      return 0.0;
    }
    if x >= up {
      return 1.0;
    }
    let lam = self.lambda.to_f64().unwrap();
    (1.0 - (-lam * (x - lo)).exp()) / (1.0 - self.tail_ratio)
  }
}

/// Truncated Beta restricted to $[\text{lower}, \text{upper}] \subseteq [0, 1]$.
/// Rejection from the base [`SimdBeta`]; emits the clamped midpoint after
/// 1000 unsuccessful tries (the boundary itself is hit with measure zero).
pub struct SimdTruncatedBeta<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  alpha: T,
  beta: T,
  lower: T,
  upper: T,
  base: SimdBeta<T, R>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdTruncatedBeta<T, R> {
  /// Create a truncated beta.
  ///
  /// - `alpha`, `beta` — shape parameters of the untruncated base
  ///   [`SimdBeta`] (both > 0), matching `SimdBeta::new`'s own roles.
  /// - `lower`, `upper` — truncation interval bounds, both in [0, 1]
  ///   with `lower < upper`.
  pub fn new<S: SeedExt>(alpha: T, beta: T, lower: T, upper: T, seed: &S) -> Self {
    assert!(alpha > T::zero() && beta > T::zero(), "α, β > 0");
    let lo = lower.to_f64().unwrap();
    let up = upper.to_f64().unwrap();
    assert!(
      (0.0..=1.0).contains(&lo) && (0.0..=1.0).contains(&up),
      "bounds must lie in [0,1]"
    );
    assert!(lower < upper, "lower < upper");
    Self {
      alpha,
      beta,
      lower,
      upper,
      base: SimdBeta::<T, R>::new(alpha, beta, seed),
    }
  }

  #[inline]
  pub fn sample_fast(&self) -> T {
    for _ in 0..1000 {
      let x = self.base.sample_fast();
      if x >= self.lower && x <= self.upper {
        return x;
      }
    }
    (self.lower + self.upper) * T::from_f64_fast(0.5)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdTruncatedBeta<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.alpha, self.beta, self.lower, self.upper, &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Distribution<T> for SimdTruncatedBeta<T, R> {
  fn sample<Rr: Rng + ?Sized>(&self, _rng: &mut Rr) -> T {
    self.sample_fast()
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> DistributionExt for SimdTruncatedBeta<T, R> {
  fn pdf(&self, x: f64) -> f64 {
    let lo = self.lower.to_f64().unwrap();
    let up = self.upper.to_f64().unwrap();
    if x < lo || x > up {
      return 0.0;
    }
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let log_norm =
      crate::special::ln_gamma(a + b) - crate::special::ln_gamma(a) - crate::special::ln_gamma(b);
    let log_kernel = (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln();
    let base_pdf = (log_norm + log_kernel).exp();
    let norm_mass = self.cdf_helper(up) - self.cdf_helper(lo);
    base_pdf / norm_mass.max(1e-300)
  }

  fn cdf(&self, x: f64) -> f64 {
    let lo = self.lower.to_f64().unwrap();
    let up = self.upper.to_f64().unwrap();
    if x <= lo {
      return 0.0;
    }
    if x >= up {
      return 1.0;
    }
    let f_x = self.cdf_helper(x);
    let f_lo = self.cdf_helper(lo);
    let f_up = self.cdf_helper(up);
    (f_x - f_lo) / (f_up - f_lo).max(1e-300)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdTruncatedBeta<T, R> {
  fn cdf_helper(&self, x: f64) -> f64 {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    crate::special::beta_i(a, b, x.clamp(0.0, 1.0))
  }
}

/// Truncated Gamma$(k, \theta)$ restricted to $[\text{lower}, \text{upper}]$,
/// $\text{lower} \ge 0$. Rejection from base [`SimdGamma`] with the same
/// 1000-attempt fallback policy as the Beta case.
pub struct SimdTruncatedGamma<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  shape: T,
  scale: T,
  lower: T,
  upper: T,
  base: SimdGamma<T, R>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdTruncatedGamma<T, R> {
  /// Create a truncated gamma.
  ///
  /// - `shape` — shape k > 0 of the untruncated base [`SimdGamma`] (the
  ///   same role `SimdGamma::new` calls `alpha` — this wrapper uses the
  ///   `Gamma(k, θ)` letter instead).
  /// - `scale` — scale θ > 0 of the untruncated base (matches
  ///   `SimdGamma::new`'s own `scale`).
  /// - `lower`, `upper` — truncation interval bounds (`0 ≤ lower < upper`).
  pub fn new<S: SeedExt>(shape: T, scale: T, lower: T, upper: T, seed: &S) -> Self {
    assert!(shape > T::zero(), "shape > 0");
    assert!(scale > T::zero(), "scale > 0");
    assert!(lower >= T::zero(), "lower ≥ 0");
    assert!(lower < upper, "lower < upper");
    Self {
      shape,
      scale,
      lower,
      upper,
      base: SimdGamma::<T, R>::new(shape, scale, seed),
    }
  }

  #[inline]
  pub fn sample_fast(&self) -> T {
    for _ in 0..1000 {
      let x = self.base.sample_fast();
      if x >= self.lower && x <= self.upper {
        return x;
      }
    }
    (self.lower + self.upper) * T::from_f64_fast(0.5)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdTruncatedGamma<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.shape, self.scale, self.lower, self.upper, &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Distribution<T> for SimdTruncatedGamma<T, R> {
  fn sample<Rr: Rng + ?Sized>(&self, _rng: &mut Rr) -> T {
    self.sample_fast()
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> DistributionExt for SimdTruncatedGamma<T, R> {
  fn pdf(&self, x: f64) -> f64 {
    let lo = self.lower.to_f64().unwrap();
    let up = self.upper.to_f64().unwrap();
    if x < lo || x > up {
      return 0.0;
    }
    let k = self.shape.to_f64().unwrap();
    let theta = self.scale.to_f64().unwrap();
    let log_norm = -crate::special::ln_gamma(k) - k * theta.ln();
    let log_kernel = (k - 1.0) * x.ln() - x / theta;
    let base_pdf = (log_norm + log_kernel).exp();
    let mass = self.cdf_helper(up) - self.cdf_helper(lo);
    base_pdf / mass.max(1e-300)
  }

  fn cdf(&self, x: f64) -> f64 {
    let lo = self.lower.to_f64().unwrap();
    let up = self.upper.to_f64().unwrap();
    if x <= lo {
      return 0.0;
    }
    if x >= up {
      return 1.0;
    }
    let f_x = self.cdf_helper(x);
    let f_lo = self.cdf_helper(lo);
    let f_up = self.cdf_helper(up);
    (f_x - f_lo) / (f_up - f_lo).max(1e-300)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdTruncatedGamma<T, R> {
  fn cdf_helper(&self, x: f64) -> f64 {
    let k = self.shape.to_f64().unwrap();
    let theta = self.scale.to_f64().unwrap();
    crate::special::gamma_p(k, x / theta)
  }
}

/// Standard-normal CDF used by the truncated-normal CDF / normalisation
/// helpers. Mirrors the shape used in `crate::special` but kept local so
/// the truncated module is self-contained.
fn norm_cdf_scalar(z: f64) -> f64 {
  0.5 * (1.0 + crate::special::erf(z / std::f64::consts::SQRT_2))
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  /// Truncated normal samples must respect the bound.
  #[test]
  fn truncated_normal_samples_within_bounds() {
    let tn = SimdTruncatedNormal::<f64>::new(0.0, 1.0, -1.0, 2.0, &Unseeded);
    for _ in 0..5_000 {
      let x = tn.sample_fast();
      assert!((-1.0..=2.0).contains(&x), "sample {x} out of [-1, 2]");
    }
  }

  /// Truncated normal density integrates to 1 in the bulk (mid-point Riemann
  /// sum on a 1000-step grid).
  #[test]
  fn truncated_normal_pdf_normalised() {
    let tn = SimdTruncatedNormal::<f64>::new(0.0, 1.0, -1.0, 2.0, &Unseeded);
    let n = 1_000;
    let h = 3.0 / n as f64;
    let s: f64 = (0..n)
      .map(|k| tn.pdf(-1.0 + (k as f64 + 0.5) * h) * h)
      .sum();
    assert!(
      (s - 1.0).abs() < 5e-3,
      "truncated normal pdf integrates to {s}, expected 1"
    );
  }

  /// Truncated exponential: closed-form CDF must round-trip to inputs.
  #[test]
  fn truncated_exp_cdf_round_trips() {
    let te = SimdTruncatedExp::<f64>::new(2.0, 0.0, 1.5, &Unseeded);
    for x in [0.0, 0.3, 0.7, 1.0, 1.5] {
      let f = te.cdf(x);
      assert!((0.0..=1.0).contains(&f));
    }
    assert_eq!(te.cdf(-0.1), 0.0);
    assert_eq!(te.cdf(2.0), 1.0);
  }

  /// Truncated exponential samples in bounds with the right approximate
  /// mean (closed-form check on tight [0, 0.5] band of Exp(1)).
  #[test]
  fn truncated_exp_samples_mean() {
    let te = SimdTruncatedExp::<f64>::new(1.0, 0.0, 0.5, &Unseeded);
    let n = 20_000;
    let mut sum = 0.0;
    for _ in 0..n {
      let x = te.sample_fast();
      assert!((0.0..=0.5).contains(&x));
      sum += x;
    }
    let mean = sum / n as f64;
    // Closed-form mean of Truncated Exp(1) on [0, 0.5]:
    //   E[X | 0 ≤ X ≤ 0.5] = ∫₀^0.5 x · e^{-x} dx / (1 - e^{-0.5})
    //                       = [1 - 1.5 · e^{-0.5}] / (1 - e^{-0.5})
    let half = 0.5_f64;
    let expected = (1.0 - 1.5 * (-half).exp()) / (1.0 - (-half).exp());
    assert!(
      (mean - expected).abs() < 0.01,
      "truncated Exp(1) mean = {mean}, expected ≈ {expected}"
    );
  }

  /// A far-tail interval keeps its draws inside its bounds and keeps the
  /// right conditional mean.
  ///
  /// The inverse transform cannot reach here: `F(lower)` and `F(upper)`
  /// both round to 1 past about four standard deviations, their difference
  /// loses every digit, and `ndtri` of the saturated quantile is `+inf` —
  /// 17 % of the draws on [8, 8.5], every one of them on [10, 12] and on
  /// its mirror image. The reference mean comes from Simpson quadrature on
  /// the conditional density of `Z - a`, which is proportional to
  /// `exp(-a t - t^2 / 2)` and so never has to form a CDF either.
  #[test]
  fn truncated_normal_far_tail_stays_in_bounds() {
    for (lo, hi) in [(8.0_f64, 8.5_f64), (10.0, 12.0), (-12.0, -10.0)] {
      let tn = SimdTruncatedNormal::<f64>::new(0.0, 1.0, lo, hi, &Deterministic::new(2718));
      let n = 40_000;
      let mut sum = 0.0;
      for _ in 0..n {
        let x = tn.sample_fast();
        assert!((lo..=hi).contains(&x), "sample {x} out of [{lo}, {hi}]");
        sum += x;
      }
      let a = lo.abs().min(hi.abs());
      let width = hi - lo;
      let (mut num, mut den) = (0.0, 0.0);
      let steps = 4_000;
      let h = width / steps as f64;
      for k in 0..=steps {
        let t = k as f64 * h;
        let w = if k == 0 || k == steps {
          1.0
        } else if k % 2 == 1 {
          4.0
        } else {
          2.0
        };
        let f = (-a * t - 0.5 * t * t).exp();
        num += w * t * f;
        den += w * f;
      }
      let want = (a + num / den) * lo.signum();
      let mean = sum / n as f64;
      assert!(
        (mean - want).abs() < 0.01,
        "[{lo}, {hi}]: mean = {mean}, expected ≈ {want}"
      );
    }
  }

  /// A far interval keeps its truncated-exponential draws inside its bounds.
  ///
  /// `F(x) = 1 - e^{-λx}` saturates at 1 once `λ · lower` passes about 36:
  /// both bounds round to the same double, the normalising mass comes out
  /// zero, and every draw collapsed onto 690.78 — the `-ln(1e-300)` the old
  /// guard against `ln 0` left behind, outside the interval entirely.
  #[test]
  fn truncated_exp_far_interval_stays_in_bounds() {
    for (lam, lo, hi) in [(1.0_f64, 50.0_f64, 60.0_f64), (100.0, 0.5, 0.6)] {
      let te = SimdTruncatedExp::<f64>::new(lam, lo, hi, &Deterministic::new(2718));
      let n = 40_000;
      let mut sum = 0.0;
      for _ in 0..n {
        let x = te.sample_fast();
        assert!((lo..=hi).contains(&x), "sample {x} out of [{lo}, {hi}]");
        sum += x;
      }
      // E[X] = lower + 1/λ − (upper − lower)·r/(1 − r), r = e^{−λ(upper−lower)}.
      let r = (-lam * (hi - lo)).exp();
      let want = lo + 1.0 / lam - (hi - lo) * r / (1.0 - r);
      let mean = sum / n as f64;
      assert!(
        (mean - want).abs() < 5.0 / lam / (n as f64).sqrt(),
        "lambda = {lam} on [{lo}, {hi}]: mean = {mean}, expected ≈ {want}"
      );
    }
  }

  /// Truncated Beta in [0.2, 0.8] respects bounds and has uniform-like
  /// support across many samples.
  #[test]
  fn truncated_beta_samples_within_bounds() {
    let tb = SimdTruncatedBeta::<f64>::new(2.0, 2.0, 0.2, 0.8, &Unseeded);
    for _ in 0..3_000 {
      let x = tb.sample_fast();
      assert!((0.2..=0.8).contains(&x));
    }
  }

  /// Truncated Gamma stays in bounds.
  #[test]
  fn truncated_gamma_samples_within_bounds() {
    let tg = SimdTruncatedGamma::<f64>::new(2.0, 1.0, 1.0, 5.0, &Unseeded);
    for _ in 0..3_000 {
      let x = tg.sample_fast();
      assert!((1.0..=5.0).contains(&x));
    }
  }

  /// PDF / CDF degenerate cases: outside the bounds must produce 0 PDF
  /// and {0, 1} CDF.
  #[test]
  fn truncated_pdf_cdf_outside_bounds() {
    let tn = SimdTruncatedNormal::<f64>::new(0.0, 1.0, -1.0, 1.0, &Unseeded);
    assert_eq!(tn.pdf(-1.5), 0.0);
    assert_eq!(tn.pdf(1.5), 0.0);
    assert_eq!(tn.cdf(-2.0), 0.0);
    assert_eq!(tn.cdf(2.0), 1.0);
  }
}

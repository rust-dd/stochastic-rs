//! Truncated normal: rejection on the base normal, Robert's one-sided tail proposals, or the inverse cdf.
//!
//! Reference: Johnson, N. L., Kotz, S. & Balakrishnan, N. (1994), *Continuous Univariate Distributions*, vol. 1, 2nd ed., Wiley, ch. 13 §10.1, ISBN 0-471-58495-9 (truncated normal moments).
//! Reference: Kan, R. & Robotti, C. (2017), "On Moments of Folded and Truncated Multivariate Normal Distributions", *Journal of Computational and Graphical Statistics* 26(4), 930-934, DOI 10.1080/10618600.2017.1322092 (the moment recursion at n = 1 and the mgf).

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use crate::normal::SimdNormal;
use crate::seeded::StreamState;
use crate::source::AnyRng;
use crate::source::uniform53;
use crate::traits::DistributionExt;
use crate::traits::SimdFloatExt;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

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
    let alpha_star = 0.5 * (a + (a * a + 4.0).sqrt());
    // Not Robert's faster rule (2.1): switching would move the pinned `truncated_normal_uniform_tail_f64` stream.
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

/// Truncated normal law $\mathcal{N}(\mu, \sigma^2)$ on $[\text{lower}, \text{upper}]$: parameters and the interval's
/// cdf constants; a [`Seeded`](crate::Seeded) stream draws it.
///
/// Robert, C.P. (1995), "Simulation of truncated normal variables", *Statistics and Computing* 5(2), 121-125, DOI 10.1007/BF00143942.
/// Devroye, L. (1986), *Non-Uniform Random Variate Generation*, Springer, §II.3, DOI 10.1007/978-1-4613-8643-8.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdTruncatedNormal<T> {
  base: SimdNormal<T>,
  lower: T,
  upper: T,
  /// $F(\text{lower})$, $F(\text{upper})$ and their difference: the density's normalising mass and the inverse-cdf map.
  f_lo: f64,
  f_up: f64,
  norm_mass: f64,
  /// Set when the interval lies wholly on one side of `μ`, where the inverse cdf loses its digits first.
  tail: Option<TailSetup>,
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
    }
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

  /// $F^{-1}(F(\text{lower}) + u (F(\text{upper}) - F(\text{lower})))$, shared by the inverse-cdf draw and the quantile.
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

  /// An infinite bound stays infinite.
  fn standardised_bounds(&self) -> Option<(f64, f64)> {
    let (mu, sigma) = (self.base.mean().to_f64()?, self.base.std_dev().to_f64()?);
    Some((
      (self.lower.to_f64()? - mu) / sigma,
      (self.upper.to_f64()? - mu) / sigma,
    ))
  }

  /// $z^k \varphi(z)$, and `0` at an infinite bound, where `∞ · 0` would read NaN.
  fn tail_term(z: f64, k: i32) -> f64 {
    if z.is_infinite() {
      0.0
    } else {
      z.powi(k) * crate::special::norm_pdf(z)
    }
  }

  /// Standardised raw moments $E[Z^k]$, $k \le 4$, by the integration-by-parts recursion
  /// $E[Z^k] = (k-1)E[Z^{k-2}] + (\alpha^{k-1}\varphi(\alpha) - \beta^{k-1}\varphi(\beta))/Z$.
  fn standardised_moments(&self) -> Option<[f64; 5]> {
    let (alpha, beta) = self.standardised_bounds()?;
    let term = |k: i32| (Self::tail_term(alpha, k) - Self::tail_term(beta, k)) / self.norm_mass;
    let mut m = [1.0, term(0), 0.0, 0.0, 0.0];
    for k in 2..=4 {
      m[k] = (k as f64 - 1.0) * m[k - 2] + term(k as i32 - 1);
    }
    Some(m)
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
    let lo = self.lower.to_f64().unwrap();
    let up = self.upper.to_f64().unwrap();
    if x < lo || x > up {
      return Some(0.0);
    }
    let mu = self.base.mean().to_f64().unwrap();
    let sigma = self.base.std_dev().to_f64().unwrap();
    let z = (x - mu) / sigma;
    let phi = (-0.5 * z * z).exp() / ((2.0 * std::f64::consts::PI).sqrt() * sigma);
    Some(phi / self.norm_mass)
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
    let mu = self.base.mean().to_f64().unwrap();
    let sigma = self.base.std_dev().to_f64().unwrap();
    let f_x = norm_cdf_scalar((x - mu) / sigma);
    let f_lo = norm_cdf_scalar((lo - mu) / sigma);
    Some((f_x - f_lo) / self.norm_mass)
  }

  fn quantile(&self, p: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&p) {
      return Some(f64::NAN);
    }
    Some(self.inverse_cdf(p))
  }

  fn mean(&self) -> Option<f64> {
    Some(self.unstandardise(self.standardised_moments()?[1]))
  }

  fn median(&self) -> Option<f64> {
    self.quantile(0.5)
  }

  fn mode(&self) -> Option<f64> {
    let mu = self.base.mean().to_f64()?;
    Some(mu.clamp(self.lower.to_f64()?, self.upper.to_f64()?))
  }

  fn variance(&self) -> Option<f64> {
    let m = self.standardised_moments()?;
    let sigma = self.base.std_dev().to_f64()?;
    Some(sigma * sigma * (m[2] - m[1] * m[1]))
  }

  fn skewness(&self) -> Option<f64> {
    let m = self.standardised_moments()?;
    let var = m[2] - m[1] * m[1];
    Some((m[3] - 3.0 * m[1] * m[2] + 2.0 * m[1].powi(3)) / var.powf(1.5))
  }

  fn kurtosis(&self) -> Option<f64> {
    let m = self.standardised_moments()?;
    let var = m[2] - m[1] * m[1];
    Some(
      (m[4] - 4.0 * m[1] * m[3] + 6.0 * m[1] * m[1] * m[2] - 3.0 * m[1].powi(4)) / (var * var)
        - 3.0,
    )
  }

  /// `−E[ln f(X)] = ln(√(2π)σZ) + E[Z²]/2`, with `E[Z²]` from the moment recursion.
  fn entropy(&self) -> Option<f64> {
    let m = self.standardised_moments()?;
    let sigma = self.base.std_dev().to_f64()?;
    Some(((2.0 * std::f64::consts::PI).sqrt() * sigma * self.norm_mass).ln() + 0.5 * m[2])
  }

  fn moment_generating_function(&self, t: f64) -> Option<f64> {
    let (mu, sigma) = (self.base.mean().to_f64()?, self.base.std_dev().to_f64()?);
    let (alpha, beta) = self.standardised_bounds()?;
    let shifted = norm_cdf_scalar(beta - sigma * t) - norm_cdf_scalar(alpha - sigma * t);
    Some((mu * t + 0.5 * sigma * sigma * t * t).exp() * shifted / self.norm_mass)
  }
}

fn norm_cdf_scalar(z: f64) -> f64 {
  0.5 * (1.0 + crate::special::erf(z / std::f64::consts::SQRT_2))
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_core::simd_rng::Unseeded;

  use super::*;
  use crate::tests::scalar_draws;
  use crate::tests::scalar_ks_best_p;

  /// Truncated normal samples must respect the bound.
  #[test]
  fn truncated_normal_samples_within_bounds() {
    let mut tn = SimdTruncatedNormal::<f64>::new(0.0, 1.0, -1.0, 2.0).seeded(&Unseeded);
    for _ in 0..5_000 {
      let x = tn.sample();
      assert!((-1.0..=2.0).contains(&x), "sample {x} out of [-1, 2]");
    }
  }

  /// Truncated normal density integrates to 1 in the bulk (mid-point Riemann
  /// sum on a 1000-step grid).
  #[test]
  fn truncated_normal_pdf_normalised() {
    let tn = SimdTruncatedNormal::<f64>::new(0.0, 1.0, -1.0, 2.0);
    let n = 1_000;
    let h = 3.0 / n as f64;
    let s: f64 = (0..n)
      .map(|k| tn.pdf(-1.0 + (k as f64 + 0.5) * h).unwrap() * h)
      .sum();
    assert!(
      (s - 1.0).abs() < 5e-3,
      "truncated normal pdf integrates to {s}, expected 1"
    );
  }

  /// Draws in bounds and a conditional mean against Simpson quadrature of `exp(-a t - t²/2)`, the density of `Z - a`,
  /// which forms no cdf: the inverse transform put 17 % of [8, 8.5] and all of [10, 12] at `+inf`.
  fn assert_far_tail_means(mut draws: impl FnMut(SimdTruncatedNormal<f64>) -> Vec<f64>) {
    for (lo, hi) in [(8.0_f64, 8.5_f64), (10.0, 12.0), (-12.0, -10.0)] {
      let xs = draws(SimdTruncatedNormal::<f64>::new(0.0, 1.0, lo, hi));
      for x in &xs {
        assert!((lo..=hi).contains(x), "sample {x} out of [{lo}, {hi}]");
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
      let mean = xs.iter().sum::<f64>() / xs.len() as f64;
      assert!(
        (mean - want).abs() < 0.01,
        "[{lo}, {hi}]: mean = {mean}, expected ≈ {want}"
      );
    }
  }

  /// A far-tail interval keeps its draws inside its bounds and keeps the right conditional mean.
  #[test]
  fn truncated_normal_far_tail_stays_in_bounds() {
    assert_far_tail_means(|d| {
      let mut s = d.seeded(&Deterministic::new(2718));
      (0..40_000).map(|_| s.sample()).collect()
    });
  }

  /// The honest draw keeps the same far-tail bounds and conditional means on the caller's rng.
  #[test]
  fn scalar_far_tail_stays_in_bounds() {
    assert_far_tail_means(|d| scalar_draws(&d, 2718, 40_000));
  }

  /// The honest `Distribution` agrees with the cdf on the rejection, both tail proposals, the mirror and the inverse cdf.
  #[test]
  fn scalar_sample_matches_cdf() {
    for (lo, hi) in [
      (-1.0, 2.0),
      (3.0, 6.0),
      (3.0, 3.4),
      (-6.0, -3.0),
      (-0.05, 0.05),
    ] {
      let d = SimdTruncatedNormal::<f64>::new(0.0, 1.0, lo, hi);
      let best = scalar_ks_best_p(&d, |x| d.cdf(x).unwrap());
      assert!(best > 0.01, "[{lo}, {hi}]: best p = {best}");
    }
  }

  /// PDF / CDF degenerate cases: outside the bounds must produce 0 PDF
  /// and {0, 1} CDF.
  #[test]
  fn truncated_pdf_cdf_outside_bounds() {
    let tn = SimdTruncatedNormal::<f64>::new(0.0, 1.0, -1.0, 1.0);
    assert_eq!(tn.pdf(-1.5).unwrap(), 0.0);
    assert_eq!(tn.pdf(1.5).unwrap(), 0.0);
    assert_eq!(tn.cdf(-2.0).unwrap(), 0.0);
    assert_eq!(tn.cdf(2.0).unwrap(), 1.0);
  }

  /// `mu()` and `sigma()` read back the parent's arguments, not moments of the truncated law.
  #[test]
  fn getters_return_the_parent_parameters() {
    let tn = SimdTruncatedNormal::<f64>::new(0.5, 2.0, 1.0, 4.0);
    assert_eq!(
      (tn.mu(), tn.sigma(), tn.lower(), tn.upper()),
      (0.5, 2.0, 1.0, 4.0)
    );
  }
}

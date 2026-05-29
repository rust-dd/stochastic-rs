//! # Generalized Extreme Value (GEV) distribution
//!
//! Jenkinson (1955) unification of the three Fisher-Tippett extreme-value
//! sub-families into a single three-parameter family:
//!
//! $$
//! F(x;\mu,\sigma,\xi) =
//! \begin{cases}
//!   \exp\!\Big(-\bigl(1 + \xi\,\tfrac{x - \mu}{\sigma}\bigr)^{-1/\xi}\Big),
//!     & \xi \neq 0,\ 1 + \xi(x-\mu)/\sigma > 0 \\[4pt]
//!   \exp\!\Big(-\exp\!\bigl(-(x-\mu)/\sigma\bigr)\Big),
//!     & \xi = 0
//! \end{cases}
//! $$
//!
//! with $\mu \in \mathbb{R}$ (location), $\sigma > 0$ (scale), $\xi \in
//! \mathbb{R}$ (shape).
//!
//! - $\xi > 0$: **Fréchet** (Type II) — heavy upper tail, no upper bound.
//! - $\xi = 0$: **Gumbel** (Type I) — light exponential tail.
//! - $\xi < 0$: **Reverse Weibull** (Type III) — bounded upper tail at
//!   $\mu - \sigma/\xi$.
//!
//! Used in extreme-value-theory (EVT) risk modelling (Value-at-Risk on
//! the tail block-maxima) — see McNeil-Frey-Embrechts (2015) ch. 7.2.
//!
//! ## Sampling
//!
//! Closed-form inverse CDF:
//!
//! $$
//! X = \begin{cases}
//!   \mu - \dfrac{\sigma}{\xi}\bigl(1 - (-\ln U)^{-\xi}\bigr), & \xi \neq 0 \\[4pt]
//!   \mu - \sigma\,\ln(-\ln U), & \xi = 0
//! \end{cases},
//! \qquad U \sim \mathrm{Uniform}(0, 1).
//! $$
//!
//! References:
//! - Jenkinson, A.F. (1955), "The frequency distribution of the annual
//!   maximum (or minimum) values of meteorological elements",
//!   *Quarterly Journal of the Royal Meteorological Society* 81, 158-171.
//! - Coles, S. (2001), *An Introduction to Statistical Modeling of
//!   Extreme Values*, Springer.
//! - McNeil, A.J., Frey, R., Embrechts, P. (2015),
//!   *Quantitative Risk Management*, Princeton UP, §7.2.

use rand::Rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;
use crate::traits::DistributionExt;
use crate::traits::SimdFloatExt;

/// Generalized Extreme Value distribution. Three free parameters: location
/// `μ`, scale `σ > 0`, shape `ξ`.
///
/// The struct stores the parameters; sampling and density use the
/// closed-form inverse-CDF / PDF derived in the module docs.
pub struct SimdGev<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  mu: T,
  sigma: T,
  xi: T,
  _seed: std::marker::PhantomData<R>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdGev<T, R> {
  /// Construct a GEV$(\mu, \sigma, \xi)$.
  pub fn new<S: SeedExt>(mu: T, sigma: T, xi: T, _seed: &S) -> Self {
    assert!(sigma > T::zero(), "σ must be positive");
    Self {
      mu,
      sigma,
      xi,
      _seed: std::marker::PhantomData,
    }
  }

  /// Single sample via inverse-CDF on an independent `Uniform(0, 1)`
  /// draw from the thread RNG.
  #[inline]
  pub fn sample_fast(&self) -> T {
    let mut rng = rand::rng();
    let u: f64 = rng.random_range(1e-12..1.0 - 1e-12);
    let mu = self.mu.to_f64().unwrap();
    let sigma = self.sigma.to_f64().unwrap();
    let xi = self.xi.to_f64().unwrap();
    let m_ln_u = -u.ln();
    let x = if xi.abs() < 1e-12 {
      // Gumbel limit ξ → 0: X = μ - σ · ln(-ln U).
      mu - sigma * m_ln_u.ln()
    } else {
      mu - (sigma / xi) * (1.0 - m_ln_u.powf(-xi))
    };
    T::from_f64_fast(x)
  }

  /// Closed-form support: returns `(lo, hi)` as the open interval on
  /// which the GEV density is strictly positive. Used by callers that
  /// need to clip samples or build empirical histograms.
  pub fn support(&self) -> (f64, f64) {
    let mu = self.mu.to_f64().unwrap();
    let sigma = self.sigma.to_f64().unwrap();
    let xi = self.xi.to_f64().unwrap();
    if xi.abs() < 1e-12 {
      (f64::NEG_INFINITY, f64::INFINITY)
    } else if xi > 0.0 {
      (mu - sigma / xi, f64::INFINITY)
    } else {
      (f64::NEG_INFINITY, mu - sigma / xi)
    }
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdGev<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.mu, self.sigma, self.xi, &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Distribution<T> for SimdGev<T, R> {
  fn sample<Rr: Rng + ?Sized>(&self, _rng: &mut Rr) -> T {
    self.sample_fast()
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> DistributionExt for SimdGev<T, R> {
  fn pdf(&self, x: f64) -> f64 {
    let mu = self.mu.to_f64().unwrap();
    let sigma = self.sigma.to_f64().unwrap();
    let xi = self.xi.to_f64().unwrap();
    let z = (x - mu) / sigma;
    if xi.abs() < 1e-12 {
      let m_z = -z;
      (-z - m_z.exp()).exp() / sigma
    } else {
      let t = 1.0 + xi * z;
      if t <= 0.0 {
        return 0.0;
      }
      let inv_xi = 1.0 / xi;
      let t_inv_xi = t.powf(-inv_xi);
      let t_pow = t.powf(-inv_xi - 1.0);
      (1.0 / sigma) * t_pow * (-t_inv_xi).exp()
    }
  }

  fn cdf(&self, x: f64) -> f64 {
    let mu = self.mu.to_f64().unwrap();
    let sigma = self.sigma.to_f64().unwrap();
    let xi = self.xi.to_f64().unwrap();
    let z = (x - mu) / sigma;
    if xi.abs() < 1e-12 {
      (-(-z).exp()).exp()
    } else {
      let t = 1.0 + xi * z;
      if t <= 0.0 {
        return if xi > 0.0 { 0.0 } else { 1.0 };
      }
      (-(t.powf(-1.0 / xi))).exp()
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Gumbel (ξ = 0): mean = μ + γσ (Euler-Mascheroni γ ≈ 0.5772157).
  /// Sample mean must match within 3σ on 30k draws.
  #[test]
  fn gev_gumbel_sample_mean_euler() {
    let g = SimdGev::<f64>::new(0.0, 1.0, 0.0, &Unseeded);
    let n = 30_000;
    let mut sum = 0.0;
    for _ in 0..n {
      sum += g.sample_fast();
    }
    let mean = sum / n as f64;
    let euler = 0.577_215_664_901_532_9_f64;
    assert!(
      (mean - euler).abs() < 0.05,
      "Gumbel mean = {mean}, expected ≈ γ ≈ {euler}"
    );
  }

  /// Fréchet (ξ > 0): heavy upper tail; mean is finite only when ξ < 1.
  /// Use ξ = 0.5 → E[X] = μ + σ(Γ(1 - ξ) - 1)/ξ.
  #[test]
  fn gev_frechet_sample_mean_closed_form() {
    let xi = 0.5_f64;
    let g = SimdGev::<f64>::new(0.0, 1.0, xi, &Unseeded);
    let n = 30_000;
    let mut sum = 0.0;
    for _ in 0..n {
      sum += g.sample_fast();
    }
    let mean = sum / n as f64;
    let gamma_term: f64 = crate::special::ln_gamma(1.0 - xi).exp();
    let expected = (gamma_term - 1.0) / xi;
    // Heavy-tail variance is ∞ for ξ ∈ [0.5, 1), so allow a 15% band on the mean.
    assert!(
      (mean - expected).abs() / expected.abs() < 0.15,
      "Fréchet(ξ=0.5) sample mean = {mean}, expected ≈ {expected}"
    );
  }

  /// PDF integrates to 1 within the support (numerical Riemann).
  #[test]
  fn gev_pdf_normalised_gumbel() {
    let g = SimdGev::<f64>::new(0.0, 1.0, 0.0, &Unseeded);
    let n = 5000usize;
    let lo = -10.0_f64;
    let up = 30.0_f64;
    let h = (up - lo) / n as f64;
    let s: f64 = (0..n).map(|k| g.pdf(lo + (k as f64 + 0.5) * h) * h).sum();
    assert!((s - 1.0).abs() < 1e-3, "Gumbel PDF integrates to {s}");
  }

  /// CDF matches inverse-CDF identity: F(F⁻¹(u)) = u on a grid.
  #[test]
  fn gev_cdf_inverse_round_trip() {
    let g = SimdGev::<f64>::new(0.0, 1.0, 0.2, &Unseeded);
    for u in [0.1_f64, 0.3, 0.5, 0.7, 0.9] {
      // X = μ - σ/ξ · (1 - (-ln U)^{-ξ})
      let m_ln_u = -u.ln();
      let x = -(1.0 / 0.2) * (1.0 - m_ln_u.powf(-0.2));
      let f = g.cdf(x);
      assert!((f - u).abs() < 1e-10, "F({x}) = {f}, expected {u}");
    }
  }

  /// Support edge: Reverse Weibull (ξ < 0) is bounded above by μ - σ/ξ.
  #[test]
  fn gev_reverse_weibull_bounded_support() {
    let xi = -0.5_f64;
    let g = SimdGev::<f64>::new(0.0, 1.0, xi, &Unseeded);
    let (lo, hi) = g.support();
    assert_eq!(lo, f64::NEG_INFINITY);
    assert_eq!(hi, -1.0 / xi); // = 2.0
    for _ in 0..2_000 {
      let x = g.sample_fast();
      assert!(
        x <= hi + 1e-9,
        "Reverse Weibull sample {x} exceeds bound {hi}"
      );
    }
  }
}

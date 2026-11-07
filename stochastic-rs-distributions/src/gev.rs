//! # Generalized Extreme Value (GEV) distribution
//!
//! Jenkinson (1955) unification of the three Fisher-Tippett extreme-value
//! sub-families into a single three-parameter family:
//!
//! $$
//! F(x;\mu,\sigma,\xi) =
//! \begin{cases}
//!   \exp\!\Big(-\bigl(1 + \xi\,\tfrac{x - \mu}{\sigma}\bigr)^{-1/\xi}\Big),
//!     & \xi \neq 0,\ 1 + \xi(x-\mu)/\sigma > 0 \\\\\[4pt\]
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
//!   \mu + \dfrac{\sigma}{\xi}\,\mathrm{expm1}\bigl(-\xi\ln(-\ln U)\bigr), & \xi \neq 0 \\\\\[4pt\]
//!   \mu - \sigma\,\ln(-\ln U), & \xi = 0
//! \end{cases},
//! \qquad U \sim \mathrm{Uniform}(0, 1).
//! $$
//!
//! References:
//! - Jenkinson, A.F. (1955), "The frequency distribution of the annual
//!   maximum (or minimum) values of meteorological elements",
//!   *Quarterly Journal of the Royal Meteorological Society* 81, 158-171.
//!   DOI: 10.1002/qj.49708134804
//! - Coles, S. (2001), *An Introduction to Statistical Modeling of
//!   Extreme Values*, Springer.
//! - McNeil, A.J., Frey, R., Embrechts, P. (2015),
//!   *Quantitative Risk Management*, Princeton UP, §7.2.

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use crate::seeded::StreamState;
use crate::special::SHAPE_CUT;
use crate::special::ln_1p_over_xi;
use crate::special::pow_m1_over_xi;
use crate::traits::DistributionExt;
use crate::traits::SimdFloatExt;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

mod series;

const SMALL_GEV_THRESHOLD: usize = 16;

/// Euler–Mascheroni $\gamma_E$, the Gumbel mean.
const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;

/// Generalized Extreme Value distribution. Three free parameters: location
/// `μ`, scale `σ > 0`, shape `ξ`; a [`Seeded`](crate::Seeded) stream draws it in bulk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdGev<T> {
  mu: T,
  sigma: T,
  xi: T,
}

impl<T: SimdFloatExt> SimdGev<T> {
  /// Construct a GEV$(\mu, \sigma, \xi)$.
  ///
  /// - `mu` — location μ (matches the module header's μ).
  /// - `sigma` — scale σ > 0 (matches the module header's σ).
  /// - `xi` — shape ξ (matches the module header's ξ); sign selects
  ///   Fréchet (ξ>0), Gumbel (ξ=0), or reverse-Weibull (ξ<0).
  pub fn new(mu: T, sigma: T, xi: T) -> Self {
    assert!(
      mu.is_finite(),
      "mu must satisfy `mu.is_finite()`, got mu = {mu:?}"
    );
    assert!(
      sigma.is_finite(),
      "sigma must satisfy `sigma.is_finite()`, got sigma = {sigma:?}"
    );
    assert!(
      xi.is_finite(),
      "xi must satisfy `xi.is_finite()`, got xi = {xi:?}"
    );
    assert!(
      sigma > T::zero(),
      "sigma must satisfy `sigma > T::zero()`, got sigma = {sigma:?}"
    );
    Self { mu, sigma, xi }
  }

  /// The location `μ`.
  pub fn mu(&self) -> T {
    self.mu
  }

  /// The scale `σ`.
  pub fn sigma(&self) -> T {
    self.sigma
  }

  /// The shape `ξ`.
  pub fn xi(&self) -> T {
    self.xi
  }

  /// Clamp a uniform draw to the open unit interval so the `ln` chain stays
  /// finite at the lane level (mirrors the `1e-12` guard of the scalar path).
  #[inline]
  fn clamp_open_unit(x: T) -> T {
    let eps = T::from_f64_fast(1e-12);
    x.max(eps).min(T::one() - eps)
  }

  #[inline]
  fn is_gumbel(&self) -> bool {
    self.xi.abs() < T::from_f64_fast(SHAPE_CUT)
  }

  #[inline]
  fn invert(&self, u: T, gumbel: bool) -> T {
    let u = Self::clamp_open_unit(u);
    let ln_m_ln_u = (-u.ln()).ln();
    if gumbel {
      self.mu - self.sigma * ln_m_ln_u
    } else {
      self.mu + self.sigma * pow_m1_over_xi(self.xi, ln_m_ln_u)
    }
  }

  fn fill_parts<R: SimdRngExt>(&self, rng: &mut R, out: &mut [T]) {
    let gumbel = self.is_gumbel();
    if out.len() < SMALL_GEV_THRESHOLD {
      for x in out.iter_mut() {
        *x = self.invert(T::sample_uniform_simd(rng), gumbel);
      }
      return;
    }
    let mu = T::splat(self.mu);
    let mut u = [T::zero(); 8];
    let (chunks, rem) = out.as_chunks_mut::<8>();
    for chunk in chunks {
      T::fill_uniform_simd(rng, &mut u);
      for x in u.iter_mut() {
        *x = Self::clamp_open_unit(*x);
      }
      let ln_m_ln_u = T::simd_ln(-T::simd_ln(T::simd_from_array(u)));
      let x = if gumbel {
        mu - T::splat(self.sigma) * ln_m_ln_u
      } else {
        // `σ` enters last: above the cut `σ/ξ` overflows once `σ > 1.8e289` (`3.4e19` in `f32`).
        let rise = T::simd_exp_m1(T::splat(-self.xi) * ln_m_ln_u) * T::splat(self.xi.recip());
        mu + T::splat(self.sigma) * rise
      };
      *chunk = T::simd_to_array(x);
    }
    for x in rem.iter_mut() {
      *x = self.invert(T::sample_uniform_simd(rng), gumbel);
    }
  }

  pub(crate) fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.invert(T::sample_uniform(rng), self.is_gumbel())
  }

  fn params(&self) -> (f64, f64, f64) {
    (
      self.mu.to_f64().unwrap(),
      self.sigma.to_f64().unwrap(),
      self.xi.to_f64().unwrap(),
    )
  }

  /// Closed-form support: returns `(lo, hi)` as the open interval on
  /// which the GEV density is strictly positive. Used by callers that
  /// need to clip samples or build empirical histograms.
  pub fn support(&self) -> (f64, f64) {
    let mu = self.mu.to_f64().unwrap();
    let sigma = self.sigma.to_f64().unwrap();
    let xi = self.xi.to_f64().unwrap();
    if xi == 0.0 {
      (f64::NEG_INFINITY, f64::INFINITY)
    } else if xi > 0.0 {
      (mu - sigma / xi, f64::INFINITY)
    } else {
      (f64::NEG_INFINITY, mu - sigma / xi)
    }
  }
}

impl<T: SimdFloatExt> Sealed for SimdGev<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdGev<T> {
  type State<R: SimdRngExt> = StreamState<T, R, 16>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (StreamState<T, R, 16>, u64) {
    StreamState::init(seed)
  }
}

impl<T: SimdFloatExt> SimdKernel for SimdGev<T> {
  type Item = T;

  #[inline]
  fn fill<R: SimdRngExt>(&self, state: &mut StreamState<T, R, 16>, out: &mut [T]) {
    self.fill_parts(&mut state.rng, out);
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut StreamState<T, R, 16>) -> T {
    let StreamState { rng, buf } = state;
    buf.pop(|b| self.fill_parts(rng, b))
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdGev<T> {
  /// The inverse cdf at one `[0, 1)` uniform from the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.draw_with(rng)
  }
}

impl<T: SimdFloatExt> DistributionExt for SimdGev<T> {
  fn pdf(&self, x: f64) -> Option<f64> {
    let (mu, sigma, xi) = self.params();
    let z = (x - mu) / sigma;
    if z.is_infinite() || xi * z <= -1.0 {
      return Some(0.0);
    }
    let w = ln_1p_over_xi(xi, z);
    Some((-(1.0 + xi) * w - (-w).exp()).exp() / sigma)
  }

  fn cdf(&self, x: f64) -> Option<f64> {
    let (mu, sigma, xi) = self.params();
    let z = (x - mu) / sigma;
    if z.is_infinite() || xi * z <= -1.0 {
      return Some(if z > 0.0 { 1.0 } else { 0.0 });
    }
    Some((-(-ln_1p_over_xi(xi, z)).exp()).exp())
  }

  fn quantile(&self, p: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&p) {
      return Some(f64::NAN);
    }
    if p == 0.0 {
      return Some(self.support().0);
    }
    if p == 1.0 {
      return Some(self.support().1);
    }
    let (mu, sigma, xi) = self.params();
    Some(mu + sigma * pow_m1_over_xi(xi, (-p.ln()).ln()))
  }

  /// $\mu + \sigma(\Gamma(1-\xi) - 1)/\xi$ for $\xi < 1$; `+∞` for $\xi \ge 1$.
  fn mean(&self) -> Option<f64> {
    let (mu, sigma, xi) = self.params();
    if xi < 1.0 {
      Some(mu + sigma * series::mean(xi))
    } else {
      Some(f64::INFINITY)
    }
  }

  fn median(&self) -> Option<f64> {
    let (mu, sigma, xi) = self.params();
    Some(mu + sigma * pow_m1_over_xi(xi, std::f64::consts::LN_2.ln()))
  }

  /// From ξ = −1 down the density rises to the upper end of the support, `μ − σ/ξ`.
  fn mode(&self) -> Option<f64> {
    let (mu, sigma, xi) = self.params();
    Some(if xi <= -1.0 {
      mu - sigma / xi
    } else {
      mu + sigma * pow_m1_over_xi(xi, xi.ln_1p())
    })
  }

  /// $\sigma^2(g_2 - g_1^2)/\xi^2$ with $g_k = \Gamma(1 - k\xi)$ for $\xi < 1/2$; `+∞` for $\xi \ge 1/2$.
  fn variance(&self) -> Option<f64> {
    let (_, sigma, xi) = self.params();
    if xi < 0.5 {
      Some(sigma * sigma * series::variance(xi))
    } else {
      Some(f64::INFINITY)
    }
  }

  /// `NaN` for $\xi \ge 1/3$, where the third moment is undefined.
  fn skewness(&self) -> Option<f64> {
    let (_, _, xi) = self.params();
    if xi < 1.0 / 3.0 {
      Some(series::skewness(xi))
    } else {
      Some(f64::NAN)
    }
  }

  /// Excess kurtosis; `NaN` for $\xi \ge 1/4$.
  fn kurtosis(&self) -> Option<f64> {
    let (_, _, xi) = self.params();
    if xi < 0.25 {
      Some(series::kurtosis(xi))
    } else {
      Some(f64::NAN)
    }
  }

  /// $\log\sigma + \gamma_E(\xi + 1) + 1$.
  fn entropy(&self) -> Option<f64> {
    let (_, sigma, xi) = self.params();
    Some(sigma.ln() + EULER_MASCHERONI * (xi + 1.0) + 1.0)
  }
}

#[cfg(test)]
mod tests;

py_distribution!(PyGev, SimdGev,
  sig: (mu, sigma, xi, seed=None, dtype=None),
  params: (mu: f64, sigma: f64, xi: f64)
);

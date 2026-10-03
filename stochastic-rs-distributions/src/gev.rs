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
//!   \mu - \dfrac{\sigma}{\xi}\bigl(1 - (-\ln U)^{-\xi}\bigr), & \xi \neq 0 \\\\\[4pt\]
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

use crate::seeded::Buffered;
use crate::seeded::StreamState;
use crate::traits::DistributionExt;
use crate::traits::SimdFloatExt;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

const SMALL_GEV_THRESHOLD: usize = 16;

/// Euler–Mascheroni $\gamma_E$, the Gumbel mean.
const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;

/// Apéry's constant $\zeta(3)$, in the Gumbel skewness $12\sqrt6\,\zeta(3)/\pi^3$.
const APERY: f64 = 1.202_056_903_159_594_3;

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
    assert!(sigma > T::zero(), "σ must be positive");
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
    self.xi.to_f64().unwrap().abs() < 1e-12
  }

  #[inline]
  fn invert(&self, u: T, gumbel: bool) -> T {
    let u = Self::clamp_open_unit(u);
    let m_ln_u = -u.ln();
    if gumbel {
      self.mu - self.sigma * m_ln_u.ln()
    } else {
      self.mu - (self.sigma / self.xi) * (T::one() - m_ln_u.powf(-self.xi))
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
    let one = T::splat(T::one());
    let mut u = [T::zero(); 8];
    let (chunks, rem) = out.as_chunks_mut::<8>();
    for chunk in chunks {
      T::fill_uniform_simd(rng, &mut u);
      for x in u.iter_mut() {
        *x = Self::clamp_open_unit(*x);
      }
      let m_ln_u = -T::simd_ln(T::simd_from_array(u));
      let x = if gumbel {
        mu - T::splat(self.sigma) * T::simd_ln(m_ln_u)
      } else {
        mu - T::splat(self.sigma / self.xi) * (one - T::simd_powf(m_ln_u, -self.xi))
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
    if xi.abs() < 1e-12 {
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
    let stream_seed = seed.next_seed();
    (
      StreamState {
        rng: R::from_seed(stream_seed),
        buf: Buffered::new(),
      },
      stream_seed,
    )
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

  fn inv_cdf(&self, p: f64) -> f64 {
    let (mu, sigma, xi) = self.params();
    let m_ln_p = -p.ln();
    if xi.abs() < 1e-12 {
      mu - sigma * m_ln_p.ln()
    } else {
      mu + sigma / xi * (m_ln_p.powf(-xi) - 1.0)
    }
  }

  /// $\mu + \sigma(\Gamma(1-\xi) - 1)/\xi$ for $\xi < 1$ ($\mu + \gamma_E\sigma$
  /// at $\xi = 0$); `+∞` for $\xi \ge 1$.
  fn mean(&self) -> f64 {
    let (mu, sigma, xi) = self.params();
    if xi.abs() < 1e-12 {
      mu + EULER_MASCHERONI * sigma
    } else if xi < 1.0 {
      mu + sigma * (crate::special::gamma(1.0 - xi) - 1.0) / xi
    } else {
      f64::INFINITY
    }
  }

  fn median(&self) -> f64 {
    let (mu, sigma, xi) = self.params();
    if xi.abs() < 1e-12 {
      mu - sigma * std::f64::consts::LN_2.ln()
    } else {
      mu + sigma * (std::f64::consts::LN_2.powf(-xi) - 1.0) / xi
    }
  }

  fn mode(&self) -> f64 {
    let (mu, sigma, xi) = self.params();
    if xi.abs() < 1e-12 {
      mu
    } else {
      mu + sigma * ((1.0 + xi).powf(-xi) - 1.0) / xi
    }
  }

  /// $\sigma^2(g_2 - g_1^2)/\xi^2$ with $g_k = \Gamma(1 - k\xi)$ for
  /// $\xi < 1/2$ ($\sigma^2\pi^2/6$ at $\xi = 0$); `+∞` for $\xi \ge 1/2$.
  fn variance(&self) -> f64 {
    let (_, sigma, xi) = self.params();
    if xi.abs() < 1e-12 {
      sigma * sigma * std::f64::consts::PI.powi(2) / 6.0
    } else if xi < 0.5 {
      let g1 = crate::special::gamma(1.0 - xi);
      let g2 = crate::special::gamma(1.0 - 2.0 * xi);
      sigma * sigma * (g2 - g1 * g1) / (xi * xi)
    } else {
      f64::INFINITY
    }
  }

  /// `NaN` for $\xi \ge 1/3$, where the third moment is undefined.
  fn skewness(&self) -> f64 {
    let (_, _, xi) = self.params();
    if xi.abs() < 1e-12 {
      12.0 * 6.0_f64.sqrt() * APERY / std::f64::consts::PI.powi(3)
    } else if xi < 1.0 / 3.0 {
      let g1 = crate::special::gamma(1.0 - xi);
      let g2 = crate::special::gamma(1.0 - 2.0 * xi);
      let g3 = crate::special::gamma(1.0 - 3.0 * xi);
      xi.signum() * (g3 - 3.0 * g2 * g1 + 2.0 * g1.powi(3)) / (g2 - g1 * g1).powf(1.5)
    } else {
      f64::NAN
    }
  }

  /// Excess kurtosis; `NaN` for $\xi \ge 1/4$.
  fn kurtosis(&self) -> f64 {
    let (_, _, xi) = self.params();
    if xi.abs() < 1e-12 {
      12.0 / 5.0
    } else if xi < 0.25 {
      let g1 = crate::special::gamma(1.0 - xi);
      let g2 = crate::special::gamma(1.0 - 2.0 * xi);
      let g3 = crate::special::gamma(1.0 - 3.0 * xi);
      let g4 = crate::special::gamma(1.0 - 4.0 * xi);
      (g4 - 4.0 * g3 * g1 + 6.0 * g2 * g1 * g1 - 3.0 * g1.powi(4)) / (g2 - g1 * g1).powi(2) - 3.0
    } else {
      f64::NAN
    }
  }

  /// $\log\sigma + \gamma_E(\xi + 1) + 1$.
  fn entropy(&self) -> f64 {
    let (_, sigma, xi) = self.params();
    sigma.ln() + EULER_MASCHERONI * (xi + 1.0) + 1.0
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Unseeded;

  use super::*;
  use crate::tests::scalar_ks_best_p;

  /// Gumbel (ξ = 0): mean = μ + γσ (Euler-Mascheroni γ ≈ 0.5772157).
  /// Sample mean must match within 3σ on 30k draws.
  #[test]
  fn gev_gumbel_sample_mean_euler() {
    let mut g = SimdGev::<f64>::new(0.0, 1.0, 0.0).seeded(&Unseeded);
    let n = 30_000;
    let mut sum = 0.0;
    for _ in 0..n {
      sum += g.sample();
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
    let mut g = SimdGev::<f64>::new(0.0, 1.0, xi).seeded(&Unseeded);
    let n = 30_000;
    let mut sum = 0.0;
    for _ in 0..n {
      sum += g.sample();
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

  #[test]
  fn scalar_sample_matches_cdf() {
    for xi in [0.0, 0.3, -0.3] {
      let d = SimdGev::<f64>::new(0.0, 1.0, xi);
      let best = scalar_ks_best_p(&d, |x| d.cdf(x));
      assert!(best > 0.01, "xi = {xi}: best p = {best}");
    }
  }

  /// PDF integrates to 1 within the support (numerical Riemann).
  #[test]
  fn gev_pdf_normalised_gumbel() {
    let g = SimdGev::<f64>::new(0.0, 1.0, 0.0);
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
    let g = SimdGev::<f64>::new(0.0, 1.0, 0.2);
    for u in [0.1_f64, 0.3, 0.5, 0.7, 0.9] {
      // X = μ - σ/ξ · (1 - (-ln U)^{-ξ})
      let m_ln_u = -u.ln();
      let x = -(1.0 / 0.2) * (1.0 - m_ln_u.powf(-0.2));
      let f = g.cdf(x);
      assert!((f - u).abs() < 1e-10, "F({x}) = {f}, expected {u}");
    }
  }

  /// Deterministic seeds must reproduce identical streams (the seed was
  /// silently ignored before the internal RNG landed).
  #[test]
  fn gev_deterministic_seed_reproduces_stream() {
    use stochastic_rs_core::simd_rng::Deterministic;
    let mut a = SimdGev::<f64>::new(0.5, 1.2, 0.3).seeded(&Deterministic::new(7));
    let mut b = SimdGev::<f64>::new(0.5, 1.2, 0.3).seeded(&Deterministic::new(7));
    for _ in 0..256 {
      assert_eq!(a.sample(), b.sample());
    }
  }

  /// Support edge: Reverse Weibull (ξ < 0) is bounded above by μ - σ/ξ.
  #[test]
  fn gev_reverse_weibull_bounded_support() {
    let xi = -0.5_f64;
    let g = SimdGev::<f64>::new(0.0, 1.0, xi);
    let (lo, hi) = g.support();
    assert_eq!(lo, f64::NEG_INFINITY);
    assert_eq!(hi, -1.0 / xi); // = 2.0
    let mut stream = g.seeded(&Unseeded);
    for _ in 0..2_000 {
      let x = stream.sample();
      assert!(
        x <= hi + 1e-9,
        "Reverse Weibull sample {x} exceeds bound {hi}"
      );
    }
  }
}

py_distribution!(PyGev, SimdGev,
  sig: (mu, sigma, xi, seed=None, dtype=None),
  params: (mu: f64, sigma: f64, xi: f64)
);

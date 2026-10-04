//! # Generalized Error Distribution (GED) — Subbotin family
//!
//! $$
//! f(x; \mu, \alpha, \beta) = \frac{\beta}{2\alpha\,\Gamma(1/\beta)}\,
//! \exp\!\left(-\left|\frac{x - \mu}{\alpha}\right|^\beta\right),
//! \qquad \alpha > 0,\ \beta > 0.
//! $$
//!
//! Shape parameter $\beta$ controls tail behaviour:
//! - $\beta = 1$ → Laplace (double exponential, heavy tails)
//! - $\beta = 2$ → Gaussian
//! - $\beta < 2$ → heavier-than-Gaussian (GARCH residuals, Nelson 1991)
//! - $\beta > 2$ → lighter-than-Gaussian (platykurtic)
//!
//! **Sampling.** The standard form $|X|^\beta \sim \mathrm{Gamma}(1/\beta, 1)$
//! gives the bijection
//!
//! $$
//! X = \alpha \cdot Y^{1/\beta} \cdot S + \mu,
//! \qquad Y \sim \mathrm{Gamma}(1/\beta, 1),\ S \sim \mathrm{Uniform}\{-1, +1\}.
//! $$
//!
//! Used by GARCH-type volatility models with heavy-tailed innovations
//! (Nelson 1991 EGARCH, Bollerslev 1987 GARCH-t variant).
//!
//! References:
//! - Subbotin, M.T. (1923), "On the law of frequency of error", *Matematicheskii Sbornik* 31, 296-301.
//! - Nelson, D. B. (1991), "Conditional Heteroskedasticity in Asset Returns: A New Approach", *Econometrica* 59(2), 347-370, DOI 10.2307/2938260 (the density and its variance).
//! - Nadarajah, S. (2005), "A generalized normal distribution", *Journal of Applied Statistics* 32(7), 685-694, DOI 10.1080/02664760500079464 (variance, kurtosis, Shannon entropy).

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use crate::gamma::GammaState;
use crate::gamma::SimdGamma;
use crate::seeded::Buffered;
use crate::traits::DistributionExt;
use crate::traits::SimdFloatExt;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

const SMALL_GED_THRESHOLD: usize = 16;

/// Subbotin/GED law with location `mu`, scale `alpha` and tail exponent `beta`: parameters only; a
/// [`Seeded`](crate::Seeded) stream draws it in bulk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdGed<T> {
  mu: T,
  alpha: T,
  beta: T,
  gamma: SimdGamma<T>,
}

/// A GED stream: the gamma magnitude sub-stream, the engine of the signs and the single-draw buffer.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct GedState<T: SimdFloatExt, R: SimdRngExt> {
  gamma: GammaState<T, R>,
  rng: R,
  buf: Buffered<T, 16>,
}

impl<T: SimdFloatExt> SimdGed<T> {
  /// Creates a Subbotin/GED distribution.
  ///
  /// - `mu` — location μ (matches the module header's μ).
  /// - `alpha` — scale α > 0 (matches the module header's α; NOT a
  ///   shape parameter, unlike the `alpha` in
  ///   [`SimdBeta`](crate::beta::SimdBeta)/[`SimdGamma`]).
  /// - `beta` — tail exponent β > 0 (matches the module header's β;
  ///   β=1 is Laplace, β=2 is Gaussian). Consumed internally as `1/beta`
  ///   into a `Gamma(1/beta, 1)` magnitude sampler.
  pub fn new(mu: T, alpha: T, beta: T) -> Self {
    assert!(
      mu.is_finite(),
      "mu must satisfy `mu.is_finite()`, got mu = {mu:?}"
    );
    assert!(
      alpha.is_finite(),
      "alpha must satisfy `alpha.is_finite()`, got alpha = {alpha:?}"
    );
    assert!(
      beta.is_finite(),
      "beta must satisfy `beta.is_finite()`, got beta = {beta:?}"
    );
    assert!(
      alpha > T::zero(),
      "alpha must satisfy `alpha > T::zero()`, got alpha = {alpha:?}"
    );
    assert!(
      beta > T::zero(),
      "beta must satisfy `beta > T::zero()`, got beta = {beta:?}"
    );
    let shape = T::one() / beta;
    assert!(
      shape.is_finite(),
      "beta must satisfy `(1 / beta).is_finite()`, got beta = {beta:?}"
    );
    Self {
      mu,
      alpha,
      beta,
      gamma: SimdGamma::new(shape, T::one()),
    }
  }

  /// The location `μ`.
  pub fn mu(&self) -> T {
    self.mu
  }

  /// The scale `α`.
  pub fn alpha(&self) -> T {
    self.alpha
  }

  /// The tail exponent `β`.
  pub fn beta(&self) -> T {
    self.beta
  }

  fn fill_parts<R: SimdRngExt>(&self, gamma: &mut GammaState<T, R>, rng: &mut R, out: &mut [T]) {
    let inv_beta = T::one() / self.beta;
    if out.len() < SMALL_GED_THRESHOLD {
      for x in out.iter_mut() {
        let mag = self.gamma.next(gamma).powf(inv_beta);
        let signed = if rng.next_i32() >= 0 { mag } else { -mag };
        *x = self.alpha * signed + self.mu;
      }
      return;
    }
    let alpha = T::splat(self.alpha);
    let mut ybuf = [T::zero(); 64];
    let (chunks, rem) = out.as_chunks_mut::<64>();
    for chunk in chunks {
      self.gamma.fill(gamma, &mut ybuf);
      for (sub, y8) in chunk
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(ybuf.as_chunks::<8>().0.iter())
      {
        let mag = T::simd_powf(T::simd_from_array(*y8), inv_beta);
        let t = T::simd_to_array(alpha * mag);
        let signs = rng.next_i32x8().to_array();
        for i in 0..8 {
          sub[i] = self.mu + if signs[i] >= 0 { t[i] } else { -t[i] };
        }
      }
    }
    for x in rem.iter_mut() {
      let mag = self.gamma.next(gamma).powf(inv_beta);
      let signed = if rng.next_i32() >= 0 { mag } else { -mag };
      *x = self.alpha * signed + self.mu;
    }
  }

  pub(crate) fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    let mag = self.gamma.draw_with(rng).powf(T::one() / self.beta);
    let signed = if (rng.next_u32() as i32) >= 0 {
      mag
    } else {
      -mag
    };
    self.alpha * signed + self.mu
  }
}

impl<T: SimdFloatExt> Sealed for SimdGed<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdGed<T> {
  type State<R: SimdRngExt> = GedState<T, R>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (GedState<T, R>, u64) {
    let (gamma, _) = self.gamma.init::<R, S>(seed);
    let stream_seed = seed.next_seed();
    (
      GedState {
        gamma,
        rng: R::from_seed(stream_seed),
        buf: Buffered::new(),
      },
      stream_seed,
    )
  }
}

impl<T: SimdFloatExt> SimdKernel for SimdGed<T> {
  type Item = T;

  #[inline]
  fn fill<R: SimdRngExt>(&self, state: &mut GedState<T, R>, out: &mut [T]) {
    self.fill_parts(&mut state.gamma, &mut state.rng, out);
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut GedState<T, R>) -> T {
    let GedState { gamma, rng, buf } = state;
    buf.pop(|b| self.fill_parts(gamma, rng, b))
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdGed<T> {
  /// One scalar gamma power, then a sign bit, from the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.draw_with(rng)
  }
}

impl<T: SimdFloatExt> DistributionExt for SimdGed<T> {
  fn pdf(&self, x: f64) -> Option<f64> {
    let mu = self.mu.to_f64().unwrap();
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let z = ((x - mu) / a).abs();
    let log_pdf = b.ln() - (2.0 * a).ln() - crate::special::ln_gamma(1.0 / b) - z.powf(b);
    Some(log_pdf.exp())
  }

  fn cdf(&self, x: f64) -> Option<f64> {
    let mu = self.mu.to_f64().unwrap();
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let z = (x - mu) / a;
    // F(x) = 1/2 + sign(x - μ)/2 · γ(1/β, |z|^β) / Γ(1/β)
    let zb = z.abs().powf(b);
    let half_inc = 0.5 * crate::special::gamma_p(1.0 / b, zb);
    if z >= 0.0 {
      Some(0.5 + half_inc)
    } else {
      Some(0.5 - half_inc)
    }
  }

  fn quantile(&self, p: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&p) {
      return Some(f64::NAN);
    }
    let (mu, a, b) = (self.mu.to_f64()?, self.alpha.to_f64()?, self.beta.to_f64()?);
    let u = 2.0 * p - 1.0;
    let g = SimdGamma::<f64>::new(1.0 / b, 1.0).quantile(u.abs())?;
    Some(mu + u.signum() * a * g.powf(1.0 / b))
  }

  fn mean(&self) -> Option<f64> {
    self.mu.to_f64()
  }

  fn median(&self) -> Option<f64> {
    self.mu.to_f64()
  }

  fn mode(&self) -> Option<f64> {
    self.mu.to_f64()
  }

  fn variance(&self) -> Option<f64> {
    let (a, b) = (self.alpha.to_f64()?, self.beta.to_f64()?);
    // In log space: `Γ(3/β)` overflows below β ≈ 0.0175 while the variance is still finite.
    let ln_g = crate::special::ln_gamma;
    Some(a * a * (ln_g(3.0 / b) - ln_g(1.0 / b)).exp())
  }

  fn skewness(&self) -> Option<f64> {
    Some(0.0)
  }

  fn kurtosis(&self) -> Option<f64> {
    let b = self.beta.to_f64()?;
    let ln_g = crate::special::ln_gamma;
    Some((ln_g(5.0 / b) + ln_g(1.0 / b) - 2.0 * ln_g(3.0 / b)).exp() - 3.0)
  }

  fn entropy(&self) -> Option<f64> {
    let (a, b) = (self.alpha.to_f64()?, self.beta.to_f64()?);
    Some(1.0 / b - b.ln() + (2.0 * a).ln() + crate::special::ln_gamma(1.0 / b))
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Unseeded;

  use super::*;
  use crate::tests::scalar_ks_best_p;

  /// GED with β=2 should collapse to a Gaussian with std-dev α·√(1/2).
  /// Tested via sample variance.
  #[test]
  fn ged_beta_two_is_gaussian() {
    let mut g = SimdGed::<f64>::new(0.0, std::f64::consts::SQRT_2, 2.0).seeded(&Unseeded);
    let n = 30_000;
    let mut sum_sq = 0.0;
    let mut sum = 0.0;
    for _ in 0..n {
      let x = g.sample();
      sum += x;
      sum_sq += x * x;
    }
    let mean = sum / n as f64;
    let var = sum_sq / n as f64 - mean * mean;
    assert!(
      (mean).abs() < 0.05,
      "GED(0, √2, 2) mean = {mean}, expected ~0"
    );
    // Var = α² · Γ(3/β) / Γ(1/β); α=√2, β=2 → Γ(3/2)/Γ(1/2) = 0.5 → Var = 1
    assert!(
      (var - 1.0).abs() < 0.05,
      "GED(0, √2, 2) variance = {var}, expected ~1"
    );
  }

  #[test]
  fn scalar_sample_matches_cdf() {
    let d = SimdGed::<f64>::new(0.0, 1.0, 1.5);
    let best = scalar_ks_best_p(&d, |x| d.cdf(x).unwrap());
    assert!(best > 0.01, "best p = {best}");
  }

  /// PDF normalises to 1 (numeric integration).
  #[test]
  fn ged_pdf_normalised() {
    let g = SimdGed::<f64>::new(0.0, 1.0, 1.5);
    let n = 5000;
    let lo = -20.0_f64;
    let up = 20.0_f64;
    let h = (up - lo) / n as f64;
    let s: f64 = (0..n)
      .map(|k| g.pdf(lo + (k as f64 + 0.5) * h).unwrap() * h)
      .sum();
    assert!(
      (s - 1.0).abs() < 1e-3,
      "GED(0, 1, 1.5) PDF integrates to {s}"
    );
  }

  /// CDF round-trip via PDF integration.
  #[test]
  fn ged_cdf_at_mu_is_half() {
    let g = SimdGed::<f64>::new(1.5, 0.8, 1.7);
    let c = g.cdf(1.5).unwrap();
    assert!(
      (c - 0.5).abs() < 1e-10,
      "GED(1.5, 0.8, 1.7) CDF at μ = {c}, expected 0.5"
    );
  }
}

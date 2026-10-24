//! # Generalized Hyperbolic (GH)
//!
//! Barndorff-Nielsen's normal mean-variance mixture over a generalized
//! inverse Gaussian clock,
//!
//! $$
//! X = \mu + \beta W + \sqrt{W}\,Z,\qquad W \sim \mathrm{GIG}(\lambda, \delta^2, \alpha^2 - \beta^2),\ Z \sim \mathcal N(0,1),
//! $$
//!
//! $$
//! f(x) = \frac{(\alpha^2-\beta^2)^{\lambda/2}}{\sqrt{2\pi}\,\alpha^{\lambda-1/2}\,\delta^\lambda\,K_\lambda\bigl(\delta\sqrt{\alpha^2-\beta^2}\bigr)}
//! \bigl(\delta^2 + (x-\mu)^2\bigr)^{\frac{\lambda - 1/2}{2}}
//! K_{\lambda-1/2}\!\Bigl(\alpha\sqrt{\delta^2 + (x-\mu)^2}\Bigr)\,e^{\beta(x-\mu)},
//! $$
//!
//! with $\lambda \in \mathbb R$, $\alpha > |\beta|$, $\delta > 0$, $\mu \in
//! \mathbb R$ (the sampler needs the interior of the parameter space). The
//! normal-inverse-Gaussian is $\lambda = -1/2$, the hyperbolic $\lambda = 1$,
//! and the variance gamma the $\delta \to 0$ limit for $\lambda > 0$. Moments
//! follow from the mixture: the mean is $\mu + \beta\,\mathbb E W$, the
//! variance $\mathbb E W + \beta^2\mathrm{Var}\,W$, and the third and fourth
//! central moments $\beta^3\mu_3(W) + 3\beta\mathrm{Var}\,W$ and
//! $\beta^4\mu_4(W) + 6\beta^2(\mu_3(W) + \mathbb E W\,\mathrm{Var}\,W) + 3\,\mathbb E W^2$.
//! There is no closed-form CDF or quantile.
//!
//! References:
//! - Barndorff-Nielsen, O. (1977), "Exponentially decreasing distributions
//!   for the logarithm of particle size", *Proceedings of the Royal Society
//!   A* 353(1674), 401-419. DOI: 10.1098/rspa.1977.0041
//! - Eberlein, E., Keller, U. (1995), "Hyperbolic Distributions in Finance",
//!   *Bernoulli* 1(3), 281-299. DOI: 10.2307/3318481
//! - Prause, K. (1999), *The Generalized Hyperbolic Model: Estimation,
//!   Financial Derivatives, and Risk Measures*, PhD thesis, University of
//!   Freiburg, ch. 1.
//! - Olver, F.W.J. et al. (eds.), *NIST Digital Library of Mathematical Functions*, eq. 10.30.2 (the mgf at its boundary).

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use super::SimdFloatExt;
use super::generalized_inverse_gauss::SimdGig;
use super::normal::SimdNormal;
use crate::seeded::Buffered;
use crate::seeded::StreamState;
use crate::special::exp_of_log_sum;
use crate::special::ln_bessel_ke;
use crate::special::ln_gamma;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

const SMALL_GH_THRESHOLD: usize = 16;

/// Generalized hyperbolic distribution GH$(\lambda, \alpha, \beta, \delta, \mu)$; a [`Seeded`](crate::Seeded) stream draws it in bulk.
/// Density, mgf and mean within `1e-12` to `|λ| = 300`, `3ε|λ| ln|λ|` beyond, higher moments looser (kurtosis `1e-8` at 100); NaN past every digit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdGeneralizedHyperbolic<T> {
  lambda: T,
  alpha: T,
  beta: T,
  delta: T,
  mu: T,
  gig: SimdGig<T>,
}

/// A GH stream: the GIG-clock and normal sub-streams and the single-draw buffer.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct GhState<T: SimdFloatExt, R: SimdRngExt> {
  gig: StreamState<T, R, 16>,
  normal: StreamState<T, R, 64>,
  buf: Buffered<T, 16>,
}

impl<T: SimdFloatExt> SimdGeneralizedHyperbolic<T> {
  /// Construct a GH$(\lambda, \alpha, \beta, \delta, \mu)$ over an internal
  /// GIG$(\lambda, \delta^2, \alpha^2 - \beta^2)$ clock.
  pub fn new(lambda: T, alpha: T, beta: T, delta: T, mu: T) -> Self {
    assert!(
      lambda.is_finite(),
      "lambda must satisfy `lambda.is_finite()`, got lambda = {lambda:?}"
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
      delta.is_finite(),
      "delta must satisfy `delta.is_finite()`, got delta = {delta:?}"
    );
    assert!(
      mu.is_finite(),
      "mu must satisfy `mu.is_finite()`, got mu = {mu:?}"
    );
    assert!(
      alpha > beta.abs(),
      "alpha must satisfy `alpha > beta.abs()`, got alpha = {alpha:?}, beta = {beta:?}"
    );
    assert!(
      delta > T::zero(),
      "delta must satisfy `delta > T::zero()`, got delta = {delta:?}"
    );
    let chi = delta * delta;
    let psi = alpha * alpha - beta * beta;
    assert!(
      chi > T::zero() && chi.is_finite(),
      "delta must satisfy `0 < delta * delta < ∞`, got delta = {delta:?}"
    );
    assert!(
      psi > T::zero() && psi.is_finite(),
      "alpha must satisfy `0 < alpha * alpha - beta * beta < ∞`, got alpha = {alpha:?}, beta = {beta:?}"
    );
    Self {
      lambda,
      alpha,
      beta,
      delta,
      mu,
      gig: SimdGig::new(lambda, chi, psi),
    }
  }

  /// The index `λ` of the GIG clock.
  pub fn lambda(&self) -> T {
    self.lambda
  }

  /// The tail heaviness `α`.
  pub fn alpha(&self) -> T {
    self.alpha
  }

  /// The asymmetry `β`.
  pub fn beta(&self) -> T {
    self.beta
  }

  /// The scale `δ`.
  pub fn delta(&self) -> T {
    self.delta
  }

  /// The location `μ`.
  pub fn mu(&self) -> T {
    self.mu
  }

  fn fill_parts<R: SimdRngExt>(
    &self,
    gig: &mut StreamState<T, R, 16>,
    normal: &mut StreamState<T, R, 64>,
    out: &mut [T],
  ) {
    if out.len() < SMALL_GH_THRESHOLD {
      for x in out.iter_mut() {
        let w = self.gig.next(gig);
        let z = SimdNormal::<T>::standard().next(normal);
        *x = self.mu + self.beta * w + w.sqrt() * z;
      }
      return;
    }
    let mu = T::splat(self.mu);
    let beta = T::splat(self.beta);
    let mut wbuf = [T::zero(); 64];
    let mut zbuf = [T::zero(); 64];
    let (chunks, rem) = out.as_chunks_mut::<64>();
    for chunk in chunks {
      self.gig.fill(gig, &mut wbuf);
      SimdNormal::<T>::fill_standard(&mut normal.rng, &mut zbuf);
      for (sub, (w8, z8)) in chunk.as_chunks_mut::<8>().0.iter_mut().zip(
        wbuf
          .as_chunks::<8>()
          .0
          .iter()
          .zip(zbuf.as_chunks::<8>().0.iter()),
      ) {
        let w = T::simd_from_array(*w8);
        let z = T::simd_from_array(*z8);
        *sub = T::simd_to_array(mu + beta * w + T::simd_sqrt(w) * z);
      }
    }
    if !rem.is_empty() {
      let n = rem.len();
      self.gig.fill(gig, &mut wbuf[..n]);
      SimdNormal::<T>::fill_standard(&mut normal.rng, &mut zbuf[..n]);
      for i in 0..n {
        rem[i] = self.mu + self.beta * wbuf[i] + wbuf[i].sqrt() * zbuf[i];
      }
    }
  }

  pub(crate) fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    let w = self.gig.draw_with(rng);
    let z = SimdNormal::<T>::standard().draw_with(rng);
    self.mu + self.beta * w + w.sqrt() * z
  }

  fn params(&self) -> (f64, f64, f64, f64, f64) {
    (
      self.lambda.to_f64().unwrap(),
      self.alpha.to_f64().unwrap(),
      self.beta.to_f64().unwrap(),
      self.delta.to_f64().unwrap(),
      self.mu.to_f64().unwrap(),
    )
  }
}

impl<T: SimdFloatExt> Sealed for SimdGeneralizedHyperbolic<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdGeneralizedHyperbolic<T> {
  type State<R: SimdRngExt> = GhState<T, R>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (GhState<T, R>, u64) {
    let (gig, _) = self.gig.init::<R, S>(seed);
    let (normal, _) = SimdNormal::<T>::standard().init::<R, S>(seed);
    let basis = seed.next_seed();
    (
      GhState {
        gig,
        normal,
        buf: Buffered::new(),
      },
      basis,
    )
  }
}

impl<T: SimdFloatExt> SimdKernel for SimdGeneralizedHyperbolic<T> {
  type Item = T;

  #[inline]
  fn fill<R: SimdRngExt>(&self, state: &mut GhState<T, R>, out: &mut [T]) {
    self.fill_parts(&mut state.gig, &mut state.normal, out);
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut GhState<T, R>) -> T {
    let GhState { gig, normal, buf } = state;
    buf.pop(|b| self.fill_parts(gig, normal, b))
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdGeneralizedHyperbolic<T> {
  /// `μ + βW + √W·Z` from one scalar GIG and one scalar normal draw on the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.draw_with(rng)
  }
}

impl<T: SimdFloatExt> crate::traits::DistributionExt for SimdGeneralizedHyperbolic<T> {
  fn pdf(&self, x: f64) -> Option<f64> {
    // The density vanishes at ±∞, where the log-space kernel would read ∞ − ∞.
    if x.is_infinite() {
      return Some(0.0);
    }
    let (lambda, alpha, beta, delta, mu) = self.params();
    let gamma = (alpha * alpha - beta * beta).sqrt();
    let d = x - mu;
    let root = (delta * delta + d * d).sqrt();
    let order = lambda - 0.5;
    let (t_gamma, t_alpha, t_delta) =
      (lambda * gamma.ln(), order * alpha.ln(), lambda * delta.ln());
    let (k0, dg) = (ln_bessel_ke(lambda, delta * gamma), delta * gamma);
    let log_norm =
      t_gamma - 0.5 * (2.0 * std::f64::consts::PI).ln() - t_alpha - t_delta - (k0 - dg);
    let y = alpha * root;
    let (t_root, k1, t_beta) = (order * root.ln(), ln_bessel_ke(order, y), beta * d);
    Some(exp_of_log_sum(
      log_norm + t_root + k1 - y + t_beta,
      &[t_gamma, t_alpha, t_delta, k0, dg, t_root, k1, y, t_beta],
    ))
  }

  fn mean(&self) -> Option<f64> {
    let (_, _, beta, _, mu) = self.params();
    Some(mu + beta * self.gig.raw_moment(1))
  }

  fn variance(&self) -> Option<f64> {
    let (_, _, beta, _, _) = self.params();
    let [ew, _, var, _, _] = self.gig.moments();
    Some((ew + beta * beta * var).or_nan())
  }

  fn skewness(&self) -> Option<f64> {
    let (_, _, beta, _, _) = self.params();
    let [ew, _, var, mu3, _] = self.gig.moments();
    let m2 = ew + beta * beta * var;
    Some(((beta.powi(3) * mu3 + 3.0 * beta * var) / m2.powf(1.5)).or_nan())
  }

  /// Excess kurtosis.
  fn kurtosis(&self) -> Option<f64> {
    let (_, _, beta, _, _) = self.params();
    let [ew, ew2, var, mu3, mu4] = self.gig.moments();
    let m2 = ew + beta * beta * var;
    Some(
      ((beta.powi(4) * mu4 + 6.0 * beta * beta * (mu3 + ew * var) + 3.0 * ew2) / (m2 * m2) - 3.0)
        .or_nan(),
    )
  }

  /// $e^{\mu t}(\gamma_0/\gamma_t)^\lambda K_\lambda(\delta\gamma_t)/K_\lambda(\delta\gamma_0)$, $\gamma_t = \sqrt{\alpha^2-(\beta+t)^2}$; at $\gamma_t = 0$
  /// its limit $e^{\mu t}(\delta\gamma_0/2)^\lambda\Gamma(-\lambda)/(2K_\lambda(\delta\gamma_0))$ for $\lambda < 0$, else there and beyond `+∞`.
  fn moment_generating_function(&self, t: f64) -> Option<f64> {
    if t == 0.0 {
      return Some(1.0);
    }
    let (lambda, alpha, beta, delta, mu) = self.params();
    let shifted = alpha * alpha - (beta + t).powi(2);
    if shifted < 0.0 || (shifted == 0.0 && lambda >= 0.0) {
      return Some(f64::INFINITY);
    }
    let gamma = (alpha * alpha - beta * beta).sqrt();
    if shifted == 0.0 {
      // s^{-λ} K_λ(δs) → ½ Γ(−λ) (2/δ)^{−λ} as s → 0, in log space past the overflow of Γ and K and K's underflow.
      let (k0, dg) = (ln_bessel_ke(lambda, delta * gamma), delta * gamma);
      let (t_mu, t_pow, t_gamma) = (mu * t, lambda * (0.5 * dg).ln(), ln_gamma(-lambda));
      return Some(exp_of_log_sum(
        t_mu + t_pow + t_gamma - std::f64::consts::LN_2 - (k0 - dg),
        &[t_mu, t_pow, t_gamma, k0, dg],
      ));
    }
    let gt = shifted.sqrt();
    let (t_mu, t_pow) = (mu * t, 0.5 * lambda * (gamma * gamma / shifted).ln());
    let (kt, k0, t_exp) = (
      ln_bessel_ke(lambda, delta * gt),
      ln_bessel_ke(lambda, delta * gamma),
      delta * (gamma - gt),
    );
    Some(exp_of_log_sum(
      t_mu + t_pow + kt - k0 + t_exp,
      &[t_mu, t_pow, kt, k0, t_exp],
    ))
  }
}

#[cfg(test)]
mod tests;

py_distribution!(PyGeneralizedHyperbolic, SimdGeneralizedHyperbolic,
  sig: (lambda, alpha, beta, delta, mu, seed=None, dtype=None),
  params: (lambda: f64, alpha: f64, beta: f64, delta: f64, mu: f64)
);

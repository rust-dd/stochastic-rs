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
use crate::special::KDenominator;
use crate::special::LARGE_ORDER;
use crate::special::add_parts;
use crate::special::debye_log_sum;
use crate::special::divergence;
use crate::special::exp_times_over;
use crate::special::k_ratio;
use crate::special::ldexp;
use crate::special::ln_bessel_ke;
use crate::special::ln_gamma;
use crate::special::ln_shift;
use crate::special::mul_parts;
use crate::special::offset;
use crate::special::saddle;
use crate::special::saddle_parts;
use crate::special::split_pow2;
use crate::special::two_prod;
use crate::special::two_sum;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

const SMALL_GH_THRESHOLD: usize = 16;

/// Generalized hyperbolic distribution GH$(\lambda, \alpha, \beta, \delta, \mu)$; a [`Seeded`](crate::Seeded) stream draws it in bulk.
/// The density and mgf within `ε·Σ|ln terms|` below order 50, from it `4ε(|ln f| + 20)` and `2ε(|ln M| + 5)`; the mgf adds `2ε|μt|` to its edge `|β + t| = α`, the mean `ε(|μ| + ℓ|β|E W)/|mean|` (`ℓ` `2(|ln(δγ/2)| + 9)` below order 50, `|ln β| + |ln E W|` where `E W` or `βE W` leaves the normal range, else 1), and past the clock's `r ≈ 3e34` a density `k` deviations off its mode adds `k√r·1e-31`.
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

  /// The clock's `(λ, δ², γ²)` from the parameters as doubles, `γ² = (α − β)(α + β)`, which keeps its digits at `|β| ≈ α`.
  fn clock(&self) -> (f64, f64, f64) {
    let (lambda, alpha, beta, delta, _) = self.params();
    (lambda, delta * delta, (alpha - beta) * (alpha + beta))
  }

  /// `γ_t² = (α − β − t)(α + β + t)`, `α ∓ β` exactly as two parts (`α > |β|`), so neither factor loses `α ∓ β`'s
  /// rounding where `t` cancels it.
  fn shifted(&self, t: f64) -> f64 {
    let (_, alpha, beta, _, _) = self.params();
    let (diff, sum) = (alpha - beta, alpha + beta);
    let below = (diff - t) + ((alpha - diff) - beta);
    let above = (sum + t) + ((alpha - sum) + beta);
    below * above
  }

  /// The density from `r = √(λ² + δ²γ²) = 50`: `βd + ln N(λ − ½, δ² + d², α²) − ln N(λ, δ², γ²)` meets as
  /// `H_λ(w₁) − H_λ(w₀) − ½ ln w₁ − (d − βw₁)²/2w₁` (`α² = γ² + β²`), both maximisers and `d` in two parts, plus `T₁ − T₀`.
  fn concentrated_pdf(&self, d: (f64, f64), (gamma, r0): (f64, f64)) -> f64 {
    let (lambda, alpha, beta, delta, _) = self.params();
    let root = delta.hypot(d.0);
    if !root.is_finite() {
      return 0.0;
    }
    let chi = two_prod(delta, delta);
    let psi = mul_parts(two_sum(alpha, -beta), two_sum(alpha, beta));
    let order = two_sum(lambda, -0.5);
    let w0 = saddle(lambda, delta, gamma);
    let w1 = saddle(order.0, root, alpha);
    let (w0, w1) = (
      saddle_parts(w0, (lambda, 0.0), chi, psi),
      saddle_parts(
        w1,
        order,
        add_parts(chi, mul_parts(d, d)),
        two_prod(alpha, alpha),
      ),
    );
    let (bw, bw_lo) = two_prod(beta, w1.0);
    let (g, g_lo) = two_sum(d.0, -bw);
    let gap = g + (g_lo + d.1 - bw_lo - beta * w1.1);
    let r1 = order.0.hypot(alpha * root);
    let a = divergence(lambda, (chi.0, psi.0), w0.0, offset(w1, w0)) - 0.5 * (gap / w1.0) * gap
      + debye_log_sum(order.0.abs(), r1)
      - debye_log_sum(lambda.abs(), r0)
      - 0.5 * std::f64::consts::TAU.ln();
    exp_times_over(a, (r0 / r1).sqrt(), w1.0.sqrt())
  }

  /// The interior mgf's ratio at `t`, `shifted = γ_t² > 0`.
  fn mgf_ratio(
    &self,
    t: f64,
    shifted: f64,
  ) -> ((f64, f64, f64, f64), (f64, f64), KDenominator, f64) {
    let (lambda, alpha, beta, delta, mu) = self.params();
    let psi = (alpha - beta) * (alpha + beta);
    let (gamma, gt) = (psi.sqrt(), shifted.sqrt());
    let dx = delta * t * (2.0 * beta + t) / (gamma + gt);
    (
      (mu * t, psi, shifted, 0.5 * lambda),
      (lambda, delta * gt),
      KDenominator::new(lambda, delta * gamma),
      dx,
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
    let gamma = ((alpha - beta) * (alpha + beta)).sqrt();
    let r0 = lambda.hypot(delta * gamma);
    let (d, d_lo) = two_sum(x, -mu);
    if r0 >= LARGE_ORDER {
      return Some(self.concentrated_pdf((d, d_lo), (gamma, r0)));
    }
    let (root, order, dg) = (delta.hypot(d), lambda - 0.5, delta * gamma);
    let log_norm = lambda * gamma.ln()
      - 0.5 * std::f64::consts::TAU.ln()
      - order * alpha.ln()
      - lambda * delta.ln()
      - (ln_bessel_ke(lambda, dg) - dg);
    // `βd − α·root` as `−(α ∓ β)|d| − αδ²/(root + |d|)`, of one sign, where `α ≈ |β|` would leave two large terms to cancel.
    let tilt = -(alpha - beta * d.signum()) * d.abs() - alpha * delta * (delta / (root + d.abs()));
    Some((log_norm + order * root.ln() + ln_bessel_ke(order, alpha * root) + tilt).exp())
  }

  fn mean(&self) -> Option<f64> {
    let (_, _, beta, _, mu) = self.params();
    let (lambda, chi, psi) = self.clock();
    Some(mu + SimdGig::<f64>::scaled_raw_moment_at((lambda, chi.sqrt(), psi.sqrt()), 1, beta))
  }

  /// `κ₁(W) + β²κ₂(W)` over the clock's `κₙ = sⁿρⁿ⁻¹kₙ`, `s·k₁ + (βs)²ρk₂` with `s = 2ᵉm` and `β` split, so neither term
  /// leaves the doubles before the value does.
  fn variance(&self) -> Option<f64> {
    let (_, _, beta, _, _) = self.params();
    let ((m, e), rho, [k1, k2, _, _]) = SimdGig::<f64>::cumulants_at(self.clock());
    let clock_mean = ldexp(m * k1, e);
    if beta == 0.0 {
      return Some(clock_mean);
    }
    let (b, n) = split_pow2(beta.abs());
    let tilt = n * m;
    Some(clock_mean + ldexp(tilt * (rho * k2) * tilt, 2 * (b + e)))
  }

  /// `κ₃/κ₂^{3/2}` with `κ₃ = β(3κ₂(W) + β²κ₃(W))`, over the clock's `κₙ = sⁿρⁿ⁻¹kₙ` and `B = β²sρ` as `B` or `1/B`, whichever
  /// is small; 0 at `β = 0`, where the law is symmetric.
  fn skewness(&self) -> Option<f64> {
    let (_, _, beta, _, _) = self.params();
    if beta == 0.0 {
      return Some(0.0);
    }
    let ((m, e), rho, [k1, k2, k3, _]) = SimdGig::<f64>::cumulants_at(self.clock());
    let (b, n) = split_pow2(beta.abs());
    let big = tilt_ratio((b, n), (m, e), rho);
    Some(if big <= 1.0 {
      // `|β|√s = 2^{b + ⌊e/2⌋}·n√(m·2^{e mod 2})`.
      let root = n * (m * f64::from(1 + (e & 1))).sqrt();
      let scaled = root * rho * (3.0 * k2 + big * k3) / (k1 + big * k2).powf(1.5);
      beta.signum() * ldexp(scaled, b + (e >> 1))
    } else {
      beta.signum() * rho.sqrt() * (3.0 * k2 / big + k3) / (k1 / big + k2).powf(1.5)
    })
  }

  /// Excess kurtosis `κ₄/κ₂²`, `κ₄ = 3κ₂(W) + 6β²κ₃(W) + β⁴κ₄(W)`, every term positive, scaled like the skewness.
  fn kurtosis(&self) -> Option<f64> {
    let (_, _, beta, _, _) = self.params();
    let ((m, e), rho, [k1, k2, k3, k4]) = SimdGig::<f64>::cumulants_at(self.clock());
    let big = if beta == 0.0 {
      0.0
    } else {
      tilt_ratio(split_pow2(beta.abs()), (m, e), rho)
    };
    // Where `B` is 0 the clock's `κ₃` and `κ₄` do not enter, and an infinite one must not read `0·∞`.
    if big == 0.0 {
      return Some(rho * (3.0 * k2 / (k1 * k1)));
    }
    Some(if big <= 1.0 {
      rho * (3.0 * k2 + big * (6.0 * k3 + big * k4)) / (k1 + big * k2).powi(2)
    } else {
      rho * ((3.0 * k2 / big + 6.0 * k3) / big + k4) / (k1 / big + k2).powi(2)
    })
  }

  /// $e^{\mu t}(\gamma_0/\gamma_t)^\lambda K_\lambda(\delta\gamma_t)/K_\lambda(\delta\gamma_0)$, $\gamma_t = \sqrt{\alpha^2-(\beta+t)^2}$; at $\gamma_t = 0$
  /// its limit $e^{\mu t}(\delta\gamma_0/2)^\lambda\Gamma(-\lambda)/(2K_\lambda(\delta\gamma_0))$ for $\lambda < 0$, else there and beyond `+∞`.
  fn moment_generating_function(&self, t: f64) -> Option<f64> {
    if t == 0.0 {
      return Some(1.0);
    }
    let (lambda, alpha, beta, delta, mu) = self.params();
    let shifted = self.shifted(t);
    if shifted < 0.0 || (shifted == 0.0 && lambda >= 0.0) {
      return Some(f64::INFINITY);
    }
    if lambda.abs() >= LARGE_ORDER {
      let (chi, psi) = (delta * delta, (alpha - beta) * (alpha + beta));
      return Some(
        (mu * t + ln_shift(lambda, 0.5 * t * (2.0 * beta + t), (chi, psi, shifted))).exp(),
      );
    }
    if shifted == 0.0 {
      // s^{-λ} K_λ(δs) → ½ Γ(−λ) (2/δ)^{−λ} as s → 0, in log space past the overflow of Γ and K and K's underflow.
      let dg = delta * ((alpha - beta) * (alpha + beta)).sqrt();
      return Some(
        (mu * t + lambda * (0.5 * dg).ln() + ln_gamma(-lambda)
          - std::f64::consts::LN_2
          - (ln_bessel_ke(lambda, dg) - dg))
          .exp(),
      );
    }
    let (power, num, den, dx) = self.mgf_ratio(t, shifted);
    Some(k_ratio(power, num, &den, dx))
  }
}

/// `B = β²sρ` from `β = 2ᵇn` and the clock's scale `s = 2ᵉm`, so a scale past the doubles does not read `0·∞`.
fn tilt_ratio((b, n): (i32, f64), (m, e): (f64, i32), rho: f64) -> f64 {
  ldexp((n * m) * (n * rho), 2 * b + e)
}

#[cfg(test)]
mod tests;

py_distribution!(PyGeneralizedHyperbolic, SimdGeneralizedHyperbolic,
  sig: (lambda, alpha, beta, delta, mu, seed=None, dtype=None),
  params: (lambda: f64, alpha: f64, beta: f64, delta: f64, mu: f64)
);

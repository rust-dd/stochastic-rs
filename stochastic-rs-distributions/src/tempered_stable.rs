//! # Tempered stable (positive, exponentially tilted stable)
//!
//! The one-sided stable law $S_\alpha$, $\alpha \in (0, 1)$, with Laplace
//! transform $e^{-u^\alpha}$, exponentially tilted by $\lambda \ge 0$ and
//! scaled by $\theta > 0$:
//!
//! $$
//! \mathbb E\,e^{-uX} = \exp\!\bigl(\theta\,(\lambda^\alpha - (u + \lambda)^\alpha)\bigr),\qquad
//! X \overset{\mathcal L}{=} \theta^{1/\alpha}\,S_{\alpha,\ \lambda\theta^{1/\alpha}},
//! $$
//!
//! the unit-time law of the tempered stable subordinator (the positive
//! half of a CGMY process with $Y = \alpha$, $M = \lambda$ and
//! $\theta = C\,\Gamma(1-\alpha)/\alpha$). Cumulants come straight from the
//! Laplace exponent, $\kappa_n = \theta\,(-1)^{n+1}\alpha(\alpha-1)\cdots(\alpha-n+1)\,\lambda^{\alpha-n}$;
//! there is no closed-form density or CDF.
//!
//! ## Sampling
//!
//! Devroye's exact double-rejection generator for $S_{\alpha,\lambda}$
//! (Appendix algorithm of the reference), built on Zolotarev's integral
//! representation: an auxiliary angle $U \in [0, \pi)$ is drawn from a
//! Gaussian / beta / uniform mixture hat and accepted against the Zolotarev
//! density, then $X$ from a three-piece bi-exponential hat and accepted
//! against $h(x, U)$; the return value is $1/X^{(1-\alpha)/\alpha}$. The
//! expected number of loops is uniformly bounded (below 8.12) in both
//! $\alpha$ and $\lambda$, and at $\lambda = 0$ the scheme reduces to
//! Kanter's method.
//!
//! Reference: Devroye, L. (2009), "Random variate generation for
//! exponentially and polynomially tilted stable distributions", *ACM
//! Transactions on Modeling and Computer Simulation* 19(4), Article 18.
//! DOI: 10.1145/1596519.1596523

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use super::SimdFloatExt;
use super::inverse_gauss::NormalPlusOwnState;
use crate::seeded::StreamState;
use crate::source::AnyRng;
use crate::source::NormalUniformSource;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

/// Positive tempered stable law with stability `alpha ∈ (0, 1)`, tilting
/// `lambda ≥ 0` and scale `theta > 0`; a [`Seeded`](crate::Seeded) stream draws it in bulk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdTemperedStable<T> {
  alpha: T,
  lambda: T,
  theta: T,
  /// Tilting of the unit-scale law, $\lambda\theta^{1/\alpha}$.
  tilt: f64,
  /// $\theta^{1/\alpha}$.
  scale: f64,
}

/// $\mathcal S(x) = \sin x / x$.
#[inline]
fn sinc(x: f64) -> f64 {
  if x.abs() < 1e-8 {
    1.0 - x * x / 6.0
  } else {
    x.sin() / x
  }
}

/// $B(x)/B(0) = \mathcal S(x) / \bigl(\mathcal S(\alpha x)^\alpha\,\mathcal S((1-\alpha)x)^{1-\alpha}\bigr)$.
#[inline]
fn zolotarev_ratio(alpha: f64, x: f64) -> f64 {
  sinc(x) / (sinc(alpha * x).powf(alpha) * sinc((1.0 - alpha) * x).powf(1.0 - alpha))
}

/// Zolotarev's $A(u) = \bigl((\sin\alpha u)^\alpha(\sin(1-\alpha)u)^{1-\alpha}/\sin u\bigr)^{1/(1-\alpha)}$,
/// evaluated as $B(0)^{-1/(1-\alpha)}\,(B(u)/B(0))^{-1/(1-\alpha)}$ with
/// $B(0) = \alpha^{-\alpha}(1-\alpha)^{-(1-\alpha)}$.
#[inline]
fn zolotarev_a(alpha: f64, u: f64) -> f64 {
  let b0 = alpha.powf(-alpha) * (1.0 - alpha).powf(-(1.0 - alpha));
  (b0 * zolotarev_ratio(alpha, u)).powf(-1.0 / (1.0 - alpha))
}

impl<T: SimdFloatExt> SimdTemperedStable<T> {
  /// Construct a tempered stable$(\alpha, \lambda, \theta)$.
  pub fn new(alpha: T, lambda: T, theta: T) -> Self {
    let alpha_f = alpha.to_f64().unwrap();
    let lambda_f = lambda.to_f64().unwrap();
    let theta_f = theta.to_f64().unwrap();
    assert!(
      alpha_f > 0.0 && alpha_f < 1.0,
      "TemperedStable: alpha must lie in (0, 1)"
    );
    assert!(
      lambda_f >= 0.0,
      "TemperedStable: lambda must be non-negative"
    );
    assert!(theta_f > 0.0, "TemperedStable: theta must be positive");
    let scale = theta_f.powf(1.0 / alpha_f);
    Self {
      alpha,
      lambda,
      theta,
      tilt: lambda_f * scale,
      scale,
    }
  }

  /// The stability index `α`.
  pub fn alpha(&self) -> T {
    self.alpha
  }

  /// The tilting `λ`.
  pub fn lambda(&self) -> T {
    self.lambda
  }

  /// The scale `θ`.
  pub fn theta(&self) -> T {
    self.theta
  }

  /// One draw of $S_{\alpha,\lambda'}$ with $\lambda' = $ `self.tilt` —
  /// Devroye's Appendix algorithm, step for step.
  fn draw_unit<S: NormalUniformSource<T>>(&self, src: &mut S) -> f64 {
    let alpha = self.alpha.to_f64().unwrap();
    let lambda = self.tilt;
    let pi = std::f64::consts::PI;
    let uniform = |src: &mut S| src.uniform().to_f64().unwrap();
    let normal = |src: &mut S| src.normal().to_f64().unwrap();
    let exponential = |src: &mut S| -uniform(src).max(1e-300).ln();

    let lambda_alpha = lambda.powf(alpha);
    let gamma = lambda_alpha * alpha * (1.0 - alpha);
    let root_half_pi = (pi / 2.0).sqrt();
    let xi = ((2.0 + root_half_pi) * (2.0 * gamma).sqrt() + 1.0) / pi;
    let psi = (-gamma * pi * pi / 8.0).exp() * (2.0 + root_half_pi) * (gamma * pi).sqrt() / pi;
    let w1 = xi * (pi / (2.0 * gamma)).sqrt();
    let w2 = 2.0 * psi * pi.sqrt();
    let w3 = xi * pi;
    let b = (1.0 - alpha) / alpha;
    let sqrt_gamma = gamma.sqrt();
    let sqrt_gamma_pow = sqrt_gamma.powf(1.0 / alpha);

    loop {
      let (u, z, zeta, zed) = loop {
        let v = uniform(src);
        let w_prime = uniform(src);
        let u = if gamma >= 1.0 {
          if v < w1 / (w1 + w2) {
            normal(src).abs() / sqrt_gamma
          } else {
            pi * (1.0 - w_prime * w_prime)
          }
        } else if v < w3 / (w3 + w2) {
          pi * w_prime
        } else {
          pi * (1.0 - w_prime * w_prime)
        };
        let w = uniform(src);
        let zeta = zolotarev_ratio(alpha, u).sqrt();
        let phi = (sqrt_gamma + alpha * zeta).powf(1.0 / alpha);
        let zed = phi / (phi - sqrt_gamma_pow);
        let numerator = pi
          * (-lambda_alpha * (1.0 - 1.0 / (zeta * zeta))).exp()
          * (if u >= 0.0 && gamma >= 1.0 {
            xi * (-gamma * u * u / 2.0).exp()
          } else {
            0.0
          } + if u > 0.0 && u < pi {
            psi / (pi - u).sqrt()
          } else {
            0.0
          } + if (0.0..=pi).contains(&u) && gamma < 1.0 {
            xi
          } else {
            0.0
          });
        let rho = numerator / ((1.0 + root_half_pi) * sqrt_gamma / zeta + zed);
        let z = w * rho;
        if u < pi && z <= 1.0 {
          break (u, z, zeta, zed);
        }
      };
      let _ = zeta;
      let a = zolotarev_a(alpha, u);
      let m = (b * lambda / a).powf(alpha);
      let delta = (m * alpha / a).sqrt();
      let a1 = delta * root_half_pi;
      let a2 = delta;
      let a3 = zed / a;
      let s = a1 + a2 + a3;
      let v_prime = uniform(src);
      let mut n_prime = 0.0;
      let mut e_prime = 0.0;
      // The paper's appendix prints `V' < a2/s` for the middle piece; the mixture needs `(a1 + a2)/s`.
      let x = if v_prime < a1 / s {
        n_prime = normal(src);
        m - delta * n_prime.abs()
      } else if v_prime < (a1 + a2) / s {
        m + uniform(src) * delta
      } else {
        e_prime = exponential(src);
        m + delta + e_prime * a3
      };
      let e = -z.ln();
      if x > 0.0 {
        let tilt_term = if lambda > 0.0 {
          lambda * (x.powf(-b) - m.powf(-b))
        } else {
          0.0
        };
        let penalty = if x < m { n_prime * n_prime / 2.0 } else { 0.0 }
          + if x > m + delta { e_prime } else { 0.0 };
        if a * (x - m) + tilt_term - penalty <= e {
          return x.powf(-b);
        }
      }
    }
  }

  fn fill_parts<R: SimdRngExt>(
    &self,
    normal: &mut StreamState<T, R, 64>,
    rng: &mut R,
    out: &mut [T],
  ) {
    let mut src = (normal, rng);
    for x in out.iter_mut() {
      *x = T::from_f64_fast(self.scale * self.draw_unit(&mut src));
    }
  }

  pub(crate) fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    T::from_f64_fast(self.scale * self.draw_unit(&mut AnyRng(rng)))
  }

  fn params(&self) -> (f64, f64, f64) {
    (
      self.alpha.to_f64().unwrap(),
      self.lambda.to_f64().unwrap(),
      self.theta.to_f64().unwrap(),
    )
  }

  /// $\kappa_n = \theta\,(-1)^{n+1}\,\alpha(\alpha-1)\cdots(\alpha-n+1)\,\lambda^{\alpha-n}$;
  /// `+∞` for every $n \ge 1$ when $\lambda = 0$.
  pub fn cumulant(&self, n: u32) -> f64 {
    let (alpha, lambda, theta) = self.params();
    if lambda == 0.0 {
      return f64::INFINITY;
    }
    let mut falling = 1.0;
    for k in 0..n {
      falling *= alpha - k as f64;
    }
    let sign = if n.is_multiple_of(2) { -1.0 } else { 1.0 };
    theta * sign * falling * lambda.powf(alpha - n as f64)
  }
}

impl<T: SimdFloatExt> Sealed for SimdTemperedStable<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdTemperedStable<T> {
  type State<R: SimdRngExt> = NormalPlusOwnState<T, R>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (NormalPlusOwnState<T, R>, u64) {
    NormalPlusOwnState::init(seed)
  }
}

impl<T: SimdFloatExt> SimdKernel for SimdTemperedStable<T> {
  type Item = T;

  #[inline]
  fn fill<R: SimdRngExt>(&self, state: &mut NormalPlusOwnState<T, R>, out: &mut [T]) {
    self.fill_parts(&mut state.normal, &mut state.rng, out);
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut NormalPlusOwnState<T, R>) -> T {
    let NormalPlusOwnState { normal, rng, buf } = state;
    buf.pop(|b| self.fill_parts(normal, rng, b))
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdTemperedStable<T> {
  /// One scalar Devroye double rejection on the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.draw_with(rng)
  }
}

impl<T: SimdFloatExt> crate::traits::DistributionExt for SimdTemperedStable<T> {
  fn mean(&self) -> f64 {
    self.cumulant(1)
  }

  fn variance(&self) -> f64 {
    self.cumulant(2)
  }

  fn skewness(&self) -> f64 {
    self.cumulant(3) / self.cumulant(2).powf(1.5)
  }

  /// Excess kurtosis $\kappa_4/\kappa_2^2$.
  fn kurtosis(&self) -> f64 {
    self.cumulant(4) / self.cumulant(2).powi(2)
  }

  /// $\exp\bigl(\theta(\lambda^\alpha - (\lambda - t)^\alpha)\bigr)$ for $t \le \lambda$, `NaN` beyond.
  fn moment_generating_function(&self, t: f64) -> f64 {
    let (alpha, lambda, theta) = self.params();
    if t > lambda {
      return f64::NAN;
    }
    (theta * (lambda.powf(alpha) - (lambda - t).powf(alpha))).exp()
  }

  /// $\exp\bigl(\theta(\lambda^\alpha - (\lambda - iu)^\alpha)\bigr)$.
  fn characteristic_function(&self, u: f64) -> num_complex::Complex64 {
    use num_complex::Complex64;
    let (alpha, lambda, theta) = self.params();
    (Complex64::new(theta * lambda.powf(alpha), 0.0)
      - Complex64::new(lambda, -u).powf(alpha) * theta)
      .exp()
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;
  use crate::tests::assert_moments_within;
  use crate::tests::scalar_draws;
  use crate::traits::DistributionExt;
  use crate::traits::DistributionSampler;

  fn laplace_transform(xs: &[f64], u: f64) -> f64 {
    xs.iter().map(|x| (-u * x).exp()).sum::<f64>() / xs.len() as f64
  }

  /// The empirical Laplace transform of the draws matches
  /// $\exp(\theta(\lambda^\alpha - (u+\lambda)^\alpha))$ across tilting
  /// regimes, including the untilted Kanter limit and a strong tilt where
  /// naive rejection would need $e^{\lambda^\alpha}$ tries.
  #[test]
  fn laplace_transform_matches_the_closed_form() {
    for (alpha, lambda, theta, seed) in [
      (0.5, 0.0, 1.0, 1u64),
      (0.7, 1.0, 1.0, 2),
      (0.3, 4.0, 2.0, 3),
      (0.9, 30.0, 0.5, 4),
    ] {
      let d = SimdTemperedStable::<f64>::new(alpha, lambda, theta);
      let n = 300_000;
      let mut xs = vec![0.0; n];
      d.seeded(&Deterministic::new(seed)).fill_slice(&mut xs);
      assert!(xs.iter().all(|x| *x > 0.0 && x.is_finite()));
      for u in [0.5, 1.0, 2.0] {
        let want = (theta * (lambda.powf(alpha) - (u + lambda).powf(alpha))).exp();
        let got = laplace_transform(&xs, u);
        assert!(
          (got - want).abs() < 4e-3,
          "α={alpha} λ={lambda} u={u}: {got} vs {want}"
        );
      }
    }
  }

  /// Cumulant moments: sample mean and variance against κ₁, κ₂.
  #[test]
  fn sample_moments_match_the_cumulants() {
    let d = SimdTemperedStable::<f64>::new(0.6, 2.0, 1.5);
    let n = 400_000;
    let mut xs = vec![0.0; n];
    d.seeded(&Deterministic::new(9)).fill_slice(&mut xs);
    let mean = xs.iter().sum::<f64>() / n as f64;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    assert!(
      (mean - d.mean()).abs() / d.mean() < 0.01,
      "mean {mean} vs {}",
      d.mean()
    );
    assert!(
      (var - d.variance()).abs() / d.variance() < 0.03,
      "var {var} vs {}",
      d.variance()
    );
    assert!((d.mean() - 1.5 * 0.6 * 2.0_f64.powf(-0.4)).abs() < 1e-14);
    assert!((d.variance() - 1.5 * 0.6 * 0.4 * 2.0_f64.powf(-1.4)).abs() < 1e-14);
    assert!(d.skewness() > 0.0 && d.kurtosis() > 0.0);
    assert!((d.moment_generating_function(0.0) - 1.0).abs() < 1e-15);
    assert!(d.moment_generating_function(3.0).is_nan());
    let cf = d.characteristic_function(0.0);
    assert!((cf.re - 1.0).abs() < 1e-15 && cf.im.abs() < 1e-15);
  }

  #[test]
  fn scalar_sample_moments_match_the_cumulants() {
    let d = SimdTemperedStable::<f64>::new(0.6, 2.0, 1.5);
    let xs = scalar_draws(&d, 13, 200_000);
    assert_moments_within(
      &xs,
      d.cumulant(1),
      d.cumulant(2),
      None,
      6.0,
      "tempered stable",
    );
  }

  #[test]
  fn untilted_law_has_infinite_mean() {
    let d = SimdTemperedStable::<f64>::new(0.5, 0.0, 1.0);
    assert_eq!(d.mean(), f64::INFINITY);
  }

  #[test]
  fn deterministic_seed_reproduces_stream() {
    let mut a = SimdTemperedStable::<f64>::new(0.7, 1.0, 1.0).seeded(&Deterministic::new(7));
    let mut b = SimdTemperedStable::<f64>::new(0.7, 1.0, 1.0).seeded(&Deterministic::new(7));
    for _ in 0..256 {
      assert_eq!(a.sample(), b.sample());
    }
  }

  #[test]
  #[should_panic(expected = "alpha must lie in (0, 1)")]
  fn rejects_alpha_of_one() {
    let _ = SimdTemperedStable::<f64>::new(1.0, 1.0, 1.0);
  }
}

py_distribution!(PyTemperedStable, SimdTemperedStable,
  sig: (alpha, lambda, theta, seed=None, dtype=None),
  params: (alpha: f64, lambda: f64, theta: f64)
);

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
  /// $\ln\theta/\alpha$, the log path's `scale`.
  ln_scale: f64,
  /// $\lambda'^\alpha = \lambda^\alpha\theta$, formed without $\lambda'$.
  lambda_alpha: f64,
  path: Path,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Path {
  /// Devroye's algorithm as printed.
  Plain,
  /// The same algorithm from `λ'^α` and logarithms, never forming `λ'` or `θ^{1/α}`.
  Log,
  /// The law's mean κ₁: its relative spread is below 2⁻⁵³.
  PointMass(f64),
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

/// $\ln(B(x)/B(0))$; below `x = 0.01` from $\ln\mathcal S(x) = -x^2/6 - x^4/180 - x^6/2835 - x^8/37800$, as the ratio
/// rounds its `−α(1 − α)x²/2` away next to 1, where `λ'^α` multiplies it.
fn ln_zolotarev_ratio(alpha: f64, x: f64) -> f64 {
  if x.abs() >= 0.01 {
    return zolotarev_ratio(alpha, x).ln();
  }
  let x2 = x * x;
  let mut power = x2;
  let mut sum = -0.5 * alpha * (1.0 - alpha) * x2;
  for (k, c) in [-1.0 / 180.0, -1.0 / 2835.0, -1.0 / 37800.0]
    .into_iter()
    .enumerate()
  {
    power *= x2;
    let p = 2 * k as i32 + 5;
    sum += c * (1.0 - alpha.powi(p) - (1.0 - alpha).powi(p)) * power;
  }
  sum
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
    assert!(
      alpha.is_finite(),
      "alpha must satisfy `alpha.is_finite()`, got alpha = {alpha:?}"
    );
    assert!(
      lambda.is_finite(),
      "lambda must satisfy `lambda.is_finite()`, got lambda = {lambda:?}"
    );
    assert!(
      theta.is_finite(),
      "theta must satisfy `theta.is_finite()`, got theta = {theta:?}"
    );
    let alpha_f = alpha.to_f64().unwrap();
    let lambda_f = lambda.to_f64().unwrap();
    let theta_f = theta.to_f64().unwrap();
    assert!(
      alpha_f > 0.0 && alpha_f < 1.0,
      "alpha must satisfy `alpha > 0.0 && alpha < 1.0`, got alpha = {alpha:?}"
    );
    assert!(
      lambda_f >= 0.0,
      "lambda must satisfy `lambda >= 0.0`, got lambda = {lambda:?}"
    );
    assert!(
      theta_f > 0.0,
      "theta must satisfy `theta > 0.0`, got theta = {theta:?}"
    );
    let scale = theta_f.powf(1.0 / alpha_f);
    let tilt = lambda_f * scale;
    let lambda_alpha = lambda_f.powf(alpha_f) * theta_f;
    let root_gamma = (lambda_alpha * alpha_f * (1.0 - alpha_f)).sqrt();
    // ζ at the largest angle a draw reaches, `U = pred(π)`, the smallest the hat's `(√γ + αζ)^{1/α}` meets.
    let zeta_min = zolotarev_ratio(alpha_f, std::f64::consts::PI.next_down()).sqrt();
    // The relative spread is `√((1 − α)/(α λ'^α))`, zero where `λ'^α` overflows; `αθ · λ^α/λ` overflows only with κ₁,
    // and keeps the exponent `α` exact where `α − 1` would round.
    let path = if lambda_alpha > (1.0 - alpha_f) / alpha_f * 2f64.powi(106) {
      Path::PointMass(alpha_f * theta_f * (lambda_f.powf(alpha_f) / lambda_f))
    } else if scale.is_normal()
      && (tilt.is_normal() || lambda_f == 0.0)
      // Past `2⁴⁰` the tilt term and `1 − 1/ζ²` err by about `λ'^α ε`.
      && lambda_alpha <= 2f64.powi(40)
      && root_gamma.max(alpha_f * zeta_min).ln() / alpha_f > -700.0
      && (root_gamma + alpha_f).ln() / alpha_f < 700.0
    {
      Path::Plain
    } else {
      Path::Log
    };
    Self {
      alpha,
      lambda,
      theta,
      tilt,
      scale,
      ln_scale: theta_f.ln() / alpha_f,
      lambda_alpha,
      path,
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

  /// One draw of $\theta^{1/\alpha} S_{\alpha,\lambda'}$, $\lambda' = \lambda\theta^{1/\alpha}$, by Devroye's Appendix
  /// algorithm step for step; the log path forms the hat ratio, $m$, the tilt term and the result from `λ'^α` and logs.
  fn draw_scaled<S: NormalUniformSource<T>>(&self, src: &mut S) -> f64 {
    let log = self.path == Path::Log;
    let alpha = self.alpha.to_f64().unwrap();
    let lambda = self.tilt;
    let pi = std::f64::consts::PI;
    let uniform = |src: &mut S| src.uniform().to_f64().unwrap();
    let normal = |src: &mut S| src.normal().to_f64().unwrap();
    let exponential = |src: &mut S| -uniform(src).max(1e-300).ln();

    let lambda_alpha = if log {
      self.lambda_alpha
    } else {
      lambda.powf(alpha)
    };
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
        let (zeta, decay) = if log {
          let ln_ratio = ln_zolotarev_ratio(alpha, u);
          ((0.5 * ln_ratio).exp(), lambda_alpha * (-ln_ratio).exp_m1())
        } else {
          let zeta = zolotarev_ratio(alpha, u).sqrt();
          (zeta, -lambda_alpha * (1.0 - 1.0 / (zeta * zeta)))
        };
        let zed = if log {
          -1.0 / (-(alpha * zeta / sqrt_gamma).ln_1p() / alpha).exp_m1()
        } else {
          let phi = (sqrt_gamma + alpha * zeta).powf(1.0 / alpha);
          phi / (phi - sqrt_gamma_pow)
        };
        let numerator = pi
          * decay.exp()
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
      let m = if log {
        lambda_alpha * (b / a).powf(alpha)
      } else {
        (b * lambda / a).powf(alpha)
      };
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
        // `λ' m^{−b} = λ'^α (a/b)^{1−α}`, and `x − m` is exact where the sum cancels.
        let tilt_term = if log {
          lambda_alpha * (a / b).powf(1.0 - alpha) * (-b * ((x - m) / m).ln_1p()).exp_m1()
        } else if lambda > 0.0 {
          lambda * (x.powf(-b) - m.powf(-b))
        } else {
          0.0
        };
        let penalty = if x < m { n_prime * n_prime / 2.0 } else { 0.0 }
          + if x > m + delta { e_prime } else { 0.0 };
        if a * (x - m) + tilt_term - penalty <= e {
          let unit = x.powf(-b);
          return if log || unit < f64::MIN_POSITIVE {
            (self.ln_scale - b * x.ln()).exp()
          } else {
            self.scale * unit
          };
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
    if let Path::PointMass(mean) = self.path {
      out.fill(T::from_f64_fast(mean));
      return;
    }
    let mut src = (normal, rng);
    for x in out.iter_mut() {
      *x = T::from_f64_fast(self.draw_scaled(&mut src));
    }
  }

  pub(crate) fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    T::from_f64_fast(match self.path {
      Path::PointMass(mean) => mean,
      _ => self.draw_scaled(&mut AnyRng(rng)),
    })
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
  fn mean(&self) -> Option<f64> {
    Some(self.cumulant(1))
  }

  fn variance(&self) -> Option<f64> {
    Some(self.cumulant(2))
  }

  fn skewness(&self) -> Option<f64> {
    Some(self.cumulant(3) / self.cumulant(2).powf(1.5))
  }

  /// Excess kurtosis $\kappa_4/\kappa_2^2$.
  fn kurtosis(&self) -> Option<f64> {
    Some(self.cumulant(4) / self.cumulant(2).powi(2))
  }

  /// $\exp\bigl(\theta(\lambda^\alpha - (\lambda - t)^\alpha)\bigr)$ for $t \le \lambda$, `+∞` beyond.
  fn moment_generating_function(&self, t: f64) -> Option<f64> {
    let (alpha, lambda, theta) = self.params();
    if t > lambda {
      return Some(f64::INFINITY);
    }
    Some((theta * (lambda.powf(alpha) - (lambda - t).powf(alpha))).exp())
  }

  /// $\exp\bigl(\theta(\lambda^\alpha - (\lambda - iu)^\alpha)\bigr)$.
  fn characteristic_function(&self, u: f64) -> Option<num_complex::Complex64> {
    use num_complex::Complex64;
    let (alpha, lambda, theta) = self.params();
    Some(
      (Complex64::new(theta * lambda.powf(alpha), 0.0)
        - Complex64::new(lambda, -u).powf(alpha) * theta)
        .exp(),
    )
  }
}

#[cfg(test)]
mod tests;

py_distribution!(PyTemperedStable, SimdTemperedStable,
  sig: (alpha, lambda, theta, seed=None, dtype=None),
  params: (alpha: f64, lambda: f64, theta: f64)
);

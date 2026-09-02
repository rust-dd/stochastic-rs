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

use std::cell::Cell;
use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::Unseeded;

use super::SimdFloatExt;
use super::normal::SimdNormal;
use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;

/// Positive tempered stable law with stability `alpha ∈ (0, 1)`, tilting
/// `lambda ≥ 0` and scale `theta > 0`.
pub struct SimdTemperedStable<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  alpha: T,
  lambda: T,
  theta: T,
  /// Tilting of the unit-scale law, $\lambda\theta^{1/\alpha}$.
  tilt: f64,
  /// $\theta^{1/\alpha}$.
  scale: f64,
  normal: SimdNormal<T, 64, R>,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  simd_rng: UnsafeCell<R>,
  stream_seed: Cell<u64>,
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

impl<T: SimdFloatExt, R: SimdRngExt> SimdTemperedStable<T, R> {
  /// Construct a tempered stable$(\alpha, \lambda, \theta)$.
  pub fn new<S: crate::simd_rng::SeedExt>(alpha: T, lambda: T, theta: T, seed: &S) -> Self {
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
    let normal = SimdNormal::<T, 64, R>::new(T::zero(), T::one(), seed);
    let stream_seed = seed.seed_value();
    Self {
      alpha,
      lambda,
      theta,
      tilt: lambda_f * scale,
      scale,
      normal,
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
      simd_rng: UnsafeCell::new(R::from_seed(stream_seed)),
      stream_seed: Cell::new(stream_seed),
    }
  }

  /// Builds an independent worker stream for the `stream_idx`-th chunk of a
  /// parallel `sample_matrix` fan-out; see
  /// [`DistributionSampler::fork`](crate::traits::DistributionSampler::fork).
  #[doc(hidden)]
  pub fn fork(&self, stream_idx: u64) -> Self {
    let mut basis = self.stream_seed.get();
    let call_basis = crate::simd_rng::derive_seed(&mut basis);
    self.stream_seed.set(basis);
    let child_seed = crate::simd_rng::derive_fork_seed(call_basis, stream_idx);
    Self::new(
      self.alpha,
      self.lambda,
      self.theta,
      &crate::simd_rng::Deterministic::new(child_seed),
    )
  }

  /// Returns a single sample using the internal SIMD RNG.
  #[inline]
  pub fn sample_fast(&self) -> T {
    let index = unsafe { &mut *self.index.get() };
    if *index >= 16 {
      self.refill_buffer();
    }
    let buf = unsafe { &mut *self.buffer.get() };
    let z = buf[*index];
    *index += 1;
    z
  }

  /// One draw of $S_{\alpha,\lambda'}$ with $\lambda' = $ `self.tilt` —
  /// Devroye's Appendix algorithm, step for step.
  fn draw_unit(&self, rng: &mut R) -> f64 {
    let alpha = self.alpha.to_f64().unwrap();
    let lambda = self.tilt;
    let pi = std::f64::consts::PI;
    let uniform = |rng: &mut R| T::sample_uniform_simd(rng).to_f64().unwrap();
    let normal = || self.normal.sample_fast().to_f64().unwrap();
    let exponential = |rng: &mut R| -uniform(rng).max(1e-300).ln();

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
        let v = uniform(rng);
        let w_prime = uniform(rng);
        let u = if gamma >= 1.0 {
          if v < w1 / (w1 + w2) {
            normal().abs() / sqrt_gamma
          } else {
            pi * (1.0 - w_prime * w_prime)
          }
        } else if v < w3 / (w3 + w2) {
          pi * w_prime
        } else {
          pi * (1.0 - w_prime * w_prime)
        };
        let w = uniform(rng);
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
      let v_prime = uniform(rng);
      let mut n_prime = 0.0;
      let mut e_prime = 0.0;
      let x = if v_prime < a1 / s {
        n_prime = normal();
        m - delta * n_prime.abs()
      } else if v_prime < (a1 + a2) / s {
        m + uniform(rng) * delta
      } else {
        e_prime = exponential(rng);
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

  /// Fills `out` with tempered stable draws; the double rejection is scalar
  /// on the internal SIMD uniform and normal streams.
  pub fn fill_slice(&self, out: &mut [T]) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    for x in out.iter_mut() {
      *x = T::from_f64_fast(self.scale * self.draw_unit(rng));
    }
  }

  fn refill_buffer(&self) {
    let buf = unsafe { &mut *self.buffer.get() };
    self.fill_slice(buf);
    unsafe {
      *self.index.get() = 0;
    }
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

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdTemperedStable<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.alpha, self.lambda, self.theta, &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Distribution<T> for SimdTemperedStable<T, R> {
  /// The `rng` argument is intentionally unused — this type draws from its
  /// own internal SIMD streams seeded at construction.
  fn sample<Rr: Rng + ?Sized>(&self, _rng: &mut Rr) -> T {
    let idx = unsafe { &mut *self.index.get() };
    if *idx >= 16 {
      self.refill_buffer();
    }
    let val = unsafe { (*self.buffer.get())[*idx] };
    *idx += 1;
    val
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> crate::traits::DistributionExt for SimdTemperedStable<T, R> {
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
  use crate::traits::DistributionExt;

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
      let d = SimdTemperedStable::<f64>::new(alpha, lambda, theta, &Deterministic::new(seed));
      let n = 300_000;
      let mut xs = vec![0.0; n];
      d.fill_slice(&mut xs);
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
    let d = SimdTemperedStable::<f64>::new(0.6, 2.0, 1.5, &Deterministic::new(9));
    let n = 400_000;
    let mut xs = vec![0.0; n];
    d.fill_slice(&mut xs);
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
  fn untilted_law_has_infinite_mean() {
    let d = SimdTemperedStable::<f64>::new(0.5, 0.0, 1.0, &Unseeded);
    assert_eq!(d.mean(), f64::INFINITY);
  }

  #[test]
  fn deterministic_seed_reproduces_stream() {
    let a = SimdTemperedStable::<f64>::new(0.7, 1.0, 1.0, &Deterministic::new(7));
    let b = SimdTemperedStable::<f64>::new(0.7, 1.0, 1.0, &Deterministic::new(7));
    for _ in 0..256 {
      assert_eq!(a.sample_fast(), b.sample_fast());
    }
  }

  #[test]
  #[should_panic(expected = "alpha must lie in (0, 1)")]
  fn rejects_alpha_of_one() {
    let _ = SimdTemperedStable::<f64>::new(1.0, 1.0, 1.0, &Unseeded);
  }
}

py_distribution!(PyTemperedStable, SimdTemperedStable,
  sig: (alpha, lambda, theta, seed=None, dtype=None),
  params: (alpha: f64, lambda: f64, theta: f64)
);

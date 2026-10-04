//! # Alpha Stable
//!
//! $$
//! \varphi_X(u)=\exp\!\left(i\delta u-\gamma^\alpha |u|^\alpha\left[1-i\beta\operatorname{sgn}(u)\omega(u,\alpha)\right]\right)
//! $$
//!
//! Sampling: Chambers, J.M., Mallows, C.L., Stuck, B.W. (1976), "A Method for Simulating Stable Random Variables", *Journal of the American Statistical Association* 71(354), 340-344, DOI 10.1080/01621459.1976.10480344.
//! Scale, location and the `α = 1` case: Weron, R. (1996), "On the Chambers-Mallows-Stuck method for simulating skewed stable random variables", *Statistics & Probability Letters* 28(2), 165-171, DOI 10.1016/0167-7152(95)00113-1.

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use super::SimdFloatExt;
use crate::seeded::StreamState;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

/// Alpha-stable law `S_α(scale, β, location)` in the `S₁` parametrization: parameters only; a
/// [`Seeded`](crate::Seeded) stream draws it in bulk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdAlphaStable<T> {
  alpha: T,
  beta: T,
  scale: T,
  location: T,
  /// $B_{\alpha,\beta} = \arctan(\beta\tan(\pi\alpha/2))/\alpha$ of the `α ≠ 1` formula.
  cms_b: T,
  /// $S_{\alpha,\beta} = (1 + \beta^2\tan^2(\pi\alpha/2))^{1/(2\alpha)}$ of the `α ≠ 1` formula.
  cms_s: T,
}

/// The formula the stability index selects.
#[derive(Clone, Copy)]
enum Branch {
  Gaussian,
  UnitIndex,
  General,
}

impl<T: SimdFloatExt> SimdAlphaStable<T> {
  /// Creates an alpha-stable distribution, Chambers-Mallows-Stuck sampled.
  ///
  /// - `alpha` — stability index α ∈ (0, 2] (matches the module header's
  ///   α; α=2 is Gaussian, α=1 with β=0 is Cauchy).
  /// - `beta` — skewness β ∈ [-1, 1] (matches the module header's β).
  /// - `scale` — scale γ > 0 (matches the module header's γ; NOT the
  ///   `θ` used by [`SimdGamma`](crate::gamma::SimdGamma) — unrelated
  ///   role, same word).
  /// - `location` — shift δ applied after the γ-scaled draw (matches the
  ///   module header's δ).
  pub fn new(alpha: T, beta: T, scale: T, location: T) -> Self {
    assert!(
      alpha > T::zero() && alpha <= T::from(2.0).unwrap(),
      "alpha must satisfy `alpha > T::zero() && alpha <= T::from(2.0).unwrap()`, got alpha = {alpha:?}"
    );
    assert!(
      (-T::one()..=T::one()).contains(&beta),
      "beta must satisfy `(-T::one()..=T::one()).contains(&beta)`, got beta = {beta:?}"
    );
    assert!(
      scale > T::zero(),
      "scale must satisfy `scale > T::zero()`, got scale = {scale:?}"
    );
    let tan_term = (T::from_f64_fast(std::f64::consts::PI) * alpha / T::from(2.0).unwrap()).tan();
    let beta_tan = beta * tan_term;
    Self {
      alpha,
      beta,
      scale,
      location,
      cms_b: beta_tan.atan() / alpha,
      cms_s: (T::one() + beta_tan * beta_tan).powf(T::one() / (T::from(2.0).unwrap() * alpha)),
    }
  }

  /// The stability index `α`.
  pub fn alpha(&self) -> T {
    self.alpha
  }

  /// The skewness `β`.
  pub fn beta(&self) -> T {
    self.beta
  }

  /// The scale `γ`.
  pub fn scale(&self) -> T {
    self.scale
  }

  /// The location `δ`.
  pub fn location(&self) -> T {
    self.location
  }

  fn branch(&self) -> Branch {
    let eps = T::from(1e-6).unwrap();
    if (self.alpha - T::from(2.0).unwrap()).abs() < eps {
      Branch::Gaussian
    } else if (self.alpha - T::one()).abs() < eps {
      Branch::UnitIndex
    } else {
      Branch::General
    }
  }

  fn clamp_open_unit(x: T) -> T {
    let eps = T::from(1e-12).unwrap();
    if x <= eps {
      eps
    } else if x >= T::one() - eps {
      T::one() - eps
    } else {
      x
    }
  }

  /// `σX + (2/π)βσ ln σ + μ` turns Weron's `X ~ S₁(1, β, 0)` into `S₁(σ, β, μ)`; this is its constant part.
  fn unit_index_location(&self) -> T {
    let two_over_pi = T::from(2.0).unwrap() / T::from_f64_fast(std::f64::consts::PI);
    self.location + two_over_pi * self.beta * self.scale * self.scale.ln()
  }

  /// One `α = 1` draw from the angle uniform `u` and the exponential's uniform `e`.
  #[inline]
  fn unit_index_one(&self, u: T, e: T, location: T) -> T {
    let pi = T::from_f64_fast(std::f64::consts::PI);
    let half_pi = pi / T::from(2.0).unwrap();
    let two_over_pi = T::from(2.0).unwrap() / pi;
    let u = Self::clamp_open_unit(u);
    let e = Self::clamp_open_unit(e);
    let v = pi * (u - T::from(0.5).unwrap());
    let w = -e.ln();
    let a = half_pi + self.beta * v;
    let mut ratio = (half_pi * w * v.cos()) / a.abs().max(T::min_positive_val());
    if ratio <= T::min_positive_val() {
      ratio = T::min_positive_val();
    }
    let term = a * v.tan() - self.beta * ratio.ln();
    location + self.scale * two_over_pi * term
  }

  /// The scalar twin of the 8-lane Box–Muller branch.
  fn gaussian_one(&self, u1: T, u2: T) -> T {
    let u1 = Self::clamp_open_unit(u1);
    let u2 = Self::clamp_open_unit(u2);
    let z = (-T::from(2.0).unwrap() * u1.ln()).sqrt() * (T::two_pi() * u2).cos();
    self.location + self.scale * T::from(2.0).unwrap().sqrt() * z
  }

  /// The scalar twin of the 8-lane `α ≠ 1` branch, with the same cosine floor.
  fn general_one(&self, u: T, e: T) -> T {
    let alpha = self.alpha;
    let u = Self::clamp_open_unit(u);
    let e = Self::clamp_open_unit(e);
    let v = T::pi() * (u - T::from(0.5).unwrap());
    let w = -e.ln();
    let phi = alpha * (v + self.cms_b);
    let denom = v.cos().max(T::epsilon()).powf(T::one() / alpha);
    let ratio = ((v - phi).cos() / w).max(T::min_positive_val());
    let tail = ratio.powf((T::one() - alpha) / alpha);
    self.location + self.scale * self.cms_s * (phi.sin() / denom) * tail
  }

  fn fill_gaussian_branch<R: SimdRngExt>(&self, out: &mut [T], rng: &mut R) {
    let two = T::splat(T::from(2.0).unwrap());
    let pi2 = T::splat(T::two_pi());
    let scale = T::splat(self.scale * T::from(2.0).unwrap().sqrt());
    let loc = T::splat(self.location);
    let mut u1 = [T::zero(); 8];
    let mut u2 = [T::zero(); 8];
    let (chunks, rem) = out.as_chunks_mut::<8>();
    for chunk in chunks {
      T::fill_uniform_simd(rng, &mut u1);
      T::fill_uniform_simd(rng, &mut u2);
      for i in 0..8 {
        u1[i] = Self::clamp_open_unit(u1[i]);
        u2[i] = Self::clamp_open_unit(u2[i]);
      }
      let v1 = T::simd_from_array(u1);
      let v2 = T::simd_from_array(u2);
      let r = T::simd_sqrt(-two * T::simd_ln(v1));
      let z = r * T::simd_cos(pi2 * v2);
      let x = loc + scale * z;
      *chunk = T::simd_to_array(x);
    }
    if !rem.is_empty() {
      T::fill_uniform_simd(rng, &mut u1);
      T::fill_uniform_simd(rng, &mut u2);
      for i in 0..8 {
        u1[i] = Self::clamp_open_unit(u1[i]);
        u2[i] = Self::clamp_open_unit(u2[i]);
      }
      let v1 = T::simd_from_array(u1);
      let v2 = T::simd_from_array(u2);
      let r = T::simd_sqrt(-two * T::simd_ln(v1));
      let z = r * T::simd_cos(pi2 * v2);
      let x = T::simd_to_array(loc + scale * z);
      rem.copy_from_slice(&x[..rem.len()]);
    }
  }

  fn fill_alpha_not_one_branch<R: SimdRngExt>(&self, out: &mut [T], rng: &mut R) {
    let alpha = self.alpha;
    let a = T::splat(alpha);
    let b_v = T::splat(self.cms_b);
    let s_v = T::splat(self.cms_s);
    let scale = T::splat(self.scale);
    let loc = T::splat(self.location);
    let pi = T::splat(T::pi());
    let half = T::splat(T::from(0.5).unwrap());
    let inv_alpha = T::one() / alpha;
    let exp_term = (T::one() - alpha) / alpha;
    let min_pos = T::splat(T::min_positive_val());
    let cos_floor = T::splat(T::epsilon());

    let mut u = [T::zero(); 8];
    let mut e = [T::zero(); 8];
    let (chunks, rem) = out.as_chunks_mut::<8>();
    for chunk in chunks {
      T::fill_uniform_simd(rng, &mut u);
      T::fill_uniform_simd(rng, &mut e);
      for i in 0..8 {
        u[i] = Self::clamp_open_unit(u[i]);
        e[i] = Self::clamp_open_unit(e[i]);
      }

      let u_v = T::simd_from_array(u);
      let e_v = T::simd_from_array(e);
      let v = pi * (u_v - half);
      let w = -T::simd_ln(e_v);
      let phi = a * (v + b_v);
      let numer = T::simd_sin(phi);
      // `cos(v)` is positive everywhere `v` is drawn from, but the lane-wise
      // cosine is an approximation: within an ulp of ±pi/2 it can come back a
      // small negative or a denormal, and `powf` then returns a NaN or a
      // denominator that overflows the quotient — one draw in about ten
      // million in `f32`. The floor is the type's own epsilon, which sits
      // below the smallest cosine a uniform of that precision can produce, so
      // it bites only on the approximation and not on the law.
      let denom = T::simd_powf(T::simd_max(T::simd_cos(v), cos_floor), inv_alpha);
      let ratio = T::simd_max(T::simd_cos(v - phi) / w, min_pos);
      let tail = T::simd_powf(ratio, exp_term);
      let x = loc + scale * s_v * (numer / denom) * tail;
      *chunk = T::simd_to_array(x);
    }
    if !rem.is_empty() {
      T::fill_uniform_simd(rng, &mut u);
      T::fill_uniform_simd(rng, &mut e);
      for i in 0..8 {
        u[i] = Self::clamp_open_unit(u[i]);
        e[i] = Self::clamp_open_unit(e[i]);
      }
      let u_v = T::simd_from_array(u);
      let e_v = T::simd_from_array(e);
      let v = pi * (u_v - half);
      let w = -T::simd_ln(e_v);
      let phi = a * (v + b_v);
      let numer = T::simd_sin(phi);
      // The same cosine floor as above.
      let denom = T::simd_powf(T::simd_max(T::simd_cos(v), cos_floor), inv_alpha);
      let ratio = T::simd_max(T::simd_cos(v - phi) / w, min_pos);
      let tail = T::simd_powf(ratio, exp_term);
      let x = T::simd_to_array(loc + scale * s_v * (numer / denom) * tail);
      rem.copy_from_slice(&x[..rem.len()]);
    }
  }

  fn fill_alpha_one_branch<R: SimdRngExt>(&self, out: &mut [T], rng: &mut R) {
    let location = self.unit_index_location();
    for x in out.iter_mut() {
      let u = T::sample_uniform_simd(rng);
      let e = T::sample_uniform_simd(rng);
      *x = self.unit_index_one(u, e, location);
    }
  }

  fn fill_parts<R: SimdRngExt>(&self, rng: &mut R, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    match self.branch() {
      Branch::Gaussian => self.fill_gaussian_branch(out, rng),
      Branch::UnitIndex => self.fill_alpha_one_branch(out, rng),
      Branch::General => self.fill_alpha_not_one_branch(out, rng),
    }
  }

  pub(crate) fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    let u = T::sample_uniform(rng);
    let e = T::sample_uniform(rng);
    match self.branch() {
      Branch::Gaussian => self.gaussian_one(u, e),
      Branch::UnitIndex => self.unit_index_one(u, e, self.unit_index_location()),
      Branch::General => self.general_one(u, e),
    }
  }
}

impl<T: SimdFloatExt> Sealed for SimdAlphaStable<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdAlphaStable<T> {
  type State<R: SimdRngExt> = StreamState<T, R, 16>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (StreamState<T, R, 16>, u64) {
    StreamState::init(seed)
  }
}

impl<T: SimdFloatExt> SimdKernel for SimdAlphaStable<T> {
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

impl<T: SimdFloatExt> Distribution<T> for SimdAlphaStable<T> {
  /// One scalar Chambers–Mallows–Stuck draw from two uniforms of the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.draw_with(rng)
  }
}

impl<T: SimdFloatExt> crate::traits::DistributionExt for SimdAlphaStable<T> {
  fn mean(&self) -> Option<f64> {
    let alpha = self.alpha.to_f64().unwrap();
    if alpha > 1.0 {
      Some(self.location.to_f64().unwrap())
    } else {
      Some(f64::NAN)
    }
  }

  /// The location when β = 0, where the law is symmetric and unimodal; no closed form otherwise.
  fn median(&self) -> Option<f64> {
    if self.beta.to_f64().unwrap() == 0.0 {
      Some(self.location.to_f64().unwrap())
    } else {
      None
    }
  }

  fn mode(&self) -> Option<f64> {
    if self.beta.to_f64().unwrap() == 0.0 {
      Some(self.location.to_f64().unwrap())
    } else {
      None
    }
  }

  fn variance(&self) -> Option<f64> {
    let alpha = self.alpha.to_f64().unwrap();
    if alpha == 2.0 {
      // Gaussian limit: σ² = 2 c²
      let c = self.scale.to_f64().unwrap();
      Some(2.0 * c * c)
    } else {
      Some(f64::INFINITY)
    }
  }

  fn skewness(&self) -> Option<f64> {
    if self.alpha.to_f64().unwrap() == 2.0 {
      Some(0.0)
    } else {
      Some(f64::NAN)
    }
  }

  fn kurtosis(&self) -> Option<f64> {
    if self.alpha.to_f64().unwrap() == 2.0 {
      Some(0.0)
    } else {
      Some(f64::NAN)
    }
  }

  fn characteristic_function(&self, t: f64) -> Option<num_complex::Complex64> {
    // Standard S1 parameterisation:
    //   φ(t) = exp{ iμt − |c·t|^α [ 1 − iβ sgn(t) Φ ] }
    // where Φ = tan(πα/2) for α ≠ 1, and Φ = −(2/π) ln|t| for α = 1.
    let alpha = self.alpha.to_f64().unwrap();
    let beta = self.beta.to_f64().unwrap();
    let c = self.scale.to_f64().unwrap();
    let mu = self.location.to_f64().unwrap();
    let abs_ct_alpha = (c * t.abs()).powf(alpha);
    let sgn_t = t.signum();
    let phi = if (alpha - 1.0).abs() < 1e-15 {
      -(2.0 / std::f64::consts::PI) * t.abs().ln()
    } else {
      (std::f64::consts::PI * alpha / 2.0).tan()
    };
    let bracket = num_complex::Complex64::new(1.0, -beta * sgn_t * phi);
    let exponent = num_complex::Complex64::new(0.0, mu * t) - bracket.scale(abs_ct_alpha);
    Some(exponent.exp())
  }

  fn moment_generating_function(&self, t: f64) -> Option<f64> {
    if t == 0.0 {
      Some(1.0)
    } else if self.alpha.to_f64().unwrap() == 2.0 {
      // Gaussian limit: M(t) = exp(μt + c²t²)
      let mu = self.location.to_f64().unwrap();
      let c = self.scale.to_f64().unwrap();
      Some((mu * t + c * c * t * t).exp())
    } else {
      // MGF only exists in the Gaussian limit (α = 2).
      Some(f64::NAN)
    }
  }
}

py_distribution!(PyAlphaStable, SimdAlphaStable,
  sig: (alpha, beta, scale, location, seed=None, dtype=None),
  params: (alpha: f64, beta: f64, scale: f64, location: f64)
);

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;
  use crate::normal::SimdNormal;
  use crate::tests::assert_ecf_matches;
  use crate::tests::scalar_draws;
  use crate::tests::scalar_ks_best_p;
  use crate::traits::DistributionExt;
  use crate::traits::DistributionSampler;

  /// Every single-precision draw is finite.
  ///
  /// The Chambers-Mallows-Stuck form divides by `cos(v)^(1/alpha)`, and `v`
  /// is drawn from the open interval where that cosine is positive — but the
  /// lane-wise cosine is an approximation, and within an ulp of ±pi/2 it can
  /// return a small negative, which turns the fractional power into a NaN.
  /// The rate is about one draw in ten million, so the seed is pinned to one
  /// that hits it: seed 2 fails 352 453 draws in, whatever `alpha` is, since
  /// the offending value is in the uniform stream rather than the law. A
  /// Levy fractional stable motion path turning to NaN is how it first
  /// showed up.
  #[test]
  fn single_precision_draws_stay_finite() {
    for alpha in [0.9_f64, 1.1, 1.5, 1.7, 1.9] {
      let dist = SimdAlphaStable::<f32>::new(alpha as f32, 0.2, 1.0, 0.0);
      let mut out = vec![0.0_f32; 360_000];
      dist.seeded(&Deterministic::new(2)).fill_slice(&mut out);
      let bad = out.iter().filter(|x| !x.is_finite()).count();
      assert_eq!(
        bad,
        0,
        "alpha = {alpha}: {bad} non-finite draws of {}",
        out.len()
      );
    }
  }

  #[test]
  fn alpha_stable_samples_are_finite() {
    let dist = SimdAlphaStable::<f64>::new(1.7_f64, 0.3, 1.0, 0.0);
    let mut xs = vec![0.0_f64; 1024];
    dist.seeded(&Deterministic::new(0xa1fa)).fill_slice(&mut xs);
    assert!(xs.iter().all(|x| x.is_finite()));
  }

  /// The unit-index case has `β ≠ 0` and `σ ≠ 1`, where Weron's `(2/π)βσ ln σ` shift is visible.
  #[test]
  fn scalar_sample_matches_the_characteristic_function() {
    for (alpha, beta, scale, location) in [
      (1.7, 0.3, 1.0, 0.0),
      (0.8, -0.6, 1.5, 0.4),
      (1.0, 0.5, 2.0, 0.3),
    ] {
      let d = SimdAlphaStable::<f64>::new(alpha, beta, scale, location);
      let xs = scalar_draws(&d, 11, 200_000);
      assert_ecf_matches(
        &xs,
        |u| d.characteristic_function(u).unwrap(),
        &format!("α={alpha}"),
      );
    }
  }

  #[test]
  fn unit_index_stream_matches_the_characteristic_function() {
    let d = SimdAlphaStable::<f64>::new(1.0, 0.5, 2.0, 0.3);
    let mut xs = vec![0.0; 200_000];
    d.seeded(&Deterministic::new(5)).fill_slice(&mut xs);
    assert_ecf_matches(&xs, |u| d.characteristic_function(u).unwrap(), "seeded α=1");
  }

  #[test]
  fn scalar_gaussian_index_matches_the_normal_cdf() {
    let d = SimdAlphaStable::<f64>::new(2.0, 0.0, 1.0, 0.0);
    let normal = SimdNormal::<f64>::new(0.0, std::f64::consts::SQRT_2);
    let best = scalar_ks_best_p(&d, |x| normal.cdf(x).unwrap());
    assert!(best > 0.01, "best p = {best}");
  }
}

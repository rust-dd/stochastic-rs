//! # Alpha Stable
//!
//! $$
//! \varphi_X(u)=\exp\!\left(i\delta u-\gamma^\alpha |u|^\alpha\left[1-i\beta\operatorname{sgn}(u)\omega(u,\alpha)\right]\right)
//! $$
//!
use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;

use super::SimdFloatExt;
use crate::simd_rng::SimdRng;

/// SIMD-backed alpha-stable distribution sampled with the
/// Chambers-Mallows-Stuck method.
///
/// Parameters follow a common `(alpha, beta, scale, location)` form:
/// - `alpha in (0, 2]` is the stability index
/// - `beta in [-1, 1]` is the skewness
/// - `scale > 0`
/// - `location` is the shift
pub struct SimdAlphaStable<T: SimdFloatExt> {
  alpha: T,
  beta: T,
  scale: T,
  location: T,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  simd_rng: UnsafeCell<SimdRng>,
}

impl<T: SimdFloatExt> SimdAlphaStable<T> {
  pub fn new(alpha: T, beta: T, scale: T, location: T) -> Self {
    assert!(alpha > T::zero() && alpha <= T::from(2.0).unwrap());
    assert!((-T::one()..=T::one()).contains(&beta));
    assert!(scale > T::zero());
    Self {
      alpha,
      beta,
      scale,
      location,
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
      simd_rng: UnsafeCell::new(SimdRng::new()),
    }
  }

  pub fn fill_slice<R: Rng + ?Sized>(&self, _rng: &mut R, out: &mut [T]) {
    self.fill_slice_fast(out);
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

  fn fill_gaussian_branch(&self, out: &mut [T], rng: &mut SimdRng) {
    let two = T::splat(T::from(2.0).unwrap());
    let pi2 = T::splat(T::two_pi());
    let scale = T::splat(self.scale * T::from(2.0).unwrap().sqrt());
    let loc = T::splat(self.location);
    let mut u1 = [T::zero(); 8];
    let mut u2 = [T::zero(); 8];
    let mut chunks = out.chunks_exact_mut(8);
    for chunk in &mut chunks {
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
      chunk.copy_from_slice(&T::simd_to_array(x));
    }
    let rem = chunks.into_remainder();
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

  fn fill_alpha_not_one_branch(&self, out: &mut [T], rng: &mut SimdRng) {
    let alpha = self.alpha;
    let beta = self.beta;
    let tan_term = (T::from_f64_fast(std::f64::consts::PI) * alpha / T::from(2.0).unwrap()).tan();
    let beta_tan = beta * tan_term;
    let b = (beta_tan).atan() / alpha;
    let s = (T::one() + beta_tan * beta_tan).powf(T::one() / (T::from(2.0).unwrap() * alpha));

    let a = T::splat(alpha);
    let b_v = T::splat(b);
    let s_v = T::splat(s);
    let scale = T::splat(self.scale);
    let loc = T::splat(self.location);
    let pi = T::splat(T::pi());
    let half = T::splat(T::from(0.5).unwrap());
    let inv_alpha = T::one() / alpha;
    let exp_term = (T::one() - alpha) / alpha;
    let min_pos = T::splat(T::min_positive_val());

    let mut u = [T::zero(); 8];
    let mut e = [T::zero(); 8];
    let mut chunks = out.chunks_exact_mut(8);
    for chunk in &mut chunks {
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
      let denom = T::simd_powf(T::simd_cos(v), inv_alpha);
      let ratio = T::simd_max(T::simd_cos(v - phi) / w, min_pos);
      let tail = T::simd_powf(ratio, exp_term);
      let x = loc + scale * s_v * (numer / denom) * tail;
      chunk.copy_from_slice(&T::simd_to_array(x));
    }
    let rem = chunks.into_remainder();
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
      let denom = T::simd_powf(T::simd_cos(v), inv_alpha);
      let ratio = T::simd_max(T::simd_cos(v - phi) / w, min_pos);
      let tail = T::simd_powf(ratio, exp_term);
      let x = T::simd_to_array(loc + scale * s_v * (numer / denom) * tail);
      rem.copy_from_slice(&x[..rem.len()]);
    }
  }

  fn fill_alpha_one_branch(&self, out: &mut [T], rng: &mut SimdRng) {
    let pi = T::from_f64_fast(std::f64::consts::PI);
    let half_pi = pi / T::from(2.0).unwrap();
    let two_over_pi = T::from(2.0).unwrap() / pi;
    let beta = self.beta;
    let scale = self.scale;
    let loc = self.location;
    for x in out.iter_mut() {
      let mut u = T::sample_uniform_simd(rng);
      let mut e = T::sample_uniform_simd(rng);
      u = Self::clamp_open_unit(u);
      e = Self::clamp_open_unit(e);
      let v = pi * (u - T::from(0.5).unwrap());
      let w = -e.ln();
      let a = half_pi + beta * v;
      let mut ratio = (half_pi * w * v.cos()) / a.abs().max(T::min_positive_val());
      if ratio <= T::min_positive_val() {
        ratio = T::min_positive_val();
      }
      let term = a * v.tan() - beta * ratio.ln();
      *x = loc + scale * two_over_pi * term;
    }
  }

  pub fn fill_slice_fast(&self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }

    let rng = unsafe { &mut *self.simd_rng.get() };
    let eps = T::from(1e-6).unwrap();

    if (self.alpha - T::from(2.0).unwrap()).abs() < eps {
      self.fill_gaussian_branch(out, rng);
      return;
    }

    if (self.alpha - T::one()).abs() < eps {
      self.fill_alpha_one_branch(out, rng);
      return;
    }

    self.fill_alpha_not_one_branch(out, rng);
  }

  fn refill_buffer(&self) {
    let buf = unsafe { &mut *self.buffer.get() };
    self.fill_slice_fast(buf);
    unsafe {
      *self.index.get() = 0;
    }
  }
}

impl<T: SimdFloatExt> Clone for SimdAlphaStable<T> {
  fn clone(&self) -> Self {
    Self::new(self.alpha, self.beta, self.scale, self.location)
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdAlphaStable<T> {
  fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> T {
    let idx = unsafe { &mut *self.index.get() };
    if *idx >= 16 {
      self.refill_buffer();
    }
    let val = unsafe { (*self.buffer.get())[*idx] };
    *idx += 1;
    val
  }
}

py_distribution!(PyAlphaStable, SimdAlphaStable,
  sig: (alpha, beta, scale, location, dtype=None),
  params: (alpha: f64, beta: f64, scale: f64, location: f64)
);

#[cfg(test)]
mod tests {
  use rand_distr::Distribution;

  use super::*;

  #[test]
  fn alpha_stable_samples_are_finite() {
    let dist = SimdAlphaStable::new(1.7_f64, 0.3, 1.0, 0.0);
    let mut rng = rand::rng();
    for _ in 0..1024 {
      let x = dist.sample(&mut rng);
      assert!(x.is_finite());
    }
  }
}

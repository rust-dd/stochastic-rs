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
//! - Subbotin, M.T. (1923), "On the law of frequency of error",
//!   *Matematicheskii Sbornik* 31, 296-301.
//! - Nelson, D.B. (1991), "Conditional heteroskedasticity in asset
//!   returns: a new approach", *Econometrica* 59, 347-370.

use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::gamma::SimdGamma;
use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;
use crate::traits::DistributionExt;
use crate::traits::SimdFloatExt;

const SMALL_GED_THRESHOLD: usize = 16;

pub struct SimdGed<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  mu: T,
  alpha: T,
  beta: T,
  gamma: SimdGamma<T, R>,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  simd_rng: UnsafeCell<R>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdGed<T, R> {
  pub fn new<S: SeedExt>(mu: T, alpha: T, beta: T, seed: &S) -> Self {
    assert!(alpha > T::zero(), "α must be > 0");
    assert!(beta > T::zero(), "β must be > 0");
    let inv_beta = T::one() / beta;
    Self {
      mu,
      alpha,
      beta,
      gamma: SimdGamma::<T, R>::new(inv_beta, T::one(), seed),
      buffer: UnsafeCell::new([T::zero(); 16]),
      index: UnsafeCell::new(16),
      simd_rng: UnsafeCell::new(seed.rng_ext::<R>()),
    }
  }

  /// Returns a single sample using the internal SIMD RNG.
  /// Draws from a pre-filled buffer of $X = \alpha \cdot Y^{1/\beta} \cdot S + \mu$ values.
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

  pub fn fill_slice<Rr: Rng + ?Sized>(&self, _rng: &mut Rr, out: &mut [T]) {
    self.fill_slice_fast(out);
  }

  /// Fills `out` via the gamma magnitude + random-sign bijection. Magnitudes
  /// come from the bulk gamma fill and the $Y^{1/\beta}$ power runs 8-wide;
  /// the sign bit is taken from the internal SIMD RNG's integer stream.
  pub fn fill_slice_fast(&self, out: &mut [T]) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    let inv_beta = T::one() / self.beta;
    if out.len() < SMALL_GED_THRESHOLD {
      for x in out.iter_mut() {
        let mag = self.gamma.sample_fast().powf(inv_beta);
        let signed = if rng.next_i32() >= 0 { mag } else { -mag };
        *x = self.alpha * signed + self.mu;
      }
      return;
    }
    let alpha = T::splat(self.alpha);
    let mut ybuf = [T::zero(); 64];
    let mut chunks = out.chunks_exact_mut(64);
    for chunk in &mut chunks {
      self.gamma.fill_slice_fast(&mut ybuf);
      for (sub, y8) in chunk.chunks_exact_mut(8).zip(ybuf.chunks_exact(8)) {
        let mut a = [T::zero(); 8];
        a.copy_from_slice(y8);
        let mag = T::simd_powf(T::simd_from_array(a), inv_beta);
        let t = T::simd_to_array(alpha * mag);
        let signs = rng.next_i32x8().to_array();
        for i in 0..8 {
          sub[i] = self.mu + if signs[i] >= 0 { t[i] } else { -t[i] };
        }
      }
    }
    for x in chunks.into_remainder().iter_mut() {
      let mag = self.gamma.sample_fast().powf(inv_beta);
      let signed = if rng.next_i32() >= 0 { mag } else { -mag };
      *x = self.alpha * signed + self.mu;
    }
  }

  fn refill_buffer(&self) {
    let buf = unsafe { &mut *self.buffer.get() };
    self.fill_slice_fast(buf);
    unsafe {
      *self.index.get() = 0;
    }
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdGed<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.mu, self.alpha, self.beta, &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Distribution<T> for SimdGed<T, R> {
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

impl<T: SimdFloatExt, R: SimdRngExt> DistributionExt for SimdGed<T, R> {
  fn pdf(&self, x: f64) -> f64 {
    let mu = self.mu.to_f64().unwrap();
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let z = ((x - mu) / a).abs();
    let log_pdf = b.ln() - (2.0 * a).ln() - crate::special::ln_gamma(1.0 / b) - z.powf(b);
    log_pdf.exp()
  }

  fn cdf(&self, x: f64) -> f64 {
    let mu = self.mu.to_f64().unwrap();
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let z = (x - mu) / a;
    // F(x) = 1/2 + sign(x - μ)/2 · γ(1/β, |z|^β) / Γ(1/β)
    let zb = z.abs().powf(b);
    let half_inc = 0.5 * crate::special::gamma_p(1.0 / b, zb);
    if z >= 0.0 {
      0.5 + half_inc
    } else {
      0.5 - half_inc
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// GED with β=2 should collapse to a Gaussian with std-dev α·√(1/2).
  /// Tested via sample variance.
  #[test]
  fn ged_beta_two_is_gaussian() {
    let g = SimdGed::<f64>::new(0.0, std::f64::consts::SQRT_2, 2.0, &Unseeded);
    let n = 30_000;
    let mut sum_sq = 0.0;
    let mut sum = 0.0;
    for _ in 0..n {
      let x = g.sample_fast();
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

  /// PDF normalises to 1 (numeric integration).
  #[test]
  fn ged_pdf_normalised() {
    let g = SimdGed::<f64>::new(0.0, 1.0, 1.5, &Unseeded);
    let n = 5000;
    let lo = -20.0_f64;
    let up = 20.0_f64;
    let h = (up - lo) / n as f64;
    let s: f64 = (0..n).map(|k| g.pdf(lo + (k as f64 + 0.5) * h) * h).sum();
    assert!(
      (s - 1.0).abs() < 1e-3,
      "GED(0, 1, 1.5) PDF integrates to {s}"
    );
  }

  /// CDF round-trip via PDF integration.
  #[test]
  fn ged_cdf_at_mu_is_half() {
    let g = SimdGed::<f64>::new(1.5, 0.8, 1.7, &Unseeded);
    let c = g.cdf(1.5);
    assert!(
      (c - 0.5).abs() < 1e-10,
      "GED(1.5, 0.8, 1.7) CDF at μ = {c}, expected 0.5"
    );
  }
}

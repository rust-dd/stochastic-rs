//! # Uniform
//!
//! $$
//! f(x)=\frac{1}{b-a}\mathbf{1}_{a\le x\le b}
//! $$
//!
//! Sampling: inversion, Devroye, L. (1986), *Non-Uniform Random Variate Generation*, Springer, §II.2, DOI 10.1007/978-1-4613-8643-8.

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use crate::seeded::StreamState;
use crate::traits::SimdFloatExt;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

const SMALL_UNIFORM_THRESHOLD: usize = 16;

/// Uniform law on `[low, high)`: parameters only; a [`Seeded`](crate::Seeded) stream draws it in bulk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdUniform<T> {
  low: T,
  high: T,
  scale: T,
}

impl<T: SimdFloatExt> SimdUniform<T> {
  /// # Panics
  /// `high <= low` or a bound that is not finite.
  pub fn new(low: T, high: T) -> Self {
    assert!(
      low.is_finite(),
      "low must satisfy `low.is_finite()`, got low = {low:?}"
    );
    assert!(
      high.is_finite(),
      "high must satisfy `high.is_finite()`, got high = {high:?}"
    );
    assert!(
      low < high,
      "low must satisfy `low < high`, got low = {low:?}, high = {high:?}"
    );
    Self {
      low,
      high,
      scale: high - low,
    }
  }

  pub fn unit() -> Self {
    Self::new(T::zero(), T::one())
  }

  /// The lower bound `a`.
  pub fn low(&self) -> T {
    self.low
  }

  /// The upper bound `b`.
  pub fn high(&self) -> T {
    self.high
  }

  fn fill_parts<R: SimdRngExt>(&self, rng: &mut R, out: &mut [T]) {
    if out.len() < SMALL_UNIFORM_THRESHOLD {
      for x in out.iter_mut() {
        *x = self.low + self.scale * T::sample_uniform_simd(rng);
      }
      return;
    }
    // [0, 1) fast path: fill the whole output with U(0, 1) via direct SIMD
    // stores (one engine call per 4-lane f64 chunk / 8-lane f32 chunk).
    if self.low.is_zero() && self.scale == T::one() {
      T::fill_uniform_simd(rng, out);
      return;
    }
    // Affine path: generate U(0, 1) into a 256-element stack block (one
    // tight engine loop, stays in L1), then apply the transform 8-wide on
    // the way out — `out` is written exactly once instead of the previous
    // fill-then-read-modify-write double pass over the whole slice.
    let low = T::splat(self.low);
    let scale = T::splat(self.scale);
    let mut tmp = [T::zero(); 256];
    let (chunks, rem) = out.as_chunks_mut::<256>();
    for chunk in chunks {
      T::fill_uniform_simd(rng, &mut tmp);
      for (sub, u8) in chunk
        .as_chunks_mut::<8>()
        .0
        .iter_mut()
        .zip(tmp.as_chunks::<8>().0.iter())
      {
        *sub = T::simd_to_array(low + T::simd_from_array(*u8) * scale);
      }
    }
    if !rem.is_empty() {
      let n = rem.len();
      T::fill_uniform_simd(rng, &mut tmp[..n]);
      let mut off = 0;
      let (sub, sub_rem) = rem.as_chunks_mut::<8>();
      for s in sub.iter_mut() {
        let mut a = [T::zero(); 8];
        a.copy_from_slice(&tmp[off..off + 8]);
        *s = T::simd_to_array(low + T::simd_from_array(a) * scale);
        off += 8;
      }
      for (i, x) in sub_rem.iter_mut().enumerate() {
        *x = self.low + tmp[off + i] * self.scale;
      }
    }
  }

  pub(crate) fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.low + self.scale * T::sample_uniform(rng)
  }
}

/// U(0, 1) — matches this file's own [`SimdUniform::unit`] constructor for
/// the standard uniform distribution.
impl<T: SimdFloatExt> Default for SimdUniform<T> {
  fn default() -> Self {
    Self::unit()
  }
}

impl<T: SimdFloatExt> Sealed for SimdUniform<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdUniform<T> {
  type State<R: SimdRngExt> = StreamState<T, R, 16>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (StreamState<T, R, 16>, u64) {
    StreamState::init(seed)
  }
}

impl<T: SimdFloatExt> SimdKernel for SimdUniform<T> {
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

impl<T: SimdFloatExt> Distribution<T> for SimdUniform<T> {
  /// `low + scale·u` with one `[0, 1)` uniform `u` from the caller's rng (53 bits for `f64`, 24 for `f32`).
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.draw_with(rng)
  }
}

impl<T: SimdFloatExt> crate::traits::DistributionExt for SimdUniform<T> {
  fn pdf(&self, x: f64) -> Option<f64> {
    let a = self.low.to_f64().unwrap();
    let b = self.high.to_f64().unwrap();
    if x >= a && x <= b {
      Some(1.0 / (b - a))
    } else {
      Some(0.0)
    }
  }

  fn cdf(&self, x: f64) -> Option<f64> {
    let a = self.low.to_f64().unwrap();
    let b = self.high.to_f64().unwrap();
    if x < a {
      Some(0.0)
    } else if x >= b {
      Some(1.0)
    } else {
      Some((x - a) / (b - a))
    }
  }

  fn quantile(&self, p: f64) -> Option<f64> {
    let a = self.low.to_f64().unwrap();
    let scale = self.scale.to_f64().unwrap();
    Some(a + p * scale)
  }

  fn mean(&self) -> Option<f64> {
    Some(self.low.to_f64().unwrap() + 0.5 * self.scale.to_f64().unwrap())
  }

  fn median(&self) -> Option<f64> {
    self.mean()
  }

  fn mode(&self) -> Option<f64> {
    // Any point in [a, b] is a mode; report the midpoint.
    self.mean()
  }

  fn variance(&self) -> Option<f64> {
    let scale = self.scale.to_f64().unwrap();
    Some(scale * scale / 12.0)
  }

  fn skewness(&self) -> Option<f64> {
    Some(0.0)
  }

  fn kurtosis(&self) -> Option<f64> {
    // Excess kurtosis.
    Some(-6.0 / 5.0)
  }

  fn entropy(&self) -> Option<f64> {
    Some(self.scale.to_f64().unwrap().ln())
  }

  fn characteristic_function(&self, t: f64) -> Option<num_complex::Complex64> {
    // φ(t) = (e^{itb} - e^{ita}) / (it(b-a))
    let a = self.low.to_f64().unwrap();
    let b = self.high.to_f64().unwrap();
    if t == 0.0 {
      return Some(num_complex::Complex64::new(1.0, 0.0));
    }
    let eitb = num_complex::Complex64::new(0.0, t * b).exp();
    let eita = num_complex::Complex64::new(0.0, t * a).exp();
    Some((eitb - eita) / num_complex::Complex64::new(0.0, t * (b - a)))
  }

  fn moment_generating_function(&self, t: f64) -> Option<f64> {
    let a = self.low.to_f64().unwrap();
    let b = self.high.to_f64().unwrap();
    if t == 0.0 {
      return Some(1.0);
    }
    Some(((b * t).exp() - (a * t).exp()) / (t * (b - a)))
  }
}

py_distribution!(PyUniform, SimdUniform,
  sig: (low, high, seed=None, dtype=None),
  params: (low: f64, high: f64)
);

#[cfg(test)]
mod tests {
  use super::*;
  use crate::tests::scalar_ks_best_p;
  use crate::traits::DistributionExt as _;

  /// The honest `Distribution` draws from the caller's rng and agrees with the cdf.
  #[test]
  fn scalar_sample_matches_cdf() {
    let d = SimdUniform::<f64>::new(-2.0, 3.0);
    let best = scalar_ks_best_p(&d, |x| d.cdf(x).unwrap());
    assert!(best > 0.01, "best p = {best}");
  }

  /// At the rng's largest output the honest `f32` draw stays below `high`, where a 53-bit uniform rounds up to one.
  #[test]
  fn honest_f32_draw_stays_below_high() {
    struct Top;
    impl rand::TryRng for Top {
      type Error = std::convert::Infallible;

      fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok(u32::MAX)
      }

      fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        Ok(u64::MAX)
      }

      fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        dst.fill(u8::MAX);
        Ok(())
      }
    }
    assert!(SimdUniform::<f32>::unit().sample(&mut Top) < 1.0);
  }
}

//! # Beta
//!
//! $$
//! f(x)=\frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)},\ x\in(0,1)
//! $$
//!
//! Sampling: `G1/(G1 + G2)` of two gammas, Devroye, L. (1986), *Non-Uniform Random Variate Generation*, Springer, §IX.4, DOI 10.1007/978-1-4613-8643-8.

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use super::SimdFloatExt;
use super::gamma::GammaState;
use super::gamma::SimdGamma;
use crate::seeded::Buffered;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

const SMALL_BETA_THRESHOLD: usize = 16;

/// Beta law `Beta(alpha, beta)`: parameters only; a [`Seeded`](crate::Seeded) stream draws it in bulk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdBeta<T> {
  alpha: T,
  beta: T,
  gamma1: SimdGamma<T>,
  gamma2: SimdGamma<T>,
}

/// A beta stream: the two gamma sub-streams and the single-draw buffer.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct BetaState<T: SimdFloatExt, R: SimdRngExt> {
  g1: GammaState<T, R>,
  g2: GammaState<T, R>,
  buf: Buffered<T, 16>,
}

impl<T: SimdFloatExt> SimdBeta<T> {
  /// Creates a beta distribution via the Gamma-ratio construction
  /// `X = G1/(G1+G2)`, `G1 ~ Gamma(alpha, 1)`, `G2 ~ Gamma(beta, 1)`.
  ///
  /// - `alpha` — first shape α > 0 (matches the module header's α).
  /// - `beta` — second shape β > 0 (matches the module header's β; NOT
  ///   the asymmetry role `beta` plays in
  ///   [`SimdNormalInverseGauss`](crate::normal_inverse_gauss::SimdNormalInverseGauss)
  ///   or the tail exponent it plays in [`SimdGed`](crate::ged::SimdGed)
  ///   — same word, unrelated role in each).
  pub fn new(alpha: T, beta: T) -> Self {
    assert!(
      alpha > T::zero(),
      "alpha must satisfy `alpha > T::zero()`, got alpha = {alpha:?}"
    );
    assert!(
      beta > T::zero(),
      "beta must satisfy `beta > T::zero()`, got beta = {beta:?}"
    );
    Self {
      alpha,
      beta,
      gamma1: SimdGamma::new(alpha, T::one()),
      gamma2: SimdGamma::new(beta, T::one()),
    }
  }

  /// The first shape `α`.
  pub fn alpha(&self) -> T {
    self.alpha
  }

  /// The second shape `β`.
  pub fn beta(&self) -> T {
    self.beta
  }

  /// The ratio recovered from logs, for the draws where both Gamma
  /// marginals underflowed to exactly zero and `a / (a + b)` is `0/0`.
  ///
  /// Small shapes make that ordinary rather than exotic: `Beta(0.01, 0.01)`
  /// loses a draw this way once in eight in single precision and once in
  /// five million in double. The ratio is scale-free, so re-forming it
  /// from a fresh pair of log draws is exact wherever the value path was,
  /// and where the log difference saturates `exp` the answer is the 0 or
  /// the 1 that rounding the true draw would have given.
  #[cold]
  #[inline(never)]
  fn ratio_from_logs<R: SimdRngExt>(
    &self,
    g1: &mut GammaState<T, R>,
    g2: &mut GammaState<T, R>,
  ) -> T {
    let la = self.gamma1.next_log(g1);
    let lb = self.gamma2.next_log(g2);
    T::one() / (T::one() + (lb - la).exp())
  }

  fn fill_parts<R: SimdRngExt>(
    &self,
    g1: &mut GammaState<T, R>,
    g2: &mut GammaState<T, R>,
    out: &mut [T],
  ) {
    if out.len() < SMALL_BETA_THRESHOLD {
      for x in out.iter_mut() {
        let a = self.gamma1.next(g1);
        let b = self.gamma2.next(g2);
        *x = if a + b > T::zero() {
          a / (a + b)
        } else {
          self.ratio_from_logs(g1, g2)
        };
      }
      return;
    }
    let mut ga = [T::zero(); 64];
    let mut gb = [T::zero(); 64];
    let (chunks, rem) = out.as_chunks_mut::<64>();
    for chunk in chunks {
      self.gamma1.fill(g1, &mut ga);
      self.gamma2.fill(g2, &mut gb);
      for (sub, (a8, b8)) in chunk.as_chunks_mut::<8>().0.iter_mut().zip(
        ga.as_chunks::<8>()
          .0
          .iter()
          .zip(gb.as_chunks::<8>().0.iter()),
      ) {
        let a = T::simd_from_array(*a8);
        let b = T::simd_from_array(*b8);
        *sub = T::simd_to_array(a / (a + b));
        for (j, x) in sub.iter_mut().enumerate() {
          if a8[j] + b8[j] <= T::zero() {
            *x = self.ratio_from_logs(g1, g2);
          }
        }
      }
    }
    if !rem.is_empty() {
      let n = rem.len();
      self.gamma1.fill(g1, &mut ga[..n]);
      self.gamma2.fill(g2, &mut gb[..n]);
      for i in 0..n {
        rem[i] = if ga[i] + gb[i] > T::zero() {
          ga[i] / (ga[i] + gb[i])
        } else {
          self.ratio_from_logs(g1, g2)
        };
      }
    }
  }

  pub(crate) fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    let a = self.gamma1.draw_with(rng);
    let b = self.gamma2.draw_with(rng);
    if a + b > T::zero() {
      return a / (a + b);
    }
    let la = self.gamma1.draw_log_with(rng);
    let lb = self.gamma2.draw_log_with(rng);
    T::one() / (T::one() + (lb - la).exp())
  }
}

impl<T: SimdFloatExt> Sealed for SimdBeta<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdBeta<T> {
  type State<R: SimdRngExt> = BetaState<T, R>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (BetaState<T, R>, u64) {
    let (g1, basis) = self.gamma1.init::<R, S>(seed);
    let (g2, _) = self.gamma2.init::<R, S>(seed);
    (
      BetaState {
        g1,
        g2,
        buf: Buffered::new(),
      },
      basis,
    )
  }
}

impl<T: SimdFloatExt> SimdKernel for SimdBeta<T> {
  type Item = T;

  #[inline]
  fn fill<R: SimdRngExt>(&self, state: &mut BetaState<T, R>, out: &mut [T]) {
    self.fill_parts(&mut state.g1, &mut state.g2, out);
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut BetaState<T, R>) -> T {
    let BetaState { g1, g2, buf } = state;
    buf.pop(|b| self.fill_parts(g1, g2, b))
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdBeta<T> {
  /// `G1/(G1 + G2)` of two scalar gamma draws on the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.draw_with(rng)
  }
}

impl<T: SimdFloatExt> crate::traits::DistributionExt for SimdBeta<T> {
  fn pdf(&self, x: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&x) {
      return Some(0.0);
    }
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let log_pdf = (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - crate::special::ln_beta(a, b);
    Some(log_pdf.exp())
  }

  fn cdf(&self, x: f64) -> Option<f64> {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    Some(crate::special::beta_i(a, b, x.clamp(0.0, 1.0)))
  }

  fn quantile(&self, p: f64) -> Option<f64> {
    if p <= 0.0 {
      return Some(0.0);
    }
    if p >= 1.0 {
      return Some(1.0);
    }
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    // Newton on f(x) = I_x(a,b) − p with f'(x) = pdf.
    let mut x = a / (a + b); // start at the mean
    for _ in 0..60 {
      let f = crate::special::beta_i(a, b, x) - p;
      let log_pdf = (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - crate::special::ln_beta(a, b);
      let pdf = log_pdf.exp();
      if pdf <= 0.0 {
        break;
      }
      let dx = f / pdf;
      let new_x = (x - dx).clamp(1e-14, 1.0 - 1e-14);
      if (new_x - x).abs() < 1e-14 {
        return Some(new_x);
      }
      x = new_x;
    }
    Some(x)
  }

  fn mean(&self) -> Option<f64> {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    Some(a / (a + b))
  }

  fn median(&self) -> Option<f64> {
    self.quantile(0.5)
  }

  fn mode(&self) -> Option<f64> {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    if a > 1.0 && b > 1.0 {
      Some((a - 1.0) / (a + b - 2.0))
    } else {
      Some(f64::NAN)
    }
  }

  fn variance(&self) -> Option<f64> {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let s = a + b;
    Some(a * b / (s * s * (s + 1.0)))
  }

  fn skewness(&self) -> Option<f64> {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let s = a + b;
    Some(2.0 * (b - a) * (s + 1.0).sqrt() / ((s + 2.0) * (a * b).sqrt()))
  }

  fn kurtosis(&self) -> Option<f64> {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    let s = a + b;
    let num = 6.0 * ((a - b).powi(2) * (s + 1.0) - a * b * (s + 2.0));
    let den = a * b * (s + 2.0) * (s + 3.0);
    Some(num / den)
  }

  fn entropy(&self) -> Option<f64> {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    Some(
      crate::special::ln_beta(a, b)
        - (a - 1.0) * crate::special::digamma(a)
        - (b - 1.0) * crate::special::digamma(b)
        + (a + b - 2.0) * crate::special::digamma(a + b),
    )
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;
  use crate::tests::scalar_ks_best_p;
  use crate::traits::DistributionExt as _;
  use crate::traits::DistributionSampler;

  /// Small shapes must not lose a draw to `0 / 0`.
  ///
  /// The Gamma-ratio construction underflows both marginals to exactly zero
  /// once a shape is small — 13 % of the draws at `α = β = 0.01` in single
  /// precision, and it still reaches double precision at `α = β = 0.001`.
  /// Both fill paths are exercised: a 1000-long slice takes the SIMD
  /// chunks plus the scalar remainder, an 8-long one the short path.
  #[test]
  fn small_shapes_stay_finite() {
    for seed in [7u64, 11] {
      let mut d = SimdBeta::<f32>::new(0.01, 0.01).seeded(&Deterministic::new(seed));
      let mut wide = vec![0.0_f32; 1_000];
      d.fill_slice(&mut wide);
      let bad = wide.iter().filter(|x| !x.is_finite()).count();
      assert_eq!(bad, 0, "seed {seed}: {bad} non-finite of the wide fill");
      let mut narrow = [0.0_f32; 8];
      for _ in 0..200 {
        d.fill_slice(&mut narrow);
        let bad = narrow.iter().filter(|x| !x.is_finite()).count();
        assert_eq!(bad, 0, "seed {seed}: {bad} non-finite of the short fill");
      }
    }
  }

  /// The repaired small-shape draws still carry Beta's first two moments —
  /// a fix that merely clamped the NaN away would not. Single precision is
  /// deliberate: an eighth of these draws take the repaired branch.
  #[test]
  fn small_shape_moments_hold() {
    let (a, b) = (0.01_f64, 0.01_f64);
    let mut d = SimdBeta::<f32>::new(a as f32, b as f32).seeded(&Deterministic::new(11));
    let mut buf = vec![0.0_f32; 1 << 16];
    let (mut s, mut s2) = (0.0, 0.0);
    let reps = 8;
    for _ in 0..reps {
      d.fill_slice(&mut buf);
      for &x in buf.iter() {
        s += x as f64;
        s2 += (x as f64) * (x as f64);
      }
    }
    let n = (reps * buf.len()) as f64;
    let (mean, m2) = (s / n, s2 / n);
    let want_mean = a / (a + b);
    let want_m2 = a * (a + 1.0) / ((a + b) * (a + b + 1.0));
    // The law is all but Bernoulli(1/2) at this shape, so σ ≈ 1/2 and five
    // standard errors of the mean is 2.5/√n.
    let band = 2.5 / n.sqrt();
    assert!(
      (mean - want_mean).abs() < band,
      "mean = {mean}, expected {want_mean} ± {band}"
    );
    assert!(
      (m2 - want_m2).abs() < band,
      "second moment = {m2}, expected {want_m2} ± {band}"
    );
  }

  /// The honest `Distribution` draws from the caller's rng and agrees with the cdf.
  #[test]
  fn scalar_sample_matches_cdf() {
    let d = SimdBeta::<f64>::new(2.5, 4.0);
    let best = scalar_ks_best_p(&d, |x| d.cdf(x).unwrap());
    assert!(best > 0.01, "best p = {best}");
  }
}

py_distribution!(PyBeta, SimdBeta,
  sig: (alpha, beta, seed=None, dtype=None),
  params: (alpha: f64, beta: f64)
);

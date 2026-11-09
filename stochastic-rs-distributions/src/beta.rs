//! # Beta
//!
//! $$
//! f(x)=\frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)},\ x\in(0,1)
//! $$
//!
//! Sampling: `G1/(G1 + G2)` of two gammas, Devroye, L. (1986), *Non-Uniform Random Variate Generation*, Springer, §IX.4, DOI 10.1007/978-1-4613-8643-8.

use std::mem::MaybeUninit;

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;
use stochastic_rs_core::simd_rng::as_uninit_mut;

use super::SimdFloatExt;
use super::gamma::GammaState;
use super::gamma::SimdGamma;
use crate::seeded::Buffered;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

const SMALL_BETA_THRESHOLD: usize = 16;

/// `a/(a + b)` of two positive draws, from the smaller over the larger where their sum passes the
/// doubles.
#[inline]
pub(crate) fn ratio<T: SimdFloatExt>(a: T, b: T) -> T {
  let sum = a + b;
  if sum < T::infinity() {
    return a / sum;
  }
  if a <= b {
    let r = a / b;
    r / (T::one() + r)
  } else {
    T::one() / (T::one() + b / a)
  }
}

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
  pub(crate) g1: GammaState<T, R>,
  g2: GammaState<T, R>,
  buf: Buffered<T, 16>,
}

impl<T: SimdFloatExt> SimdBeta<T> {
  /// `Gamma(α, 1)/(Gamma(α, 1) + Gamma(β, 1))`; panics unless `α` and `β` are finite and positive.
  pub fn new(alpha: T, beta: T) -> Self {
    assert!(
      alpha.is_finite(),
      "alpha must satisfy `alpha.is_finite()`, got alpha = {alpha:?}"
    );
    assert!(
      beta.is_finite(),
      "beta must satisfy `beta.is_finite()`, got beta = {beta:?}"
    );
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

  /// The ratio from logs where both gamma draws underflow to `0/0` (`Beta(0.01, 0.01)`: 1 in 8 f32,
  /// 1 in 5e6 f64 draws); scale-free, so exact where the value path was.
  #[cold]
  #[inline(never)]
  fn ratio_from_logs<R: SimdRngExt>(
    &self,
    g1: &mut GammaState<T, R>,
    g2: &mut GammaState<T, R>,
  ) -> T {
    let (a, b) = (
      self.gamma1.next_log_parts(g1),
      self.gamma2.next_log_parts(g2),
    );
    self.ratio_of_parts(a, b)
  }

  /// `1/(1 + e^{lb − la})`; where both `ln u/shape` overflow, `lb − la` is NaN and
  /// `ln u_b·(α/β) < ln u_a`, signed without a subnormal tie, picks the draw's rounded 1 or 0.
  fn ratio_of_parts(&self, (ca, ua): (T, T), (cb, ub): (T, T)) -> T {
    let la = ca + ua / self.alpha;
    let lb = cb + ub / self.beta;
    let d = lb - la;
    if d.is_nan() {
      return if ub * (self.alpha / self.beta) < ua {
        T::one()
      } else {
        T::zero()
      };
    }
    T::one() / (T::one() + d.exp())
  }

  fn fill_parts<R: SimdRngExt>(
    &self,
    g1: &mut GammaState<T, R>,
    g2: &mut GammaState<T, R>,
    out: &mut [MaybeUninit<T>],
  ) {
    if out.len() < SMALL_BETA_THRESHOLD {
      for x in out.iter_mut() {
        let a = self.gamma1.next(g1);
        let b = self.gamma2.next(g2);
        x.write(if a + b > T::zero() {
          ratio(a, b)
        } else {
          self.ratio_from_logs(g1, g2)
        });
      }
      return;
    }
    let mut ga = [T::zero(); 64];
    let mut gb = [T::zero(); 64];
    let (chunks, rem) = out.as_chunks_mut::<64>();
    for chunk in chunks {
      // SAFETY: `SimdGamma::fill` writes every element of `ga`.
      self.gamma1.fill(g1, unsafe { as_uninit_mut(&mut ga) });
      // SAFETY: `SimdGamma::fill` writes every element of `gb`.
      self.gamma2.fill(g2, unsafe { as_uninit_mut(&mut gb) });
      for (sub, (a8, b8)) in chunk.as_chunks_mut::<8>().0.iter_mut().zip(
        ga.as_chunks::<8>()
          .0
          .iter()
          .zip(gb.as_chunks::<8>().0.iter()),
      ) {
        let a = T::simd_from_array(*a8);
        let b = T::simd_from_array(*b8);
        let mut r = T::simd_to_array(a / (a + b));
        for (j, x) in r.iter_mut().enumerate() {
          let sum = a8[j] + b8[j];
          if sum <= T::zero() {
            *x = self.ratio_from_logs(g1, g2);
          } else if sum == T::infinity() {
            *x = ratio(a8[j], b8[j]);
          }
        }
        *sub = r.map(MaybeUninit::new);
      }
    }
    if !rem.is_empty() {
      let n = rem.len();
      // SAFETY: `SimdGamma::fill` writes every element of `ga[..n]`.
      self.gamma1.fill(g1, unsafe { as_uninit_mut(&mut ga[..n]) });
      // SAFETY: `SimdGamma::fill` writes every element of `gb[..n]`.
      self.gamma2.fill(g2, unsafe { as_uninit_mut(&mut gb[..n]) });
      for i in 0..n {
        rem[i].write(if ga[i] + gb[i] > T::zero() {
          ratio(ga[i], gb[i])
        } else {
          self.ratio_from_logs(g1, g2)
        });
      }
    }
  }

  pub(crate) fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    let a = self.gamma1.draw_with(rng);
    let b = self.gamma2.draw_with(rng);
    if a + b > T::zero() {
      return ratio(a, b);
    }
    let (a, b) = (
      self.gamma1.draw_log_parts_with(rng),
      self.gamma2.draw_log_parts_with(rng),
    );
    self.ratio_of_parts(a, b)
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
  fn fill<R: SimdRngExt>(&self, state: &mut BetaState<T, R>, out: &mut [MaybeUninit<T>]) {
    self.fill_parts(&mut state.g1, &mut state.g2, out);
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut BetaState<T, R>) -> T {
    let BetaState { g1, g2, buf } = state;
    // SAFETY: `fill_parts` writes every element of the buffer.
    buf.pop(|b| self.fill_parts(g1, g2, unsafe { as_uninit_mut(b) }))
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdBeta<T> {
  /// `G1/(G1 + G2)` of two scalar gamma draws on the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.draw_with(rng)
  }
}

/// The density's rare paths, out of line so the plain one inlines: an end's limit (`+∞` below a
/// unit shape, `b` or `a` at it, 0 above) and the Stirling-scaled density from a shape of 100.
#[cold]
#[inline(never)]
fn pdf_edge(a: f64, b: f64, x: f64) -> f64 {
  if x == 0.0 || x == 1.0 {
    let (shape, other) = if x == 0.0 { (a, b) } else { (b, a) };
    return if shape < 1.0 {
      f64::INFINITY
    } else if shape == 1.0 {
      other
    } else {
      0.0
    };
  }
  crate::special::beta_density(a, b, x)
}

impl<T: SimdFloatExt> crate::traits::DistributionExt for SimdBeta<T> {
  /// `(a − 1) ln x + (b − 1) ln(1 − x) − ln B(a, b)`; from a shape of 100 the incomplete beta's
  /// Stirling prefactor over `x(1 − x)`, where those terms cancel; the ends their limits.
  fn pdf(&self, x: f64) -> Option<f64> {
    if x.is_nan() {
      return Some(f64::NAN);
    }
    if !(0.0..=1.0).contains(&x) {
      return Some(0.0);
    }
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    if x == 0.0 || x == 1.0 || a.max(b) >= crate::special::BETA_LARGE_SHAPE {
      return Some(pdf_edge(a, b, x));
    }
    let log_pdf = (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - crate::special::ln_beta(a, b);
    Some(log_pdf.exp())
  }

  fn cdf(&self, x: f64) -> Option<f64> {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    Some(crate::special::beta_i(a, b, x.clamp(0.0, 1.0)))
  }

  /// Both tails keep their digits; near `x = 1` correctly rounded but at one-spacing ties whose
  /// root lies within the residual's double-precision noise of a midpoint, at worst 0.553 ulp.
  fn quantile(&self, p: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&p) {
      return Some(f64::NAN);
    }
    if p == 0.0 || p == 1.0 {
      return Some(p);
    }
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    Some(crate::special::beta_i_inv(a, b, p))
  }

  fn mean(&self) -> Option<f64> {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    Some(a / (a + b))
  }

  fn median(&self) -> Option<f64> {
    self.quantile(0.5)
  }

  /// `NaN` where no mode is unique: the U-shaped law (α, β < 1) and the uniform one (α = β = 1).
  fn mode(&self) -> Option<f64> {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    Some(if a > 1.0 && b > 1.0 {
      (a - 1.0) / (a + b - 2.0)
    } else if (a < 1.0 && b < 1.0) || (a == 1.0 && b == 1.0) {
      f64::NAN
    } else if a < b {
      0.0
    } else {
      1.0
    })
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

  /// The closed form `ln B − (a − 1)ψ(a) − (b − 1)ψ(b) + (a + b − 2)ψ(a + b)`; from a shape of 100
  /// its cancellation-free Stirling form.
  fn entropy(&self) -> Option<f64> {
    let a = self.alpha.to_f64().unwrap();
    let b = self.beta.to_f64().unwrap();
    if a.max(b) >= crate::special::BETA_LARGE_SHAPE {
      return Some(crate::special::beta_entropy(a, b));
    }
    Some(
      crate::special::ln_beta(a, b)
        - (a - 1.0) * crate::special::digamma(a)
        - (b - 1.0) * crate::special::digamma(b)
        + (a + b - 2.0) * crate::special::digamma(a + b),
    )
  }
}

#[cfg(test)]
mod tests;

py_distribution!(PyBeta, SimdBeta,
  sig: (alpha, beta, seed=None, dtype=None),
  params: (alpha: f64, beta: f64)
);

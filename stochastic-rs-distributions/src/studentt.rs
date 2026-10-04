//! # Studentt
//!
//! $$
//! f(x)=\frac{\Gamma((\nu+1)/2)}{\sqrt{\nu\pi}\,\Gamma(\nu/2)}\left(1+\frac{x^2}{\nu}\right)^{-(\nu+1)/2}
//! $$
//!
//! Sampling: `Z/sqrt(V/ν)` with `V ~ χ²_ν`, Devroye, L. (1986), *Non-Uniform Random Variate Generation*, Springer, §IX.5, DOI 10.1007/978-1-4613-8643-8.

use rand::Rng;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRngExt;

use super::SimdFloatExt;
use super::chi_square::SimdChiSquared;
use super::gamma::GammaState;
use super::normal::SimdNormal;
use crate::seeded::Buffered;
use crate::seeded::StreamState;
use crate::traits::distribution::Sealed;
use crate::traits::distribution::SimdDistribution;
use crate::traits::distribution::SimdKernel;

const SMALL_STUDENT_T_THRESHOLD: usize = 16;

/// Student's t law with `nu` degrees of freedom: parameters only; a [`Seeded`](crate::Seeded) stream draws it in bulk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimdStudentT<T> {
  nu: T,
  chisq: SimdChiSquared<T>,
}

/// A Student-t stream: the normal and chi-squared sub-streams and the single-draw buffer.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct StudentTState<T: SimdFloatExt, R: SimdRngExt> {
  normal: StreamState<T, R, 64>,
  chisq: GammaState<T, R>,
  buf: Buffered<T, 16>,
}

impl<T: SimdFloatExt> SimdStudentT<T> {
  /// Creates a Student's t-distribution via `Z/sqrt(V/nu)`, `Z` standard
  /// normal, `V ~ ChiSquared(nu)`.
  ///
  /// - `nu` — degrees of freedom ν (matches the module header's ν).
  pub fn new(nu: T) -> Self {
    assert!(
      nu > T::zero(),
      "nu must satisfy `nu > T::zero()`, got nu = {nu:?}"
    );
    Self {
      nu,
      chisq: SimdChiSquared::new(nu),
    }
  }

  /// The degrees of freedom `ν`.
  pub fn nu(&self) -> T {
    self.nu
  }

  fn fill_parts<R: SimdRngExt>(
    &self,
    normal: &mut StreamState<T, R, 64>,
    chisq: &mut GammaState<T, R>,
    out: &mut [T],
  ) {
    if out.len() < SMALL_STUDENT_T_THRESHOLD {
      for x in out.iter_mut() {
        let z = SimdNormal::<T>::standard().next(normal);
        let v = self.chisq.next(chisq);
        *x = z / (v / self.nu).sqrt();
      }
      return;
    }
    let inv_nu = T::splat(T::one() / self.nu);
    let mut zbuf = [T::zero(); 64];
    let mut vbuf = [T::zero(); 64];
    let (chunks, rem) = out.as_chunks_mut::<64>();
    for chunk in chunks {
      SimdNormal::<T>::fill_standard(&mut normal.rng, &mut zbuf);
      self.chisq.fill(chisq, &mut vbuf);
      for (sub, (z8, v8)) in chunk.as_chunks_mut::<8>().0.iter_mut().zip(
        zbuf
          .as_chunks::<8>()
          .0
          .iter()
          .zip(vbuf.as_chunks::<8>().0.iter()),
      ) {
        let x = T::simd_from_array(*z8) / T::simd_sqrt(T::simd_from_array(*v8) * inv_nu);
        *sub = T::simd_to_array(x);
      }
    }
    if !rem.is_empty() {
      let n = rem.len();
      SimdNormal::<T>::fill_standard(&mut normal.rng, &mut zbuf[..n]);
      self.chisq.fill(chisq, &mut vbuf[..n]);
      for i in 0..n {
        rem[i] = zbuf[i] / (vbuf[i] / self.nu).sqrt();
      }
    }
  }

  pub(crate) fn draw_with<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    let z = SimdNormal::<T>::standard().draw_with(rng);
    let v = self.chisq.draw_with(rng);
    z / (v / self.nu).sqrt()
  }
}

/// ν=5 — matches this crate's own `tests/distribution_ext_vs_reference.rs`
/// fixture, also used by the umbrella crate's workspace-root
/// `benches/distributions.rs` and `benches/dist_multicore.rs`.
impl<T: SimdFloatExt> Default for SimdStudentT<T> {
  fn default() -> Self {
    Self::new(T::from(5.0).unwrap())
  }
}

impl<T: SimdFloatExt> Sealed for SimdStudentT<T> {}

impl<T: SimdFloatExt> SimdDistribution for SimdStudentT<T> {
  type State<R: SimdRngExt> = StudentTState<T, R>;

  fn init<R: SimdRngExt, S: SeedExt>(&self, seed: &S) -> (StudentTState<T, R>, u64) {
    let (normal, _) = SimdNormal::<T>::standard().init::<R, S>(seed);
    let (chisq, _) = self.chisq.init::<R, S>(seed);
    // No engine reads this fourth draw: it is the basis, and it keeps the seed budget later streams depend on.
    let basis = seed.next_seed();
    (
      StudentTState {
        normal,
        chisq,
        buf: Buffered::new(),
      },
      basis,
    )
  }
}

impl<T: SimdFloatExt> SimdKernel for SimdStudentT<T> {
  type Item = T;

  #[inline]
  fn fill<R: SimdRngExt>(&self, state: &mut StudentTState<T, R>, out: &mut [T]) {
    self.fill_parts(&mut state.normal, &mut state.chisq, out);
  }

  #[inline]
  fn next<R: SimdRngExt>(&self, state: &mut StudentTState<T, R>) -> T {
    let StudentTState { normal, chisq, buf } = state;
    buf.pop(|b| self.fill_parts(normal, chisq, b))
  }
}

impl<T: SimdFloatExt> Distribution<T> for SimdStudentT<T> {
  /// `Z/sqrt(V/ν)` from one scalar normal and one scalar chi-squared draw on the caller's rng.
  fn sample<G: Rng + ?Sized>(&self, rng: &mut G) -> T {
    self.draw_with(rng)
  }
}

impl<T: SimdFloatExt> crate::traits::DistributionExt for SimdStudentT<T> {
  fn pdf(&self, x: f64) -> Option<f64> {
    let nu = self.nu.to_f64().unwrap();
    // f(x) = Γ((ν+1)/2) / (√(νπ) Γ(ν/2)) · (1 + x²/ν)^(−(ν+1)/2)
    let log_norm = crate::special::ln_gamma(0.5 * (nu + 1.0))
      - 0.5 * (nu * std::f64::consts::PI).ln()
      - crate::special::ln_gamma(0.5 * nu);
    let log_kernel = -0.5 * (nu + 1.0) * (1.0 + x * x / nu).ln();
    Some((log_norm + log_kernel).exp())
  }

  fn cdf(&self, x: f64) -> Option<f64> {
    // For x ≥ 0:  F(x) = 1 − ½ I_{ν/(ν+x²)}(ν/2, ½)
    // By symmetry F(−x) = 1 − F(x).
    let nu = self.nu.to_f64().unwrap();
    let t = nu / (nu + x * x);
    let half = 0.5 * crate::special::beta_i(0.5 * nu, 0.5, t);
    if x >= 0.0 {
      Some(1.0 - half)
    } else {
      Some(half)
    }
  }

  fn quantile(&self, p: f64) -> Option<f64> {
    if p <= 0.0 {
      return Some(f64::NEG_INFINITY);
    }
    if p >= 1.0 {
      return Some(f64::INFINITY);
    }
    let nu = self.nu.to_f64().unwrap();
    // Use the Cornish-Fisher-style normal seed and refine with Newton's method.
    let z = crate::special::ndtri(p);
    let mut x = z * (1.0 + (z * z + 1.0) / (4.0 * nu));
    for _ in 0..40 {
      let cdf = {
        let t = nu / (nu + x * x);
        let half = 0.5 * crate::special::beta_i(0.5 * nu, 0.5, t);
        if x >= 0.0 { 1.0 - half } else { half }
      };
      let f = cdf - p;
      let log_norm = crate::special::ln_gamma(0.5 * (nu + 1.0))
        - 0.5 * (nu * std::f64::consts::PI).ln()
        - crate::special::ln_gamma(0.5 * nu);
      let log_kernel = -0.5 * (nu + 1.0) * (1.0 + x * x / nu).ln();
      let pdf = (log_norm + log_kernel).exp();
      if pdf <= 0.0 {
        break;
      }
      let dx = f / pdf;
      let new_x = x - dx;
      if (new_x - x).abs() < 1e-14 * (1.0 + x.abs()) {
        return Some(new_x);
      }
      x = new_x;
    }
    Some(x)
  }

  /// `NaN` at `nu <= 1` (`nu = 1` is the Cauchy distribution — see
  /// [`crate::cauchy`]'s `SimdCauchy::mean`, same underlying non-convergent
  /// integral): the mean does not exist there, so it is `NaN` rather than
  /// `0.0` even though `0.0` is what every finite-mean case below returns.
  fn mean(&self) -> Option<f64> {
    if self.nu.to_f64().unwrap() > 1.0 {
      Some(0.0)
    } else {
      Some(f64::NAN)
    }
  }

  fn median(&self) -> Option<f64> {
    Some(0.0)
  }

  fn mode(&self) -> Option<f64> {
    Some(0.0)
  }

  /// Three-way split on `nu`, all mathematically forced by the tail
  /// behaviour of the Student-t density: `nu > 2` gives the standard finite
  /// `nu/(nu-2)`; `1 < nu <= 2` (`nu = 2` is the common finance choice for
  /// "just barely infinite variance") diverges to `+∞`, a definite value;
  /// `nu <= 1` has no mean to build a variance from at all, so `NaN`.
  fn variance(&self) -> Option<f64> {
    let nu = self.nu.to_f64().unwrap();
    if nu > 2.0 {
      Some(nu / (nu - 2.0))
    } else if nu > 1.0 {
      Some(f64::INFINITY)
    } else {
      Some(f64::NAN)
    }
  }

  /// `NaN` at `nu <= 3`: `nu = 3` is itself a common fat-tail choice in
  /// finance, and it already sits at this threshold — the third central
  /// moment does not exist there, so skewness is `NaN`, not `0.0`.
  fn skewness(&self) -> Option<f64> {
    if self.nu.to_f64().unwrap() > 3.0 {
      Some(0.0)
    } else {
      Some(f64::NAN)
    }
  }

  /// Three-way split mirroring `variance` one moment order up: finite for
  /// `nu > 4`, `+∞` for `2 < nu <= 4` (a definite divergence), `NaN` for
  /// `nu <= 2` (no variance to build on).
  fn kurtosis(&self) -> Option<f64> {
    let nu = self.nu.to_f64().unwrap();
    if nu > 4.0 {
      Some(6.0 / (nu - 4.0))
    } else if nu > 2.0 {
      Some(f64::INFINITY)
    } else {
      Some(f64::NAN)
    }
  }

  fn entropy(&self) -> Option<f64> {
    let nu = self.nu.to_f64().unwrap();
    let half_nu = 0.5 * nu;
    let half_nu_p1 = 0.5 * (nu + 1.0);
    Some(
      half_nu_p1 * (crate::special::digamma(half_nu_p1) - crate::special::digamma(half_nu))
        + 0.5 * nu.ln()
        + crate::special::ln_gamma(half_nu)
        - crate::special::ln_gamma(half_nu_p1)
        + 0.5 * std::f64::consts::PI.ln(),
    )
  }

  /// 1 at `t = 0`, else `NaN`: the `|x|^{-nu-1}` tail makes `E[e^{tX}]` diverge at every `t != 0`,
  /// however large `nu` is.
  fn moment_generating_function(&self, t: f64) -> Option<f64> {
    Some(if t == 0.0 { 1.0 } else { f64::NAN })
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::SimdRng;

  use super::*;
  use crate::tests::scalar_ks_best_p;
  use crate::traits::DistributionExt;

  /// Backs the doc comments on `mean`/`variance`/`skewness`/`kurtosis`:
  /// `nu = 3` is a common fat-tail choice in finance. It clears the
  /// mean/variance thresholds (`nu > 1`, `nu > 2`) but sits at exactly the
  /// skewness threshold (`nu > 3` required, so `nu = 3` itself is NaN) and
  /// inside kurtosis's divergent-but-defined middle band (`2 < nu <= 4`).
  #[test]
  fn studentt_low_nu_moments_match_documented_thresholds() {
    let t = SimdStudentT::<f64>::new(3.0);
    assert_eq!(t.mean().unwrap(), 0.0, "nu=3 > 1, mean should be 0");
    assert!(
      t.variance().unwrap().is_finite(),
      "nu=3 > 2, variance should be finite"
    );
    assert!(
      t.skewness().unwrap().is_nan(),
      "nu=3 is not > 3, skewness must be NaN"
    );
    assert_eq!(
      t.kurtosis().unwrap(),
      f64::INFINITY,
      "nu=3 is in (2,4], kurtosis diverges to +inf"
    );
    assert!(
      t.moment_generating_function(0.5).unwrap().is_nan(),
      "MGF must be NaN for every nu"
    );
  }

  /// `nu = 2` (the other boundary of variance's three-way split) clears
  /// the mean threshold, sits in variance's divergent-but-defined middle
  /// band (`1 < nu <= 2`), and sits at exactly kurtosis's `NaN` threshold
  /// (`nu > 2` required there, so `nu = 2` itself has no variance to build
  /// a kurtosis from at all).
  #[test]
  fn studentt_nu_two_variance_diverges_kurtosis_is_nan() {
    let t = SimdStudentT::<f64>::new(2.0);
    assert_eq!(t.mean().unwrap(), 0.0, "nu=2 > 1, mean should be 0");
    assert_eq!(
      t.variance().unwrap(),
      f64::INFINITY,
      "nu=2 is in (1,2], variance diverges to +inf"
    );
    assert!(
      t.kurtosis().unwrap().is_nan(),
      "nu=2 is not > 2, kurtosis must be NaN"
    );
  }

  /// At `nu = 1` (Cauchy) neither mean nor variance exists.
  #[test]
  fn studentt_nu_one_is_cauchy_like() {
    let t = SimdStudentT::<f64>::new(1.0);
    assert!(t.mean().unwrap().is_nan(), "nu=1 has no mean");
    assert!(t.variance().unwrap().is_nan(), "nu=1 has no variance");
  }

  /// Above every threshold, all four moments must be finite real numbers.
  #[test]
  fn studentt_high_nu_moments_are_all_finite() {
    let t = SimdStudentT::<f64>::new(10.0);
    assert!(t.mean().unwrap().is_finite());
    assert!(t.variance().unwrap().is_finite());
    assert!(t.skewness().unwrap().is_finite());
    assert!(t.kurtosis().unwrap().is_finite());
  }

  /// The honest `Distribution` draws from the caller's rng and agrees with the cdf.
  #[test]
  fn scalar_sample_matches_cdf() {
    let d = SimdStudentT::<f64>::new(6.0);
    let best = scalar_ks_best_p(&d, |x| d.cdf(x).unwrap());
    assert!(best > 0.01, "best p = {best}");
  }

  /// Every 5-wide `fill_with` builds a fresh stream from the caller's rng and takes the small-slice path.
  #[test]
  fn fill_with_is_deterministic_in_the_caller_rng() {
    let d = SimdStudentT::<f64>::new(6.0);
    let fill = |seed: u64| {
      let mut rng = SimdRng::from_seed(seed);
      let mut out = [0.0f64; 5];
      (0..4_000)
        .flat_map(|_| {
          d.fill_with(&mut rng, &mut out);
          out.map(f64::to_bits)
        })
        .collect::<Vec<_>>()
    };
    assert_eq!(fill(5), fill(5));
    assert_ne!(fill(5), fill(6));
  }
}

py_distribution!(PyStudentT, SimdStudentT,
  sig: (nu, seed=None, dtype=None),
  params: (nu: f64)
);

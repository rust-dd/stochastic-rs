//! # Studentt
//!
//! $$
//! f(x)=\frac{\Gamma((\nu+1)/2)}{\sqrt{\nu\pi}\,\Gamma(\nu/2)}\left(1+\frac{x^2}{\nu}\right)^{-(\nu+1)/2}
//! $$
//!
use std::cell::Cell;
use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::Unseeded;

use super::SimdFloatExt;
use super::chi_square::SimdChiSquared;
use super::normal::SimdNormal;
use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;

const SMALL_STUDENT_T_THRESHOLD: usize = 16;

pub struct SimdStudentT<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  nu: T,
  normal: SimdNormal<T, 64, R>,
  chisq: SimdChiSquared<T, R>,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  simd_rng: UnsafeCell<R>,
  stream_seed: Cell<u64>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdStudentT<T, R> {
  /// Creates a Student's t-distribution via `Z/sqrt(V/nu)`, `Z` standard
  /// normal, `V ~ ChiSquared(nu)`.
  ///
  /// - `nu` — degrees of freedom ν (matches the module header's ν).
  ///
  /// RNGs come from a [`SeedExt`](crate::simd_rng::SeedExt) source; each
  /// sub-component (normal, chisq, main rng) gets an independent stream.
  pub fn new<S: crate::simd_rng::SeedExt>(nu: T, seed: &S) -> Self {
    let normal = SimdNormal::<T, 64, R>::new(T::zero(), T::one(), seed);
    let chisq = SimdChiSquared::new(nu, seed);
    let stream_seed = seed.seed_value();
    Self {
      nu,
      normal,
      chisq,
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
    Self::new(self.nu, &crate::simd_rng::Deterministic::new(child_seed))
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

  /// Fills `out` using the internal SIMD RNG stream — the only stream this
  /// sampler draws from (see the crate-level RNG policy).
  pub fn fill_slice(&self, out: &mut [T]) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    if out.len() < SMALL_STUDENT_T_THRESHOLD {
      for x in out.iter_mut() {
        let z = self.normal.sample(rng);
        let v = self.chisq.sample(rng);
        *x = z / (v / self.nu).sqrt();
      }
      return;
    }
    let inv_nu = T::splat(T::one() / self.nu);
    let mut zbuf = [T::zero(); 64];
    let mut vbuf = [T::zero(); 64];
    let mut chunks = out.chunks_exact_mut(64);
    for chunk in &mut chunks {
      self.normal.fill_standard_fast(&mut zbuf);
      self.chisq.fill_slice(&mut vbuf);
      for (sub, (z8, v8)) in chunk
        .chunks_exact_mut(8)
        .zip(zbuf.chunks_exact(8).zip(vbuf.chunks_exact(8)))
      {
        let mut za = [T::zero(); 8];
        let mut va = [T::zero(); 8];
        za.copy_from_slice(z8);
        va.copy_from_slice(v8);
        let x = T::simd_from_array(za) / T::simd_sqrt(T::simd_from_array(va) * inv_nu);
        sub.copy_from_slice(&T::simd_to_array(x));
      }
    }
    let rem = chunks.into_remainder();
    if !rem.is_empty() {
      let n = rem.len();
      self.normal.fill_standard_fast(&mut zbuf[..n]);
      self.chisq.fill_slice(&mut vbuf[..n]);
      for i in 0..n {
        rem[i] = zbuf[i] / (vbuf[i] / self.nu).sqrt();
      }
    }
  }

  fn refill_buffer(&self) {
    let buf = unsafe { &mut *self.buffer.get() };
    self.fill_slice(buf);
    unsafe {
      *self.index.get() = 0;
    }
  }
}

/// ν=5 — matches this crate's own `tests/distribution_ext_vs_reference.rs`
/// fixture, also used by the umbrella crate's workspace-root
/// `benches/distributions.rs` and `benches/dist_multicore.rs`.
impl<T: SimdFloatExt, R: SimdRngExt> Default for SimdStudentT<T, R> {
  fn default() -> Self {
    Self::new(T::from(5.0).unwrap(), &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdStudentT<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.nu, &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Distribution<T> for SimdStudentT<T, R> {
  /// The `rng` argument is intentionally unused — this type draws from its
  /// own internal SIMD stream seeded at construction. Use `Deterministic`
  /// in the constructor for reproducibility.
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

impl<T: SimdFloatExt, R: SimdRngExt> crate::traits::DistributionExt for SimdStudentT<T, R> {
  fn pdf(&self, x: f64) -> f64 {
    let nu = self.nu.to_f64().unwrap();
    // f(x) = Γ((ν+1)/2) / (√(νπ) Γ(ν/2)) · (1 + x²/ν)^(−(ν+1)/2)
    let log_norm = crate::special::ln_gamma(0.5 * (nu + 1.0))
      - 0.5 * (nu * std::f64::consts::PI).ln()
      - crate::special::ln_gamma(0.5 * nu);
    let log_kernel = -0.5 * (nu + 1.0) * (1.0 + x * x / nu).ln();
    (log_norm + log_kernel).exp()
  }

  fn cdf(&self, x: f64) -> f64 {
    // For x ≥ 0:  F(x) = 1 − ½ I_{ν/(ν+x²)}(ν/2, ½)
    // By symmetry F(−x) = 1 − F(x).
    let nu = self.nu.to_f64().unwrap();
    let t = nu / (nu + x * x);
    let half = 0.5 * crate::special::beta_i(0.5 * nu, 0.5, t);
    if x >= 0.0 { 1.0 - half } else { half }
  }

  fn inv_cdf(&self, p: f64) -> f64 {
    if p <= 0.0 {
      return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
      return f64::INFINITY;
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
        return new_x;
      }
      x = new_x;
    }
    x
  }

  /// `NaN` at `nu <= 1` (`nu = 1` is the Cauchy distribution — see
  /// [`crate::cauchy`]'s `SimdCauchy::mean`, same underlying non-convergent
  /// integral): the mean does not exist there, so it is `NaN` rather than
  /// `0.0` even though `0.0` is what every finite-mean case below returns.
  fn mean(&self) -> f64 {
    if self.nu.to_f64().unwrap() > 1.0 {
      0.0
    } else {
      f64::NAN
    }
  }

  fn median(&self) -> f64 {
    0.0
  }

  fn mode(&self) -> f64 {
    0.0
  }

  /// Three-way split on `nu`, all mathematically forced by the tail
  /// behaviour of the Student-t density: `nu > 2` gives the standard finite
  /// `nu/(nu-2)`; `1 < nu <= 2` (`nu = 2` is the common finance choice for
  /// "just barely infinite variance") diverges to `+∞`, a definite value;
  /// `nu <= 1` has no mean to build a variance from at all, so `NaN`.
  fn variance(&self) -> f64 {
    let nu = self.nu.to_f64().unwrap();
    if nu > 2.0 {
      nu / (nu - 2.0)
    } else if nu > 1.0 {
      f64::INFINITY
    } else {
      f64::NAN
    }
  }

  /// `NaN` at `nu <= 3`: `nu = 3` is itself a common fat-tail choice in
  /// finance, and it already sits at this threshold — the third central
  /// moment does not exist there, so skewness is `NaN`, not `0.0`.
  fn skewness(&self) -> f64 {
    if self.nu.to_f64().unwrap() > 3.0 {
      0.0
    } else {
      f64::NAN
    }
  }

  /// Three-way split mirroring `variance` one moment order up: finite for
  /// `nu > 4`, `+∞` for `2 < nu <= 4` (a definite divergence), `NaN` for
  /// `nu <= 2` (no variance to build on).
  fn kurtosis(&self) -> f64 {
    let nu = self.nu.to_f64().unwrap();
    if nu > 4.0 {
      6.0 / (nu - 4.0)
    } else if nu > 2.0 {
      f64::INFINITY
    } else {
      f64::NAN
    }
  }

  fn entropy(&self) -> f64 {
    let nu = self.nu.to_f64().unwrap();
    let half_nu = 0.5 * nu;
    let half_nu_p1 = 0.5 * (nu + 1.0);
    half_nu_p1 * (crate::special::digamma(half_nu_p1) - crate::special::digamma(half_nu))
      + 0.5 * nu.ln()
      + crate::special::ln_gamma(half_nu)
      - crate::special::ln_gamma(half_nu_p1)
      + 0.5 * std::f64::consts::PI.ln()
  }

  /// `NaN` for every `nu`: the Student-t tail decays polynomially
  /// (`~|x|^{-nu-1}`), too slowly for `e^{tx}` to be integrable at any
  /// `t != 0`, regardless of how large `nu` is.
  fn moment_generating_function(&self, _t: f64) -> f64 {
    f64::NAN
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::traits::DistributionExt;

  /// Backs the doc comments on `mean`/`variance`/`skewness`/`kurtosis`:
  /// `nu = 3` is a common fat-tail choice in finance. It clears the
  /// mean/variance thresholds (`nu > 1`, `nu > 2`) but sits at exactly the
  /// skewness threshold (`nu > 3` required, so `nu = 3` itself is NaN) and
  /// inside kurtosis's divergent-but-defined middle band (`2 < nu <= 4`).
  #[test]
  fn studentt_low_nu_moments_match_documented_thresholds() {
    let t = SimdStudentT::<f64>::new(3.0, &Unseeded);
    assert_eq!(t.mean(), 0.0, "nu=3 > 1, mean should be 0");
    assert!(
      t.variance().is_finite(),
      "nu=3 > 2, variance should be finite"
    );
    assert!(
      t.skewness().is_nan(),
      "nu=3 is not > 3, skewness must be NaN"
    );
    assert_eq!(
      t.kurtosis(),
      f64::INFINITY,
      "nu=3 is in (2,4], kurtosis diverges to +inf"
    );
    assert!(
      t.moment_generating_function(0.5).is_nan(),
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
    let t = SimdStudentT::<f64>::new(2.0, &Unseeded);
    assert_eq!(t.mean(), 0.0, "nu=2 > 1, mean should be 0");
    assert_eq!(
      t.variance(),
      f64::INFINITY,
      "nu=2 is in (1,2], variance diverges to +inf"
    );
    assert!(
      t.kurtosis().is_nan(),
      "nu=2 is not > 2, kurtosis must be NaN"
    );
  }

  /// At `nu = 1` (Cauchy) neither mean nor variance exists.
  #[test]
  fn studentt_nu_one_is_cauchy_like() {
    let t = SimdStudentT::<f64>::new(1.0, &Unseeded);
    assert!(t.mean().is_nan(), "nu=1 has no mean");
    assert!(t.variance().is_nan(), "nu=1 has no variance");
  }

  /// Above every threshold, all four moments must be finite real numbers.
  #[test]
  fn studentt_high_nu_moments_are_all_finite() {
    let t = SimdStudentT::<f64>::new(10.0, &Unseeded);
    assert!(t.mean().is_finite());
    assert!(t.variance().is_finite());
    assert!(t.skewness().is_finite());
    assert!(t.kurtosis().is_finite());
  }
}

py_distribution!(PyStudentT, SimdStudentT,
  sig: (nu, seed=None, dtype=None),
  params: (nu: f64)
);

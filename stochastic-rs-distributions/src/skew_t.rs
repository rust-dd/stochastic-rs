//! # Skewed Student-t (Hansen 1994)
//!
//! Hansen's standardised skewed t — zero mean and unit variance by
//! construction, degrees of freedom $\eta > 2$ and skew $\lambda \in (-1, 1)$:
//!
//! $$
//! g(z) = b\,c\left(1 + \frac{1}{\eta - 2}\Bigl(\frac{bz + a}{1 \mp \lambda}\Bigr)^2\right)^{-(\eta+1)/2},
//! \qquad \text{“$-$” for } z < -a/b,\ \text{“$+$” for } z \ge -a/b,
//! $$
//!
//! $$
//! a = 4\lambda c\,\frac{\eta - 2}{\eta - 1},\qquad b^2 = 1 + 3\lambda^2 - a^2,\qquad
//! c = \frac{\Gamma\!\bigl(\tfrac{\eta+1}{2}\bigr)}{\sqrt{\pi(\eta - 2)}\,\Gamma\!\bigl(\tfrac\eta2\bigr)}.
//! $$
//!
//! Each side is a rescaled unit-variance Student-t, so the CDF and
//! quantile are piecewise Student-t CDFs and quantiles, and a draw is
//! $z = (sW - a)/b$ with $W = |T|\sqrt{(\eta-2)/\eta}$, $T \sim t_\eta$,
//! and $s = 1 + \lambda$ with probability $(1 + \lambda)/2$, $s = -(1 -
//! \lambda)$ otherwise. Skewness needs $\eta > 3$, kurtosis $\eta > 4$.
//! This is the density used by the GARCH literature and the `arch`
//! package, not Azzalini's skew-t.
//!
//! Reference: Hansen, B.E. (1994), "Autoregressive Conditional Density
//! Estimation", *International Economic Review* 35(3), 705-730.
//! DOI: 10.2307/2527081

use std::cell::Cell;
use std::cell::UnsafeCell;

use rand::Rng;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::Unseeded;

use super::SimdFloatExt;
use super::studentt::SimdStudentT;
use crate::simd_rng::SimdRng;
use crate::simd_rng::SimdRngExt;
use crate::special::ln_gamma;
use crate::traits::DistributionExt;

const SMALL_SKEW_T_THRESHOLD: usize = 16;

/// Hansen's standardised skewed Student-t with `η > 2` degrees of freedom
/// and skew `λ ∈ (−1, 1)`.
pub struct SimdSkewT<T: SimdFloatExt, R: SimdRngExt = SimdRng> {
  eta: T,
  lambda: T,
  a: f64,
  b: f64,
  c: f64,
  student: SimdStudentT<T, R>,
  buffer: UnsafeCell<[T; 16]>,
  index: UnsafeCell<usize>,
  simd_rng: UnsafeCell<R>,
  stream_seed: Cell<u64>,
}

impl<T: SimdFloatExt, R: SimdRngExt> SimdSkewT<T, R> {
  /// Construct Hansen's skew-t$(\eta, \lambda)$.
  pub fn new<S: crate::simd_rng::SeedExt>(eta: T, lambda: T, seed: &S) -> Self {
    let eta_f = eta.to_f64().unwrap();
    let lambda_f = lambda.to_f64().unwrap();
    assert!(eta_f > 2.0, "SkewT: eta must exceed 2");
    assert!(lambda_f.abs() < 1.0, "SkewT: lambda must lie in (-1, 1)");
    let c = (ln_gamma(0.5 * (eta_f + 1.0)) - ln_gamma(0.5 * eta_f)).exp()
      / (std::f64::consts::PI * (eta_f - 2.0)).sqrt();
    let a = 4.0 * lambda_f * c * (eta_f - 2.0) / (eta_f - 1.0);
    let b = (1.0 + 3.0 * lambda_f * lambda_f - a * a).sqrt();
    let student = SimdStudentT::<T, R>::new(eta, seed);
    let stream_seed = seed.seed_value();
    Self {
      eta,
      lambda,
      a,
      b,
      c,
      student,
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
      self.eta,
      self.lambda,
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

  /// Fills `out` from the two-piece representation: a unit-variance
  /// Student-t magnitude gets the right-hand scale `1 + λ` with
  /// probability `(1 + λ)/2` and the left-hand scale `−(1 − λ)` otherwise.
  pub fn fill_slice(&self, out: &mut [T]) {
    let rng = unsafe { &mut *self.simd_rng.get() };
    let eta = self.eta.to_f64().unwrap();
    let lambda = self.lambda.to_f64().unwrap();
    let scale = ((eta - 2.0) / eta).sqrt();
    let right = T::from_f64_fast((1.0 + lambda) * scale / self.b);
    let left = T::from_f64_fast(-(1.0 - lambda) * scale / self.b);
    let shift = T::from_f64_fast(-self.a / self.b);
    let p_right = T::from_f64_fast(0.5 * (1.0 + lambda));
    if out.len() < SMALL_SKEW_T_THRESHOLD {
      for x in out.iter_mut() {
        let w = self.student.sample_fast().abs();
        let u = T::sample_uniform_simd(rng);
        *x = if u < p_right { right * w } else { left * w } + shift;
      }
      return;
    }
    let mut tbuf = [T::zero(); 64];
    let mut ubuf = [T::zero(); 64];
    let (chunks, rem) = out.as_chunks_mut::<64>();
    for chunk in chunks {
      self.student.fill_slice(&mut tbuf);
      T::fill_uniform_simd(rng, &mut ubuf);
      for i in 0..64 {
        let w = tbuf[i].abs();
        chunk[i] = if ubuf[i] < p_right {
          right * w
        } else {
          left * w
        } + shift;
      }
    }
    if !rem.is_empty() {
      let n = rem.len();
      self.student.fill_slice(&mut tbuf[..n]);
      T::fill_uniform_simd(rng, &mut ubuf[..n]);
      for i in 0..n {
        let w = tbuf[i].abs();
        rem[i] = if ubuf[i] < p_right {
          right * w
        } else {
          left * w
        } + shift;
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

  /// The mode $-a/b$, where the two pieces meet.
  pub fn knot(&self) -> f64 {
    -self.a / self.b
  }

  /// Side scale `1 ∓ λ` for the piece containing `z`.
  fn side(&self, z: f64) -> f64 {
    let lambda = self.lambda.to_f64().unwrap();
    if z < self.knot() {
      1.0 - lambda
    } else {
      1.0 + lambda
    }
  }

  /// $\sqrt{\eta/(\eta - 2)}$: unit-variance to raw Student-t scale.
  fn t_scale(&self) -> f64 {
    let eta = self.eta.to_f64().unwrap();
    (eta / (eta - 2.0)).sqrt()
  }

  /// $\mathbb E|W|^3$ of the unit-variance Student-t magnitude, $\eta > 3$.
  fn abs_third_moment(&self) -> f64 {
    let eta = self.eta.to_f64().unwrap();
    (eta - 2.0).powf(1.5) * (ln_gamma(0.5 * (eta - 3.0)) - ln_gamma(0.5 * eta)).exp()
      / std::f64::consts::PI.sqrt()
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Clone for SimdSkewT<T, R> {
  fn clone(&self) -> Self {
    Self::new(self.eta, self.lambda, &Unseeded)
  }
}

impl<T: SimdFloatExt, R: SimdRngExt> Distribution<T> for SimdSkewT<T, R> {
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

impl<T: SimdFloatExt, R: SimdRngExt> DistributionExt for SimdSkewT<T, R> {
  fn pdf(&self, z: f64) -> f64 {
    let eta = self.eta.to_f64().unwrap();
    let w = (self.b * z + self.a) / self.side(z);
    self.b * self.c * (1.0 + w * w / (eta - 2.0)).powf(-0.5 * (eta + 1.0))
  }

  fn cdf(&self, z: f64) -> f64 {
    let lambda = self.lambda.to_f64().unwrap();
    let w = (self.b * z + self.a) / self.side(z) * self.t_scale();
    let t = self.student.cdf(w);
    if z < self.knot() {
      (1.0 - lambda) * t
    } else {
      0.5 * (1.0 - lambda) + (1.0 + lambda) * (t - 0.5)
    }
  }

  fn inv_cdf(&self, p: f64) -> f64 {
    let lambda = self.lambda.to_f64().unwrap();
    let split = 0.5 * (1.0 - lambda);
    let (side, q) = if p < split {
      (1.0 - lambda, p / (1.0 - lambda))
    } else {
      (1.0 + lambda, (p - split) / (1.0 + lambda) + 0.5)
    };
    (side * self.student.inv_cdf(q) / self.t_scale() - self.a) / self.b
  }

  fn mean(&self) -> f64 {
    0.0
  }

  fn median(&self) -> f64 {
    self.inv_cdf(0.5)
  }

  fn mode(&self) -> f64 {
    self.knot()
  }

  fn variance(&self) -> f64 {
    1.0
  }

  /// $\mathbb E z^3$ from the two-piece moments; `NaN` for $\eta \le 3$.
  fn skewness(&self) -> f64 {
    let eta = self.eta.to_f64().unwrap();
    if eta <= 3.0 {
      return f64::NAN;
    }
    let l = self.lambda.to_f64().unwrap();
    let (a, b) = (self.a, self.b);
    let m3w = self.abs_third_moment();
    (4.0 * l * (1.0 + l * l) * m3w - 3.0 * a * (1.0 + 3.0 * l * l) + 2.0 * a.powi(3)) / b.powi(3)
  }

  /// Excess kurtosis $\mathbb E z^4 - 3$; `NaN` for $\eta \le 4$.
  fn kurtosis(&self) -> f64 {
    let eta = self.eta.to_f64().unwrap();
    if eta <= 4.0 {
      return f64::NAN;
    }
    let l = self.lambda.to_f64().unwrap();
    let (a, b) = (self.a, self.b);
    let m3w = self.abs_third_moment();
    let m4w = 3.0 * (eta - 2.0) / (eta - 4.0);
    ((1.0 + 10.0 * l * l + 5.0 * l.powi(4)) * m4w - 16.0 * a * l * (1.0 + l * l) * m3w
      + 6.0 * a * a * (1.0 + 3.0 * l * l)
      - 3.0 * a.powi(4))
      / b.powi(4)
      - 3.0
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * b.abs().max(1e-300)
  }

  /// `arch.univariate.distribution.SkewStudent` (Hansen's density): the
  /// per-observation likelihood, `cdf`, `ppf`, and the third and fourth
  /// moments of the density integrated numerically with `scipy.integrate`.
  #[test]
  fn matches_arch_skew_student() {
    let cases: [((f64, f64), [(f64, f64, f64); 6], [f64; 3], [f64; 2]); 2] = [
      (
        (5.0, -0.3),
        [
          (-3.0, 0.011_968_363_228_410_123, 0.010_908_787_904_991_42),
          (-1.0, 0.173_461_332_479_957_76, 0.131_343_308_198_292_43),
          (-0.2, 0.400_166_177_565_298_43, 0.356_174_523_423_168_74),
          (0.0, 0.453_941_038_826_430_6, 0.441_776_736_834_836_87),
          (0.5, 0.502_052_313_672_497_9, 0.687_806_461_738_387_8),
          (2.0, 0.022_804_512_034_078_815, 0.989_606_509_260_385),
        ],
        [
          -1.732_379_684_017_731_5,
          0.124_519_972_477_625_78,
          1.050_050_376_587_227,
        ],
        [-1.233_482_286_442_846_3, 11.883_107_935_250_047],
      ),
      (
        (8.0, 0.4),
        [
          (
            -3.0,
            0.000_927_488_132_394_006_1,
            0.000_362_441_895_588_431_5,
          ),
          (-1.0, 0.319_618_350_480_818_44, 0.126_876_513_847_731_7),
          (-0.2, 0.441_400_364_923_063_6, 0.475_940_259_652_003_26),
          (0.0, 0.407_546_825_530_346_93, 0.561_014_236_823_786),
          (0.5, 0.293_335_838_830_014_85, 0.737_209_495_229_636_3),
          (2.0, 0.052_559_540_646_527_58, 0.961_147_044_034_035_1),
        ],
        [
          -1.334_570_755_211_834_9,
          -0.144_990_071_783_220_7,
          1.291_494_477_029_585_5,
        ],
        [0.990_074_339_587_914_7, 5.595_401_399_341_534],
      ),
    ];
    for ((eta, lambda), grid, ppf, moments) in cases {
      let d = SimdSkewT::<f64>::new(eta, lambda, &Unseeded);
      for (z, pdf, cdf) in grid {
        assert!(close(d.pdf(z), pdf, 1e-12), "pdf({z}) = {}", d.pdf(z));
        assert!(close(d.cdf(z), cdf, 1e-11), "cdf({z}) = {}", d.cdf(z));
      }
      for (p, want) in [0.05, 0.5, 0.9].into_iter().zip(ppf) {
        assert!(
          close(d.inv_cdf(p), want, 1e-9),
          "ppf({p}) = {}",
          d.inv_cdf(p)
        );
      }
      assert!(
        close(d.skewness(), moments[0], 1e-7),
        "skewness {}",
        d.skewness()
      );
      assert!(
        close(d.kurtosis() + 3.0, moments[1], 1e-7),
        "kurtosis {}",
        d.kurtosis()
      );
      assert_eq!(d.median(), d.inv_cdf(0.5));
      assert!(close(d.mode(), -d.a / d.b, 1e-15));
    }
  }

  /// The density integrates to one with zero mean and unit variance.
  #[test]
  fn density_is_standardised() {
    let d = SimdSkewT::<f64>::new(5.0, -0.3, &Unseeded);
    let (lo, hi, n) = (-60.0_f64, 60.0_f64, 1_200_000usize);
    let h = (hi - lo) / n as f64;
    let (mut mass, mut m1, mut m2) = (0.0, 0.0, 0.0);
    for k in 0..n {
      let z = lo + (k as f64 + 0.5) * h;
      let f = d.pdf(z) * h;
      mass += f;
      m1 += z * f;
      m2 += z * z * f;
    }
    assert!((mass - 1.0).abs() < 1e-6, "mass {mass}");
    assert!(m1.abs() < 1e-5, "mean {m1}");
    assert!((m2 - 1.0).abs() < 1e-4, "second moment {m2}");
  }

  /// The two-piece sampler reproduces the standardisation and the sign of
  /// the skew.
  #[test]
  fn sample_moments_match_closed_forms() {
    for (eta, lambda, seed) in [(5.0, -0.3, 3u64), (8.0, 0.4, 5)] {
      let d = SimdSkewT::<f64>::new(eta, lambda, &Deterministic::new(seed));
      let n = 400_000;
      let mut xs = vec![0.0; n];
      d.fill_slice(&mut xs);
      let mean = xs.iter().sum::<f64>() / n as f64;
      let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
      let m3 = xs.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n as f64;
      assert!(mean.abs() < 0.01, "mean {mean}");
      assert!((var - 1.0).abs() < 0.03, "var {var}");
      assert!(
        (m3 - d.skewness()).abs() < 0.15,
        "m3 {m3} vs {}",
        d.skewness()
      );
      let below = xs.iter().filter(|x| **x < d.knot()).count() as f64 / n as f64;
      assert!(
        (below - 0.5 * (1.0 - lambda)).abs() < 0.01,
        "mass below the knot {below}"
      );
    }
  }

  #[test]
  fn deterministic_seed_reproduces_stream() {
    let a = SimdSkewT::<f64>::new(5.0, -0.3, &Deterministic::new(7));
    let b = SimdSkewT::<f64>::new(5.0, -0.3, &Deterministic::new(7));
    for _ in 0..256 {
      assert_eq!(a.sample_fast(), b.sample_fast());
    }
  }

  #[test]
  #[should_panic(expected = "eta must exceed 2")]
  fn rejects_small_eta() {
    let _ = SimdSkewT::<f64>::new(2.0, 0.0, &Unseeded);
  }
}

py_distribution!(PySkewT, SimdSkewT,
  sig: (eta, lambda, seed=None, dtype=None),
  params: (eta: f64, lambda: f64)
);

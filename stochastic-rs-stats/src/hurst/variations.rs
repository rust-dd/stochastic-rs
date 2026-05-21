//! Variations-ratio Hurst estimators (Daubechies filter, central
//! finite-difference, `k`-th order `p`-power variation).
//!
//! These all share the same scaling argument: for a self-similar
//! process with index `H`, the chosen variation measured at strides
//! `1` and `2` satisfies `V(2) / V(1) ≈ 2^{pH}` (up to the order-
//! specific factor), so `H` follows from a single log-ratio.
//!
//! References:
//! - Coeurjolly (2001) — *Estimating the parameters of a fractional
//!   Brownian motion by discrete variations of its sample paths*,
//!   Statistical Inference for Stochastic Processes 4(2), 199–227.
//! - Brouste & Iacus (2013) — *Parameter estimation for the discretely
//!   observed fractional Ornstein-Uhlenbeck process*, arXiv:1703.09372.

use std::f64::consts::SQRT_2;

use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::array;

use super::HurstDiagnostic;
use super::HurstError;
use super::HurstEstimator;
use super::HurstResult;
use crate::traits::FloatExt;

/// Selector for the underlying variation kernel used by [`Variations`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VariationKind {
  /// Daubechies-4 low-pass / dilated kernel (Coeurjolly variant).
  Daubechies,
  /// Second / fourth-order central difference moments ratio.
  CentralDiff,
  /// `k`-th order `p`-power variation at strides 1 and 2
  /// (arXiv:1703.09372).
  PowerVariation { k: usize, p: f64 },
}

/// Variations-ratio Hurst estimator.
///
/// Three flavours selectable via [`VariationKind`].  None of them
/// allocate beyond the auxiliary filter buffer.
#[derive(Clone, Copy, Debug)]
pub struct Variations {
  pub kind: VariationKind,
}

impl Default for Variations {
  fn default() -> Self {
    Self {
      kind: VariationKind::CentralDiff,
    }
  }
}

impl Variations {
  #[must_use]
  pub fn new(kind: VariationKind) -> Self {
    Self { kind }
  }
}

impl<T: FloatExt> HurstEstimator<T> for Variations {
  fn estimate(&self, x: ArrayView1<T>) -> Result<HurstResult<T>, HurstError> {
    let n = x.len();
    match self.kind {
      VariationKind::Daubechies => {
        if n < 8 {
          return Err(HurstError::TooFewObservations { got: n, required: 8 });
        }
        let (h, v1, v2) = daubechies_h_inner::<T>(x)?;
        Ok(HurstResult {
          hurst: T::from_f64_fast(h),
          std_err: None,
          n_obs: n,
          diagnostic: HurstDiagnostic::Variations {
            v_short: T::from_f64_fast(v1),
            v_long: T::from_f64_fast(v2),
          },
        })
      }
      VariationKind::CentralDiff => {
        if n < 5 {
          return Err(HurstError::TooFewObservations { got: n, required: 5 });
        }
        let (h, v1, v2) = central_diff_h_inner::<T>(x)?;
        Ok(HurstResult {
          hurst: T::from_f64_fast(h),
          std_err: None,
          n_obs: n,
          diagnostic: HurstDiagnostic::Variations {
            v_short: T::from_f64_fast(v1),
            v_long: T::from_f64_fast(v2),
          },
        })
      }
      VariationKind::PowerVariation { k, p } => {
        if k == 0 {
          return Err(HurstError::InvalidParameter("k", k as f64));
        }
        if !(p.is_finite() && p > 0.0) {
          return Err(HurstError::InvalidParameter("p", p));
        }
        let required = k + 2 + 1;
        if n < required {
          return Err(HurstError::TooFewObservations {
            got: n,
            required,
          });
        }
        let (h, v1, v2) = power_variation_h_inner::<T>(x, k, p)?;
        Ok(HurstResult {
          hurst: T::from_f64_fast(h),
          std_err: None,
          n_obs: n,
          diagnostic: HurstDiagnostic::Variations {
            v_short: T::from_f64_fast(v1),
            v_long: T::from_f64_fast(v2),
          },
        })
      }
    }
  }
}

/// Daubechies-4 wavelet coefficients used by both the Hurst variations
/// estimator and the multi-scale [`super::wavelet::Wavelet`] estimator.
#[inline]
pub(crate) fn daubechies_coeffs() -> Array1<f64> {
  array![
    0.482962913144534 / SQRT_2,
    -0.836516303737808 / SQRT_2,
    0.224143868042013 / SQRT_2,
    0.12940952255126 / SQRT_2,
  ]
}

/// Insert a zero between every coefficient (filter dilation `a_2`).
#[inline]
pub(crate) fn a2_coefficients(a: &Array1<f64>) -> Array1<f64> {
  let mut a_2 = Array1::<f64>::zeros(a.len() * 2);
  for (i, &val) in a.iter().enumerate() {
    a_2[i * 2 + 1] = val;
  }
  a_2
}

/// Direct-form-II IIR filter `lfilter` (subset of
/// `scipy.signal.lfilter`): `y[i] = Σⱼ b[j] x[i-j] - Σⱼ≥1 a[j] y[i-j]`
/// with zero initial conditions.  `x` is read as `f64`.
pub(crate) fn lfilter_f64<T: FloatExt>(
  b: &Array1<f64>,
  a: &Array1<f64>,
  x: ArrayView1<T>,
) -> Array1<f64> {
  let n = x.len();
  let mut y = Array1::<f64>::zeros(n);
  for i in 0..n {
    let mut acc = 0.0;
    for j in 0..b.len() {
      if i >= j {
        acc += b[j] * x[i - j].to_f64().unwrap_or(0.0);
      }
    }
    for j in 1..a.len() {
      if i >= j {
        acc -= a[j] * y[i - j];
      }
    }
    y[i] = acc;
  }
  y
}

/// Daubechies-filter based H estimator returning `(H, V_short, V_long)`.
pub(crate) fn daubechies_h_inner<T: FloatExt>(
  x: ArrayView1<T>,
) -> Result<(f64, f64, f64), HurstError> {
  let a = daubechies_coeffs();
  let a_2 = a2_coefficients(&a);
  let v1 = lfilter_f64::<T>(&a, &array![1.0], x).mapv(|y| y * y).sum();
  let v2 = lfilter_f64::<T>(&a_2, &array![1.0], x).mapv(|y| y * y).sum();
  if !(v1 > 0.0 && v2 > 0.0 && v1.is_finite() && v2.is_finite()) {
    return Err(HurstError::DegeneratePath);
  }
  let h = 0.5 * (v2 / v1).log2();
  if !h.is_finite() {
    return Err(HurstError::DegeneratePath);
  }
  Ok((h, v1, v2))
}

/// Central finite-difference based H estimator (2nd vs 4th order
/// squared-difference ratio).  Returns `(H, V_short, V_long)`.
pub(crate) fn central_diff_h_inner<T: FloatExt>(
  x: ArrayView1<T>,
) -> Result<(f64, f64, f64), HurstError> {
  let n = x.len();
  let to = |i: usize| x[i].to_f64().unwrap_or(0.0);
  let sum1: f64 = (0..(n - 4))
    .map(|i| {
      let d = to(i + 4) - 2.0 * to(i + 2) + to(i);
      d * d
    })
    .sum();
  let sum2: f64 = (0..(n - 2))
    .map(|i| {
      let d = to(i + 2) - 2.0 * to(i + 1) + to(i);
      d * d
    })
    .sum();
  if !(sum1 > 0.0 && sum2 > 0.0 && sum1.is_finite() && sum2.is_finite()) {
    return Err(HurstError::DegeneratePath);
  }
  let h = 0.5 * (sum1 / sum2).log2();
  if !h.is_finite() {
    return Err(HurstError::DegeneratePath);
  }
  Ok((h, sum2, sum1))
}

/// `k`-th order `p`-power variation Hurst estimator
/// (arXiv:1703.09372).  Returns `(H, V_short, V_long)`.
///
/// Uses the non-overlapping (subsample-by-stride) variant so that
/// `V_{stride=2} / V_{stride=1} → 2^{pH - 1}` (Coeurjolly scaling),
/// matching the closed-form `H = (1 + log₂(V₂/V₁)) / p`.  The
/// overlapping [`power_variation`] (used by
/// [`crate::fou_estimator::estimate_fou_v4`]) does not satisfy this
/// relation and produces biased Hurst estimates.
pub(crate) fn power_variation_h_inner<T: FloatExt>(
  x: ArrayView1<T>,
  k: usize,
  p: f64,
) -> Result<(f64, f64, f64), HurstError> {
  let v1 = power_variation_nonoverlap::<T>(x, k, p, 1);
  let v2 = power_variation_nonoverlap::<T>(x, k, p, 2);
  if !(v1 > 0.0 && v2 > 0.0 && v1.is_finite() && v2.is_finite()) {
    return Err(HurstError::DegeneratePath);
  }
  let h = (1.0 + (v2 / v1).log2()) / p;
  if !h.is_finite() {
    return Err(HurstError::DegeneratePath);
  }
  Ok((h, v1, v2))
}

/// `k`-th order `p`-power variation at the given stride, **overlapping**
/// windows (step 1).  Used by
/// [`crate::fou_estimator::estimate_fou_v4`] (its bit-exact regression
/// test pins this exact formula).  Caller checks length / parameter
/// validity.
pub(crate) fn power_variation<T: FloatExt>(
  x: ArrayView1<T>,
  k: usize,
  p: f64,
  stride: usize,
) -> f64 {
  let n = x.len();
  let span = k * stride;
  if n <= span {
    return 0.0;
  }
  let mut v = 0.0;
  for i in 0..(n - span) {
    let mut d = 0.0;
    for j in 0..=k {
      let coeff = diff_coeff(k, j);
      d += coeff * x[i + j * stride].to_f64().unwrap_or(0.0);
    }
    v += d.abs().powf(p);
  }
  v
}

/// `k`-th order `p`-power variation at the given stride,
/// **non-overlapping** (windows step by `stride`).  This is the
/// Coeurjolly / Brouste-Iacus variant and gives the unbiased
/// Hurst estimate via the `(1 + log₂(V₂/V₁)) / p` formula.
fn power_variation_nonoverlap<T: FloatExt>(
  x: ArrayView1<T>,
  k: usize,
  p: f64,
  stride: usize,
) -> f64 {
  let n = x.len();
  let span = k * stride;
  if n <= span {
    return 0.0;
  }
  let mut v = 0.0;
  let mut i = 0;
  while i + span < n {
    let mut d = 0.0;
    for j in 0..=k {
      let coeff = diff_coeff(k, j);
      d += coeff * x[i + j * stride].to_f64().unwrap_or(0.0);
    }
    v += d.abs().powf(p);
    i += stride;
  }
  v
}

#[inline]
pub(crate) fn diff_coeff(k: usize, j: usize) -> f64 {
  let sign = if ((k - j) & 1) == 0 { 1.0 } else { -1.0 };
  sign * binomial(k, j)
}

#[inline]
pub(crate) fn binomial(n: usize, k: usize) -> f64 {
  if k > n {
    return 0.0;
  }
  let k = k.min(n - k);
  if k == 0 {
    return 1.0;
  }
  let mut c = 1.0;
  for i in 1..=k {
    c *= (n - k + i) as f64 / i as f64;
  }
  c
}

#[cfg(test)]
mod tests {
  use super::*;
  use stochastic_rs_core::simd_rng::Unseeded;
  use stochastic_rs_stochastic::process::fbm::Fbm;

  use crate::traits::ProcessExt;

  #[test]
  fn central_diff_matches_known_h_on_fbm() {
    let h = 0.7_f64;
    let m = 32;
    let mut acc = 0.0;
    for _ in 0..m {
      let fbm = Fbm::new(h, 4096, Some(1.0), Unseeded);
      let path = fbm.sample();
      let r = Variations {
        kind: VariationKind::CentralDiff,
      }
      .estimate(path.view())
      .expect("central diff variations");
      acc += r.hurst;
    }
    let h_est = acc / m as f64;
    assert!(
      (h_est - h).abs() < 0.05,
      "central diff H={h_est:.3}, expected {h:.3}"
    );
  }

  #[test]
  fn daubechies_matches_known_h_on_fbm() {
    let h = 0.3_f64;
    let m = 32;
    let mut acc = 0.0;
    for _ in 0..m {
      let fbm = Fbm::new(h, 4096, Some(1.0), Unseeded);
      let path = fbm.sample();
      let r = Variations {
        kind: VariationKind::Daubechies,
      }
      .estimate(path.view())
      .expect("daubechies variations");
      acc += r.hurst;
    }
    let h_est = acc / m as f64;
    assert!(
      (h_est - h).abs() < 0.08,
      "daubechies H={h_est:.3}, expected {h:.3}"
    );
  }
}

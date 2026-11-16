//! Variations-ratio Hurst estimators: a filter's variation at its dilation over that at lag 1, as
//! Brouste & Iacus's plain sums for the two filters and Coeurjolly's means for the power variation.
//!
//! $$
//! \hat H=\tfrac12\log_2\frac{V_{\mathbf a^2}}{V_{\mathbf a}},\qquad
//! \hat H=\frac1p\log_2\frac{V_{k,p}(2)/(n-2k)}{V_{k,p}(1)/(n-k)},\qquad
//! \text{each ratio}\to2^{pH}\ (p=2\text{ for the filters})
//! $$
//!
//! Reference: Coeurjolly, J.-F. (2001), "Estimating the parameters of a fractional Brownian motion by discrete variations of its sample paths", Statistical Inference for Stochastic Processes 4(2), 199–227, DOI 10.1023/A:1017507306245.
//!
//! Reference: Brouste, A. & Iacus, S. M. (2013), "Parameter estimation for the discretely observed fractional Ornstein–Uhlenbeck process and the Yuima R package", Computational Statistics 28(4), 1529–1547, DOI 10.1007/s00180-012-0365-6, arXiv:1112.3777.
//!
//! Reference: Istas, J. & Lang, G. (1997), "Quadratic variations and estimation of the local Hölder index of a Gaussian process", Annales de l'Institut Henri Poincaré (B) Probability and Statistics 33(4), 407–436, DOI 10.1016/S0246-0203(97)80099-4.

use std::f64::consts::SQRT_2;

use ndarray::ArrayView1;

use super::HurstDiagnostic;
use super::HurstEstimator;
use super::HurstFit;
use crate::StatsError;
use crate::traits::FloatExt;

/// Selector for the underlying variation kernel used by [`Variations`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum VariationKind {
  /// Brouste & Iacus's order-2 Daubechies filter and its dilation, plain sums over full windows.
  Daubechies,
  /// The second difference at lags 1 and 2, Brouste & Iacus's plain sums with `a = (1, −2, 1)`.
  CentralDiff,
  /// The `k`-th difference's `p`-power variations at lags 1 and 2 at every position, each over its
  /// count (Coeurjolly 2001), as [`crate::fou::fit_fou_power_variation`] reads `H`.
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
  fn estimate(&self, x: ArrayView1<T>) -> Result<HurstFit<T>, StatsError> {
    let n = x.len();
    match self.kind {
      VariationKind::Daubechies => {
        if n < 8 {
          return Err(StatsError::InsufficientData { needed: 8, got: n });
        }
        let (h, v1, v2) = daubechies_h_inner::<T>(x)?;
        Ok(HurstFit {
          hurst: T::from_f64_fast(h),
          std_err: None,
          nobs: n,
          diagnostic: HurstDiagnostic::Variations {
            v_short: T::from_f64_fast(v1),
            v_long: T::from_f64_fast(v2),
          },
        })
      }
      VariationKind::CentralDiff => {
        if n < 5 {
          return Err(StatsError::InsufficientData { needed: 5, got: n });
        }
        let (h, v1, v2) = central_diff_h_inner::<T>(x)?;
        Ok(HurstFit {
          hurst: T::from_f64_fast(h),
          std_err: None,
          nobs: n,
          diagnostic: HurstDiagnostic::Variations {
            v_short: T::from_f64_fast(v1),
            v_long: T::from_f64_fast(v2),
          },
        })
      }
      VariationKind::PowerVariation { k, p } => {
        if k == 0 {
          return Err(StatsError::InvalidParameter {
            name: "k",
            value: k as f64,
            constraint: "k >= 1",
          });
        }
        if !(p.is_finite() && p > 0.0) {
          return Err(StatsError::InvalidParameter {
            name: "p",
            value: p,
            constraint: "p > 0",
          });
        }
        let required = k.saturating_mul(2).saturating_add(1);
        if n < required {
          return Err(StatsError::InsufficientData {
            needed: required,
            got: n,
          });
        }
        let (h, v1, v2) = power_variation_h_inner::<T>(x, k, p)?;
        Ok(HurstFit {
          hurst: T::from_f64_fast(h),
          std_err: None,
          nobs: n,
          diagnostic: HurstDiagnostic::Variations {
            v_short: T::from_f64_fast(v1),
            v_long: T::from_f64_fast(v2),
          },
        })
      }
    }
  }
}

/// Brouste & Iacus's order-2 Daubechies filter, their Remark 1's digits.
pub(crate) fn daubechies_coeffs() -> [f64; 4] {
  [
    0.4829629131445341 / SQRT_2,
    -0.8365163037378077 / SQRT_2,
    0.2241438680420134 / SQRT_2,
    0.1294095225512603 / SQRT_2,
  ]
}

/// Brouste & Iacus's `V_{N,a} = Σᵢ (Σₖ aₖ X_{i+k})²` over full windows, with the filter dilated
/// `dilation` times (`a²` at 2); a plain sum.
pub(crate) fn filtered_variation<T: FloatExt>(x: ArrayView1<T>, a: &[f64], dilation: usize) -> f64 {
  let span = (a.len() - 1) * dilation;
  let n = x.len();
  if n <= span {
    return 0.0;
  }
  let mut v = 0.0;
  for i in 0..(n - span) {
    let mut d = 0.0;
    for (k, &ak) in a.iter().enumerate() {
      d += ak * x[i + k * dilation].to_f64().unwrap_or(0.0);
    }
    v += d * d;
  }
  v
}

/// Brouste & Iacus's `H` with their Daubechies filter, plain sums; returns `(H, V_short, V_long)`.
pub(crate) fn daubechies_h_inner<T: FloatExt>(
  x: ArrayView1<T>,
) -> Result<(f64, f64, f64), StatsError> {
  let a = daubechies_coeffs();
  let v1 = filtered_variation(x, &a, 1);
  let v2 = filtered_variation(x, &a, 2);
  if !(v1 > 0.0 && v2 > 0.0 && v1.is_finite() && v2.is_finite()) {
    return Err(StatsError::DegenerateData("constant or zero-variance path"));
  }
  let h = 0.5 * (v2 / v1).log2();
  if !h.is_finite() {
    return Err(StatsError::DegenerateData("constant or zero-variance path"));
  }
  Ok((h, v1, v2))
}

/// Brouste & Iacus's `H` with the second difference, plain sums; returns `(H, V_short, V_long)`.
pub(crate) fn central_diff_h_inner<T: FloatExt>(
  x: ArrayView1<T>,
) -> Result<(f64, f64, f64), StatsError> {
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
    return Err(StatsError::DegenerateData("constant or zero-variance path"));
  }
  let h = 0.5 * (sum1 / sum2).log2();
  if !h.is_finite() {
    return Err(StatsError::DegenerateData("constant or zero-variance path"));
  }
  Ok((h, sum2, sum1))
}

/// [`power_variation_hurst`] off the variations at lags 1 and 2; returns `(H, V_short, V_long)`.
pub(crate) fn power_variation_h_inner<T: FloatExt>(
  x: ArrayView1<T>,
  k: usize,
  p: f64,
) -> Result<(f64, f64, f64), StatsError> {
  let v1 = power_variation::<T>(x, k, p, 1);
  let v2 = power_variation::<T>(x, k, p, 2);
  if !(v1 > 0.0 && v2 > 0.0 && v1.is_finite() && v2.is_finite()) {
    return Err(StatsError::DegenerateData("constant or zero-variance path"));
  }
  let h = power_variation_hurst(v1, v2, x.len(), k, p);
  if !h.is_finite() {
    return Err(StatsError::DegenerateData("constant or zero-variance path"));
  }
  Ok((h, v1, v2))
}

/// `k`-th order `p`-power variation at lag `stride`, summed over every position; the caller checks
/// that a window fits.
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

/// Coeurjolly's `log₂(V̄(2)/V̄(1))/p` on `n` points, the lag-`s` variation divided by its `n − ks`
/// windows; the caller checks `n ≥ 2k + 1`.
pub(crate) fn power_variation_hurst(v1: f64, v2: f64, n: usize, k: usize, p: f64) -> f64 {
  ((v2 / (n - 2 * k) as f64) / (v1 / (n - k) as f64)).log2() / p
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
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_core::simd_rng::Unseeded;
  use stochastic_rs_stochastic::process::fbm::Fbm;

  use super::*;
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
  fn power_variation_reads_the_fou_fit_hurst() {
    let path = Fbm::new(0.7, 512, Some(1.0), Deterministic::new(5)).sample();
    for (k, p) in [(1, 2.0), (2, 2.0), (3, 1.5)] {
      let kind = Variations::new(VariationKind::PowerVariation { k, p });
      let fit = HurstEstimator::<f64>::estimate(&kind, path.view()).unwrap();
      let fou = crate::fou::fit_fou_power_variation(path.view(), None, k, p, None, Some(1.0));
      assert_eq!(
        fit.hurst.to_bits(),
        fou.unwrap().hurst.to_bits(),
        "k = {k}, p = {p}"
      );
    }
  }

  #[test]
  fn power_variation_needs_2k_plus_1_points() {
    let path = |n: usize| ndarray::Array1::from_iter((0..n).map(|i| (2.0 * i as f64).sin()));
    let kind = |k| Variations::new(VariationKind::PowerVariation { k, p: 2.0 });
    assert_eq!(
      HurstEstimator::<f64>::estimate(&kind(3), path(6).view()).unwrap_err(),
      StatsError::InsufficientData { needed: 7, got: 6 }
    );
    assert!(HurstEstimator::<f64>::estimate(&kind(1), path(3).view()).is_ok());
  }

  #[test]
  fn a_level_shift_leaves_the_daubechies_estimate_alone() {
    let path = Fbm::new(0.3, 1024, Some(1.0), Deterministic::new(3)).sample();
    let daubechies = Variations::new(VariationKind::Daubechies);
    let level = HurstEstimator::<f64>::estimate(&daubechies, path.view()).unwrap();
    let shifted = HurstEstimator::<f64>::estimate(&daubechies, (&path + 5.0).view()).unwrap();
    assert!(
      (shifted.hurst - level.hurst).abs() < 1e-9,
      "{level:?} {shifted:?}"
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

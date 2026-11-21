//! σ as its share `w ∈ [0, 1]` of the way from its floor to Feller's cap, which is then a bound of
//! the coordinate's box: above a projected cap σ had no slope, and a fit never left it.

use ndarray::Array2;
use ndarray::ArrayView1;

use super::heston::params::feller_sigma_cap;

pub(crate) const P_SHARE: (f64, f64) = (0.0, 1.0);

/// The largest σ with `σ² ≤ 2κθ`, at most the box's `ceiling`.
pub(crate) fn cap(kappa: f64, theta: f64, ceiling: f64) -> f64 {
  feller_sigma_cap(kappa, theta).min(ceiling)
}

/// $\sigma = \sigma_{lo} + w\,(\mathrm{cap}(\kappa,\theta) - \sigma_{lo})$ in σ's box
/// `(σ_lo, ceiling)`.
pub(crate) fn sigma(share: f64, kappa: f64, theta: f64, (floor, ceiling): (f64, f64)) -> f64 {
  let cap = cap(kappa, theta, ceiling);
  // At `w = 1` the sum can round an ulp past the cap.
  (floor + share * (cap - floor)).min(cap)
}

/// [`sigma`]'s inverse at a point of the box; 1 for a σ past the cap.
pub(crate) fn share(sigma: f64, kappa: f64, theta: f64, (floor, ceiling): (f64, f64)) -> f64 {
  ((sigma - floor) / (cap(kappa, theta, ceiling) - floor)).clamp(0.0, 1.0)
}

/// `[∂σ/∂κ, ∂σ/∂θ, ∂σ/∂w]` of [`sigma`]: the cap moves with `κ` and `θ` only under the ceiling.
pub(crate) fn slopes(share: f64, kappa: f64, theta: f64, (floor, ceiling): (f64, f64)) -> [f64; 3] {
  let cap = cap(kappa, theta, ceiling);
  let moving = if cap < ceiling { share * cap } else { 0.0 };
  [moving / (2.0 * kappa), moving / (2.0 * theta), cap - floor]
}

/// `∂θ/∂x` of the natural parameters: the folds' slopes on the diagonal, and for each `(κ's
/// index, slopes)` σ's row, two below `κ`'s, through its share.
pub(crate) fn natural_jacobian(
  folds: ArrayView1<f64>,
  sigmas: &[(usize, [f64; 3])],
) -> Array2<f64> {
  let mut jacobian = Array2::from_diag(&folds);
  for &(kappa, slopes) in sigmas {
    for (offset, slope) in slopes.into_iter().enumerate() {
      jacobian[(kappa + 2, kappa + offset)] = slope * folds[kappa + offset];
    }
  }
  jacobian
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Over the calibrators' `κ ≥ 0.1`, `θ ≥ 0.001` the cap is at least `0.01414`, above σ's floor.
  #[test]
  fn the_map_rises_strictly_in_the_share_and_inverts() {
    let range = (0.01, 1.0);
    for (kappa, theta) in [(0.1, 0.001), (1.5, 0.04), (20.0, 0.4), (20.0, 1.0)] {
      let cap = cap(kappa, theta, range.1);
      assert!(cap >= 0.01414 && cap * cap <= 2.0 * kappa * theta, "{cap}");
      let shares = (0..=1000).map(|i| f64::from(i) / 1000.0);
      let sigmas = shares.clone().map(|w| sigma(w, kappa, theta, range));
      let sigmas = sigmas.collect::<Vec<_>>();
      assert!(sigmas.windows(2).all(|pair| pair[0] < pair[1]));
      assert_eq!((sigmas[0], sigmas[1000]), (range.0, cap));
      for (w, s) in shares.zip(sigmas) {
        assert!((share(s, kappa, theta, range) - w).abs() < 1e-13, "{w}");
      }
    }
    assert_eq!(share(0.5, 1.5, 0.04, range), 1.0);
  }

  #[test]
  fn the_slopes_are_the_maps_differences() {
    let range = (0.01, 1.0);
    for (w, kappa, theta) in [(0.7, 1.5, 0.04), (0.3, 0.1, 0.001), (0.9, 12.0, 0.3)] {
      let h = 1e-6;
      let by = |dk: f64, dt: f64, dw: f64| {
        let up = sigma(w + dw, kappa + dk, theta + dt, range);
        (up - sigma(w - dw, kappa - dk, theta - dt, range)) / (2.0 * h)
      };
      let numeric = [by(h, 0.0, 0.0), by(0.0, h, 0.0), by(0.0, 0.0, h)];
      for (analytic, numeric) in slopes(w, kappa, theta, range).into_iter().zip(numeric) {
        let gap = (analytic - numeric).abs();
        assert!(
          gap < 1e-6 * analytic.abs().max(1.0),
          "{analytic} against {numeric}"
        );
      }
    }
  }
}

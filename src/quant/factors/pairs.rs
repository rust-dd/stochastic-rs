//! Cointegrated pairs-trading framework.
//!
//! 1. Estimate the static hedge ratio $\hat\beta$ from
//!    $y_t = \alpha + \beta x_t + \varepsilon_t$ by OLS.
//! 2. Form the spread $s_t = y_t - \hat\alpha - \hat\beta x_t$.
//! 3. Standardise to a z-score using a rolling or full-sample mean and
//!    standard deviation.
//! 4. Emit entry / exit signals around the supplied thresholds.
//!
//! For a formal mean-reversion check the residual $\varepsilon_t$ should be
//! tested for unit-root with [`crate::stats::stationarity::adf::adf_test`].
//!
//! Reference: Engle, Granger, "Co-Integration and Error Correction:
//! Representation, Estimation, and Testing", Econometrica, 55(2), 251-276
//! (1987). DOI: 10.2307/1913236
//!
//! Reference: Gatev, Goetzmann, Rouwenhorst, "Pairs Trading: Performance of a
//! Relative-Value Arbitrage Rule", Review of Financial Studies, 19(3),
//! 797-827 (2006). DOI: 10.1093/rfs/hhj020

use std::fmt::Display;

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray_linalg::LeastSquaresSvd;

/// Side of an open pairs position.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PairsSignal {
  /// No position held.
  #[default]
  Flat,
  /// Long the spread $s_t = y_t - \beta x_t$ (buy $y$, sell scaled $x$).
  LongSpread,
  /// Short the spread (sell $y$, buy scaled $x$).
  ShortSpread,
}

impl Display for PairsSignal {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Flat => write!(f, "Flat"),
      Self::LongSpread => write!(f, "Long spread"),
      Self::ShortSpread => write!(f, "Short spread"),
    }
  }
}

/// Configuration of a cointegrated pairs-trading strategy.
#[derive(Debug, Clone)]
pub struct PairsStrategy {
  /// Estimated intercept $\hat\alpha$.
  pub alpha: f64,
  /// Estimated hedge ratio $\hat\beta$.
  pub beta: f64,
  /// Realised spread $s_t = y_t - \hat\alpha - \hat\beta x_t$.
  pub spread: Array1<f64>,
  /// Z-score series of the spread.
  pub z_score: Array1<f64>,
  /// Signal series.
  pub signals: Array1<PairsSignal>,
  /// Mean of the spread used to compute the z-score.
  pub spread_mean: f64,
  /// Standard deviation of the spread used to compute the z-score.
  pub spread_std: f64,
}

/// Compute pairs-trading signals.
///
/// `entry_z`: enter when `|z| ≥ entry_z`. `exit_z`: close when `|z| ≤ exit_z`.
/// Otherwise the previous signal persists. `entry_z` must exceed `exit_z`.
pub fn pairs_signals(
  y: ArrayView1<f64>,
  x: ArrayView1<f64>,
  entry_z: f64,
  exit_z: f64,
) -> PairsStrategy {
  assert_eq!(y.len(), x.len(), "y and x must have equal length");
  assert!(entry_z > exit_z, "entry_z must exceed exit_z");
  let n = y.len();
  assert!(n >= 3, "need at least three observations");
  let mut design = Array2::<f64>::zeros((n, 2));
  for i in 0..n {
    design[[i, 0]] = 1.0;
    design[[i, 1]] = x[i];
  }
  let y_owned = y.to_owned();
  let sol = design
    .least_squares(&y_owned)
    .expect("hedge ratio OLS failed");
  let alpha = sol.solution[0];
  let beta = sol.solution[1];
  let mut spread = Array1::<f64>::zeros(n);
  for i in 0..n {
    spread[i] = y[i] - alpha - beta * x[i];
  }
  let mean = spread.iter().sum::<f64>() / n as f64;
  let var = spread.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
  let sd = var.sqrt().max(1e-12);
  let z = spread.mapv(|v| (v - mean) / sd);
  let mut signals = Array1::<PairsSignal>::from_elem(n, PairsSignal::Flat);
  let mut state = PairsSignal::Flat;
  for i in 0..n {
    let zi = z[i];
    state = match state {
      PairsSignal::Flat => {
        if zi >= entry_z {
          PairsSignal::ShortSpread
        } else if zi <= -entry_z {
          PairsSignal::LongSpread
        } else {
          PairsSignal::Flat
        }
      }
      PairsSignal::LongSpread => {
        if zi.abs() <= exit_z {
          PairsSignal::Flat
        } else {
          PairsSignal::LongSpread
        }
      }
      PairsSignal::ShortSpread => {
        if zi.abs() <= exit_z {
          PairsSignal::Flat
        } else {
          PairsSignal::ShortSpread
        }
      }
    };
    signals[i] = state;
  }
  PairsStrategy {
    alpha,
    beta,
    spread,
    z_score: z,
    signals,
    spread_mean: mean,
    spread_std: sd,
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;
  use crate::distributions::normal::SimdNormal;

  #[test]
  fn signals_flat_when_spread_within_band() {
    let y = array![1.0_f64, 1.01, 1.0, 1.02, 1.005];
    let x = array![0.5_f64, 0.51, 0.5, 0.52, 0.505];
    let s = pairs_signals(y.view(), x.view(), 2.0, 0.5);
    assert!(s.signals.iter().all(|&v| matches!(v, PairsSignal::Flat)));
  }

  #[test]
  fn long_spread_triggered_by_negative_extreme() {
    let mut y_buf = vec![0.0_f64; 200];
    let mut x_buf = vec![0.0_f64; 200];
    let dist = SimdNormal::<f64>::with_seed(0.0, 0.01, 1);
    let mut shocks = vec![0.0_f64; 200];
    dist.fill_slice_fast(&mut shocks);
    for i in 0..200 {
      x_buf[i] = 100.0 + 0.05 * i as f64;
      y_buf[i] = 2.0 * x_buf[i] + 1.0 + shocks[i];
    }
    y_buf[150] -= 5.0;
    let y = Array1::from(y_buf);
    let x = Array1::from(x_buf);
    let s = pairs_signals(y.view(), x.view(), 1.5, 0.5);
    assert!(matches!(s.signals[150], PairsSignal::LongSpread));
  }

  #[test]
  fn beta_recovered_from_linear_relationship() {
    let dist = SimdNormal::<f64>::with_seed(0.0, 0.005, 7);
    let mut shocks = vec![0.0_f64; 500];
    dist.fill_slice_fast(&mut shocks);
    let mut x_buf = vec![0.0_f64; 500];
    let mut y_buf = vec![0.0_f64; 500];
    for i in 0..500 {
      x_buf[i] = (i as f64) * 0.01;
      y_buf[i] = 0.3 + 1.7 * x_buf[i] + shocks[i];
    }
    let y = Array1::from(y_buf);
    let x = Array1::from(x_buf);
    let s = pairs_signals(y.view(), x.view(), 2.0, 0.5);
    assert!((s.beta - 1.7).abs() < 0.01);
    assert!((s.alpha - 0.3).abs() < 0.05);
  }
}

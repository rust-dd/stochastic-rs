//! Monte Carlo Greeks by Malliavin integration-by-parts weights, for GBM and Heston:
//!
//! $$
//! \Delta=\mathbb E\!\left[e^{-rT}\Phi(S_T)\,\frac{W_T}{S_0\sigma T}\right]
//! $$
//!

mod gbm;
mod heston;
mod heston_el_khatib;

use stochastic_rs_stochastic::mc::McEstimate;

use crate::pricing::mc_stats::std_err_from_sums;

pub use gbm::GbmMalliavinGreeks;
pub use heston::HestonMalliavinGreeks;

fn estimate(sum: f64, sum_sq: f64, n: usize) -> McEstimate<f64> {
  McEstimate {
    mean: sum / n as f64,
    std_err: std_err_from_sums(sum, sum_sq, n),
    n_samples: n,
  }
}

/// A query no path is simulated for.
fn undefined() -> McEstimate<f64> {
  McEstimate {
    mean: f64::NAN,
    std_err: f64::NAN,
    n_samples: 0,
  }
}

#[cfg(test)]
mod tests;

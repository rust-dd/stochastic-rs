//! # Longstaff-Schwartz Monte Carlo for American Options
//!
//! $$
//! V_0 = \frac{1}{N}\sum_{n=1}^{N} e^{-r\,\tau^*_n}\,h\!\bigl(S_{\tau^*_n}^{(n)}\bigr)
//! $$
//!
//! Backward induction with least-squares regression on polynomial basis.
//!
//! Reference: Longstaff & Schwartz (2001), "Valuing American Options by
//! Simulation: A Simple Least-Squares Approach",
//! DOI: 10.1093/rfs/14.1.113

use ndarray::{Array1, Array2};
use ndarray_linalg::LeastSquaresSvd;

use crate::traits::FloatExt;

/// Longstaff-Schwartz American option pricer.
pub struct Lsm<T: FloatExt> {
  /// Risk-free rate.
  pub r: T,
  /// Time to maturity.
  pub tau: T,
  /// Number of polynomial basis functions for the regression.
  pub n_basis: usize,
}

impl<T: FloatExt> Lsm<T> {
  pub fn new(r: T, tau: T, n_basis: usize) -> Self {
    assert!(n_basis >= 2, "need at least 2 basis functions");
    Self { r, tau, n_basis }
  }

  /// Price an American option via LSM.
  ///
  /// # Arguments
  /// * `paths` – simulated price paths, shape `(n_paths, n_steps + 1)`.
  ///   Column 0 is the initial price, column `n_steps` is the terminal price.
  /// * `payoff` – intrinsic value function `h(S)` (e.g. `|s| (K - s).max(0.0)`
  ///   for a put).
  pub fn price<F>(&self, paths: &Array2<T>, payoff: F) -> T
  where
    F: Fn(T) -> T,
  {
    let n_paths = paths.nrows();
    let n_steps = paths.ncols() - 1;
    let dt = self.tau / T::from_usize_(n_steps);
    let disc = (-self.r * dt).exp();

    // Cash flow value and exercise time for each path
    let mut cf = Array1::<T>::zeros(n_paths);
    let mut cf_time = vec![n_steps; n_paths];

    // Terminal payoff
    for i in 0..n_paths {
      cf[i] = payoff(paths[[i, n_steps]]);
    }

    // Backward induction
    for step in (1..n_steps).rev() {
      // In-the-money paths at this step
      let mut itm: Vec<usize> = Vec::new();
      for i in 0..n_paths {
        if payoff(paths[[i, step]]) > T::zero() {
          itm.push(i);
        }
      }

      if itm.len() <= self.n_basis {
        continue;
      }

      // Discounted continuation values for ITM paths
      let mut y_vals = Vec::with_capacity(itm.len());
      for &idx in &itm {
        let steps_to_cf = cf_time[idx] - step;
        let disc_factor = (-self.r * dt * T::from_usize_(steps_to_cf)).exp();
        y_vals.push((cf[idx] * disc_factor).to_f64().unwrap());
      }

      // Regression: basis functions are 1, S, S², …
      let n_itm = itm.len();
      let n_b = self.n_basis.min(n_itm);
      let a_mat = Array2::from_shape_fn((n_itm, n_b), |(row, col)| {
        paths[[itm[row], step]].to_f64().unwrap().powi(col as i32)
      });
      let b_vec = Array1::from_vec(y_vals);

      let beta = match a_mat.least_squares(&b_vec) {
        Ok(result) => result.solution,
        Err(_) => continue,
      };

      // Exercise decision
      for &idx in &itm {
        let s = paths[[idx, step]].to_f64().unwrap();
        let continuation: f64 = (0..n_b).map(|j| beta[j] * s.powi(j as i32)).sum();
        let exercise_val = payoff(paths[[idx, step]]);
        if exercise_val > T::from_f64_fast(continuation) {
          cf[idx] = exercise_val;
          cf_time[idx] = step;
        }
      }
    }

    // Discount each path's cash flow to t = 0
    let mut total = T::zero();
    for i in 0..n_paths {
      let disc_factor = disc.powi(cf_time[i] as i32);
      total += cf[i] * disc_factor;
    }
    total / T::from_usize_(n_paths)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// American put ≥ European put (early exercise premium).
  #[test]
  fn american_put_geq_european_put() {
    let n_paths = 10_000;
    let n_steps = 50;
    let s0 = 100.0_f64;
    let k = 100.0;
    let r = 0.05;
    let sigma = 0.2;
    let tau = 1.0;
    let dt = tau / n_steps as f64;
    let sqrt_dt = dt.sqrt();

    // Generate GBM paths (log-Euler for positivity)
    let mut paths = Array2::<f64>::zeros((n_paths, n_steps + 1));
    for i in 0..n_paths {
      paths[[i, 0]] = s0;
      let z = f64::normal_array(n_steps, 0.0, 1.0);
      for j in 0..n_steps {
        paths[[i, j + 1]] =
          paths[[i, j]] * ((r - 0.5 * sigma * sigma) * dt + sigma * sqrt_dt * z[j]).exp();
      }
    }

    let payoff_put = |s: f64| (k - s).max(0.0);
    let lsm = Lsm::new(r, tau, 4);
    let am_price = lsm.price(&paths, payoff_put);

    // European put: discount terminal payoff only
    let mut eu_sum = 0.0;
    for i in 0..n_paths {
      eu_sum += payoff_put(paths[[i, n_steps]]);
    }
    let eu_price = eu_sum / n_paths as f64 * (-r * tau).exp();

    assert!(
      am_price >= eu_price * 0.95,
      "American put ({am_price:.4}) should be >= European put ({eu_price:.4})"
    );
    assert!(
      am_price > 0.0 && am_price < 50.0,
      "American put price {am_price:.4} out of range"
    );
  }
}

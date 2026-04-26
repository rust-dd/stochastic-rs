//! # Bermudan
//!
//! Bermudan option pricer via Longstaff-Schwartz Monte Carlo. The holder
//! may exercise on a finite set of pre-specified exercise dates inside
//! $[0, T]$.
//!
//! $$
//! V_0 = \sup_{\tau \in \mathcal T} E^{\mathbb Q}\!\big[e^{-r\tau} h(S_\tau)\big],
//! \qquad \mathcal T = \{t_{e_1}, \ldots, t_{e_K}\}
//! $$
//!
//! Source:
//! - Longstaff, F. & Schwartz, E. (2001), "Valuing American Options by Simulation:
//!   A Simple Least-Squares Approach", DOI: 10.1093/rfs/14.1.113
//! - Glasserman, P. (2004), "Monte Carlo Methods in Financial Engineering", §8
//!
use ndarray::Array1;
use ndarray::Array2;
use ndarray_linalg::LeastSquaresSvd;

use stochastic_rs_stochastic::diffusion::gbm_log::GBMLog;
use crate::traits::ProcessExt;

/// Bermudan pricer driven by a path matrix and a set of exercise indices.
///
/// `paths` has shape `(n_paths, n_steps + 1)` with column 0 the initial
/// price. `exercise_steps` are the column indices at which the holder may
/// exercise; they must be strictly increasing and the last one is typically
/// `n_steps` (maturity).
#[derive(Debug, Clone)]
pub struct BermudanLsmPricer {
  /// Risk-free rate (constant).
  pub r: f64,
  /// Total maturity in years.
  pub tau: f64,
  /// Number of polynomial basis functions for the regression.
  pub n_basis: usize,
}

impl BermudanLsmPricer {
  pub fn new(r: f64, tau: f64, n_basis: usize) -> Self {
    assert!(n_basis >= 2, "need at least 2 basis functions");
    Self { r, tau, n_basis }
  }

  /// Price a Bermudan option. `exercise_steps` lists the *column indices*
  /// in `paths` at which the holder may exercise.
  pub fn price<F>(&self, paths: &Array2<f64>, exercise_steps: &[usize], payoff: F) -> f64
  where
    F: Fn(f64) -> f64,
  {
    assert!(!exercise_steps.is_empty(), "no exercise dates");
    let n_paths = paths.nrows();
    let n_steps = paths.ncols() - 1;
    assert!(
      exercise_steps.iter().all(|&s| s <= n_steps),
      "exercise step out of range"
    );
    let dt = self.tau / n_steps as f64;
    let disc_step = (-self.r * dt).exp();

    let last_idx = *exercise_steps.last().unwrap();
    let mut cf = Array1::<f64>::zeros(n_paths);
    let mut cf_time: Vec<usize> = vec![last_idx; n_paths];

    // Initial cash flow is the payoff at the latest exercise date
    for i in 0..n_paths {
      cf[i] = payoff(paths[[i, last_idx]]);
    }

    // Backward induction over exercise dates only
    for &step in exercise_steps.iter().rev().skip(1) {
      let mut itm: Vec<usize> = Vec::new();
      for i in 0..n_paths {
        if payoff(paths[[i, step]]) > 0.0 {
          itm.push(i);
        }
      }
      if itm.len() <= self.n_basis {
        continue;
      }

      let n_itm = itm.len();
      let n_b = self.n_basis.min(n_itm);
      let mut y_vals = Vec::with_capacity(n_itm);
      for &idx in &itm {
        let steps_to_cf = cf_time[idx] - step;
        let disc_factor = disc_step.powi(steps_to_cf as i32);
        y_vals.push(cf[idx] * disc_factor);
      }
      let a_mat = Array2::from_shape_fn((n_itm, n_b), |(row, col)| {
        paths[[itm[row], step]].powi(col as i32)
      });
      let b_vec = Array1::from_vec(y_vals);

      let beta = match a_mat.least_squares(&b_vec) {
        Ok(result) => result.solution,
        Err(_) => continue,
      };
      for &idx in &itm {
        let s = paths[[idx, step]];
        let continuation: f64 = (0..n_b).map(|j| beta[j] * s.powi(j as i32)).sum();
        let exercise_val = payoff(s);
        if exercise_val > continuation {
          cf[idx] = exercise_val;
          cf_time[idx] = step;
        }
      }
    }

    let mut total = 0.0;
    for i in 0..n_paths {
      let disc = disc_step.powi(cf_time[i] as i32);
      total += cf[i] * disc;
    }
    total / n_paths as f64
  }
}

/// Generate `n_paths` independent GBM paths on a uniform time grid using
/// the project's [`GBMLog`] process (log-Euler scheme). Returns an
/// `(n_paths, n_steps + 1)` matrix where column `0` is the spot and column
/// `n_steps` the terminal value.
pub fn generate_gbm_paths(
  s0: f64,
  r: f64,
  q: f64,
  sigma: f64,
  t: f64,
  n_paths: usize,
  n_steps: usize,
) -> Array2<f64> {
  let b = r - q;
  let process = GBMLog::<f64>::new(
    None,
    Some(b),
    None,
    None,
    sigma,
    n_steps + 1,
    Some(s0),
    Some(t),
  );
  let samples = process.sample_par(n_paths);
  let mut paths = Array2::<f64>::zeros((n_paths, n_steps + 1));
  for (i, sample) in samples.into_iter().enumerate() {
    paths.row_mut(i).assign(&sample);
  }
  paths
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Bermudan put on the *full* exercise grid (every step) should match an
  /// American LSM and exceed the European put.
  #[test]
  fn bermudan_full_grid_above_european() {
    let s0 = 100.0;
    let k = 100.0;
    let r = 0.05;
    let sigma = 0.20;
    let t = 1.0;
    let n_paths = 20_000;
    let n_steps = 50;

    let paths = generate_gbm_paths(s0, r, 0.0, sigma, t, n_paths, n_steps);
    let exercise: Vec<usize> = (1..=n_steps).collect();
    let bermudan = BermudanLsmPricer::new(r, t, 4);
    let payoff = |s: f64| (k - s).max(0.0);
    let berm_price = bermudan.price(&paths, &exercise, payoff);

    let mut eu_sum = 0.0;
    for i in 0..n_paths {
      eu_sum += payoff(paths[[i, n_steps]]);
    }
    let eu = eu_sum / n_paths as f64 * (-r * t).exp();

    assert!(
      berm_price >= eu * 0.99,
      "Bermudan ({berm_price}) should be ~>= European ({eu})"
    );
    assert!(berm_price > 0.0 && berm_price < 50.0);
  }

  /// Bermudan price is monotone (modulo MC noise) in the number of
  /// exercise dates.
  #[test]
  fn bermudan_price_monotone_in_exercise_count() {
    let s0 = 100.0;
    let k = 100.0;
    let r = 0.05;
    let sigma = 0.30;
    let t = 1.0;
    let n_paths = 20_000;
    let n_steps = 60;

    let paths = generate_gbm_paths(s0, r, 0.0, sigma, t, n_paths, n_steps);
    let bermudan = BermudanLsmPricer::new(r, t, 4);
    let payoff = |s: f64| (k - s).max(0.0);

    let mut prev = f64::NEG_INFINITY;
    for n_exercise in [2usize, 4, 12, 30, 60] {
      let stride = n_steps / n_exercise;
      let exercise: Vec<usize> = (1..=n_exercise).map(|i| i * stride).collect();
      let p = bermudan.price(&paths, &exercise, payoff);
      assert!(
        p >= prev - 0.05,
        "n_exercise={n_exercise} gives {p}, prev={prev}"
      );
      prev = p;
    }
  }

  /// Bermudan put with a single exercise = European put (terminal only).
  #[test]
  fn bermudan_single_exercise_equals_european() {
    let s0 = 100.0;
    let k = 105.0;
    let r = 0.05;
    let sigma = 0.20;
    let t = 1.0;
    let n_paths = 50_000;
    let n_steps = 30;

    let paths = generate_gbm_paths(s0, r, 0.0, sigma, t, n_paths, n_steps);
    let bermudan = BermudanLsmPricer::new(r, t, 4);
    let payoff = |s: f64| (k - s).max(0.0);
    let berm = bermudan.price(&paths, &[n_steps], payoff);

    let mut eu_sum = 0.0;
    for i in 0..n_paths {
      eu_sum += payoff(paths[[i, n_steps]]);
    }
    let eu = eu_sum / n_paths as f64 * (-r * t).exp();
    let diff = (berm - eu).abs();
    assert!(diff < 1e-9, "berm={berm}, eu={eu}");
  }
}

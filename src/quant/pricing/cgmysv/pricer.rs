//! CGMYSV Monte Carlo pricers for European, American, Asian, and Barrier options.
//!
//! - **European**: discounted terminal payoff (Section 4.1)
//! - **American**: Longstaff–Schwartz LSM with basis $\{1,S,S^2,\sigma,\sigma^2,\sigma S\}$ (Section 4.2)
//! - **Asian**: arithmetic average payoff (Section 4.3)
//! - **Barrier**: knock-in / knock-out with discrete monitoring (Section 4.3)
//!
//! Reference: Kim, Y. S. (2021), arXiv:2101.11001, Section 4.

use std::fmt;

use ndarray::Array2;

use super::model::CgmysvParams;
use super::path_gen::CgmysvPathGen;
use crate::quant::pricing::barrier::BarrierType;
use crate::quant::OptionType;

/// Monte Carlo pricing result with standard error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct McResult {
  pub price: f64,
  pub std_error: f64,
}

impl fmt::Display for McResult {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{:.4} ± {:.4}", self.price, self.std_error)
  }
}

/// Monte Carlo pricer for the CGMYSV model.
#[derive(Debug, Clone)]
pub struct CgmysvPricer {
  /// CGMYSV model parameters.
  pub params: CgmysvParams,
  /// Spot price $S_0$.
  pub s: f64,
  /// Risk-free rate $r$.
  pub r: f64,
  /// Continuous dividend yield $q$.
  pub q: f64,
  /// Number of Monte Carlo paths ($N$).
  pub n_paths: usize,
  /// Number of time steps per path ($M$).
  pub n_steps: usize,
  /// Truncation terms for the jump series ($J$).
  pub n_jumps: usize,
}

impl CgmysvPricer {
  /// Generate risk-neutral price paths $S_{n,m}$ and variance paths $v_{n,m}$.
  ///
  /// $S_{n,m} = S_0\exp\!\bigl((r-q)\,t_m + L_{n,m} - \omega(t_m)\bigr)$
  fn generate_price_paths(&self, tau: f64) -> (Array2<f64>, Array2<f64>) {
    let path_gen = CgmysvPathGen {
      params: self.params.clone(),
      n_steps: self.n_steps,
      n_jumps: self.n_jumps,
      t: tau,
    };

    let (l_paths, v_paths) = path_gen.generate(self.n_paths);
    let dt = tau / self.n_steps as f64;

    let mut omega = vec![0.0; self.n_steps + 1];
    for m in 1..=self.n_steps {
      omega[m] = self.params.omega(m as f64 * dt);
    }

    let mut s_paths = Array2::zeros((self.n_paths, self.n_steps + 1));
    for i in 0..self.n_paths {
      s_paths[[i, 0]] = self.s;
      for m in 1..=self.n_steps {
        let t_m = m as f64 * dt;
        s_paths[[i, m]] =
          self.s * ((self.r - self.q) * t_m + l_paths[[i, m]] - omega[m]).exp();
      }
    }

    (s_paths, v_paths)
  }

  /// Price a European option via Monte Carlo (Section 4.1).
  pub fn price_european(&self, k: f64, tau: f64, option_type: OptionType) -> McResult {
    let (s_paths, _) = self.generate_price_paths(tau);
    let discount = (-self.r * tau).exp();

    let payoffs: Vec<f64> = (0..self.n_paths)
      .map(|i| {
        let s_t = s_paths[[i, self.n_steps]];
        match option_type {
          OptionType::Call => (s_t - k).max(0.0),
          OptionType::Put => (k - s_t).max(0.0),
        }
      })
      .collect();

    mc_stats(&payoffs, discount)
  }

  /// Price an American option via LSM (Section 4.2).
  ///
  /// Uses the regression basis $\{1,\,S_t,\,S_t^2,\,\sigma_t,\,\sigma_t^2,\,\sigma_t S_t\}$
  /// following Rachev et al. (2011), Chapter 15.
  pub fn price_american(&self, k: f64, tau: f64, option_type: OptionType) -> McResult {
    let (s_paths, v_paths) = self.generate_price_paths(tau);
    let dt = tau / self.n_steps as f64;

    let payoff = |s: f64| match option_type {
      OptionType::Call => (s - k).max(0.0),
      OptionType::Put => (k - s).max(0.0),
    };

    let n = self.n_paths;
    let mut cf = vec![0.0_f64; n];
    let mut cf_time = vec![self.n_steps; n];

    for i in 0..n {
      cf[i] = payoff(s_paths[[i, self.n_steps]]);
    }

    // Backward induction with least-squares regression
    let n_basis = 6;
    for step in (1..self.n_steps).rev() {
      let itm: Vec<usize> = (0..n)
        .filter(|&i| payoff(s_paths[[i, step]]) > 0.0)
        .collect();

      if itm.len() <= n_basis + 1 {
        continue;
      }

      let n_itm = itm.len();

      // Discounted continuation values
      let y_vec: Vec<f64> = itm
        .iter()
        .map(|&i| {
          let steps_fwd = cf_time[i] - step;
          cf[i] * (-self.r * dt * steps_fwd as f64).exp()
        })
        .collect();

      // Design matrix [1, S, S², σ, σ², σ·S]
      let mut x_data = vec![0.0_f64; n_itm * n_basis];
      for (row, &idx) in itm.iter().enumerate() {
        let s_val = s_paths[[idx, step]];
        let sigma = v_paths[[idx, step]].max(0.0).sqrt();
        let base = row * n_basis;
        x_data[base] = 1.0;
        x_data[base + 1] = s_val;
        x_data[base + 2] = s_val * s_val;
        x_data[base + 3] = sigma;
        x_data[base + 4] = sigma * sigma;
        x_data[base + 5] = sigma * s_val;
      }

      let x_mat =
        nalgebra::DMatrix::from_row_slice(n_itm, n_basis, &x_data);
      let y_nal = nalgebra::DVector::from_vec(y_vec);

      let beta = match x_mat.clone().svd(true, true).solve(&y_nal, 1e-10) {
        Ok(b) => b,
        Err(_) => continue,
      };

      for (row, &idx) in itm.iter().enumerate() {
        let continuation: f64 = (0..n_basis)
          .map(|j| x_mat[(row, j)] * beta[j])
          .sum();
        let exercise_val = payoff(s_paths[[idx, step]]);
        if exercise_val > continuation {
          cf[idx] = exercise_val;
          cf_time[idx] = step;
        }
      }
    }

    let disc = (-self.r * dt).exp();
    let payoffs: Vec<f64> = (0..n)
      .map(|i| cf[i] * disc.powi(cf_time[i] as i32))
      .collect();

    let np = n as f64;
    let mean = payoffs.iter().sum::<f64>() / np;
    let variance = payoffs.iter().map(|&p| (p - mean).powi(2)).sum::<f64>() / (np - 1.0);

    McResult {
      price: mean,
      std_error: (variance / np).sqrt(),
    }
  }

  /// Price an Asian option (arithmetic average) via Monte Carlo (Section 4.3).
  pub fn price_asian(&self, k: f64, tau: f64, option_type: OptionType) -> McResult {
    let (s_paths, _) = self.generate_price_paths(tau);
    let discount = (-self.r * tau).exp();

    let payoffs: Vec<f64> = (0..self.n_paths)
      .map(|i| {
        let avg: f64 =
          (1..=self.n_steps).map(|m| s_paths[[i, m]]).sum::<f64>() / self.n_steps as f64;
        match option_type {
          OptionType::Call => (avg - k).max(0.0),
          OptionType::Put => (k - avg).max(0.0),
        }
      })
      .collect();

    mc_stats(&payoffs, discount)
  }

  /// Price a barrier option via Monte Carlo (Section 4.3).
  pub fn price_barrier(
    &self,
    k: f64,
    tau: f64,
    barrier: f64,
    barrier_type: BarrierType,
    option_type: OptionType,
  ) -> McResult {
    let (s_paths, _) = self.generate_price_paths(tau);
    let discount = (-self.r * tau).exp();

    let payoffs: Vec<f64> = (0..self.n_paths)
      .map(|i| {
        let mut knocked = false;
        for m in 1..=self.n_steps {
          let s_m = s_paths[[i, m]];
          match barrier_type {
            BarrierType::DownAndOut | BarrierType::DownAndIn => {
              if s_m <= barrier {
                knocked = true;
                break;
              }
            }
            BarrierType::UpAndOut | BarrierType::UpAndIn => {
              if s_m >= barrier {
                knocked = true;
                break;
              }
            }
          }
        }

        let alive = match barrier_type {
          BarrierType::DownAndOut | BarrierType::UpAndOut => !knocked,
          BarrierType::DownAndIn | BarrierType::UpAndIn => knocked,
        };

        if alive {
          let s_t = s_paths[[i, self.n_steps]];
          match option_type {
            OptionType::Call => (s_t - k).max(0.0),
            OptionType::Put => (k - s_t).max(0.0),
          }
        } else {
          0.0
        }
      })
      .collect();

    mc_stats(&payoffs, discount)
  }
}

/// Compute MC price estimate and standard error from discounted payoffs.
fn mc_stats(payoffs: &[f64], discount: f64) -> McResult {
  let n = payoffs.len() as f64;
  let mean = payoffs.iter().sum::<f64>() / n;
  let variance = payoffs.iter().map(|&p| (p - mean).powi(2)).sum::<f64>() / (n - 1.0);
  McResult {
    price: discount * mean,
    std_error: discount * (variance / n).sqrt(),
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn paper_params() -> CgmysvParams {
    // Table 2, call calibration — Kim (2021)
    CgmysvParams {
      alpha: 0.5184,
      lambda_plus: 25.4592,
      lambda_minus: 4.6040,
      kappa: 1.0029,
      eta: 0.0711,
      zeta: 0.3443,
      rho: -2.0283,
      v0: 0.006381,
    }
  }

  fn paper_pricer(n_paths: usize) -> CgmysvPricer {
    CgmysvPricer {
      params: paper_params(),
      s: 2488.11,
      r: 0.01213,
      q: 0.01884,
      n_paths,
      n_steps: 100,
      n_jumps: 1024,
    }
  }

  #[test]
  fn european_call_in_range() {
    // Table 4: FFT call = 19.6590, MCS(10000) = 19.6840
    let pricer = paper_pricer(5000);
    let tau = 28.0 / 365.0;
    let result = pricer.price_european(2500.0, tau, OptionType::Call);
    assert!(
      result.price > 5.0 && result.price < 50.0,
      "European call price {result} out of range"
    );
  }

  #[test]
  fn european_put_in_range() {
    // Table 4: FFT put = 32.9541, MCS(10000) = 32.6914
    let pricer = paper_pricer(5000);
    let tau = 28.0 / 365.0;
    let result = pricer.price_european(2500.0, tau, OptionType::Put);
    assert!(
      result.price > 10.0 && result.price < 80.0,
      "European put price {result} out of range"
    );
  }

  #[test]
  fn asian_call_positive() {
    // Table 8: MCS(10000) call = 21.6513
    let pricer = paper_pricer(2000);
    let tau = 25.0 / 365.0;
    let result = pricer.price_asian(2500.0, tau, OptionType::Call);
    assert!(result.price > 0.0, "Asian call price {result} should be positive");
  }

  #[test]
  fn barrier_knockout_leq_european() {
    let pricer = paper_pricer(2000);
    let tau = 25.0 / 365.0;
    let eu = pricer.price_european(2500.0, tau, OptionType::Call);
    let barrier = pricer.price_barrier(
      2500.0,
      tau,
      2400.0,
      BarrierType::DownAndOut,
      OptionType::Call,
    );
    // Down-and-out call ≤ vanilla European call (allow MC noise)
    assert!(
      barrier.price <= eu.price + 5.0 * (eu.std_error + barrier.std_error),
      "Barrier {barrier} should be ≤ European {eu}"
    );
  }
}

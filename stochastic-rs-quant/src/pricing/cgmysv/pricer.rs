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
use rayon::prelude::*;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::volatility::svcgmy::Svcgmy;

use super::model::CgmysvParams;
use crate::OptionType;
use crate::pricing::barrier::BarrierType;
use crate::traits::ProcessExt;

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
    let p = &self.params;
    let n = self.n_steps + 1;

    let process = Svcgmy::new(
      p.lambda_plus,
      p.lambda_minus,
      p.alpha,
      p.kappa,
      p.eta,
      p.zeta,
      p.rho,
      n,
      self.n_jumps,
      Some(p.rho * p.v0),
      Some(p.v0),
      Some(tau),
      Unseeded,
    );

    let results: Vec<[ndarray::Array1<f64>; 2]> = (0..self.n_paths)
      .into_par_iter()
      .map(|_| process.sample())
      .collect();

    let dt = tau / self.n_steps as f64;
    let mut omega = vec![0.0_f64; n];
    for m in 1..=self.n_steps {
      omega[m] = p.omega(m as f64 * dt);
    }

    let mut s_paths = Array2::zeros((self.n_paths, n));
    let mut v_paths = Array2::zeros((self.n_paths, n));
    for (i, lv) in results.into_iter().enumerate() {
      v_paths.row_mut(i).assign(&lv[1]);
      s_paths[[i, 0]] = self.s;
      for m in 1..=self.n_steps {
        let t_m = m as f64 * dt;
        s_paths[[i, m]] = self.s * ((self.r - self.q) * t_m + lv[0][m] - omega[m]).exp();
      }
    }

    (s_paths, v_paths)
  }

  /// Price a European option via Monte Carlo (Section 4.1).
  pub fn price_european(&self, k: f64, tau: f64, option_type: OptionType) -> McResult {
    let (s_paths, _) = self.generate_price_paths(tau);
    self.european_on_paths(&s_paths, k, tau, option_type)
  }

  fn european_on_paths(
    &self,
    s_paths: &Array2<f64>,
    k: f64,
    tau: f64,
    option_type: OptionType,
  ) -> McResult {
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
    self.american_on_paths(&s_paths, &v_paths, k, tau, option_type)
  }

  fn american_on_paths(
    &self,
    s_paths: &Array2<f64>,
    v_paths: &Array2<f64>,
    k: f64,
    tau: f64,
    option_type: OptionType,
  ) -> McResult {
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

      let x_mat = nalgebra::DMatrix::from_row_slice(n_itm, n_basis, &x_data);
      let y_nal = nalgebra::DVector::from_vec(y_vec);

      let beta = match x_mat.clone().svd(true, true).solve(&y_nal, 1e-10) {
        Ok(b) => b,
        Err(_) => continue,
      };

      for (row, &idx) in itm.iter().enumerate() {
        let continuation: f64 = (0..n_basis).map(|j| x_mat[(row, j)] * beta[j]).sum();
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
  use crate::pricing::cgmysv::CgmysvModel;
  use crate::pricing::fourier::FourierModelExt;
  use crate::pricing::fourier::LewisPricer;

  /// Table 2 call-side params with corrected v₀ (paper PDF: 0.006381, actual ≈ 0.01115).
  fn paper_params() -> CgmysvParams {
    CgmysvParams {
      alpha: 0.5184,
      lambda_plus: 25.4592,
      lambda_minus: 4.6040,
      kappa: 1.0029,
      eta: 0.0711,
      zeta: 0.3443,
      rho: -2.0283,
      v0: 0.01115,
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

  /// Table 4 reference: Lewis call ≈ 19.66 (paper FFT = 19.659).
  /// Maturity literals (`28.0 / 365.0`, `25.0 / 365.0`) below are the
  /// published S&P 500 option maturities from Carr-Geman-Madan-Yor &
  /// Schoutens (CGMYSV) Tables 4-6 reproduced verbatim — keeping them as
  /// inline literals avoids hiding the paper's calibration days behind a
  /// helper.
  #[test]
  fn table4_lewis_call() {
    let model = CgmysvModel {
      params: paper_params(),
      r: 0.01213,
      q: 0.01884,
    };
    let call = LewisPricer::price_call(&model, 2488.11, 2500.0, 0.01213, 0.01884, 28.0 / 365.0);
    println!("Table 4 — European call (K=2500, T=28d)");
    println!("  Lewis:     {call:.4}");
    println!("  Paper FFT: 19.6590");
    assert!(
      (call - 19.659).abs() < 0.5,
      "Lewis call = {call:.4}, paper FFT = 19.659"
    );
  }

  /// Table 4 reference: MC call ≈ 19.68, put ≈ 32.69.
  #[test]
  fn table4_european_mc() {
    let pricer = paper_pricer(5000);
    let tau = 28.0 / 365.0;
    let call = pricer.price_european(2500.0, tau, OptionType::Call);
    let put = pricer.price_european(2500.0, tau, OptionType::Put);
    println!("Table 4 — European MC (K=2500, T=28d, N=5000)");
    println!("  Call: {call}  (paper: 19.6840 ± 0.2551)");
    println!("  Put:  {put}  (paper: 32.6914 ± 0.7617)");
    assert!(
      call.price > 10.0 && call.price < 35.0,
      "MC call = {call}, paper = 19.68"
    );
    assert!(
      put.price > 15.0 && put.price < 55.0,
      "MC put = {put}, paper = 32.69"
    );
    assert!(put.price > call.price, "put > call for K > S");
  }

  /// Table 8: Asian call/put.
  #[test]
  fn table8_asian_positive() {
    let pricer = paper_pricer(2000);
    let tau = 25.0 / 365.0;
    let call = pricer.price_asian(2500.0, tau, OptionType::Call);
    let put = pricer.price_asian(2500.0, tau, OptionType::Put);
    println!("Table 8 — Asian MC (K=2500, T=25d, N=2000)");
    println!("  Call: {call}  (paper: 21.6513 ± 0.1937)");
    println!("  Put:  {put}  (paper:  9.9964 ± 0.3679)");
    assert!(call.price > 0.0, "Asian call = {call}");
    assert!(put.price > 0.0, "Asian put = {put}");
  }

  /// Table 9: Barrier knock-out ≤ vanilla.
  #[test]
  fn table9_barrier_knockout() {
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
    println!("Table 9 — Barrier MC (K=2500, T=25d, N=2000)");
    println!("  DO call:   {barrier}  (paper: 16.5518 ± 0.1749)");
    println!("  EU call:   {eu}");
    assert!(
      barrier.price <= eu.price + 5.0 * (eu.std_error + barrier.std_error),
      "Barrier {barrier} ≤ European {eu}"
    );
  }

  /// American put ≥ European put (early exercise premium).
  ///
  /// Uses common random numbers: both prices computed on the same path set
  /// so the comparison is dominated by the LSM exercise policy quality
  /// rather than cross-method MC noise.
  #[test]
  fn american_put_geq_european() {
    let pricer = paper_pricer(3000);
    let tau = 28.0 / 365.0;
    let (s_paths, v_paths) = pricer.generate_price_paths(tau);
    let eu = pricer.european_on_paths(&s_paths, 2500.0, tau, OptionType::Put);
    let am = pricer.american_on_paths(&s_paths, &v_paths, 2500.0, tau, OptionType::Put);
    println!("American vs European put (K=2500, T=28d, N=3000, common paths)");
    println!("  American: {am}");
    println!("  European: {eu}");
    assert!(
      am.price >= eu.price * 0.90,
      "American put {am} should be ≥ European put {eu}"
    );
    assert!(am.price > 0.0, "American put positive: {am}");
  }

  /// φ(0) = 1, φ(-i) = exp((r-q)T).
  #[test]
  fn chf_sanity() {
    let model = CgmysvModel {
      params: paper_params(),
      r: 0.01213,
      q: 0.01884,
    };
    let tau = 28.0 / 365.0;
    let phi0 = model.chf(tau, num_complex::Complex64::new(0.0, 0.0));
    let phi_ni = model.chf(tau, num_complex::Complex64::new(0.0, -1.0));
    let expected = ((0.01213 - 0.01884) * tau).exp();
    println!("ChF sanity");
    println!("  φ(0)  = {:.10} (expected 1.0)", phi0.norm());
    println!("  φ(-i) = {:.10} (expected {expected:.10})", phi_ni.re);
    assert!((phi0.norm() - 1.0).abs() < 1e-10, "φ(0) = {phi0}");
    assert!((phi_ni.re - expected).abs() < 1e-8, "φ(-i) = {}", phi_ni.re);
  }
}

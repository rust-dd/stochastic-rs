use std::cell::RefCell;

use impl_new_derive::ImplNew;
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{DMatrix, DVector, Dyn, Owned};
use ndarray::Array1;

use crate::{
  quant::{
    calibration::CalibrationHistory,
    pricing::heston::HestonPricer,
    r#trait::{CalibrationLossExt, PricerExt},
    CalibrationLossScore, OptionType,
  },
  stats::mle::nmle_heston,
};

#[derive(Clone, Debug)]
pub struct HestonParams {
  /// Initial variance v0 (not volatility) in Heston model
  pub v0: f64,
  /// Mean reversion speed
  pub kappa: f64,
  /// Long-run variance
  pub theta: f64,
  /// Volatility of variance
  pub sigma: f64,
  /// Correlation between price and variance Brownian motions
  pub rho: f64,
}

impl From<HestonParams> for DVector<f64> {
  fn from(params: HestonParams) -> Self {
    DVector::from_vec(vec![
      params.v0,
      params.kappa,
      params.theta,
      params.sigma,
      params.rho,
    ])
  }
}

impl From<DVector<f64>> for HestonParams {
  fn from(params: DVector<f64>) -> Self {
    HestonParams {
      v0: params[0],
      kappa: params[1],
      theta: params[2],
      sigma: params[3],
      rho: params[4],
    }
  }
}

/// A Heston model calibrator (price-based) using Levenberg-Marquardt.
#[derive(ImplNew, Clone)]
pub struct HestonCalibrator {
  /// Params to calibrate (v0, kappa, theta, sigma, rho).
  /// If None, an initial guess will be inferred using heston_mle (requires mle_* fields).
  pub params: Option<HestonParams>,
  /// Option prices from the market.
  pub c_market: DVector<f64>,
  /// Underlying spot per quote (allows small variations per strike/maturity bucket).
  pub s: DVector<f64>,
  /// Strikes per quote.
  pub k: DVector<f64>,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield.
  pub q: Option<f64>,
  /// Time to maturity (years) used for all quotes in this calibrator.
  pub tau: f64,
  /// Option type of the quotes.
  pub option_type: OptionType,
  /// Optional: time series for MLE-based initial guess
  pub mle_s: Option<Array1<f64>>, // stock prices time series
  pub mle_v: Option<Array1<f64>>, // variance (or instantaneous variance proxy) time series
  pub mle_r: Option<f64>,         // risk-free rate used for MLE
  /// History of iterations (residuals, params, loss metrics).
  calibration_history: RefCell<Vec<CalibrationHistory<HestonParams>>>,
}

impl CalibrationLossExt for HestonCalibrator {}

impl HestonCalibrator {
  pub fn calibrate(&self) {
    // Prepare a problem clone with an initial guess if needed
    let mut problem = self.clone();
    problem.ensure_initial_guess();

    println!("Initial guess: {:?}", problem.params);

    let (result, ..) = LevenbergMarquardt::new().minimize(problem);

    // Print the c_market
    println!("Market prices: {:?}", self.c_market);

    let residuals = result.residuals().unwrap();

    // Print the c_model (residuals = market - model, so model = market - residuals)
    println!("Model prices: {:?}", self.c_market.clone() - residuals);

    // Print the result of the calibration
    println!("Calibration report: {:?}", result.params);
  }

  pub fn set_initial_guess(&mut self, params: HestonParams) {
    self.params = Some(params);
  }

  fn ensure_initial_guess(&mut self) {
    if self.params.is_none() {
      if let (Some(s), Some(v), Some(r)) = (self.mle_s.clone(), self.mle_v.clone(), self.mle_r) {
        let mut p = nmle_heston(s, v, r);
        // Clamp to reasonable bounds
        p.v0 = p.v0.max(1e-8);
        p.kappa = p.kappa.max(1e-8);
        p.theta = p.theta.max(1e-8);
        p.sigma = p.sigma.abs().max(1e-8);
        p.rho = p.rho.max(-0.9999).min(0.9999);
        self.params = Some(p);
      } else {
        // Fallback conservative guess
        self.params = Some(HestonParams {
          v0: 0.04,
          kappa: 1.5,
          theta: 0.04,
          sigma: 0.5,
          rho: -0.5,
        });
      }
    }
  }

  fn effective_params(&self) -> HestonParams {
    if let Some(p) = &self.params {
      return p.clone();
    }
    if let (Some(s), Some(v), Some(r)) = (self.mle_s.clone(), self.mle_v.clone(), self.mle_r) {
      let mut p = nmle_heston(s, v, r);
      p.v0 = p.v0.max(1e-8);
      p.kappa = p.kappa.max(1e-8);
      p.theta = p.theta.max(1e-8);
      p.sigma = p.sigma.abs().max(1e-8);
      p.rho = p.rho.max(-0.9999).min(0.9999);
      return p;
    }
    HestonParams {
      v0: 0.04,
      kappa: 1.5,
      theta: 0.04,
      sigma: 0.5,
      rho: -0.5,
    }
  }

  fn compute_model_prices_for(&self, params: &HestonParams) -> DVector<f64> {
    let mut c_model = DVector::zeros(self.c_market.len());

    for (idx, _) in self.c_market.iter().enumerate() {
      let pricer = HestonPricer::new(
        self.s[idx],
        params.v0,
        self.k[idx],
        self.r,
        self.q,
        params.rho,
        params.kappa,
        params.theta,
        params.sigma,
        Some(0.0), // lambda (market price of vol risk), set to 0 in most calibrations
        Some(self.tau),
        None,
        None,
      );
      let (call, put) = pricer.calculate_call_put();

      match self.option_type {
        OptionType::Call => c_model[idx] = call.max(0.0),
        OptionType::Put => c_model[idx] = put.max(0.0),
      }
    }

    c_model
  }

  fn residuals_for(&self, params: &HestonParams) -> DVector<f64> {
    self.c_market.clone() - self.compute_model_prices_for(params)
  }

  /// Numerically approximate the Jacobian via central differences.
  fn numeric_jacobian(&self, params: &HestonParams) -> DMatrix<f64> {
    let n = self.c_market.len();
    let p = 5usize; // v0, kappa, theta, sigma, rho

    let base_params_vec: DVector<f64> = params.clone().into();
    let mut J = DMatrix::zeros(n, p);

    for col in 0..p {
      let x = base_params_vec[col];
      let mut h = 1e-5_f64.max(1e-3 * x.abs());

      let mut params_plus = params.clone();
      let mut params_minus = params.clone();

      match col {
        0 => {
          // v0 >= 0
          params_plus.v0 = (x + h).max(1e-8);
          params_minus.v0 = (x - h).max(1e-8);
        }
        1 => {
          // kappa > 0
          params_plus.kappa = (x + h).max(1e-8);
          params_minus.kappa = (x - h).max(1e-8);
        }
        2 => {
          // theta > 0
          params_plus.theta = (x + h).max(1e-8);
          params_minus.theta = (x - h).max(1e-8);
        }
        3 => {
          // sigma >= 0
          params_plus.sigma = (x + h).max(1e-8);
          params_minus.sigma = (x - h).max(1e-8);
        }
        4 => {
          // -1 < rho < 1
          let clamp = |y: f64| y.max(-0.9999).min(0.9999);
          params_plus.rho = clamp(x + h);
          params_minus.rho = clamp(x - h);
          // Use symmetric step if clamped too hard
          if (params_plus.rho - params_minus.rho).abs() < 0.5 * h {
            h = 1e-4;
            params_plus.rho = clamp(x + h);
            params_minus.rho = clamp(x - h);
          }
        }
        _ => unreachable!(),
      }

      // Optional: enforce (soft) Feller condition in probes by nudging theta
      // 2*kappa*theta > sigma^2
      let enforce_feller = |p: &mut HestonParams| {
        if 2.0 * p.kappa * p.theta <= p.sigma * p.sigma {
          p.theta = (p.sigma * p.sigma) / (2.0 * p.kappa) + 1e-8;
        }
      };
      enforce_feller(&mut params_plus);
      enforce_feller(&mut params_minus);

      let r_plus = self.residuals_for(&params_plus);
      let r_minus = self.residuals_for(&params_minus);

      let diff = (r_plus - r_minus) / (2.0 * h);
      for row in 0..n {
        J[(row, col)] = diff[row];
      }
    }

    J
  }
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for HestonCalibrator {
  type JacobianStorage = Owned<f64, Dyn, Dyn>;
  type ParameterStorage = Owned<f64, Dyn>;
  type ResidualStorage = Owned<f64, Dyn>;

  fn set_params(&mut self, params: &DVector<f64>) {
    self.params = Some(HestonParams::from(params.clone()));
  }

  fn params(&self) -> DVector<f64> {
    if let Some(p) = &self.params {
      p.clone().into()
    } else if let (Some(s), Some(v), Some(r)) = (self.mle_s.clone(), self.mle_v.clone(), self.mle_r)
    {
      let p = nmle_heston(s, v, r);
      p.into()
    } else {
      HestonParams {
        v0: 0.04,
        kappa: 1.5,
        theta: 0.04,
        sigma: 0.5,
        rho: -0.5,
      }
      .into()
    }
  }

  fn residuals(&self) -> Option<DVector<f64>> {
    let params_eff = self.effective_params();
    let c_model = self.compute_model_prices_for(&params_eff);

    // Push history for the current iterate
    self
      .calibration_history
      .borrow_mut()
      .push(CalibrationHistory {
        residuals: self.c_market.clone() - c_model.clone(),
        call_put: self
          .c_market
          .iter()
          .enumerate()
          .map(|(i, _)| {
            let pricer = HestonPricer::new(
              self.s[i],
              params_eff.v0,
              self.k[i],
              self.r,
              self.q,
              params_eff.rho,
              params_eff.kappa,
              params_eff.theta,
              params_eff.sigma,
              Some(0.0),
              Some(self.tau),
              None,
              None,
            );
            pricer.calculate_call_put()
          })
          .collect::<Vec<(f64, f64)>>()
          .into(),
        params: params_eff.clone(),
        loss_scores: CalibrationLossScore {
          mae: self.mae(&self.c_market, &c_model),
          mse: self.mse(&self.c_market, &c_model),
          rmse: self.rmse(&self.c_market, &c_model),
          mpe: self.mpe(&self.c_market, &c_model),
          mape: self.mape(&self.c_market, &c_model),
          mspe: self.mspe(&self.c_market, &c_model),
          rmspe: self.rmspe(&self.c_market, &c_model),
          mre: self.mre(&self.c_market, &c_model),
          mrpe: self.mrpe(&self.c_market, &c_model),
        },
      });

    Some(self.c_market.clone() - c_model)
  }

  fn jacobian(&self) -> Option<DMatrix<f64>> {
    // Use our own numeric Jacobian to keep the solver stable
    let p = self.effective_params();
    Some(self.numeric_jacobian(&p))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::stochastic::noise::cgns::CGNS;
  use crate::stochastic::volatility::heston::Heston as HestonProcess;
  use crate::stochastic::volatility::HestonPow;
  use crate::stochastic::Sampling2DExt;
  use ndarray::Array1;

  #[test]
  fn test_heston_calibrate() {
    // Example dataset across strikes for a single maturity bucket.
    let s = vec![
      100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
    ];

    let k = vec![
      80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0,
    ];

    // Hypothetical market call prices (monotone across strikes)
    let c_market = vec![
      21.5, 17.9, 14.2, 11.0, 8.2, 6.0, 4.3, 3.1, 2.2, 1.6, 1.2, 0.9,
    ];

    let r = 0.01;
    let q = Some(0.0);
    let tau = 0.5;
    let option_type = OptionType::Call;

    let calibrator = HestonCalibrator::new(
      Some(HestonParams {
        v0: 0.04,
        kappa: 1.5,
        theta: 0.04,
        sigma: 0.5,
        rho: -0.7,
      }),
      c_market.clone().into(),
      s.clone().into(),
      k.clone().into(),
      r,
      q,
      tau,
      option_type,
      None,
      None,
      None,
    );

    calibrator.calibrate();
  }

  #[test]
  fn test_heston_calibrate_with_mle_seed() {
    // Simulate a short Heston path to seed MLE
    let s0 = 100.0;
    let v0 = 0.04;
    let true_params = HestonParams {
      v0,
      kappa: 2.0,
      theta: 0.04,
      sigma: 0.5,
      rho: -0.6,
    };
    let n = 256usize;
    let t = 1.0;
    let mu = 0.0; // drift not needed for MLE besides r in formula

    let process = HestonProcess::new(
      Some(s0),
      Some(v0),
      true_params.kappa,
      true_params.theta,
      true_params.sigma,
      true_params.rho,
      mu,
      n,
      Some(t),
      HestonPow::Sqrt,
      Some(true),
      None,
      CGNS::new(true_params.rho, n, None, None),
    );

    let [s_ts, v_ts] = process.sample();

    // Build synthetic market prices from true parameters
    let strikes = vec![80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0];
    let s_grid = vec![s0; strikes.len()];
    let r = 0.01;
    let q = Some(0.0);
    let tau = 0.5;

    let mut c_market = Vec::with_capacity(strikes.len());
    for &kk in &strikes {
      let pr = HestonPricer::new(
        s0,
        true_params.v0,
        kk,
        r,
        q,
        true_params.rho,
        true_params.kappa,
        true_params.theta,
        true_params.sigma,
        Some(0.0),
        Some(tau),
        None,
        None,
      );
      let (call, _) = pr.calculate_call_put();
      c_market.push(call);
    }

    let calibrator = HestonCalibrator::new(
      None,
      c_market.clone().into(),
      s_grid.clone().into(),
      strikes.clone().into(),
      r,
      q,
      tau,
      OptionType::Call,
      Some(s_ts),
      Some(v_ts),
      Some(r),
    );

    calibrator.calibrate();
  }

  #[test]
  fn test_heston_calibrate_ls_dataset() {
    let r = 0.04;
    let q = Some(0.06);
    let tau = 0.083;

    let strikes_ls: Vec<f64> = vec![
      5220.318, 6090.371, 6960.424, 7830.477, 8265.5035, 8483.01675, 8700.53, 8918.04325,
      9135.5565, 9570.583, 10440.636, 11310.689,
    ];
    let spots_ls: Vec<f64> = vec![
      8700.53, 8700.53, 8700.53, 8700.53, 8700.53, 8700.53, 8700.53, 8700.53, 8700.53, 8700.53,
      8700.53, 8700.53,
    ];
    let vol_ls: Vec<f64> = vec![
      0.3669, 0.3082, 0.2218, 0.1799, 0.1393, 0.1156, 0.1019, 0.0923, 0.0915, 0.1086, 0.1237, 0.136,
    ];
    let markets_ls: Vec<f64> = vec![
      3499.564, 2632.74, 1765.933, 901.846, 479.055, 279.313, 118.848, 28.79, 4.23, 0.143,
      1.799e-5, 1.259e-9,
    ];

    // MLE seed from pseudo time-series built from LS vectors
    let s_ts = Array1::from(spots_ls.clone());
    let v_ts = Array1::from(vol_ls.iter().map(|x| x * x).collect::<Vec<f64>>());

    let calibrator = HestonCalibrator::new(
      None,
      markets_ls.clone().into(),
      spots_ls.clone().into(),
      strikes_ls.clone().into(),
      r,
      q,
      tau,
      OptionType::Call,
      Some(s_ts),
      Some(v_ts),
      Some(r),
    );

    calibrator.calibrate();
  }
}

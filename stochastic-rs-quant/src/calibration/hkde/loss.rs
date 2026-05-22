use std::f64::consts::PI;

use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::DMatrix;
use nalgebra::DVector;
use nalgebra::Dyn;
use nalgebra::Owned;

use super::calibrator::HKDECalibrator;
use super::params::ETA1_MIN;
use super::params::ETA2_MIN;
use super::params::HKDEParams;
use super::params::KAPPA_MIN;
use super::params::P_P_UP;
use super::params::RHO_BOUND;
use super::params::SIGMA_V_MIN;
use super::params::THETA_MIN;
use crate::CalibrationLossScore;
use crate::OptionType;
use crate::calibration::CalibrationHistory;
use crate::pricing::bsm::BSMCoc;
use crate::pricing::bsm::BSMPricer;
use crate::pricing::fourier::HKDEFourier;
use crate::traits::ModelPricer;
use crate::traits::PricerExt;

const EPS: f64 = 1e-8;

impl HKDECalibrator {
  pub(super) fn build_model(&self, p: &HKDEParams) -> HKDEFourier {
    HKDEFourier {
      v0: p.v0,
      kappa: p.kappa,
      theta: p.theta,
      sigma_v: p.sigma_v,
      rho: p.rho,
      r: self.r,
      q: self.q.unwrap_or(0.0),
      lam: p.lambda,
      p_up: p.p_up,
      eta1: p.eta1,
      eta2: p.eta2,
    }
  }

  pub(super) fn compute_model_prices_for(&self, p: &HKDEParams) -> DVector<f64> {
    let n = self.c_market.len();
    let model = self.build_model(p);
    let r = self.r;
    let q = self.q.unwrap_or(0.0);

    let mut c_model = DVector::zeros(n);
    for idx in 0..n {
      let price = model.price_option(
        self.s[idx],
        self.k[idx],
        r,
        q,
        self.flat_t[idx],
        self.option_type,
      );
      c_model[idx] = price.max(0.0);
    }
    c_model
  }

  pub(super) fn weighted_residuals_for(&self, p: &HKDEParams) -> DVector<f64> {
    let c_model = self.compute_model_prices_for(p);
    let n = self.c_market.len();
    let mut r = DVector::zeros(n);
    for i in 0..n {
      r[i] = self.sqrt_weights[i] * (c_model[i] - self.c_market[i]);
    }
    r
  }

  /// Central finite-difference Jacobian of the weighted residuals.
  pub(super) fn numeric_jacobian(&self, params: &HKDEParams) -> DMatrix<f64> {
    let n = self.c_market.len();
    let p_dim = 9usize;
    let base_vec: DVector<f64> = (*params).into();
    let mut j_mat = DMatrix::zeros(n, p_dim);

    for col in 0..p_dim {
      let x = base_vec[col];
      let h = 1e-5_f64.max(1e-3 * x.abs());

      let mut plus = *params;
      let mut minus = *params;
      match col {
        0 => {
          plus.v0 = (x + h).max(0.0);
          minus.v0 = (x - h).max(0.0);
        }
        1 => {
          plus.kappa = (x + h).max(KAPPA_MIN);
          minus.kappa = (x - h).max(KAPPA_MIN);
        }
        2 => {
          plus.theta = (x + h).max(THETA_MIN);
          minus.theta = (x - h).max(THETA_MIN);
        }
        3 => {
          plus.sigma_v = (x + h).abs().max(SIGMA_V_MIN);
          minus.sigma_v = (x - h).abs().max(SIGMA_V_MIN);
        }
        4 => {
          plus.rho = (x + h).clamp(-RHO_BOUND, RHO_BOUND);
          minus.rho = (x - h).clamp(-RHO_BOUND, RHO_BOUND);
        }
        5 => {
          plus.lambda = (x + h).max(0.0);
          minus.lambda = (x - h).max(0.0);
        }
        6 => {
          plus.p_up = (x + h).clamp(P_P_UP.0, P_P_UP.1);
          minus.p_up = (x - h).clamp(P_P_UP.0, P_P_UP.1);
        }
        7 => {
          plus.eta1 = (x + h).max(ETA1_MIN);
          minus.eta1 = (x - h).max(ETA1_MIN);
        }
        8 => {
          plus.eta2 = (x + h).max(ETA2_MIN);
          minus.eta2 = (x - h).max(ETA2_MIN);
        }
        _ => unreachable!(),
      }
      plus.project_in_place();
      minus.project_in_place();

      let r_plus = self.weighted_residuals_for(&plus);
      let r_minus = self.weighted_residuals_for(&minus);
      let diff = (r_plus - r_minus) / (2.0 * h);
      for row in 0..n {
        j_mat[(row, col)] = diff[row];
      }
    }

    j_mat
  }
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for HKDECalibrator {
  type JacobianStorage = Owned<f64, Dyn, Dyn>;
  type ParameterStorage = Owned<f64, Dyn>;
  type ResidualStorage = Owned<f64, Dyn>;

  fn set_params(&mut self, params: &DVector<f64>) {
    let p = HKDEParams::from(params.clone()).projected();
    self.params = Some(p);
  }

  fn params(&self) -> DVector<f64> {
    self.effective_params().into()
  }

  fn residuals(&self) -> Option<DVector<f64>> {
    let params_eff = self.effective_params();
    let c_model = self.compute_model_prices_for(&params_eff);

    if self.record_history {
      let call_put: Vec<(f64, f64)> = (0..self.c_market.len())
        .map(|idx| {
          let tau = self.flat_t[idx];
          let s_i = self.s[idx];
          let k_i = self.k[idx];
          let disc_r = (-self.r * tau).exp();
          let disc_q = (-self.q.unwrap_or(0.0) * tau).exp();
          let (call, put) = match self.option_type {
            OptionType::Call => {
              let call = c_model[idx];
              (call, (call - s_i * disc_q + k_i * disc_r).max(0.0))
            }
            OptionType::Put => {
              let put = c_model[idx];
              ((put + s_i * disc_q - k_i * disc_r).max(0.0), put)
            }
          };
          (call, put)
        })
        .collect();

      self
        .calibration_history
        .borrow_mut()
        .push(CalibrationHistory {
          residuals: self.c_market.clone() - c_model.clone(),
          call_put: call_put.into(),
          params: params_eff,
          loss_scores: CalibrationLossScore::compute_selected(
            self.c_market.as_slice(),
            c_model.as_slice(),
            self.loss_metrics,
          ),
        });
    }

    let n = self.c_market.len();
    let mut r = DVector::zeros(n);
    for i in 0..n {
      r[i] = self.sqrt_weights[i] * (c_model[i] - self.c_market[i]);
    }
    Some(r)
  }

  fn jacobian(&self) -> Option<DMatrix<f64>> {
    Some(self.numeric_jacobian(&self.effective_params()))
  }
}

/// Compute $\sqrt{w_j^{(n)}}$ for each market quote (Eq. 13 of the paper).
///
/// The market implied volatility $\sigma_j^{(n)}$ is recovered from the
/// observed price by inverting Black-Scholes; the resulting normalised
/// vega $S_0\,\psi(d_1)\,\sqrt{T}$ is then used as weight.
pub(super) fn compute_sqrt_weights(
  s: &[f64],
  k: &[f64],
  t: &[f64],
  r: f64,
  q: f64,
  prices: &[f64],
  option_type: OptionType,
) -> Vec<f64> {
  let n = prices.len();
  let mut out = Vec::with_capacity(n);
  for i in 0..n {
    let w = market_weight(s[i], k[i], r, q, t[i], prices[i], option_type);
    out.push(w.sqrt());
  }
  out
}

/// Single-quote weight $w_j^{(n)} = (S_0\,\psi(d_1)\,\sqrt{T_n})^{-1}$
/// evaluated at the market implied volatility.
fn market_weight(
  s: f64,
  k: f64,
  r: f64,
  q: f64,
  t: f64,
  market_price: f64,
  option_type: OptionType,
) -> f64 {
  if t <= 0.0 || s <= 0.0 || k <= 0.0 {
    return 1.0;
  }

  let pricer = BSMPricer::new(
    s,
    0.2,
    k,
    r,
    None,
    None,
    Some(q),
    Some(t),
    None,
    None,
    option_type,
    BSMCoc::Bsm1973,
  );
  let sigma_mkt = pricer.implied_volatility(market_price, option_type);

  if !sigma_mkt.is_finite() || sigma_mkt <= EPS {
    return 1.0;
  }

  let sqrt_t = t.sqrt();
  let d1 = ((s / k).ln() + (r - q + 0.5 * sigma_mkt * sigma_mkt) * t) / (sigma_mkt * sqrt_t);
  let phi = (-0.5 * d1 * d1).exp() / (2.0 * PI).sqrt();
  let vega_norm = s * phi * sqrt_t;
  if vega_norm < EPS {
    return 1.0;
  }
  1.0 / vega_norm
}

/// Reference calibrated Hkde parameters from Agazzotti et al. (2025), Table 1.
///
/// All values correspond to single-name equity option surfaces observed on
/// 2024-02-20. They are exposed as constants so downstream users and tests
/// can check their own implementations against the published results.
///
/// Note: several of these sets do not satisfy the Feller condition and lie
/// outside the default box used by [`HKDEParams::project_in_place`] — the
/// paper does not enforce those constraints. When feeding them through the
/// projection the calibrator will adjust them to the admissible set.
pub mod paper_table1 {
  use super::HKDEParams;

  /// AMZN, 2024-02-20 (Agazzotti et al. 2025, Table 1). Feller violated
  /// ($2\kappa\theta \approx 0.707 < \sigma_v^2 \approx 1.608$) and
  /// $\lambda$, $\sigma_v$ sit outside the default projection box.
  pub const AMZN: HKDEParams = HKDEParams {
    v0: 0.023,
    kappa: 5.275,
    theta: 0.067,
    sigma_v: 1.268,
    rho: -0.691,
    lambda: 53.165,
    p_up: 0.999,
    eta1: 49.799,
    eta2: 2.587,
  };

  /// NFLX, 2024-02-20 (Agazzotti et al. 2025, Table 1). Feller violated,
  /// several parameters exceed the default projection box.
  pub const NFLX: HKDEParams = HKDEParams {
    v0: 0.001,
    kappa: 13.355,
    theta: 0.091,
    sigma_v: 4.797,
    rho: -0.498,
    lambda: 103.622,
    p_up: 0.272,
    eta1: 42.945,
    eta2: 65.011,
  };

  /// SHOP, 2024-02-20 (Agazzotti et al. 2025, Table 1). Fully admissible:
  /// Feller holds ($2\kappa\theta \approx 0.278 \ge \sigma_v^2 \approx 0.038$)
  /// and every parameter lies inside the default projection box.
  pub const SHOP: HKDEParams = HKDEParams {
    v0: 0.176,
    kappa: 0.191,
    theta: 0.728,
    sigma_v: 0.194,
    rho: -0.718,
    lambda: 1.009,
    p_up: 0.958,
    eta1: 8.739,
    eta2: 0.733,
  };

  /// SPOT, 2024-02-20 (Agazzotti et al. 2025, Table 1). Feller violated,
  /// $\lambda$ and $\eta_2$ lie outside the default projection box.
  pub const SPOT: HKDEParams = HKDEParams {
    v0: 0.064,
    kappa: 6.796,
    theta: 0.163,
    sigma_v: 1.698,
    rho: -0.391,
    lambda: 17.725,
    p_up: 1.0,
    eta1: 35.555,
    eta2: 0.049,
  };
}

/// Reference calibration error metrics from Agazzotti et al. (2025), Table 2.
///
/// These values correspond to Hkde calibrated against market option data on
/// 2024-02-20 and are reproduced here as a documentation / sanity aid.
/// Reaching them exactly requires the underlying proprietary quotes.
#[allow(dead_code)]
pub mod paper_table2 {
  /// Hkde mean absolute percentage error per ticker, Table 2.
  pub const MAPE: [(&str, f64); 4] = [
    ("AMZN", 0.0261),
    ("NFLX", 0.0488),
    ("SHOP", 0.0266),
    ("SPOT", 0.0339),
  ];
  /// Hkde root-mean-square error per ticker, Table 2.
  pub const RMSE: [(&str, f64); 4] = [
    ("AMZN", 0.01433),
    ("NFLX", 0.10048),
    ("SHOP", 0.02938),
    ("SPOT", 0.03173),
  ];
}

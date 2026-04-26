//! # Double Heston Calibration
//!
//! $$
//! \begin{aligned}
//! dS_t &= (r-q)S_t\,dt + \sqrt{v_{1,t}}\,S_t\,dW_{1,t}^S + \sqrt{v_{2,t}}\,S_t\,dW_{2,t}^S \\
//! dv_{1,t} &= \kappa_1(\theta_1 - v_{1,t})\,dt + \sigma_1\sqrt{v_{1,t}}\,dW_{1,t}^v \\
//! dv_{2,t} &= \kappa_2(\theta_2 - v_{2,t})\,dt + \sigma_2\sqrt{v_{2,t}}\,dW_{2,t}^v
//! \end{aligned}
//! $$
//! with $d\langle W_1^S,W_1^v\rangle_t=\rho_1\,dt$,
//! $d\langle W_2^S,W_2^v\rangle_t=\rho_2\,dt$, and every other Brownian
//! motion pair independent.
//!
//! The characteristic function of $\ln(S_T/S_0)$ factorises into a sum of
//! two Heston contributions:
//! $$
//! \phi_T(u) = \exp\!\left(iu(r-q)T + \sum_{j=1}^2\bigl[C_j(u,T) + D_j(u,T)\,v_{j,0}\bigr]\right)
//! $$
//!
//! Source:
//! - Christoffersen, Heston & Jacobs (2009), "The Shape and Term Structure of
//!   the Index Option Smirk: Why Multifactor Stochastic Volatility Models Work
//!   So Well", Management Science 55(12), 1914-1932,
//!   <https://doi.org/10.1287/mnsc.1090.1065>
//! - Mehrdoust, Noorani & Hamdi (2021), "Calibration of the double Heston
//!   model and an analytical formula in pricing American put option",
//!   J. Comput. Appl. Math. 392, 113422,
//!   <https://doi.org/10.1016/j.cam.2021.113422>
//! - Levenberg (1944), <https://doi.org/10.1090/qam/10666>
//! - Marquardt (1963), <https://doi.org/10.1137/0111030>

use std::cell::RefCell;
use std::f64::consts::FRAC_1_PI;
use std::rc::Rc;

use levenberg_marquardt::LeastSquaresProblem;
use levenberg_marquardt::LevenbergMarquardt;
use nalgebra::DMatrix;
use nalgebra::DVector;
use nalgebra::Dyn;
use nalgebra::Owned;
use num_complex::Complex64;

use super::GL_U_MAX;
use super::gauss_legendre_64;
use super::periodic_map;
use crate::CalibrationLossScore;
use crate::LossMetric;
use crate::OptionType;
use crate::calibration::CalibrationHistory;

const EPS: f64 = 1e-8;
const RHO_BOUND: f64 = 0.9999;
const KAPPA_MIN: f64 = 1e-3;
const THETA_MIN: f64 = 1e-8;
const SIGMA_MIN: f64 = 1e-8;

const P_V0: (f64, f64) = (0.001, 0.25);
const P_KAPPA: (f64, f64) = (0.1, 20.0);
const P_THETA: (f64, f64) = (0.001, 0.4);
const P_SIGMA: (f64, f64) = (0.01, 1.0);
const P_RHO: (f64, f64) = (-1.0, 1.0);

/// Double Heston model parameters.
#[derive(Clone, Copy, Debug)]
pub struct DoubleHestonParams {
  /// Initial variance of factor 1.
  pub v1_0: f64,
  /// Mean-reversion speed of factor 1.
  pub kappa1: f64,
  /// Long-run variance of factor 1.
  pub theta1: f64,
  /// Volatility-of-variance of factor 1.
  pub sigma1: f64,
  /// Spot-variance correlation for factor 1.
  pub rho1: f64,
  /// Initial variance of factor 2.
  pub v2_0: f64,
  /// Mean-reversion speed of factor 2.
  pub kappa2: f64,
  /// Long-run variance of factor 2.
  pub theta2: f64,
  /// Volatility-of-variance of factor 2.
  pub sigma2: f64,
  /// Spot-variance correlation for factor 2.
  pub rho2: f64,
}

impl DoubleHestonParams {
  /// Project parameters to satisfy admissibility: box constraints via periodic
  /// mapping plus per-factor Feller condition $2\kappa_j\theta_j\ge\sigma_j^2$.
  pub fn project_in_place(&mut self) {
    self.v1_0 = periodic_map(self.v1_0, P_V0.0, P_V0.1).max(0.0);
    self.kappa1 = periodic_map(self.kappa1, P_KAPPA.0, P_KAPPA.1).max(KAPPA_MIN);
    self.theta1 = periodic_map(self.theta1, P_THETA.0, P_THETA.1).max(THETA_MIN);
    self.sigma1 = periodic_map(self.sigma1, P_SIGMA.0, P_SIGMA.1)
      .abs()
      .max(SIGMA_MIN);
    self.rho1 = periodic_map(self.rho1, P_RHO.0, P_RHO.1).clamp(-RHO_BOUND, RHO_BOUND);

    self.v2_0 = periodic_map(self.v2_0, P_V0.0, P_V0.1).max(0.0);
    self.kappa2 = periodic_map(self.kappa2, P_KAPPA.0, P_KAPPA.1).max(KAPPA_MIN);
    self.theta2 = periodic_map(self.theta2, P_THETA.0, P_THETA.1).max(THETA_MIN);
    self.sigma2 = periodic_map(self.sigma2, P_SIGMA.0, P_SIGMA.1)
      .abs()
      .max(SIGMA_MIN);
    self.rho2 = periodic_map(self.rho2, P_RHO.0, P_RHO.1).clamp(-RHO_BOUND, RHO_BOUND);

    // Feller, factor 1.
    if 2.0 * self.kappa1 * self.theta1 < self.sigma1 * self.sigma1 {
      let sigma_star = (2.0 * self.kappa1 * self.theta1).sqrt();
      if sigma_star >= P_SIGMA.0 {
        self.sigma1 = sigma_star.min(P_SIGMA.1);
      } else {
        let theta_star = ((self.sigma1 * self.sigma1) / (2.0 * self.kappa1)).max(THETA_MIN) + EPS;
        self.theta1 = theta_star.min(P_THETA.1);
      }
    }

    // Feller, factor 2.
    if 2.0 * self.kappa2 * self.theta2 < self.sigma2 * self.sigma2 {
      let sigma_star = (2.0 * self.kappa2 * self.theta2).sqrt();
      if sigma_star >= P_SIGMA.0 {
        self.sigma2 = sigma_star.min(P_SIGMA.1);
      } else {
        let theta_star = ((self.sigma2 * self.sigma2) / (2.0 * self.kappa2)).max(THETA_MIN) + EPS;
        self.theta2 = theta_star.min(P_THETA.1);
      }
    }
  }

  pub fn projected(mut self) -> Self {
    self.project_in_place();
    self
  }

  /// Convert to a [`DoubleHestonFourier`] model for pricing / vol surface generation.
  pub fn to_model(&self, r: f64, q: f64) -> crate::pricing::fourier::DoubleHestonFourier {
    crate::pricing::fourier::DoubleHestonFourier {
      v1_0: self.v1_0,
      kappa1: self.kappa1,
      theta1: self.theta1,
      sigma1: self.sigma1,
      rho1: self.rho1,
      v2_0: self.v2_0,
      kappa2: self.kappa2,
      theta2: self.theta2,
      sigma2: self.sigma2,
      rho2: self.rho2,
      r,
      q,
    }
  }
}

impl crate::traits::ToModel for DoubleHestonParams {
  fn to_model(&self, r: f64, q: f64) -> Box<dyn crate::traits::ModelPricer> {
    Box::new(DoubleHestonParams::to_model(self, r, q))
  }
}

impl From<DoubleHestonParams> for DVector<f64> {
  fn from(p: DoubleHestonParams) -> Self {
    DVector::from_vec(vec![
      p.v1_0, p.kappa1, p.theta1, p.sigma1, p.rho1, p.v2_0, p.kappa2, p.theta2, p.sigma2, p.rho2,
    ])
  }
}

impl From<DVector<f64>> for DoubleHestonParams {
  fn from(v: DVector<f64>) -> Self {
    DoubleHestonParams {
      v1_0: v[0],
      kappa1: v[1],
      theta1: v[2],
      sigma1: v[3],
      rho1: v[4],
      v2_0: v[5],
      kappa2: v[6],
      theta2: v[7],
      sigma2: v[8],
      rho2: v[9],
    }
  }
}

/// Calibration result for the double Heston model.
#[derive(Clone, Debug)]
pub struct DoubleHestonCalibrationResult {
  pub v1_0: f64,
  pub kappa1: f64,
  pub theta1: f64,
  pub sigma1: f64,
  pub rho1: f64,
  pub v2_0: f64,
  pub kappa2: f64,
  pub theta2: f64,
  pub sigma2: f64,
  pub rho2: f64,
  pub loss: CalibrationLossScore,
  pub converged: bool,
}

impl DoubleHestonCalibrationResult {
  pub fn params(&self) -> DoubleHestonParams {
    DoubleHestonParams {
      v1_0: self.v1_0,
      kappa1: self.kappa1,
      theta1: self.theta1,
      sigma1: self.sigma1,
      rho1: self.rho1,
      v2_0: self.v2_0,
      kappa2: self.kappa2,
      theta2: self.theta2,
      sigma2: self.sigma2,
      rho2: self.rho2,
    }
  }

  pub fn to_model(&self, r: f64, q: f64) -> crate::pricing::fourier::DoubleHestonFourier {
    self.params().to_model(r, q)
  }
}

impl crate::traits::ToModel for DoubleHestonCalibrationResult {
  fn to_model(&self, r: f64, q: f64) -> Box<dyn crate::traits::ModelPricer> {
    Box::new(self.params().to_model(r, q))
  }
}

/// Double Heston characteristic function $\phi_T(u)$ of $\ln(S_T/S_0)$,
/// computed as the sum of two independent Heston contributions.
fn double_heston_cf(p: &DoubleHestonParams, r: f64, q: f64, tau: f64, u: Complex64) -> Complex64 {
  let i = Complex64::i();

  let (c1, d1) = factor_cd(p.kappa1, p.theta1, p.sigma1, p.rho1, tau, u);
  let (c2, d2) = factor_cd(p.kappa2, p.theta2, p.sigma2, p.rho2, tau, u);

  (c1 + c2 + d1 * p.v1_0 + d2 * p.v2_0 + i * u * (r - q) * tau).exp()
}

/// Single Heston factor $(C_j, D_j)$ at Fourier argument `u`.
fn factor_cd(
  kappa: f64,
  theta: f64,
  sigma: f64,
  rho: f64,
  tau: f64,
  u: Complex64,
) -> (Complex64, Complex64) {
  let i = Complex64::i();
  let sigma2 = sigma * sigma;
  let rsi = rho * sigma * i;

  let d = ((kappa - rsi * u).powi(2) + sigma2 * (i * u + u * u)).sqrt();
  let g = (kappa - rsi * u - d) / (kappa - rsi * u + d);
  let exp_dt = (-d * tau).exp();

  let c_val = (kappa * theta / sigma2)
    * ((kappa - rsi * u - d) * tau - 2.0 * ((1.0 - g * exp_dt) / (1.0 - g)).ln());
  let d_val = ((kappa - rsi * u - d) / sigma2) * (1.0 - exp_dt) / (1.0 - g * exp_dt);

  (c_val, d_val)
}

/// Price a European call option under the double Heston model via the
/// Gil-Pelaez quadrature over 64-point Gauss-Legendre nodes.
fn double_heston_call_price(
  p: &DoubleHestonParams,
  s: f64,
  k: f64,
  r: f64,
  q: f64,
  tau: f64,
) -> f64 {
  let (nodes, weights) = gauss_legendre_64();
  let scale = 0.5 * GL_U_MAX;
  let ln_ks = (k / s).ln();

  let mut p1_int = 0.0_f64;
  let mut p2_int = 0.0_f64;

  let phi_neg_i = double_heston_cf(p, r, q, tau, Complex64::new(0.0, -1.0));
  let phi_neg_i_norm = phi_neg_i.norm();

  for (&x, &w) in nodes.iter().zip(weights.iter()) {
    let u_real = scale * (x + 1.0);
    let w_s = scale * w;
    if u_real <= EPS {
      continue;
    }

    let xi = Complex64::new(u_real, 0.0);
    let xi_shift = Complex64::new(u_real, -1.0);

    let phi = double_heston_cf(p, r, q, tau, xi);
    let phi_shift = double_heston_cf(p, r, q, tau, xi_shift);

    let kernel =
      (Complex64::new(0.0, -u_real * ln_ks)).exp() / (Complex64::i() * Complex64::new(u_real, 0.0));

    p2_int += w_s * (kernel * phi).re;
    if phi_neg_i_norm > 1e-30 {
      p1_int += w_s * (kernel * phi_shift / phi_neg_i).re;
    }
  }

  let p1 = 0.5 + FRAC_1_PI * p1_int;
  let p2 = 0.5 + FRAC_1_PI * p2_int;

  let call = s * (-q * tau).exp() * p1 - k * (-r * tau).exp() * p2;
  call.max(0.0)
}

/// Double Heston least-squares calibrator using Levenberg-Marquardt.
#[derive(Clone)]
pub struct DoubleHestonCalibrator {
  pub params: Option<DoubleHestonParams>,
  pub c_market: DVector<f64>,
  pub s: DVector<f64>,
  pub k: DVector<f64>,
  pub r: f64,
  pub q: Option<f64>,
  pub flat_t: Vec<f64>,
  pub option_type: OptionType,
  pub record_history: bool,
  pub loss_metrics: &'static [LossMetric],
  calibration_history: Rc<RefCell<Vec<CalibrationHistory<DoubleHestonParams>>>>,
}

impl DoubleHestonCalibrator {
  /// Create a calibrator for a single maturity slice.
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    params: Option<DoubleHestonParams>,
    c_market: DVector<f64>,
    s: DVector<f64>,
    k: DVector<f64>,
    r: f64,
    q: Option<f64>,
    tau: f64,
    option_type: OptionType,
    record_history: bool,
  ) -> Self {
    let n = c_market.len();
    assert_eq!(n, s.len(), "c_market and s must have the same length");
    assert_eq!(n, k.len(), "c_market and k must have the same length");
    assert!(
      tau.is_finite() && tau > 0.0,
      "tau must be a finite positive value"
    );

    Self {
      params,
      c_market,
      s,
      k,
      r,
      q,
      flat_t: vec![tau; n],
      option_type,
      record_history,
      loss_metrics: &LossMetric::ALL,
      calibration_history: Rc::new(RefCell::new(Vec::new())),
    }
  }

  /// Create a calibrator from multiple maturity slices for joint surface calibration.
  pub fn from_slices(
    params: Option<DoubleHestonParams>,
    slices: &[super::levy::MarketSlice],
    s: f64,
    r: f64,
    q: Option<f64>,
    option_type: OptionType,
    record_history: bool,
  ) -> Self {
    let mut flat_prices = Vec::new();
    let mut flat_strikes = Vec::new();
    let mut flat_t = Vec::new();
    let mut flat_s = Vec::new();

    for slice in slices {
      for i in 0..slice.strikes.len() {
        flat_prices.push(slice.prices[i]);
        flat_strikes.push(slice.strikes[i]);
        flat_t.push(slice.t);
        flat_s.push(s);
      }
    }

    Self {
      params,
      c_market: DVector::from_vec(flat_prices),
      s: DVector::from_vec(flat_s),
      k: DVector::from_vec(flat_strikes),
      r,
      q,
      flat_t,
      option_type,
      record_history,
      loss_metrics: &LossMetric::ALL,
      calibration_history: Rc::new(RefCell::new(Vec::new())),
    }
  }

  /// Run the calibration via Levenberg-Marquardt.
  pub fn calibrate(
    &self,
    initial_params: Option<DoubleHestonParams>,
  ) -> DoubleHestonCalibrationResult {
    let mut problem = self.clone();
    if let Some(p) = initial_params {
      problem.params = Some(p.projected());
    }
    problem.ensure_initial_guess();

    let (result, report) = LevenbergMarquardt::new().minimize(problem);

    let p = result.effective_params();
    let c_model = result.compute_model_prices_for(&p);
    let loss = CalibrationLossScore::compute_selected(
      result.c_market.as_slice(),
      c_model.as_slice(),
      result.loss_metrics,
    );

    DoubleHestonCalibrationResult {
      v1_0: p.v1_0,
      kappa1: p.kappa1,
      theta1: p.theta1,
      sigma1: p.sigma1,
      rho1: p.rho1,
      v2_0: p.v2_0,
      kappa2: p.kappa2,
      theta2: p.theta2,
      sigma2: p.sigma2,
      rho2: p.rho2,
      loss,
      converged: report.termination.was_successful(),
    }
  }

  pub fn set_initial_guess(&mut self, params: DoubleHestonParams) {
    self.params = Some(params.projected());
  }

  pub fn set_record_history(&mut self, record: bool) {
    self.record_history = record;
  }

  pub fn history(&self) -> Vec<CalibrationHistory<DoubleHestonParams>> {
    self.calibration_history.borrow().clone()
  }

  /// Fallback: a fast-and-slow factor split centred around realistic
  /// equity-index values (fast factor mean-reverts in ≈ 4 months, slow factor
  /// in ≈ 2 years).
  fn fallback_params() -> DoubleHestonParams {
    DoubleHestonParams {
      v1_0: 0.02,
      kappa1: 3.0,
      theta1: 0.02,
      sigma1: 0.3,
      rho1: -0.6,
      v2_0: 0.02,
      kappa2: 0.5,
      theta2: 0.02,
      sigma2: 0.15,
      rho2: -0.3,
    }
    .projected()
  }

  fn ensure_initial_guess(&mut self) {
    if self.params.is_none() {
      self.params = Some(Self::fallback_params());
    }
  }

  fn effective_params(&self) -> DoubleHestonParams {
    if let Some(p) = &self.params {
      return (*p).projected();
    }
    Self::fallback_params()
  }

  fn compute_model_prices_for(&self, p: &DoubleHestonParams) -> DVector<f64> {
    let n = self.c_market.len();
    let mut c_model = DVector::zeros(n);
    let q_val = self.q.unwrap_or(0.0);

    for idx in 0..n {
      let tau = self.flat_t[idx];
      let call = double_heston_call_price(p, self.s[idx], self.k[idx], self.r, q_val, tau);
      c_model[idx] = match self.option_type {
        OptionType::Call => call.max(0.0),
        OptionType::Put => {
          let put = call - self.s[idx] * (-q_val * tau).exp() + self.k[idx] * (-self.r * tau).exp();
          put.max(0.0)
        }
      };
    }

    c_model
  }

  fn residuals_for(&self, p: &DoubleHestonParams) -> DVector<f64> {
    self.c_market.clone() - self.compute_model_prices_for(p)
  }

  /// Central finite-difference Jacobian over the 10 parameters.
  fn numeric_jacobian(&self, params: &DoubleHestonParams) -> DMatrix<f64> {
    let n = self.c_market.len();
    let p_dim = 10usize;

    let base_params_vec: DVector<f64> = (*params).into();
    let mut j_mat = DMatrix::zeros(n, p_dim);

    for col in 0..p_dim {
      let x = base_params_vec[col];
      let mut h = 1e-5_f64.max(1e-3 * x.abs());

      let mut params_plus = *params;
      let mut params_minus = *params;

      let field_plus_minus = |x: f64, h: f64, lo: f64, hi: f64| -> (f64, f64) {
        (periodic_map(x + h, lo, hi), periodic_map(x - h, lo, hi))
      };

      match col {
        0 => {
          let (a, b) = field_plus_minus(x, h, P_V0.0, P_V0.1);
          params_plus.v1_0 = a.max(0.0);
          params_minus.v1_0 = b.max(0.0);
        }
        1 => {
          let (a, b) = field_plus_minus(x, h, P_KAPPA.0, P_KAPPA.1);
          params_plus.kappa1 = a.max(KAPPA_MIN);
          params_minus.kappa1 = b.max(KAPPA_MIN);
        }
        2 => {
          let (a, b) = field_plus_minus(x, h, P_THETA.0, P_THETA.1);
          params_plus.theta1 = a.max(THETA_MIN);
          params_minus.theta1 = b.max(THETA_MIN);
        }
        3 => {
          let (a, b) = field_plus_minus(x, h, P_SIGMA.0, P_SIGMA.1);
          params_plus.sigma1 = a.abs().max(SIGMA_MIN);
          params_minus.sigma1 = b.abs().max(SIGMA_MIN);
        }
        4 => {
          let clamp = |y: f64| y.clamp(-RHO_BOUND, RHO_BOUND);
          params_plus.rho1 = clamp(x + h);
          params_minus.rho1 = clamp(x - h);
          if (params_plus.rho1 - params_minus.rho1).abs() < 0.5 * h {
            h = 1e-4;
            params_plus.rho1 = clamp(x + h);
            params_minus.rho1 = clamp(x - h);
          }
        }
        5 => {
          let (a, b) = field_plus_minus(x, h, P_V0.0, P_V0.1);
          params_plus.v2_0 = a.max(0.0);
          params_minus.v2_0 = b.max(0.0);
        }
        6 => {
          let (a, b) = field_plus_minus(x, h, P_KAPPA.0, P_KAPPA.1);
          params_plus.kappa2 = a.max(KAPPA_MIN);
          params_minus.kappa2 = b.max(KAPPA_MIN);
        }
        7 => {
          let (a, b) = field_plus_minus(x, h, P_THETA.0, P_THETA.1);
          params_plus.theta2 = a.max(THETA_MIN);
          params_minus.theta2 = b.max(THETA_MIN);
        }
        8 => {
          let (a, b) = field_plus_minus(x, h, P_SIGMA.0, P_SIGMA.1);
          params_plus.sigma2 = a.abs().max(SIGMA_MIN);
          params_minus.sigma2 = b.abs().max(SIGMA_MIN);
        }
        9 => {
          let clamp = |y: f64| y.clamp(-RHO_BOUND, RHO_BOUND);
          params_plus.rho2 = clamp(x + h);
          params_minus.rho2 = clamp(x - h);
          if (params_plus.rho2 - params_minus.rho2).abs() < 0.5 * h {
            h = 1e-4;
            params_plus.rho2 = clamp(x + h);
            params_minus.rho2 = clamp(x - h);
          }
        }
        _ => unreachable!(),
      }

      params_plus.project_in_place();
      params_minus.project_in_place();

      let r_plus = self.residuals_for(&params_plus);
      let r_minus = self.residuals_for(&params_minus);

      let diff = (r_plus - r_minus) / (2.0 * h);
      for row in 0..n {
        j_mat[(row, col)] = diff[row];
      }
    }

    j_mat
  }
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for DoubleHestonCalibrator {
  type JacobianStorage = Owned<f64, Dyn, Dyn>;
  type ParameterStorage = Owned<f64, Dyn>;
  type ResidualStorage = Owned<f64, Dyn>;

  fn set_params(&mut self, params: &DVector<f64>) {
    let p = DoubleHestonParams::from(params.clone()).projected();
    self.params = Some(p);
  }

  fn params(&self) -> DVector<f64> {
    self.effective_params().into()
  }

  fn residuals(&self) -> Option<DVector<f64>> {
    let params_eff = self.effective_params();
    let c_model = self.compute_model_prices_for(&params_eff);

    if self.record_history {
      let q_val = self.q.unwrap_or(0.0);
      self
        .calibration_history
        .borrow_mut()
        .push(CalibrationHistory {
          residuals: self.c_market.clone() - c_model.clone(),
          call_put: self
            .c_market
            .iter()
            .enumerate()
            .map(|(idx, _)| {
              let tau = self.flat_t[idx];
              let call =
                double_heston_call_price(&params_eff, self.s[idx], self.k[idx], self.r, q_val, tau);
              let put =
                call - self.s[idx] * (-q_val * tau).exp() + self.k[idx] * (-self.r * tau).exp();
              (call.max(0.0), put.max(0.0))
            })
            .collect::<Vec<(f64, f64)>>()
            .into(),
          params: params_eff,
          loss_scores: CalibrationLossScore::compute_selected(
            self.c_market.as_slice(),
            c_model.as_slice(),
            self.loss_metrics,
          ),
        });
    }

    Some(self.c_market.clone() - c_model)
  }

  fn jacobian(&self) -> Option<DMatrix<f64>> {
    Some(self.numeric_jacobian(&self.effective_params()))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  // Reference Heston (v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.7)
  // S=100, r=0.05, q=0, T=1.0. A double Heston with v1_0+v2_0 = 0.04 and
  // factor-1 degenerate (v1_0 ≈ 0) collapses onto the single Heston answer.
  const HESTON_REF: [f64; 9] = [
    25.095178, 20.976171, 17.106937, 13.548230, 10.361869, 7.604362, 5.317953, 3.519953, 2.193310,
  ];
  const STRIKES: [f64; 9] = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0];

  #[test]
  fn double_heston_reduces_to_heston_when_one_factor_vanishes() {
    // Factor 2 has v2_0=0 and theta2≈0 so its contribution is negligible.
    let p = DoubleHestonParams {
      v1_0: 0.04,
      kappa1: 1.5,
      theta1: 0.04,
      sigma1: 0.3,
      rho1: -0.7,
      v2_0: 1e-6,
      kappa2: 5.0,
      theta2: 1e-6,
      sigma2: 0.01,
      rho2: 0.0,
    };
    for (i, &k) in STRIKES.iter().enumerate() {
      let price = double_heston_call_price(&p, 100.0, k, 0.05, 0.0, 1.0);
      assert!(
        (price - HESTON_REF[i]).abs() < 0.25,
        "Double Heston (1 active factor) K={k}: got {price:.6}, expected {:.6}",
        HESTON_REF[i]
      );
    }
  }

  #[test]
  fn double_heston_two_factors_produces_sensible_smile() {
    // Standard Christoffersen 2-factor split: one fast-mean-reverting factor,
    // one slow-mean-reverting factor. Prices should be positive, finite,
    // monotonic in strike, and satisfy basic bounds.
    let p = DoubleHestonParams {
      v1_0: 0.02,
      kappa1: 3.0,
      theta1: 0.02,
      sigma1: 0.4,
      rho1: -0.6,
      v2_0: 0.02,
      kappa2: 0.5,
      theta2: 0.03,
      sigma2: 0.2,
      rho2: -0.3,
    };
    let mut prev = f64::INFINITY;
    for &k in STRIKES.iter() {
      let price = double_heston_call_price(&p, 100.0, k, 0.05, 0.0, 1.0);
      // Intrinsic value lower bound: max(S e^{-qT} - K e^{-rT}, 0)
      let intrinsic = (100.0_f64 - k * (-0.05_f64 * 1.0).exp()).max(0.0);
      assert!(
        price.is_finite() && price >= intrinsic - 1e-6 && price <= 100.0 + 1e-6,
        "Double Heston K={k}: got {price:.6} outside [{intrinsic:.6}, 100]"
      );
      assert!(
        price < prev + 1e-6,
        "Double Heston prices should be monotone decreasing in strike: {prev:.6} → {price:.6} at K={k}"
      );
      prev = price;
    }
  }

  #[test]
  fn double_heston_calibrate_to_heston_surface() {
    // Calibrate to Heston prices: we expect the RMSE to be small.
    let n = STRIKES.len();
    let calibrator = DoubleHestonCalibrator::new(
      Some(DoubleHestonParams {
        v1_0: 0.03,
        kappa1: 2.0,
        theta1: 0.03,
        sigma1: 0.25,
        rho1: -0.5,
        v2_0: 0.01,
        kappa2: 0.8,
        theta2: 0.02,
        sigma2: 0.15,
        rho2: -0.4,
      }),
      HESTON_REF.to_vec().into(),
      vec![100.0; n].into(),
      STRIKES.to_vec().into(),
      0.05,
      Some(0.0),
      1.0,
      OptionType::Call,
      false,
    );

    let result = calibrator.calibrate(None);
    assert!(
      result.loss.get(LossMetric::Rmse) < 0.6,
      "Double Heston calibration RMSE={:.6}",
      result.loss.get(LossMetric::Rmse)
    );
    // Verify the calibrated price is close to the reference
    let p_out = result.params();
    for (i, &k) in STRIKES.iter().enumerate() {
      let price = double_heston_call_price(&p_out, 100.0, k, 0.05, 0.0, 1.0);
      assert!(
        (price - HESTON_REF[i]).abs() < 1.5,
        "Calibrated price K={k}: got {price:.4}, ref {:.4}",
        HESTON_REF[i]
      );
    }
  }

  #[test]
  fn double_heston_params_to_model() {
    let p = DoubleHestonParams {
      v1_0: 0.02,
      kappa1: 3.0,
      theta1: 0.02,
      sigma1: 0.3,
      rho1: -0.6,
      v2_0: 0.02,
      kappa2: 0.5,
      theta2: 0.02,
      sigma2: 0.15,
      rho2: -0.3,
    };
    let model = p.to_model(0.03, 0.01);
    assert_eq!(model.v1_0, 0.02);
    assert_eq!(model.kappa2, 0.5);
    assert_eq!(model.r, 0.03);
    assert_eq!(model.q, 0.01);
  }
}

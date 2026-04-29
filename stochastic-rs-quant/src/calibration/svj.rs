//! # SVJ (Bates) Calibration
//!
//! $$
//! \begin{aligned}
//! dS_t &= (r-q-\lambda\bar k)S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S + J_t\,S_t\,dN_t \\
//! dv_t &= \kappa(\theta-v_t)\,dt + \sigma_v\sqrt{v_t}\,dW_t^v, \quad
//! d\langle W^S,W^v\rangle_t = \rho\,dt
//! \end{aligned}
//! $$
//!
//! where $J_t\sim\mathcal{N}(\mu_J,\sigma_J^2)$ are i.i.d. log-jump sizes and
//! $N_t$ is a Poisson process with intensity $\lambda$.
//!
//! The characteristic function is
//! $$
//! \phi_T(\xi) = \phi_T^{\mathrm{Heston}}(\xi)\cdot
//! \exp\!\bigl[T\,\lambda\bigl(e^{i\mu_J\xi - \tfrac12\sigma_J^2\xi^2} - 1\bigr)\bigr].
//! $$
//!
//! Source:
//! - Bates, D. (1996), "Jumps and Stochastic Volatility"
//!   https://doi.org/10.1093/rfs/9.1.69
//! - Heston, S. L. (1993)
//!   https://doi.org/10.1093/rfs/6.2.327

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
const SIGMA_V_MIN: f64 = 1e-8;

// Parameter ranges for periodic/box projection
const P_V0: (f64, f64) = (0.005, 0.25);
const P_KAPPA: (f64, f64) = (0.1, 20.0);
const P_THETA: (f64, f64) = (0.001, 0.4);
const P_SIGMA_V: (f64, f64) = (0.01, 1.0);
const P_RHO: (f64, f64) = (-1.0, 1.0);
const P_LAMBDA: (f64, f64) = (0.0, 10.0);
const P_MU_J: (f64, f64) = (-0.5, 0.5);
const P_SIGMA_J: (f64, f64) = (0.001, 1.0);

/// SVJ / Bates model parameters.
#[derive(Clone, Copy, Debug)]
pub struct SVJParams {
  /// Initial variance $v_0$.
  pub v0: f64,
  /// Mean-reversion speed $\kappa$.
  pub kappa: f64,
  /// Long-run variance $\theta$.
  pub theta: f64,
  /// Volatility of variance $\sigma_v$.
  pub sigma_v: f64,
  /// Correlation $\rho$ between price and variance Brownian motions.
  pub rho: f64,
  /// Jump intensity $\lambda$ (Poisson arrival rate).
  pub lambda: f64,
  /// Mean log-jump size $\mu_J$.
  pub mu_j: f64,
  /// Jump-size volatility $\sigma_J$.
  pub sigma_j: f64,
}

impl SVJParams {
  /// Project parameters to satisfy admissibility constraints (box + Feller condition).
  pub fn project_in_place(&mut self) {
    self.v0 = periodic_map(self.v0, P_V0.0, P_V0.1).max(0.0);
    self.kappa = periodic_map(self.kappa, P_KAPPA.0, P_KAPPA.1).max(KAPPA_MIN);
    self.theta = periodic_map(self.theta, P_THETA.0, P_THETA.1).max(THETA_MIN);
    self.sigma_v = periodic_map(self.sigma_v, P_SIGMA_V.0, P_SIGMA_V.1)
      .abs()
      .max(SIGMA_V_MIN);
    self.rho = periodic_map(self.rho, P_RHO.0, P_RHO.1).clamp(-RHO_BOUND, RHO_BOUND);
    self.lambda = periodic_map(self.lambda, P_LAMBDA.0, P_LAMBDA.1).max(0.0);
    self.mu_j = periodic_map(self.mu_j, P_MU_J.0, P_MU_J.1);
    self.sigma_j = periodic_map(self.sigma_j, P_SIGMA_J.0, P_SIGMA_J.1)
      .abs()
      .max(P_SIGMA_J.0);

    // Feller condition: 2*kappa*theta >= sigma_v^2
    if 2.0 * self.kappa * self.theta < self.sigma_v * self.sigma_v {
      let sigma_star = (2.0 * self.kappa * self.theta).sqrt();
      if sigma_star >= P_SIGMA_V.0 {
        self.sigma_v = sigma_star.min(P_SIGMA_V.1);
      } else {
        let theta_star = ((self.sigma_v * self.sigma_v) / (2.0 * self.kappa)).max(THETA_MIN) + EPS;
        self.theta = theta_star.min(P_THETA.1);
      }
    }
  }

  pub fn projected(mut self) -> Self {
    self.project_in_place();
    self
  }
}

impl From<SVJParams> for DVector<f64> {
  fn from(p: SVJParams) -> Self {
    DVector::from_vec(vec![
      p.v0, p.kappa, p.theta, p.sigma_v, p.rho, p.lambda, p.mu_j, p.sigma_j,
    ])
  }
}

impl From<DVector<f64>> for SVJParams {
  fn from(v: DVector<f64>) -> Self {
    SVJParams {
      v0: v[0],
      kappa: v[1],
      theta: v[2],
      sigma_v: v[3],
      rho: v[4],
      lambda: v[5],
      mu_j: v[6],
      sigma_j: v[7],
    }
  }
}

/// Calibration result for the SVJ (Bates) model.
#[derive(Clone, Debug)]
pub struct SVJCalibrationResult {
  /// Initial variance.
  pub v0: f64,
  /// Mean-reversion speed.
  pub kappa: f64,
  /// Long-run variance.
  pub theta: f64,
  /// Volatility of variance.
  pub sigma_v: f64,
  /// Correlation.
  pub rho: f64,
  /// Jump intensity.
  pub lambda: f64,
  /// Mean log-jump size.
  pub mu_j: f64,
  /// Jump-size volatility.
  pub sigma_j: f64,
  /// Calibration loss metrics.
  pub loss: CalibrationLossScore,
  /// Whether the optimiser converged.
  pub converged: bool,
}

impl crate::traits::ToModel for SVJCalibrationResult {
  type Model = crate::pricing::fourier::BatesFourier;
  fn to_model(&self, r: f64, q: f64) -> Self::Model {
    SVJCalibrationResult::to_model(self, r, q)
  }
}

impl SVJCalibrationResult {
  /// Convert to a [`BatesFourier`] model for pricing / vol surface generation.
  pub fn to_model(&self, r: f64, q: f64) -> crate::pricing::fourier::BatesFourier {
    crate::pricing::fourier::BatesFourier {
      v0: self.v0,
      kappa: self.kappa,
      theta: self.theta,
      sigma_v: self.sigma_v,
      rho: self.rho,
      lambda: self.lambda,
      mu_j: self.mu_j,
      sigma_j: self.sigma_j,
      r,
      q,
    }
  }
}

impl crate::traits::CalibrationResult for SVJCalibrationResult {
  type Params = SVJParams;
  fn rmse(&self) -> f64 {
    self.loss.get(LossMetric::Rmse)
  }
  fn converged(&self) -> bool {
    self.converged
  }
  fn params(&self) -> Self::Params {
    SVJParams {
      v0: self.v0,
      kappa: self.kappa,
      theta: self.theta,
      sigma_v: self.sigma_v,
      rho: self.rho,
      lambda: self.lambda,
      mu_j: self.mu_j,
      sigma_j: self.sigma_j,
    }
  }
  fn loss_score(&self) -> Option<&CalibrationLossScore> {
    Some(&self.loss)
  }
}

impl crate::traits::Calibrator for SVJCalibrator {
  type InitialGuess = SVJParams;
  type Params = SVJParams;
  type Output = SVJCalibrationResult;
  type Error = anyhow::Error;

  fn calibrate(&self, initial: Option<Self::InitialGuess>) -> Result<Self::Output, Self::Error> {
    Ok(self.solve(initial))
  }
}

/// SVJ (Bates) least-squares calibrator using Levenberg-Marquardt.
///
/// Source:
/// - Levenberg (1944), https://doi.org/10.1090/qam/10666
/// - Marquardt (1963), https://doi.org/10.1137/0111030
/// - Bates (1996), https://doi.org/10.1093/rfs/9.1.69
#[derive(Clone)]
pub struct SVJCalibrator {
  /// Params to calibrate.
  pub params: Option<SVJParams>,
  /// Option prices from the market (flattened across all maturities).
  pub c_market: DVector<f64>,
  /// Underlying spot per quote.
  pub s: DVector<f64>,
  /// Strikes per quote (flattened).
  pub k: DVector<f64>,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield.
  pub q: Option<f64>,
  /// Time to maturity per quote (flattened). Supports multi-maturity joint calibration.
  pub flat_t: Vec<f64>,
  /// Option type.
  pub option_type: OptionType,
  /// If true, record per-iteration calibration history.
  pub record_history: bool,
  /// Which loss metrics to compute when recording history.
  pub loss_metrics: &'static [LossMetric],
  /// History of iterations.
  calibration_history: Rc<RefCell<Vec<CalibrationHistory<SVJParams>>>>,
}

/// Bates/SVJ characteristic function $\phi_T(\xi)$.
///
/// $$
/// \phi_T(\xi) = \exp\!\bigl[C(\xi,T)+D(\xi,T)\,v_0+i\xi\ln S\bigr]
/// $$
///
/// where $C$ and $D$ are the Heston terms augmented by the jump compensator:
/// $$
/// C_{\mathrm{Bates}} = C_{\mathrm{Heston}} + T\,\lambda\bigl(e^{i\mu_J\xi-\frac12\sigma_J^2\xi^2}-1\bigr)
///   - i\xi\,T\,\lambda\bigl(e^{\mu_J+\frac12\sigma_J^2}-1\bigr)
///
/// $$
fn bates_cf(p: &SVJParams, s: f64, r: f64, q: f64, tau: f64, u: Complex64) -> Complex64 {
  let i = Complex64::i();
  let iu = i * u;
  let rq = r - q;

  // Heston characteristic function components (log-stock formulation)
  let xi_h = Complex64::new(p.kappa, 0.0) - p.sigma_v * p.rho * iu;
  let d = (xi_h * xi_h + p.sigma_v * p.sigma_v * (u * u + iu)).sqrt();

  let exp_neg_dt = (-d * tau).exp();
  let g = (xi_h - d) / (xi_h + d);

  let sigma_v2 = p.sigma_v * p.sigma_v;

  let c_heston = iu * (s.ln() + rq * tau)
    + (p.kappa * p.theta / sigma_v2)
      * ((xi_h - d) * tau - 2.0 * ((1.0 - g * exp_neg_dt) / (1.0 - g)).ln());

  let d_heston = ((xi_h - d) / sigma_v2) * ((1.0 - exp_neg_dt) / (1.0 - g * exp_neg_dt));

  // Jump compensator: martingale correction
  // k_bar = E[e^J - 1] = exp(mu_j + 0.5*sigma_j^2) - 1
  let k_bar = (p.mu_j + 0.5 * p.sigma_j * p.sigma_j).exp() - 1.0;
  let jump_cf = p.lambda * ((i * p.mu_j * u - 0.5 * p.sigma_j * p.sigma_j * u * u).exp() - 1.0);
  let jump_drift = -p.lambda * k_bar * iu;

  let log_phi = c_heston + d_heston * p.v0 + (jump_cf + jump_drift) * tau;
  log_phi.exp()
}

/// Price a European call option under the Bates/SVJ model using
/// Gauss-Legendre quadrature over the Gil-Pelaez integral.
fn bates_call_price(p: &SVJParams, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
  let (nodes, weights) = gauss_legendre_64();
  let scale = 0.5 * GL_U_MAX;

  let mut i1 = 0.0_f64;
  let mut i2 = 0.0_f64;

  for (&x, &w) in nodes.iter().zip(weights.iter()) {
    let u_real = scale * (x + 1.0);
    let w_s = scale * w;
    if u_real <= EPS {
      continue;
    }

    let xi = Complex64::new(u_real, 0.0);
    let xi_shift = Complex64::new(u_real, -1.0);

    let phi = bates_cf(p, s, r, q, tau, xi);
    let phi_shift = bates_cf(p, s, r, q, tau, xi_shift);

    let kernel = (Complex64::new(0.0, -u_real * k.ln())).exp()
      / (Complex64::i() * Complex64::new(u_real, 0.0));

    i1 += w_s * (kernel * phi_shift).re;
    i2 += w_s * (kernel * phi).re;
  }

  let disc_r = (-r * tau).exp();
  let disc_q = (-q * tau).exp();
  let call = 0.5 * (s * disc_q - k * disc_r) + disc_r * FRAC_1_PI * (i1 - k * i2);
  call.max(0.0)
}

impl SVJCalibrator {
  /// Create a calibrator for a single maturity slice (backwards compatible).
  pub fn new(
    params: Option<SVJParams>,
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
    params: Option<SVJParams>,
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
}

impl SVJCalibrator {
  fn solve(&self, initial_params: Option<SVJParams>) -> SVJCalibrationResult {
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

    SVJCalibrationResult {
      v0: p.v0,
      kappa: p.kappa,
      theta: p.theta,
      sigma_v: p.sigma_v,
      rho: p.rho,
      lambda: p.lambda,
      mu_j: p.mu_j,
      sigma_j: p.sigma_j,
      loss,
      converged: report.termination.was_successful(),
    }
  }

  pub fn set_initial_guess(&mut self, params: SVJParams) {
    self.params = Some(params.projected());
  }

  pub fn set_record_history(&mut self, record: bool) {
    self.record_history = record;
  }

  pub fn history(&self) -> Vec<CalibrationHistory<SVJParams>> {
    self.calibration_history.borrow().clone()
  }

  fn fallback_params() -> SVJParams {
    SVJParams {
      v0: 0.04,
      kappa: 1.5,
      theta: 0.04,
      sigma_v: 0.5,
      rho: -0.5,
      lambda: 0.5,
      mu_j: -0.05,
      sigma_j: 0.1,
    }
    .projected()
  }

  fn ensure_initial_guess(&mut self) {
    if self.params.is_none() {
      self.params = Some(Self::fallback_params());
    }
  }

  fn effective_params(&self) -> SVJParams {
    if let Some(p) = &self.params {
      return (*p).projected();
    }
    Self::fallback_params()
  }

  fn compute_model_prices_for(&self, p: &SVJParams) -> DVector<f64> {
    let n = self.c_market.len();
    let mut c_model = DVector::zeros(n);
    let q_val = self.q.unwrap_or(0.0);

    for idx in 0..n {
      let tau = self.flat_t[idx];
      let call = bates_call_price(p, self.s[idx], self.k[idx], self.r, q_val, tau);
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

  fn residuals_for(&self, p: &SVJParams) -> DVector<f64> {
    self.c_market.clone() - self.compute_model_prices_for(p)
  }

  /// Central finite-difference Jacobian.
  fn numeric_jacobian(&self, params: &SVJParams) -> DMatrix<f64> {
    let n = self.c_market.len();
    let p = 8usize; // v0, kappa, theta, sigma_v, rho, lambda, mu_j, sigma_j

    let base_params_vec: DVector<f64> = (*params).into();
    let mut j_mat = DMatrix::zeros(n, p);

    for col in 0..p {
      let x = base_params_vec[col];
      let mut h = 1e-5_f64.max(1e-3 * x.abs());

      let mut params_plus = *params;
      let mut params_minus = *params;

      match col {
        0 => {
          params_plus.v0 = (x + h).max(0.0);
          params_minus.v0 = (x - h).max(0.0);
        }
        1 => {
          params_plus.kappa = (x + h).max(KAPPA_MIN);
          params_minus.kappa = (x - h).max(KAPPA_MIN);
        }
        2 => {
          params_plus.theta = (x + h).max(THETA_MIN);
          params_minus.theta = (x - h).max(THETA_MIN);
        }
        3 => {
          params_plus.sigma_v = (x + h).abs().max(SIGMA_V_MIN);
          params_minus.sigma_v = (x - h).abs().max(SIGMA_V_MIN);
        }
        4 => {
          let clamp = |y: f64| y.clamp(-RHO_BOUND, RHO_BOUND);
          params_plus.rho = clamp(x + h);
          params_minus.rho = clamp(x - h);
          if (params_plus.rho - params_minus.rho).abs() < 0.5 * h {
            h = 1e-4;
            params_plus.rho = clamp(x + h);
            params_minus.rho = clamp(x - h);
          }
        }
        5 => {
          params_plus.lambda = (x + h).max(0.0);
          params_minus.lambda = (x - h).max(0.0);
        }
        6 => {
          params_plus.mu_j = (x + h).clamp(P_MU_J.0, P_MU_J.1);
          params_minus.mu_j = (x - h).clamp(P_MU_J.0, P_MU_J.1);
        }
        7 => {
          params_plus.sigma_j = (x + h).abs().max(P_SIGMA_J.0);
          params_minus.sigma_j = (x - h).abs().max(P_SIGMA_J.0);
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

impl LeastSquaresProblem<f64, Dyn, Dyn> for SVJCalibrator {
  type JacobianStorage = Owned<f64, Dyn, Dyn>;
  type ParameterStorage = Owned<f64, Dyn>;
  type ResidualStorage = Owned<f64, Dyn>;

  fn set_params(&mut self, params: &DVector<f64>) {
    let p = SVJParams::from(params.clone()).projected();
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
                bates_call_price(&params_eff, self.s[idx], self.k[idx], self.r, q_val, tau);
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
  use crate::traits::Calibrator;

  // Analytical reference: Heston (v0=0.04, kappa=1.5, theta=0.04, sigma_v=0.3, rho=-0.7)
  // S=100, r=0.05, q=0, T=1.0
  const HESTON_REF: [f64; 9] = [
    25.095178, 20.976171, 17.106937, 13.548230, 10.361869, 7.604362, 5.317953, 3.519953, 2.193310,
  ];
  const STRIKES: [f64; 9] = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0];

  #[test]
  fn bates_pricer_matches_heston_reference() {
    // With lambda=0, SVJ reduces to Heston. Verify we reproduce reference Heston prices.
    let p = SVJParams {
      v0: 0.04,
      kappa: 1.5,
      theta: 0.04,
      sigma_v: 0.3,
      rho: -0.7,
      lambda: 0.0,
      mu_j: 0.0,
      sigma_j: 0.01, // small nonzero to avoid division issues
    };
    for (i, &k) in STRIKES.iter().enumerate() {
      let price = bates_call_price(&p, 100.0, k, 0.05, 0.0, 1.0);
      assert!(
        (price - HESTON_REF[i]).abs() < 0.1,
        "Heston K={k}: got {price:.6}, expected {:.6}",
        HESTON_REF[i]
      );
    }
  }

  #[test]
  fn svj_calibrate_recovers_heston_prices() {
    // Use reference Heston prices as market data. SVJ with lambda≈0 should recover them.
    let n = STRIKES.len();
    let calibrator = SVJCalibrator::new(
      Some(SVJParams {
        v0: 0.06,
        kappa: 2.0,
        theta: 0.06,
        sigma_v: 0.4,
        rho: -0.5,
        lambda: 0.1,
        mu_j: 0.0,
        sigma_j: 0.1,
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

    let result = calibrator.calibrate(None).unwrap();
    // SVJ has 8 params fitting 9 points from a 5-param submodel; local minima expected
    assert!(
      result.loss.get(LossMetric::Rmse) < 0.5,
      "SVJ→Heston RMSE={:.6}",
      result.loss.get(LossMetric::Rmse)
    );
    println!(
      "SVJ→Heston: v0={:.4}, kappa={:.4}, theta={:.4}, sigma_v={:.4}, rho={:.4}, lambda={:.4}",
      result.v0, result.kappa, result.theta, result.sigma_v, result.rho, result.lambda
    );
  }

  #[test]
  fn test_svj_calibrate() {
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

    let calibrator = SVJCalibrator::new(
      Some(SVJParams {
        v0: 0.04,
        kappa: 1.5,
        theta: 0.04,
        sigma_v: 0.5,
        rho: -0.7,
        lambda: 0.5,
        mu_j: -0.05,
        sigma_j: 0.1,
      }),
      c_market.into(),
      s.into(),
      k.into(),
      r,
      q,
      tau,
      option_type,
      true,
    );

    let result = calibrator.calibrate(None).unwrap();
    println!(
      "SVJ result: v0={}, kappa={}, theta={}, sigma_v={}, rho={}, lambda={}, mu_j={}, sigma_j={}",
      result.v0,
      result.kappa,
      result.theta,
      result.sigma_v,
      result.rho,
      result.lambda,
      result.mu_j,
      result.sigma_j,
    );
    println!("Loss: {:?}", result.loss);
  }
}

//! # HKDE (Heston + Kou Double-Exponential) Calibration
//!
//! $$
//! \begin{aligned}
//! dS_t &= (r-q-\lambda\bar k)S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S + S_t(e^{J_t}-1)\,dN_t \\
//! dv_t &= \kappa(\theta-v_t)\,dt + \sigma_v\sqrt{v_t}\,dW_t^v, \quad
//! d\langle W^S,W^v\rangle_t = \rho\,dt
//! \end{aligned}
//! $$
//!
//! where $N_t$ is a Poisson process with intensity $\lambda$ and jump sizes
//! follow an asymmetric double-exponential distribution
//!
//! $$
//! f_J(y) = p\,\eta_1 e^{-\eta_1 y}\,\mathbb{1}_{y\ge 0}
//!        + (1-p)\,\eta_2 e^{\eta_2 y}\,\mathbb{1}_{y<0},\qquad \eta_1>1,\ \eta_2>0.
//! $$
//!
//! The characteristic function factorises as
//! $\phi_{\mathrm{HKDE}}(\xi,t) = \phi_{\mathrm{Hes}}(\xi,t)\,\phi_{\mathrm{Kou}}(\xi,t)$
//! and is provided by [`HKDEFourier`](crate::quant::pricing::fourier::HKDEFourier).
//!
//! The nine-dimensional parameter vector $\theta=(v_0,\kappa,\theta,\sigma_v,\rho,
//! \lambda,p,\eta_1,\eta_2)$ is calibrated against a set of market quotes by
//! minimising the vega-weighted least-squares objective (Eq. 12 of the paper):
//!
//! $$
//! \theta^{\*} = \arg\min_{\theta\in\Theta}
//! \sum_{n=1}^{N}\sum_{j=1}^{N_n} w_j^{(n)}\bigl(\mathcal V_j^{(n)}(\theta)-v_j^{(n)}\bigr)^2,
//! $$
//!
//! with vega weights (Eq. 13)
//!
//! $$
//! w_j^{(n)} := \bigl(S_0\,\psi(d_1)\,\sqrt{T_n}\bigr)^{-1},\qquad
//! d_1 = \frac{\log(S_0/K_j^{(n)}) + T_n\bigl(r_n-q_n+\tfrac12(\sigma_j^{(n)})^2\bigr)}
//!            {\sigma_j^{(n)}\sqrt{T_n}},
//! $$
//!
//! where $\psi$ is the standard-normal pdf and $\sigma_j^{(n)}$ is the market
//! implied volatility of the quote. The weights are computed once from the
//! observed market prices and kept fixed during the Levenberg-Marquardt
//! iterations.
//!
//! Source:
//! - Agazzotti, Aglieri Rinella, Aguilar, Kirkby (2025),
//!   "Calibration and Option Pricing with stochastic volatility and double
//!   exponential jumps", arXiv: 2502.13824
//! - Kou, S. G. (2002), "A jump-diffusion model for option pricing",
//!   Management Science 48(8), https://doi.org/10.1287/mnsc.48.8.1086.166
//! - Heston, S. L. (1993), "A closed-form solution for options with stochastic
//!   volatility", https://doi.org/10.1093/rfs/6.2.327

use std::cell::RefCell;
use std::f64::consts::PI;
use std::rc::Rc;

use levenberg_marquardt::LeastSquaresProblem;
use levenberg_marquardt::LevenbergMarquardt;
use nalgebra::DMatrix;
use nalgebra::DVector;
use nalgebra::Dyn;
use nalgebra::Owned;

use super::periodic_map;
use crate::quant::CalibrationLossScore;
use crate::quant::LossMetric;
use crate::quant::OptionType;
use crate::quant::calibration::CalibrationHistory;
use crate::quant::pricing::bsm::BSMCoc;
use crate::quant::pricing::bsm::BSMPricer;
use crate::quant::pricing::fourier::HKDEFourier;
use crate::traits::ModelPricer;
use crate::traits::PricerExt;

const EPS: f64 = 1e-8;
const RHO_BOUND: f64 = 0.9999;
const KAPPA_MIN: f64 = 1e-3;
const THETA_MIN: f64 = 1e-8;
const SIGMA_V_MIN: f64 = 1e-8;
/// Kou admissibility: $\eta_1>1$ is required for $\mathbb E[e^J]$ to be finite.
const ETA1_MIN: f64 = 1.0 + 1e-3;
const ETA2_MIN: f64 = 1e-3;

/// Box ranges used by the periodic projection.
///
/// The upper bounds were chosen to encompass the calibrated parameter sets
/// reported by Agazzotti et al. (2025), Table 1, which routinely produce
/// $\theta$ values above $0.4$ on single-name equities (e.g. SHOP: $\theta=0.728$).
const P_V0: (f64, f64) = (0.005, 0.25);
const P_KAPPA: (f64, f64) = (0.1, 20.0);
const P_THETA: (f64, f64) = (0.001, 1.0);
const P_SIGMA_V: (f64, f64) = (0.01, 1.0);
const P_RHO: (f64, f64) = (-1.0, 1.0);
const P_LAMBDA: (f64, f64) = (0.0, 10.0);
const P_P_UP: (f64, f64) = (0.001, 0.999);
const P_ETA1: (f64, f64) = (1.01, 50.0);
const P_ETA2: (f64, f64) = (0.1, 50.0);

/// HKDE model parameters — Heston stochastic volatility augmented by a
/// Kou double-exponential jump component.
#[derive(Clone, Copy, Debug)]
pub struct HKDEParams {
  /// Initial variance $v_0$.
  pub v0: f64,
  /// Mean-reversion speed $\kappa$.
  pub kappa: f64,
  /// Long-run variance $\theta$.
  pub theta: f64,
  /// Volatility of variance $\sigma_v$.
  pub sigma_v: f64,
  /// Correlation $\rho$ between the price and variance Brownian motions.
  pub rho: f64,
  /// Poisson jump intensity $\lambda$.
  pub lambda: f64,
  /// Probability $p$ of an upward jump.
  pub p_up: f64,
  /// Upward jump rate $\eta_1>1$.
  pub eta1: f64,
  /// Downward jump rate $\eta_2>0$.
  pub eta2: f64,
}

impl HKDEParams {
  /// Project the parameter vector onto the admissible set.
  ///
  /// The projection enforces the box bounds, the Feller condition
  /// $2\kappa\theta\ge\sigma_v^2$ and the Kou finiteness requirement
  /// $\eta_1>1$.
  pub fn project_in_place(&mut self) {
    self.v0 = periodic_map(self.v0, P_V0.0, P_V0.1).max(0.0);
    self.kappa = periodic_map(self.kappa, P_KAPPA.0, P_KAPPA.1).max(KAPPA_MIN);
    self.theta = periodic_map(self.theta, P_THETA.0, P_THETA.1).max(THETA_MIN);
    self.sigma_v = periodic_map(self.sigma_v, P_SIGMA_V.0, P_SIGMA_V.1)
      .abs()
      .max(SIGMA_V_MIN);
    self.rho = periodic_map(self.rho, P_RHO.0, P_RHO.1).clamp(-RHO_BOUND, RHO_BOUND);
    self.lambda = periodic_map(self.lambda, P_LAMBDA.0, P_LAMBDA.1).max(0.0);
    self.p_up = periodic_map(self.p_up, P_P_UP.0, P_P_UP.1).clamp(P_P_UP.0, P_P_UP.1);
    self.eta1 = periodic_map(self.eta1, P_ETA1.0, P_ETA1.1).max(ETA1_MIN);
    self.eta2 = periodic_map(self.eta2, P_ETA2.0, P_ETA2.1).max(ETA2_MIN);

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

  /// Return a copy with the admissibility projection applied.
  pub fn projected(mut self) -> Self {
    self.project_in_place();
    self
  }
}

impl From<HKDEParams> for DVector<f64> {
  fn from(p: HKDEParams) -> Self {
    DVector::from_vec(vec![
      p.v0, p.kappa, p.theta, p.sigma_v, p.rho, p.lambda, p.p_up, p.eta1, p.eta2,
    ])
  }
}

impl From<DVector<f64>> for HKDEParams {
  fn from(v: DVector<f64>) -> Self {
    HKDEParams {
      v0: v[0],
      kappa: v[1],
      theta: v[2],
      sigma_v: v[3],
      rho: v[4],
      lambda: v[5],
      p_up: v[6],
      eta1: v[7],
      eta2: v[8],
    }
  }
}

/// Calibration result for the HKDE model.
#[derive(Clone, Debug)]
pub struct HKDECalibrationResult {
  pub v0: f64,
  pub kappa: f64,
  pub theta: f64,
  pub sigma_v: f64,
  pub rho: f64,
  pub lambda: f64,
  pub p_up: f64,
  pub eta1: f64,
  pub eta2: f64,
  /// Calibration loss metrics on the unweighted price residuals.
  pub loss: CalibrationLossScore,
  /// Whether the optimiser converged.
  pub converged: bool,
}

impl crate::traits::ToModel for HKDECalibrationResult {
  fn to_model(&self, r: f64, q: f64) -> Box<dyn crate::traits::ModelPricer> {
    Box::new(HKDECalibrationResult::to_model(self, r, q))
  }
}

impl HKDECalibrationResult {
  /// Convert to a [`HKDEFourier`] model for pricing / vol-surface generation.
  pub fn to_model(&self, r: f64, q: f64) -> HKDEFourier {
    HKDEFourier {
      v0: self.v0,
      kappa: self.kappa,
      theta: self.theta,
      sigma_v: self.sigma_v,
      rho: self.rho,
      r,
      q,
      lam: self.lambda,
      p_up: self.p_up,
      eta1: self.eta1,
      eta2: self.eta2,
    }
  }
}

/// HKDE least-squares calibrator using Levenberg-Marquardt.
///
/// Source:
/// - Levenberg (1944), https://doi.org/10.1090/qam/10666
/// - Marquardt (1963), https://doi.org/10.1137/0111030
/// - Agazzotti et al. (2025), arXiv: 2502.13824
#[derive(Clone)]
pub struct HKDECalibrator {
  /// Current parameter iterate. `None` triggers the fallback initial guess.
  pub params: Option<HKDEParams>,
  /// Market option prices (flattened across all maturities).
  pub c_market: DVector<f64>,
  /// Underlying spot per quote.
  pub s: DVector<f64>,
  /// Strike per quote.
  pub k: DVector<f64>,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield.
  pub q: Option<f64>,
  /// Time to maturity per quote (flattened). Supports joint multi-maturity
  /// calibration.
  pub flat_t: Vec<f64>,
  /// Option type.
  pub option_type: OptionType,
  /// Precomputed vega weights $\sqrt{w_j^{(n)}}$ applied to each residual
  /// (Eq. 13 of the paper). Length equals the number of quotes.
  pub sqrt_weights: Vec<f64>,
  /// If true, record per-iteration calibration history.
  pub record_history: bool,
  /// Which loss metrics to compute when recording history.
  pub loss_metrics: &'static [LossMetric],
  calibration_history: Rc<RefCell<Vec<CalibrationHistory<HKDEParams>>>>,
}

impl HKDECalibrator {
  /// Create a calibrator for a single maturity slice.
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    params: Option<HKDEParams>,
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

    let flat_t = vec![tau; n];
    let q_val = q.unwrap_or(0.0);
    let sqrt_weights = compute_sqrt_weights(
      s.as_slice(),
      k.as_slice(),
      &flat_t,
      r,
      q_val,
      c_market.as_slice(),
      option_type,
    );

    Self {
      params,
      c_market,
      s,
      k,
      r,
      q,
      flat_t,
      option_type,
      sqrt_weights,
      record_history,
      loss_metrics: &LossMetric::ALL,
      calibration_history: Rc::new(RefCell::new(Vec::new())),
    }
  }

  /// Create a calibrator from multiple maturity slices for joint surface
  /// calibration. Mirrors the API of the Heston / SVJ / BSM calibrators.
  pub fn from_slices(
    params: Option<HKDEParams>,
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

    let q_val = q.unwrap_or(0.0);
    let sqrt_weights = compute_sqrt_weights(
      &flat_s,
      &flat_strikes,
      &flat_t,
      r,
      q_val,
      &flat_prices,
      option_type,
    );

    Self {
      params,
      c_market: DVector::from_vec(flat_prices),
      s: DVector::from_vec(flat_s),
      k: DVector::from_vec(flat_strikes),
      r,
      q,
      flat_t,
      option_type,
      sqrt_weights,
      record_history,
      loss_metrics: &LossMetric::ALL,
      calibration_history: Rc::new(RefCell::new(Vec::new())),
    }
  }

  /// Run the calibration. If `initial_params` is `Some`, it overrides the
  /// calibrator's current initial guess.
  pub fn calibrate(&self, initial_params: Option<HKDEParams>) -> HKDECalibrationResult {
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

    HKDECalibrationResult {
      v0: p.v0,
      kappa: p.kappa,
      theta: p.theta,
      sigma_v: p.sigma_v,
      rho: p.rho,
      lambda: p.lambda,
      p_up: p.p_up,
      eta1: p.eta1,
      eta2: p.eta2,
      loss,
      converged: report.termination.was_successful(),
    }
  }

  pub fn set_initial_guess(&mut self, params: HKDEParams) {
    self.params = Some(params.projected());
  }

  pub fn set_record_history(&mut self, record: bool) {
    self.record_history = record;
  }

  pub fn history(&self) -> Vec<CalibrationHistory<HKDEParams>> {
    self.calibration_history.borrow().clone()
  }

  fn fallback_params() -> HKDEParams {
    HKDEParams {
      v0: 0.04,
      kappa: 1.5,
      theta: 0.04,
      sigma_v: 0.3,
      rho: -0.5,
      lambda: 0.5,
      p_up: 0.4,
      eta1: 10.0,
      eta2: 5.0,
    }
    .projected()
  }

  fn ensure_initial_guess(&mut self) {
    if self.params.is_none() {
      self.params = Some(Self::fallback_params());
    }
  }

  fn effective_params(&self) -> HKDEParams {
    if let Some(p) = &self.params {
      return (*p).projected();
    }
    Self::fallback_params()
  }

  fn build_model(&self, p: &HKDEParams) -> HKDEFourier {
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

  fn compute_model_prices_for(&self, p: &HKDEParams) -> DVector<f64> {
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

  fn weighted_residuals_for(&self, p: &HKDEParams) -> DVector<f64> {
    let c_model = self.compute_model_prices_for(p);
    let n = self.c_market.len();
    let mut r = DVector::zeros(n);
    for i in 0..n {
      r[i] = self.sqrt_weights[i] * (c_model[i] - self.c_market[i]);
    }
    r
  }

  /// Central finite-difference Jacobian of the weighted residuals.
  fn numeric_jacobian(&self, params: &HKDEParams) -> DMatrix<f64> {
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
fn compute_sqrt_weights(
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

/// Reference calibrated HKDE parameters from Agazzotti et al. (2025), Table 1.
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
/// These values correspond to HKDE calibrated against market option data on
/// 2024-02-20 and are reproduced here as a documentation / sanity aid.
/// Reaching them exactly requires the underlying proprietary quotes.
#[allow(dead_code)]
pub mod paper_table2 {
  /// HKDE mean absolute percentage error per ticker, Table 2.
  pub const MAPE: [(&str, f64); 4] = [
    ("AMZN", 0.0261),
    ("NFLX", 0.0488),
    ("SHOP", 0.0266),
    ("SPOT", 0.0339),
  ];
  /// HKDE root-mean-square error per ticker, Table 2.
  pub const RMSE: [(&str, f64); 4] = [
    ("AMZN", 0.01433),
    ("NFLX", 0.10048),
    ("SHOP", 0.02938),
    ("SPOT", 0.03173),
  ];
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Analytical reference Heston prices at
  /// (v0=0.04, kappa=1.5, theta=0.04, sigma_v=0.3, rho=-0.7),
  /// S=100, r=0.05, q=0, T=1.
  const HESTON_REF: [f64; 9] = [
    25.095178, 20.976171, 17.106937, 13.548230, 10.361869, 7.604362, 5.317953, 3.519953, 2.193310,
  ];
  const STRIKES: [f64; 9] = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0];

  fn ref_hkde_params() -> HKDEParams {
    HKDEParams {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma_v: 0.3,
      rho: -0.7,
      lambda: 2.0,
      p_up: 0.4,
      eta1: 10.0,
      eta2: 5.0,
    }
  }

  #[test]
  fn hkde_projection_enforces_eta1_above_one() {
    let mut p = ref_hkde_params();
    p.eta1 = 0.5;
    p.project_in_place();
    assert!(p.eta1 > 1.0, "eta1={} should be > 1", p.eta1);
  }

  #[test]
  fn hkde_projection_enforces_feller() {
    let mut p = ref_hkde_params();
    p.kappa = 0.5;
    p.theta = 0.01;
    p.sigma_v = 0.9;
    p.project_in_place();
    assert!(
      2.0 * p.kappa * p.theta >= p.sigma_v * p.sigma_v - 1e-10,
      "Feller violated after projection: 2*k*theta={}, sigma_v^2={}",
      2.0 * p.kappa * p.theta,
      p.sigma_v * p.sigma_v
    );
  }

  #[test]
  fn hkde_calibrate_recovers_heston_prices() {
    // HKDE with lam=0 must reproduce Heston prices. We use the analytical
    // Heston reference as synthetic market data and verify the calibrator
    // finds parameters that match it closely.
    let n = STRIKES.len();
    let calibrator = HKDECalibrator::new(
      Some(HKDEParams {
        v0: 0.05,
        kappa: 1.8,
        theta: 0.05,
        sigma_v: 0.4,
        rho: -0.6,
        lambda: 0.05,
        p_up: 0.5,
        eta1: 10.0,
        eta2: 10.0,
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
    // 9 parameters, 9 data points — expect a good fit.
    assert!(
      result.loss.get(LossMetric::Rmse) < 0.5,
      "HKDE→Heston RMSE={:.6}",
      result.loss.get(LossMetric::Rmse)
    );
  }

  #[test]
  fn hkde_calibrate_self_consistency() {
    // Generate synthetic market prices from a known HKDE model, then
    // verify the calibrator recovers a loss close to zero when started
    // from a perturbed initial guess.
    let truth = ref_hkde_params();
    let model = HKDEFourier {
      v0: truth.v0,
      kappa: truth.kappa,
      theta: truth.theta,
      sigma_v: truth.sigma_v,
      rho: truth.rho,
      r: 0.03,
      q: 0.0,
      lam: truth.lambda,
      p_up: truth.p_up,
      eta1: truth.eta1,
      eta2: truth.eta2,
    };

    let s_val = 100.0;
    let r = 0.03;
    let q = 0.0;
    let tau = 0.75;
    let market: Vec<f64> = STRIKES
      .iter()
      .map(|&k| model.price_call(s_val, k, r, q, tau).max(0.0))
      .collect();

    let initial = HKDEParams {
      v0: 0.05,
      kappa: 1.5,
      theta: 0.05,
      sigma_v: 0.25,
      rho: -0.5,
      lambda: 1.0,
      p_up: 0.5,
      eta1: 8.0,
      eta2: 6.0,
    };

    let calibrator = HKDECalibrator::new(
      Some(initial),
      market.into(),
      vec![s_val; STRIKES.len()].into(),
      STRIKES.to_vec().into(),
      r,
      Some(q),
      tau,
      OptionType::Call,
      false,
    );
    let result = calibrator.calibrate(None);
    assert!(
      result.loss.get(LossMetric::Rmse) < 1.0,
      "HKDE self-consistency RMSE={:.6}",
      result.loss.get(LossMetric::Rmse)
    );
  }

  #[test]
  fn hkde_calibrate_from_slices_multi_maturity() {
    // Joint calibration across three maturity slices. Truth is a known
    // HKDE model; we verify the calibrator produces a small aggregate
    // pricing error when started from a perturbed guess.
    use crate::quant::calibration::levy::MarketSlice;

    let truth = ref_hkde_params();
    let r = 0.02;
    let q = 0.0;
    let s0 = 100.0;
    let make_model = || HKDEFourier {
      v0: truth.v0,
      kappa: truth.kappa,
      theta: truth.theta,
      sigma_v: truth.sigma_v,
      rho: truth.rho,
      r,
      q,
      lam: truth.lambda,
      p_up: truth.p_up,
      eta1: truth.eta1,
      eta2: truth.eta2,
    };
    let slice_strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];

    let make_slice = |t: f64| -> MarketSlice {
      let model = make_model();
      let prices: Vec<f64> = slice_strikes
        .iter()
        .map(|&k| model.price_call(s0, k, r, q, t).max(0.0))
        .collect();
      MarketSlice {
        strikes: slice_strikes.clone(),
        prices,
        is_call: vec![true; slice_strikes.len()],
        t,
      }
    };

    let slices = vec![make_slice(0.25), make_slice(0.5), make_slice(1.0)];

    let calibrator = HKDECalibrator::from_slices(
      Some(HKDEParams {
        v0: 0.05,
        kappa: 1.8,
        theta: 0.05,
        sigma_v: 0.25,
        rho: -0.5,
        lambda: 1.5,
        p_up: 0.5,
        eta1: 12.0,
        eta2: 6.0,
      }),
      &slices,
      s0,
      r,
      Some(q),
      OptionType::Call,
      false,
    );
    let result = calibrator.calibrate(None);
    assert!(
      result.loss.get(LossMetric::Rmse) < 1.0,
      "HKDE multi-maturity RMSE={:.6}",
      result.loss.get(LossMetric::Rmse)
    );
  }

  /// The SHOP parameter set from Agazzotti et al. (2025), Table 1, is fully
  /// admissible. Check that the projection is a no-op on it: this guarantees
  /// that the published Table 1 numbers can be used verbatim as inputs to
  /// our calibrator / pricer without silent clipping.
  #[test]
  fn paper_table1_shop_is_admissible() {
    let original = paper_table1::SHOP;
    let projected = original.projected();
    let eps = 1e-12;
    assert!((projected.v0 - original.v0).abs() < eps);
    assert!((projected.kappa - original.kappa).abs() < eps);
    assert!((projected.theta - original.theta).abs() < eps);
    assert!((projected.sigma_v - original.sigma_v).abs() < eps);
    assert!((projected.rho - original.rho).abs() < eps);
    assert!((projected.lambda - original.lambda).abs() < eps);
    assert!((projected.p_up - original.p_up).abs() < eps);
    assert!((projected.eta1 - original.eta1).abs() < eps);
    assert!((projected.eta2 - original.eta2).abs() < eps);
    assert!(
      2.0 * projected.kappa * projected.theta >= projected.sigma_v * projected.sigma_v,
      "Feller should hold for SHOP"
    );
  }

  /// Self-consistency test against the published SHOP calibration
  /// (Agazzotti et al. 2025, Table 1). We generate synthetic market prices
  /// from the paper's SHOP parameters and verify the calibrator recovers
  /// them starting from a perturbed guess.
  #[test]
  fn paper_table1_shop_self_consistency() {
    let truth = paper_table1::SHOP;
    let r = 0.05;
    let q = 0.0;
    let s0 = 100.0;
    let tau = 0.5;
    let strikes = [80.0_f64, 90.0, 100.0, 110.0, 120.0];

    let model = HKDEFourier {
      v0: truth.v0,
      kappa: truth.kappa,
      theta: truth.theta,
      sigma_v: truth.sigma_v,
      rho: truth.rho,
      r,
      q,
      lam: truth.lambda,
      p_up: truth.p_up,
      eta1: truth.eta1,
      eta2: truth.eta2,
    };
    let market: Vec<f64> = strikes
      .iter()
      .map(|&k| model.price_call(s0, k, r, q, tau).max(0.0))
      .collect();

    let initial = HKDEParams {
      v0: 0.12,
      kappa: 0.3,
      theta: 0.6,
      sigma_v: 0.25,
      rho: -0.55,
      lambda: 0.8,
      p_up: 0.9,
      eta1: 10.0,
      eta2: 1.0,
    };

    let calibrator = HKDECalibrator::new(
      Some(initial),
      market.clone().into(),
      vec![s0; strikes.len()].into(),
      strikes.to_vec().into(),
      r,
      Some(q),
      tau,
      OptionType::Call,
      false,
    );
    let result = calibrator.calibrate(None);
    // The weighted-least-squares landscape is multi-modal; we do not
    // demand the exact truth vector back, only a small residual pricing
    // error across the smile.
    assert!(
      result.loss.get(LossMetric::Rmse) < 0.5,
      "SHOP self-consistency RMSE={:.6}",
      result.loss.get(LossMetric::Rmse)
    );
    // Refit once from the recovered point to confirm convergence is stable.
    let refined = HKDECalibrator::new(
      Some(HKDEParams {
        v0: result.v0,
        kappa: result.kappa,
        theta: result.theta,
        sigma_v: result.sigma_v,
        rho: result.rho,
        lambda: result.lambda,
        p_up: result.p_up,
        eta1: result.eta1,
        eta2: result.eta2,
      }),
      market.into(),
      vec![s0; strikes.len()].into(),
      strikes.to_vec().into(),
      r,
      Some(q),
      tau,
      OptionType::Call,
      false,
    )
    .calibrate(None);
    assert!(
      refined.loss.get(LossMetric::Rmse) <= result.loss.get(LossMetric::Rmse) + 1e-6,
      "refit should not degrade the fit"
    );
  }
}

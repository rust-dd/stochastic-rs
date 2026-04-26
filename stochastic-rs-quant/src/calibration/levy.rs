//! # Lévy Model Calibration
//!
//! $$
//! \hat\theta=\arg\min_\theta\sum_i\left(C_i^{\mathrm{model}}(\theta)-C_i^{\mathrm{mkt}}\right)^2
//! $$
//!
//! Calibrates Lévy process parameters to observed option prices using
//! characteristic-function-based pricing and Levenberg-Marquardt optimisation.
//!
//! Supported models:
//! - Variance Gamma (VG)
//! - Normal Inverse Gaussian (NIG)
//! - CGMY
//! - Merton Jump-Diffusion
//! - Kou Double-Exponential Jump-Diffusion
//!
//! Source:
//! - Carr, P. & Madan, D. (1999), "Option valuation using the fast Fourier transform"
//!   https://doi.org/10.1016/S0165-1889(98)00038-5
//! - Madan, D., Carr, P. & Chang, E. (1998), "The Variance Gamma Process and Option Pricing"
//!   https://doi.org/10.1023/A:1009703431535

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
use crate::CalibrationLossScore;
use crate::LossMetric;
use crate::calibration::CalibrationHistory;

const EPS: f64 = 1e-8;

/// Supported Lévy model types for calibration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LevyModelType {
  /// Variance Gamma: $\psi(\xi)=-\frac{1}{\nu}\ln\!\bigl(1-i\theta\nu\xi+\tfrac12\sigma^2\nu\xi^2\bigr)$
  VarianceGamma,
  /// Normal Inverse Gaussian: $\psi(\xi)=\delta\bigl(\sqrt{\alpha^2-\beta^2}-\sqrt{\alpha^2-(\beta+i\xi)^2}\bigr)$
  NIG,
  /// CGMY: $\psi(\xi)=C\,\Gamma(-Y)\bigl[(M-i\xi)^Y-M^Y+(G+i\xi)^Y-G^Y\bigr]$
  CGMY,
  /// Merton Jump-Diffusion: $\psi(\xi)=-\tfrac12\sigma^2\xi^2+\lambda\bigl(e^{i\mu_J\xi-\frac12\sigma_J^2\xi^2}-1\bigr)$
  MertonJD,
  /// Kou Double-Exponential: $\psi(\xi)=-\tfrac12\sigma^2\xi^2+\lambda\bigl(\frac{p\eta_1}{\eta_1-i\xi}+\frac{(1-p)\eta_2}{\eta_2+i\xi}-1\bigr)$
  Kou,
}

/// Market data for a single maturity slice.
#[derive(Clone, Debug)]
pub struct MarketSlice {
  /// Strike prices.
  pub strikes: Vec<f64>,
  /// Market option prices.
  pub prices: Vec<f64>,
  /// `true` for call, `false` for put.
  pub is_call: Vec<bool>,
  /// Time to maturity in years.
  pub t: f64,
}

/// Calibration result for a Lévy model.
#[derive(Clone, Debug)]
pub struct LevyCalibrationResult {
  /// Calibrated parameter vector.
  pub params: Vec<f64>,
  /// Model type that was calibrated.
  pub model_type: LevyModelType,
  /// Calibration loss metrics.
  pub loss: CalibrationLossScore,
  /// Whether the optimiser converged.
  pub converged: bool,
  /// Number of LM iterations performed.
  pub iterations: usize,
}

/// Variant-dispatching wrapper around the five Lévy Fourier pricers.
///
/// Used by [`LevyCalibrationResult::to_model`] so the result remains a
/// concrete `ModelPricer` (no `Box<dyn>`).
#[derive(Clone, Debug)]
pub enum LevyModel {
  VarianceGamma(crate::pricing::fourier::VarianceGammaFourier),
  Nig(crate::pricing::fourier::CGMYFourier),
  Cgmy(crate::pricing::fourier::CGMYFourier),
  MertonJd(crate::pricing::fourier::MertonJDFourier),
  Kou(crate::pricing::fourier::KouFourier),
}

impl crate::traits::ModelPricer for LevyModel {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    match self {
      LevyModel::VarianceGamma(m) => m.price_call(s, k, r, q, tau),
      LevyModel::Nig(m) => m.price_call(s, k, r, q, tau),
      LevyModel::Cgmy(m) => m.price_call(s, k, r, q, tau),
      LevyModel::MertonJd(m) => m.price_call(s, k, r, q, tau),
      LevyModel::Kou(m) => m.price_call(s, k, r, q, tau),
    }
  }
}

impl crate::traits::ToModel for LevyCalibrationResult {
  type Model = LevyModel;
  fn to_model(&self, r: f64, q: f64) -> Self::Model {
    LevyCalibrationResult::to_model(self, r, q)
  }
}

impl LevyCalibrationResult {
  /// Convert to a Fourier model for pricing / vol surface generation.
  pub fn to_model(&self, r: f64, q: f64) -> LevyModel {
    use crate::pricing::fourier::*;
    let p = &self.params;
    match self.model_type {
      LevyModelType::VarianceGamma => LevyModel::VarianceGamma(VarianceGammaFourier {
        sigma: p[0],
        theta: p[1],
        nu: p[2],
        r,
        q,
      }),
      LevyModelType::NIG => LevyModel::Nig(CGMYFourier {
        c: p[0],
        g: p[1],
        m: p[2],
        y: 0.5,
        r,
        q,
      }),
      LevyModelType::CGMY => LevyModel::Cgmy(CGMYFourier {
        c: p[0],
        g: p[1],
        m: p[2],
        y: p[3],
        r,
        q,
      }),
      LevyModelType::MertonJD => LevyModel::MertonJd(MertonJDFourier {
        sigma: p[0],
        lambda: p[1],
        mu_j: p[2],
        sigma_j: p[3],
        r,
        q,
      }),
      LevyModelType::Kou => LevyModel::Kou(KouFourier {
        sigma: p[0],
        lambda: p[1],
        p_up: p[2],
        eta1: p[3],
        eta2: p[4],
        r,
        q,
      }),
    }
  }
}

/// Lévy model calibrator via Fourier pricing + Levenberg-Marquardt.
///
/// Source:
/// - Levenberg (1944), https://doi.org/10.1090/qam/10666
/// - Marquardt (1963), https://doi.org/10.1137/0111030
#[derive(Clone)]
pub struct LevyCalibrator {
  /// Lévy model to calibrate.
  pub model_type: LevyModelType,
  /// Underlying spot price.
  pub s: f64,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield.
  pub q: f64,
  /// Market data slices (one per maturity).
  pub market_data: Vec<MarketSlice>,
  /// If true, record per-iteration calibration history.
  pub record_history: bool,
  /// Internal: current parameter vector (set by LM).
  params: Vec<f64>,
  /// Internal: flattened market prices for the residual vector.
  flat_prices: Vec<f64>,
  /// Internal: flattened strikes.
  flat_strikes: Vec<f64>,
  /// Internal: flattened maturities.
  flat_t: Vec<f64>,
  /// Internal: flattened is_call flags.
  flat_is_call: Vec<bool>,
  /// Which loss metrics to compute when recording history.
  pub loss_metrics: &'static [LossMetric],
  /// History of iterations.
  calibration_history: Rc<RefCell<Vec<CalibrationHistory<Vec<f64>>>>>,
}

/// Compute the Lévy characteristic exponent $\psi(\xi)$ such that
/// $\phi_T(\xi) = \exp\bigl(i\xi (r-q)T + T\,\psi(\xi) - T\,\psi(-i)\bigr)$.
///
/// The martingale correction $-T\,\psi(-i)$ ensures that $E[S_T] = S_0 e^{(r-q)T}$.
fn levy_char_exponent(model_type: LevyModelType, params: &[f64], xi: Complex64) -> Complex64 {
  let i = Complex64::i();
  match model_type {
    LevyModelType::VarianceGamma => {
      // params: [sigma, theta, nu]
      let sigma = params[0];
      let theta = params[1];
      let nu = params[2];
      // psi(xi) = -(1/nu) * ln(1 - i*theta*nu*xi + 0.5*sigma^2*nu*xi^2)
      let inner = Complex64::new(1.0, 0.0) - i * theta * nu * xi
        + Complex64::new(0.5 * sigma * sigma * nu, 0.0) * xi * xi;
      -inner.ln() / nu
    }
    LevyModelType::NIG => {
      // params: [alpha, beta, delta]
      let alpha = params[0];
      let beta = params[1];
      let delta = params[2];
      // psi(xi) = delta * (sqrt(alpha^2 - beta^2) - sqrt(alpha^2 - (beta + i*xi)^2))
      let a2 = alpha * alpha;
      let b2 = beta * beta;
      let base = Complex64::new(a2 - b2, 0.0).sqrt();
      let shifted = Complex64::new(beta, 0.0) + i * xi;
      let branch = (Complex64::new(a2, 0.0) - shifted * shifted).sqrt();
      delta * (base - branch)
    }
    LevyModelType::CGMY => {
      // params: [C, G, M, Y]
      let c = params[0];
      let g = params[1];
      let m = params[2];
      let y = params[3];
      // psi(xi) = C * Gamma(-Y) * [(M - i*xi)^Y - M^Y + (G + i*xi)^Y - G^Y]
      let gamma_neg_y = gamma_neg_y_fn(y);
      let m_shift = (Complex64::new(m, 0.0) - i * xi).powf(y);
      let g_shift = (Complex64::new(g, 0.0) + i * xi).powf(y);
      let m_y = Complex64::new(m.powf(y), 0.0);
      let g_y = Complex64::new(g.powf(y), 0.0);
      c * gamma_neg_y * (m_shift - m_y + g_shift - g_y)
    }
    LevyModelType::MertonJD => {
      // params: [sigma, lambda, mu_j, sigma_j]
      let sigma = params[0];
      let lambda = params[1];
      let mu_j = params[2];
      let sigma_j = params[3];
      // psi(xi) = -0.5*sigma^2*xi^2 + lambda*(exp(i*mu_j*xi - 0.5*sigma_j^2*xi^2) - 1)
      let diffusion = Complex64::new(-0.5 * sigma * sigma, 0.0) * xi * xi;
      let jump_exp = (i * mu_j * xi - Complex64::new(0.5 * sigma_j * sigma_j, 0.0) * xi * xi).exp();
      diffusion + lambda * (jump_exp - 1.0)
    }
    LevyModelType::Kou => {
      // params: [sigma, lambda, p_up, eta1, eta2]
      let sigma = params[0];
      let lambda = params[1];
      let p = params[2];
      let eta1 = params[3];
      let eta2 = params[4];
      // psi(xi) = -0.5*sigma^2*xi^2 + lambda*(p*eta1/(eta1 - i*xi) + (1-p)*eta2/(eta2 + i*xi) - 1)
      let diffusion = Complex64::new(-0.5 * sigma * sigma, 0.0) * xi * xi;
      let up = p * eta1 / (Complex64::new(eta1, 0.0) - i * xi);
      let dn = (1.0 - p) * eta2 / (Complex64::new(eta2, 0.0) + i * xi);
      diffusion + lambda * (up + dn - 1.0)
    }
  }
}

/// $\Gamma(-Y)$ for CGMY, using reflection formula.
fn gamma_neg_y_fn(y: f64) -> Complex64 {
  // For Y < 0, Gamma(-Y) = Gamma(|Y|) is real positive, straightforward.
  // For 0 < Y < 2, Y != 1, use reflection: Gamma(-Y) = -pi / (Y * sin(pi*Y) * Gamma(Y))
  if y.abs() < EPS {
    // Y ≈ 0: Gamma(0) is ±∞, but C*Gamma(-Y) remains finite via limiting form.
    // Use a large approximation.
    return Complex64::new(1e15, 0.0);
  }
  if y < 0.0 {
    Complex64::new(statrs::function::gamma::gamma(-y), 0.0)
  } else if (y - 1.0).abs() < EPS {
    // Y = 1 is a pole; clamp to nearby value.
    Complex64::new(statrs::function::gamma::gamma(-0.999), 0.0)
  } else {
    // Reflection: Gamma(-Y) = -pi / (Y * sin(pi*Y) * Gamma(Y))
    let g = statrs::function::gamma::gamma(y);
    let sin_val = (std::f64::consts::PI * y).sin();
    if sin_val.abs() < EPS || g.abs() < EPS {
      Complex64::new(1e15, 0.0)
    } else {
      Complex64::new(-std::f64::consts::PI / (y * sin_val * g), 0.0)
    }
  }
}

/// Price a European call option using the characteristic function and
/// Gauss-Legendre quadrature over the Gil-Pelaez inversion integral.
///
/// $$
/// C = S e^{-qT} P_1 - K e^{-rT} P_2
/// $$
///
/// where $P_j = \frac{1}{2} + \frac{1}{\pi}\int_0^\infty \mathrm{Re}\!\bigl[\cdots\bigr]\,du$.
fn fourier_call_price(
  model_type: LevyModelType,
  params: &[f64],
  s: f64,
  k: f64,
  r: f64,
  q: f64,
  t: f64,
) -> f64 {
  let (nodes, weights) = gauss_legendre_64();
  let scale = 0.5 * GL_U_MAX;
  let ln_s = s.ln();
  let ln_k = k.ln();
  let rq = r - q;

  // Martingale correction: omega = -psi(-i) so that E[S_T] = S_0*exp((r-q)*T)
  let psi_neg_i = levy_char_exponent(model_type, params, Complex64::new(0.0, -1.0));
  let omega = -psi_neg_i;

  let mut i1 = 0.0_f64;
  let mut i2 = 0.0_f64;

  for (&x, &w) in nodes.iter().zip(weights.iter()) {
    let u = scale * (x + 1.0);
    let w_s = scale * w;
    if u <= EPS {
      continue;
    }

    let xi = Complex64::new(u, 0.0);
    let psi = levy_char_exponent(model_type, params, xi);

    // Full log-characteristic function: i*xi*(ln(S) + (r-q+omega)*T) + psi(xi)*T
    let log_cf = Complex64::i() * xi * (ln_s + (rq + omega.re) * t) + psi * t;
    let phi = log_cf.exp();

    // Shifted for P1: xi -> xi - i
    let xi_shift = Complex64::new(u, -1.0);
    let psi_shift = levy_char_exponent(model_type, params, xi_shift);
    let log_cf_shift = Complex64::i() * xi_shift * (ln_s + (rq + omega.re) * t) + psi_shift * t;
    let phi_shift = log_cf_shift.exp();

    let kernel = (Complex64::new(0.0, -u * ln_k)).exp() / (Complex64::i() * xi);

    i1 += w_s * (kernel * phi_shift).re;
    i2 += w_s * (kernel * phi).re;
  }

  let disc_r = (-r * t).exp();
  let disc_q = (-q * t).exp();

  let call = 0.5 * (s * disc_q - k * disc_r) + disc_r * FRAC_1_PI * (i1 - k * i2);
  call.max(0.0)
}

/// Price a European option (call or put) via put-call parity from the Fourier call price.
fn fourier_option_price(
  model_type: LevyModelType,
  params: &[f64],
  s: f64,
  k: f64,
  r: f64,
  q: f64,
  t: f64,
  is_call: bool,
) -> f64 {
  let call = fourier_call_price(model_type, params, s, k, r, q, t);
  if is_call {
    call
  } else {
    // Put-call parity: P = C - S*exp(-q*T) + K*exp(-r*T)
    let put = call - s * (-q * t).exp() + k * (-r * t).exp();
    put.max(0.0)
  }
}

fn param_count(model_type: LevyModelType) -> usize {
  match model_type {
    LevyModelType::VarianceGamma => 3, // sigma, theta, nu
    LevyModelType::NIG => 3,           // alpha, beta, delta
    LevyModelType::CGMY => 4,          // C, G, M, Y
    LevyModelType::MertonJD => 4,      // sigma, lambda, mu_j, sigma_j
    LevyModelType::Kou => 5,           // sigma, lambda, p_up, eta1, eta2
  }
}

fn param_bounds(model_type: LevyModelType) -> Vec<(f64, f64)> {
  match model_type {
    LevyModelType::VarianceGamma => {
      vec![(0.01, 2.0), (-1.0, 1.0), (0.01, 5.0)]
    }
    LevyModelType::NIG => {
      vec![(0.01, 50.0), (-50.0, 50.0), (0.001, 5.0)]
    }
    LevyModelType::CGMY => {
      vec![(0.001, 10.0), (0.01, 50.0), (0.01, 50.0), (-1.0, 1.999)]
    }
    LevyModelType::MertonJD => {
      vec![(0.01, 2.0), (0.01, 20.0), (-1.0, 1.0), (0.01, 2.0)]
    }
    LevyModelType::Kou => {
      vec![
        (0.01, 2.0),
        (0.01, 20.0),
        (0.01, 0.99),
        (1.0, 100.0),
        (1.0, 100.0),
      ]
    }
  }
}

fn default_params(model_type: LevyModelType) -> Vec<f64> {
  match model_type {
    LevyModelType::VarianceGamma => vec![0.2, -0.1, 0.5],
    LevyModelType::NIG => vec![15.0, -5.0, 0.5],
    LevyModelType::CGMY => vec![1.0, 10.0, 15.0, 0.5],
    LevyModelType::MertonJD => vec![0.15, 1.0, -0.05, 0.1],
    LevyModelType::Kou => vec![0.15, 3.0, 0.5, 10.0, 10.0],
  }
}

/// Project parameters into their admissible bounds.
fn project_params(model_type: LevyModelType, params: &mut [f64]) {
  let bounds = param_bounds(model_type);
  for (p, (lo, hi)) in params.iter_mut().zip(bounds.iter()) {
    *p = p.clamp(*lo, *hi);
  }
  // NIG additional constraint: alpha > |beta|
  if model_type == LevyModelType::NIG {
    let alpha = params[0];
    let beta = params[1];
    if alpha <= beta.abs() {
      params[0] = beta.abs() + 0.01;
    }
  }
}

impl LevyCalibrator {
  pub fn new(
    model_type: LevyModelType,
    s: f64,
    r: f64,
    q: f64,
    market_data: Vec<MarketSlice>,
  ) -> Self {
    let (flat_prices, flat_strikes, flat_t, flat_is_call) = Self::flatten(&market_data);
    let params = default_params(model_type);

    Self {
      model_type,
      s,
      r,
      q,
      market_data,
      record_history: false,
      params,
      flat_prices,
      flat_strikes,
      flat_t,
      flat_is_call,
      loss_metrics: &LossMetric::ALL,
      calibration_history: Rc::new(RefCell::new(Vec::new())),
    }
  }

  fn flatten(market_data: &[MarketSlice]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<bool>) {
    let mut prices = Vec::new();
    let mut strikes = Vec::new();
    let mut ts = Vec::new();
    let mut is_call = Vec::new();
    for slice in market_data {
      for i in 0..slice.strikes.len() {
        prices.push(slice.prices[i]);
        strikes.push(slice.strikes[i]);
        ts.push(slice.t);
        is_call.push(slice.is_call[i]);
      }
    }
    (prices, strikes, ts, is_call)
  }

  pub fn set_record_history(&mut self, record: bool) {
    self.record_history = record;
  }

  pub fn history(&self) -> Vec<CalibrationHistory<Vec<f64>>> {
    self.calibration_history.borrow().clone()
  }

  /// Run the calibration, returning the result.
  pub fn calibrate(&self, initial_params: Option<Vec<f64>>) -> LevyCalibrationResult {
    let mut problem = self.clone();
    if let Some(p) = initial_params {
      assert_eq!(
        p.len(),
        param_count(self.model_type),
        "initial_params length must match model parameter count"
      );
      problem.params = p;
    }
    project_params(problem.model_type, &mut problem.params);

    let (result, report) = LevenbergMarquardt::new().minimize(problem);

    let final_params = result.params.clone();
    let c_model = result.compute_model_prices();
    let loss =
      CalibrationLossScore::compute_selected(&result.flat_prices, &c_model, result.loss_metrics);

    LevyCalibrationResult {
      params: final_params,
      model_type: self.model_type,
      loss,
      converged: report.termination.was_successful(),
      iterations: report.number_of_evaluations,
    }
  }

  fn compute_model_prices(&self) -> Vec<f64> {
    (0..self.flat_prices.len())
      .map(|i| {
        fourier_option_price(
          self.model_type,
          &self.params,
          self.s,
          self.flat_strikes[i],
          self.r,
          self.q,
          self.flat_t[i],
          self.flat_is_call[i],
        )
      })
      .collect()
  }

  fn effective_params(&self) -> Vec<f64> {
    let mut p = self.params.clone();
    project_params(self.model_type, &mut p);
    p
  }

  fn numeric_jacobian(&self) -> DMatrix<f64> {
    let n = self.flat_prices.len();
    let m = param_count(self.model_type);
    let bounds = param_bounds(self.model_type);
    let mut j_mat = DMatrix::zeros(n, m);
    let base = self.effective_params();

    for col in 0..m {
      let x = base[col];
      let h = 1e-5_f64.max(1e-3 * x.abs());

      let mut p_plus = base.clone();
      let mut p_minus = base.clone();
      p_plus[col] = (x + h).clamp(bounds[col].0, bounds[col].1);
      p_minus[col] = (x - h).clamp(bounds[col].0, bounds[col].1);
      project_params(self.model_type, &mut p_plus);
      project_params(self.model_type, &mut p_minus);

      let actual_h = p_plus[col] - p_minus[col];
      if actual_h.abs() < EPS {
        continue;
      }

      for i in 0..n {
        let f_plus = fourier_option_price(
          self.model_type,
          &p_plus,
          self.s,
          self.flat_strikes[i],
          self.r,
          self.q,
          self.flat_t[i],
          self.flat_is_call[i],
        );
        let f_minus = fourier_option_price(
          self.model_type,
          &p_minus,
          self.s,
          self.flat_strikes[i],
          self.r,
          self.q,
          self.flat_t[i],
          self.flat_is_call[i],
        );
        // residual = market - model, so d(residual)/dparam = -d(model)/dparam
        j_mat[(i, col)] = -(f_plus - f_minus) / actual_h;
      }
    }

    j_mat
  }
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for LevyCalibrator {
  type JacobianStorage = Owned<f64, Dyn, Dyn>;
  type ParameterStorage = Owned<f64, Dyn>;
  type ResidualStorage = Owned<f64, Dyn>;

  fn set_params(&mut self, params: &DVector<f64>) {
    let mut p: Vec<f64> = params.as_slice().to_vec();
    project_params(self.model_type, &mut p);
    self.params = p;
  }

  fn params(&self) -> DVector<f64> {
    DVector::from_vec(self.effective_params())
  }

  fn residuals(&self) -> Option<DVector<f64>> {
    let c_model = self.compute_model_prices();
    let n = self.flat_prices.len();

    if self.record_history {
      self
        .calibration_history
        .borrow_mut()
        .push(CalibrationHistory {
          residuals: DVector::from_iterator(
            n,
            self
              .flat_prices
              .iter()
              .zip(c_model.iter())
              .map(|(m, d)| m - d),
          ),
          call_put: self
            .flat_prices
            .iter()
            .enumerate()
            .map(|(i, _)| {
              let call = fourier_call_price(
                self.model_type,
                &self.params,
                self.s,
                self.flat_strikes[i],
                self.r,
                self.q,
                self.flat_t[i],
              );
              let put = call - self.s * (-self.q * self.flat_t[i]).exp()
                + self.flat_strikes[i] * (-self.r * self.flat_t[i]).exp();
              (call, put.max(0.0))
            })
            .collect::<Vec<(f64, f64)>>()
            .into(),
          params: self.params.clone(),
          loss_scores: CalibrationLossScore::compute_selected(
            &self.flat_prices,
            &c_model,
            self.loss_metrics,
          ),
        });
    }

    let mut residuals = DVector::zeros(n);
    for i in 0..n {
      residuals[i] = self.flat_prices[i] - c_model[i];
    }

    Some(residuals)
  }

  fn jacobian(&self) -> Option<DMatrix<f64>> {
    Some(self.numeric_jacobian())
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  // Analytical reference prices (Gil-Pelaez inversion)
  // S=100, r=0.05, q=0, T=1.0
  const STRIKES: [f64; 9] = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0];

  // VG: sigma=0.2, theta=-0.1, nu=0.5
  const VG_REF: [f64; 9] = [
    25.056158, 20.941767, 17.091301, 13.572373, 10.453503, 7.795810, 5.640544, 3.991334, 2.793823,
  ];

  // MJD: sigma=0.15, lam=1.0, muj=-0.05, sigj=0.1
  const MJD_REF: [f64; 9] = [
    24.537096, 20.322713, 16.420092, 12.915553, 9.877019, 7.340385, 5.303578, 3.729825, 2.557790,
  ];

  #[test]
  fn vg_pricer_matches_reference() {
    // Verify our internal Fourier pricer reproduces reference VG prices
    let params = [0.2, -0.1, 0.5]; // sigma, theta, nu
    for (i, &k) in STRIKES.iter().enumerate() {
      let price = fourier_call_price(
        LevyModelType::VarianceGamma,
        &params,
        100.0,
        k,
        0.05,
        0.0,
        1.0,
      );
      assert!(
        (price - VG_REF[i]).abs() < 0.05,
        "VG K={k}: got {price:.6}, expected {:.6}",
        VG_REF[i]
      );
    }
  }

  #[test]
  fn mjd_pricer_matches_reference() {
    // Verify our internal Fourier pricer reproduces reference MJD prices
    let params = [0.15, 1.0, -0.05, 0.1]; // sigma, lambda, mu_j, sigma_j
    for (i, &k) in STRIKES.iter().enumerate() {
      let price = fourier_call_price(LevyModelType::MertonJD, &params, 100.0, k, 0.05, 0.0, 1.0);
      assert!(
        (price - MJD_REF[i]).abs() < 0.05,
        "MJD K={k}: got {price:.6}, expected {:.6}",
        MJD_REF[i]
      );
    }
  }

  #[test]
  fn vg_calibrate_recovers_reference_prices() {
    let market = MarketSlice {
      strikes: STRIKES.to_vec(),
      prices: VG_REF.to_vec(),
      is_call: vec![true; 9],
      t: 1.0,
    };

    let calibrator =
      LevyCalibrator::new(LevyModelType::VarianceGamma, 100.0, 0.05, 0.0, vec![market]);

    let result = calibrator.calibrate(None);
    assert!(
      result.loss.get(LossMetric::Rmse) < 0.1,
      "VG RMSE={:.6}",
      result.loss.get(LossMetric::Rmse)
    );
    println!("VG recovered params: {:?}", result.params);
  }

  #[test]
  fn mjd_calibrate_recovers_reference_prices() {
    let market = MarketSlice {
      strikes: STRIKES.to_vec(),
      prices: MJD_REF.to_vec(),
      is_call: vec![true; 9],
      t: 1.0,
    };

    let calibrator = LevyCalibrator::new(LevyModelType::MertonJD, 100.0, 0.05, 0.0, vec![market]);

    let result = calibrator.calibrate(None);
    assert!(
      result.loss.get(LossMetric::Rmse) < 0.1,
      "MJD RMSE={:.6}",
      result.loss.get(LossMetric::Rmse)
    );
    println!("MJD recovered params: {:?}", result.params);
  }

  #[test]
  fn test_levy_vg_calibrate() {
    let market = MarketSlice {
      strikes: vec![90.0, 95.0, 100.0, 105.0, 110.0],
      prices: vec![12.5, 9.0, 6.2, 4.0, 2.3],
      is_call: vec![true, true, true, true, true],
      t: 0.5,
    };

    let calibrator = LevyCalibrator::new(
      LevyModelType::VarianceGamma,
      100.0,
      0.03,
      0.01,
      vec![market],
    );

    let result = calibrator.calibrate(None);
    println!("VG params: {:?}, loss: {:?}", result.params, result.loss);
  }

  #[test]
  fn test_levy_merton_calibrate() {
    let market = MarketSlice {
      strikes: vec![90.0, 95.0, 100.0, 105.0, 110.0],
      prices: vec![12.5, 9.0, 6.2, 4.0, 2.3],
      is_call: vec![true, true, true, true, true],
      t: 0.5,
    };

    let calibrator = LevyCalibrator::new(LevyModelType::MertonJD, 100.0, 0.03, 0.01, vec![market]);

    let result = calibrator.calibrate(None);
    println!(
      "Merton params: {:?}, loss: {:?}",
      result.params, result.loss
    );
  }
}

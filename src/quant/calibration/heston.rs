//! # Heston
//!
//! $$
//! \begin{aligned}dS_t&=\mu S_tdt+\sqrt{v_t}S_tdW_t^S\\dv_t&=\kappa(\theta-v_t)dt+\xi\sqrt{v_t}dW_t^v,\ d\langle W^S,W^v\rangle_t=\rho dt\end{aligned}
//! $$
//!
use std::cell::RefCell;
use std::f64::consts::FRAC_1_PI;
use std::rc::Rc;
use std::sync::OnceLock;

use levenberg_marquardt::LeastSquaresProblem;
use levenberg_marquardt::LevenbergMarquardt;
use nalgebra::DMatrix;
use nalgebra::DVector;
use nalgebra::Dyn;
use nalgebra::Owned;
use ndarray::Array1;
use num_complex::Complex64;

use crate::quant::CalibrationLossScore;
use crate::quant::OptionType;
use crate::quant::calibration::CalibrationHistory;
use crate::quant::loss;
use crate::quant::pricing::heston::HestonPricer;
use crate::stats::heston_mle::HestonMleResult;
use crate::stats::heston_mle::nmle_heston;
use crate::stats::heston_mle::nmle_heston_with_delta;
use crate::stats::heston_mle::pmle_heston;
use crate::stats::heston_mle::pmle_heston_with_delta;
use crate::stats::heston_nml_cekf::HestonNMLECEKFConfig;
use crate::stats::heston_nml_cekf::nmle_cekf_heston;
use crate::traits::PricerExt;

const EPS: f64 = 1e-8;
const RHO_BOUND: f64 = 0.9999;
const KAPPA_MIN: f64 = 1e-3;
const THETA_MIN: f64 = 1e-8;
const SIGMA_MIN: f64 = 1e-8;
const CUI_GL_N: usize = 64;
const CUI_U_MAX: f64 = 100.0;

// Use periodic linear extension mapping into these ranges
const P_KAPPA: (f64, f64) = (0.1, 20.0);
const P_THETA: (f64, f64) = (0.001, 0.4);
const P_SIGMA: (f64, f64) = (0.01, 0.6);
const P_RHO: (f64, f64) = (-1.0, 1.0);
const P_V0: (f64, f64) = (0.005, 0.25);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
/// Jacobian strategy used by the Heston least-squares calibration.
///
/// Source:
/// - Cui et al. (2017), analytic Heston calibration Jacobian
///   https://doi.org/10.1016/j.ejor.2017.05.018
pub enum HestonJacobianMethod {
  /// Central finite-difference Jacobian.
  NumericFiniteDiff,
  /// Closed-form analytic Jacobian for Heston calibration integrals.
  ///
  /// Source:
  /// - Cui et al. (2017)
  ///   https://doi.org/10.1016/j.ejor.2017.05.018
  #[default]
  CuiAnalytic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
/// Seed estimators for Heston calibration from historical time series.
///
/// Source:
/// - Wang et al. (2018), NMLE/PMLE/NMLE-CEKF
///   https://doi.org/10.1007/s11432-017-9215-8
///   http://scis.scichina.com/en/2018/042202.pdf
pub enum HestonMleSeedMethod {
  #[default]
  /// Nonlinear MLE based on square-root variance dynamics.
  Nmle,
  /// Pseudo-MLE closed-form approximation.
  Pmle,
  /// NMLE with consistent EKF latent-variance filtering.
  NmleCekf,
}

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

impl HestonParams {
  fn periodic_map(x: f64, c: f64, d: f64) -> f64 {
    if c <= x && x <= d {
      x
    } else {
      let range = d - c;
      if range <= 0.0 {
        return c;
      }
      let n = ((x - c) / range).floor();
      let n_int = n as i64;
      if n_int % 2 == 0 {
        x - n * range
      } else {
        d + n * range - (x - c)
      }
    }
  }

  /// Project parameters to satisfy Heston admissibility constraints and periodic-range mapping.
  /// Steps:
  /// 1) Periodic mapping into fixed parameter ranges
  /// 2) Enforce basic positivity/box constraints
  /// 3) Enforce Feller by lowering sigma when needed (otherwise minimally bump theta)
  ///
  /// Source:
  /// - Heston (1993), variance process admissibility context
  ///   https://doi.org/10.1093/rfs/6.2.327
  pub fn project_in_place(&mut self) {
    self.kappa = Self::periodic_map(self.kappa, P_KAPPA.0, P_KAPPA.1);
    self.theta = Self::periodic_map(self.theta, P_THETA.0, P_THETA.1);
    self.sigma = Self::periodic_map(self.sigma, P_SIGMA.0, P_SIGMA.1).abs();
    self.rho = Self::periodic_map(self.rho, P_RHO.0, P_RHO.1);
    self.v0 = Self::periodic_map(self.v0, P_V0.0, P_V0.1);

    self.v0 = self.v0.max(0.0);
    self.kappa = self.kappa.max(KAPPA_MIN);
    self.theta = self.theta.max(THETA_MIN);
    self.sigma = self.sigma.abs().max(SIGMA_MIN);
    self.rho = self.rho.clamp(-RHO_BOUND, RHO_BOUND);

    // 3) Feller condition: 2*kappa*theta ≥ sigma^2.
    if 2.0 * self.kappa * self.theta < self.sigma * self.sigma {
      let sigma_star = (2.0 * self.kappa * self.theta).sqrt();
      if sigma_star >= P_SIGMA.0 {
        // Prefer reducing sigma, but keep within the range lower bound as well.
        self.sigma = sigma_star.min(P_SIGMA.1);
      } else {
        // As a fallback (when sigma would go below minimum), bump theta minimally, respecting the range upper bound.
        let theta_star = ((self.sigma * self.sigma) / (2.0 * self.kappa)).max(THETA_MIN) + EPS;
        self.theta = theta_star.min(P_THETA.1);
      }
    }
  }

  pub fn projected(mut self) -> Self {
    self.project_in_place();
    self
  }
}

impl From<HestonMleResult> for HestonParams {
  fn from(mle: HestonMleResult) -> Self {
    HestonParams {
      v0: mle.v0,
      kappa: mle.kappa,
      theta: mle.theta,
      sigma: mle.sigma,
      rho: mle.rho,
    }
  }
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

#[derive(Clone, Copy, Debug)]
struct CuiCfTerms {
  iu: Complex64,
  u2_iu: Complex64,
  xi: Complex64,
  d: Complex64,
  sinh_z: Complex64,
  cosh_z: Complex64,
  a2: Complex64,
  a: Complex64,
  b: Complex64,
  dlog: Complex64,
  phi: Complex64,
  exp_half_kappa_t: f64,
}

fn finite_c64(z: Complex64) -> bool {
  z.re.is_finite() && z.im.is_finite()
}

fn gauss_legendre_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
  let mut x = vec![0.0; n];
  let mut w = vec![0.0; n];
  let m = n.div_ceil(2);
  let eps = 1e-14;

  for i in 0..m {
    let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
    loop {
      let mut p1 = 1.0;
      let mut p2 = 0.0;
      for j in 1..=n {
        let p3 = p2;
        p2 = p1;
        p1 = ((2.0 * j as f64 - 1.0) * z * p2 - (j as f64 - 1.0) * p3) / j as f64;
      }
      let pp = n as f64 * (z * p1 - p2) / (z * z - 1.0);
      let z_next = z - p1 / pp;
      if (z_next - z).abs() <= eps {
        z = z_next;
        break;
      }
      z = z_next;
    }

    x[i] = -z;
    x[n - 1 - i] = z;

    let mut p1 = 1.0;
    let mut p2 = 0.0;
    for j in 1..=n {
      let p3 = p2;
      p2 = p1;
      p1 = ((2.0 * j as f64 - 1.0) * z * p2 - (j as f64 - 1.0) * p3) / j as f64;
    }
    let pp = n as f64 * (z * p1 - p2) / (z * z - 1.0);
    let wi = 2.0 / ((1.0 - z * z) * pp * pp);
    w[i] = wi;
    w[n - 1 - i] = wi;
  }

  (x, w)
}

fn gauss_legendre_64() -> (&'static [f64], &'static [f64]) {
  static GL64: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
  let (x, w) = GL64.get_or_init(|| gauss_legendre_nodes_weights(CUI_GL_N));
  (x.as_slice(), w.as_slice())
}

#[derive(Clone)]
/// Heston least-squares calibrator using Levenberg-Marquardt iterations.
///
/// Source:
/// - Levenberg (1944), https://doi.org/10.1090/qam/10666
/// - Marquardt (1963), https://doi.org/10.1137/0111030
/// - Heston model (1993), https://doi.org/10.1093/rfs/6.2.327
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
  /// Seed method for the MLE-based initial guess.
  pub mle_seed_method: HestonMleSeedMethod,
  /// Optional explicit sampling step used by MLE seed estimators.
  pub mle_delta: Option<f64>,
  /// Optional config for NMLE-CEKF seed when `mle_seed_method = NmleCekf`.
  pub nmle_cekf_config: Option<HestonNMLECEKFConfig>,
  /// If true, record per-iteration calibration history.
  pub record_history: bool,
  /// Jacobian/method choice for calibration.
  pub jacobian_method: HestonJacobianMethod,
  /// History of iterations (residuals, params, loss metrics).
  calibration_history: Rc<RefCell<Vec<CalibrationHistory<HestonParams>>>>,
}

impl HestonCalibrator {
  pub fn new(
    params: Option<HestonParams>,
    c_market: DVector<f64>,
    s: DVector<f64>,
    k: DVector<f64>,
    r: f64,
    q: Option<f64>,
    tau: f64,
    option_type: OptionType,
    mle_s: Option<Array1<f64>>,
    mle_v: Option<Array1<f64>>,
    mle_r: Option<f64>,
    record_history: bool,
  ) -> Self {
    assert_eq!(
      c_market.len(),
      s.len(),
      "c_market and s must have the same length"
    );
    assert_eq!(
      c_market.len(),
      k.len(),
      "c_market and k must have the same length"
    );
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
      tau,
      option_type,
      mle_s,
      mle_v,
      mle_r,
      mle_seed_method: HestonMleSeedMethod::default(),
      mle_delta: None,
      nmle_cekf_config: None,
      record_history,
      jacobian_method: HestonJacobianMethod::default(),
      calibration_history: Rc::new(RefCell::new(Vec::new())),
    }
  }
}

impl HestonCalibrator {
  pub fn calibrate(&self) {
    // Prepare a problem clone with an initial guess if needed
    let mut problem = self.clone();
    problem.ensure_initial_guess();

    println!(
      "Initial guess ({:?}): {:?}",
      problem.jacobian_method, problem.params
    );

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
    self.params = Some(params.projected());
  }

  /// Enable or disable recording of per-iteration calibration history.
  pub fn set_record_history(&mut self, record: bool) {
    self.record_history = record;
  }

  pub fn set_jacobian_method(&mut self, method: HestonJacobianMethod) {
    self.jacobian_method = method;
  }

  pub fn set_mle_seed_method(&mut self, method: HestonMleSeedMethod) {
    self.mle_seed_method = method;
  }

  pub fn set_mle_delta(&mut self, delta: Option<f64>) {
    self.mle_delta = delta;
  }

  pub fn set_nmle_cekf_config(&mut self, cfg: HestonNMLECEKFConfig) {
    self.nmle_cekf_config = Some(cfg);
  }

  pub fn calibrate_cui(&mut self) {
    self.jacobian_method = HestonJacobianMethod::CuiAnalytic;
    self.calibrate();
  }

  /// Retrieve the collected calibration history.
  pub fn history(&self) -> Vec<CalibrationHistory<HestonParams>> {
    self.calibration_history.borrow().clone()
  }

  fn fallback_params() -> HestonParams {
    HestonParams {
      v0: 0.04,
      kappa: 1.5,
      theta: 0.04,
      sigma: 0.5,
      rho: -0.5,
    }
    .projected()
  }

  fn inferred_mle_delta(&self, n_obs: usize) -> f64 {
    if let Some(dt) = self.mle_delta
      && dt.is_finite()
      && dt > 0.0
    {
      return dt;
    }
    1.0 / n_obs.saturating_sub(1).max(1) as f64
  }

  /// Build an initial parameter guess from historical series via NMLE/PMLE/NMLE-CEKF.
  ///
  /// Source:
  /// - Wang et al. (2018)
  ///   https://doi.org/10.1007/s11432-017-9215-8
  ///   http://scis.scichina.com/en/2018/042202.pdf
  fn infer_initial_guess_from_series(&self) -> Option<HestonParams> {
    match self.mle_seed_method {
      HestonMleSeedMethod::Nmle => {
        let (s, v, r) = (self.mle_s.clone()?, self.mle_v.clone()?, self.mle_r?);
        if s.len() < 2 || v.len() < 2 || s.len() != v.len() {
          return None;
        }
        let p: HestonParams = if self.mle_delta.is_some() {
          let delta = self.inferred_mle_delta(s.len());
          nmle_heston_with_delta(s, v, r, delta).into()
        } else {
          nmle_heston(s, v, r).into()
        };
        Some(p.projected())
      }
      HestonMleSeedMethod::Pmle => {
        let (s, v, r) = (self.mle_s.clone()?, self.mle_v.clone()?, self.mle_r?);
        if s.len() < 2 || v.len() < 2 || s.len() != v.len() {
          return None;
        }
        let p: HestonParams = if self.mle_delta.is_some() {
          let delta = self.inferred_mle_delta(s.len());
          pmle_heston_with_delta(s, v, r, delta).into()
        } else {
          pmle_heston(s, v, r).into()
        };
        Some(p.projected())
      }
      HestonMleSeedMethod::NmleCekf => {
        let (s, r) = (self.mle_s.clone()?, self.mle_r?);
        if s.len() < 2 {
          return None;
        }
        let mut cfg = self.nmle_cekf_config.clone().unwrap_or_default();
        cfg.r = r;
        cfg.delta = self.inferred_mle_delta(s.len());
        if let Some(v_ts) = self.mle_v.as_ref()
          && !v_ts.is_empty()
          && v_ts[0].is_finite()
          && v_ts[0] > 0.0
        {
          cfg.initial_v0 = v_ts[0];
        }
        let out = nmle_cekf_heston(s, cfg);
        let p: HestonParams = out.params.into();
        Some(p.projected())
      }
    }
  }

  fn ensure_initial_guess(&mut self) {
    if self.params.is_none() {
      self.params = Some(
        self
          .infer_initial_guess_from_series()
          .unwrap_or_else(Self::fallback_params),
      );
    }
  }

  fn effective_params(&self) -> HestonParams {
    if let Some(p) = &self.params {
      return p.clone().projected();
    }
    self
      .infer_initial_guess_from_series()
      .unwrap_or_else(Self::fallback_params)
  }

  fn compute_model_prices_for_numeric(&self, params: &HestonParams) -> DVector<f64> {
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

  fn cui_terms_for(
    &self,
    params: &HestonParams,
    s: f64,
    u: Complex64,
    tau: f64,
  ) -> Option<CuiCfTerms> {
    let iu = Complex64::i() * u;
    let u2_iu = u * u + iu;
    let xi = Complex64::new(params.kappa, 0.0) - params.sigma * params.rho * iu;
    let d = (xi * xi + (params.sigma * params.sigma) * u2_iu).sqrt();
    let z = d * (0.5 * tau);
    let sinh_z = z.sinh();
    let cosh_z = z.cosh();
    let a1 = u2_iu * sinh_z;
    let a2 = d * cosh_z + xi * sinh_z;
    if a2.norm() < EPS {
      return None;
    }

    let a = a1 / a2;
    let exp_half_kappa_t = (0.5 * params.kappa * tau).exp();
    let b = d * exp_half_kappa_t / a2;

    let exp_neg_dt = (-d * tau).exp();
    let half = Complex64::new(0.5, 0.0);
    let log_arg = half * (d + xi) + half * (d - xi) * exp_neg_dt;
    if log_arg.norm() < EPS {
      return None;
    }
    let dlog = d.ln() + (params.kappa - d) * (0.5 * tau) - log_arg.ln();

    let rq = self.r - self.q.unwrap_or(0.0);
    let exponent = iu * (s.ln() + rq * tau)
      - tau * params.kappa * params.theta * params.rho * iu / params.sigma
      - params.v0 * a
      + (2.0 * params.kappa * params.theta / (params.sigma * params.sigma)) * dlog;
    let phi = exponent.exp();

    let terms = CuiCfTerms {
      iu,
      u2_iu,
      xi,
      d,
      sinh_z,
      cosh_z,
      a2,
      a,
      b,
      dlog,
      phi,
      exp_half_kappa_t,
    };

    if finite_c64(terms.d)
      && finite_c64(terms.a)
      && finite_c64(terms.b)
      && finite_c64(terms.dlog)
      && finite_c64(terms.phi)
    {
      Some(terms)
    } else {
      None
    }
  }

  fn cui_da_db_ddlog(
    &self,
    terms: &CuiCfTerms,
    d_prime: Complex64,
    xi_prime: Complex64,
    tau: f64,
    with_kappa_prefactor: bool,
  ) -> Option<(Complex64, Complex64, Complex64)> {
    let z_prime = d_prime * (0.5 * tau);
    let da1 = terms.u2_iu * terms.cosh_z * z_prime;
    let da2 = d_prime * terms.cosh_z
      + terms.d * terms.sinh_z * z_prime
      + xi_prime * terms.sinh_z
      + terms.xi * terms.cosh_z * z_prime;
    if da2.re.is_nan() || da2.im.is_nan() || terms.a2.norm() < EPS {
      return None;
    }

    let da = (da1 - terms.a * da2) / terms.a2;
    let mut db =
      terms.exp_half_kappa_t * (d_prime / terms.a2 - terms.d * da2 / (terms.a2 * terms.a2));
    if with_kappa_prefactor {
      db += terms.b * (0.5 * tau);
    }
    let ddlog = db / terms.b;

    if finite_c64(da) && finite_c64(db) && finite_c64(ddlog) {
      Some((da, db, ddlog))
    } else {
      None
    }
  }

  fn cui_price_and_grad_for_quote(
    &self,
    params: &HestonParams,
    s: f64,
    k: f64,
    tau: f64,
  ) -> Option<(f64, [f64; 5])> {
    let (nodes, weights) = gauss_legendre_64();
    let scale = 0.5 * CUI_U_MAX;
    let sigma = params.sigma;
    let sigma2 = sigma * sigma;
    let sigma3 = sigma2 * sigma;

    let mut i1 = 0.0_f64;
    let mut i2 = 0.0_f64;
    let mut g1 = [0.0_f64; 5];
    let mut g2 = [0.0_f64; 5];

    for (&x, &w) in nodes.iter().zip(weights.iter()) {
      let u_real = scale * (x + 1.0);
      let w_scaled = scale * w;
      if u_real <= EPS {
        continue;
      }
      let k_kernel = (Complex64::new(0.0, -u_real * k.ln())).exp()
        / (Complex64::i() * Complex64::new(u_real, 0.0));

      let terms_u = self.cui_terms_for(params, s, Complex64::new(u_real, 0.0), tau)?;
      let terms_shift = self.cui_terms_for(params, s, Complex64::new(u_real, -1.0), tau)?;

      let d_prime_rho_u = -sigma * terms_u.iu * terms_u.xi / terms_u.d;
      let d_prime_rho_shift = -sigma * terms_shift.iu * terms_shift.xi / terms_shift.d;
      let xi_prime_rho_u = -sigma * terms_u.iu;
      let xi_prime_rho_shift = -sigma * terms_shift.iu;

      let d_prime_kappa_u = terms_u.xi / terms_u.d;
      let d_prime_kappa_shift = terms_shift.xi / terms_shift.d;
      let xi_prime_kappa = Complex64::new(1.0, 0.0);

      let d_prime_sigma_u =
        (sigma * terms_u.u2_iu - params.rho * terms_u.iu * terms_u.xi) / terms_u.d;
      let d_prime_sigma_shift =
        (sigma * terms_shift.u2_iu - params.rho * terms_shift.iu * terms_shift.xi) / terms_shift.d;
      let xi_prime_sigma_u = -params.rho * terms_u.iu;
      let xi_prime_sigma_shift = -params.rho * terms_shift.iu;

      let (d_a_rho_u, _d_b_rho_u, d_dlog_rho_u) =
        self.cui_da_db_ddlog(&terms_u, d_prime_rho_u, xi_prime_rho_u, tau, false)?;
      let (d_a_rho_shift, _d_b_rho_shift, d_dlog_rho_shift) = self.cui_da_db_ddlog(
        &terms_shift,
        d_prime_rho_shift,
        xi_prime_rho_shift,
        tau,
        false,
      )?;

      let (d_a_kappa_u, _d_b_kappa_u, d_dlog_kappa_u) =
        self.cui_da_db_ddlog(&terms_u, d_prime_kappa_u, xi_prime_kappa, tau, true)?;
      let (d_a_kappa_shift, _d_b_kappa_shift, d_dlog_kappa_shift) =
        self.cui_da_db_ddlog(&terms_shift, d_prime_kappa_shift, xi_prime_kappa, tau, true)?;

      let (d_a_sigma_u, _d_b_sigma_u, d_dlog_sigma_u) =
        self.cui_da_db_ddlog(&terms_u, d_prime_sigma_u, xi_prime_sigma_u, tau, false)?;
      let (d_a_sigma_shift, _d_b_sigma_shift, d_dlog_sigma_shift) = self.cui_da_db_ddlog(
        &terms_shift,
        d_prime_sigma_shift,
        xi_prime_sigma_shift,
        tau,
        false,
      )?;

      let h_v0_u = -terms_u.a;
      let h_theta_u = 2.0 * params.kappa / sigma2 * terms_u.dlog
        - tau * params.kappa * params.rho * terms_u.iu / sigma;
      let h_rho_u = -params.v0 * d_a_rho_u
        + (2.0 * params.kappa * params.theta / sigma2) * d_dlog_rho_u
        - tau * params.kappa * params.theta * terms_u.iu / sigma;
      let h_kappa_u = -params.v0 * d_a_kappa_u
        + (2.0 * params.theta / sigma2) * terms_u.dlog
        + (2.0 * params.kappa * params.theta / sigma2) * d_dlog_kappa_u
        - tau * params.theta * params.rho * terms_u.iu / sigma;
      let h_sigma_u = -params.v0 * d_a_sigma_u
        - (4.0 * params.kappa * params.theta / sigma3) * terms_u.dlog
        + (2.0 * params.kappa * params.theta / sigma2) * d_dlog_sigma_u
        + tau * params.kappa * params.theta * params.rho * terms_u.iu / sigma2;

      let h_v0_shift = -terms_shift.a;
      let h_theta_shift = 2.0 * params.kappa / sigma2 * terms_shift.dlog
        - tau * params.kappa * params.rho * terms_shift.iu / sigma;
      let h_rho_shift = -params.v0 * d_a_rho_shift
        + (2.0 * params.kappa * params.theta / sigma2) * d_dlog_rho_shift
        - tau * params.kappa * params.theta * terms_shift.iu / sigma;
      let h_kappa_shift = -params.v0 * d_a_kappa_shift
        + (2.0 * params.theta / sigma2) * terms_shift.dlog
        + (2.0 * params.kappa * params.theta / sigma2) * d_dlog_kappa_shift
        - tau * params.theta * params.rho * terms_shift.iu / sigma;
      let h_sigma_shift = -params.v0 * d_a_sigma_shift
        - (4.0 * params.kappa * params.theta / sigma3) * terms_shift.dlog
        + (2.0 * params.kappa * params.theta / sigma2) * d_dlog_sigma_shift
        + tau * params.kappa * params.theta * params.rho * terms_shift.iu / sigma2;

      let dphi_u = [
        terms_u.phi * h_v0_u,
        terms_u.phi * h_kappa_u,
        terms_u.phi * h_theta_u,
        terms_u.phi * h_sigma_u,
        terms_u.phi * h_rho_u,
      ];
      let dphi_shift = [
        terms_shift.phi * h_v0_shift,
        terms_shift.phi * h_kappa_shift,
        terms_shift.phi * h_theta_shift,
        terms_shift.phi * h_sigma_shift,
        terms_shift.phi * h_rho_shift,
      ];

      i1 += w_scaled * (k_kernel * terms_shift.phi).re;
      i2 += w_scaled * (k_kernel * terms_u.phi).re;
      for j in 0..5 {
        g1[j] += w_scaled * (k_kernel * dphi_shift[j]).re;
        g2[j] += w_scaled * (k_kernel * dphi_u[j]).re;
      }
    }

    let disc_r = (-self.r * tau).exp();
    let disc_q = (-self.q.unwrap_or(0.0) * tau).exp();
    let call = 0.5 * (s * disc_q - k * disc_r) + disc_r * FRAC_1_PI * (i1 - k * i2);
    let mut grad = [0.0_f64; 5];
    for j in 0..5 {
      grad[j] = disc_r * FRAC_1_PI * (g1[j] - k * g2[j]);
    }

    if call.is_finite() && grad.iter().all(|g| g.is_finite()) {
      Some((call, grad))
    } else {
      None
    }
  }

  /// Compute model prices and residual Jacobian via the Cui analytic formulation.
  ///
  /// Source:
  /// - Cui et al. (2017), fast and accurate Heston calibration
  ///   https://doi.org/10.1016/j.ejor.2017.05.018
  fn compute_model_prices_and_residual_jacobian_cui(
    &self,
    params: &HestonParams,
  ) -> Option<(DVector<f64>, DMatrix<f64>)> {
    let n = self.c_market.len();
    let mut c_model = DVector::zeros(n);
    let mut j_residual = DMatrix::zeros(n, 5);

    for idx in 0..n {
      let s = self.s[idx];
      let k = self.k[idx];
      let (call_raw, grad_call) = self.cui_price_and_grad_for_quote(params, s, k, self.tau)?;
      let disc_r = (-self.r * self.tau).exp();
      let disc_q = (-self.q.unwrap_or(0.0) * self.tau).exp();
      let put_raw = call_raw - s * disc_q + k * disc_r;

      let (model_raw, grad_model) = match self.option_type {
        OptionType::Call => (call_raw, grad_call),
        OptionType::Put => (put_raw, grad_call),
      };

      if model_raw > 0.0 {
        c_model[idx] = model_raw;
        for col in 0..5 {
          // residual = market - model
          j_residual[(idx, col)] = -grad_model[col];
        }
      } else {
        c_model[idx] = 0.0;
      }
    }

    Some((c_model, j_residual))
  }

  fn compute_model_prices_for_cui(&self, params: &HestonParams) -> Option<DVector<f64>> {
    self
      .compute_model_prices_and_residual_jacobian_cui(params)
      .map(|(c_model, _)| c_model)
  }

  fn compute_model_prices_for(&self, params: &HestonParams) -> DVector<f64> {
    match self.jacobian_method {
      HestonJacobianMethod::NumericFiniteDiff => self.compute_model_prices_for_numeric(params),
      HestonJacobianMethod::CuiAnalytic => self
        .compute_model_prices_for_cui(params)
        .unwrap_or_else(|| self.compute_model_prices_for_numeric(params)),
    }
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
          // v0 ≥ 0
          params_plus.v0 = (x + h).max(0.0);
          params_minus.v0 = (x - h).max(0.0);
        }
        1 => {
          // kappa > 0
          params_plus.kappa = (x + h).max(KAPPA_MIN);
          params_minus.kappa = (x - h).max(KAPPA_MIN);
        }
        2 => {
          // theta > 0
          params_plus.theta = (x + h).max(THETA_MIN);
          params_minus.theta = (x - h).max(THETA_MIN);
        }
        3 => {
          // sigma ≥ 0
          params_plus.sigma = (x + h).abs();
          params_minus.sigma = (x - h).abs();
        }
        4 => {
          // −1 < rho < 1
          let clamp = |y: f64| y.clamp(-RHO_BOUND, RHO_BOUND);
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

      // Enforce full projection (incl. Feller) on probes
      params_plus.project_in_place();
      params_minus.project_in_place();

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
    let p = HestonParams::from(params.clone()).projected();
    self.params = Some(p);
  }

  fn params(&self) -> DVector<f64> {
    self.effective_params().into()
  }

  fn residuals(&self) -> Option<DVector<f64>> {
    let params_eff = self.effective_params();
    let c_model = self.compute_model_prices_for(&params_eff);

    // Push history for the current iterate if enabled
    if self.record_history {
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
            mae: loss::mae(self.c_market.as_slice(), c_model.as_slice()),
            mse: loss::mse(self.c_market.as_slice(), c_model.as_slice()),
            rmse: loss::rmse(self.c_market.as_slice(), c_model.as_slice()),
            mpe: loss::mpe(self.c_market.as_slice(), c_model.as_slice()),
            mape: loss::mape(self.c_market.as_slice(), c_model.as_slice()),
            mspe: loss::mspe(self.c_market.as_slice(), c_model.as_slice()),
            rmspe: loss::rmspe(self.c_market.as_slice(), c_model.as_slice()),
            mre: loss::mre(self.c_market.as_slice(), c_model.as_slice()),
            mrpe: loss::mrpe(self.c_market.as_slice(), c_model.as_slice()),
          },
        });
    }

    Some(self.c_market.clone() - c_model)
  }

  fn jacobian(&self) -> Option<DMatrix<f64>> {
    let p = self.effective_params();
    match self.jacobian_method {
      HestonJacobianMethod::NumericFiniteDiff => Some(self.numeric_jacobian(&p)),
      HestonJacobianMethod::CuiAnalytic => {
        if let Some((_, jac)) = self.compute_model_prices_and_residual_jacobian_cui(&p) {
          Some(jac)
        } else {
          Some(self.numeric_jacobian(&p))
        }
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;

  use super::*;
  use crate::stochastic::volatility::HestonPow;
  use crate::stochastic::volatility::heston::Heston as HestonProcess;
  use crate::traits::ProcessExt;

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
      true,
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
      true,
    );

    calibrator.calibrate();
  }

  #[test]
  fn test_heston_calibrate_with_pmle_seed() {
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
    let mu = 0.0;

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
    );
    let [s_ts, v_ts] = process.sample();

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

    let mut calibrator = HestonCalibrator::new(
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
      true,
    );
    calibrator.set_mle_seed_method(HestonMleSeedMethod::Pmle);
    calibrator.set_mle_delta(Some(t / (n - 1) as f64));

    calibrator.calibrate();
  }

  #[test]
  fn test_heston_calibrate_with_nmle_cekf_seed() {
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
    let mu = 0.0;

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
    );
    let [s_ts, _v_ts] = process.sample();

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

    let mut calibrator = HestonCalibrator::new(
      None,
      c_market.clone().into(),
      s_grid.clone().into(),
      strikes.clone().into(),
      r,
      q,
      tau,
      OptionType::Call,
      Some(s_ts),
      None,
      Some(r),
      true,
    );
    calibrator.set_mle_seed_method(HestonMleSeedMethod::NmleCekf);
    calibrator.set_mle_delta(Some(t / (n - 1) as f64));
    calibrator.set_nmle_cekf_config(HestonNMLECEKFConfig {
      max_iters: 6,
      tol: 1e-5,
      param_damping: 0.6,
      initial_v0: v0,
      ..HestonNMLECEKFConfig::default()
    });

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
      true,
    );

    calibrator.calibrate();
    let history = calibrator.history();
    println!("{:?}", history);
  }

  #[test]
  fn test_heston_cui_price_and_jacobian_finite() {
    let params = HestonParams {
      v0: 0.04,
      kappa: 1.8,
      theta: 0.05,
      sigma: 0.45,
      rho: -0.55,
    };
    let s = vec![100.0; 6];
    let k = vec![80.0, 90.0, 95.0, 100.0, 110.0, 120.0];
    let r = 0.01;
    let q = Some(0.0);
    let tau = 0.75;
    let option_type = OptionType::Call;

    let market = vec![22.0, 14.8, 11.9, 9.5, 6.2, 4.0];
    let mut calibrator = HestonCalibrator::new(
      Some(params.clone()),
      market.into(),
      s.clone().into(),
      k.clone().into(),
      r,
      q,
      tau,
      option_type,
      None,
      None,
      None,
      false,
    );
    calibrator.set_jacobian_method(HestonJacobianMethod::CuiAnalytic);

    let (c_model, jac) = calibrator
      .compute_model_prices_and_residual_jacobian_cui(&params)
      .expect("Cui model/jacobian should be computable");
    assert_eq!(c_model.len(), k.len());
    assert_eq!(jac.nrows(), k.len());
    assert_eq!(jac.ncols(), 5);
    assert!(c_model.iter().all(|x| x.is_finite()));
    assert!(jac.iter().all(|x| x.is_finite()));

    for (i, &strike) in k.iter().enumerate() {
      let pr = HestonPricer::new(
        s[i],
        params.v0,
        strike,
        r,
        q,
        params.rho,
        params.kappa,
        params.theta,
        params.sigma,
        Some(0.0),
        Some(tau),
        None,
        None,
      );
      let (call_ref, _) = pr.calculate_call_put();
      let rel = ((c_model[i] - call_ref).abs()) / (1.0 + call_ref.abs());
      assert!(rel < 5e-2, "quote {} relative gap too large: {}", i, rel);
    }
  }

  #[test]
  fn test_heston_cui_jacobian_matches_numeric() {
    let params = HestonParams {
      v0: 0.05,
      kappa: 1.4,
      theta: 0.06,
      sigma: 0.35,
      rho: -0.45,
    };
    let s = vec![100.0; 5];
    let k = vec![85.0, 95.0, 100.0, 105.0, 115.0];
    let r = 0.015;
    let q = Some(0.0);
    let tau = 0.6;

    let market = vec![18.0, 11.0, 8.5, 6.4, 3.7];
    let calibrator = HestonCalibrator::new(
      Some(params.clone()),
      market.into(),
      s.into(),
      k.into(),
      r,
      q,
      tau,
      OptionType::Call,
      None,
      None,
      None,
      false,
    );

    let jac_num = calibrator.numeric_jacobian(&params);
    let (_, jac_cui) = calibrator
      .compute_model_prices_and_residual_jacobian_cui(&params)
      .expect("Cui Jacobian should be computable");

    for row in 0..jac_num.nrows() {
      for col in 0..jac_num.ncols() {
        let n = jac_num[(row, col)];
        let a = jac_cui[(row, col)];
        let rel = (a - n).abs() / (1.0 + n.abs());
        assert!(
          rel < 2e-1,
          "Jacobian mismatch at ({}, {}): analytic={}, numeric={}, rel={}",
          row,
          col,
          a,
          n,
          rel
        );
      }
    }
  }
}

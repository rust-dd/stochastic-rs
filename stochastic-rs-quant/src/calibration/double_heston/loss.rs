use std::f64::consts::FRAC_1_PI;

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use num_complex::Complex64;

use super::super::integrate_gl_to_convergence;
use super::super::periodic_map;
use super::calibrator::DoubleHestonCalibrator;
use super::params::DoubleHestonParams;
use super::params::P_SIGMA;
use super::params::P_V0;
use crate::OptionType;
use crate::calibration::clip_negative;
use crate::calibration::heston::params::KAPPA_MIN;
use crate::calibration::heston::params::P_KAPPA;
use crate::calibration::heston::params::P_RHO;
use crate::calibration::heston::params::P_THETA;
use crate::calibration::heston::params::RHO_BOUND;
use crate::calibration::heston::params::SIGMA_MIN;
use crate::calibration::heston::params::THETA_MIN;
use crate::calibration::least_squares::LeastSquaresProblem;
use crate::calibration::least_squares::Trail;
use crate::calibration::periodic_slopes;
use crate::calibration::scale_columns;
use crate::pricing::fourier::factor_cd;

/// Double Heston characteristic function $\phi_T(u)$ of $\ln(S_T/S_0)$,
/// computed as the sum of two independent Heston contributions.
pub(super) fn double_heston_cf(
  p: &DoubleHestonParams,
  r: f64,
  q: f64,
  tau: f64,
  u: Complex64,
) -> Complex64 {
  let i = Complex64::i();

  let (c1, d1) = factor_cd(&p.first, tau, u);
  let (c2, d2) = factor_cd(&p.second, tau, u);

  (c1 + c2 + d1 * p.first.v0 + d2 * p.second.v0 + i * u * (r - q) * tau).exp()
}

/// Price a European call option under the double Heston model via a
/// convergence-controlled Gil-Pelaez quadrature.
pub(super) fn double_heston_call_price(
  p: &DoubleHestonParams,
  s: f64,
  k: f64,
  r: f64,
  q: f64,
  tau: f64,
) -> f64 {
  let ln_ks = (k / s).ln();
  let phi_neg_i = double_heston_cf(p, r, q, tau, Complex64::new(0.0, -1.0));
  let phi_neg_i_norm = phi_neg_i.norm();
  let disc_r = (-r * tau).exp();
  let disc_q = (-q * tau).exp();

  let Some([integral]) = integrate_gl_to_convergence(
    |u_real| {
      let xi = Complex64::new(u_real, 0.0);
      let xi_shift = Complex64::new(u_real, -1.0);

      let phi = double_heston_cf(p, r, q, tau, xi);
      let phi_shift = double_heston_cf(p, r, q, tau, xi_shift);

      let kernel = (Complex64::new(0.0, -u_real * ln_ks)).exp()
        / (Complex64::i() * Complex64::new(u_real, 0.0));

      let p1 = if phi_neg_i_norm > 1e-30 {
        (kernel * phi_shift / phi_neg_i).re
      } else {
        0.0
      };
      Some([s * disc_q * p1 - k * disc_r * (kernel * phi).re])
    },
    1e-8,
  ) else {
    return f64::NAN;
  };

  let call = 0.5 * (s * disc_q - k * disc_r) + FRAC_1_PI * integral;
  clip_negative(call)
}

pub(super) struct DoubleHestonProblem<'a> {
  calibrator: &'a DoubleHestonCalibrator,
  params: DoubleHestonParams,
  /// `dθⱼ/dxⱼ` of the box's reflection at the optimiser's raw coordinates.
  reflections: Array1<f64>,
  pub(super) market: Vec<f64>,
  pub(super) trail: Trail<DoubleHestonParams>,
}

impl<'a> DoubleHestonProblem<'a> {
  pub(super) fn new(calibrator: &'a DoubleHestonCalibrator, start: DoubleHestonParams) -> Self {
    Self {
      calibrator,
      params: start,
      reflections: Array1::ones(10),
      market: calibrator.quotes.iter().map(|quote| quote.price).collect(),
      trail: Trail::new(),
    }
  }

  pub(super) fn effective_params(&self) -> DoubleHestonParams {
    self.params.projected()
  }

  pub(super) fn compute_model_prices_for(&self, p: &DoubleHestonParams) -> Vec<f64> {
    let DoubleHestonCalibrator { s, r, q, .. } = *self.calibrator;
    self
      .calibrator
      .quotes
      .iter()
      .map(|quote| {
        let (k, tau) = (quote.strike, quote.tau);
        let call = double_heston_call_price(p, s, k, r, q, tau);
        match quote.option_type {
          OptionType::Call => call,
          OptionType::Put => clip_negative(call - s * (-q * tau).exp() + k * (-r * tau).exp()),
        }
      })
      .collect()
  }

  pub(super) fn residuals_for(&self, p: &DoubleHestonParams) -> Array1<f64> {
    self.market_minus(&self.compute_model_prices_for(p))
  }

  fn market_minus(&self, model: &[f64]) -> Array1<f64> {
    Array1::from_iter(
      self
        .market
        .iter()
        .zip(model)
        .map(|(market, model)| market - model),
    )
  }

  /// Central finite-difference Jacobian over the 10 parameters.
  pub(super) fn numeric_jacobian(&self, params: &DoubleHestonParams) -> Array2<f64> {
    let n = self.calibrator.quotes.len();
    let p_dim = 10usize;

    let base_params_vec: Array1<f64> = (*params).into();
    let mut j_mat = Array2::zeros((n, p_dim));

    for col in 0..p_dim {
      let x = base_params_vec[col];
      let mut h = 1e-5_f64.max(1e-3 * x.abs());

      let mut params_plus = *params;
      let mut params_minus = *params;
      let (plus, minus) = if col < 5 {
        (&mut params_plus.first, &mut params_minus.first)
      } else {
        (&mut params_plus.second, &mut params_minus.second)
      };

      let field_plus_minus = |x: f64, h: f64, lo: f64, hi: f64| -> (f64, f64) {
        (periodic_map(x + h, lo, hi), periodic_map(x - h, lo, hi))
      };

      match col % 5 {
        0 => {
          let (a, b) = field_plus_minus(x, h, P_V0.0, P_V0.1);
          plus.v0 = a.max(0.0);
          minus.v0 = b.max(0.0);
        }
        1 => {
          let (a, b) = field_plus_minus(x, h, P_KAPPA.0, P_KAPPA.1);
          plus.kappa = a.max(KAPPA_MIN);
          minus.kappa = b.max(KAPPA_MIN);
        }
        2 => {
          let (a, b) = field_plus_minus(x, h, P_THETA.0, P_THETA.1);
          plus.theta = a.max(THETA_MIN);
          minus.theta = b.max(THETA_MIN);
        }
        3 => {
          let (a, b) = field_plus_minus(x, h, P_SIGMA.0, P_SIGMA.1);
          plus.sigma = a.abs().max(SIGMA_MIN);
          minus.sigma = b.abs().max(SIGMA_MIN);
        }
        _ => {
          let clamp = |y: f64| y.clamp(-RHO_BOUND, RHO_BOUND);
          plus.rho = clamp(x + h);
          minus.rho = clamp(x - h);
          if (plus.rho - minus.rho).abs() < 0.5 * h {
            h = 1e-4;
            plus.rho = clamp(x + h);
            minus.rho = clamp(x - h);
          }
        }
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

impl LeastSquaresProblem for DoubleHestonProblem<'_> {
  fn set_params(&mut self, params: ArrayView1<f64>) {
    let factor = [P_V0, P_KAPPA, P_THETA, P_SIGMA, P_RHO];
    self.reflections = periodic_slopes(params, &[factor, factor].concat());
    self.params = DoubleHestonParams::from(params.to_owned()).projected();
  }

  fn params(&self) -> Array1<f64> {
    self.effective_params().into()
  }

  fn residuals(&mut self) -> Option<Array1<f64>> {
    let params_eff = self.effective_params();
    let model = self.compute_model_prices_for(&params_eff);
    let residuals = self.market_minus(&model);
    let options = &self.calibrator.options;
    self
      .trail
      .stage(options, &self.market, &model, &params_eff, residuals.view());
    Some(residuals)
  }

  fn jacobian(&self) -> Option<Array2<f64>> {
    let jacobian = self.numeric_jacobian(&self.effective_params());
    Some(scale_columns(jacobian, self.reflections.view()))
  }

  fn accept(&mut self) {
    self.trail.accept();
  }
}

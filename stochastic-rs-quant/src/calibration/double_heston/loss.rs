use std::f64::consts::FRAC_1_PI;

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use num_complex::Complex64;

use super::super::integrate_gl_to_convergence;
use super::calibrator::DoubleHestonCalibrator;
use super::params::BOXED;
use super::params::DoubleHestonParams;
use super::params::FOLD;
use crate::OptionType;
use crate::calibration::clip_negative;
use crate::calibration::feller_share::natural_jacobian;
use crate::calibration::least_squares::LeastSquaresProblem;
use crate::calibration::least_squares::Trail;
use crate::calibration::periodic_slopes;
use crate::calibration::scale_columns;
use crate::calibration::stencil::derivative;
use crate::calibration::stencil::step;
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
  /// The folded point: each `sigma` holds its σ's share of the way to Feller's cap.
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
      params: start.to_share(),
      reflections: Array1::ones(10),
      market: calibrator.quotes.iter().map(|quote| quote.price).collect(),
      trail: Trail::new(),
    }
  }

  pub(super) fn effective_params(&self) -> DoubleHestonParams {
    self.params.at_share()
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

  /// `∂r/∂x` at the folded point, each stencil point through the σ shares; `scale` times the
  /// shared step.
  pub(super) fn difference_jacobian(&self, scale: f64) -> Array2<f64> {
    let point = Array1::from(self.params);
    let mut jacobian = Array2::zeros((self.market.len(), point.len()));
    for (j, mut column) in jacobian.columns_mut().into_iter().enumerate() {
      let x = point[j];
      column.assign(&derivative(x, scale * step(x), BOXED[j], |value| {
        let mut moved = point.clone();
        moved[j] = value;
        self.residuals_for(&DoubleHestonParams::from(moved).at_share())
      }));
    }
    jacobian
  }
}

impl LeastSquaresProblem for DoubleHestonProblem<'_> {
  fn set_params(&mut self, params: ArrayView1<f64>) {
    self.reflections = periodic_slopes(params, &FOLD);
    self.params = DoubleHestonParams::from(params.to_owned()).folded();
  }

  fn params(&self) -> Array1<f64> {
    self.params.into()
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
    let jacobian = self.difference_jacobian(1.0);
    Some(scale_columns(jacobian, self.reflections.view()))
  }

  fn accept(&mut self) {
    self.trail.accept();
  }

  fn natural_params(&self) -> Vec<f64> {
    Array1::from(self.effective_params()).to_vec()
  }

  fn natural_jacobian(&self) -> Array2<f64> {
    let [first, second] = self.params.share_slopes();
    natural_jacobian(self.reflections.view(), &[(1, first), (6, second)])
  }
}

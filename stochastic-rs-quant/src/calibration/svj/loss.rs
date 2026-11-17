use std::f64::consts::FRAC_1_PI;

use ndarray::Array1;
use ndarray::Array2;
use num_complex::Complex64;

use super::SvjParams;
use super::calibrator::KAPPA_MIN;
use super::calibrator::SIGMA_V_MIN;
use super::calibrator::SvjCalibrator;
use super::calibrator::THETA_MIN;
use super::params::P_MU_J;
use super::params::P_SIGMA_J;
use super::params::RHO_BOUND;
use crate::OptionType;
use crate::calibration::clip_negative;
use crate::calibration::integrate_gl_to_convergence;
use crate::calibration::least_squares::LeastSquaresProblem;
use crate::calibration::least_squares::Trail;

/// Bates/SVJ characteristic function $\phi_T(\xi)$.
///
/// $$
/// \phi_T(\xi) = \exp\!\bigl[C(\xi,T)+D(\xi,T)\,v_0+i\xi\ln S\bigr]
/// $$
///
/// where $C$ and $D$ are the Heston terms augmented by the jump compensator:
/// $$
/// C_{\mathrm{Bates}} = C_{\mathrm{Heston}} +
///   T\,\lambda\bigl(e^{i\mu_J\xi-\frac12\sigma_J^2\xi^2}-1\bigr) -
///   i\xi\,T\,\lambda\bigl(e^{\mu_J+\frac12\sigma_J^2}-1\bigr)
/// $$
///
/// The Heston diffusion part uses the Albrecher-Mayer-Schoutens-Tistaert (2007)
/// "Little Heston Trap" form (`g̃ = 1/g_original`, `exp(-d·τ)`) so the
/// log-argument stays on the principal branch at long τ / high `|ρ|`.
pub(super) fn bates_cf(p: &SvjParams, s: f64, r: f64, q: f64, tau: f64, u: Complex64) -> Complex64 {
  let i = Complex64::i();
  let iu = i * u;
  let rq = r - q;

  let xi_h = Complex64::new(p.kappa, 0.0) - p.sigma_v * p.rho * iu;
  let d = (xi_h * xi_h + p.sigma_v * p.sigma_v * (u * u + iu)).sqrt();

  let exp_neg_dt = (-d * tau).exp();
  let g = (xi_h - d) / (xi_h + d);

  let sigma_v2 = p.sigma_v * p.sigma_v;

  let c_heston = iu * (s.ln() + rq * tau)
    + (p.kappa * p.theta / sigma_v2)
      * ((xi_h - d) * tau - 2.0 * ((1.0 - g * exp_neg_dt) / (1.0 - g)).ln());

  let d_heston = ((xi_h - d) / sigma_v2) * ((1.0 - exp_neg_dt) / (1.0 - g * exp_neg_dt));

  let k_bar = (p.mu_j + 0.5 * p.sigma_j * p.sigma_j).exp() - 1.0;
  let jump_cf = p.lambda * ((i * p.mu_j * u - 0.5 * p.sigma_j * p.sigma_j * u * u).exp() - 1.0);
  let jump_drift = -p.lambda * k_bar * iu;

  let log_phi = c_heston + d_heston * p.v0 + (jump_cf + jump_drift) * tau;
  log_phi.exp()
}

/// Price a European call option under the Bates/SVJ model using a
/// convergence-controlled Gauss-Legendre quadrature.
pub(super) fn bates_call_price(p: &SvjParams, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
  let Some([integral]) = integrate_gl_to_convergence(
    |u_real| {
      let xi = Complex64::new(u_real, 0.0);
      let xi_shift = Complex64::new(u_real, -1.0);
      let phi = bates_cf(p, s, r, q, tau, xi);
      let phi_shift = bates_cf(p, s, r, q, tau, xi_shift);
      let kernel = (Complex64::new(0.0, -u_real * k.ln())).exp()
        / (Complex64::i() * Complex64::new(u_real, 0.0));
      Some([(kernel * (phi_shift - k * phi)).re])
    },
    1e-8,
  ) else {
    return f64::NAN;
  };

  let disc_r = (-r * tau).exp();
  let disc_q = (-q * tau).exp();
  let call = 0.5 * (s * disc_q - k * disc_r) + disc_r * FRAC_1_PI * integral;
  clip_negative(call)
}

/// One fit: the current point, the quotes' prices and the trail.
pub(super) struct SvjProblem<'a> {
  calibrator: &'a SvjCalibrator,
  params: SvjParams,
  pub(super) market: Vec<f64>,
  pub(super) trail: Trail<SvjParams>,
}

impl<'a> SvjProblem<'a> {
  pub(super) fn new(calibrator: &'a SvjCalibrator, start: SvjParams) -> Self {
    Self {
      calibrator,
      params: start,
      market: calibrator.quotes.iter().map(|quote| quote.price).collect(),
      trail: Trail::new(),
    }
  }

  pub(super) fn effective_params(&self) -> SvjParams {
    self.params.projected()
  }

  pub(super) fn compute_model_prices_for(&self, p: &SvjParams) -> Vec<f64> {
    let SvjCalibrator { s, r, q, .. } = *self.calibrator;
    self
      .calibrator
      .quotes
      .iter()
      .map(|quote| {
        let (k, tau) = (quote.strike, quote.tau);
        let call = bates_call_price(p, s, k, r, q, tau);
        match quote.option_type {
          OptionType::Call => call,
          OptionType::Put => clip_negative(call - s * (-q * tau).exp() + k * (-r * tau).exp()),
        }
      })
      .collect()
  }

  fn residuals_for(&self, p: &SvjParams) -> Array1<f64> {
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

  /// Central finite-difference Jacobian.
  fn numeric_jacobian(&self, params: &SvjParams) -> Array2<f64> {
    let n = self.calibrator.quotes.len();
    let p = 8usize;

    let base_params_vec: Array1<f64> = (*params).into();
    let mut j_mat = Array2::zeros((n, p));

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

impl LeastSquaresProblem for SvjProblem<'_> {
  fn set_params(&mut self, params: &Array1<f64>) {
    self.params = SvjParams::from(params.clone()).projected();
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
      .stage(options, &self.market, &model, &params_eff, &residuals);
    Some(residuals)
  }

  fn jacobian(&self) -> Option<Array2<f64>> {
    Some(self.numeric_jacobian(&self.effective_params()))
  }

  fn accept(&mut self) {
    self.trail.accept();
  }
}

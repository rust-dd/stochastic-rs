use std::f64::consts::FRAC_1_PI;

use nalgebra::DMatrix;
use nalgebra::DVector;
use num_complex::Complex64;

use super::super::integrate_gl_to_convergence;
use super::super::periodic_map;
use super::calibrator::DoubleHestonCalibrator;
use super::params::DoubleHestonParams;
use super::params::KAPPA_MIN;
use super::params::P_KAPPA;
use super::params::P_SIGMA;
use super::params::P_THETA;
use super::params::P_V0;
use super::params::RHO_BOUND;
use super::params::SIGMA_MIN;
use super::params::THETA_MIN;
use crate::OptionType;

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

  let (c1, d1) = factor_cd(p.kappa1, p.theta1, p.sigma1, p.rho1, tau, u);
  let (c2, d2) = factor_cd(p.kappa2, p.theta2, p.sigma2, p.rho2, tau, u);

  (c1 + c2 + d1 * p.v1_0 + d2 * p.v2_0 + i * u * (r - q) * tau).exp()
}

/// Single Heston factor $(C_j, D_j)$ at Fourier argument `u`.
///
/// Uses the Albrecher-Mayer-Schoutens-Tistaert (2007) "Little Heston Trap"
/// form (`g̃ = 1/g_original`, `exp(-d·τ)`) so each independent factor stays on
/// the principal log-branch at long τ / high `|ρ_j|`.
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
  if call.is_finite() {
    call.max(0.0)
  } else {
    call
  }
}

impl DoubleHestonCalibrator {
  pub(super) fn compute_model_prices_for(&self, p: &DoubleHestonParams) -> DVector<f64> {
    let n = self.c_market.len();
    let mut c_model = DVector::zeros(n);
    let q_val = self.q.unwrap_or(0.0);

    for idx in 0..n {
      let tau = self.flat_t[idx];
      let call = double_heston_call_price(p, self.s[idx], self.k[idx], self.r, q_val, tau);
      c_model[idx] = match self.option_type {
        OptionType::Call => call,
        OptionType::Put => {
          let put = call - self.s[idx] * (-q_val * tau).exp() + self.k[idx] * (-self.r * tau).exp();
          if put.is_finite() { put.max(0.0) } else { put }
        }
      };
    }

    c_model
  }

  pub(super) fn residuals_for(&self, p: &DoubleHestonParams) -> DVector<f64> {
    self.c_market.clone() - self.compute_model_prices_for(p)
  }

  /// Central finite-difference Jacobian over the 10 parameters.
  pub(super) fn numeric_jacobian(&self, params: &DoubleHestonParams) -> DMatrix<f64> {
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

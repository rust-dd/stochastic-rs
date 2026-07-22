use std::f64::consts::FRAC_1_PI;

use nalgebra::DMatrix;
use nalgebra::DVector;
use num_complex::Complex64;

use super::calibrator::HestonCalibrator;
use super::params::CuiCfTerms;
use super::params::EPS;
use super::params::HestonParams;
use super::params::finite_c64;
use crate::OptionType;
use crate::calibration::integrate_gl_to_convergence;

impl HestonCalibrator {
  pub(super) fn cui_terms_for(
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

  pub(super) fn cui_da_db_ddlog(
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

  pub(super) fn cui_price_and_grad_for_quote(
    &self,
    params: &HestonParams,
    s: f64,
    k: f64,
    tau: f64,
  ) -> Option<(f64, [f64; 5])> {
    let sigma = params.sigma;
    let sigma2 = sigma * sigma;
    let sigma3 = sigma2 * sigma;

    let integrals = integrate_gl_to_convergence(
      |u_real| {
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
        let d_prime_sigma_shift = (sigma * terms_shift.u2_iu
          - params.rho * terms_shift.iu * terms_shift.xi)
          / terms_shift.d;
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

        Some([
          (k_kernel * (terms_shift.phi - k * terms_u.phi)).re,
          (k_kernel * (dphi_shift[0] - k * dphi_u[0])).re,
          (k_kernel * (dphi_shift[1] - k * dphi_u[1])).re,
          (k_kernel * (dphi_shift[2] - k * dphi_u[2])).re,
          (k_kernel * (dphi_shift[3] - k * dphi_u[3])).re,
          (k_kernel * (dphi_shift[4] - k * dphi_u[4])).re,
        ])
      },
      1e-8,
    )?;

    let disc_r = (-self.r * tau).exp();
    let disc_q = (-self.q.unwrap_or(0.0) * tau).exp();
    let call = 0.5 * (s * disc_q - k * disc_r) + disc_r * FRAC_1_PI * integrals[0];
    let mut grad = [0.0_f64; 5];
    for j in 0..5 {
      grad[j] = disc_r * FRAC_1_PI * integrals[1 + j];
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
  pub(super) fn compute_model_prices_and_residual_jacobian_cui(
    &self,
    params: &HestonParams,
  ) -> Option<(DVector<f64>, DMatrix<f64>)> {
    let n = self.c_market.len();
    let mut c_model = DVector::zeros(n);
    let mut j_residual = DMatrix::zeros(n, 5);

    for idx in 0..n {
      let s = self.s[idx];
      let k = self.k[idx];
      let tau_idx = self.flat_t[idx];
      let (call_raw, grad_call) = self.cui_price_and_grad_for_quote(params, s, k, tau_idx)?;
      let disc_r = (-self.r * tau_idx).exp();
      let disc_q = (-self.q.unwrap_or(0.0) * tau_idx).exp();
      let put_raw = call_raw - s * disc_q + k * disc_r;

      let (model_raw, grad_model) = match self.option_type {
        OptionType::Call => (call_raw, grad_call),
        OptionType::Put => (put_raw, grad_call),
      };

      if model_raw > 0.0 {
        c_model[idx] = model_raw;
        for col in 0..5 {
          j_residual[(idx, col)] = -grad_model[col];
        }
      } else {
        c_model[idx] = 0.0;
      }
    }

    Some((c_model, j_residual))
  }

  pub(super) fn compute_model_prices_for_cui(&self, params: &HestonParams) -> Option<DVector<f64>> {
    self
      .compute_model_prices_and_residual_jacobian_cui(params)
      .map(|(c_model, _)| c_model)
  }
}

use std::f64::consts::FRAC_1_PI;

use num_complex::Complex64;

use super::EPS;
use super::types::LevyModelType;
use crate::pricing::cf_quadrature::integrate_to_convergence;

/// Compute the Lévy characteristic exponent $\psi(\xi)$ such that
/// $\phi_T(\xi) = \exp\bigl(i\xi (r-q)T + T\,\psi(\xi) - T\,\psi(-i)\bigr)$.
///
/// The martingale correction $-T\,\psi(-i)$ ensures that $E[S_T] = S_0 e^{(r-q)T}$.
pub(super) fn levy_char_exponent(
  model_type: LevyModelType,
  params: &[f64],
  xi: Complex64,
) -> Complex64 {
  let i = Complex64::i();
  match model_type {
    LevyModelType::VarianceGamma => {
      let sigma = params[0];
      let theta = params[1];
      let nu = params[2];
      let inner = Complex64::new(1.0, 0.0) - i * theta * nu * xi
        + Complex64::new(0.5 * sigma * sigma * nu, 0.0) * xi * xi;
      -inner.ln() / nu
    }
    LevyModelType::Nig => {
      let alpha = params[0];
      let beta = params[1];
      let delta = params[2];
      let a2 = alpha * alpha;
      let b2 = beta * beta;
      let base = Complex64::new(a2 - b2, 0.0).sqrt();
      let shifted = Complex64::new(beta, 0.0) + i * xi;
      let branch = (Complex64::new(a2, 0.0) - shifted * shifted).sqrt();
      delta * (base - branch)
    }
    LevyModelType::Cgmy => {
      let c = params[0];
      let g = params[1];
      let m = params[2];
      let y = params[3];
      let gamma_neg_y = gamma_neg_y_fn(y);
      let m_shift = (Complex64::new(m, 0.0) - i * xi).powf(y);
      let g_shift = (Complex64::new(g, 0.0) + i * xi).powf(y);
      let m_y = Complex64::new(m.powf(y), 0.0);
      let g_y = Complex64::new(g.powf(y), 0.0);
      c * gamma_neg_y * (m_shift - m_y + g_shift - g_y)
    }
    LevyModelType::MertonJD => {
      let sigma = params[0];
      let lambda = params[1];
      let mu_j = params[2];
      let sigma_j = params[3];
      let diffusion = Complex64::new(-0.5 * sigma * sigma, 0.0) * xi * xi;
      let jump_exp = (i * mu_j * xi - Complex64::new(0.5 * sigma_j * sigma_j, 0.0) * xi * xi).exp();
      diffusion + lambda * (jump_exp - 1.0)
    }
    LevyModelType::Kou => {
      let sigma = params[0];
      let lambda = params[1];
      let p = params[2];
      let eta1 = params[3];
      let eta2 = params[4];
      let diffusion = Complex64::new(-0.5 * sigma * sigma, 0.0) * xi * xi;
      let up = p * eta1 / (Complex64::new(eta1, 0.0) - i * xi);
      let dn = (1.0 - p) * eta2 / (Complex64::new(eta2, 0.0) + i * xi);
      diffusion + lambda * (up + dn - 1.0)
    }
  }
}

/// $\Gamma(-Y)$ for Cgmy, using reflection formula.
fn gamma_neg_y_fn(y: f64) -> Complex64 {
  if y.abs() < EPS {
    return Complex64::new(1e15, 0.0);
  }
  if y < 0.0 {
    Complex64::new(stochastic_rs_distributions::special::gamma(-y), 0.0)
  } else if (y - 1.0).abs() < EPS {
    Complex64::new(stochastic_rs_distributions::special::gamma(-0.999), 0.0)
  } else {
    let g = stochastic_rs_distributions::special::gamma(y);
    let sin_val = (std::f64::consts::PI * y).sin();
    if sin_val.abs() < EPS || g.abs() < EPS {
      Complex64::new(1e15, 0.0)
    } else {
      Complex64::new(-std::f64::consts::PI / (y * sin_val * g), 0.0)
    }
  }
}

/// Price a European call option using the damped Lewis inversion integral.
///
/// $$
/// C = S e^{-qT} - \frac{\sqrt{SK}e^{-rT}}{\pi}
/// \int_0^\infty \operatorname{Re}\!\left[
/// \frac{e^{-iu\log(K/S)}\phi_T(u-i/2)}{u^2+1/4}
/// \right]du.
/// $$
pub(super) fn fourier_call_price(
  model_type: LevyModelType,
  params: &[f64],
  s: f64,
  k: f64,
  r: f64,
  q: f64,
  t: f64,
) -> f64 {
  let ln_ks = (k / s).ln();
  let rq = r - q;

  // The correction enforces E[S_T] = S_0 exp((r-q)T).
  let psi_neg_i = levy_char_exponent(model_type, params, Complex64::new(0.0, -1.0));
  let omega = -psi_neg_i;

  let integral = integrate_to_convergence(
    |u| {
      let xi = Complex64::new(u, -0.5);
      let psi = levy_char_exponent(model_type, params, xi);
      let log_cf = Complex64::i() * xi * ((rq + omega.re) * t) + psi * t;
      let phi = log_cf.exp();
      let kernel = (Complex64::new(0.0, -u * ln_ks)).exp() / (u * u + 0.25);
      (kernel * phi).re
    },
    0.0,
    1e-9,
  );

  let disc_r = (-r * t).exp();
  let disc_q = (-q * t).exp();

  let call = s * disc_q - (s * k).sqrt() * disc_r * FRAC_1_PI * integral;
  if call.is_finite() {
    call.max(0.0)
  } else {
    call
  }
}

/// Price a European option (call or put) via put-call parity from the Fourier call price.
pub(super) fn fourier_option_price(
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
    let put = call - s * (-q * t).exp() + k * (-r * t).exp();
    if put.is_finite() { put.max(0.0) } else { put }
  }
}

pub(super) fn param_count(model_type: LevyModelType) -> usize {
  match model_type {
    LevyModelType::VarianceGamma => 3,
    LevyModelType::Nig => 3,
    LevyModelType::Cgmy => 4,
    LevyModelType::MertonJD => 4,
    LevyModelType::Kou => 5,
  }
}

pub(super) fn param_bounds(model_type: LevyModelType) -> Vec<(f64, f64)> {
  match model_type {
    LevyModelType::VarianceGamma => {
      vec![(0.01, 2.0), (-1.0, 1.0), (0.01, 5.0)]
    }
    LevyModelType::Nig => {
      vec![(0.01, 50.0), (-50.0, 50.0), (0.001, 5.0)]
    }
    LevyModelType::Cgmy => {
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

pub(super) fn default_params(model_type: LevyModelType) -> Vec<f64> {
  match model_type {
    LevyModelType::VarianceGamma => vec![0.2, -0.1, 0.5],
    LevyModelType::Nig => vec![15.0, -5.0, 0.5],
    LevyModelType::Cgmy => vec![1.0, 10.0, 15.0, 0.5],
    LevyModelType::MertonJD => vec![0.15, 1.0, -0.05, 0.1],
    LevyModelType::Kou => vec![0.15, 3.0, 0.5, 10.0, 10.0],
  }
}

/// Project parameters into their admissible bounds.
pub(super) fn project_params(model_type: LevyModelType, params: &mut [f64]) {
  let bounds = param_bounds(model_type);
  for (p, (lo, hi)) in params.iter_mut().zip(bounds.iter()) {
    *p = p.clamp(*lo, *hi);
  }
  // The NIG exponent is defined only when alpha > |beta|.
  if model_type == LevyModelType::Nig {
    let alpha = params[0];
    let beta = params[1];
    if alpha <= beta.abs() {
      params[0] = beta.abs() + 0.01;
    }
  }
}

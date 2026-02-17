//! # Heston NMLE-CEKF
//!
//! $$
//! \hat V_{k+1}=\bar V_{k+1}+K_{k+1}\bigl(z_{k+1}-h(\bar V_{k+1})\bigr)
//! $$
//!
use ndarray::Array1;

use crate::stats::heston_mle::HestonMleResult;
use crate::stats::heston_mle::nmle_heston_with_delta;

const EPS: f64 = 1e-12;
const RHO_MAX: f64 = 0.9999;

#[derive(Clone, Copy, Debug)]
pub struct HestonNMLECEKFParams {
  pub kappa: f64,
  pub theta: f64,
  pub sigma: f64,
  pub rho: f64,
}

impl Default for HestonNMLECEKFParams {
  fn default() -> Self {
    Self {
      kappa: 1.5,
      theta: 0.04,
      sigma: 0.4,
      rho: -0.5,
    }
  }
}

impl From<HestonMleResult> for HestonNMLECEKFParams {
  fn from(v: HestonMleResult) -> Self {
    Self {
      kappa: v.kappa,
      theta: v.theta,
      sigma: v.sigma,
      rho: v.rho,
    }
  }
}

impl HestonNMLECEKFParams {
  fn projected(mut self) -> Self {
    self.kappa = self.kappa.abs().max(EPS);
    self.theta = self.theta.abs().max(EPS);
    self.sigma = self.sigma.abs().max(EPS);
    self.rho = self.rho.clamp(-RHO_MAX, RHO_MAX);
    self
  }
}

#[derive(Clone, Debug)]
pub struct HestonNMLECEKFConfig {
  pub r: f64,
  pub delta: f64,
  pub max_iters: usize,
  pub tol: f64,
  pub param_damping: f64,
  pub initial_v0: f64,
  pub initial_p0: f64,
  pub initial_params: HestonNMLECEKFParams,
  pub q11: f64,
  pub q12: f64,
  pub q22: f64,
  /// Enables consistent-EKF correction terms (`ΔQ_k`, `ΔR_{k+1}`) from NMLE-CEKF.
  ///
  /// Source:
  /// - Wang et al. (2018), NMLE-CEKF consistent terms
  ///   https://doi.org/10.1007/s11432-017-9215-8
  ///   http://scis.scichina.com/en/2018/042202.pdf
  pub use_consistent_terms: bool,
}

impl Default for HestonNMLECEKFConfig {
  fn default() -> Self {
    Self {
      r: 0.0,
      delta: 1.0 / 252.0,
      max_iters: 12,
      tol: 1e-6,
      param_damping: 0.7,
      initial_v0: 0.04,
      initial_p0: 0.1,
      initial_params: HestonNMLECEKFParams::default(),
      q11: 1.0,
      q12: 0.0,
      q22: 1.0,
      use_consistent_terms: true,
    }
  }
}

#[derive(Clone, Debug)]
pub struct HestonNMLECEKFResult {
  pub params: HestonMleResult,
  pub vol_path: Array1<f64>,
  pub cov_path: Array1<f64>,
  pub iterations: usize,
  pub converged: bool,
}

fn blend_params(
  old: HestonNMLECEKFParams,
  new: HestonNMLECEKFParams,
  alpha: f64,
) -> HestonNMLECEKFParams {
  let w = alpha.clamp(0.0, 1.0);
  HestonNMLECEKFParams {
    kappa: (1.0 - w) * old.kappa + w * new.kappa,
    theta: (1.0 - w) * old.theta + w * new.theta,
    sigma: (1.0 - w) * old.sigma + w * new.sigma,
    rho: (1.0 - w) * old.rho + w * new.rho,
  }
}

fn cekf_pass(
  s: &Array1<f64>,
  params: HestonNMLECEKFParams,
  cfg: &HestonNMLECEKFConfig,
) -> (Array1<f64>, Array1<f64>) {
  let n_obs = s.len();
  let n = n_obs - 1;

  let mut v_hat = Array1::<f64>::zeros(n_obs);
  let mut p_hat = Array1::<f64>::zeros(n_obs);
  v_hat[0] = cfg.initial_v0.max(EPS);
  p_hat[0] = cfg.initial_p0.max(EPS);

  for k in 0..n {
    let v_k = v_hat[k].max(EPS);
    let p_k = p_hat[k].max(EPS);

    let f = 1.0 - params.kappa * cfg.delta;
    let l2 = params.sigma * (v_k * cfg.delta).max(EPS).sqrt();

    let l_q_l = cfg.q22 * l2 * l2;
    let delta_q = if cfg.use_consistent_terms {
      let bound_q = p_k * f.abs().powi(2)
        + (cfg.delta * (params.kappa * params.theta).abs()).powi(2)
        + params.sigma.abs().powi(2) * cfg.delta * v_k * cfg.q22;
      (bound_q - (f * f * p_k + l_q_l)).max(0.0)
    } else {
      0.0
    };

    let bar_p = (f * f * p_k + l_q_l + delta_q).max(EPS);
    let bar_v = (v_k + params.kappa * (params.theta - v_k) * cfg.delta).max(EPS);

    // z_k = ln S_{k+1} - ln S_k, h(V) = (r - 0.5 V) dt
    let h = -0.5 * cfg.delta;
    let sqrt_vdt = (bar_v * cfg.delta).max(EPS).sqrt();
    let m1 = ((1.0 - params.rho * params.rho).max(0.0)).sqrt() * sqrt_vdt;
    let m2 = params.rho * sqrt_vdt;

    let m_q_m = cfg.q11 * m1 * m1 + 2.0 * cfg.q12 * m1 * m2 + cfg.q22 * m2 * m2;
    let l_q_m = l2 * (cfg.q12 * m1 + cfg.q22 * m2);
    let xi = (h * h * bar_p + m_q_m + 2.0 * h * l_q_m).max(EPS);
    let k_gain = (bar_p * h + l_q_m) / xi;

    let z_obs = s[k + 1].max(EPS).ln() - s[k].max(EPS).ln();
    let innovation = z_obs - (cfg.r - 0.5 * bar_v) * cfg.delta;
    let v_next = (bar_v + k_gain * innovation).max(EPS);

    let m_l_t = m2 * l2;
    let delta_r = if cfg.use_consistent_terms {
      let q_mix =
        (1.0 - params.rho * params.rho).max(0.0) * cfg.q11 + params.rho * params.rho * cfg.q22;
      let bound_r = bar_p * (1.0 + 0.5 * k_gain * cfg.delta).powi(2)
        + 2.0 * k_gain * k_gain * cfg.delta * bar_v * q_mix
        - bar_p
        + k_gain * (h * bar_p + m_l_t);
      bound_r.max(0.0)
    } else {
      0.0
    };

    let p_next = (bar_p - k_gain * (h * bar_p + m_l_t) + delta_r).max(EPS);

    v_hat[k + 1] = v_next;
    p_hat[k + 1] = p_next;
  }

  (v_hat, p_hat)
}

/// Heston NMLE-CEKF loop:
/// 1) CEKF latent variance filtering
/// 2) NMLE parameter refresh
/// 3) Damped fixed-point iteration
///
/// Source:
/// - Wang et al. (2018), NMLE-CEKF framework
///   https://doi.org/10.1007/s11432-017-9215-8
///   http://scis.scichina.com/en/2018/042202.pdf
pub fn nmle_cekf_heston(s: Array1<f64>, cfg: HestonNMLECEKFConfig) -> HestonNMLECEKFResult {
  assert!(s.len() >= 2, "nmle_cekf_heston requires at least 2 prices");
  assert!(
    cfg.delta.is_finite() && cfg.delta > 0.0,
    "delta must be finite and positive"
  );
  assert!(cfg.max_iters > 0, "max_iters must be positive");
  assert!(cfg.tol.is_finite() && cfg.tol > 0.0, "tol must be positive");
  assert!(
    cfg.initial_v0.is_finite() && cfg.initial_v0 > 0.0,
    "initial_v0 must be positive"
  );
  assert!(
    cfg.initial_p0.is_finite() && cfg.initial_p0 > 0.0,
    "initial_p0 must be positive"
  );
  assert!(
    cfg.q11 >= 0.0 && cfg.q22 >= 0.0,
    "q11 and q22 must be non-negative"
  );
  assert!(
    cfg.q11 * cfg.q22 - cfg.q12 * cfg.q12 >= -1e-12,
    "noise covariance must be positive semidefinite"
  );

  let mut params = cfg.initial_params.projected();
  let mut converged = false;
  let mut iters = 0;

  for i in 0..cfg.max_iters {
    let (v_hat, _p_hat) = cekf_pass(&s, params, &cfg);
    let nmle = nmle_heston_with_delta(s.clone(), v_hat.clone(), cfg.r, cfg.delta);
    let updated = HestonNMLECEKFParams::from(nmle).projected();
    let blended = blend_params(params, updated, cfg.param_damping).projected();

    let max_diff = (blended.kappa - params.kappa)
      .abs()
      .max((blended.theta - params.theta).abs())
      .max((blended.sigma - params.sigma).abs())
      .max((blended.rho - params.rho).abs());

    params = blended;
    iters = i + 1;

    if max_diff < cfg.tol {
      converged = true;
      break;
    }
  }

  // Keep output state trajectory consistent with the reported final parameters.
  let (v_final, p_final) = cekf_pass(&s, params, &cfg);
  let mle_final = nmle_heston_with_delta(s, v_final.clone(), cfg.r, cfg.delta);
  let p = HestonNMLECEKFParams::from(mle_final).projected();

  HestonNMLECEKFResult {
    params: HestonMleResult {
      v0: v_final[0].max(0.0),
      kappa: p.kappa,
      theta: p.theta,
      sigma: p.sigma,
      rho: p.rho,
    },
    vol_path: v_final,
    cov_path: p_final,
    iterations: iters,
    converged,
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use rand::SeedableRng;
  use rand::rngs::StdRng;
  use rand_distr::Distribution;
  use rand_distr::StandardNormal;

  use super::HestonNMLECEKFConfig;
  use super::HestonNMLECEKFParams;
  use super::nmle_cekf_heston;

  fn simulate_heston_prices(
    n_obs: usize,
    delta: f64,
    r: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
  ) -> (Array1<f64>, Array1<f64>) {
    let mut s = Array1::<f64>::zeros(n_obs);
    let mut v = Array1::<f64>::zeros(n_obs);
    s[0] = 100.0;
    v[0] = theta.max(1e-8);

    let mut rng = StdRng::seed_from_u64(42);
    let normal = StandardNormal;

    for i in 1..n_obs {
      let zs: f64 = normal.sample(&mut rng);
      let zv: f64 = normal.sample(&mut rng);

      let dw_s = delta.sqrt() * zs;
      let dw_v = delta.sqrt() * zv;

      let v_prev = v[i - 1].max(1e-10);
      let v_next = v_prev + kappa * (theta - v_prev) * delta + sigma * v_prev.sqrt() * dw_v;
      v[i] = v_next.max(1e-10);

      let log_s_next = s[i - 1].ln()
        + (r - 0.5 * v_prev) * delta
        + ((1.0 - rho * rho).max(0.0) * v_prev).sqrt() * dw_s
        + rho * v_prev.sqrt() * dw_v;
      s[i] = log_s_next.exp();
    }

    (s, v)
  }

  #[test]
  fn nmle_cekf_heston_runs_and_returns_finite_outputs() {
    let n_obs = 320usize;
    let delta = 1.0 / (n_obs as f64 - 1.0);
    let r = 0.01;
    let (s, _v_true) = simulate_heston_prices(n_obs, delta, r, 1.5, 0.04, 0.35, -0.6);

    let cfg = HestonNMLECEKFConfig {
      r,
      delta,
      max_iters: 8,
      tol: 1e-5,
      param_damping: 0.6,
      initial_v0: 0.06,
      initial_p0: 0.2,
      initial_params: HestonNMLECEKFParams {
        kappa: 1.0,
        theta: 0.03,
        sigma: 0.4,
        rho: -0.3,
      },
      ..HestonNMLECEKFConfig::default()
    };

    let out = nmle_cekf_heston(s, cfg);

    assert_eq!(out.vol_path.len(), n_obs);
    assert_eq!(out.cov_path.len(), n_obs);
    assert!(out.iterations >= 1);
    assert!(out.vol_path.iter().all(|x| x.is_finite() && *x > 0.0));
    assert!(out.cov_path.iter().all(|x| x.is_finite() && *x > 0.0));
    assert!(out.params.kappa.is_finite() && out.params.kappa > 0.0);
    assert!(out.params.theta.is_finite() && out.params.theta > 0.0);
    assert!(out.params.sigma.is_finite() && out.params.sigma > 0.0);
    assert!(out.params.rho.is_finite() && out.params.rho.abs() <= 1.0);
  }

  #[test]
  fn nmle_cekf_heston_runs_without_consistent_terms() {
    let n_obs = 256usize;
    let delta = 1.0 / (n_obs as f64 - 1.0);
    let r = 0.0;
    let (s, _v_true) = simulate_heston_prices(n_obs, delta, r, 1.2, 0.05, 0.25, -0.4);

    let cfg = HestonNMLECEKFConfig {
      r,
      delta,
      max_iters: 5,
      tol: 1e-5,
      use_consistent_terms: false,
      ..HestonNMLECEKFConfig::default()
    };

    let out = nmle_cekf_heston(s, cfg);
    assert!(out.vol_path.iter().all(|x| x.is_finite() && *x > 0.0));
    assert!(out.params.kappa > 0.0);
    assert!(out.params.theta > 0.0);
    assert!(out.params.sigma > 0.0);
  }
}

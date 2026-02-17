//! # Mle
//!
//! $$
//! \hat\theta=\arg\max_\theta \sum_{i=1}^n \log f(x_i\mid\theta)
//! $$
//!
use ndarray::Array1;

#[derive(Clone, Debug)]
pub struct HestonMleResult {
  /// Initial variance/volatility level.
  pub v0: f64,
  /// Mean-reversion speed parameter.
  pub kappa: f64,
  /// Long-run target level parameter.
  pub theta: f64,
  /// Volatility / diffusion scale parameter.
  pub sigma: f64,
  /// Correlation parameter.
  pub rho: f64,
}

/// NMLE estimator for Heston parameters from observed `(S_t, V_t)` samples.
///
/// Source:
/// - Wang et al. (2018), NMLE closed-form estimators
///   https://doi.org/10.1007/s11432-017-9215-8
///   http://scis.scichina.com/en/2018/042202.pdf
pub fn nmle_heston(s: Array1<f64>, v: Array1<f64>, r: f64) -> HestonMleResult {
  assert_eq!(s.len(), v.len(), "s and v must have the same length");
  let n_obs = v.len();
  assert!(n_obs >= 2, "nmle_heston requires at least 2 observations");
  let delta = 1.0 / (n_obs - 1) as f64;
  nmle_heston_with_delta(s, v, r, delta)
}

/// NMLE estimator with explicit sampling interval `delta`.
///
/// Source:
/// - Wang et al. (2018), Eq. (19)-(21) and rho moment estimator
///   https://doi.org/10.1007/s11432-017-9215-8
///   http://scis.scichina.com/en/2018/042202.pdf
pub fn nmle_heston_with_delta(
  s: Array1<f64>,
  v: Array1<f64>,
  r: f64,
  delta: f64,
) -> HestonMleResult {
  const EPS: f64 = 1e-12;

  assert_eq!(s.len(), v.len(), "s and v must have the same length");
  let n_obs = v.len();
  assert!(n_obs >= 2, "nmle_heston requires at least 2 observations");
  assert!(
    delta.is_finite() && delta > 0.0,
    "delta must be finite and positive"
  );

  let n_f = (n_obs - 1) as f64;

  // Following Wang et al. (2018), Eq. (19)-(21):
  // P = kappa * theta - sigma^2 / 4, with closed-form estimators from sqrt-variance dynamics.
  let mut sum_inv_v_prev = 0.0;
  let mut sum_v_prev = 0.0;
  let mut sum_sqrt_ratio = 0.0;
  let mut sum_sqrt_prod = 0.0;

  for i in 1..n_obs {
    let v_prev = v[i - 1].max(EPS);
    let v_curr = v[i].max(EPS);

    sum_inv_v_prev += 1.0 / v_prev;
    sum_v_prev += v_prev;
    sum_sqrt_ratio += (v_curr / v_prev).sqrt();
    sum_sqrt_prod += (v_curr * v_prev).sqrt();
  }

  let denom_inner = n_f - (sum_v_prev * sum_inv_v_prev) / n_f;
  let denom_p = 0.5 * delta * denom_inner;
  let p_hat = if denom_p.abs() > EPS {
    (sum_sqrt_prod - (sum_v_prev * sum_sqrt_ratio) / n_f) / denom_p
  } else {
    0.0
  };

  let mut kappa_hat =
    (2.0 / delta) * (1.0 + 0.5 * delta * p_hat * (sum_inv_v_prev / n_f) - (sum_sqrt_ratio / n_f));
  if !kappa_hat.is_finite() {
    kappa_hat = EPS;
  }
  kappa_hat = kappa_hat.max(EPS);

  let mut sum_res2 = 0.0;
  for i in 1..n_obs {
    let v_prev = v[i - 1].max(EPS);
    let v_curr = v[i].max(EPS);
    let drift_term = (delta / (2.0 * v_prev.sqrt())) * (p_hat - kappa_hat * v_prev);
    let resid = v_curr.sqrt() - v_prev.sqrt() - drift_term;
    sum_res2 += resid * resid;
  }

  let sigma2_hat = (4.0 * sum_res2 / (delta * n_f)).max(EPS);
  let sigma_hat = sigma2_hat.sqrt();

  let mut theta_hat = (p_hat + 0.25 * sigma2_hat) / kappa_hat;
  if !theta_hat.is_finite() {
    theta_hat = v[0].max(EPS);
  }
  theta_hat = theta_hat.max(EPS);

  let mut sum_dw1dw2 = 0.0;
  for i in 1..n_obs {
    let v_prev = v[i - 1].max(EPS);
    let v_curr = v[i].max(EPS);
    let s_prev = s[i - 1].max(EPS);
    let s_curr = s[i].max(EPS);

    let dw1_i = (s_curr.ln() - s_prev.ln() - (r - 0.5 * v_prev) * delta) / v_prev.sqrt();
    let dw2_i =
      (v_curr - v_prev - kappa_hat * (theta_hat - v_prev) * delta) / (sigma_hat * v_prev.sqrt());
    sum_dw1dw2 += dw1_i * dw2_i;
  }

  let rho_hat = (sum_dw1dw2 / (n_f * delta)).clamp(-1.0, 1.0);

  HestonMleResult {
    v0: v[0].max(0.0),
    kappa: kappa_hat,
    theta: theta_hat,
    sigma: sigma_hat,
    rho: rho_hat,
  }
}

/// PMLE estimator for Heston parameters from observed `(S_t, V_t)` samples.
///
/// Source:
/// - Wang et al. (2018), PMLE closed-form estimators
///   https://doi.org/10.1007/s11432-017-9215-8
///   http://scis.scichina.com/en/2018/042202.pdf
pub fn pmle_heston(s: Array1<f64>, v: Array1<f64>, r: f64) -> HestonMleResult {
  assert_eq!(s.len(), v.len(), "s and v must have the same length");
  let n_obs = v.len();
  assert!(n_obs >= 2, "pmle_heston requires at least 2 observations");
  let delta = 1.0 / (n_obs - 1) as f64;
  pmle_heston_with_delta(s, v, r, delta)
}

/// PMLE estimator with explicit sampling interval `delta`.
///
/// Source:
/// - Wang et al. (2018), PMLE formulas and rho moment estimator
///   https://doi.org/10.1007/s11432-017-9215-8
///   http://scis.scichina.com/en/2018/042202.pdf
pub fn pmle_heston_with_delta(
  s: Array1<f64>,
  v: Array1<f64>,
  r: f64,
  delta: f64,
) -> HestonMleResult {
  const EPS: f64 = 1e-12;

  assert_eq!(s.len(), v.len(), "s and v must have the same length");
  let n_obs = v.len();
  assert!(n_obs >= 2, "pmle_heston requires at least 2 observations");
  assert!(
    delta.is_finite() && delta > 0.0,
    "delta must be finite and positive"
  );

  let n_f = (n_obs - 1) as f64;

  let mut sum_v_k = 0.0;
  let mut sum_v_prev = 0.0;
  let mut sum_inv_v_prev = 0.0;
  let mut sum_ratio = 0.0;

  for i in 1..n_obs {
    let v_prev = v[i - 1].max(EPS);
    let v_k = v[i].max(EPS);
    sum_v_k += v_k;
    sum_v_prev += v_prev;
    sum_inv_v_prev += 1.0 / v_prev;
    sum_ratio += v_k / v_prev;
  }

  let mean_v_k = sum_v_k / n_f;
  let mean_v_prev = sum_v_prev / n_f;
  let mean_inv_v_prev = sum_inv_v_prev / n_f;
  let mean_ratio = sum_ratio / n_f;

  let beta1_num = mean_v_k * mean_inv_v_prev - mean_ratio;
  let beta1_den = mean_v_prev * mean_inv_v_prev - 1.0;
  let mut beta1_hat = if beta1_den.abs() > EPS {
    beta1_num / beta1_den
  } else {
    1.0 - 1e-6
  };
  beta1_hat = beta1_hat.clamp(EPS, 1.0 - 1e-6);

  let beta2_den = (1.0 - beta1_hat) * mean_inv_v_prev;
  let mut beta2_hat = if beta2_den.abs() > EPS {
    (mean_ratio - beta1_hat) / beta2_den
  } else {
    mean_v_k
  };
  beta2_hat = beta2_hat.max(EPS);

  let mut sum_beta3 = 0.0;
  for i in 1..n_obs {
    let v_prev = v[i - 1].max(EPS);
    let v_k = v[i].max(EPS);
    let residual = v_k - v_prev * beta1_hat - beta2_hat * (1.0 - beta1_hat);
    sum_beta3 += residual * residual / v_prev;
  }
  let beta3_hat = (sum_beta3 / n_f).max(EPS);

  let kappa_hat = (-(beta1_hat.ln()) / delta).max(EPS);
  let theta_hat = beta2_hat;
  let sigma2_hat =
    ((2.0 * kappa_hat * beta3_hat) / (1.0 - beta1_hat * beta1_hat).max(EPS)).max(EPS);
  let sigma_hat = sigma2_hat.sqrt();

  let mut sum_dw1dw2 = 0.0;
  for i in 1..n_obs {
    let v_prev = v[i - 1].max(EPS);
    let v_k = v[i].max(EPS);
    let s_prev = s[i - 1].max(EPS);
    let s_curr = s[i].max(EPS);

    let dw1_i = (s_curr.ln() - s_prev.ln() - (r - 0.5 * v_prev) * delta) / v_prev.sqrt();
    let dw2_i =
      (v_k - v_prev - kappa_hat * (theta_hat - v_prev) * delta) / (sigma_hat * v_prev.sqrt());
    sum_dw1dw2 += dw1_i * dw2_i;
  }

  let rho_hat = (sum_dw1dw2 / (n_f * delta)).clamp(-1.0, 1.0);

  HestonMleResult {
    v0: v[0].max(0.0),
    kappa: kappa_hat,
    theta: theta_hat,
    sigma: sigma_hat,
    rho: rho_hat,
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use rand::SeedableRng;
  use rand::rngs::StdRng;
  use rand_distr::Distribution;
  use rand_distr::StandardNormal;

  use super::nmle_heston;
  use super::nmle_heston_with_delta;
  use super::pmle_heston;
  use super::pmle_heston_with_delta;

  #[test]
  #[should_panic(expected = "s and v must have the same length")]
  fn nmle_heston_panics_on_mismatched_lengths() {
    let s = Array1::from(vec![100.0, 101.0, 102.0]);
    let v = Array1::from(vec![0.04, 0.05]);
    let _ = nmle_heston(s, v, 0.01);
  }

  #[test]
  #[should_panic(expected = "at least 2 observations")]
  fn nmle_heston_panics_on_short_series() {
    let s = Array1::from(vec![100.0]);
    let v = Array1::from(vec![0.04]);
    let _ = nmle_heston(s, v, 0.01);
  }

  #[test]
  #[should_panic(expected = "delta must be finite and positive")]
  fn nmle_heston_with_delta_panics_on_non_positive_step() {
    let s = Array1::from(vec![100.0, 101.0]);
    let v = Array1::from(vec![0.04, 0.05]);
    let _ = nmle_heston_with_delta(s, v, 0.01, 0.0);
  }

  #[test]
  #[should_panic(expected = "delta must be finite and positive")]
  fn pmle_heston_with_delta_panics_on_non_positive_step() {
    let s = Array1::from(vec![100.0, 101.0]);
    let v = Array1::from(vec![0.04, 0.05]);
    let _ = pmle_heston_with_delta(s, v, 0.01, 0.0);
  }

  #[test]
  fn nmle_heston_recovers_reasonable_params_from_synthetic_path() {
    let n = 4096usize;
    let m = n - 1;
    let dt = 1.0 / m as f64;
    let sqrt_dt = dt.sqrt();

    let r = 0.01;
    let kappa_true = 1.2;
    let theta_true = 0.09;
    let sigma_true = 0.20;
    let rho_true = -0.35;
    let beta1_true = (-kappa_true * dt).exp();
    let beta3_true = sigma_true * sigma_true * (1.0 - beta1_true * beta1_true) / (2.0 * kappa_true);

    let mut s = Array1::<f64>::zeros(n);
    let mut v = Array1::<f64>::zeros(n);
    s[0] = 100.0;
    v[0] = 0.09;

    let mut rng = StdRng::seed_from_u64(7);
    let normal = StandardNormal;

    for i in 1..n {
      let z1: f64 = normal.sample(&mut rng);
      let z2: f64 = normal.sample(&mut rng);

      let v_prev = v[i - 1].max(1e-10);
      let v_next =
        beta1_true * v_prev + theta_true * (1.0 - beta1_true) + (beta3_true * v_prev).sqrt() * z2;
      v[i] = v_next.max(1e-10);

      let dw2 =
        (v[i] - v_prev - kappa_true * (theta_true - v_prev) * dt) / (sigma_true * v_prev.sqrt());
      let dw1 = rho_true * dw2 + (1.0 - rho_true * rho_true).sqrt() * sqrt_dt * z1;

      let log_s_next = s[i - 1].ln() + (r - 0.5 * v_prev) * dt + v_prev.sqrt() * dw1;
      s[i] = log_s_next.exp();
    }

    let est = nmle_heston(s, v, r);

    assert!(est.kappa.is_finite() && est.kappa > 0.0);
    assert!(est.theta.is_finite() && est.theta > 0.0);
    assert!(est.sigma.is_finite() && est.sigma > 0.0);
    assert!(est.rho.is_finite());

    assert!(
      (est.kappa - kappa_true).abs() < 4.0,
      "kappa estimate too far: est={}, true={}",
      est.kappa,
      kappa_true
    );
    assert!(
      (est.theta - theta_true).abs() < 0.03,
      "theta estimate too far: est={}, true={}",
      est.theta,
      theta_true
    );
    assert!(
      (est.sigma - sigma_true).abs() < 0.10,
      "sigma estimate too far: est={}, true={}",
      est.sigma,
      sigma_true
    );
    assert!(
      (est.rho - rho_true).abs() < 0.12,
      "rho estimate too far: est={}, true={}",
      est.rho,
      rho_true
    );
  }

  #[test]
  fn pmle_heston_returns_finite_params_from_synthetic_path() {
    let n = 2048usize;
    let m = n - 1;
    let dt = 1.0 / m as f64;

    let r = 0.01;
    let kappa_true = 1.2;
    let theta_true = 0.08;
    let sigma_true = 0.2;
    let rho_true = -0.4;

    let mut s = Array1::<f64>::zeros(n);
    let mut v = Array1::<f64>::zeros(n);
    s[0] = 100.0;
    v[0] = theta_true;

    let mut rng = StdRng::seed_from_u64(17);
    let normal = StandardNormal;

    for i in 1..n {
      let z1: f64 = normal.sample(&mut rng);
      let z2: f64 = normal.sample(&mut rng);

      let dw1 = dt.sqrt() * z1;
      let dw2 = dt.sqrt() * (rho_true * z1 + (1.0 - rho_true * rho_true).sqrt() * z2);

      let v_prev = v[i - 1].max(1e-10);
      let v_next =
        v_prev + kappa_true * (theta_true - v_prev) * dt + sigma_true * v_prev.sqrt() * dw2;
      v[i] = v_next.max(1e-10);

      let log_s_next = s[i - 1].ln() + (r - 0.5 * v_prev) * dt + v_prev.sqrt() * dw1;
      s[i] = log_s_next.exp();
    }

    let est = pmle_heston(s, v, r);
    assert!(est.kappa.is_finite() && est.kappa > 0.0);
    assert!(est.theta.is_finite() && est.theta > 0.0);
    assert!(est.sigma.is_finite() && est.sigma > 0.0);
    assert!(est.rho.is_finite() && est.rho.abs() <= 1.0);
  }
}

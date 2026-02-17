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

pub fn nmle_heston(s: Array1<f64>, v: Array1<f64>, r: f64) -> HestonMleResult {
  let n = v.len();
  let delta = 1.0 / n as f64;
  let mut sum = [0.0; 4];

  for i in 1..n {
    // sum of (V_i / V_{i-1}
    sum[0] += v[i] / v[i - 1];

    // sum of V_i
    sum[1] += v[i];

    // sum of V_{i-1}
    sum[2] += v[i - 1];

    // sum of 1 / V_{i-1}
    sum[3] += 1.0 / v[i - 1];
  }

  let beta_hat1 = ((n as f64).powi(-2) * sum[1] * sum[3] - (n as f64).powi(-1) * sum[0])
    / ((n as f64).powi(-2) * sum[2] * sum[3] - 1.0);
  let beta_hat2 =
    ((n as f64).powi(-1) * sum[0] - beta_hat1) / ((1.0 - beta_hat1) * (n as f64).powi(-1) * sum[3]);

  let sum_beta_hat3 = {
    let mut sum = 0.0;
    for i in 1..n {
      sum +=
        (v[i] - v[i - 1] * beta_hat1 - beta_hat2 * (1.0 - beta_hat1).powi(2)) * (1.0 / v[i - 1])
    }
    sum
  };
  let beta_hat3 = (n as f64).powi(-1) * sum_beta_hat3;
  let kappa_hat = -(1.0 / delta) * beta_hat1.ln();
  let theta_hat = beta_hat2;
  let sigma_hat = (2.0 * kappa_hat * beta_hat3) / (1.0 - beta_hat1.powi(2)).sqrt();

  let mut sum_dw1dw2 = 0.0;
  for i in 1..n {
    let dw1_i = (s[i].ln() - s[i - 1].ln() - (r - 0.5 * v[i - 1]) * delta) / v[i - 1].sqrt();
    let dw2_i = (v[i] - v[i - 1] - kappa_hat * (theta_hat - v[i - 1]) * delta)
      / (sigma_hat * v[i - 1].sqrt());

    sum_dw1dw2 += dw1_i * dw2_i;
  }

  let rho_hat = sum_dw1dw2 / (n as f64 * delta);

  HestonMleResult {
    v0: v[0],
    kappa: kappa_hat,
    theta: theta_hat,
    sigma: sigma_hat,
    rho: rho_hat,
  }
}
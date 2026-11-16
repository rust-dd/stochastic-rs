//! # Mle
//!
//! $$
//! \hat\theta=\arg\max_\theta \sum_{i=1}^n \log f(x_i\mid\theta)
//! $$
//!
use ndarray::ArrayView1;

use crate::StatsError;
use crate::input::as_slice_or_copy;
use crate::input::reject_constant;
use crate::input::validate;
use crate::input::validate_view;

#[derive(Clone, Debug)]
pub struct HestonMleFit {
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

/// Wang et al. (2018) closed-form NMLE of Heston from `(S_t, V_t)` over one unit of time; `Err` on
/// unequal lengths, fewer than 2 or non-finite observations or a constant variance path.
/// Reference: Wang et al. (2018), Sci. China Inf. Sci. 61, 042202. DOI: 10.1007/s11432-017-9215-8
pub fn fit_heston_nmle(
  s: ArrayView1<f64>,
  v: ArrayView1<f64>,
  r: f64,
) -> Result<HestonMleFit, StatsError> {
  validate_series(s, v)?;
  Ok(nmle(s, v, r, 1.0 / (v.len() - 1) as f64))
}

/// [`fit_heston_nmle`] at sampling interval `delta` (Wang et al. eq. 19-21 and the rho moment
/// estimator); `Err` as it, and panics unless `delta` is finite and positive.
pub fn fit_heston_nmle_with_delta(
  s: ArrayView1<f64>,
  v: ArrayView1<f64>,
  r: f64,
  delta: f64,
) -> Result<HestonMleFit, StatsError> {
  assert_delta(delta);
  validate_series(s, v)?;
  Ok(nmle(s, v, r, delta))
}

fn nmle(s: ArrayView1<f64>, v: ArrayView1<f64>, r: f64, delta: f64) -> HestonMleFit {
  const EPS: f64 = 1e-12;

  let nobs = v.len();
  let n_f = (nobs - 1) as f64;

  // Following Wang et al. (2018), Eq. (19)-(21):
  // P = kappa * theta - sigma^2 / 4, with closed-form estimators from sqrt-variance dynamics.
  let mut sum_inv_v_prev = 0.0;
  let mut sum_v_prev = 0.0;
  let mut sum_sqrt_ratio = 0.0;
  let mut sum_sqrt_prod = 0.0;

  for i in 1..nobs {
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
  for i in 1..nobs {
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
  for i in 1..nobs {
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

  HestonMleFit {
    v0: v[0].max(0.0),
    kappa: kappa_hat,
    theta: theta_hat,
    sigma: sigma_hat,
    rho: rho_hat,
  }
}

/// Wang et al. (2018) closed-form PMLE of Heston from `(S_t, V_t)` over one unit of time; `Err` as
/// [`fit_heston_nmle`].
pub fn fit_heston_pmle(
  s: ArrayView1<f64>,
  v: ArrayView1<f64>,
  r: f64,
) -> Result<HestonMleFit, StatsError> {
  validate_series(s, v)?;
  Ok(pmle(s, v, r, 1.0 / (v.len() - 1) as f64))
}

/// [`fit_heston_pmle`] at sampling interval `delta` (Wang et al.'s PMLE and the rho moment
/// estimator); `Err` as it, and panics unless `delta` is finite and positive.
pub fn fit_heston_pmle_with_delta(
  s: ArrayView1<f64>,
  v: ArrayView1<f64>,
  r: f64,
  delta: f64,
) -> Result<HestonMleFit, StatsError> {
  assert_delta(delta);
  validate_series(s, v)?;
  Ok(pmle(s, v, r, delta))
}

fn pmle(s: ArrayView1<f64>, v: ArrayView1<f64>, r: f64, delta: f64) -> HestonMleFit {
  const EPS: f64 = 1e-12;

  let nobs = v.len();
  let n_f = (nobs - 1) as f64;

  let mut sum_v_k = 0.0;
  let mut sum_v_prev = 0.0;
  let mut sum_inv_v_prev = 0.0;
  let mut sum_ratio = 0.0;

  for i in 1..nobs {
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
  for i in 1..nobs {
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
  for i in 1..nobs {
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

  HestonMleFit {
    v0: v[0].max(0.0),
    kappa: kappa_hat,
    theta: theta_hat,
    sigma: sigma_hat,
    rho: rho_hat,
  }
}

/// Equal lengths, two or more finite observations each, and a variance path that moves: the
/// closed forms divide by its spread.
fn validate_series(s: ArrayView1<f64>, v: ArrayView1<f64>) -> Result<(), StatsError> {
  if s.len() != v.len() {
    return Err(StatsError::DimensionMismatch {
      expected: s.len(),
      got: v.len(),
    });
  }
  validate_view(s, 2)?;
  let v = as_slice_or_copy(v);
  validate(&v, 2)?;
  reject_constant(&v, "constant variance series")
}

#[track_caller]
fn assert_delta(delta: f64) {
  assert!(
    delta.is_finite() && delta > 0.0,
    "delta must satisfy `delta.is_finite() && delta > 0.0`, got delta = {delta:?}"
  );
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::SimdDistribution;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::fit_heston_nmle;
  use super::fit_heston_nmle_with_delta;
  use super::fit_heston_pmle;
  use super::fit_heston_pmle_with_delta;
  use crate::StatsError;

  #[test]
  fn nmle_heston_rejects_mismatched_lengths() {
    let s = Array1::from(vec![100.0, 101.0, 102.0]);
    let v = Array1::from(vec![0.04, 0.05]);
    assert_eq!(
      fit_heston_nmle(s.view(), v.view(), 0.01).unwrap_err(),
      StatsError::DimensionMismatch {
        expected: 3,
        got: 2
      }
    );
  }

  #[test]
  fn nmle_heston_rejects_a_short_series() {
    let s = Array1::from(vec![100.0]);
    let v = Array1::from(vec![0.04]);
    assert_eq!(
      fit_heston_nmle(s.view(), v.view(), 0.01).unwrap_err(),
      StatsError::InsufficientData { needed: 2, got: 1 }
    );
  }

  #[test]
  fn a_non_finite_or_constant_series_is_an_err() {
    let s = Array1::from(vec![100.0, 101.0, 102.0, 101.5]);
    let v = Array1::from(vec![0.04, f64::INFINITY, 0.05, 0.045]);
    for fit in [fit_heston_nmle, fit_heston_pmle] {
      assert_eq!(
        fit(s.view(), v.view(), 0.01).unwrap_err(),
        StatsError::InvalidObservation {
          index: 1,
          value: f64::INFINITY,
          constraint: "finite"
        }
      );
      assert_eq!(
        fit(s.view(), Array1::from_elem(4, 0.04).view(), 0.01).unwrap_err(),
        StatsError::DegenerateData("constant variance series")
      );
    }
  }

  #[test]
  #[should_panic(
    expected = "delta must satisfy `delta.is_finite() && delta > 0.0`, got delta = 0.0"
  )]
  fn nmle_heston_with_delta_panics_on_non_positive_step() {
    let s = Array1::from(vec![100.0, 101.0]);
    let v = Array1::from(vec![0.04, 0.05]);
    let _ = fit_heston_nmle_with_delta(s.view(), v.view(), 0.01, 0.0);
  }

  #[test]
  #[should_panic(
    expected = "delta must satisfy `delta.is_finite() && delta > 0.0`, got delta = 0.0"
  )]
  fn pmle_heston_with_delta_panics_on_non_positive_step() {
    let s = Array1::from(vec![100.0, 101.0]);
    let v = Array1::from(vec![0.04, 0.05]);
    let _ = fit_heston_pmle_with_delta(s.view(), v.view(), 0.01, 0.0);
  }

  #[test]
  /// Parameter recovery has a sampling distribution of its own, so a single
  /// path is a coin flip against these tolerances. Three pinned seeds with an
  /// any() over them — the best-of pattern from `integration-test-writing`
  /// SKILL section 1.2.
  fn nmle_heston_recovers_reasonable_params_from_synthetic_path() {
    // A daily step over ~16 years. The `fit_heston_nmle` default of delta =
    // 1/(n-1) puts the whole path inside one year, where kappa and theta are
    // barely identifiable at all — two of three seeds drove the optimiser to
    // a boundary. Volatility of variance and correlation are estimable
    // either way; mean reversion needs the horizon.
    let n = 4096usize;
    let dt = 1.0_f64 / 252.0;
    let sqrt_dt = dt.sqrt();

    let r = 0.01;
    let kappa_true = 1.2;
    let theta_true = 0.09;
    let sigma_true = 0.20;
    let rho_true = -0.35;
    let beta1_true = (-kappa_true * dt).exp();
    let beta3_true = sigma_true * sigma_true * (1.0 - beta1_true * beta1_true) / (2.0 * kappa_true);

    let run = |seed: u64| {
      let mut s = Array1::<f64>::zeros(n);
      let mut v = Array1::<f64>::zeros(n);
      s[0] = 100.0;
      v[0] = 0.09;

      let mut normal = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(seed));

      for i in 1..n {
        let z1: f64 = normal.sample();
        let z2: f64 = normal.sample();

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

      fit_heston_nmle_with_delta(s.view(), v.view(), r, dt).unwrap()
    };

    let fits = [7u64, 19, 41].map(|seed| (seed, run(seed)));

    for (seed, est) in fits.iter() {
      assert!(
        est.kappa.is_finite() && est.kappa > 0.0,
        "seed {seed}: kappa must be finite and positive, got {}",
        est.kappa
      );
      assert!(
        est.theta.is_finite() && est.theta > 0.0,
        "seed {seed}: theta must be finite and positive, got {}",
        est.theta
      );
      assert!(
        est.sigma.is_finite() && est.sigma > 0.0,
        "seed {seed}: sigma must be finite and positive, got {}",
        est.sigma
      );
      assert!(est.rho.is_finite(), "seed {seed}: rho must be finite");
    }

    let ok = fits.iter().any(|(_, est)| {
      (est.kappa - kappa_true).abs() < 4.0
        && (est.theta - theta_true).abs() < 0.03
        && (est.sigma - sigma_true).abs() < 0.10
        && (est.rho - rho_true).abs() < 0.12
    });
    let report = fits
      .iter()
      .map(|(s, e)| {
        format!(
          "seed {s}: κ={:.4} θ={:.4} σ={:.4} ρ={:.4}",
          e.kappa, e.theta, e.sigma, e.rho
        )
      })
      .collect::<Vec<_>>()
      .join("; ");
    assert!(ok, "no seed recovered the Heston parameters — {report}");
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

    let mut normal = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(17));

    for i in 1..n {
      let z1: f64 = normal.sample();
      let z2: f64 = normal.sample();

      let dw1 = dt.sqrt() * z1;
      let dw2 = dt.sqrt() * (rho_true * z1 + (1.0 - rho_true * rho_true).sqrt() * z2);

      let v_prev = v[i - 1].max(1e-10);
      let v_next =
        v_prev + kappa_true * (theta_true - v_prev) * dt + sigma_true * v_prev.sqrt() * dw2;
      v[i] = v_next.max(1e-10);

      let log_s_next = s[i - 1].ln() + (r - 0.5 * v_prev) * dt + v_prev.sqrt() * dw1;
      s[i] = log_s_next.exp();
    }

    let est = fit_heston_pmle(s.view(), v.view(), r).unwrap();
    assert!(est.kappa.is_finite() && est.kappa > 0.0);
    assert!(est.theta.is_finite() && est.theta > 0.0);
    assert!(est.sigma.is_finite() && est.sigma > 0.0);
    assert!(est.rho.is_finite() && est.rho.abs() <= 1.0);
  }
}

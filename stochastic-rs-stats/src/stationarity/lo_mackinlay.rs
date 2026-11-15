//! Lo-MacKinlay (1988) variance-ratio test for random-walk (unit-root)
//! behaviour of asset prices.
//!
//! Under the null $H_0$ that log-prices follow a random walk with drift,
//! the $q$-period return variance equals $q$ times the one-period variance,
//! so the variance ratio
//! $$
//! VR(q) = \frac{\widehat{\mathrm{Var}}(r_t(q))}{q \cdot \widehat{\mathrm{Var}}(r_t)}
//! $$
//! satisfies $VR(q) \to 1$ in probability. Two test statistics are computed
//! from the same overlapping-window variance estimator:
//!
//! - **M1(q)** assumes IID innovations (homoskedasticity):
//!   $Z_1(q) = \sqrt{nq}\,(VR(q)-1)/\sqrt{\phi(q)}$ with
//!   $\phi(q) = 2(2q-1)(q-1)/(3q)$.
//! - **M2(q)** is robust to conditional heteroskedasticity:
//!   $Z_2(q) = \sqrt{nq}\,(VR(q)-1)/\sqrt{\phi^*(q)}$ with
//!   $\phi^*(q) = \sum_{j=1}^{q-1} [2(q-j)/q]^2 \delta(j)$ and
//!   $\delta(j) = \frac{\sum_t (r_t-\bar r)^2 (r_{t-j}-\bar r)^2}{[\sum_t (r_t-\bar r)^2]^2}$.
//!
//! Both statistics are asymptotically $\mathcal{N}(0,1)$ under $H_0$. A
//! two-sided test rejects when $|Z_k| > z_{1-\alpha/2}$.
//!
//! Reference: Lo, A.W. and MacKinlay, A.C. (1988), "Stock Market Prices Do
//! Not Follow Random Walks: Evidence from a Simple Specification Test",
//! *Review of Financial Studies* 1(1), 41-66.

use ndarray::ArrayView1;

use crate::StatsError;
use crate::critical_values::CriticalValues;
use crate::critical_values::two_sided_p_value;
use crate::input::as_slice_or_copy;
use crate::input::assert_alpha;
use crate::input::reject_constant;
use crate::input::validate;

/// Configuration for the Lo-MacKinlay variance-ratio test.
#[derive(Debug, Clone, Copy)]
pub struct LoMacKinlayConfig {
  /// Aggregation horizon $q \ge 2$. Typical choices: 2, 4, 8, 16.
  pub q: usize,
  /// Significance level used for two-sided rejection.
  pub alpha: f64,
  /// Inputs are interpreted as log-prices (`true`) or as already-differenced
  /// log-returns (`false`).
  pub input_is_log_prices: bool,
}

impl Default for LoMacKinlayConfig {
  fn default() -> Self {
    Self {
      q: 2,
      alpha: 0.05,
      input_is_log_prices: true,
    }
  }
}

/// Lo–MacKinlay variance-ratio test (null: a random walk), decided by the two-sided normal p-value.
#[derive(Debug, Clone)]
pub struct LoMacKinlayTest {
  /// Lo-MacKinlay M2 statistic — heteroskedasticity-robust; NaN when $\phi^*(q)$ is zero.
  pub statistic: f64,
  /// Two-sided p-value $2\Phi(-|M_2|)$.
  pub p_value: Option<f64>,
  pub critical_values: Option<CriticalValues>,
  /// Whether the random-walk null is rejected at `alpha` by the robust statistic.
  pub rejects_null: bool,
  /// Aggregation horizon used.
  pub q: usize,
  /// Variance ratio $VR(q)$. Under the random-walk null, $VR(q) \to 1$.
  pub variance_ratio: f64,
  /// Lo-MacKinlay M1 statistic — IID innovations.
  pub z_iid: f64,
  /// Two-sided p-value $2\Phi(-|M_1|)$.
  pub p_value_iid: f64,
  /// Whether the random-walk null is rejected at `alpha` by the IID statistic.
  pub rejects_null_iid: bool,
}

impl crate::traits::HypothesisTest for LoMacKinlayTest {
  fn statistic(&self) -> f64 {
    self.statistic
  }

  fn null_rejected(&self) -> Option<bool> {
    Some(self.rejects_null)
  }
}

/// Lo–MacKinlay (1988) variance-ratio test; `Err` below `4 q` returns, on a non-finite input or on
/// zero-variance increments, and panics unless `q >= 2` and `0 < alpha < 1`.
pub fn lo_mackinlay_test(
  series: ArrayView1<f64>,
  cfg: LoMacKinlayConfig,
) -> Result<LoMacKinlayTest, StatsError> {
  assert!(cfg.q >= 2, "q must satisfy `q >= 2`, got q = {:?}", cfg.q);
  assert_alpha(cfg.alpha);
  let raw = as_slice_or_copy(series);
  let returns: Vec<f64> = if cfg.input_is_log_prices {
    validate(&raw, cfg.q * 4 + 1)?;
    (1..raw.len()).map(|i| raw[i] - raw[i - 1]).collect()
  } else {
    validate(&raw, cfg.q * 4)?;
    raw.to_vec()
  };
  reject_constant(&returns, "zero-variance increments")?;

  let nq = returns.len();
  let nq_f = nq as f64;
  let q = cfg.q;
  let q_f = q as f64;

  let mean = returns.iter().sum::<f64>() / nq_f;

  // One-period variance — sample variance with n-1 denominator (Lo-Mac eq.5a).
  let mut var1 = 0.0_f64;
  for &r in &returns {
    let d = r - mean;
    var1 += d * d;
  }
  var1 /= nq_f - 1.0;
  if var1 <= 0.0 {
    return Err(StatsError::DegenerateData("zero-variance increments"));
  }

  // q-period overlapping variance (Lo-Mac eq.7) with the small-sample
  // unbiased denominator m = q (nq - q + 1) (1 - q/nq). The `q` factor in m
  // already absorbs the per-period rescaling, so σ_c²(q) is on the same
  // one-period scale as σ_a²(1) — VR = σ_c²(q) / σ_a²(1) directly, with no
  // additional q-factor.
  let m = q_f * (nq_f - q_f + 1.0) * (1.0 - q_f / nq_f);
  let mut var_q = 0.0_f64;
  for t in q..=nq {
    let mut sum_r = 0.0_f64;
    for j in 0..q {
      sum_r += returns[t - 1 - j];
    }
    let d = sum_r - q_f * mean;
    var_q += d * d;
  }
  var_q /= m;

  let variance_ratio = var_q / var1;

  // IID statistic (Lo-Mac eq.14): φ(q) = 2(2q-1)(q-1)/(3q).
  let phi_iid = 2.0 * (2.0 * q_f - 1.0) * (q_f - 1.0) / (3.0 * q_f);
  let z_iid = nq_f.sqrt() * (variance_ratio - 1.0) / phi_iid.sqrt();

  // Heteroskedasticity-robust statistic (Lo-Mac eq.18).
  let denom_delta_squared: f64 = returns
    .iter()
    .map(|r| (r - mean).powi(2))
    .sum::<f64>()
    .powi(2);
  let mut phi_robust = 0.0_f64;
  if denom_delta_squared > 0.0 {
    for j in 1..q {
      let mut delta_j = 0.0_f64;
      for t in (j + 1)..=nq {
        delta_j += (returns[t - 1] - mean).powi(2) * (returns[t - 1 - j] - mean).powi(2);
      }
      delta_j /= denom_delta_squared;
      let weight = 2.0 * (q_f - j as f64) / q_f;
      phi_robust += weight * weight * delta_j;
    }
  }
  // φ* picks up an extra √(nq) scaling because δ(j) is normalised by the
  // squared sum of all returns rather than the sample variance (Lo-Mac eq.17
  // vs eq.18); multiply through by nq to recover the same scale as φ.
  phi_robust *= nq_f;
  let z_robust = if phi_robust > 0.0 {
    nq_f.sqrt() * (variance_ratio - 1.0) / phi_robust.sqrt()
  } else {
    f64::NAN
  };

  let p_value = two_sided_p_value(z_robust);
  let p_value_iid = two_sided_p_value(z_iid);

  Ok(LoMacKinlayTest {
    statistic: z_robust,
    p_value: Some(p_value),
    critical_values: None,
    rejects_null: p_value < cfg.alpha,
    q,
    variance_ratio,
    z_iid,
    p_value_iid,
    rejects_null_iid: p_value_iid < cfg.alpha,
  })
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::SimdDistribution;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::LoMacKinlayConfig;
  use super::lo_mackinlay_test;

  fn simulate_log_prices_random_walk(n: usize, sigma: f64, seed: u64) -> Vec<f64> {
    let mut dist = SimdNormal::<f64>::new(0.0, sigma).seeded(&Deterministic::new(seed));
    let mut x = vec![0.0_f64; n];
    for t in 1..n {
      x[t] = x[t - 1] + dist.sample();
    }
    x
  }

  fn simulate_log_prices_ar1_returns(phi: f64, n: usize, sigma: f64, seed: u64) -> Vec<f64> {
    let mut dist = SimdNormal::<f64>::new(0.0, sigma).seeded(&Deterministic::new(seed));
    let mut x = vec![0.0_f64; n];
    let mut r_prev = 0.0_f64;
    for t in 1..n {
      let r = phi * r_prev + dist.sample();
      x[t] = x[t - 1] + r;
      r_prev = r;
    }
    x
  }

  #[test]
  fn lo_mackinlay_does_not_reject_random_walk() {
    let x = simulate_log_prices_random_walk(4000, 0.01, 0x4C4F4D31);
    let res = lo_mackinlay_test(
      ndarray::ArrayView1::from(&x),
      LoMacKinlayConfig {
        q: 4,
        alpha: 0.05,
        input_is_log_prices: true,
      },
    )
    .unwrap();
    assert!(
      !res.rejects_null && !res.rejects_null_iid,
      "expected no rejection, got {res:?}"
    );
    assert!(
      (res.variance_ratio - 1.0).abs() < 0.1,
      "VR should be near 1, got {}",
      res.variance_ratio
    );
  }

  #[test]
  fn lo_mackinlay_rejects_positive_autocorrelation() {
    // Strong positive AR(1) on returns → variance ratio > 1.
    let x = simulate_log_prices_ar1_returns(0.4, 4000, 0.01, 0x4C4F4D32);
    let res = lo_mackinlay_test(
      ndarray::ArrayView1::from(&x),
      LoMacKinlayConfig {
        q: 4,
        alpha: 0.05,
        input_is_log_prices: true,
      },
    )
    .unwrap();
    assert!(res.rejects_null, "expected robust-Z rejection, got {res:?}");
    assert!(
      res.variance_ratio > 1.05,
      "VR should be > 1 under positive autocorrelation, got {}",
      res.variance_ratio
    );
  }

  #[test]
  fn lo_mackinlay_rejects_negative_autocorrelation() {
    // Strong negative AR(1) on returns → variance ratio < 1 (mean-reversion).
    let x = simulate_log_prices_ar1_returns(-0.4, 4000, 0.01, 0x4C4F4D33);
    let res = lo_mackinlay_test(
      ndarray::ArrayView1::from(&x),
      LoMacKinlayConfig {
        q: 4,
        alpha: 0.05,
        input_is_log_prices: true,
      },
    )
    .unwrap();
    assert!(res.rejects_null, "expected robust-Z rejection, got {res:?}");
    assert!(
      res.variance_ratio < 0.95,
      "VR should be < 1 under negative autocorrelation, got {}",
      res.variance_ratio
    );
  }

  #[test]
  fn constant_returns_are_degenerate_data() {
    let returns = ndarray::Array1::<f64>::from_elem(40, 0.1);
    let cfg = LoMacKinlayConfig {
      input_is_log_prices: false,
      ..LoMacKinlayConfig::default()
    };
    assert_eq!(
      lo_mackinlay_test(returns.view(), cfg).unwrap_err(),
      crate::StatsError::DegenerateData("zero-variance increments")
    );
  }

  #[test]
  fn lo_mackinlay_accepts_returns_input_directly() {
    let prices = simulate_log_prices_random_walk(2000, 0.01, 0x4C4F4D34);
    let returns: Vec<f64> = (1..prices.len())
      .map(|i| prices[i] - prices[i - 1])
      .collect();
    let res_returns = lo_mackinlay_test(
      ndarray::ArrayView1::from(&returns),
      LoMacKinlayConfig {
        q: 4,
        alpha: 0.05,
        input_is_log_prices: false,
      },
    )
    .unwrap();
    let res_prices = lo_mackinlay_test(
      ndarray::ArrayView1::from(&prices),
      LoMacKinlayConfig {
        q: 4,
        alpha: 0.05,
        input_is_log_prices: true,
      },
    )
    .unwrap();
    // Both paths should yield the same variance ratio modulo fp tolerance.
    assert!(
      (res_returns.variance_ratio - res_prices.variance_ratio).abs() < 1e-10,
      "{} vs {}",
      res_returns.variance_ratio,
      res_prices.variance_ratio
    );
  }
}

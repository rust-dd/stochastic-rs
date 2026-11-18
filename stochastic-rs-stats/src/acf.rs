//! Sample ACF with the biased `c_0` denominator, Durbin–Levinson PACF and the Ljung–Box test.
//!
//! $$
//! \hat\rho_k = \frac{\sum_{t=k}^{n-1} (x_t - \bar x)(x_{t-k} - \bar x)}
//! {\sum_{t=0}^{n-1} (x_t - \bar x)^2}, \qquad
//! Q = n(n+2) \sum_{k=1}^{h} \frac{\hat\rho_k^2}{n-k} \overset{H_0}{\sim} \chi^2_{h-\text{fit\_df}}
//! $$
//!
//! Reference: Box, Jenkins, Reinsel & Ljung (2015), Time Series Analysis, 5th ed., §2.1.4, §3.2.6
//! Reference: Ljung & Box (1978), Biometrika 65(2), 297-303, DOI: 10.1093/biomet/65.2.297
//! Reference: Durbin (1960), Revue de l'Institut International de Statistique 28(3), 233-244

use ndarray::Array1;
use ndarray::ArrayView1;
use stochastic_rs_distributions::special::gamma_q;

use crate::StatsError;
use crate::critical_values::CriticalValues;
use crate::input::assert_alpha;
use crate::input::validate_view;
use crate::traits::FloatExt;

/// Biased-denominator sample ACF (statsmodels' default), Box–Jenkins §2.1.4; `Err` on at most
/// `max_lag` points, a non-finite one or a constant series, and panics unless `max_lag >= 1`.
/// $$
/// \hat\rho_k = \frac{\sum_{t=k}^{n-1} (x_t - \bar x)(x_{t-k} - \bar x)}
/// {\sum_{t=0}^{n-1} (x_t - \bar x)^2}, \quad k = 1, \dots, \text{max\_lag}.
/// $$
pub fn acf<T: FloatExt>(x: ArrayView1<T>, max_lag: usize) -> Result<Array1<T>, StatsError> {
  assert!(
    max_lag >= 1,
    "max_lag must satisfy `max_lag >= 1`, got max_lag = {max_lag:?}"
  );
  validate_view(x, max_lag.saturating_add(1))?;
  if x.iter().all(|&v| v == x[0]) {
    return Err(StatsError::DegenerateData("constant series"));
  }
  let n = x.len();

  let mean = x.iter().copied().sum::<T>() / T::from_usize_(n);
  let centered = x.iter().map(|&v| v - mean).collect::<Vec<_>>();
  let c0 = centered.iter().map(|&d| d * d).sum::<T>();

  let mut rho = Array1::<T>::zeros(max_lag);
  for k in 1..=max_lag {
    let ck = (k..n).map(|t| centered[t] * centered[t - k]).sum::<T>();
    rho[k - 1] = ck / c0;
  }
  Ok(rho)
}

/// PACF by the Durbin–Levinson recursion over [`acf`] (Box–Jenkins §3.2.6), index `k − 1` holding
/// $\phi_{k,k}$; `Err` and panics as [`acf`].
/// $$
/// \phi_{k,k} = \frac{\hat\rho_k - \sum_{j=1}^{k-1} \phi_{k-1,j}\hat\rho_{k-j}}
/// {1 - \sum_{j=1}^{k-1} \phi_{k-1,j}\hat\rho_j}, \qquad
/// \phi_{k,j} = \phi_{k-1,j} - \phi_{k,k}\,\phi_{k-1,k-j} \quad (j = 1,\dots,k-1).
/// $$
pub fn pacf<T: FloatExt>(x: ArrayView1<T>, max_lag: usize) -> Result<Array1<T>, StatsError> {
  let rho = acf(x, max_lag)?;

  let mut phi = Array1::<T>::zeros(max_lag);
  phi[0] = rho[0];

  // `prev` holds phi_{k-1,1..=k-1}; `curr` is the scratch buffer for
  // phi_{k,1..=k} before it becomes `prev` for the next iteration.
  let mut prev = vec![T::zero(); max_lag];
  let mut curr = vec![T::zero(); max_lag];
  prev[0] = rho[0];

  for k in 2..=max_lag {
    let mut num = rho[k - 1];
    let mut den = T::one();
    for j in 1..k {
      num -= prev[j - 1] * rho[k - j - 1];
      den -= prev[j - 1] * rho[j - 1];
    }
    let phi_kk = num / den;

    for j in 1..k {
      curr[j - 1] = prev[j - 1] - phi_kk * prev[k - j - 1];
    }
    curr[k - 1] = phi_kk;

    phi[k - 1] = phi_kk;
    prev[..k].copy_from_slice(&curr[..k]);
  }

  Ok(phi)
}

/// Configuration for [`ljung_box`].
#[derive(Debug, Clone, Copy)]
pub struct LjungBoxConfig {
  /// Autocorrelation lags in the statistic.
  pub lags: usize,
  /// Parameters fitted to the series before the test: `p + q` for ARMA(p, q) residuals.
  pub fit_df: usize,
  /// Significance level used to compute `rejects_null`.
  pub alpha: f64,
}

impl Default for LjungBoxConfig {
  fn default() -> Self {
    Self {
      lags: 20,
      fit_df: 0,
      alpha: 0.05,
    }
  }
}

/// Ljung–Box portmanteau test (null: no autocorrelation up to `lags`), decided by its p-value.
#[derive(Debug, Clone)]
pub struct LjungBoxTest {
  /// Q test statistic.
  pub statistic: f64,
  /// p-value under the chi-square(`df`) asymptotic null (upper-tail survival function).
  pub p_value: Option<f64>,
  pub critical_values: Option<CriticalValues>,
  /// Whether the no-autocorrelation null is rejected at `alpha`.
  pub rejects_null: bool,
  /// Degrees of freedom, `lags - fit_df`.
  pub df: usize,
  /// Number of autocorrelation lags included in the statistic.
  pub lags: usize,
}

impl crate::traits::HypothesisTest for LjungBoxTest {
  fn statistic(&self) -> f64 {
    self.statistic
  }

  fn null_rejected(&self) -> Option<bool> {
    Some(self.rejects_null)
  }
}

/// Ljung–Box test of the first `lags` autocorrelations against `χ²(lags − fit_df)`; `Err` as
/// [`acf`]'s on the series, and panics unless `lags > fit_df` and `0 < alpha < 1`.
pub fn ljung_box<T: FloatExt>(
  x: ArrayView1<T>,
  cfg: LjungBoxConfig,
) -> Result<LjungBoxTest, StatsError> {
  let LjungBoxConfig {
    lags,
    fit_df,
    alpha,
  } = cfg;
  assert!(
    lags > fit_df,
    "lags must satisfy `lags > fit_df`, got lags = {lags:?}, fit_df = {fit_df:?}"
  );
  assert_alpha(alpha);

  let n = x.len();
  let rho = acf(x, lags)?;

  let n_f = n as f64;
  let mut sum = 0.0_f64;
  for k in 1..=lags {
    let rho_k = rho[k - 1].to_f64().unwrap();
    sum += rho_k * rho_k / (n_f - k as f64);
  }
  let statistic = n_f * (n_f + 2.0) * sum;

  let df = lags - fit_df;
  let p_value = gamma_q(df as f64 / 2.0, statistic / 2.0);

  Ok(LjungBoxTest {
    statistic,
    p_value: Some(p_value),
    critical_values: None,
    rejects_null: p_value < alpha,
    df,
    lags,
  })
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use ndarray::ArrayView1;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::DistributionSampler;
  use stochastic_rs_distributions::SimdDistribution;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::LjungBoxConfig;
  use super::StatsError;
  use super::acf;
  use super::ljung_box;
  use super::pacf;

  fn lags_10() -> LjungBoxConfig {
    LjungBoxConfig {
      lags: 10,
      ..LjungBoxConfig::default()
    }
  }

  /// An AR(1) path with standard normal innovations.
  fn simulate_ar1(phi: f64, n: usize, seed: u64) -> Vec<f64> {
    let innovations = {
      let mut dist = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(seed));
      (0..n).map(|_| dist.sample()).collect::<Vec<_>>()
    };
    let mut x = vec![0.0; n];
    for t in 1..n {
      x[t] = phi * x[t - 1] + innovations[t];
    }
    x
  }

  /// `x = 1..5` has deviations `[-2, -1, 0, 1, 2]`: `c0 = 10`, `c1 = 4`, `c2 = -1`, `c3 = -4`, so
  /// `ρ = [1, 0.4, -0.1, -0.4]`.
  #[test]
  fn acf_matches_hand_computation() {
    let x = Array1::<f64>::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let rho = acf(x.view(), 3).unwrap();
    assert!((rho[0] - 0.4).abs() < 1e-12, "rho1 = {}", rho[0]);
    assert!((rho[1] - (-0.1)).abs() < 1e-12, "rho2 = {}", rho[1]);
    assert!((rho[2] - (-0.4)).abs() < 1e-12, "rho3 = {}", rho[2]);
  }

  #[test]
  fn acf_rejects_constant_series() {
    let x = Array1::<f64>::from(vec![5.0; 20]);
    assert_eq!(
      acf(x.view(), 3).unwrap_err(),
      StatsError::DegenerateData("constant series")
    );
  }

  #[test]
  fn acf_rejects_non_finite_series() {
    let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    v[2] = f64::NAN;
    let x = Array1::<f64>::from(v);
    assert!(matches!(
      acf(x.view(), 3).unwrap_err(),
      StatsError::InvalidObservation { index: 2, .. }
    ));
  }

  #[test]
  fn acf_needs_more_points_than_lags() {
    let x = Array1::<f64>::from(vec![1.0, 2.0, 3.0]);
    assert_eq!(
      acf(x.view(), 3).unwrap_err(),
      StatsError::InsufficientData { needed: 4, got: 3 }
    );
  }

  #[test]
  fn a_lag_bound_of_usize_max_is_insufficient_data() {
    let x = Array1::<f64>::linspace(0.0, 1.0, 20);
    let expected = StatsError::InsufficientData {
      needed: usize::MAX,
      got: 20,
    };
    assert_eq!(acf(x.view(), usize::MAX).unwrap_err(), expected);
    assert_eq!(pacf(x.view(), usize::MAX).unwrap_err(), expected);
    let cfg = LjungBoxConfig {
      lags: usize::MAX,
      ..LjungBoxConfig::default()
    };
    assert_eq!(ljung_box(x.view(), cfg).unwrap_err(), expected);
  }

  /// Durbin-Levinson base case: phi_{1,1} = rho_1 exactly.
  #[test]
  fn pacf_lag1_equals_acf_lag1() {
    let x = Array1::<f64>::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let rho = acf(x.view(), 3).unwrap();
    let phi = pacf(x.view(), 3).unwrap();
    assert!(
      (phi[0] - rho[0]).abs() < 1e-12,
      "phi[0]={}, rho[0]={}",
      phi[0],
      rho[0]
    );
  }

  #[test]
  fn pacf_ar1_cuts_off_after_lag1() {
    let best_badness = [2718u64, 999, 42]
      .into_iter()
      .map(|seed| {
        let x = simulate_ar1(0.7, 20_000, seed);
        let phi = pacf(ArrayView1::from(&x), 5).unwrap();
        let lag1_err = (phi[0] - 0.7).abs();
        let tail_max = phi
          .slice(ndarray::s![1..5])
          .iter()
          .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        lag1_err.max(tail_max)
      })
      .fold(f64::INFINITY, f64::min);

    assert!(
      best_badness < 0.05,
      "no seed satisfied |pacf[0]-0.7|<0.05 and max|pacf[1..5]|<0.05 (best badness {best_badness})"
    );
  }

  #[test]
  fn ljung_box_iid_high_p() {
    let best_p = [2718u64, 999, 42]
      .into_iter()
      .map(|seed| {
        let mut dist = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(seed));
        let mut x = vec![0.0; 2000];
        dist.fill_slice(&mut x);
        ljung_box(ArrayView1::from(&x), lags_10())
          .unwrap()
          .p_value
          .unwrap()
      })
      .fold(0.0_f64, f64::max);

    assert!(
      best_p > 0.01,
      "every seed rejected iid noise (best p {best_p})"
    );
  }

  #[test]
  fn ljung_box_ar1_rejects() {
    for seed in [2718u64, 999, 42] {
      let x = simulate_ar1(0.5, 2000, seed);
      let res = ljung_box(ArrayView1::from(&x), lags_10()).unwrap();
      assert!(
        res.p_value.unwrap() < 1e-6,
        "seed {seed}: expected p < 1e-6 for AR(1) residual autocorrelation, got {:?}",
        res.p_value
      );
    }
  }

  #[test]
  #[should_panic(expected = "lags must satisfy `lags > fit_df`, got lags = 3, fit_df = 3")]
  fn ljung_box_rejects_bad_df() {
    let x = Array1::<f64>::from(vec![1.0; 20]);
    let _ = ljung_box(
      x.view(),
      LjungBoxConfig {
        lags: 3,
        fit_df: 3,
        alpha: 0.05,
      },
    );
  }
}

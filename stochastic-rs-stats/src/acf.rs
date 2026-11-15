//! # Autocorrelation, partial autocorrelation, and the Ljung-Box test
//!
//! Sample ACF with the biased $c_0$ denominator (the statsmodels default),
//! the Durbin-Levinson recursion for the PACF, and the Ljung-Box
//! portmanteau statistic for testing whether a batch of sample
//! autocorrelations is jointly zero.
//!
//! $$
//! \hat\rho_k = \frac{\sum_{t=k}^{n-1} (x_t - \bar x)(x_{t-k} - \bar x)}
//! {\sum_{t=0}^{n-1} (x_t - \bar x)^2}, \qquad
//! Q = n(n+2) \sum_{k=1}^{h} \frac{\hat\rho_k^2}{n-k} \overset{H_0}{\sim} \chi^2_{h-\text{fit\_df}}
//! $$
//!
//! References:
//! - Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
//!   *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
//!   §2.1.4 (sample ACF), §3.2.6 (Durbin-Levinson PACF recursion).
//! - Ljung, G. M., & Box, G. E. P. (1978). "On a Measure of Lack of Fit in
//!   Time Series Models". *Biometrika*, 65(2), 297-303.
//!   DOI: 10.1093/biomet/65.2.297
//! - Durbin, J. (1960). "The Fitting of Time-Series Models". *Revue de
//!   l'Institut International de Statistique*, 28(3), 233-244.

use ndarray::Array1;
use ndarray::ArrayView1;
use stochastic_rs_distributions::special::gamma_q;

use crate::critical_values::CriticalValues;
use crate::traits::FloatExt;

/// Sample autocorrelation function (ACF) up to `max_lag`.
///
/// Biased estimator with denominator $c_0$ (statsmodels' `acf(..., fft=False,
/// adjusted=False)` default), Box-Jenkins §2.1.4:
/// $$
/// \hat\rho_k = \frac{\sum_{t=k}^{n-1} (x_t - \bar x)(x_{t-k} - \bar x)}
/// {\sum_{t=0}^{n-1} (x_t - \bar x)^2}, \quad k = 1, \dots, \text{max\_lag}.
/// $$
///
/// The returned array has length `max_lag`; index `0` is $\hat\rho_1$ (the
/// trivial $\hat\rho_0 = 1$ is not included).
///
/// # Panics
/// Panics if `x` has fewer than 2 observations, `max_lag == 0`,
/// `max_lag >= x.len()`, `x` contains a non-finite value, or `x` is
/// constant (`c_0 = 0`, which would make every `ρ̂_k` an undefined `0/0`).
pub fn acf<T: FloatExt>(x: ArrayView1<T>, max_lag: usize) -> Array1<T> {
  let n = x.len();
  assert!(n >= 2, "acf requires at least 2 observations");
  assert!(max_lag > 0, "max_lag must be positive");
  assert!(max_lag < n, "max_lag must be less than the sample size");
  assert!(
    x.iter().all(|v| v.is_finite()),
    "acf requires finite observations"
  );

  let mean = x.iter().copied().sum::<T>() / T::from_usize_(n);
  let centered = x.iter().map(|&v| v - mean).collect::<Vec<_>>();
  let c0 = centered.iter().map(|&d| d * d).sum::<T>();
  assert!(c0 > T::zero(), "acf requires a non-constant series");

  let mut rho = Array1::<T>::zeros(max_lag);
  for k in 1..=max_lag {
    let ck = (k..n).map(|t| centered[t] * centered[t - k]).sum::<T>();
    rho[k - 1] = ck / c0;
  }
  rho
}

/// Partial autocorrelation function (PACF) up to `max_lag`, via the
/// Durbin-Levinson recursion over the sample [`acf`] (Box-Jenkins §3.2.6):
/// $$
/// \phi_{k,k} = \frac{\hat\rho_k - \sum_{j=1}^{k-1} \phi_{k-1,j}\hat\rho_{k-j}}
/// {1 - \sum_{j=1}^{k-1} \phi_{k-1,j}\hat\rho_j}, \qquad
/// \phi_{k,j} = \phi_{k-1,j} - \phi_{k,k}\,\phi_{k-1,k-j} \quad (j = 1,\dots,k-1).
/// $$
///
/// The returned array has length `max_lag`; index `0` is
/// $\phi_{1,1} = \hat\rho_1$, index `k-1` is $\phi_{k,k}$.
///
/// # Panics
/// Same as [`acf`] (fewer than 2 observations, `max_lag == 0`,
/// `max_lag >= x.len()`, a non-finite value, or a constant series) — this
/// function calls it internally to obtain
/// $\hat\rho_1,\dots,\hat\rho_{\text{max\_lag}}$.
pub fn pacf<T: FloatExt>(x: ArrayView1<T>, max_lag: usize) -> Array1<T> {
  let rho = acf(x, max_lag);

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

  phi
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

/// Ljung–Box test of the first `lags` autocorrelations against `χ²(lags − fit_df)`; panics unless
/// `lags > fit_df` and `0 < alpha < 1`, and as [`acf`] does on the series.
pub fn ljung_box<T: FloatExt>(x: ArrayView1<T>, cfg: LjungBoxConfig) -> LjungBoxTest {
  let LjungBoxConfig {
    lags,
    fit_df,
    alpha,
  } = cfg;
  assert!(
    lags > fit_df,
    "lags must exceed fit_df (lags = {lags}, fit_df = {fit_df})"
  );
  assert!(
    alpha > 0.0 && alpha < 1.0,
    "alpha must satisfy `0.0 < alpha < 1.0`, got alpha = {alpha:?}"
  );

  let n = x.len();
  let rho = acf(x, lags);

  let n_f = n as f64;
  let mut sum = 0.0_f64;
  for k in 1..=lags {
    let rho_k = rho[k - 1].to_f64().unwrap();
    sum += rho_k * rho_k / (n_f - k as f64);
  }
  let statistic = n_f * (n_f + 2.0) * sum;

  let df = lags - fit_df;
  let p_value = gamma_q(df as f64 / 2.0, statistic / 2.0);

  LjungBoxTest {
    statistic,
    p_value: Some(p_value),
    critical_values: None,
    rejects_null: p_value < alpha,
    df,
    lags,
  }
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
  use super::acf;
  use super::ljung_box;
  use super::pacf;

  fn lags_10() -> LjungBoxConfig {
    LjungBoxConfig {
      lags: 10,
      ..LjungBoxConfig::default()
    }
  }

  /// Self-contained AR(1) generator, mirroring
  /// `stationarity::adf::tests::simulate_ar1` (not imported: that module is
  /// faer-backed).
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

  /// Hand-computed: x = [1,2,3,4,5], mean x̄ = 3, deviations d = [-2,-1,0,1,2].
  /// c0 = (-2)^2+(-1)^2+0^2+1^2+2^2 = 4+1+0+1+4 = 10.
  /// c1 = d1*d0 + d2*d1 + d3*d2 + d4*d3
  ///    = (-1)(-2) + (0)(-1) + (1)(0) + (2)(1) = 2+0+0+2 = 4  ⟹ ρ1 = 4/10 = 0.4.
  /// c2 = d2*d0 + d3*d1 + d4*d2 = (0)(-2) + (1)(-1) + (2)(0) = 0-1+0 = -1 ⟹ ρ2 = -1/10 = -0.1.
  /// c3 = d3*d0 + d4*d1 = (1)(-2) + (2)(-1) = -2-2 = -4 ⟹ ρ3 = -4/10 = -0.4.
  #[test]
  fn acf_matches_hand_computation() {
    let x = Array1::<f64>::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let rho = acf(x.view(), 3);
    assert!((rho[0] - 0.4).abs() < 1e-12, "rho1 = {}", rho[0]);
    assert!((rho[1] - (-0.1)).abs() < 1e-12, "rho2 = {}", rho[1]);
    assert!((rho[2] - (-0.4)).abs() < 1e-12, "rho3 = {}", rho[2]);
  }

  #[test]
  #[should_panic(expected = "acf requires a non-constant series")]
  fn acf_rejects_constant_series() {
    let x = Array1::<f64>::from(vec![5.0; 20]);
    let _ = acf(x.view(), 3);
  }

  #[test]
  #[should_panic(expected = "acf requires finite observations")]
  fn acf_rejects_non_finite_series() {
    let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    v[2] = f64::NAN;
    let x = Array1::<f64>::from(v);
    let _ = acf(x.view(), 3);
  }

  /// Durbin-Levinson base case: phi_{1,1} = rho_1 exactly.
  #[test]
  fn pacf_lag1_equals_acf_lag1() {
    let x = Array1::<f64>::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let rho = acf(x.view(), 3);
    let phi = pacf(x.view(), 3);
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
        let phi = pacf(ArrayView1::from(&x), 5);
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
        ljung_box(ArrayView1::from(&x), lags_10()).p_value.unwrap()
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
      let res = ljung_box(ArrayView1::from(&x), lags_10());
      assert!(
        res.p_value.unwrap() < 1e-6,
        "seed {seed}: expected p < 1e-6 for AR(1) residual autocorrelation, got {:?}",
        res.p_value
      );
    }
  }

  #[test]
  #[should_panic(expected = "lags must exceed fit_df")]
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

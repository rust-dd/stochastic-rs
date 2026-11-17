//! Granger causality test based on a VAR(p) F-test of nested OLS.
//!
//! Tests $H_0$: lags of $X$ do not Granger-cause $Y$ versus $H_1$ that they
//! do, comparing restricted ($Y$ on its own lags) and unrestricted ($Y$ on
//! its own lags + lags of $X$) regression sums of squares.
//!
//! Reference: Granger, "Investigating Causal Relations by Econometric Models
//! and Cross-Spectral Methods", Econometrica, 37(3), 424-438 (1969).
//! DOI: 10.2307/1912791

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use stochastic_rs_distributions::linalg::lstsq;
use stochastic_rs_distributions::special::beta_i as beta_reg;

use crate::StatsError;
use crate::critical_values::CriticalValues;
use crate::input::assert_alpha;
use crate::stationarity::common::prepared;
use crate::stationarity::common::reject_exact_fit;

/// Granger-causality F-test (null: `x` does not Granger-cause `y`), decided by its p-value.
#[derive(Debug, Clone)]
pub struct GrangerTest {
  /// $F$ statistic.
  pub statistic: f64,
  /// $p$-value under $\mathcal F_{p, T-2p-1}$.
  pub p_value: Option<f64>,
  pub critical_values: Option<CriticalValues>,
  /// Whether $H_0$ (no Granger causality) is rejected at the supplied significance level.
  pub rejects_null: bool,
  /// Number of lag terms tested.
  pub lags: usize,
  /// Effective sample size (rows of the VAR design).
  pub nobs: usize,
}

impl crate::traits::HypothesisTest for GrangerTest {
  fn statistic(&self) -> f64 {
    self.statistic
  }

  fn null_rejected(&self) -> Option<bool> {
    Some(self.rejects_null)
  }
}

/// Granger F-test of "does `x` cause `y`?" at `lags` lags; `Err` below `3 lags + 3` points, a
/// non-finite or constant series, unequal lengths, a singular or exact fit; panics on bad config.
pub fn granger_causality_test(
  y: ArrayView1<f64>,
  x: ArrayView1<f64>,
  lags: usize,
  alpha: f64,
) -> Result<GrangerTest, StatsError> {
  assert!(
    lags >= 1,
    "lags must satisfy `lags >= 1`, got lags = {lags:?}"
  );
  assert_alpha(alpha);
  if x.len() != y.len() {
    return Err(StatsError::DimensionMismatch {
      expected: y.len(),
      got: x.len(),
    });
  }
  let needed = lags.saturating_mul(3).saturating_add(3);
  // Each series at its own power of two, which `F` does not see, keeps every square in range.
  let (y, x) = (prepared(y, needed)?, prepared(x, needed)?);
  let n_total = y.len();
  let n = n_total - lags;

  let mut design_unrest = Array2::<f64>::zeros((n, 1 + 2 * lags));
  let mut target = Array1::<f64>::zeros(n);
  for i in 0..n {
    target[i] = y[lags + i];
    design_unrest[[i, 0]] = 1.0;
    for l in 1..=lags {
      design_unrest[[i, l]] = y[lags + i - l];
      design_unrest[[i, lags + l]] = x[lags + i - l];
    }
  }
  let mut design_rest = Array2::<f64>::zeros((n, 1 + lags));
  for i in 0..n {
    design_rest[[i, 0]] = 1.0;
    for l in 1..=lags {
      design_rest[[i, l]] = y[lags + i - l];
    }
  }
  let (residuals, rss_unrest) = ols_rss(&design_unrest, &target)?;
  reject_exact_fit(&y, residuals.as_slice().unwrap(), 1.0)?;
  let (_, rss_rest) = ols_rss(&design_rest, &target)?;
  let q = lags as f64;
  let dof = (n as f64) - (1.0 + 2.0 * q);
  let f = ((rss_rest - rss_unrest) / q) / (rss_unrest / dof.max(1.0));
  if !f.is_finite() {
    return Err(StatsError::Numerical {
      stage: "F statistic",
    });
  }
  // The restricted fit is nested in the full one, so a negative `F` is the subtraction's rounding.
  let f = f.max(0.0);
  let p_value = if f > 0.0 {
    let d1 = q;
    let d2 = dof.max(1.0);
    let x = (d1 * f) / (d1 * f + d2);
    1.0 - beta_reg(d1 / 2.0, d2 / 2.0, x)
  } else {
    1.0
  };
  Ok(GrangerTest {
    statistic: f,
    p_value: Some(p_value),
    critical_values: None,
    rejects_null: p_value < alpha,
    lags,
    nobs: n,
  })
}

fn ols_rss(x: &Array2<f64>, y: &Array1<f64>) -> Result<(Array1<f64>, f64), StatsError> {
  let beta = lstsq(x.view(), y.view());
  let yhat = x.dot(&beta);
  let residuals = y - &yhat;
  let rss = residuals.iter().map(|v| v * v).sum::<f64>();
  if rss.is_finite() {
    Ok((residuals, rss))
  } else {
    Err(StatsError::Singular("OLS design"))
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::DistributionSampler;
  use stochastic_rs_distributions::SimdDistribution;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::*;

  #[test]
  fn granger_does_not_reject_independent_series() {
    let mut dist = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(7));
    let mut x_buf = vec![0.0_f64; 500];
    dist.fill_slice(&mut x_buf);
    let mut dist2 = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(13));
    let mut y_buf = vec![0.0_f64; 500];
    dist2.fill_slice(&mut y_buf);
    let x = Array1::from(x_buf);
    let y = Array1::from(y_buf);
    let res = granger_causality_test(y.view(), x.view(), 4, 0.05).unwrap();
    assert!(!res.rejects_null);
  }

  #[test]
  fn granger_rejects_when_x_drives_y() {
    let mut dist = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(17));
    let mut x_buf = vec![0.0_f64; 500];
    dist.fill_slice(&mut x_buf);
    let mut dist_eps = SimdNormal::<f64>::new(0.0, 0.3).seeded(&Deterministic::new(19));
    let mut eps = vec![0.0_f64; 500];
    dist_eps.fill_slice(&mut eps);
    let mut y = vec![0.0_f64; 500];
    for i in 2..500 {
      y[i] = 0.5 * y[i - 1] + 0.7 * x_buf[i - 1] - 0.3 * x_buf[i - 2] + eps[i];
    }
    let xa = Array1::from(x_buf);
    let ya = Array1::from(y);
    let res = granger_causality_test(ya.view(), xa.view(), 2, 0.05).unwrap();
    assert!(res.rejects_null);
  }

  #[test]
  fn series_of_unequal_lengths_are_a_dimension_mismatch() {
    let y = Array1::from_iter((0..40).map(|i| (i as f64 * 0.7).sin()));
    let x = Array1::from_iter((0..41).map(|i| (i as f64 * 0.3).cos()));
    assert_eq!(
      granger_causality_test(y.view(), x.view(), 1, 0.05).unwrap_err(),
      StatsError::DimensionMismatch {
        expected: 40,
        got: 41
      }
    );
  }

  #[test]
  fn a_constant_series_is_degenerate_data() {
    let constant = Array1::<f64>::from_elem(64, 0.3);
    let moving = Array1::from_iter((0..64).map(|i| (i as f64 * 0.7).sin()));
    for (y, x) in [(&constant, &moving), (&moving, &constant)] {
      assert_eq!(
        granger_causality_test(y.view(), x.view(), 1, 0.05).unwrap_err(),
        StatsError::DegenerateData("constant series")
      );
    }
  }

  #[test]
  fn a_lag_bound_of_usize_max_is_insufficient_data() {
    let y = Array1::from_iter((0..40).map(|i| (i as f64 * 0.7).sin()));
    let x = Array1::from_iter((0..40).map(|i| (i as f64 * 0.3).cos()));
    assert_eq!(
      granger_causality_test(y.view(), x.view(), usize::MAX, 0.05).unwrap_err(),
      StatsError::InsufficientData {
        needed: usize::MAX,
        got: 40
      }
    );
  }

  /// A zero target after the first lag fits exactly through zero arithmetic, on any platform.
  #[test]
  fn an_exact_fit_is_degenerate_data() {
    let mut y = Array1::<f64>::zeros(12);
    y[0] = 5.0;
    let x = Array1::from_iter((0..12).map(|i| i as f64));
    assert_eq!(
      granger_causality_test(y.view(), x.view(), 1, 0.05).unwrap_err(),
      StatsError::DegenerateData("zero residual variance")
    );
  }

  const SCALES: [f64; 9] = [1e-300, 1e-200, 1e-160, 1e-6, 1.0, 1e6, 1e160, 1e200, 1e300];

  fn normals(seed: u64) -> Array1<f64> {
    let mut z = vec![0.0_f64; 200];
    SimdNormal::<f64>::new(0.0, 1.0)
      .seeded(&Deterministic::new(seed))
      .fill_slice(&mut z);
    Array1::from(z)
  }

  /// `y` driven by the last `x` with noise far below `1e-12` of it fits to rounding at every scale.
  #[test]
  fn a_fit_to_rounding_is_degenerate_data_at_every_scale() {
    let (x, noise) = (normals(23), normals(29));
    let y = Array1::from_iter((0..200).map(|t| {
      let driven = if t == 0 { 0.4 } else { 0.7 * x[t - 1] };
      driven + 1e-15 * noise[t]
    }));
    for c in SCALES {
      assert_eq!(
        granger_causality_test(y.mapv(|v| v * c).view(), x.mapv(|v| v * c).view(), 2, 0.05)
          .unwrap_err(),
        StatsError::DegenerateData("zero residual variance"),
        "c = {c:e}"
      );
    }
  }

  /// `F` does not see the scale of either series, scaled together or apart.
  #[test]
  fn an_ordinary_pair_keeps_its_statistic_at_every_scale() {
    let (x, w) = (normals(31), normals(37));
    let driven = Array1::from_iter((0..200).map(|t| {
      let drive = if t == 0 { 0.0 } else { 0.3 * x[t - 1] };
      drive + w[t]
    }));
    for y in [&w, &driven] {
      let unit = granger_causality_test(y.view(), x.view(), 2, 0.05)
        .unwrap()
        .statistic;
      for c in SCALES {
        for (cy, cx) in [(c, c), (c, 1.0), (1.0, c)] {
          let got = granger_causality_test(
            y.mapv(|v| v * cy).view(),
            x.mapv(|v| v * cx).view(),
            2,
            0.05,
          )
          .map(|r| r.statistic);
          assert!(
            got.as_ref().is_ok_and(|f| (f - unit).abs() <= 1e-12 * unit),
            "y × {cy:e}, x × {cx:e}: {got:?} against {unit}"
          );
        }
      }
    }
  }
}

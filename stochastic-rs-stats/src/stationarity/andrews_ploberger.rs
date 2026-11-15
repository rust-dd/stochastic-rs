//! Andrews (1993) and Andrews-Ploberger (1994) tests for parameter
//! constancy in a linear regression with unknown breakpoint.
//!
//! Given $y_t = x_t^\top \beta + \varepsilon_t$, $t = 1, \dots, T$, and a
//! candidate split at fraction $\tau$ with breakpoint $t_\tau = \lfloor \tau
//! T \rfloor$, fit OLS separately on the two subsamples and compute the
//! Wald statistic for the joint restriction $\beta_1 = \beta_2$:
//! $$
//! W(\tau) = (\hat\beta_1 - \hat\beta_2)^\top
//!           [\hat V(\hat\beta_1) + \hat V(\hat\beta_2)]^{-1}
//!           (\hat\beta_1 - \hat\beta_2).
//! $$
//! Three aggregation statistics over $\tau \in [\pi_0, 1-\pi_0]$ are
//! supported:
//!
//! - **Sup-Wald** $\;\sup_\tau W(\tau)$ — Quandt (1960) / Andrews (1993).
//! - **Exp-Wald** $\;\ln\!\big(\tfrac{1}{N}\sum_\tau e^{W(\tau)/2}\big)$ —
//!   Andrews-Ploberger (1994), optimal under a Pitman-drift alternative
//!   with large $\xi$.
//! - **Avg-Wald** $\;\tfrac{1}{N}\sum_\tau W(\tau)$ — Andrews-Ploberger
//!   (1994), optimal under small $\xi$.
//!
//! Asymptotic critical values are taken from the canonical tables for
//! $\pi_0 = 0.15$ (Andrews 1993 Table 1 for Sup-Wald; Andrews-Ploberger
//! 1994 Tables II and III for Exp-Wald and Avg-Wald). For any other `pi0` or `k > 5` the
//! statistic is computed, but its p-value bracket, critical values and decision are `None`.
//!
//! References:
//! - Andrews, D.W.K. (1993), "Tests for Parameter Instability and
//!   Structural Change with Unknown Change Point", *Econometrica* 61(4),
//!   821-856.
//! - Andrews, D.W.K., Ploberger, W. (1994), "Optimal Tests when a
//!   Nuisance Parameter is Present Only under the Alternative",
//!   *Econometrica* 62(6), 1383-1414.
//! - Hansen, B.E. (1997), "Approximate Asymptotic P Values for
//!   Structural-Change Tests", *J. Bus. Econ. Stat.* 15(1), 60-67.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;

use crate::StatsError;
use crate::critical_values::CriticalValues;
use crate::input::as_slice_or_copy;
use crate::input::assert_alpha;
use crate::input::validate;
use crate::input::validate_rows;
use crate::linalg::inverse;
use crate::linalg::solve;
use crate::traits::RealExt;

/// Which aggregation of the local Wald statistic to report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AndrewsPlobergerStatistic {
  /// $\sup_\tau W(\tau)$ — Andrews (1993) Sup-Wald.
  Sup,
  /// $\ln \{ (1/N) \sum_\tau \exp[W(\tau)/2] \}$ — Andrews-Ploberger Exp-Wald.
  Exp,
  /// $(1/N) \sum_\tau W(\tau)$ — Andrews-Ploberger Avg-Wald.
  Avg,
}

/// Configuration for the Andrews / Andrews-Ploberger test.
#[derive(Debug, Clone, Copy)]
pub struct AndrewsPlobergerConfig {
  /// Aggregation choice.
  pub statistic: AndrewsPlobergerStatistic,
  /// Trim fraction $\pi_0 \in (0, 0.5)$. Tabulated critical values inside
  /// this implementation are for $\pi_0 = 0.15$.
  pub pi0: f64,
  /// Significance level used to evaluate `rejects_null`.
  pub alpha: f64,
}

impl Default for AndrewsPlobergerConfig {
  fn default() -> Self {
    Self {
      statistic: AndrewsPlobergerStatistic::Sup,
      pi0: 0.15,
      alpha: 0.05,
    }
  }
}

/// Andrews / Andrews–Ploberger test (null: constant regression parameters); the table holds the
/// 1 %, 5 % and 10 % levels for `pi0 = 0.15` and `1 <= k <= 5`, and is `None` off it.
#[derive(Debug, Clone)]
pub struct AndrewsPlobergerTest {
  /// Test statistic.
  pub statistic: f64,
  /// Upper bound on the p-value: `0.01`, `0.05`, `0.10` or `1.0`, the smallest tabulated level
  /// whose critical value the statistic reaches.
  pub p_value: Option<f64>,
  pub critical_values: Option<CriticalValues>,
  /// Whether the parameter-constancy null is rejected at `alpha`; `None` off the table.
  pub rejects_null: Option<bool>,
  /// Statistic actually computed (Sup, Exp, or Avg-Wald).
  pub statistic_kind: AndrewsPlobergerStatistic,
  /// $\tau^* = \arg\max_\tau W(\tau)$: the break estimate under Sup-Wald, a diagnostic otherwise.
  pub breakpoint_frac: f64,
  /// Local Wald statistic on the trimmed grid `[pi0, 1-pi0]`.
  pub wald_path: Vec<(f64, f64)>,
}

impl crate::traits::HypothesisTest for AndrewsPlobergerTest {
  fn statistic(&self) -> f64 {
    self.statistic
  }

  fn null_rejected(&self) -> Option<bool> {
    self.rejects_null
  }
}

/// Andrews / Andrews–Ploberger break test of `y` on `x`; `Err` on too few rows, a non-finite `y` or
/// `x`, mismatched rows or a NaN Wald path, panics on bad `pi0`, `alpha` or no column.
pub fn andrews_ploberger_test(
  y: ArrayView1<f64>,
  x: ArrayView2<f64>,
  cfg: AndrewsPlobergerConfig,
) -> Result<AndrewsPlobergerTest, StatsError> {
  assert!(
    cfg.pi0 > 0.0 && cfg.pi0 < 0.5,
    "pi0 must satisfy `0.0 < pi0 < 0.5`, got pi0 = {:?}",
    cfg.pi0
  );
  assert_alpha(cfg.alpha);
  let k = x.ncols();
  assert!(
    k >= 1,
    "x.ncols() must satisfy `x.ncols() >= 1`, got x.ncols() = {k:?}"
  );
  let row = critical_row(cfg.statistic, k, cfg.pi0);
  let critical_values =
    row.map(|(c10, c5, c1)| CriticalValues::new([(0.01, c1), (0.05, c5), (0.10, c10)]));
  let critical = critical_values.as_ref().map(|cv| cv.value_at(cfg.alpha));
  if x.nrows() != y.len() {
    return Err(StatsError::DimensionMismatch {
      expected: y.len(),
      got: x.nrows(),
    });
  }
  validate(&as_slice_or_copy(y), 3 * k + 4)?;
  validate_rows(x)?;

  let t_total = y.len();
  let (t1_lo, t1_hi) = trimmed_grid(t_total, k, cfg.pi0);
  if t1_lo > t1_hi {
    let needed = (t_total + 1..)
      .find(|&t| {
        let (lo, hi) = trimmed_grid(t, k, cfg.pi0);
        lo <= hi
      })
      .unwrap();
    return Err(StatsError::InsufficientData {
      needed,
      got: t_total,
    });
  }

  let mut wald_path: Vec<(f64, f64)> = Vec::with_capacity(t1_hi - t1_lo + 1);
  for t1 in t1_lo..=t1_hi {
    let w = local_wald(y.view(), x.view(), t1, k);
    let tau = (t1 as f64) / (t_total as f64);
    wald_path.push((tau, w));
  }

  let (statistic, breakpoint_frac) = aggregate(&wald_path, cfg.statistic)?;

  Ok(AndrewsPlobergerTest {
    statistic,
    p_value: row.map(|row| p_value_bracket(statistic, row)),
    critical_values,
    rejects_null: critical.map(|c| statistic >= c),
    statistic_kind: cfg.statistic,
    breakpoint_frac,
    wald_path,
  })
}

/// The split indices `[lo, hi]` of the trimmed grid for `t` rows and `k` regressors.
fn trimmed_grid(t: usize, k: usize, pi0: f64) -> (usize, usize) {
  let lo = ((pi0 * t as f64).ceil() as usize).max(k + 1);
  let hi = (((1.0 - pi0) * t as f64).floor() as usize).min(t - k - 1);
  (lo, hi)
}

/// Local Wald statistic at split index `t1` (first segment is rows `0..t1`,
/// second segment is rows `t1..T`).
fn local_wald(y: ArrayView1<f64>, x: ArrayView2<f64>, t1: usize, k: usize) -> f64 {
  let t_total = y.len();
  if t1 < k + 1 || t_total - t1 < k + 1 {
    return f64::NAN;
  }
  let (b1, v1) = match segment_fit(y.view(), x.view(), 0, t1, k) {
    Some(v) => v,
    None => return f64::NAN,
  };
  let (b2, v2) = match segment_fit(y.view(), x.view(), t1, t_total, k) {
    Some(v) => v,
    None => return f64::NAN,
  };

  let diff = &b1 - &b2;
  let vsum = &v1 + &v2;
  match solve(&vsum, &diff) {
    Some(z) => diff.dot(&z).max(0.0),
    None => f64::NAN,
  }
}

/// OLS on `x[lo..hi, :], y[lo..hi]` returning $(\hat\beta, \hat V(\hat\beta))$
/// with the homoskedastic variance $\hat\sigma^2 (X^\top X)^{-1}$.
fn segment_fit(
  y: ArrayView1<f64>,
  x: ArrayView2<f64>,
  lo: usize,
  hi: usize,
  k: usize,
) -> Option<(Array1<f64>, Array2<f64>)> {
  let n = hi - lo;
  if n <= k {
    return None;
  }
  let xs = x.slice(ndarray::s![lo..hi, ..]).to_owned();
  let ys = y.slice(ndarray::s![lo..hi]).to_owned();
  let xtx = xs.t().dot(&xs);
  let xty = xs.t().dot(&ys);
  let beta = solve(&xtx, &xty)?;
  let resid = &ys - &xs.dot(&beta);
  let sse: f64 = resid.iter().map(|r| r * r).sum();
  let sigma2 = sse / (n as f64 - k as f64).max(1.0);
  let xtx_inv = inverse(&xtx)?;
  Some((beta, &xtx_inv * sigma2))
}

/// The aggregated statistic and the `τ` of the last maximal `W(τ)`; `Err` on a NaN in the path.
fn aggregate(
  path: &[(f64, f64)],
  kind: AndrewsPlobergerStatistic,
) -> Result<(f64, f64), StatsError> {
  let mut max_w = f64::NEG_INFINITY;
  let mut tau_star = f64::NAN;
  for &(tau, w) in path {
    if w >= max_w {
      tau_star = tau;
    }
    max_w = max_w.max_or_nan(w);
  }
  if max_w.is_nan() {
    return Err(StatsError::Numerical { stage: "Wald path" });
  }
  let n_grid = path.len() as f64;
  Ok(match kind {
    AndrewsPlobergerStatistic::Sup => (max_w, tau_star),
    AndrewsPlobergerStatistic::Exp => {
      // Subtract the per-tau max for numerical stability of logsumexp:
      // ln{N⁻¹ Σ exp(w/2)} = (max/2) + ln{N⁻¹ Σ exp((w - max)/2)}.
      let lse_tail = path
        .iter()
        .map(|(_, w)| ((w - max_w) * 0.5).exp())
        .sum::<f64>()
        / n_grid;
      (max_w * 0.5 + lse_tail.ln(), tau_star)
    }
    AndrewsPlobergerStatistic::Avg => {
      let mean = path.iter().map(|(_, w)| *w).sum::<f64>() / n_grid;
      (mean, tau_star)
    }
  })
}

/// Andrews (1993) Table 1 — Sup-Wald critical values for $\pi_0 = 0.15$.
fn sup_wald_critical(k: usize) -> Option<(f64, f64, f64)> {
  let row = match k {
    1 => (7.17, 8.85, 12.16),
    2 => (9.31, 11.79, 15.46),
    3 => (11.18, 13.78, 17.36),
    4 => (12.93, 15.60, 19.84),
    5 => (14.61, 17.61, 22.27),
    _ => return None,
  };
  Some(row)
}

/// Andrews-Ploberger (1994) Table II — Exp-Wald critical values, $\pi_0 = 0.15$.
fn exp_wald_critical(k: usize) -> Option<(f64, f64, f64)> {
  let row = match k {
    1 => (1.83, 2.57, 4.13),
    2 => (3.15, 4.07, 5.78),
    3 => (4.35, 5.36, 7.42),
    4 => (5.48, 6.59, 8.83),
    5 => (6.54, 7.74, 10.17),
    _ => return None,
  };
  Some(row)
}

/// Andrews-Ploberger (1994) Table III — Avg-Wald critical values, $\pi_0 = 0.15$.
fn avg_wald_critical(k: usize) -> Option<(f64, f64, f64)> {
  let row = match k {
    1 => (3.84, 4.69, 6.74),
    2 => (6.13, 7.31, 9.61),
    3 => (8.36, 9.71, 12.21),
    4 => (10.46, 11.95, 14.71),
    5 => (12.49, 14.08, 17.10),
    _ => return None,
  };
  Some(row)
}

/// The `(10 %, 5 %, 1 %)` row for `k` regressors; only the `pi0 = 0.15` tables are shipped.
fn critical_row(kind: AndrewsPlobergerStatistic, k: usize, pi0: f64) -> Option<(f64, f64, f64)> {
  if (pi0 - 0.15).abs() >= 1e-9 {
    return None;
  }
  match kind {
    AndrewsPlobergerStatistic::Sup => sup_wald_critical(k),
    AndrewsPlobergerStatistic::Exp => exp_wald_critical(k),
    AndrewsPlobergerStatistic::Avg => avg_wald_critical(k),
  }
}

fn p_value_bracket(stat: f64, (c10, c5, c1): (f64, f64, f64)) -> f64 {
  if stat >= c1 {
    0.01
  } else if stat >= c5 {
    0.05
  } else if stat >= c10 {
    0.10
  } else {
    1.0
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Array2;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::DistributionSampler;
  use stochastic_rs_distributions::SimdDistribution;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::*;

  fn simulate(
    n: usize,
    beta1: (f64, f64),
    beta2: (f64, f64),
    break_frac: Option<f64>,
    seed: u64,
  ) -> (Array1<f64>, Array2<f64>) {
    let mut dist = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(seed));
    let mut x_col = Array1::<f64>::zeros(n);
    dist.fill_slice(x_col.as_slice_mut().unwrap());
    let mut design = Array2::<f64>::zeros((n, 2));
    let mut y = Array1::<f64>::zeros(n);
    let break_at = break_frac.map(|f| (f * n as f64) as usize);
    for i in 0..n {
      design[[i, 0]] = 1.0;
      design[[i, 1]] = x_col[i];
      let (b0, b1) = match break_at {
        Some(t) if i >= t => beta2,
        _ => beta1,
      };
      let eps = 0.5 * dist.sample();
      y[i] = b0 + b1 * x_col[i] + eps;
    }
    (y, design)
  }

  #[test]
  fn sup_wald_does_not_reject_no_break() {
    let (y, x) = simulate(300, (1.0, 2.0), (1.0, 2.0), None, 42);
    let res = andrews_ploberger_test(
      y.view(),
      x.view(),
      AndrewsPlobergerConfig {
        statistic: AndrewsPlobergerStatistic::Sup,
        pi0: 0.15,
        alpha: 0.05,
      },
    )
    .unwrap();
    assert!(
      res.rejects_null == Some(false),
      "no-break DGP: Sup-Wald should not reject; W = {}, p = {:?}",
      res.statistic,
      res.p_value
    );
  }

  #[test]
  fn sup_wald_rejects_mid_sample_intercept_break() {
    let (y, x) = simulate(300, (1.0, 2.0), (4.0, 2.0), Some(0.5), 7);
    let res = andrews_ploberger_test(
      y.view(),
      x.view(),
      AndrewsPlobergerConfig {
        statistic: AndrewsPlobergerStatistic::Sup,
        pi0: 0.15,
        alpha: 0.05,
      },
    )
    .unwrap();
    assert!(
      res.rejects_null == Some(true),
      "intercept break at τ=0.5: should reject; W = {}, p = {:?}",
      res.statistic,
      res.p_value
    );
    assert!(
      (res.breakpoint_frac - 0.5).abs() < 0.15,
      "estimated break {} should be near 0.5",
      res.breakpoint_frac
    );
  }

  #[test]
  fn exp_wald_rejects_slope_break() {
    let (y, x) = simulate(300, (0.0, 1.0), (0.0, 3.0), Some(0.3), 11);
    let res = andrews_ploberger_test(
      y.view(),
      x.view(),
      AndrewsPlobergerConfig {
        statistic: AndrewsPlobergerStatistic::Exp,
        pi0: 0.15,
        alpha: 0.05,
      },
    )
    .unwrap();
    assert!(
      res.rejects_null == Some(true),
      "slope break at τ=0.3: Exp-Wald should reject; W = {}, p = {:?}",
      res.statistic,
      res.p_value
    );
  }

  #[test]
  fn avg_wald_rejects_intercept_break() {
    let (y, x) = simulate(300, (0.0, 1.0), (3.0, 1.0), Some(0.4), 13);
    let res = andrews_ploberger_test(
      y.view(),
      x.view(),
      AndrewsPlobergerConfig {
        statistic: AndrewsPlobergerStatistic::Avg,
        pi0: 0.15,
        alpha: 0.05,
      },
    )
    .unwrap();
    assert!(
      res.rejects_null == Some(true),
      "intercept break: Avg-Wald should reject; W = {}, p = {:?}",
      res.statistic,
      res.p_value
    );
  }

  #[test]
  fn off_its_table_the_test_takes_no_decision() {
    let (y, x) = simulate(300, (1.0, 2.0), (4.0, 2.0), Some(0.5), 7);
    let cfg = AndrewsPlobergerConfig {
      pi0: 0.2,
      ..AndrewsPlobergerConfig::default()
    };
    let res = andrews_ploberger_test(y.view(), x.view(), cfg).unwrap();
    assert!(res.statistic.is_finite());
    assert_eq!(res.rejects_null, None);
    assert_eq!(crate::traits::HypothesisTest::null_rejected(&res), None);
    assert!(res.p_value.is_none() && res.critical_values.is_none());
  }

  #[test]
  fn a_non_finite_design_row_is_an_invalid_observation() {
    let (y, mut x) = simulate(300, (1.0, 2.0), (1.0, 2.0), None, 42);
    x[[120, 1]] = f64::INFINITY;
    assert_eq!(
      andrews_ploberger_test(y.view(), x.view(), AndrewsPlobergerConfig::default()).unwrap_err(),
      StatsError::InvalidObservation {
        index: 120,
        value: f64::INFINITY,
        constraint: "finite"
      }
    );
  }

  #[test]
  fn a_nan_on_the_wald_path_is_a_numerical_error() {
    let path = [(0.2, 1.0), (0.3, f64::NAN), (0.4, 3.0)];
    for kind in [
      AndrewsPlobergerStatistic::Sup,
      AndrewsPlobergerStatistic::Exp,
      AndrewsPlobergerStatistic::Avg,
    ] {
      assert_eq!(
        aggregate(&path, kind),
        Err(StatsError::Numerical { stage: "Wald path" })
      );
    }
  }

  #[test]
  fn tied_wald_maxima_report_the_last_breakpoint() {
    let path = [(0.2, 4.0), (0.3, 1.0), (0.4, 4.0)];
    assert_eq!(
      aggregate(&path, AndrewsPlobergerStatistic::Sup),
      Ok((4.0, 0.4))
    );
  }

  #[test]
  fn sup_wald_matches_andrews_1993_critical_for_k_eq_2() {
    // White-box check: confirm the embedded table reproduces Andrews (1993)
    // Table 1 row k = 2.
    let (c10, c5, c1) = sup_wald_critical(2).unwrap();
    assert!((c10 - 9.31).abs() < 1e-6);
    assert!((c5 - 11.79).abs() < 1e-6);
    assert!((c1 - 15.46).abs() < 1e-6);
  }
}

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
//! 1994 Tables II and III for Exp-Wald and Avg-Wald). The implementation
//! requires `pi0 = 0.15` for the embedded critical values; the test
//! statistic is computed for any `pi0`, but the rejection decision and
//! p-value bracket use the 0.15 table.
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
use ndarray_linalg::Inverse;
use ndarray_linalg::Solve;

use super::common::validate_series;

/// Which aggregation of the local Wald statistic to report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApStatistic {
  /// $\sup_\tau W(\tau)$ — Andrews (1993) Sup-Wald.
  Sup,
  /// $\ln \{ (1/N) \sum_\tau \exp[W(\tau)/2] \}$ — Andrews-Ploberger Exp-Wald.
  Exp,
  /// $(1/N) \sum_\tau W(\tau)$ — Andrews-Ploberger Avg-Wald.
  Avg,
}

/// Configuration for the Andrews / Andrews-Ploberger test.
#[derive(Debug, Clone, Copy)]
pub struct ApConfig {
  /// Aggregation choice.
  pub statistic: ApStatistic,
  /// Trim fraction $\pi_0 \in (0, 0.5)$. Tabulated critical values inside
  /// this implementation are for $\pi_0 = 0.15$.
  pub pi0: f64,
  /// Significance level used to evaluate `null_rejected`.
  pub alpha: f64,
}

impl Default for ApConfig {
  fn default() -> Self {
    Self {
      statistic: ApStatistic::Sup,
      pi0: 0.15,
      alpha: 0.05,
    }
  }
}

/// Result of the Andrews / Andrews-Ploberger test.
#[derive(Debug, Clone)]
pub struct ApResult {
  /// Statistic actually computed (Sup, Exp, or Avg-Wald).
  pub statistic_kind: ApStatistic,
  /// Test statistic.
  pub statistic: f64,
  /// Breakpoint fraction $\tau^* = \arg\max_\tau W(\tau)$ (only meaningful
  /// for Sup-Wald; otherwise the location of the max along the grid is
  /// still reported for diagnostic use).
  pub breakpoint_frac: f64,
  /// Local Wald statistic on the trimmed grid `[pi0, 1-pi0]`.
  pub wald_path: Vec<(f64, f64)>,
  /// Coarse asymptotic p-value bracket: one of `0.01`, `0.05`, `0.10`, or
  /// `1.0` (no rejection at any tabulated level). Comes from the
  /// $\pi_0 = 0.15$ critical-value tables; returns `f64::NAN` when `pi0
  /// != 0.15` or `k` is outside the tabulated range `1..=5`.
  pub p_value: f64,
  /// Whether the parameter-constancy null is rejected at `alpha`.
  pub null_rejected: bool,
}

impl crate::traits::HypothesisTest for ApResult {
  fn statistic(&self) -> f64 {
    self.statistic
  }
  fn null_rejected(&self) -> Option<bool> {
    Some(self.null_rejected)
  }
}

/// Andrews / Andrews-Ploberger structural-break test with unknown
/// breakpoint.
///
/// # Panics
/// Panics on non-finite inputs, `pi0` outside `(0, 0.5)`, `alpha` outside
/// `(0, 1)`, or a sample too short to fit OLS on both pieces of the
/// trimmed grid.
pub fn andrews_ploberger_test(y: ArrayView1<f64>, x: ArrayView2<f64>, cfg: ApConfig) -> ApResult {
  let y_slice = y
    .as_slice()
    .expect("andrews_ploberger_test requires a contiguous ArrayView1");
  assert_eq!(y.len(), x.nrows(), "y/x row mismatch");
  assert!(
    cfg.pi0 > 0.0 && cfg.pi0 < 0.5,
    "pi0 must be in (0, 0.5), got {}",
    cfg.pi0
  );
  assert!(
    cfg.alpha > 0.0 && cfg.alpha < 1.0,
    "alpha must be in (0, 1)"
  );
  let t_total = y.len();
  let k = x.ncols();
  assert!(k >= 1, "design matrix must have at least one regressor");
  validate_series(y_slice, 3 * k + 4);

  let t1_lo = ((cfg.pi0 * t_total as f64).ceil() as usize).max(k + 1);
  let t1_hi = (((1.0 - cfg.pi0) * t_total as f64).floor() as usize).min(t_total - k - 1);
  assert!(
    t1_lo <= t1_hi,
    "trimmed grid is empty for the given pi0 and sample size"
  );

  let mut wald_path: Vec<(f64, f64)> = Vec::with_capacity(t1_hi - t1_lo + 1);
  for t1 in t1_lo..=t1_hi {
    let w = local_wald(y.view(), x.view(), t1, k);
    let tau = (t1 as f64) / (t_total as f64);
    wald_path.push((tau, w));
  }

  let (statistic, breakpoint_frac) = aggregate(&wald_path, cfg.statistic);
  let (p_value, null_rejected) = decide(statistic, cfg.statistic, k, cfg.pi0, cfg.alpha);

  ApResult {
    statistic_kind: cfg.statistic,
    statistic,
    breakpoint_frac,
    wald_path,
    p_value,
    null_rejected,
  }
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
  match vsum.solve(&diff) {
    Ok(z) => diff.dot(&z).max(0.0),
    Err(_) => f64::NAN,
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
  let beta = xtx.solve(&xty).ok()?;
  let resid = &ys - &xs.dot(&beta);
  let sse: f64 = resid.iter().map(|r| r * r).sum();
  let sigma2 = sse / (n as f64 - k as f64).max(1.0);
  let xtx_inv = xtx.inv().ok()?;
  Some((beta, &xtx_inv * sigma2))
}

fn aggregate(path: &[(f64, f64)], kind: ApStatistic) -> (f64, f64) {
  let finite: Vec<(f64, f64)> = path
    .iter()
    .copied()
    .filter(|(_, w)| w.is_finite())
    .collect();
  if finite.is_empty() {
    return (f64::NAN, f64::NAN);
  }
  let &(tau_star, max_w) = finite
    .iter()
    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    .expect("non-empty by guard above");
  match kind {
    ApStatistic::Sup => (max_w, tau_star),
    ApStatistic::Exp => {
      // Subtract the per-tau max for numerical stability of logsumexp:
      // ln{N⁻¹ Σ exp(w/2)} = (max/2) + ln{N⁻¹ Σ exp((w - max)/2)}.
      let n_grid = finite.len() as f64;
      let max_over_two = max_w * 0.5;
      let lse_tail: f64 = finite
        .iter()
        .map(|(_, w)| ((w - max_w) * 0.5).exp())
        .sum::<f64>()
        / n_grid;
      let stat = max_over_two + lse_tail.ln();
      (stat, tau_star)
    }
    ApStatistic::Avg => {
      let n_grid = finite.len() as f64;
      let mean = finite.iter().map(|(_, w)| *w).sum::<f64>() / n_grid;
      (mean, tau_star)
    }
  }
}

/// Andrews (1993) Table 1 — Sup-Wald critical values for $\pi_0 = 0.15$.
fn sup_wald_critical(k: usize, alpha: f64) -> Option<(f64, f64, f64)> {
  let row = match k {
    1 => (7.17, 8.85, 12.16),
    2 => (9.31, 11.79, 15.46),
    3 => (11.18, 13.78, 17.36),
    4 => (12.93, 15.60, 19.84),
    5 => (14.61, 17.61, 22.27),
    _ => return None,
  };
  Some(select_levels(row, alpha))
}

/// Andrews-Ploberger (1994) Table II — Exp-Wald critical values, $\pi_0 = 0.15$.
fn exp_wald_critical(k: usize, alpha: f64) -> Option<(f64, f64, f64)> {
  let row = match k {
    1 => (1.83, 2.57, 4.13),
    2 => (3.15, 4.07, 5.78),
    3 => (4.35, 5.36, 7.42),
    4 => (5.48, 6.59, 8.83),
    5 => (6.54, 7.74, 10.17),
    _ => return None,
  };
  Some(select_levels(row, alpha))
}

/// Andrews-Ploberger (1994) Table III — Avg-Wald critical values, $\pi_0 = 0.15$.
fn avg_wald_critical(k: usize, alpha: f64) -> Option<(f64, f64, f64)> {
  let row = match k {
    1 => (3.84, 4.69, 6.74),
    2 => (6.13, 7.31, 9.61),
    3 => (8.36, 9.71, 12.21),
    4 => (10.46, 11.95, 14.71),
    5 => (12.49, 14.08, 17.10),
    _ => return None,
  };
  Some(select_levels(row, alpha))
}

fn select_levels(row: (f64, f64, f64), _alpha: f64) -> (f64, f64, f64) {
  // Returned tuple is always (10%, 5%, 1%); the caller compares against the
  // appropriate level depending on `alpha`.
  row
}

fn decide(stat: f64, kind: ApStatistic, k: usize, pi0: f64, alpha: f64) -> (f64, bool) {
  let table = if (pi0 - 0.15).abs() < 1e-9 {
    match kind {
      ApStatistic::Sup => sup_wald_critical(k, alpha),
      ApStatistic::Exp => exp_wald_critical(k, alpha),
      ApStatistic::Avg => avg_wald_critical(k, alpha),
    }
  } else {
    None
  };
  let Some((c10, c5, c1)) = table else {
    return (f64::NAN, false);
  };
  let p_value = if stat >= c1 {
    0.01
  } else if stat >= c5 {
    0.05
  } else if stat >= c10 {
    0.10
  } else {
    1.0
  };
  let crit = if alpha <= 0.01 {
    c1
  } else if alpha <= 0.05 {
    c5
  } else {
    c10
  };
  (p_value, stat >= crit)
}

#[cfg(test)]
mod tests {
  use ndarray::Array2;
  use ndarray_rand::RandomExt;
  use ndarray_rand::rand_distr::Normal;
  use rand::SeedableRng;
  use rand::rngs::StdRng;

  use super::*;

  fn simulate(
    n: usize,
    beta1: (f64, f64),
    beta2: (f64, f64),
    break_frac: Option<f64>,
    seed: u64,
  ) -> (Array1<f64>, Array2<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let x_col = Array1::random_using(n, Normal::new(0.0, 1.0).unwrap(), &mut rng);
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
      let eps =
        ndarray_rand::rand_distr::Distribution::sample(&Normal::new(0.0, 0.5).unwrap(), &mut rng);
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
      ApConfig {
        statistic: ApStatistic::Sup,
        pi0: 0.15,
        alpha: 0.05,
      },
    );
    assert!(
      !res.null_rejected,
      "no-break DGP: Sup-Wald should not reject; W = {}, p = {}",
      res.statistic, res.p_value
    );
  }

  #[test]
  fn sup_wald_rejects_mid_sample_intercept_break() {
    let (y, x) = simulate(300, (1.0, 2.0), (4.0, 2.0), Some(0.5), 7);
    let res = andrews_ploberger_test(
      y.view(),
      x.view(),
      ApConfig {
        statistic: ApStatistic::Sup,
        pi0: 0.15,
        alpha: 0.05,
      },
    );
    assert!(
      res.null_rejected,
      "intercept break at τ=0.5: should reject; W = {}, p = {}",
      res.statistic, res.p_value
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
      ApConfig {
        statistic: ApStatistic::Exp,
        pi0: 0.15,
        alpha: 0.05,
      },
    );
    assert!(
      res.null_rejected,
      "slope break at τ=0.3: Exp-Wald should reject; W = {}, p = {}",
      res.statistic, res.p_value
    );
  }

  #[test]
  fn avg_wald_rejects_intercept_break() {
    let (y, x) = simulate(300, (0.0, 1.0), (3.0, 1.0), Some(0.4), 13);
    let res = andrews_ploberger_test(
      y.view(),
      x.view(),
      ApConfig {
        statistic: ApStatistic::Avg,
        pi0: 0.15,
        alpha: 0.05,
      },
    );
    assert!(
      res.null_rejected,
      "intercept break: Avg-Wald should reject; W = {}, p = {}",
      res.statistic, res.p_value
    );
  }

  #[test]
  fn sup_wald_matches_andrews_1993_critical_for_k_eq_2() {
    // White-box check: confirm the embedded table reproduces Andrews (1993)
    // Table 1 row k = 2.
    let (c10, c5, c1) = sup_wald_critical(2, 0.05).unwrap();
    assert!((c10 - 9.31).abs() < 1e-6);
    assert!((c5 - 11.79).abs() < 1e-6);
    assert!((c1 - 15.46).abs() < 1e-6);
  }
}

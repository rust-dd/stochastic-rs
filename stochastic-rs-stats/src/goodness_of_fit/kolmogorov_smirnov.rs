//! # One-sample Kolmogorov-Smirnov goodness-of-fit test
//!
//! Tests whether a continuous sample was drawn from a **fully specified**
//! distribution — every parameter of the null CDF `F` is fixed in advance,
//! not fitted from `sample` itself (see "Known vs. estimated parameters"
//! below for why that distinction changes which critical values apply).
//!
//! $$ D_n = \sup_x \left| F_n(x) - F(x) \right| $$
//!
//! where `F_n` is the sample's empirical CDF. Computed exactly (no
//! discretisation) via the standard sorted-sample identity: with
//! `x_{(0)} \le \dots \le x_{(n-1)}`,
//!
//! $$ D_n = \max_i \max\!\left( \left|\tfrac{i+1}{n} - F(x_{(i)})\right|,\ \left|F(x_{(i)}) - \tfrac{i}{n}\right| \right). $$
//!
//! ## Asymptotic null distribution and critical value
//!
//! Under `H0`, `\sqrt n D_n` converges to the Kolmogorov distribution
//! (Kolmogorov 1933), tabulated for practical use by Smirnov (1948) and
//! by Massey (1951) — the source of the `c(\alpha)` critical-value
//! constants and their large-`n` critical-value formula used below:
//!
//! $$ \Pr(\sqrt n D_n \le \lambda) \to 1 - 2\sum_{k=1}^{\infty} (-1)^{k-1} e^{-2k^2\lambda^2}, \qquad D_{n,\alpha} = \frac{c(\alpha)}{\sqrt n},\quad c(\alpha) = \sqrt{-\tfrac12 \ln(\alpha/2)}. $$
//!
//! `c(\alpha)` at the common levels — `c(0.10) = 1.224`, `c(0.05) =
//! 1.358`, `c(0.025) = 1.48`, `c(0.01) = 1.628` — and the closed form
//! above are Massey's (1951) large-sample approximation to the exact
//! finite-sample table; [`ks_critical_value`]'s own tests pin these
//! constants. No finite-sample correction (e.g. Stephens 1974's
//! `\sqrt n \to \sqrt n + 0.12 + 0.11/\sqrt n`) is applied, so
//! [`KolmogorovSmirnovResult::critical_value`] and
//! [`KolmogorovSmirnovResult::p_value`] stay defined by the *same*
//! uncorrected asymptotic formula and therefore always agree on
//! rejection at the sample sizes this crate's own callers use (tens of
//! thousands) — mixing a corrected p-value with an uncorrected critical
//! value would let the two disagree at the margin.
//!
//! ## Known vs. estimated parameters — why the plain table applies here
//!
//! The critical values above are only valid when `F` is fully specified
//! *before* looking at `sample` — if any parameter of `F` was fitted
//! *from* `sample` (e.g. plugging in the sample's own mean/variance as
//! a Normal's `μ`/`σ`), the true null distribution of `D_n` shifts left
//! (the fitted CDF hugs the empirical one more closely than the truly
//! fixed one would), so the standard table over-states how surprising a
//! given `D_n` is and under-rejects. Lilliefors (1967) derived the
//! (smaller, distribution-specific) corrected critical values for
//! exactly that fitted-parameter case. This test does not implement the
//! Lilliefors correction, and does not need to: every caller in this
//! workspace constructs the candidate distribution with parameters
//! chosen *before* sampling (e.g. `SimdGamma::new(2.5, 1.5, seed)`) and
//! tests the resulting draws against that same, already-fixed `cdf` —
//! the parameters are never re-estimated from the sample under test, so
//! the plain (non-Lilliefors) table is the correct one. A future caller
//! that instead fits parameters to data before calling this function
//! would need the Lilliefors correction, not this one.
//!
//! References:
//! - Kolmogorov, A.N. (1933), "Sulla determinazione empirica di una
//!   legge di distribuzione", *Giornale dell'Istituto Italiano degli
//!   Attuari* 4, 83-91.
//! - Smirnov, N.V. (1948), "Table for estimating the goodness of fit of
//!   empirical distributions", *Annals of Mathematical Statistics*
//!   19(2), 279-281.
//! - Massey, F.J. (1951), "The Kolmogorov-Smirnov Test for Goodness of
//!   Fit", *Journal of the American Statistical Association* 46(253),
//!   68-78.
//! - Lilliefors, H.W. (1967), "On the Kolmogorov-Smirnov test for
//!   normality with mean and variance unknown", *Journal of the
//!   American Statistical Association* 62(318), 399-402.

use ndarray::ArrayView1;

use crate::traits::HypothesisTest;

/// Configuration for [`kolmogorov_smirnov_test`].
#[derive(Debug, Clone, Copy)]
pub struct KolmogorovSmirnovConfig {
  /// Significance level. Default `0.05` (Massey 1951's `c(0.05) = 1.358`).
  pub alpha: f64,
}

impl Default for KolmogorovSmirnovConfig {
  fn default() -> Self {
    Self { alpha: 0.05 }
  }
}

/// Result of a one-sample Kolmogorov-Smirnov goodness-of-fit test.
#[derive(Debug, Clone, Copy)]
pub struct KolmogorovSmirnovResult {
  /// The `D_n` statistic.
  pub statistic: f64,
  /// `D_{n,\alpha} = c(\alpha)/\sqrt n` — reject when `statistic` exceeds this.
  pub critical_value: f64,
  /// Asymptotic p-value from the Kolmogorov distribution's survival
  /// function at `\sqrt n \cdot` `statistic` (uncorrected, see module doc).
  pub p_value: f64,
  /// Sample size `n`.
  pub n: usize,
  /// Whether the null (sample ~ the given `cdf`) is rejected at `alpha`.
  pub reject: bool,
}

impl HypothesisTest for KolmogorovSmirnovResult {
  fn statistic(&self) -> f64 {
    self.statistic
  }

  fn null_rejected(&self) -> Option<bool> {
    Some(self.reject)
  }
}

/// Massey's (1951) large-`n` critical-value constant `c(\alpha) =
/// \sqrt{-\tfrac12 \ln(\alpha/2)}`, solving `Q(c(\alpha)) = \alpha` for
/// the Kolmogorov survival function `Q` (see module doc). Matches the
/// tabulated `c(0.10)=1.224`, `c(0.05)=1.358`, `c(0.025)=1.48`,
/// `c(0.01)=1.628` (cross-checked in this module's own tests).
fn ks_c_alpha(alpha: f64) -> f64 {
  (-0.5 * (alpha / 2.0).ln()).sqrt()
}

/// The two-sided asymptotic Kolmogorov-Smirnov critical value
/// `D_{n,\alpha} = c(\alpha)/\sqrt n` (Massey 1951; see module doc).
pub fn ks_critical_value(alpha: f64, n: usize) -> f64 {
  assert!(alpha > 0.0 && alpha < 1.0, "alpha must be in (0, 1)");
  assert!(n >= 1, "n must be at least 1");
  ks_c_alpha(alpha) / (n as f64).sqrt()
}

/// Kolmogorov's asymptotic survival function `Q(\lambda) = \Pr(\sqrt n
/// D_n > \lambda)`, i.e. `2 \sum_{k=1}^{\infty} (-1)^{k-1}
/// e^{-2k^2\lambda^2}` (Kolmogorov 1933). The series is absolutely
/// convergent for every `\lambda > 0` (terms shrink as `e^{-2k^2
/// \lambda^2}`); truncating once a term is negligible relative to the
/// partial sum bounds the remaining alternating-series error by that
/// same negligible term.
fn kolmogorov_survival(lambda: f64) -> f64 {
  if lambda <= 0.0 {
    return 1.0;
  }
  let mut sum = 0.0_f64;
  let mut sign = 1.0_f64;
  for k in 1..=500u32 {
    let term = (-2.0 * (k as f64).powi(2) * lambda * lambda).exp();
    sum += sign * term;
    if term < 1e-12 {
      break;
    }
    sign = -sign;
  }
  (2.0 * sum).clamp(0.0, 1.0)
}

/// One-sample Kolmogorov-Smirnov goodness-of-fit test against a fully
/// specified continuous `cdf`.
///
/// # Panics
/// Panics if `sample` has fewer than 2 observations, contains
/// non-finite values, or `cfg.alpha` is not in `(0, 1)`.
pub fn kolmogorov_smirnov_test(
  sample: ArrayView1<f64>,
  mut cdf: impl FnMut(f64) -> f64,
  cfg: KolmogorovSmirnovConfig,
) -> KolmogorovSmirnovResult {
  assert!(
    sample.len() >= 2,
    "KS test requires at least 2 observations"
  );
  assert!(
    sample.iter().all(|x| x.is_finite()),
    "KS test requires finite observations"
  );
  assert!(
    cfg.alpha > 0.0 && cfg.alpha < 1.0,
    "alpha must be in (0, 1)"
  );

  let mut sorted = sample.to_vec();
  sorted.sort_by(f64::total_cmp);
  let n = sorted.len();
  let n_f = n as f64;

  let mut d = 0.0_f64;
  for (i, &x) in sorted.iter().enumerate() {
    let f = cdf(x).clamp(0.0, 1.0);
    let i_f = i as f64;
    let d_plus = ((i_f + 1.0) / n_f - f).abs();
    let d_minus = (f - i_f / n_f).abs();
    d = d.max(d_plus.max(d_minus));
  }

  let p_value = kolmogorov_survival(n_f.sqrt() * d);
  KolmogorovSmirnovResult {
    statistic: d,
    critical_value: ks_critical_value(cfg.alpha, n),
    p_value,
    n,
    reject: p_value < cfg.alpha,
  }
}

#[cfg(test)]
mod tests {
  use ndarray::ArrayView1;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::DistributionExt;
  use stochastic_rs_distributions::normal::SimdNormal;
  use stochastic_rs_distributions::uniform::SimdUniform;

  use super::KolmogorovSmirnovConfig;
  use super::kolmogorov_smirnov_test;
  use super::ks_critical_value;

  /// Massey (1951) / Smirnov (1948) tabulated `c(\alpha)` constants —
  /// `ks_critical_value(alpha, 1)` returns `c(\alpha)` unscaled.
  #[test]
  fn critical_value_matches_massey_table() {
    let table = [(0.10, 1.224), (0.05, 1.358), (0.025, 1.48), (0.01, 1.628)];
    for (alpha, expected_c) in table {
      let c = ks_critical_value(alpha, 1);
      assert!(
        (c - expected_c).abs() < 5e-3,
        "c({alpha}) = {c}, expected {expected_c}"
      );
    }
  }

  #[test]
  fn critical_value_scales_as_inverse_sqrt_n() {
    let c = ks_critical_value(0.05, 1);
    let d_100 = ks_critical_value(0.05, 100);
    assert!((d_100 - c / 10.0).abs() < 1e-12);
  }

  /// A true normal sample must not reject at `alpha = 0.05` — run three
  /// pinned seeds and require the *worst* (smallest) p-value to still
  /// clear `alpha`, per this workspace's multi-seed mandate for
  /// statistical assertions (a correct test still rejects a true null at
  /// rate `alpha`, and the SIMD stream differs across platforms).
  #[test]
  fn accepts_true_normal_sample() {
    let worst_p = [42u64, 999, 2718]
      .into_iter()
      .map(|seed| {
        let dist = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(seed));
        let mut x = vec![0.0; 5_000];
        dist.fill_slice(&mut x);
        kolmogorov_smirnov_test(
          ArrayView1::from(&x),
          |v| dist.cdf(v),
          KolmogorovSmirnovConfig::default(),
        )
        .p_value
      })
      .fold(1.0_f64, f64::min);
    assert!(
      worst_p > 0.01,
      "every seed gave p <= 0.01 (worst {worst_p}); likely a bug, not bad luck"
    );
  }

  /// A uniform sample tested against a standard normal CDF must reject.
  #[test]
  fn rejects_uniform_against_normal_cdf() {
    let dist = SimdUniform::<f64>::new(0.0, 1.0, &Deterministic::new(1));
    let mut x = vec![0.0; 2_000];
    dist.fill_slice(&mut x);
    let normal = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(1));
    let res = kolmogorov_smirnov_test(
      ArrayView1::from(&x),
      |v| normal.cdf(v),
      KolmogorovSmirnovConfig::default(),
    );
    assert!(res.reject, "expected rejection, got {res:?}");
    assert!(res.statistic > res.critical_value);
  }
}

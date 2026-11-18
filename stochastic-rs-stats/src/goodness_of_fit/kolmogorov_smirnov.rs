//! One-sample Kolmogorov–Smirnov test of a continuous sample against a fully specified CDF.
//!
//! $$
//! D_n = \sup_x \left| F_n(x) - F(x) \right| = \max_i \max\!\left(
//! \left|\tfrac{i+1}{n} - F(x_{(i)})\right|,\ \left|F(x_{(i)}) - \tfrac{i}{n}\right| \right)
//! $$
//!
//! $$
//! \Pr(\sqrt n D_n \le \lambda) \to 1 - 2\sum_{k=1}^{\infty} (-1)^{k-1} e^{-2k^2\lambda^2}, \qquad
//! D_{n,\alpha} = \frac{c(\alpha)}{\sqrt n},\quad c(\alpha) = \sqrt{-\tfrac12 \ln(\alpha/2)}
//! $$
//!
//! No finite-sample or Lilliefors correction: `F` must not be fitted from the sample under test.
//!
//! Reference: Kolmogorov (1933), Giornale dell'Istituto Italiano degli Attuari 4, 83-91
//! Reference: Smirnov (1948), Annals of Mathematical Statistics 19(2), 279-281
//! Reference: Massey (1951), JASA 46(253), 68-78
//! Reference: Lilliefors (1967), JASA 62(318), 399-402

use ndarray::ArrayView1;

use crate::StatsError;
use crate::critical_values::CriticalValues;
use crate::input::assert_alpha;
use crate::input::validate;
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

/// One-sample Kolmogorov–Smirnov test (null: the given continuous `cdf`), decided by its p-value.
#[derive(Debug, Clone)]
pub struct KolmogorovSmirnovTest {
  /// The `D_n` statistic.
  pub statistic: f64,
  /// Asymptotic p-value from the Kolmogorov distribution's survival
  /// function at `\sqrt n \cdot` `statistic` (uncorrected, see module doc).
  pub p_value: Option<f64>,
  /// `D_{n,\alpha} = c(\alpha)/\sqrt n` at the `alpha` the test was run with.
  pub critical_values: Option<CriticalValues>,
  /// Whether the null (sample ~ the given `cdf`) is rejected at `alpha`.
  pub rejects_null: bool,
  /// Sample size `n`.
  pub nobs: usize,
}

impl HypothesisTest for KolmogorovSmirnovTest {
  fn statistic(&self) -> f64 {
    self.statistic
  }

  fn null_rejected(&self) -> Option<bool> {
    Some(self.rejects_null)
  }
}

/// Massey's (1951) large-`n` constant $c(\alpha) = \sqrt{-\tfrac12 \ln(\alpha/2)}$, which solves
/// $Q(c) = \alpha$: `c(0.10) = 1.224`, `c(0.05) = 1.358`, `c(0.025) = 1.48`, `c(0.01) = 1.628`.
fn ks_c_alpha(alpha: f64) -> f64 {
  (-0.5 * (alpha / 2.0).ln()).sqrt()
}

/// The two-sided asymptotic critical value $D_{n,\alpha} = c(\alpha)/\sqrt n$ (Massey 1951); panics
/// unless `0 < alpha < 1` and `n >= 1`.
pub fn ks_critical_value(alpha: f64, n: usize) -> f64 {
  assert!(alpha > 0.0 && alpha < 1.0, "alpha must be in (0, 1)");
  assert!(n >= 1, "n must be at least 1");
  ks_c_alpha(alpha) / (n as f64).sqrt()
}

/// Kolmogorov's survival function $Q(\lambda) = 2 \sum_{k \ge 1} (-1)^{k-1} e^{-2k^2\lambda^2}$,
/// summed until a term is negligible against the partial sum, which bounds the alternating rest.
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

/// One-sample Kolmogorov–Smirnov test against a fully specified continuous `cdf`; `Err` below 2
/// points or on a non-finite one, and panics unless `0 < alpha < 1`.
pub fn kolmogorov_smirnov_test(
  sample: ArrayView1<f64>,
  mut cdf: impl FnMut(f64) -> f64,
  cfg: KolmogorovSmirnovConfig,
) -> Result<KolmogorovSmirnovTest, StatsError> {
  assert_alpha(cfg.alpha);
  let mut sorted = sample.to_vec();
  validate(&sorted, 2)?;
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
  Ok(KolmogorovSmirnovTest {
    statistic: d,
    p_value: Some(p_value),
    critical_values: Some(CriticalValues::new([(
      cfg.alpha,
      ks_critical_value(cfg.alpha, n),
    )])),
    rejects_null: p_value < cfg.alpha,
    nobs: n,
  })
}

#[cfg(test)]
mod tests {
  use ndarray::ArrayView1;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::DistributionExt;
  use stochastic_rs_distributions::DistributionSampler;
  use stochastic_rs_distributions::SimdDistribution;
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

  /// A true normal sample is not rejected at `alpha = 0.05`: the largest p-value of three pinned
  /// seeds clears it, as a correct test rejects a true null at rate `alpha` on any one seed.
  #[test]
  fn accepts_true_normal_sample() {
    let best_p = [42u64, 999, 2718]
      .into_iter()
      .map(|seed| {
        let dist = SimdNormal::<f64>::new(0.0, 1.0);
        let mut x = vec![0.0; 5_000];
        dist.seeded(&Deterministic::new(seed)).fill_slice(&mut x);
        kolmogorov_smirnov_test(
          ArrayView1::from(&x),
          |v| dist.cdf(v).unwrap(),
          KolmogorovSmirnovConfig::default(),
        )
        .unwrap()
        .p_value
        .unwrap()
      })
      .fold(0.0_f64, f64::max);
    assert!(
      best_p > 0.01,
      "every seed gave p <= 0.01 (best {best_p}); likely a bug, not bad luck"
    );
  }

  /// A uniform sample tested against a standard normal CDF must reject.
  #[test]
  fn rejects_uniform_against_normal_cdf() {
    let mut dist = SimdUniform::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(1));
    let mut x = vec![0.0; 2_000];
    dist.fill_slice(&mut x);
    let normal = SimdNormal::<f64>::new(0.0, 1.0);
    let res = kolmogorov_smirnov_test(
      ArrayView1::from(&x),
      |v| normal.cdf(v).unwrap(),
      KolmogorovSmirnovConfig::default(),
    )
    .unwrap();
    assert!(res.rejects_null, "expected rejection, got {res:?}");
    assert!(res.statistic > res.critical_values.as_ref().unwrap().at(0.05).unwrap());
  }
}

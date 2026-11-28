//! One-sample Kolmogorov–Smirnov test against a fully specified CDF, with the asymptotic p-value.
//!
//! $$
//! D_n = \sup_x \left| F_n(x) - F(x) \right| = \max_i \max\!\left(
//! \left|\tfrac{i+1}{n} - F(x_{(i)})\right|,\ \left|F(x_{(i)}) - \tfrac{i}{n}\right| \right)
//! $$
//!
//! $$
//! \Pr(\sqrt n D_n > \lambda) \to Q(\lambda) = 2\sum_{k=1}^{\infty} (-1)^{k-1} e^{-2k^2\lambda^2},
//! \qquad D_{n,\alpha} = \frac{c(\alpha)}{\sqrt n},\quad Q(c(\alpha)) = \alpha
//! $$
//!
//! Nothing is fitted, so a constant sample tests as any; `F` must not come from it (Lilliefors).
//!
//! Reference: Kolmogorov (1933), Giornale dell'Istituto Italiano degli Attuari 4, 83-91
//! Reference: Smirnov (1948), Ann. Math. Stat. 19(2), 279-281, DOI: 10.1214/aoms/1177730256
//! Reference: Massey (1951), JASA 46(253), 68-78, DOI: 10.1080/01621459.1951.10500769
//! Reference: Lilliefors (1967), JASA 62(318), 399-402, DOI: 10.1080/01621459.1967.10482916

use ndarray::ArrayView1;

use crate::StatsError;
use crate::critical_values::CriticalValues;
use crate::input::assert_alpha;
use crate::input::validate;
use crate::traits::HypothesisTest;
use crate::traits::RealExt;

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
  /// Asymptotic p-value, Kolmogorov's survival function $Q$ at $\sqrt n$ times `statistic`
  /// (uncorrected, see module doc).
  pub p_value: Option<f64>,
  /// $D_{n,\alpha} = c(\alpha)/\sqrt n$ at the `alpha` the test was run with.
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

/// $c(\alpha)$ with $Q(c) = \alpha$, bisected below Massey's (1951) one-term
/// $\sqrt{-\tfrac12 \ln(\alpha/2)}$, where the terms of $Q$ past its first sum to less than zero.
fn ks_c_alpha(alpha: f64) -> f64 {
  let (mut low, mut high) = (0.0_f64, (-0.5 * (alpha / 2.0).ln()).sqrt());
  loop {
    let mid = 0.5 * (low + high);
    if mid <= low || mid >= high {
      return high;
    }
    if kolmogorov_survival(mid) > alpha {
      low = mid;
    } else {
      high = mid;
    }
  }
}

/// The two-sided asymptotic critical value $D_{n,\alpha} = c(\alpha)/\sqrt n$, where the p-value
/// meets `alpha`; panics unless `0 < alpha < 1` and `n >= 1`.
pub fn ks_critical_value(alpha: f64, n: usize) -> f64 {
  assert_alpha(alpha);
  assert!(n >= 1, "n must satisfy `n >= 1`, got n = {n:?}");
  ks_c_alpha(alpha) / (n as f64).sqrt()
}

/// Kolmogorov's survival function $Q(\lambda) = 2 \sum_{k \ge 1} (-1)^{k-1} e^{-2k^2\lambda^2}$,
/// summed until a term falls below `1e-12`, which bounds the alternating rest.
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
/// points, on a non-finite one or a `cdf` value NaN or off `[0, 1]`; panics unless `0 < alpha < 1`.
pub fn kolmogorov_smirnov_test(
  sample: ArrayView1<f64>,
  mut cdf: impl FnMut(f64) -> f64,
  cfg: KolmogorovSmirnovConfig,
) -> Result<KolmogorovSmirnovTest, StatsError> {
  assert_alpha(cfg.alpha);
  let mut sorted = sample.to_vec();
  validate(&sorted, 2)?;
  sorted.sort_by(f64::nan_last);
  let n = sorted.len();
  let n_f = n as f64;

  let mut d = 0.0_f64;
  let mut off = Vec::new();
  for (i, &x) in sorted.iter().enumerate() {
    let f = cdf(x);
    if !(0.0..=1.0).contains(&f) {
      off.push(x);
      continue;
    }
    let i_f = i as f64;
    let d_plus = ((i_f + 1.0) / n_f - f).abs();
    let d_minus = (f - i_f / n_f).abs();
    d = d.max_or_nan(d_plus.max_or_nan(d_minus));
  }
  if !off.is_empty() {
    return Err(off_the_unit_interval(sample, &off));
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

/// `InvalidObservation` at the first observation of `sample` in `off`, sorted: those whose `cdf`
/// left `[0, 1]`.
fn off_the_unit_interval(sample: ArrayView1<f64>, off: &[f64]) -> StatsError {
  let (index, &value) = sample
    .iter()
    .enumerate()
    .find(|(_, v)| off.binary_search_by(|o| o.nan_last(v)).is_ok())
    .expect("every point of `off` is drawn from the sample");
  StatsError::InvalidObservation {
    index,
    value,
    constraint: "0 <= cdf(x) <= 1",
  }
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
  use super::kolmogorov_survival;
  use super::ks_critical_value;
  use crate::StatsError;

  /// Massey (1951) / Smirnov (1948) tabulated $c(\alpha)$ constants —
  /// `ks_critical_value(alpha, 1)` returns $c(\alpha)$ unscaled.
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

  /// The critical value is the edge of the region `p < alpha` that `rejects_null` reads: a hair
  /// past it rejects and a hair short of it does not.
  #[test]
  fn the_critical_value_is_where_the_p_value_crosses_alpha() {
    for alpha in [0.001, 0.01, 0.05, 0.10, 0.25, 0.5, 0.9] {
      let c = ks_critical_value(alpha, 1);
      let past = kolmogorov_survival(c * (1.0 + 1e-9));
      let short = kolmogorov_survival(c * (1.0 - 1e-9));
      assert!(
        past < alpha && short > alpha,
        "alpha {alpha}: Q = {past} past {c} and {short} short of it"
      );
    }
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

  /// A `cdf` NaN at one point, NaN everywhere or above 1 is refused at the first such observation
  /// of the sample, not dropped from the supremum or clamped into it.
  #[test]
  fn a_cdf_off_the_unit_interval_is_invalid_observation() {
    let x = [0.3, -0.5, 0.9, -0.5, 0.1];
    let normal = SimdNormal::<f64>::new(0.0, 1.0);
    let cfg = KolmogorovSmirnovConfig::default();
    let refused = |index: usize, value: f64| StatsError::InvalidObservation {
      index,
      value,
      constraint: "0 <= cdf(x) <= 1",
    };
    let nan_at_one = |v: f64| {
      if v < 0.0 {
        f64::NAN
      } else {
        normal.cdf(v).unwrap()
      }
    };
    assert_eq!(
      kolmogorov_smirnov_test(ArrayView1::from(&x), nan_at_one, cfg).unwrap_err(),
      refused(1, -0.5)
    );
    assert_eq!(
      kolmogorov_smirnov_test(ArrayView1::from(&x), |_| f64::NAN, cfg).unwrap_err(),
      refused(0, 0.3)
    );
    let above_one = |v: f64| if v > 0.5 { 1.5 } else { normal.cdf(v).unwrap() };
    assert_eq!(
      kolmogorov_smirnov_test(ArrayView1::from(&x), above_one, cfg).unwrap_err(),
      refused(2, 0.9)
    );
  }
}

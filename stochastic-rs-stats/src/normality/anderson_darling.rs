//! Anderson–Darling test for normality with the mean and the variance estimated from the sample.
//!
//! The p-value of `A²(1 + 0.75/n + 2.25/n²)` is Stephens' (1986) Table 4.9, as nortest computes it.
//!
//! Reference: Anderson–Darling (1952), Ann. Math. Stat. 23(2), 193–212, 10.1214/aoms/1177729437
//! Reference: Stephens (1974), JASA 69(347), 730–737, DOI: 10.1080/01621459.1974.10480196
//! Reference: Stephens (1986), Goodness-of-Fit Techniques, Table 4.9 via nortest, ISBN 0824774876
//! Reference: Gross & Ligges (2015), nortest 1.0-4, `ad.test`, DOI: 10.32614/CRAN.package.nortest

use ndarray::ArrayView1;
use stochastic_rs_distributions::special::ln_norm_cdf;

use crate::StatsError;
use crate::critical_values::CriticalValues;
use crate::input::as_slice_or_copy;
use crate::input::assert_alpha;
use crate::input::reject_constant;
use crate::input::validate;
use crate::traits::RealExt;

/// Configuration for the Anderson-Darling normality test.
#[derive(Debug, Clone, Copy)]
pub struct AndersonDarlingConfig {
  /// Significance level used to compute `rejects_null`.
  pub alpha: f64,
}

impl Default for AndersonDarlingConfig {
  fn default() -> Self {
    Self { alpha: 0.05 }
  }
}

/// Anderson–Darling test (null: normality), decided by Stephens' (1986) p-value.
#[derive(Debug, Clone)]
pub struct AndersonDarlingTest {
  /// Raw A^2 statistic, the sample standardised by its `n − 1` deviation, as nortest's `sd`.
  pub statistic: f64,
  /// Stephens' (1986) p-value of `adjusted_statistic`, `3.7e-24` from 10 on.
  pub p_value: Option<f64>,
  pub critical_values: Option<CriticalValues>,
  /// Whether normality is rejected at `alpha`.
  pub rejects_null: bool,
  /// Stephens finite-sample adjusted statistic.
  pub adjusted_statistic: f64,
}

impl crate::traits::HypothesisTest for AndersonDarlingTest {
  fn statistic(&self) -> f64 {
    self.statistic
  }

  fn null_rejected(&self) -> Option<bool> {
    Some(self.rejects_null)
  }
}

fn mean_sd(sample: &[f64]) -> (f64, f64) {
  let n = sample.len() as f64;
  let mean = sample.iter().sum::<f64>() / n;
  let var = sample
    .iter()
    .map(|&x| {
      let d = x - mean;
      d * d
    })
    .sum::<f64>()
    / (n - 1.0);
  (mean, var.sqrt())
}

/// Table 4.9's four formulas, the last only below 10, where nortest answers `3.7e-24` instead: its
/// exponent turns up at 153.5 and passes zero at 306.7.
fn ad_pvalue_from_adjusted_statistic(a: f64) -> f64 {
  if a < 0.2 {
    1.0 - (-13.436 + 101.14 * a - 223.73 * a * a).exp()
  } else if a < 0.34 {
    1.0 - (-8.318 + 42.796 * a - 59.938 * a * a).exp()
  } else if a < 0.6 {
    (0.9177 - 4.279 * a - 1.38 * a * a).exp()
  } else if a < 10.0 {
    (1.2937 - 5.709 * a + 0.0186 * a * a).exp()
  } else if a >= 10.0 {
    3.7e-24
  } else {
    f64::NAN
  }
}

/// Anderson–Darling normality test with estimated mean and variance; `Err` below 8 points, on a
/// non-finite or a constant sample, and panics unless `0 < alpha < 1`.
pub fn anderson_darling_normal_test(
  sample: ArrayView1<f64>,
  cfg: AndersonDarlingConfig,
) -> Result<AndersonDarlingTest, StatsError> {
  assert_alpha(cfg.alpha);
  let sample = as_slice_or_copy(sample);
  validate(&sample, 8)?;

  let mut sorted = sample.to_vec();
  sorted.sort_by(f64::nan_last);
  let n = sorted.len();
  let n_f = n as f64;
  reject_constant(&sorted, "constant sample")?;

  let (mean, sd) = mean_sd(&sorted);
  // `ln Φ` and `ln Φ(−z)` keep the far tails, where `Φ` itself rounds to 0 or 1.
  let ln_cdf = sorted
    .iter()
    .map(|&x| ln_norm_cdf((x - mean) / sd))
    .collect::<Vec<_>>();
  let ln_sf = sorted
    .iter()
    .map(|&x| ln_norm_cdf(-(x - mean) / sd))
    .collect::<Vec<_>>();
  let sum = (0..n)
    .map(|i| (2 * i + 1) as f64 * (ln_cdf[i] + ln_sf[n - 1 - i]))
    .sum::<f64>();

  let statistic = -n_f - sum / n_f;
  let adjusted_statistic = statistic * (1.0 + 0.75 / n_f + 2.25 / (n_f * n_f));
  let p_value = ad_pvalue_from_adjusted_statistic(adjusted_statistic);

  Ok(AndersonDarlingTest {
    statistic,
    p_value: Some(p_value),
    critical_values: None,
    rejects_null: p_value < cfg.alpha,
    adjusted_statistic,
  })
}

#[cfg(test)]
mod tests {
  use ndarray::ArrayView1;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::DistributionSampler;
  use stochastic_rs_distributions::SimdDistribution;
  use stochastic_rs_distributions::exp::SimdExp;
  use stochastic_rs_distributions::normal::SimdNormal;
  use stochastic_rs_distributions::uniform::SimdUniform;

  use super::AndersonDarlingConfig;
  use super::ad_pvalue_from_adjusted_statistic;
  use super::anderson_darling_normal_test;
  use crate::StatsError;

  #[test]
  fn anderson_darling_accepts_normal_sample() {
    // The best of three seeds: the test rejects a true normal about 1% of the time and the stream
    // differs by platform; the seed goes to the constructor, the one place `fill_slice` reads it.
    let best_p = [42u64, 123, 999]
      .into_iter()
      .map(|seed| {
        let mut dist = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(seed));
        let mut x = vec![0.0; 4000];
        dist.fill_slice(&mut x);
        anderson_darling_normal_test(ArrayView1::from(&x), AndersonDarlingConfig::default())
          .unwrap()
          .p_value
          .unwrap()
      })
      .fold(0.0_f64, f64::max);

    assert!(
      best_p > 0.01,
      "all seeds gave p-value <= 0.01 (best {best_p}); likely a bug"
    );
  }

  #[test]
  fn anderson_darling_rejects_uniform_sample() {
    let mut dist = SimdUniform::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(42));
    let mut x = vec![0.0; 4000];
    dist.fill_slice(&mut x);

    let res =
      anderson_darling_normal_test(ArrayView1::from(&x), AndersonDarlingConfig::default()).unwrap();
    assert!(
      res.rejects_null,
      "expected rejection for non-normal sample, got {res:?}"
    );
  }

  #[test]
  fn a_constant_sample_is_degenerate_data() {
    let x = [2.5; 10];
    assert_eq!(
      anderson_darling_normal_test(ArrayView1::from(&x), AndersonDarlingConfig::default())
        .unwrap_err(),
      StatsError::DegenerateData("constant sample")
    );
  }

  /// nortest's `ad.test` on the same formula, mpmath at 40 digits; within `1e-12` for libm's `exp`.
  #[test]
  fn the_p_value_is_stephens_table_on_a_grid_of_the_adjusted_statistic() {
    for (a, want) in [
      (0.0, 0.999998538430939),
      (0.05, 0.9998687392821478),
      (0.1, 0.9961485285157409),
      (0.15, 0.9630771618746098),
      (0.199, 0.8857163462249079),
      (0.2, 0.8842497006682849),
      (0.25, 0.7446512446013425),
      (0.3, 0.5825623136156666),
      (0.339, 0.5025652555891493),
      (0.34, 0.498232720934432),
      (0.4, 0.36251116687118035),
      (0.5, 0.20871199326901024),
      (0.599, 0.11758822453925745),
      (0.6, 0.11943249053580202),
      (0.75, 0.0509155195066488),
      (1.0, 0.012317922048460174),
      (1.5, 0.0007260076543596115),
      (2.0, 4.319006785118328e-05),
      (3.0, 1.5717594347363418e-07),
      (5.0, 2.3273286691614228e-12),
      (7.5, 2.6352595389362055e-18),
      (9.99, 3.97138178913265e-24),
      (10.0, 3.7e-24),
      (20.0, 3.7e-24),
      (100.0, 3.7e-24),
      (153.5, 3.7e-24),
      (306.7, 3.7e-24),
      (463.76, 3.7e-24),
      (1000.0, 3.7e-24),
      (10000.0, 3.7e-24),
    ] {
      let got = ad_pvalue_from_adjusted_statistic(a);
      assert!(
        ((got - want) / want).abs() < 1e-12,
        "p({a}) = {got}, want {want}"
      );
    }
  }

  /// Across `[0, 1e4]` the p-value never rises but at 0.6, where Table 4.9's last two formulas
  /// meet (`0.1176` below, `0.1194` at it).
  #[test]
  fn the_p_value_falls_with_the_statistic_on_each_side_of_0_6() {
    let grid = (0..=200_000)
      .map(|k| 1e-4 * k as f64)
      .chain((0..=4_000).map(|k| 20.0 * 500.0_f64.powf(k as f64 / 4_000.0)))
      .collect::<Vec<_>>();
    for pair in grid.windows(2) {
      let (a, b) = (pair[0], pair[1]);
      let (pa, pb) = (
        ad_pvalue_from_adjusted_statistic(a),
        ad_pvalue_from_adjusted_statistic(b),
      );
      assert!(
        pb <= pa || (a < 0.6 && b >= 0.6),
        "p({b}) = {pb} above p({a}) = {pa}"
      );
    }
  }

  /// R's `pnorm(log.p = TRUE)`, `mean` and `sd` in nortest's `ad.test`, retyped, on two rational
  /// samples, the second with a point 9.9 deviations out, past where `Φ` rounds to zero.
  #[test]
  fn the_statistic_is_nortests_on_two_pinned_samples() {
    let small = (0..40)
      .map(|i| ((i * 37) % 101) as f64 / 101.0 + ((i * 11) % 7) as f64 / 7.0)
      .collect::<Vec<f64>>();
    let mut outlier = (0..99)
      .map(|j| ((j * 37) % 101) as f64 / 1e4)
      .collect::<Vec<f64>>();
    outlier.push(-10.0);
    for (x, a, aa, p) in [
      (
        small,
        0.13292790444771185,
        0.13560723252173607,
        0.9783970933167602,
      ),
      (outlier, 37.97305836882143, 38.26640024472057, 3.7e-24),
    ] {
      let res =
        anderson_darling_normal_test(ArrayView1::from(&x), AndersonDarlingConfig::default())
          .unwrap();
      let close = |got: f64, want: f64| ((got - want) / want).abs() < 1e-10;
      assert!(
        close(res.statistic, a)
          && close(res.adjusted_statistic, aa)
          && close(res.p_value.unwrap(), p),
        "{res:?} against A {a}, A* {aa}, p {p}"
      );
    }
  }

  /// 10,000 standard-exponential draws, `A²* ≈ 464`, past the 306.7 where the last formula's
  /// exponent turns positive and the p-value read 1.
  #[test]
  fn ten_thousand_exponential_draws_reject_normality() {
    let mut dist = SimdExp::<f64>::new(1.0).seeded(&Deterministic::new(42));
    let mut x = vec![0.0; 10_000];
    dist.fill_slice(&mut x);
    let res =
      anderson_darling_normal_test(ArrayView1::from(&x), AndersonDarlingConfig::default()).unwrap();
    assert!(
      res.adjusted_statistic > 306.7 && res.p_value == Some(3.7e-24) && res.rejects_null,
      "{res:?}"
    );
  }
}

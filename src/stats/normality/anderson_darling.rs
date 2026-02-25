use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

/// Configuration for the Anderson-Darling normality test.
#[derive(Debug, Clone, Copy)]
pub struct AndersonDarlingConfig {
  /// Significance level used to compute `reject_normality`.
  pub alpha: f64,
}

impl Default for AndersonDarlingConfig {
  fn default() -> Self {
    Self { alpha: 0.05 }
  }
}

/// Result of the Anderson-Darling normality test.
#[derive(Debug, Clone, Copy)]
pub struct AndersonDarlingResult {
  /// Raw A^2 statistic.
  pub statistic: f64,
  /// Stephens finite-sample adjusted statistic.
  pub adjusted_statistic: f64,
  /// Approximate p-value for normality (Stephens-style approximation).
  pub p_value: f64,
  /// Whether normality is rejected at `alpha`.
  pub reject_normality: bool,
}

fn mean_std(sample: &[f64]) -> (f64, f64) {
  let n = sample.len() as f64;
  let mean = sample.iter().sum::<f64>() / n;
  let var = sample
    .iter()
    .map(|&x| {
      let d = x - mean;
      d * d
    })
    .sum::<f64>()
    / n;
  (mean, var.sqrt())
}

fn ad_pvalue_from_adjusted_statistic(a2_star: f64) -> f64 {
  // Piecewise approximation widely used for Anderson-Darling normality testing.
  let p = if a2_star < 0.2 {
    1.0 - (-13.436 + 101.14 * a2_star - 223.73 * a2_star * a2_star).exp()
  } else if a2_star < 0.34 {
    1.0 - (-8.318 + 42.796 * a2_star - 59.938 * a2_star * a2_star).exp()
  } else if a2_star < 0.6 {
    (0.9177 - 4.279 * a2_star - 1.38 * a2_star * a2_star).exp()
  } else {
    (1.2937 - 5.709 * a2_star + 0.0186 * a2_star * a2_star).exp()
  };
  p.clamp(0.0, 1.0)
}

/// Anderson-Darling test for normality with estimated mean and variance.
///
/// # Panics
/// Panics if the sample has fewer than 8 points or contains non-finite values.
pub fn anderson_darling_normal_test(
  sample: &[f64],
  cfg: AndersonDarlingConfig,
) -> AndersonDarlingResult {
  assert!(
    sample.len() >= 8,
    "Anderson-Darling requires at least 8 observations"
  );
  assert!(
    sample.iter().all(|x| x.is_finite()),
    "Anderson-Darling requires finite observations"
  );
  assert!(
    cfg.alpha > 0.0 && cfg.alpha < 1.0,
    "alpha must be in (0, 1)"
  );

  let mut sorted = sample.to_vec();
  sorted.sort_by(f64::total_cmp);
  let n = sorted.len();
  let n_f = n as f64;

  let (mean, std) = mean_std(&sorted);
  assert!(std > 0.0, "Anderson-Darling requires non-constant sample");

  let normal = Normal::new(mean, std).expect("normal params must be valid");
  let eps = 1e-15;

  let mut sum = 0.0;
  for i in 0..n {
    let f_i = normal.cdf(sorted[i]).clamp(eps, 1.0 - eps);
    let f_j = normal.cdf(sorted[n - 1 - i]).clamp(eps, 1.0 - eps);
    let k = (2 * (i + 1) - 1) as f64;
    sum += k * (f_i.ln() + (1.0 - f_j).ln());
  }

  let statistic = -n_f - sum / n_f;
  let adjusted_statistic = statistic * (1.0 + 0.75 / n_f + 2.25 / (n_f * n_f));
  let p_value = ad_pvalue_from_adjusted_statistic(adjusted_statistic);

  AndersonDarlingResult {
    statistic,
    adjusted_statistic,
    p_value,
    reject_normality: p_value < cfg.alpha,
  }
}

#[cfg(test)]
mod tests {
  use super::AndersonDarlingConfig;
  use super::anderson_darling_normal_test;
  use crate::distributions::normal::SimdNormal;
  use crate::distributions::uniform::SimdUniform;

  #[test]
  fn anderson_darling_accepts_normal_sample() {
    let dist = SimdNormal::<f64>::new(0.0, 1.0);
    let mut rng = rand::rng();
    let mut x = vec![0.0; 4000];
    dist.fill_slice(&mut rng, &mut x);

    let res = anderson_darling_normal_test(&x, AndersonDarlingConfig::default());
    assert!(
      res.p_value > 0.01,
      "p-value too small for normal sample: {res:?}"
    );
  }

  #[test]
  fn anderson_darling_rejects_uniform_sample() {
    let dist = SimdUniform::<f64>::new(0.0, 1.0);
    let mut rng = rand::rng();
    let mut x = vec![0.0; 4000];
    dist.fill_slice(&mut rng, &mut x);

    let res = anderson_darling_normal_test(&x, AndersonDarlingConfig::default());
    assert!(
      res.reject_normality,
      "expected rejection for non-normal sample, got {res:?}"
    );
  }
}

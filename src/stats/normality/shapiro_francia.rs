use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::Distribution;
use rand_distr::StandardNormal;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Normal;

/// Configuration for the Shapiro-Francia normality test.
#[derive(Debug, Clone, Copy)]
pub struct ShapiroFranciaConfig {
  /// Significance level used to compute `reject_normality`.
  pub alpha: f64,
  /// Number of parametric bootstrap samples used for p-value approximation.
  pub bootstrap_samples: usize,
  /// Seed for bootstrap reproducibility.
  pub bootstrap_seed: u64,
}

impl Default for ShapiroFranciaConfig {
  fn default() -> Self {
    Self {
      alpha: 0.05,
      bootstrap_samples: 512,
      bootstrap_seed: 42,
    }
  }
}

/// Result of the Shapiro-Francia normality test.
#[derive(Debug, Clone, Copy)]
pub struct ShapiroFranciaResult {
  /// Shapiro-Francia W statistic.
  pub statistic: f64,
  /// Bootstrap p-value (left tail, small W indicates non-normality).
  pub p_value: f64,
  /// Whether normality is rejected at `alpha`.
  pub reject_normality: bool,
}

fn shapiro_francia_statistic_sorted(sorted: &[f64]) -> f64 {
  let n = sorted.len();
  let n_f = n as f64;
  let mean = sorted.iter().sum::<f64>() / n_f;
  let s2 = sorted
    .iter()
    .map(|&x| {
      let d = x - mean;
      d * d
    })
    .sum::<f64>();

  if s2 <= 0.0 {
    return 1.0;
  }

  let std_normal = Normal::new(0.0, 1.0).expect("standard normal must be valid");
  let mut m = Vec::with_capacity(n);
  for i in 0..n {
    let p = (i as f64 + 1.0 - 0.375) / (n_f + 0.25);
    m.push(std_normal.inverse_cdf(p));
  }
  let m_norm = m.iter().map(|v| v * v).sum::<f64>().sqrt();
  if m_norm <= 0.0 {
    return 1.0;
  }

  let num = m
    .iter()
    .zip(sorted.iter())
    .map(|(mi, xi)| (mi / m_norm) * xi)
    .sum::<f64>();
  (num * num / s2).clamp(0.0, 1.0)
}

/// Shapiro-Francia normality test using bootstrap p-values.
///
/// # Panics
/// Panics if the sample has fewer than 8 points or contains non-finite values.
pub fn shapiro_francia_test(sample: &[f64], cfg: ShapiroFranciaConfig) -> ShapiroFranciaResult {
  assert!(
    sample.len() >= 8,
    "Shapiro-Francia requires at least 8 observations"
  );
  assert!(
    sample.iter().all(|x| x.is_finite()),
    "Shapiro-Francia requires finite observations"
  );
  assert!(
    cfg.alpha > 0.0 && cfg.alpha < 1.0,
    "alpha must be in (0, 1)"
  );
  assert!(
    cfg.bootstrap_samples > 0,
    "bootstrap_samples must be positive"
  );

  let mut obs = sample.to_vec();
  obs.sort_by(f64::total_cmp);
  let obs_stat = shapiro_francia_statistic_sorted(&obs);

  let n = sample.len();
  let mut rng = StdRng::seed_from_u64(cfg.bootstrap_seed);
  let mut normal_draw = vec![0.0; n];

  let mut left_tail_hits = 0usize;
  for _ in 0..cfg.bootstrap_samples {
    for v in &mut normal_draw {
      *v = StandardNormal.sample(&mut rng);
    }
    normal_draw.sort_by(f64::total_cmp);
    let w = shapiro_francia_statistic_sorted(&normal_draw);
    if w <= obs_stat {
      left_tail_hits += 1;
    }
  }

  let p_value = (left_tail_hits as f64 + 1.0) / (cfg.bootstrap_samples as f64 + 1.0);

  ShapiroFranciaResult {
    statistic: obs_stat,
    p_value,
    reject_normality: p_value < cfg.alpha,
  }
}

#[cfg(test)]
mod tests {
  use super::ShapiroFranciaConfig;
  use super::shapiro_francia_test;
  use crate::distributions::exp::SimdExp;
  use crate::distributions::normal::SimdNormal;

  #[test]
  fn shapiro_francia_accepts_normal_sample() {
    let dist = SimdNormal::<f64>::new(0.0, 1.0);
    let mut rng = rand::rng();
    let mut x = vec![0.0; 700];
    dist.fill_slice(&mut rng, &mut x);

    let cfg = ShapiroFranciaConfig {
      bootstrap_samples: 256,
      bootstrap_seed: 7,
      ..ShapiroFranciaConfig::default()
    };
    let res = shapiro_francia_test(&x, cfg);
    assert!(
      res.p_value > 0.01,
      "p-value too small for normal sample: {res:?}"
    );
  }

  #[test]
  fn shapiro_francia_rejects_skewed_sample() {
    let dist = SimdExp::<f64>::new(1.0);
    let mut rng = rand::rng();
    let mut x = vec![0.0; 700];
    dist.fill_slice(&mut rng, &mut x);

    let cfg = ShapiroFranciaConfig {
      bootstrap_samples: 256,
      bootstrap_seed: 11,
      ..ShapiroFranciaConfig::default()
    };
    let res = shapiro_francia_test(&x, cfg);
    assert!(
      res.reject_normality,
      "expected rejection for non-normal sample, got {res:?}"
    );
  }
}

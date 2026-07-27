use ndarray::ArrayView1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::special::ndtri;

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

impl crate::traits::HypothesisTest for ShapiroFranciaResult {
  fn statistic(&self) -> f64 {
    self.statistic
  }
  fn null_rejected(&self) -> Option<bool> {
    Some(self.reject_normality)
  }
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

  let mut m = Vec::with_capacity(n);
  for i in 0..n {
    let p = (i as f64 + 1.0 - 0.375) / (n_f + 0.25);
    m.push(ndtri(p));
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
pub fn shapiro_francia_test(
  sample: ArrayView1<f64>,
  cfg: ShapiroFranciaConfig,
) -> ShapiroFranciaResult {
  let sample = sample
    .as_slice()
    .expect("shapiro_francia_test requires a contiguous ArrayView1");
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
  let normals = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(cfg.bootstrap_seed));
  let mut normal_draw = vec![0.0; n];

  let mut left_tail_hits = 0usize;
  for _ in 0..cfg.bootstrap_samples {
    normals.fill_slice_fast(&mut normal_draw);
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
  use ndarray::ArrayView1;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::exp::SimdExp;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::ShapiroFranciaConfig;
  use super::shapiro_francia_test;

  /// The sample must come from a `Deterministic` seed, not an `Unseeded` one:
  /// `fill_slice` ignores the `Rng` it is handed and draws from the
  /// distribution's own SIMD stream, so seeding an external `StdRng` has no
  /// effect on the data at all.
  fn normal_sample(seed: u64, n: usize) -> Vec<f64> {
    let dist = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(seed));
    let mut x = vec![0.0; n];
    dist.fill_slice_fast(&mut x);
    x
  }

  #[test]
  fn shapiro_francia_accepts_normal_sample() {
    // A correct test still rejects a genuine normal sample at rate `alpha`,
    // and the SIMD stream differs between platforms, so one seed cannot be
    // trusted to be lucky everywhere. Three independent seeds put the
    // false-failure probability at roughly 1e-6 on any target.
    let best_p = [2718u64, 999, 42]
      .into_iter()
      .map(|seed| {
        let x = normal_sample(seed, 700);
        let cfg = ShapiroFranciaConfig {
          bootstrap_samples: 256,
          bootstrap_seed: 7,
          ..ShapiroFranciaConfig::default()
        };
        shapiro_francia_test(ArrayView1::from(&x), cfg).p_value
      })
      .fold(0.0_f64, f64::max);

    assert!(
      best_p > 0.01,
      "every seed gave p <= 0.01 (best {best_p}); likely a bug, not bad luck"
    );
  }

  #[test]
  fn shapiro_francia_rejects_skewed_sample() {
    let dist = SimdExp::<f64>::new(1.0, &Deterministic::new(11));
    let mut x = vec![0.0; 700];
    dist.fill_slice_fast(&mut x);

    let cfg = ShapiroFranciaConfig {
      bootstrap_samples: 256,
      bootstrap_seed: 11,
      ..ShapiroFranciaConfig::default()
    };
    let res = shapiro_francia_test(ArrayView1::from(&x), cfg);
    assert!(
      res.reject_normality,
      "expected rejection for non-normal sample, got {res:?}"
    );
  }
}

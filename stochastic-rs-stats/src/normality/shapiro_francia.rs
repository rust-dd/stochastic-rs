//! Shapiro-Francia test for normality: squared correlation between the
//! ordered sample and Blom-type expected normal order-statistic
//! approximations. The p-value is obtained here by parametric bootstrap
//! under the null rather than Shapiro & Francia's original
//! normal-theory approximation formula for the mean/variance of `W`.
//!
//! Reference: Shapiro S. S. & Francia R. S. (1972) — *An Approximate
//! Analysis of Variance Test for Normality*, Journal of the American
//! Statistical Association 67(337), 215–216,
//! DOI: 10.1080/01621459.1972.10481232.

use ndarray::ArrayView1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::special::ndtri;

use crate::StatsError;
use crate::critical_values::CriticalValues;
use crate::input::as_slice_or_copy;
use crate::input::assert_alpha;
use crate::input::validate;

/// Configuration for the Shapiro-Francia normality test.
#[derive(Debug, Clone, Copy)]
pub struct ShapiroFranciaConfig {
  /// Significance level used to compute `rejects_null`.
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

/// Shapiro–Francia test (null: normality), decided by its parametric-bootstrap p-value.
#[derive(Debug, Clone)]
pub struct ShapiroFranciaTest {
  /// Shapiro-Francia W statistic.
  pub statistic: f64,
  /// Bootstrap p-value (left tail, small W indicates non-normality).
  pub p_value: Option<f64>,
  pub critical_values: Option<CriticalValues>,
  /// Whether normality is rejected at `alpha`.
  pub rejects_null: bool,
}

impl crate::traits::HypothesisTest for ShapiroFranciaTest {
  fn statistic(&self) -> f64 {
    self.statistic
  }

  fn null_rejected(&self) -> Option<bool> {
    Some(self.rejects_null)
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

/// Shapiro–Francia normality test with a bootstrap p-value; `Err` below 8 points, on a non-finite
/// or constant sample, and panics unless `0 < alpha < 1` and `bootstrap_samples >= 1`.
pub fn shapiro_francia_test(
  sample: ArrayView1<f64>,
  cfg: ShapiroFranciaConfig,
) -> Result<ShapiroFranciaTest, StatsError> {
  assert_alpha(cfg.alpha);
  assert!(
    cfg.bootstrap_samples >= 1,
    "bootstrap_samples must satisfy `bootstrap_samples >= 1`, got bootstrap_samples = {:?}",
    cfg.bootstrap_samples
  );
  let sample = as_slice_or_copy(sample);
  validate(&sample, 8)?;

  let mut obs = sample.to_vec();
  obs.sort_by(f64::total_cmp);
  if obs[0] == obs[obs.len() - 1] {
    return Err(StatsError::DegenerateData("constant sample"));
  }
  let obs_stat = shapiro_francia_statistic_sorted(&obs);

  let n = sample.len();
  let mut normals =
    SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(cfg.bootstrap_seed));
  let mut normal_draw = vec![0.0; n];

  let mut left_tail_hits = 0usize;
  for _ in 0..cfg.bootstrap_samples {
    normals.fill_slice(&mut normal_draw);
    normal_draw.sort_by(f64::total_cmp);
    let w = shapiro_francia_statistic_sorted(&normal_draw);
    if w <= obs_stat {
      left_tail_hits += 1;
    }
  }

  let p_value = (left_tail_hits as f64 + 1.0) / (cfg.bootstrap_samples as f64 + 1.0);

  Ok(ShapiroFranciaTest {
    statistic: obs_stat,
    p_value: Some(p_value),
    critical_values: None,
    rejects_null: p_value < cfg.alpha,
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

  use super::ShapiroFranciaConfig;
  use super::shapiro_francia_test;

  /// The sample must come from a `Deterministic` seed, not an `Unseeded` one:
  /// `fill_slice` ignores the `Rng` it is handed and draws from the
  /// distribution's own SIMD stream, so seeding an external `StdRng` has no
  /// effect on the data at all.
  fn normal_sample(seed: u64, n: usize) -> Vec<f64> {
    let mut dist = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(seed));
    let mut x = vec![0.0; n];
    dist.fill_slice(&mut x);
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
        shapiro_francia_test(ArrayView1::from(&x), cfg)
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

  #[test]
  fn shapiro_francia_rejects_skewed_sample() {
    let mut dist = SimdExp::<f64>::new(1.0).seeded(&Deterministic::new(11));
    let mut x = vec![0.0; 700];
    dist.fill_slice(&mut x);

    let cfg = ShapiroFranciaConfig {
      bootstrap_samples: 256,
      bootstrap_seed: 11,
      ..ShapiroFranciaConfig::default()
    };
    let res = shapiro_francia_test(ArrayView1::from(&x), cfg).unwrap();
    assert!(
      res.rejects_null,
      "expected rejection for non-normal sample, got {res:?}"
    );
  }
}

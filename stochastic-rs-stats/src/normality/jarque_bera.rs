//! Jarque-Bera test for normality, comparing sample skewness and excess
//! kurtosis against their values under normality via the asymptotic
//! chi-square(2) statistic.
//!
//! References:
//! - Jarque C. M. & Bera A. K. (1980) — *Efficient Tests for Normality,
//!   Homoscedasticity and Serial Independence of Regression Residuals*,
//!   Economics Letters 6(3), 255–259, DOI: 10.1016/0165-1765(80)90024-5.
//! - Jarque C. M. & Bera A. K. (1987) — *A Test for Normality of
//!   Observations and Regression Residuals*, International Statistical
//!   Review 55(2), 163–172, DOI: 10.2307/1403192.

use ndarray::ArrayView1;
use stochastic_rs_distributions::special::gamma_p;

/// Configuration for the Jarque-Bera normality test.
#[derive(Debug, Clone, Copy)]
pub struct JarqueBeraConfig {
  /// Significance level used to compute `reject_normality`.
  pub alpha: f64,
}

impl Default for JarqueBeraConfig {
  fn default() -> Self {
    Self { alpha: 0.05 }
  }
}

/// Result of the Jarque-Bera normality test.
#[derive(Debug, Clone, Copy)]
pub struct JarqueBeraResult {
  /// JB test statistic.
  pub statistic: f64,
  /// p-value under chi-square(2) asymptotics.
  pub p_value: f64,
  /// Sample skewness.
  pub skewness: f64,
  /// Sample excess kurtosis.
  pub excess_kurtosis: f64,
  /// Whether normality is rejected at `alpha`.
  pub reject_normality: bool,
}

impl crate::traits::HypothesisTest for JarqueBeraResult {
  fn statistic(&self) -> f64 {
    self.statistic
  }

  fn null_rejected(&self) -> Option<bool> {
    Some(self.reject_normality)
  }
}

/// Jarque-Bera test for normality.
///
/// # Panics
/// Panics if the sample has fewer than 8 points or contains non-finite values.
pub fn jarque_bera_test(sample: ArrayView1<f64>, cfg: JarqueBeraConfig) -> JarqueBeraResult {
  let sample = sample
    .as_slice()
    .expect("jarque_bera_test requires a contiguous ArrayView1");
  assert!(
    sample.len() >= 8,
    "Jarque-Bera requires at least 8 observations"
  );
  assert!(
    sample.iter().all(|x| x.is_finite()),
    "Jarque-Bera requires finite observations"
  );
  assert!(
    cfg.alpha > 0.0 && cfg.alpha < 1.0,
    "alpha must be in (0, 1)"
  );

  let n = sample.len() as f64;
  let mean = sample.iter().sum::<f64>() / n;

  let mut m2 = 0.0;
  let mut m3 = 0.0;
  let mut m4 = 0.0;
  for &x in sample {
    let d = x - mean;
    let d2 = d * d;
    m2 += d2;
    m3 += d2 * d;
    m4 += d2 * d2;
  }
  m2 /= n;
  m3 /= n;
  m4 /= n;

  if m2 <= 0.0 || !m2.is_finite() {
    return JarqueBeraResult {
      statistic: f64::INFINITY,
      p_value: 0.0,
      skewness: 0.0,
      excess_kurtosis: f64::INFINITY,
      reject_normality: true,
    };
  }

  let skewness = m3 / m2.powf(1.5);
  let kurtosis = m4 / (m2 * m2);
  let excess_kurtosis = kurtosis - 3.0;
  let statistic = (n / 6.0) * (skewness * skewness + 0.25 * excess_kurtosis * excess_kurtosis);

  // χ²(df=2) survival function: P(X > t) = 1 − P(1, t/2) = exp(−t/2).
  // Use the regularised lower incomplete gamma to make the dependency
  // explicit and uniform with other CDF use sites.
  let p_value = (1.0 - gamma_p(1.0, 0.5 * statistic)).clamp(0.0, 1.0);

  JarqueBeraResult {
    statistic,
    p_value,
    skewness,
    excess_kurtosis,
    reject_normality: p_value < cfg.alpha,
  }
}

#[cfg(test)]
mod tests {
  use ndarray::ArrayView1;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::normal::SimdNormal;
  use stochastic_rs_distributions::uniform::SimdUniform;

  use super::JarqueBeraConfig;
  use super::jarque_bera_test;

  /// The sample must come from a `Deterministic` seed, not an `Unseeded` one:
  /// `fill_slice` takes no RNG at all and draws from the distribution's own
  /// SIMD stream, so only the constructor's seed controls the data.
  fn normal_sample(seed: u64, n: usize) -> Vec<f64> {
    let dist = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(seed));
    let mut x = vec![0.0; n];
    dist.fill_slice(&mut x);
    x
  }

  #[test]
  fn jarque_bera_accepts_normal_sample() {
    // A correct test still rejects a genuine normal sample at rate `alpha`,
    // and the SIMD stream differs between platforms, so one seed cannot be
    // trusted to be lucky everywhere. Three independent seeds put the
    // false-failure probability at roughly 1e-6 on any target.
    let best_p = [1u64, 999, 7]
      .into_iter()
      .map(|seed| {
        let x = normal_sample(seed, 5000);
        jarque_bera_test(ArrayView1::from(&x), JarqueBeraConfig::default()).p_value
      })
      .fold(0.0_f64, f64::max);

    assert!(
      best_p > 0.01,
      "every seed gave p <= 0.01 (best {best_p}); likely a bug, not bad luck"
    );
  }

  #[test]
  fn jarque_bera_rejects_heavy_tail_sample() {
    let mut x = normal_sample(42, 5000);

    // The bimodal +/-2 shift is what makes the sample non-normal, so the
    // coin flips genuinely drive the data and get their own seed, distinct
    // from the one behind `normal_sample`.
    let mut coin = vec![0.0_f64; x.len()];
    SimdUniform::<f64>::new(0.0, 1.0, &Deterministic::new(43)).fill_slice(&mut coin);
    for (v, u) in x.iter_mut().zip(&coin) {
      *v += if *u < 0.5 { -2.0 } else { 2.0 };
    }

    let res = jarque_bera_test(ArrayView1::from(&x), JarqueBeraConfig::default());
    assert!(
      res.reject_normality,
      "expected rejection for non-normal sample, got {res:?}"
    );
  }
}

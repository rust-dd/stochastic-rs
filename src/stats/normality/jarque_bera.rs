use statrs::distribution::ChiSquared;
use statrs::distribution::ContinuousCDF;

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

/// Jarque-Bera test for normality.
///
/// # Panics
/// Panics if the sample has fewer than 8 points or contains non-finite values.
pub fn jarque_bera_test(sample: &[f64], cfg: JarqueBeraConfig) -> JarqueBeraResult {
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

  let chi2 = ChiSquared::new(2.0).expect("chi-square df=2 must be valid");
  let p_value = (1.0 - chi2.cdf(statistic)).clamp(0.0, 1.0);

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
  use rand::Rng;

  use super::JarqueBeraConfig;
  use super::jarque_bera_test;
  use crate::distributions::normal::SimdNormal;

  #[test]
  fn jarque_bera_accepts_normal_sample() {
    let dist = SimdNormal::<f64>::new(0.0, 1.0);
    let mut rng = rand::rng();
    let mut x = vec![0.0; 5000];
    dist.fill_slice(&mut rng, &mut x);

    let res = jarque_bera_test(&x, JarqueBeraConfig::default());
    assert!(
      res.p_value > 0.01,
      "p-value too small for normal sample: {res:?}"
    );
  }

  #[test]
  fn jarque_bera_rejects_heavy_tail_sample() {
    let dist = SimdNormal::<f64>::new(0.0, 1.0);
    let mut rng = rand::rng();
    let mut x = vec![0.0; 5000];
    dist.fill_slice(&mut rng, &mut x);

    for v in &mut x {
      let u: f64 = rng.random();
      *v += if u < 0.5 { -2.0 } else { 2.0 };
    }

    let res = jarque_bera_test(&x, JarqueBeraConfig::default());
    assert!(
      res.reject_normality,
      "expected rejection for non-normal sample, got {res:?}"
    );
  }
}

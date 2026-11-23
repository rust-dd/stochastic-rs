//! Bipower variation, MinRV, MedRV and the Barndorff-Nielsen–Shephard jump test.
//!
//! Reference: Barndorff-Nielsen & Shephard (2004), JFEC 2(1), 1-37, DOI: 10.1093/jjfinec/nbh001
//! Reference: Barndorff-Nielsen & Shephard (2006), JFEC 4(1), 1-30, DOI: 10.1093/jjfinec/nbi022
//! Reference: Andersen et al. (2012), J. Econom. 169(1), 75-93, DOI: 10.1016/j.jeconom.2012.01.011
//! Reference: Huang & Tauchen (2005), JFEC 3(4), 456-499, DOI: 10.1093/jjfinec/nbi025

use ndarray::ArrayView1;

use crate::StatsError;
use crate::critical_values::CriticalValues;
use crate::critical_values::two_sided_p_value;
use crate::evt::gpd::params_in_data_units;
use crate::input::as_slice_or_copy;
use crate::input::assert_alpha;
use crate::input::reject_constant;
use crate::input::validate_view;
use crate::realized::variance::realized_variance;
use crate::stationarity::common::unit_scaled;
use crate::traits::FloatExt;

/// Bipower variation $BV = \frac{\pi}{2}\sum_{i=2}^{n} |r_{i-1}|\,|r_i|$, robust to finite-activity
/// jumps (BN-Shephard 2004); NaN below two returns.
pub fn bipower_variation<T: FloatExt>(returns: ArrayView1<T>) -> T {
  let n = returns.len();
  if n < 2 {
    return T::nan();
  }
  let mu1_inv2 = T::from_f64_fast(std::f64::consts::FRAC_PI_2);
  let mut sum = T::zero();
  for i in 1..n {
    sum += returns[i - 1].abs() * returns[i].abs();
  }
  mu1_inv2 * sum
}

/// MinRV $= \frac{\pi}{\pi - 2}\frac{n}{n - 1}\sum_{i=2}^{n} \min(|r_{i-1}|, |r_i|)^2$, the
/// jump-robust nearest-neighbour truncation of Andersen, Dobrev, Schaumburg (2012); NaN below two.
pub fn minrv<T: FloatExt>(returns: ArrayView1<T>) -> T {
  let n = returns.len();
  if n < 2 {
    return T::nan();
  }
  let pi = T::from_f64_fast(std::f64::consts::PI);
  let scale = pi / (pi - T::from_f64_fast(2.0));
  let mut sum = T::zero();
  for i in 1..n {
    let m = returns[i - 1].abs().min_or_nan(returns[i].abs());
    sum += m * m;
  }
  let nn = T::from_usize_(n);
  scale * (nn / (nn - T::one())) * sum
}

/// MedRV $= \frac{\pi}{6 - 4\sqrt 3 + \pi}\frac{n}{n - 2}\sum_{i=3}^{n} \mathrm{med}(|r_{i-2}|,
/// |r_{i-1}|, |r_i|)^2$ (Andersen, Dobrev, Schaumburg 2012); NaN below three returns.
pub fn medrv<T: FloatExt>(returns: ArrayView1<T>) -> T {
  let n = returns.len();
  if n < 3 {
    return T::nan();
  }
  let pi = T::from_f64_fast(std::f64::consts::PI);
  let denom = T::from_f64_fast(6.0) - T::from_f64_fast(4.0) * T::from_f64_fast(3.0).sqrt() + pi;
  let scale = pi / denom;
  let mut sum = T::zero();
  for i in 2..n {
    let (a, b, c) = (returns[i - 2].abs(), returns[i - 1].abs(), returns[i].abs());
    let m = a
      .max_or_nan(b)
      .min_or_nan(a.max_or_nan(c))
      .min_or_nan(b.max_or_nan(c));
    sum += m * m;
  }
  let nn = T::from_usize_(n);
  scale * (nn / (nn - T::from_f64_fast(2.0))) * sum
}

/// Tripower quarticity $TPQ = n\,\mu_{4/3}^{-3}\sum_{i=3}^n |r_{i-2} r_{i-1} r_i|^{4/3}$, the
/// jump-robust integrated quarticity of the BNS test; NaN below three returns.
pub fn tripower_quarticity<T: FloatExt>(returns: ArrayView1<T>) -> T {
  let n = returns.len();
  if n < 3 {
    return T::nan();
  }
  let p43 = T::from_f64_fast(4.0 / 3.0);
  let mu_43 = mu_p::<T>(4.0 / 3.0);
  let mu_inv3 = T::one() / (mu_43.powi(3));
  let mut sum = T::zero();
  for i in 2..n {
    sum +=
      returns[i - 2].abs().powf(p43) * returns[i - 1].abs().powf(p43) * returns[i].abs().powf(p43);
  }
  T::from_usize_(n) * mu_inv3 * sum
}

/// BNS jump test in Huang–Tauchen's ratio form (null: no jump), decided by its normal p-value.
#[derive(Debug, Clone)]
pub struct BnsJumpTest {
  /// Standardised log-difference $\sqrt n\,(\ln RV - \ln BV) / \sqrt{\theta\,\max(1, TPQ / BV^2)}$,
  /// $\theta = (\pi/2)^2 + \pi - 5$.
  pub statistic: f64,
  /// Two-sided p-value `2 Φ(−|statistic|)` under `N(0, 1)`.
  pub p_value: Option<f64>,
  pub critical_values: Option<CriticalValues>,
  /// Whether the no-jump null is rejected at the given $\alpha$.
  pub rejects_null: bool,
  /// Realized variance.
  pub rv: f64,
  /// Bipower variation.
  pub bv: f64,
  /// Tripower quarticity (jump-robust estimate of $\int \sigma^4_t dt$).
  pub tpq: f64,
}

impl crate::traits::HypothesisTest for BnsJumpTest {
  fn statistic(&self) -> f64 {
    self.statistic
  }

  fn null_rejected(&self) -> Option<bool> {
    Some(self.rejects_null)
  }
}

/// BNS jump test, `N(0, 1)` under no jump; `Err` below 3 returns, on a non-finite one, a constant
/// series, a zero bipower variation or measures past the doubles; panics unless `0 < alpha < 1`.
pub fn bns_jump_test<T: FloatExt>(
  returns: ArrayView1<T>,
  alpha: f64,
) -> Result<BnsJumpTest, StatsError> {
  assert_alpha(alpha);
  validate_view(returns, 3)?;
  reject_constant(&as_slice_or_copy(returns), "constant series")?;
  let n = returns.len();
  let (scaled, unit) = unit_scaled(returns.to_vec());
  let scaled = ArrayView1::from(&scaled);
  let rv = realized_variance(scaled).to_f64().unwrap();
  let bv = bipower_variation(scaled).to_f64().unwrap();
  let tpq = tripower_quarticity(scaled).to_f64().unwrap();
  let theta = (std::f64::consts::FRAC_PI_2).powi(2) + std::f64::consts::PI - 5.0;
  let nn = n as f64;
  if !(bv > 0.0 && rv > 0.0) {
    return Err(StatsError::DegenerateData("zero bipower variation"));
  }
  // TPQ alone scales the quarticity: realized quarticity is not jump-robust and would absorb the
  // jump under test.
  let ratio = tpq / (bv * bv);
  let denom = (theta * ratio.max(1.0)).sqrt();
  let statistic = nn.sqrt() * (rv.ln() - bv.ln()) / denom;
  let p_value = two_sided_p_value(statistic);
  let in_units = params_in_data_units([rv, bv, tpq], unit.to_f64().unwrap(), [2, 2, 4])?;
  Ok(BnsJumpTest {
    statistic,
    p_value: Some(p_value),
    critical_values: None,
    rejects_null: p_value < alpha,
    rv: in_units[0],
    bv: in_units[1],
    tpq: in_units[2],
  })
}

fn mu_p<T: FloatExt>(p: f64) -> T {
  // E|Z|^p with Z ~ N(0, 1) = 2^{p/2} \Gamma((p+1)/2) / \sqrt{\pi}
  let g = stochastic_rs_distributions::special::gamma((p + 1.0) / 2.0);
  T::from_f64_fast(2.0_f64.powf(p / 2.0) * g / std::f64::consts::PI.sqrt())
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use ndarray::array;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::DistributionSampler;
  use stochastic_rs_distributions::SimdDistribution;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::*;

  fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
  }

  fn iid_normal(seed: u64, n: usize, std: f64) -> Array1<f64> {
    let mut dist = SimdNormal::<f64>::new(0.0, std).seeded(&Deterministic::new(seed));
    let mut out = Array1::<f64>::zeros(n);
    dist.fill_slice(out.as_slice_mut().unwrap());
    out
  }

  #[test]
  fn bv_constant_returns_equals_pi_over_two_times_n_minus_one_times_r_squared() {
    let r = Array1::<f64>::from_elem(100, 0.01);
    let bv = bipower_variation(r.view());
    let expected = std::f64::consts::FRAC_PI_2 * 99.0 * 0.0001;
    assert!(approx(bv, expected, 1e-12));
  }

  #[test]
  fn bv_close_to_rv_under_normal_returns() {
    let r = iid_normal(7, 5_000, 0.01);
    let rv: f64 = realized_variance(r.view());
    let bv: f64 = bipower_variation(r.view());
    assert!((rv - bv).abs() / rv < 0.1);
  }

  #[test]
  fn bv_robust_to_a_single_jump() {
    let mut r = iid_normal(7, 5_000, 0.01);
    r[2_500] = 1.0;
    let rv: f64 = realized_variance(r.view());
    let bv: f64 = bipower_variation(r.view());
    assert!(rv > 1.5 * bv);
  }

  #[test]
  fn jump_test_does_not_reject_pure_diffusion() {
    let r = iid_normal(11, 5_000, 0.01);
    let test = bns_jump_test(r.view(), 0.05).unwrap();
    assert!(!test.rejects_null);
  }

  #[test]
  fn jump_test_rejects_with_large_jumps() {
    let mut r = iid_normal(13, 5_000, 0.005);
    for &i in &[1_000usize, 2_500, 4_000] {
      r[i] = 0.3;
    }
    let test = bns_jump_test(r.view(), 0.05).unwrap();
    assert!(test.rejects_null);
  }

  #[test]
  fn jump_test_rejects_a_single_moderate_jump() {
    // One jump of about 15 sigma in 256 returns: plain to the eye, yet a denominator on a
    // quarticity that is not jump-robust would swamp it and hide the rejection.
    let mut r = iid_normal(17, 256, 0.005);
    r[128] = 0.077;
    let test = bns_jump_test(r.view(), 0.01).unwrap();
    assert!(
      test.rejects_null,
      "single jump must be detected, statistic was {}",
      test.statistic
    );
    assert!(
      test.statistic > 0.0,
      "jump evidence lives in the upper tail"
    );
  }

  #[test]
  fn jump_test_statistic_grows_with_jump_size() {
    let base = iid_normal(19, 256, 0.005);
    let mut small = base.clone();
    small[128] = 0.02;
    let mut large = base.clone();
    large[128] = 0.10;
    let z_small = bns_jump_test(small.view(), 0.01).unwrap().statistic;
    let z_large = bns_jump_test(large.view(), 0.01).unwrap().statistic;
    assert!(
      z_large > z_small,
      "a bigger jump must not weaken the evidence: {z_small} -> {z_large}"
    );
  }

  #[test]
  fn medrv_is_nan_when_too_short() {
    let r = array![0.01_f64, -0.02];
    assert!(medrv(r.view()).is_nan());
  }

  #[test]
  fn medrv_takes_the_middle_of_three_without_a_sort() {
    let r = array![0.03_f64, -0.01, 0.02, -0.05, 0.04];
    let pi = std::f64::consts::PI;
    let medians = [0.02_f64, 0.02, 0.04];
    let want = pi / (6.0 - 4.0 * 3.0_f64.sqrt() + pi)
      * (5.0 / 3.0)
      * medians.iter().map(|m| m * m).sum::<f64>();
    assert_eq!(medrv(r.view()), want);
  }

  #[test]
  fn a_zero_bipower_variation_is_degenerate_data() {
    let r = Array1::from_iter((0..40).map(|i| if i % 2 == 0 { 0.0 } else { 0.01 }));
    assert_eq!(
      bns_jump_test(r.view(), 0.05).unwrap_err(),
      StatsError::DegenerateData("zero bipower variation")
    );
  }

  #[test]
  fn constant_and_zero_returns_are_a_constant_series() {
    for r in [Array1::from_elem(200, 0.001), Array1::<f64>::zeros(200)] {
      assert_eq!(
        bns_jump_test(r.view(), 0.05).unwrap_err(),
        StatsError::DegenerateData("constant series")
      );
    }
  }

  #[test]
  fn a_zero_tripower_quarticity_keeps_the_statistic_defined() {
    let r = Array1::from_iter((0..90).map(|i| [0.01, -0.012, 0.0][i % 3]));
    let test = bns_jump_test(r.view(), 0.05).unwrap();
    assert_eq!(test.tpq, 0.0);
    assert!(test.bv > 0.0 && test.statistic.is_finite(), "{test:?}");
  }

  #[test]
  fn a_far_tail_statistic_keeps_a_positive_p_value() {
    let mut r = iid_normal(29, 5_000, 0.01);
    r[2_500] = 0.5;
    let test = bns_jump_test(r.view(), 0.05).unwrap();
    assert!(test.statistic > 8.27, "z = {}", test.statistic);
    assert!(test.p_value.unwrap() > 0.0, "p = {:?}", test.p_value);
    assert_eq!(test.p_value, Some(two_sided_p_value(test.statistic)));
  }
}

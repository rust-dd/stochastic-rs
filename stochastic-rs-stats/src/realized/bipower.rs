//! Bipower variation, MinRV, MedRV and the Barndorff-Nielsen / Shephard jump test.
//!
//! Reference: Barndorff-Nielsen, Shephard, "Power and Bipower Variation with
//! Stochastic Volatility and Jumps", Journal of Financial Econometrics, 2(1),
//! 1-37 (2004). DOI: 10.1093/jjfinec/nbh001
//!
//! Reference: Barndorff-Nielsen, Shephard, "Econometrics of Testing for Jumps
//! in Financial Economics Using Bipower Variation", Journal of Financial
//! Econometrics, 4(1), 1-30 (2006). DOI: 10.1093/jjfinec/nbi022
//!
//! Reference: Andersen, Dobrev, Schaumburg, "Jump-Robust Volatility Estimation
//! Using Nearest Neighbor Truncation", Journal of Econometrics, 169(1), 75-93
//! (2012). DOI: 10.1016/j.jeconom.2012.01.011
//!
//! Reference: Huang, Tauchen, "The Relative Contribution of Jumps to Total
//! Price Variance", Journal of Financial Econometrics, 3(4), 456-499 (2005).
//! DOI: 10.1093/jjfinec/nbi025

use ndarray::ArrayView1;
use stochastic_rs_distributions::special::erf;

use crate::realized::variance::realized_quarticity;
use crate::realized::variance::realized_variance;
use crate::traits::FloatExt;

#[inline]
fn standard_normal_cdf(z: f64) -> f64 {
  0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Bipower variation: $BV = \mu_1^{-2}\sum_{i=2}^{n} |r_{i-1}|\,|r_i|$
/// with $\mu_1 = \mathbb E|Z| = \sqrt{2/\pi}$, so $\mu_1^{-2} = \pi/2$.
///
/// Robust to finite-activity jumps; converges to integrated variance under no
/// jumps (BN-Shephard 2004).
pub fn bipower_variation<T: FloatExt>(returns: ArrayView1<T>) -> T {
  let n = returns.len();
  if n < 2 {
    return T::zero();
  }
  let mu1_inv2 = T::from_f64_fast(std::f64::consts::FRAC_PI_2);
  let mut sum = T::zero();
  for i in 1..n {
    sum += returns[i - 1].abs() * returns[i].abs();
  }
  mu1_inv2 * sum
}

/// MinRV (Andersen, Dobrev, Schaumburg 2012): nearest-neighbour-truncation
/// jump-robust estimator,
/// $MinRV = \frac{\pi}{\pi - 2}\sum_{i=2}^{n} \min(|r_{i-1}|, |r_i|)^2$.
pub fn minrv<T: FloatExt>(returns: ArrayView1<T>) -> T {
  let n = returns.len();
  if n < 2 {
    return T::zero();
  }
  let pi = T::from_f64_fast(std::f64::consts::PI);
  let scale = pi / (pi - T::from_f64_fast(2.0));
  let mut sum = T::zero();
  for i in 1..n {
    let m = returns[i - 1].abs().min(returns[i].abs());
    sum += m * m;
  }
  let nn = T::from_usize_(n);
  scale * (nn / (nn - T::one())) * sum
}

/// MedRV (Andersen, Dobrev, Schaumburg 2012):
/// $MedRV = \frac{\pi}{6 - 4\sqrt 3 + \pi}\sum_{i=3}^{n} \mathrm{med}(|r_{i-2}|, |r_{i-1}|, |r_i|)^2$.
pub fn medrv<T: FloatExt>(returns: ArrayView1<T>) -> T {
  let n = returns.len();
  if n < 3 {
    return T::zero();
  }
  let pi = T::from_f64_fast(std::f64::consts::PI);
  let denom = T::from_f64_fast(6.0) - T::from_f64_fast(4.0) * T::from_f64_fast(3.0).sqrt() + pi;
  let scale = pi / denom;
  let mut sum = T::zero();
  for i in 2..n {
    let mut t = [returns[i - 2].abs(), returns[i - 1].abs(), returns[i].abs()];
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = t[1];
    sum += m * m;
  }
  let nn = T::from_usize_(n);
  scale * (nn / (nn - T::from_f64_fast(2.0))) * sum
}

/// Tripower quarticity: $TPQ = n \cdot \mu_{4/3}^{-3}\sum_{i=3}^n
/// |r_{i-2}|^{4/3}|r_{i-1}|^{4/3}|r_i|^{4/3}$, used as the jump-robust
/// estimator of integrated quarticity in the BNS jump test.
pub fn tripower_quarticity<T: FloatExt>(returns: ArrayView1<T>) -> T {
  let n = returns.len();
  if n < 3 {
    return T::zero();
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

/// Result of the BNS (Huang–Tauchen ratio-form) jump test.
#[derive(Debug, Clone, Copy)]
pub struct BnsJumpTest {
  /// Realized variance.
  pub rv: f64,
  /// Bipower variation.
  pub bv: f64,
  /// Tripower quarticity (jump-robust estimate of $\int \sigma^4_t dt$).
  pub tpq: f64,
  /// Standardised log-difference statistic
  /// $\sqrt n\,(\ln RV - \ln BV) / \sqrt{(\theta - 2)\,TPQ / BV^2}$
  /// with $\theta = (\pi/2)^2 + \pi - 5$.
  pub statistic: f64,
  /// Two-sided p-value under $\mathcal N(0,1)$.
  pub p_value: f64,
  /// Whether the no-jump null is rejected at the given $\alpha$.
  pub reject_no_jump: bool,
}

/// Barndorff-Nielsen / Shephard ratio-form jump test
/// (Huang–Tauchen 2005 finite-sample variant).
///
/// Returns `BnsJumpTest::statistic ~ N(0, 1)` under the no-jump null.
pub fn bns_jump_test<T: FloatExt>(returns: ArrayView1<T>, alpha: f64) -> BnsJumpTest {
  assert!(alpha > 0.0 && alpha < 1.0, "alpha must lie in (0, 1)");
  let n = returns.len();
  let rv = realized_variance(returns).to_f64().unwrap();
  let bv = bipower_variation(returns).to_f64().unwrap();
  let tpq = tripower_quarticity(returns).to_f64().unwrap();
  let rq = realized_quarticity(returns).to_f64().unwrap();
  let theta = (std::f64::consts::FRAC_PI_2).powi(2) + std::f64::consts::PI - 5.0;
  let nn = n as f64;
  let ratio = (tpq.max(rq).max(1e-30)) / bv.max(1e-30).powi(2);
  let denom = (theta * ratio.max(1.0)).sqrt();
  let statistic = if bv > 0.0 && rv > 0.0 && denom > 0.0 {
    nn.sqrt() * (rv.ln() - bv.ln()) / denom
  } else {
    0.0
  };
  let cdf = standard_normal_cdf(statistic);
  let p_value = 2.0 * (1.0_f64.min(cdf).min(1.0 - cdf));
  let reject_no_jump = p_value < alpha;
  BnsJumpTest {
    rv,
    bv,
    tpq,
    statistic,
    p_value,
    reject_no_jump,
  }
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
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::*;

  fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
  }

  fn iid_normal(seed: u64, n: usize, std: f64) -> Array1<f64> {
    let dist = SimdNormal::<f64>::new(0.0, std, &stochastic_rs_core::simd_rng::Deterministic::new(seed));
    let mut out = Array1::<f64>::zeros(n);
    dist.fill_slice_fast(out.as_slice_mut().unwrap());
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
    let test = bns_jump_test(r.view(), 0.05);
    assert!(!test.reject_no_jump);
  }

  #[test]
  fn jump_test_rejects_with_large_jumps() {
    let mut r = iid_normal(13, 5_000, 0.005);
    for &i in &[1_000usize, 2_500, 4_000] {
      r[i] = 0.3;
    }
    let test = bns_jump_test(r.view(), 0.05);
    assert!(test.reject_no_jump);
  }

  #[test]
  fn medrv_zero_when_too_short() {
    let r = array![0.01_f64];
    assert_eq!(medrv(r.view()), 0.0);
  }
}

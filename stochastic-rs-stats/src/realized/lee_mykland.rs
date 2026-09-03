//! Lee–Mykland nonparametric jump test on intraday log-returns.
//!
//! Each return is standardised by a bipower-variation estimate of the local
//! volatility from the $K$ observations just before it (Definition 1),
//!
//! $$
//! \mathcal L(i) = \frac{r_i}{\hat\sigma(t_i)},\qquad
//! \hat\sigma(t_i)^2 = \frac{1}{K-2}\sum_{j=i-K+2}^{i-1} |r_j|\,|r_{j-1}|,
//! $$
//!
//! so that under the no-jump null $\mathcal L(i) \approx U_i / c$ with
//! $U_i$ standard normal and $c = \mathbb E|U_i| = \sqrt{2/\pi}$
//! (Theorem 1). A return is declared a jump when its statistic leaves the
//! Gumbel region of the sample maximum (Lemma 1),
//!
//! $$
//! \frac{|\mathcal L(i)| - C_n}{S_n} > \beta^*,\qquad
//! C_n = \frac{\sqrt{2\log n}}{c} - \frac{\log\pi + \log\log n}{2c\sqrt{2\log n}},\qquad
//! S_n = \frac{1}{c\sqrt{2\log n}},
//! $$
//!
//! with $\beta^* = -\log(-\log(1-\alpha))$ and $n$ the number of return
//! observations. The window must grow like $\Delta t^{\gamma}$ with
//! $-1 < \gamma < -1/2$; the paper's rule of thumb picks the smallest
//! integer above $\sqrt{252 \times \text{observations per day}}$, which is
//! $K = 16$ for daily data ([`lee_mykland_window`]).
//!
//! Reference: Lee, Mykland, "Jumps in Financial Markets: A New Nonparametric
//! Test and Jump Dynamics", Review of Financial Studies, 21(6), 2535-2563
//! (2008). DOI: 10.1093/rfs/hhm056

use ndarray::Array1;
use ndarray::ArrayView1;

use crate::traits::FloatExt;

/// $c = \mathbb E|U| = \sqrt{2/\pi}$ for a standard normal $U$: the scale of
/// the statistic under the null (Theorem 1).
pub const LEE_MYKLAND_C: f64 = std::f64::consts::FRAC_2_SQRT_PI / std::f64::consts::SQRT_2;

/// Result of the Lee–Mykland jump test over a return series.
#[derive(Debug, Clone)]
pub struct LeeMyklandTest {
  /// Statistics $\mathcal L(i)$ aligned with the input; the first
  /// `window - 1` entries are NaN because their volatility window is
  /// incomplete.
  pub statistics: Array1<f64>,
  /// Local volatility $\hat\sigma(t_i)$ behind each statistic, NaN where
  /// the statistic is.
  pub local_volatility: Array1<f64>,
  /// Indices of the returns declared jumps, ascending.
  pub jump_indices: Vec<usize>,
  /// Rejection threshold on $|\mathcal L(i)|$: $C_n + \beta^* S_n$.
  pub threshold: f64,
  /// Gumbel centring $C_n$.
  pub c_n: f64,
  /// Gumbel scaling $S_n$.
  pub s_n: f64,
  /// $\beta^* = -\log(-\log(1-\alpha))$.
  pub beta_star: f64,
  /// Window size $K$.
  pub window: usize,
  /// Significance level.
  pub alpha: f64,
  /// Number of return observations $n$.
  pub nobs: usize,
}

/// Lee–Mykland jump test of every return in `returns` at level `alpha`,
/// with the local volatility taken from the `window` observations before it.
///
/// The first `window - 1` statistics and local volatilities are NaN, and so
/// is any statistic where a zero return meets a zero local volatility.
///
/// # Panics
///
/// If `window < 3`, if there are fewer than `window` returns, or if `alpha`
/// is outside `(0, 1)`.
pub fn lee_mykland_test<T: FloatExt>(
  returns: ArrayView1<T>,
  window: usize,
  alpha: f64,
) -> LeeMyklandTest {
  let n = returns.len();
  assert!(window >= 3, "window must be at least 3, got {window}");
  assert!(
    n >= window,
    "need at least window = {window} returns, got {n}"
  );
  assert!(
    alpha > 0.0 && alpha < 1.0,
    "alpha must lie in (0, 1), got {alpha}"
  );

  let r: Vec<f64> = returns
    .iter()
    .map(|x| x.to_f64().unwrap_or(f64::NAN))
    .collect();
  let mut statistics = Array1::<f64>::from_elem(n, f64::NAN);
  let mut local_volatility = Array1::<f64>::from_elem(n, f64::NAN);
  let scale = 1.0 / (window - 2) as f64;
  for i in (window - 1)..n {
    let mut sum = 0.0;
    for j in (i + 2 - window)..i {
      sum += r[j].abs() * r[j - 1].abs();
    }
    let sigma = (scale * sum).sqrt();
    local_volatility[i] = sigma;
    statistics[i] = r[i] / sigma;
  }

  let (c_n, s_n) = gumbel_constants(n);
  let beta_star = -(-(1.0 - alpha).ln()).ln();
  let threshold = c_n + beta_star * s_n;
  let jump_indices = statistics
    .iter()
    .enumerate()
    .filter(|(_, l)| l.abs() > threshold)
    .map(|(i, _)| i)
    .collect();

  LeeMyklandTest {
    statistics,
    local_volatility,
    jump_indices,
    threshold,
    c_n,
    s_n,
    beta_star,
    window,
    alpha,
    nobs: n,
  }
}

/// The paper's window rule (§1.3): the smallest integer above
/// $\sqrt{252 \times \text{observations per day}}$ — 16 for daily returns,
/// 141 for 78 five-minute returns a day.
///
/// # Panics
///
/// If `observations_per_day` is zero.
pub fn lee_mykland_window(observations_per_day: usize) -> usize {
  assert!(
    observations_per_day >= 1,
    "observations_per_day must be at least 1"
  );
  (252.0 * observations_per_day as f64).sqrt().floor() as usize + 1
}

/// Lemma 1's centring and scaling of the maximum of $n$ statistics.
fn gumbel_constants(n: usize) -> (f64, f64) {
  let log_n = (n as f64).ln();
  let root = (2.0 * log_n).sqrt();
  let c_n =
    root / LEE_MYKLAND_C - (std::f64::consts::PI.ln() + log_n.ln()) / (2.0 * LEE_MYKLAND_C * root);
  let s_n = 1.0 / (LEE_MYKLAND_C * root);
  (c_n, s_n)
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use ndarray::array;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::*;

  fn iid_normal(seed: u64, n: usize, std: f64) -> Array1<f64> {
    let dist = SimdNormal::<f64>::new(0.0, std, &Deterministic::new(seed));
    let mut out = Array1::<f64>::zeros(n);
    dist.fill_slice(out.as_slice_mut().unwrap());
    out
  }

  /// Definition 1 by hand with $K = 4$: the window behind $r_i$ holds the
  /// two products $|r_{i-3}||r_{i-2}|$ and $|r_{i-2}||r_{i-1}|$.
  #[test]
  fn statistic_follows_definition_one() {
    let r = array![0.01_f64, -0.02, 0.015, -0.005, 0.03, 0.002];
    let out = lee_mykland_test(r.view(), 4, 0.01);
    let sigma3 = (0.5_f64 * (0.02 * 0.01 + 0.015 * 0.02)).sqrt();
    let sigma4 = (0.5_f64 * (0.015 * 0.02 + 0.005 * 0.015)).sqrt();
    assert!((out.local_volatility[3] - sigma3).abs() < 1e-15);
    assert!((out.statistics[3] - (-0.005 / sigma3)).abs() < 1e-12);
    assert!((out.statistics[4] - 0.03 / sigma4).abs() < 1e-12);
    assert!(out.statistics.iter().take(3).all(|v| v.is_nan()));
    assert!(out.statistics.iter().skip(3).all(|v| v.is_finite()));
    assert_eq!((out.window, out.nobs), (4, 6));
  }

  /// Lemma 1's constants, and the paper's quoted 1% threshold
  /// $\beta^* = -\log(-\log 0.99) = 4.6001$.
  #[test]
  fn threshold_follows_lemma_one() {
    let out = lee_mykland_test(iid_normal(3, 300, 0.01).view(), 16, 0.01);
    assert!((out.beta_star - 4.6001).abs() < 1e-4, "{}", out.beta_star);
    let n = 300.0_f64;
    let root = (2.0 * n.ln()).sqrt();
    let c = (2.0_f64 / std::f64::consts::PI).sqrt();
    let c_n = root / c - (std::f64::consts::PI.ln() + n.ln().ln()) / (2.0 * c * root);
    let s_n = 1.0 / (c * root);
    assert!((out.c_n - c_n).abs() < 1e-12 && (out.s_n - s_n).abs() < 1e-12);
    assert!((out.threshold - (c_n + out.beta_star * s_n)).abs() < 1e-12);
    assert!((LEE_MYKLAND_C - c).abs() < 1e-15);
  }

  /// §1.3: $K^{opt} = 16$ for daily data; $\sqrt{252 \cdot 78} = 140.2$
  /// for five-minute returns.
  #[test]
  fn window_rule_recovers_the_papers_daily_sixteen() {
    assert_eq!(lee_mykland_window(1), 16);
    assert_eq!(lee_mykland_window(78), 141);
  }

  /// Theorem 1: under the null the statistic is $U/c$, so its standard
  /// deviation is $1/c \approx 1.2533$. Three pinned seeds, best case.
  #[test]
  fn null_statistics_scale_like_one_over_c() {
    let closest = [2718u64, 999, 42]
      .into_iter()
      .map(|seed| {
        let out = lee_mykland_test(iid_normal(seed, 4_000, 0.01).view(), 270, 0.01);
        let vals: Vec<f64> = out
          .statistics
          .iter()
          .copied()
          .filter(|v| v.is_finite())
          .collect();
        let m = vals.iter().sum::<f64>() / vals.len() as f64;
        let var = vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (vals.len() as f64 - 1.0);
        (var.sqrt() - 1.0 / LEE_MYKLAND_C).abs()
      })
      .fold(f64::INFINITY, f64::min);
    assert!(
      closest < 0.05,
      "every seed missed 1/c by at least {closest}"
    );
  }

  /// Pure diffusion: nothing is flagged at the 1% level on the best of
  /// three pinned seeds.
  #[test]
  fn pure_diffusion_is_not_flagged() {
    let fewest = [2718u64, 999, 42]
      .into_iter()
      .map(|seed| {
        lee_mykland_test(iid_normal(seed, 4_000, 0.01).view(), 270, 0.01)
          .jump_indices
          .len()
      })
      .min()
      .unwrap();
    assert_eq!(fewest, 0);
  }

  /// A 20σ return is located exactly, and the bipower window keeps an
  /// earlier jump from masking a second one 40 returns later.
  #[test]
  fn locates_jumps_and_is_robust_to_an_earlier_one() {
    let mut r = iid_normal(17, 2_000, 0.005);
    r[1_000] = 0.10;
    r[1_040] = -0.08;
    let out = lee_mykland_test(r.view(), 100, 0.01);
    assert!(
      out.jump_indices.contains(&1_000) && out.jump_indices.contains(&1_040),
      "{:?}",
      out.jump_indices
    );
    assert!(out.statistics[1_000] > out.threshold);
    assert!(out.statistics[1_040] < -out.threshold);
  }

  #[test]
  #[should_panic(expected = "window must be at least 3")]
  fn rejects_a_window_below_three() {
    let _ = lee_mykland_test(array![0.01_f64, 0.02, 0.03].view(), 2, 0.01);
  }

  #[test]
  #[should_panic(expected = "need at least window = 5 returns")]
  fn rejects_a_series_shorter_than_the_window() {
    let _ = lee_mykland_test(array![0.01_f64, 0.02, 0.03].view(), 5, 0.01);
  }

  #[test]
  #[should_panic(expected = "alpha must lie in (0, 1)")]
  fn rejects_a_unit_alpha() {
    let _ = lee_mykland_test(array![0.01_f64, 0.02, 0.03, 0.04].view(), 3, 1.0);
  }

  #[test]
  #[should_panic(expected = "observations_per_day must be at least 1")]
  fn window_rule_rejects_zero_frequency() {
    let _ = lee_mykland_window(0);
  }
}

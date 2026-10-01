//! # Multi-Level Monte Carlo (MLMC)
//!
//! $\mathbb{E}\[P_L\] = \mathbb{E}\[P_0\]
//!   + \sum_{\ell=1}^{L}\mathbb{E}[P_\ell - P_{\ell-1}]$
//!
//! Giles (2008) eq. (12) allocation at cost $C_\ell \propto h_\ell^{-1}$ ($M = 2$); $V_\ell$ is the
//! per-level `n − 1` sample variance (Welford); levels grow while $|\hat Y_L| > \epsilon/\sqrt{2}$:
//!
//! $$
//! N_\ell = \Bigl\lceil 2\epsilon^{-2}\sqrt{V_\ell / C_\ell}\,
//!   \sum_{k=0}^{L}\sqrt{V_k C_k}\Bigr\rceil
//! $$
//!
//! Giles (2008), "Multilevel Monte Carlo Path Simulation", DOI: 10.1287/opre.1070.0496.
//! Giles (2015), "Multilevel Monte Carlo methods", DOI: 10.1017/S096249291500001X.
//! Welford (1962), "Note on a Method for Calculating Corrected Sums of Squares and Products", DOI: 10.1080/00401706.1962.10490022.

use ndarray::Array1;

use super::Welford;
use crate::traits::FloatExt;

/// MLMC configuration.
#[derive(Debug, Clone)]
pub struct Mlmc<T: FloatExt> {
  /// Target root-mean-square error.
  pub epsilon: T,
  /// Minimum number of levels.
  pub l_min: usize,
  /// Maximum number of levels.
  pub l_max: usize,
  /// Initial samples per level, at least 2.
  pub n0: usize,
}

/// Result of an MLMC estimation.
#[derive(Debug, Clone)]
pub struct MlmcResult<T: FloatExt> {
  /// Estimated mean.
  pub mean: T,
  /// Number of levels used.
  pub n_levels: usize,
  /// Samples generated at each level.
  pub samples_per_level: Vec<usize>,
  /// Estimated variance $V_\ell$ of one sample at each level: the sample
  /// variance (`n − 1` denominator) of the samples drawn there.
  pub variance_per_level: Vec<T>,
}

impl<T: FloatExt> Mlmc<T> {
  /// Configure an MLMC run.
  ///
  /// # Panics
  ///
  /// If `epsilon <= 0`, `l_max < l_min` or `n0 < 2`.
  pub fn new(epsilon: T, l_min: usize, l_max: usize, n0: usize) -> Self {
    assert!(epsilon > T::zero(), "epsilon must be positive");
    assert!(l_max >= l_min, "l_max must be >= l_min");
    assert!(
      n0 >= 2,
      "n0 must be at least 2: a level variance needs two samples"
    );
    Self {
      epsilon,
      l_min,
      l_max,
      n0,
    }
  }

  /// Run the adaptive MLMC algorithm.
  ///
  /// `level_sampler(l, n)` must return `n` coupled differences
  /// `P_l − P_{l−1}` for level `l > 0`, or `n` values of `P_0` for `l = 0`.
  /// The caller is responsible for using the same Brownian increments on the
  /// fine and coarse paths (strong coupling).
  pub fn estimate<F>(&self, level_sampler: F) -> MlmcResult<T>
  where
    F: Fn(usize, usize) -> Array1<T>,
  {
    let two = T::from_f64_fast(2.0);
    let draw_initial = |level: usize| {
      let mut acc = Welford::default();
      acc.extend(level_sampler(level, self.n0));
      acc
    };

    let mut l = self.l_min;
    // Initial samples at each level
    let mut levels = (0..=l).map(&draw_initial).collect::<Vec<_>>();

    for _ in 0..20 {
      // Variance estimates V_l
      let var_l = level_variances::<T>(&levels);

      // Cost model: C_l = 2^l (Euler-Maruyama steps per sample)
      let cost_l: Vec<T> = (0..=l).map(|i| two.powi(i as i32)).collect();

      // Optimal allocation: N_l = ⌈2ε⁻² · √(V_l/C_l) · Σ√(V_l·C_l)⌉
      let sum_sqrt_vc: T = var_l
        .iter()
        .zip(&cost_l)
        .map(|(&v, &c)| (v * c).sqrt())
        .sum();
      let inv_eps_sq = T::one() / (self.epsilon * self.epsilon);
      let optimal: Vec<usize> = var_l
        .iter()
        .zip(&cost_l)
        .map(|(&v, &c)| {
          let n_opt = two * inv_eps_sq * (v / c).sqrt() * sum_sqrt_vc;
          n_opt.to_f64().unwrap().ceil().max(1.0) as usize
        })
        .collect();

      // Generate additional samples where needed
      let mut converged = true;
      for (level, acc) in levels.iter_mut().enumerate() {
        if optimal[level] > acc.count() {
          acc.extend(level_sampler(level, optimal[level] - acc.count()));
          converged = false;
        }
      }

      if converged {
        // Bias check: remaining bias ≈ |E[Y_L]| should be < ε/√2
        let mean_last = T::from_f64_fast(levels[l].mean().abs());
        if mean_last > self.epsilon / two.sqrt() && l < self.l_max {
          l += 1;
          levels.push(draw_initial(l));
        } else {
          break;
        }
      }
    }

    // Final estimate: sum of level means
    let mean = T::from_f64_fast(levels.iter().map(Welford::mean).sum::<f64>());

    MlmcResult {
      mean,
      n_levels: l + 1,
      samples_per_level: levels.iter().map(Welford::count).collect(),
      variance_per_level: level_variances(&levels),
    }
  }
}

fn level_variances<T: FloatExt>(levels: &[Welford]) -> Vec<T> {
  levels
    .iter()
    .map(|acc| T::from_f64_fast(acc.sample_variance()))
    .collect()
}

#[cfg(test)]
mod tests {
  use std::cell::RefCell;

  use super::*;

  /// Level `l` alternates `base_l ± δ_l` with `δ_l² = ¼·2⁻ˡ` (only level 0 on `base0`), so its
  /// sample variance after `n` draws has the closed form [`alternating_variance`].
  fn alternating_levels(base0: f64) -> impl Fn(usize, usize) -> Array1<f64> {
    let drawn = RefCell::new(vec![0usize; 16]);
    move |level, n| {
      let mut drawn = drawn.borrow_mut();
      let delta = (0.25 * 0.5f64.powi(level as i32)).sqrt();
      let base = if level == 0 { base0 } else { 0.0 };
      let start = drawn[level];
      drawn[level] += n;
      Array1::from_iter((start..start + n).map(|k| base + if k % 2 == 0 { delta } else { -delta }))
    }
  }

  fn alternating_variance(level: usize, n: usize) -> f64 {
    let delta_sq = 0.25 * 0.5f64.powi(level as i32);
    let n = n as f64;
    if n % 2.0 == 0.0 {
      delta_sq * n / (n - 1.0)
    } else {
      delta_sq * (n + 1.0) / n
    }
  }

  /// A constant level has no variance. The offset has no exact binary form,
  /// so `sum_sq / n − mean²` leaves a spurious `0.25` behind.
  #[test]
  fn a_constant_level_has_exactly_zero_variance() {
    let mlmc = Mlmc::new(1.0, 2, 4, 100);
    let result = mlmc.estimate(|_, n| Array1::<f64>::from_elem(n, 12_345_678.9));
    assert!(
      result.variance_per_level.iter().all(|&v| v == 0.0),
      "variances {:?}",
      result.variance_per_level
    );
    assert!(result.samples_per_level.iter().all(|&n| n == 100));
  }

  /// Level 0 sits on an offset of 1e8 with spread 0.5: the cancelled variance reads `0.0` and
  /// allocates one sample, while the true one asks for more than the initial 500.
  #[test]
  fn an_offset_level_keeps_its_variance_and_its_samples() {
    let result = Mlmc::new(0.05, 2, 4, 500).estimate(alternating_levels(1.0e8));
    for (level, (&n, &v)) in result
      .samples_per_level
      .iter()
      .zip(&result.variance_per_level)
      .enumerate()
    {
      let exact = alternating_variance(level, n);
      assert!(
        ((v - exact) / exact).abs() < 1e-9,
        "level {level}: variance {v}, exact {exact}"
      );
    }
    assert!(
      result.samples_per_level[0] > 500,
      "level 0 was never topped up: {:?}",
      result.samples_per_level
    );
  }

  /// A level variance needs two samples.
  #[test]
  #[should_panic(expected = "n0 must be at least 2")]
  fn rejects_a_single_initial_sample() {
    Mlmc::new(1.0, 1, 2, 1);
  }

  /// MLMC for a Gbm European call with Euler discretization.
  ///
  /// BS call price for S=100, K=100, r=5%, σ=20%, T=1 ≈ 10.45.
  #[test]
  fn mlmc_gbm_call_converges() {
    let r = 0.05_f64;
    let sigma = 0.2;
    let s0 = 100.0;
    let k = 100.0;
    let tau = 1.0;

    let mlmc = Mlmc::new(1.0, 2, 8, 500);

    let sampler = |level: usize, n: usize| -> Array1<f64> {
      let m_fine = 2usize.pow(level as u32 + 1);
      let dt_fine = tau / m_fine as f64;
      let sqrt_dt_fine = dt_fine.sqrt();
      let disc = (-r * tau).exp();
      let mut out = Array1::<f64>::zeros(n);

      for i in 0..n {
        let z = f64::normal_array(m_fine, 0.0, 1.0);

        // Fine path (Euler)
        let mut s_f = s0;
        for j in 0..m_fine {
          s_f += r * s_f * dt_fine + sigma * s_f * sqrt_dt_fine * z[j];
        }
        let pf = (s_f - k).max(0.0) * disc;

        if level == 0 {
          out[i] = pf;
        } else {
          // Coarse path with coupled Brownian increments
          let m_coarse = m_fine / 2;
          let dt_coarse = tau / m_coarse as f64;
          let mut s_c = s0;
          for j in 0..m_coarse {
            let dw = sqrt_dt_fine * (z[2 * j] + z[2 * j + 1]);
            s_c += r * s_c * dt_coarse + sigma * s_c * dw;
          }
          let pc = (s_c - k).max(0.0) * disc;
          out[i] = pf - pc;
        }
      }
      out
    };

    let result = mlmc.estimate(sampler);
    assert!(
      result.mean > 5.0 && result.mean < 20.0,
      "MLMC price = {:.4}, expected ≈ 10.45",
      result.mean
    );
    assert!(result.n_levels >= 3);
  }
}

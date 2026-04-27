//! # Multi-Level Monte Carlo (MLMC)
//!
//!
//! $\mathbb{E}[P_L] = \mathbb{E}[P_0]
//!   + \sum_{\ell=1}^{L}\mathbb{E}[P_\ell - P_{\ell-1}]$
//!
//! Adaptive MLMC with optimal sample allocation following Giles (2008).
//!
//! Reference: Giles (2015), "Multilevel Monte Carlo methods",
//! arXiv: 1304.5472, DOI: 10.1017/S096249291500001X

use ndarray::Array1;

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
  /// Initial samples per level.
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
  /// Estimated variance at each level.
  pub variance_per_level: Vec<T>,
}

impl<T: FloatExt> Mlmc<T> {
  pub fn new(epsilon: T, l_min: usize, l_max: usize, n0: usize) -> Self {
    assert!(epsilon > T::zero(), "epsilon must be positive");
    assert!(l_max >= l_min, "l_max must be >= l_min");
    assert!(n0 > 0, "n0 must be positive");
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

    let mut l = self.l_min;
    let mut sums: Vec<T> = vec![T::zero(); l + 1];
    let mut sum_sqs: Vec<T> = vec![T::zero(); l + 1];
    let mut n_l: Vec<usize> = vec![0; l + 1];

    // Initial samples at each level
    for level in 0..=l {
      let y = level_sampler(level, self.n0);
      sums[level] = y.iter().copied().sum();
      sum_sqs[level] = y.iter().map(|&v| v * v).sum();
      n_l[level] = self.n0;
    }

    for _ in 0..20 {
      // Variance estimates V_l
      let var_l: Vec<T> = (0..=l)
        .map(|i| {
          let n = T::from_usize_(n_l[i]);
          let m = sums[i] / n;
          (sum_sqs[i] / n - m * m).max(T::zero())
        })
        .collect();

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
      for level in 0..=l {
        if optimal[level] > n_l[level] {
          let extra = optimal[level] - n_l[level];
          let y = level_sampler(level, extra);
          sums[level] += y.iter().copied().sum::<T>();
          sum_sqs[level] += y.iter().map(|&v| v * v).sum::<T>();
          n_l[level] = optimal[level];
          converged = false;
        }
      }

      if converged {
        // Bias check: remaining bias ≈ |E[Y_L]| should be < ε/√2
        let mean_last = (sums[l] / T::from_usize_(n_l[l])).abs();
        if mean_last > self.epsilon / two.sqrt() && l < self.l_max {
          l += 1;
          sums.push(T::zero());
          sum_sqs.push(T::zero());
          n_l.push(0);
          let y = level_sampler(l, self.n0);
          sums[l] = y.iter().copied().sum();
          sum_sqs[l] = y.iter().map(|&v| v * v).sum();
          n_l[l] = self.n0;
        } else {
          break;
        }
      }
    }

    // Final estimate: sum of level means
    let mean: T = sums
      .iter()
      .zip(&n_l)
      .map(|(&s, &n)| s / T::from_usize_(n))
      .sum();
    let variance_per_level: Vec<T> = (0..=l)
      .map(|i| {
        let n = T::from_usize_(n_l[i]);
        let m = sums[i] / n;
        (sum_sqs[i] / n - m * m).max(T::zero())
      })
      .collect();

    MlmcResult {
      mean,
      n_levels: l + 1,
      samples_per_level: n_l,
      variance_per_level,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

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

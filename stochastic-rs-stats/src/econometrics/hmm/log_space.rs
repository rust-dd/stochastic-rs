//! Rabiner's scaled forward-backward recursions (§V.A) on logarithms, every sum by log-sum-exp,
//! so that no state's weight leaves the double range whatever the transitions.
//!
//! Reference: Rabiner, "A Tutorial on Hidden Markov Models and Selected Applications in Speech
//! Recognition", Proceedings of the IEEE, 77(2), 257-286 (1989). DOI: 10.1109/5.18626

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;

use super::GaussianHmm;
use super::log_gauss_pdf;
use crate::traits::RealExt;

/// Rows of `ln α̂_t`, the log densities `ln b_t(j)`, the step scalings `ln ĉ_t` and their sum.
pub(super) struct LogForward {
  pub(super) log_alpha: Array2<f64>,
  pub(super) log_emissions: Array2<f64>,
  pub(super) log_scaling: Array1<f64>,
  pub(super) log_likelihood: f64,
}

impl GaussianHmm {
  pub(super) fn log_forward(&self, observations: ArrayView1<f64>) -> LogForward {
    let n = observations.len();
    let k = self.n_states();
    let ln_stds = self.stds.mapv(f64::ln);
    let ln_transitions = self.transitions.mapv(f64::ln);
    let log_emissions = Array2::from_shape_fn((n, k), |(t, s)| {
      log_gauss_pdf(observations[t], self.means[s], self.stds[s], ln_stds[s])
    });
    let mut log_alpha = Array2::<f64>::zeros((n, k));
    let mut log_scaling = Array1::<f64>::zeros(n);
    let mut log_likelihood = 0.0;
    let mut terms = vec![0.0; k];
    for t in 0..n {
      for j in 0..k {
        let predicted = if t == 0 {
          self.initial[j].ln()
        } else {
          for i in 0..k {
            terms[i] = log_alpha[[t - 1, i]] + ln_transitions[[i, j]];
          }
          log_sum_exp(&terms)
        };
        log_alpha[[t, j]] = predicted + log_emissions[[t, j]];
      }
      // Not `log_sum_exp`: a step no state explains is `−∞ − (−∞)`, NaN, as in the scaled pass.
      let shift = f64::max_or_nan_fold(log_alpha.row(t).iter().copied());
      let log_c = shift
        + log_alpha
          .row(t)
          .iter()
          .map(|v| (v - shift).exp())
          .sum::<f64>()
          .ln();
      for j in 0..k {
        log_alpha[[t, j]] -= log_c;
      }
      log_scaling[t] = log_c;
      log_likelihood += log_c;
    }
    LogForward {
      log_alpha,
      log_emissions,
      log_scaling,
      log_likelihood,
    }
  }

  /// Rabiner's `γ_t` and `Σ_t ξ_t`, each step normalised as in the scaled pass, and the
  /// log-likelihood.
  pub(super) fn log_expectations(
    &self,
    observations: ArrayView1<f64>,
  ) -> (Array2<f64>, Array2<f64>, f64) {
    let LogForward {
      log_alpha,
      log_emissions,
      log_scaling,
      log_likelihood,
    } = self.log_forward(observations);
    let (n, k) = log_alpha.dim();
    let ln_transitions = self.transitions.mapv(f64::ln);
    let mut log_beta = Array2::<f64>::zeros((n, k));
    let mut terms = vec![0.0; k];
    for t in (0..(n - 1)).rev() {
      for i in 0..k {
        for j in 0..k {
          terms[j] = ln_transitions[[i, j]] + log_emissions[[t + 1, j]] + log_beta[[t + 1, j]];
        }
        log_beta[[t, i]] = log_sum_exp(&terms) - log_scaling[t + 1];
      }
    }
    let mut gamma = Array2::<f64>::zeros((n, k));
    for t in 0..n {
      let mut z = 0.0;
      for s in 0..k {
        gamma[[t, s]] = (log_alpha[[t, s]] + log_beta[[t, s]]).exp();
        z += gamma[[t, s]];
      }
      for s in 0..k {
        gamma[[t, s]] /= z;
      }
    }
    let mut xi_sum = Array2::<f64>::zeros((k, k));
    let mut xi = Array2::<f64>::zeros((k, k));
    for t in 0..(n - 1) {
      let mut z = 0.0;
      for i in 0..k {
        for j in 0..k {
          xi[[i, j]] = (log_alpha[[t, i]]
            + ln_transitions[[i, j]]
            + log_emissions[[t + 1, j]]
            + log_beta[[t + 1, j]]
            - log_scaling[t + 1])
            .exp();
          z += xi[[i, j]];
        }
      }
      for i in 0..k {
        for j in 0..k {
          xi_sum[[i, j]] += xi[[i, j]] / z;
        }
      }
    }
    (gamma, xi_sum, log_likelihood)
  }
}

/// `ln Σ_i e^{v_i}`, shifted by the largest `v_i`: `−∞` when every `v_i` is, NaN if any is.
fn log_sum_exp(values: &[f64]) -> f64 {
  let shift = f64::max_or_nan_fold(values.iter().copied());
  if shift == f64::NEG_INFINITY {
    return shift;
  }
  shift + values.iter().map(|v| (v - shift).exp()).sum::<f64>().ln()
}

#[cfg(test)]
mod tests;

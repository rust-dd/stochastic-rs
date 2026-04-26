//! Hidden Markov Model with univariate Gaussian emissions.
//!
//! Implements the forward-backward algorithm with log-space scaling, the
//! Baum-Welch EM training procedure for $(\pi, A, \mu, \sigma)$, and the
//! Viterbi most-likely-path decoder.
//!
//! Reference: Baum, Petrie, Soules, Weiss, "A Maximization Technique
//! Occurring in the Statistical Analysis of Probabilistic Functions of Markov
//! Chains", Annals of Mathematical Statistics, 41(1), 164-171 (1970).
//! DOI: 10.1214/aoms/1177697196
//!
//! Reference: Rabiner, "A Tutorial on Hidden Markov Models and Selected
//! Applications in Speech Recognition", Proceedings of the IEEE, 77(2),
//! 257-286 (1989). DOI: 10.1109/5.18626

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;

/// Result of a Baum-Welch training pass.
#[derive(Debug, Clone)]
pub struct HmmFit {
  /// Number of EM iterations actually executed.
  pub iterations: usize,
  /// Final log-likelihood after the last iteration.
  pub log_likelihood: f64,
  /// `true` if the convergence criterion was met before `max_iter`.
  pub converged: bool,
}

/// Gaussian-emission HMM with `K` hidden states.
#[derive(Debug, Clone)]
pub struct GaussianHmm {
  /// Initial state distribution $\pi$ (length `K`).
  pub initial: Array1<f64>,
  /// Row-stochastic transition matrix $A$ (`K x K`).
  pub transitions: Array2<f64>,
  /// Per-state emission means $\mu_k$ (length `K`).
  pub means: Array1<f64>,
  /// Per-state emission standard deviations $\sigma_k$ (length `K`).
  pub stds: Array1<f64>,
}

impl GaussianHmm {
  /// Construct a new HMM with shapes validated.
  pub fn new(
    initial: Array1<f64>,
    transitions: Array2<f64>,
    means: Array1<f64>,
    stds: Array1<f64>,
  ) -> Self {
    let k = initial.len();
    assert_eq!(transitions.dim(), (k, k), "transitions must be K x K");
    assert_eq!(means.len(), k, "means length must equal K");
    assert_eq!(stds.len(), k, "stds length must equal K");
    assert!(stds.iter().all(|&s| s > 0.0), "stds must be positive");
    Self {
      initial,
      transitions,
      means,
      stds,
    }
  }

  /// Number of hidden states.
  pub fn n_states(&self) -> usize {
    self.initial.len()
  }

  /// Forward-backward log-likelihood with log-space scaling.
  pub fn log_likelihood(&self, observations: ArrayView1<f64>) -> f64 {
    let (alpha, scaling) = self.forward(observations);
    let _ = alpha;
    scaling.iter().map(|s| s.ln()).sum()
  }

  /// Viterbi most-likely state sequence.
  pub fn viterbi(&self, observations: ArrayView1<f64>) -> Array1<usize> {
    let n = observations.len();
    let k = self.n_states();
    let mut delta = Array2::<f64>::from_elem((n, k), f64::NEG_INFINITY);
    let mut psi = Array2::<usize>::zeros((n, k));
    for s in 0..k {
      delta[[0, s]] = self.initial[s].max(1e-300).ln()
        + log_gauss_pdf(observations[0], self.means[s], self.stds[s]);
    }
    for t in 1..n {
      for s in 0..k {
        let emission = log_gauss_pdf(observations[t], self.means[s], self.stds[s]);
        let mut best = f64::NEG_INFINITY;
        let mut arg = 0usize;
        for prev in 0..k {
          let trans = self.transitions[[prev, s]].max(1e-300).ln();
          let candidate = delta[[t - 1, prev]] + trans;
          if candidate > best {
            best = candidate;
            arg = prev;
          }
        }
        delta[[t, s]] = best + emission;
        psi[[t, s]] = arg;
      }
    }
    let mut path = Array1::<usize>::zeros(n);
    let mut best = f64::NEG_INFINITY;
    let mut arg = 0usize;
    for s in 0..k {
      if delta[[n - 1, s]] > best {
        best = delta[[n - 1, s]];
        arg = s;
      }
    }
    path[n - 1] = arg;
    for t in (1..n).rev() {
      path[t - 1] = psi[[t, path[t]]];
    }
    path
  }

  fn forward(&self, observations: ArrayView1<f64>) -> (Array2<f64>, Array1<f64>) {
    let n = observations.len();
    let k = self.n_states();
    let mut alpha = Array2::<f64>::zeros((n, k));
    let mut scaling = Array1::<f64>::zeros(n);
    for s in 0..k {
      alpha[[0, s]] = self.initial[s] * gauss_pdf(observations[0], self.means[s], self.stds[s]);
    }
    let c0: f64 = alpha.row(0).iter().sum::<f64>().max(1e-300);
    scaling[0] = c0;
    for s in 0..k {
      alpha[[0, s]] /= c0;
    }
    for t in 1..n {
      for j in 0..k {
        let mut acc = 0.0;
        for i in 0..k {
          acc += alpha[[t - 1, i]] * self.transitions[[i, j]];
        }
        alpha[[t, j]] = acc * gauss_pdf(observations[t], self.means[j], self.stds[j]);
      }
      let ct: f64 = alpha.row(t).iter().sum::<f64>().max(1e-300);
      scaling[t] = ct;
      for j in 0..k {
        alpha[[t, j]] /= ct;
      }
    }
    (alpha, scaling)
  }

  fn backward(&self, observations: ArrayView1<f64>, scaling: &Array1<f64>) -> Array2<f64> {
    let n = observations.len();
    let k = self.n_states();
    let mut beta = Array2::<f64>::zeros((n, k));
    for s in 0..k {
      beta[[n - 1, s]] = 1.0 / scaling[n - 1];
    }
    for t in (0..(n - 1)).rev() {
      for i in 0..k {
        let mut acc = 0.0;
        for j in 0..k {
          acc += self.transitions[[i, j]]
            * gauss_pdf(observations[t + 1], self.means[j], self.stds[j])
            * beta[[t + 1, j]];
        }
        beta[[t, i]] = acc / scaling[t];
      }
    }
    beta
  }

  /// Baum-Welch EM training. `max_iter` iterations, stopped early when the
  /// log-likelihood improves by less than `tol`.
  pub fn baum_welch(&mut self, observations: ArrayView1<f64>, max_iter: usize, tol: f64) -> HmmFit {
    let n = observations.len();
    let k = self.n_states();
    assert!(
      n >= 2 && k >= 1,
      "need at least two observations and one state"
    );
    let mut prev_ll = f64::NEG_INFINITY;
    let mut converged = false;
    let mut iter_done = 0;
    for it in 0..max_iter {
      iter_done = it + 1;
      let (alpha, scaling) = self.forward(observations);
      let beta = self.backward(observations, &scaling);
      let ll: f64 = scaling.iter().map(|s| s.ln()).sum();
      let mut gamma = Array2::<f64>::zeros((n, k));
      for t in 0..n {
        let mut z = 0.0;
        for s in 0..k {
          gamma[[t, s]] = alpha[[t, s]] * beta[[t, s]];
          z += gamma[[t, s]];
        }
        if z > 0.0 {
          for s in 0..k {
            gamma[[t, s]] /= z;
          }
        }
      }
      let mut xi_sum = Array2::<f64>::zeros((k, k));
      for t in 0..(n - 1) {
        let mut z = 0.0;
        let mut tmp = Array2::<f64>::zeros((k, k));
        for i in 0..k {
          for j in 0..k {
            tmp[[i, j]] = alpha[[t, i]]
              * self.transitions[[i, j]]
              * gauss_pdf(observations[t + 1], self.means[j], self.stds[j])
              * beta[[t + 1, j]];
            z += tmp[[i, j]];
          }
        }
        if z > 0.0 {
          for i in 0..k {
            for j in 0..k {
              xi_sum[[i, j]] += tmp[[i, j]] / z;
            }
          }
        }
      }
      for s in 0..k {
        self.initial[s] = gamma[[0, s]];
      }
      let denom: Array1<f64> = (0..k)
        .map(|i| (0..(n - 1)).map(|t| gamma[[t, i]]).sum::<f64>())
        .collect::<Vec<_>>()
        .into();
      for i in 0..k {
        if denom[i] > 0.0 {
          for j in 0..k {
            self.transitions[[i, j]] = xi_sum[[i, j]] / denom[i];
          }
        }
      }
      let totals: Array1<f64> = (0..k)
        .map(|s| (0..n).map(|t| gamma[[t, s]]).sum::<f64>())
        .collect::<Vec<_>>()
        .into();
      for s in 0..k {
        if totals[s] > 1e-12 {
          let mu = (0..n).map(|t| gamma[[t, s]] * observations[t]).sum::<f64>() / totals[s];
          let var = (0..n)
            .map(|t| gamma[[t, s]] * (observations[t] - mu).powi(2))
            .sum::<f64>()
            / totals[s];
          self.means[s] = mu;
          self.stds[s] = var.max(1e-8).sqrt();
        }
      }
      if (ll - prev_ll).abs() < tol {
        converged = true;
        prev_ll = ll;
        break;
      }
      prev_ll = ll;
    }
    HmmFit {
      iterations: iter_done,
      log_likelihood: prev_ll,
      converged,
    }
  }
}

#[inline]
fn gauss_pdf(x: f64, mean: f64, std: f64) -> f64 {
  let z = (x - mean) / std;
  (-0.5 * z * z).exp() / (std * (2.0 * std::f64::consts::PI).sqrt())
}

#[inline]
fn log_gauss_pdf(x: f64, mean: f64, std: f64) -> f64 {
  let z = (x - mean) / std;
  -0.5 * z * z - std.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;
  use stochastic_rs_distributions::normal::SimdNormal;

  #[test]
  fn viterbi_recovers_two_state_path() {
    let init = array![0.5, 0.5];
    let trans = array![[0.95, 0.05], [0.05, 0.95]];
    let means = array![-1.0, 1.0];
    let stds = array![0.3, 0.3];
    let hmm = GaussianHmm::new(init, trans, means, stds);
    let obs = array![
      -1.0_f64, -0.9, -1.1, -1.05, -0.95, -1.0, 1.0, 1.1, 0.9, 1.05, 0.95, 1.0
    ];
    let path = hmm.viterbi(obs.view());
    assert!(path.iter().take(6).all(|&s| s == 0));
    assert!(path.iter().skip(6).all(|&s| s == 1));
  }

  #[test]
  fn baum_welch_log_likelihood_does_not_decrease() {
    let dist0 = SimdNormal::<f64>::with_seed(-1.0, 0.5, 1);
    let dist1 = SimdNormal::<f64>::with_seed(1.5, 0.4, 2);
    let mut a = vec![0.0_f64; 100];
    let mut b = vec![0.0_f64; 100];
    dist0.fill_slice_fast(&mut a);
    dist1.fill_slice_fast(&mut b);
    let mut obs: Vec<f64> = a.into_iter().chain(b).collect();
    obs.shrink_to_fit();
    let observations = ndarray::Array1::from(obs);
    let init = array![0.5, 0.5];
    let trans = array![[0.9, 0.1], [0.1, 0.9]];
    let means = array![-0.5, 0.5];
    let stds = array![1.0, 1.0];
    let mut hmm = GaussianHmm::new(init, trans, means, stds);
    let ll0 = hmm.log_likelihood(observations.view());
    let fit = hmm.baum_welch(observations.view(), 100, 1e-6);
    assert!(fit.log_likelihood >= ll0 - 1e-6);
    assert!(hmm.stds.iter().all(|&s| s > 0.0));
  }
}

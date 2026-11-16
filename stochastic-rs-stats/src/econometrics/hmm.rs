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
use ndarray::ArrayViewMut1;

use crate::StatsError;
use crate::input::as_slice_or_copy;
use crate::input::reject_constant;
use crate::input::validate;
use crate::traits::RealExt;

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

/// A forward pass: the filtered rows, each step's densities scaled by the largest one the step can
/// reach, the sums `c̃_t` of those scaled rows and the log-likelihood `Σ_t (shift_t + ln c̃_t)`.
struct Forward {
  alpha: Array2<f64>,
  emissions: Array2<f64>,
  scaling: Array1<f64>,
  log_likelihood: f64,
}

impl GaussianHmm {
  /// A `K`-state HMM; panics unless `K >= 1`, the shapes agree, the entries are finite, the stds
  /// positive, and `initial` and each row of `transitions` are non-negative and sum to 1 ± 1e-6.
  pub fn new(
    initial: Array1<f64>,
    transitions: Array2<f64>,
    means: Array1<f64>,
    stds: Array1<f64>,
  ) -> Self {
    let hmm = Self {
      initial,
      transitions,
      means,
      stds,
    };
    hmm.assert_parameters();
    hmm
  }

  /// [`Self::new`]'s checks, which every entry point repeats for a struct literal.
  #[track_caller]
  fn assert_parameters(&self) {
    let Self {
      initial,
      transitions,
      means,
      stds,
    } = self;
    let k = initial.len();
    assert!(
      k >= 1,
      "initial.len() must satisfy `initial.len() >= 1`, got initial.len() = {k:?}"
    );
    assert!(
      initial.iter().all(|p| p.is_finite()),
      "initial must satisfy `initial.iter().all(|p| p.is_finite())`, got initial = {initial:?}"
    );
    assert!(
      initial.iter().all(|&p| p >= 0.0),
      "initial must satisfy `initial.iter().all(|&p| p >= 0.0)`, got initial = {initial:?}"
    );
    let sum = initial.sum();
    assert!(
      (sum - 1.0).abs() < 1e-6,
      "initial must satisfy `(initial.sum() - 1.0).abs() < 1e-6`, got initial.sum() = {sum:?}"
    );
    assert!(
      transitions.dim() == (k, k),
      "transitions must satisfy `transitions.dim() == (k, k)`, got transitions.dim() = {:?}, k = {k}",
      transitions.dim()
    );
    assert!(
      transitions.iter().all(|p| p.is_finite()),
      "transitions must satisfy `transitions.iter().all(|p| p.is_finite())`, got transitions = {transitions:?}"
    );
    assert!(
      transitions.iter().all(|&p| p >= 0.0),
      "transitions must satisfy `transitions.iter().all(|&p| p >= 0.0)`, got transitions = {transitions:?}"
    );
    for (i, row) in transitions.rows().into_iter().enumerate() {
      let sum = row.sum();
      assert!(
        (sum - 1.0).abs() < 1e-6,
        "transitions must satisfy `(transitions.row(i).sum() - 1.0).abs() < 1e-6` for every row, got transitions.row({i}).sum() = {sum:?}"
      );
    }
    assert!(
      means.len() == k,
      "means must satisfy `means.len() == k`, got means.len() = {}, k = {k}",
      means.len()
    );
    assert!(
      means.iter().all(|m| m.is_finite()),
      "means must satisfy `means.iter().all(|m| m.is_finite())`, got means = {means:?}"
    );
    assert!(
      stds.len() == k,
      "stds must satisfy `stds.len() == k`, got stds.len() = {}, k = {k}",
      stds.len()
    );
    assert!(
      stds.iter().all(|&s| s > 0.0 && s.is_finite()),
      "stds must satisfy `stds.iter().all(|&s| s > 0.0 && s.is_finite())`, got stds = {stds:?}"
    );
  }

  /// Number of hidden states.
  pub fn n_states(&self) -> usize {
    self.initial.len()
  }

  /// Forward log-likelihood, summed in log space; `0.0` for an empty series and NaN on a non-finite
  /// observation; panics as [`Self::new`] does on a struct literal.
  pub fn log_likelihood(&self, observations: ArrayView1<f64>) -> f64 {
    self.assert_parameters();
    if observations.is_empty() {
      return 0.0;
    }
    self.forward(observations).log_likelihood
  }

  /// Filtered `P(s_t = k | y_1..y_t)` per observation, which later data never move (unlike
  /// [`Self::viterbi`]'s states); NaN rows from a NaN, and panics as [`Self::new`] on a literal.
  pub fn filtered_state_probs(&self, observations: ArrayView1<f64>) -> Array2<f64> {
    self.assert_parameters();
    self.forward(observations).alpha
  }

  /// The filtered distribution one observation after `prev` ([`Self::initial`] to start), `O(K²)`;
  /// NaN from a NaN input, and panics as [`Self::new`] does or unless `prev.len() == K`.
  pub fn filter_step(&self, prev: ArrayView1<f64>, observation: f64) -> Array1<f64> {
    self.assert_parameters();
    let k = self.n_states();
    assert_eq!(prev.len(), k, "prev must have one entry per state");
    let mut predicted = Array1::<f64>::zeros(k);
    for j in 0..k {
      let mut acc = 0.0;
      for i in 0..k {
        acc += prev[i] * self.transitions[[i, j]];
      }
      predicted[j] = acc;
    }
    let ln_stds = self.stds.mapv(f64::ln);
    let mut scaled = Array1::<f64>::zeros(k);
    let mut next = Array1::<f64>::zeros(k);
    self.update(
      observation,
      predicted.view(),
      ln_stds.view(),
      scaled.view_mut(),
      next.view_mut(),
    );
    next
  }

  /// Viterbi most-likely state sequence, a zero probability scoring `ln 0 = −∞`; empty for an empty
  /// series, and panics as [`Self::new`] does on a struct literal.
  pub fn viterbi(&self, observations: ArrayView1<f64>) -> Array1<usize> {
    self.assert_parameters();
    let n = observations.len();
    if n == 0 {
      return Array1::zeros(0);
    }
    let k = self.n_states();
    let ln_stds = self.stds.mapv(f64::ln);
    let ln_transitions = self.transitions.mapv(f64::ln);
    let mut delta = Array2::<f64>::from_elem((n, k), f64::NEG_INFINITY);
    let mut psi = Array2::<usize>::zeros((n, k));
    for s in 0..k {
      delta[[0, s]] = self.initial[s].ln()
        + log_gauss_pdf(observations[0], self.means[s], self.stds[s], ln_stds[s]);
    }
    for t in 1..n {
      for s in 0..k {
        let emission = log_gauss_pdf(observations[t], self.means[s], self.stds[s], ln_stds[s]);
        let mut best = f64::NEG_INFINITY;
        let mut arg = 0usize;
        for prev in 0..k {
          let candidate = delta[[t - 1, prev]] + ln_transitions[[prev, s]];
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

  /// One filtering step from the predicted distribution: each density it reaches, scaled by the
  /// largest, into `scaled`, the normalised filtered row into `filtered`; returns `(c̃, shift)`.
  fn update(
    &self,
    observation: f64,
    predicted: ArrayView1<f64>,
    ln_stds: ArrayView1<f64>,
    mut scaled: ArrayViewMut1<f64>,
    mut filtered: ArrayViewMut1<f64>,
  ) -> (f64, f64) {
    let k = predicted.len();
    let mut shift = f64::NEG_INFINITY;
    for j in 0..k {
      if predicted[j] > 0.0 {
        let ln_density = log_gauss_pdf(observation, self.means[j], self.stds[j], ln_stds[j]);
        scaled[j] = ln_density;
        shift = shift.max_or_nan(ln_density);
      }
    }
    let mut sum = 0.0;
    for j in 0..k {
      let density = if predicted[j] > 0.0 {
        (scaled[j] - shift).exp()
      } else {
        0.0
      };
      scaled[j] = density;
      filtered[j] = predicted[j] * density;
      sum += filtered[j];
    }
    for j in 0..k {
      filtered[j] /= sum;
    }
    (sum, shift)
  }

  fn forward(&self, observations: ArrayView1<f64>) -> Forward {
    let n = observations.len();
    let k = self.n_states();
    let ln_stds = self.stds.mapv(f64::ln);
    let mut alpha = Array2::<f64>::zeros((n, k));
    let mut emissions = Array2::<f64>::zeros((n, k));
    let mut scaling = Array1::<f64>::zeros(n);
    let mut predicted = self.initial.clone();
    let mut log_likelihood = 0.0;
    for t in 0..n {
      if t > 0 {
        for j in 0..k {
          let mut acc = 0.0;
          for i in 0..k {
            acc += alpha[[t - 1, i]] * self.transitions[[i, j]];
          }
          predicted[j] = acc;
        }
      }
      let (sum, shift) = self.update(
        observations[t],
        predicted.view(),
        ln_stds.view(),
        emissions.row_mut(t),
        alpha.row_mut(t),
      );
      scaling[t] = sum;
      log_likelihood += shift + sum.ln();
    }
    Forward {
      alpha,
      emissions,
      scaling,
      log_likelihood,
    }
  }

  fn backward(&self, emissions: &Array2<f64>, scaling: &Array1<f64>) -> Array2<f64> {
    let (n, k) = emissions.dim();
    let mut beta = Array2::<f64>::zeros((n, k));
    for s in 0..k {
      beta[[n - 1, s]] = 1.0 / scaling[n - 1];
    }
    for t in (0..(n - 1)).rev() {
      for i in 0..k {
        let mut acc = 0.0;
        for j in 0..k {
          acc += self.transitions[[i, j]] * emissions[[t + 1, j]] * beta[[t + 1, j]];
        }
        beta[[t, i]] = acc / scaling[t];
      }
    }
    beta
  }

  /// Baum–Welch EM, at most `max_iter` steps or until the gain is below `tol`; panics as `new`;
  /// `Err`, the last full update kept, on 0 states, < 2, non-finite or constant points or overflow.
  pub fn baum_welch(
    &mut self,
    observations: ArrayView1<f64>,
    max_iter: usize,
    tol: f64,
  ) -> Result<HmmFit, StatsError> {
    let k = self.n_states();
    if k == 0 {
      return Err(StatsError::InvalidParameter {
        name: "initial.len()",
        value: 0.0,
        constraint: "initial.len() >= 1",
      });
    }
    self.assert_parameters();
    let y = as_slice_or_copy(observations);
    validate(&y, 2)?;
    reject_constant(&y, "constant series")?;
    let n = observations.len();
    let mut prev_ll = f64::NEG_INFINITY;
    let mut converged = false;
    let mut iter_done = 0;
    for it in 0..max_iter {
      iter_done = it + 1;
      let Forward {
        alpha,
        emissions,
        scaling,
        log_likelihood: ll,
      } = self.forward(observations);
      if !ll.is_finite() {
        return Err(StatsError::Numerical {
          stage: "forward pass",
        });
      }
      let beta = self.backward(&emissions, &scaling);
      let mut gamma = Array2::<f64>::zeros((n, k));
      for t in 0..n {
        let mut z = 0.0;
        for s in 0..k {
          gamma[[t, s]] = alpha[[t, s]] * beta[[t, s]];
          z += gamma[[t, s]];
        }
        if !(z > 0.0 && z.is_finite()) {
          return Err(StatsError::Numerical {
            stage: "backward pass",
          });
        }
        for s in 0..k {
          gamma[[t, s]] /= z;
        }
      }
      let mut xi_sum = Array2::<f64>::zeros((k, k));
      let mut tmp = Array2::<f64>::zeros((k, k));
      for t in 0..(n - 1) {
        let mut z = 0.0;
        for i in 0..k {
          for j in 0..k {
            tmp[[i, j]] =
              alpha[[t, i]] * self.transitions[[i, j]] * emissions[[t + 1, j]] * beta[[t + 1, j]];
            z += tmp[[i, j]];
          }
        }
        if !(z > 0.0 && z.is_finite()) {
          return Err(StatsError::Numerical {
            stage: "backward pass",
          });
        }
        for i in 0..k {
          for j in 0..k {
            xi_sum[[i, j]] += tmp[[i, j]] / z;
          }
        }
      }
      let totals: Array1<f64> = (0..k)
        .map(|s| (0..n).map(|t| gamma[[t, s]]).sum::<f64>())
        .collect::<Vec<_>>()
        .into();
      let mut emission = Vec::with_capacity(k);
      for s in 0..k {
        if totals[s] > 1e-12 {
          let mu = (0..n).map(|t| gamma[[t, s]] * observations[t]).sum::<f64>() / totals[s];
          let var = (0..n)
            .map(|t| gamma[[t, s]] * (observations[t] - mu).powi(2))
            .sum::<f64>()
            / totals[s];
          if !var.is_finite() {
            return Err(StatsError::Numerical {
              stage: "emission variance",
            });
          }
          emission.push(Some((mu, var)));
        } else {
          emission.push(None);
        }
      }
      for s in 0..k {
        self.initial[s] = gamma[[0, s]];
      }
      // Rabiner's (40b) divides by `Σ_t γ_t(i)`, which (38) makes this row's sum; the row's own sum
      // keeps it a probability vector (43b) to rounding even where the state's mass is subnormal.
      for i in 0..k {
        let row = (0..k).map(|j| xi_sum[[i, j]]).sum::<f64>();
        if row > 0.0 {
          for j in 0..k {
            self.transitions[[i, j]] = xi_sum[[i, j]] / row;
          }
        }
      }
      for (s, update) in emission.into_iter().enumerate() {
        if let Some((mu, var)) = update {
          self.means[s] = mu;
          // Rabiner (1989), §V.D, eq. (113b): a reestimated variance is held at a floor δ.
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
    Ok(HmmFit {
      iterations: iter_done,
      log_likelihood: prev_ll,
      converged,
    })
  }
}

/// `ln N(x; mean, std²)`, with `ln_std = std.ln()` taken once per state.
#[inline]
fn log_gauss_pdf(x: f64, mean: f64, std: f64, ln_std: f64) -> f64 {
  let z = (x - mean) / std;
  -0.5 * z * z - ln_std - 0.5 * (2.0 * std::f64::consts::PI).ln()
}

#[cfg(test)]
mod tests;

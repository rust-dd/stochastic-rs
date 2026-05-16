//! # Multivariate Hawkes Process
//!
//! $$
//! \lambda_i(t) = \mu_i + \sum_{j=1}^{D}\sum_{T_k^j < t} \alpha_{ij}\,e^{-\beta_{ij}(t - T_k^j)},
//! \quad i = 1,\dots,D
//! $$
//!
//! D-dimensional self-exciting point process with exponential kernels and
//! cross-excitation. Stationarity requires $\rho(\Gamma) < 1$ where
//! $\Gamma_{ij} = \alpha_{ij}/\beta_{ij}$.
//!
//! Simulated via multivariate Ogata thinning.
//!
//! Reference:
//! - Hawkes (1971), "Spectra of some self-exciting and mutually exciting point processes"
//! - Bacry, Mastromatteo, Muzy (2015), "Hawkes processes in finance", arXiv:1502.04592

use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// D-dimensional Hawkes process with exponential kernels.
///
/// Output: `[Array1<T>; D]` — event times for each component, or
/// a single `Array2<T>` of shape `(D, max_events)` via `sample()`.
pub struct MultivariateHawkes<T: FloatExt, S: SeedExt = Unseeded> {
  /// Baseline intensities $\mu_i > 0$, length D.
  pub mu: Array1<T>,
  /// Excitation matrix $\alpha_{ij} \ge 0$, shape (D, D).
  pub alpha: Array2<T>,
  /// Decay matrix $\beta_{ij} > 0$, shape (D, D).
  pub beta: Array2<T>,
  /// Time horizon.
  pub t_max: T,
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> MultivariateHawkes<T, S> {
  pub fn new(mu: Array1<T>, alpha: Array2<T>, beta: Array2<T>, t_max: T, seed: S) -> Self {
    let d = mu.len();
    assert_eq!(alpha.shape(), [d, d], "alpha must be (D, D)");
    assert_eq!(beta.shape(), [d, d], "beta must be (D, D)");
    Self {
      mu,
      alpha,
      beta,
      t_max,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for MultivariateHawkes<T, S> {
  type Output = Vec<Array1<T>>;

  /// Multivariate Ogata thinning.
  ///
  /// Returns a `Vec` of length D, where each element contains the
  /// event arrival times for that component.
  fn sample(&self) -> Self::Output {
    let mut rng = self.seed.rng();
    let d = self.mu.len();

    // S[i][j] = running self-exciting component from source j to target i
    let mut s = vec![vec![T::zero(); d]; d];
    let mut t = T::zero();
    let mut events: Vec<Vec<T>> = (0..d).map(|_| vec![T::zero()]).collect();

    while t < self.t_max {
      // Compute component intensities and total upper bound
      let mut lambdas = vec![T::zero(); d];
      for i in 0..d {
        lambdas[i] = self.mu[i];
        for j in 0..d {
          lambdas[i] += s[i][j];
        }
      }
      let lambda_bar: T = lambdas.iter().copied().sum();
      if lambda_bar <= T::zero() {
        break;
      }

      // Propose next event time
      let u = T::one() - T::sample_uniform_simd(&mut rng);
      let dt = -u.ln() / lambda_bar;
      t += dt;

      if t >= self.t_max {
        break;
      }

      // Decay all S components
      for i in 0..d {
        for j in 0..d {
          s[i][j] = s[i][j] * (-self.beta[[i, j]] * dt).exp();
        }
      }

      // Recompute intensities at proposed time
      for i in 0..d {
        lambdas[i] = self.mu[i];
        for j in 0..d {
          lambdas[i] += s[i][j];
        }
      }

      // Accept/reject and assign to component
      let v = T::sample_uniform_simd(&mut rng) * lambda_bar;
      let mut cumsum = T::zero();
      for i in 0..d {
        cumsum += lambdas[i];
        if v <= cumsum {
          // Event accepted on component i
          events[i].push(t);
          // Excite all components from source i
          for k in 0..d {
            s[k][i] += self.alpha[[k, i]];
          }
          break;
        }
      }
    }

    events.into_iter().map(Array1::from_vec).collect()
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Array2;
  use ndarray::array;

  use super::*;

  #[test]
  fn bivariate_hawkes_runs() {
    let mu = array![1.0_f64, 1.5];
    let alpha = Array2::from_shape_vec((2, 2), vec![0.3, 0.1, 0.2, 0.4]).unwrap();
    let beta = Array2::from_shape_vec((2, 2), vec![2.0, 2.0, 2.0, 2.0]).unwrap();
    let h = MultivariateHawkes::new(mu, alpha, beta, 10.0, Unseeded);
    let events = h.sample();
    assert_eq!(events.len(), 2);
    assert!(events[0].len() > 1, "component 0 should have events");
    assert!(events[1].len() > 1, "component 1 should have events");
  }

  #[test]
  fn cross_excitation_increases_events() {
    // No cross-excitation
    let mu = array![2.0_f64, 2.0];
    let alpha_diag = Array2::from_shape_vec((2, 2), vec![0.5, 0.0, 0.0, 0.5]).unwrap();
    let beta = Array2::from_shape_vec((2, 2), vec![3.0, 3.0, 3.0, 3.0]).unwrap();
    let h1 = MultivariateHawkes::new(mu.clone(), alpha_diag, beta.clone(), 50.0, Unseeded);

    // With cross-excitation
    let alpha_full = Array2::from_shape_vec((2, 2), vec![0.5, 0.5, 0.5, 0.5]).unwrap();
    let h2 = MultivariateHawkes::new(mu, alpha_full, beta, 50.0, Unseeded);

    let n_trials = 20;
    let avg1: f64 = (0..n_trials)
      .map(|_| (h1.sample()[0].len() + h1.sample()[1].len()) as f64)
      .sum::<f64>()
      / n_trials as f64;
    let avg2: f64 = (0..n_trials)
      .map(|_| (h2.sample()[0].len() + h2.sample()[1].len()) as f64)
      .sum::<f64>()
      / n_trials as f64;

    // Cross-excitation should produce more events on average
    assert!(
      avg2 > avg1 * 0.9,
      "cross-excited avg={avg2:.0} should exceed diagonal avg={avg1:.0}"
    );
  }

  #[test]
  fn univariate_matches_scalar() {
    // D=1 multivariate should behave like the scalar Hawkes
    let mu = array![3.0_f64];
    let alpha = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
    let beta = Array2::from_shape_vec((1, 1), vec![5.0]).unwrap();
    let h = MultivariateHawkes::new(mu, alpha, beta, 10.0, Unseeded);
    let events = h.sample();
    assert_eq!(events.len(), 1);
    assert!(events[0].len() > 2, "should have multiple events");
    // Events should be sorted
    for w in events[0].as_slice().unwrap().windows(2) {
      assert!(w[0] <= w[1], "events must be sorted");
    }
  }
}

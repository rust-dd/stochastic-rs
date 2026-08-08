//! Online joint covariance for Heston Monte Carlo observables.

use super::EstimateWithError;
use super::HestonMalliavinError;

#[derive(Debug, Default)]
pub(super) struct OnlineScalar {
  count: usize,
  mean: f64,
  second_moment: f64,
}

impl OnlineScalar {
  pub(super) fn push(&mut self, value: f64) -> Result<(), HestonMalliavinError> {
    if !value.is_finite() {
      return Err(HestonMalliavinError::NonFiniteSimulation);
    }
    self.count += 1;
    let difference = value - self.mean;
    self.mean += difference / self.count as f64;
    self.second_moment += difference * (value - self.mean);
    Ok(())
  }

  pub(super) fn finish(self) -> Result<EstimateWithError, HestonMalliavinError> {
    if self.count < 2 {
      return Err(HestonMalliavinError::InsufficientSamples);
    }
    let sample_variance = self.second_moment / (self.count - 1) as f64;
    Ok(EstimateWithError {
      value: self.mean,
      standard_error: (sample_variance.max(0.0) / self.count as f64).sqrt(),
    })
  }
}

/// Welford accumulator for `N` jointly observed Monte Carlo values.
#[derive(Debug)]
pub(super) struct OnlineCovariance<const N: usize> {
  count: usize,
  mean: [f64; N],
  second_moment: [[f64; N]; N],
}

impl<const N: usize> Default for OnlineCovariance<N> {
  fn default() -> Self {
    Self {
      count: 0,
      mean: [0.0; N],
      second_moment: [[0.0; N]; N],
    }
  }
}

/// Sample summary of jointly accumulated observations.
#[derive(Debug, Clone, Copy)]
pub(super) struct CovarianceSummary<const N: usize> {
  pub(super) mean: [f64; N],
  pub(super) sample_covariance: [[f64; N]; N],
  pub(super) estimator_covariance: [[f64; N]; N],
  pub(super) independent_samples: usize,
}

impl<const N: usize> CovarianceSummary<N> {
  pub(super) fn estimate(&self, index: usize) -> EstimateWithError {
    EstimateWithError {
      value: self.mean[index],
      standard_error: self.estimator_covariance[index][index].max(0.0).sqrt(),
    }
  }
}

impl<const N: usize> OnlineCovariance<N> {
  pub(super) fn push(&mut self, observation: [f64; N]) {
    self.count += 1;
    let count = self.count as f64;
    let delta = std::array::from_fn::<_, N, _>(|i| observation[i] - self.mean[i]);
    for (mean, difference) in self.mean.iter_mut().zip(delta) {
      *mean += difference / count;
    }
    let adjusted = std::array::from_fn::<_, N, _>(|i| observation[i] - self.mean[i]);
    for (i, row) in self.second_moment.iter_mut().enumerate() {
      for (j, entry) in row.iter_mut().enumerate() {
        *entry += delta[i] * adjusted[j];
      }
    }
  }

  pub(super) fn finish(self) -> Result<CovarianceSummary<N>, HestonMalliavinError> {
    if self.count < 2 {
      return Err(HestonMalliavinError::InsufficientSamples);
    }
    let divisor = (self.count - 1) as f64;
    let sample_covariance =
      std::array::from_fn(|i| std::array::from_fn(|j| self.second_moment[i][j] / divisor));
    let estimator_covariance: [[f64; N]; N] =
      std::array::from_fn(|i| std::array::from_fn(|j| sample_covariance[i][j] / self.count as f64));
    Ok(CovarianceSummary {
      mean: self.mean,
      sample_covariance,
      estimator_covariance,
      independent_samples: self.count,
    })
  }
}

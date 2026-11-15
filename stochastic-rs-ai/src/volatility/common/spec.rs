use anyhow::Result;
use anyhow::bail;

#[derive(Clone, Debug)]
pub struct TrainConfig {
  pub test_ratio: f32,
  pub batch_size: usize,
  pub epochs: usize,
  pub learning_rate: f64,
  pub random_seed: u64,
  pub shuffle: bool,
}

impl Default for TrainConfig {
  fn default() -> Self {
    Self {
      test_ratio: 0.15,
      batch_size: 32,
      epochs: 200,
      learning_rate: 1e-3,
      random_seed: 42,
      shuffle: true,
    }
  }
}

#[derive(Clone, Debug)]
pub struct EpochMetrics {
  pub epoch: usize,
  pub train_rmse: f32,
  pub val_rmse: f32,
}

#[derive(Clone, Debug)]
pub struct TrainReport {
  pub epochs: Vec<EpochMetrics>,
}

/// The contract of a surrogate: the parameter box and the strike-maturity grid its output is laid out on.
#[derive(Clone, Debug)]
pub struct StochVolModelSpec {
  pub model_id: String,
  pub hidden_dim: usize,
  pub param_lb: Vec<f32>,
  pub param_ub: Vec<f32>,
  /// Strikes as `K / S0` in the network's column order (descending for the shipped Heston set).
  pub strikes: Vec<f64>,
  /// Maturities in years, strictly ascending; the output is maturity-major.
  pub maturities: Vec<f64>,
}

impl StochVolModelSpec {
  pub fn new(
    model_id: impl Into<String>,
    hidden_dim: usize,
    param_lb: Vec<f32>,
    param_ub: Vec<f32>,
    strikes: Vec<f64>,
    maturities: Vec<f64>,
  ) -> Result<Self> {
    if param_lb.is_empty() || param_lb.len() != param_ub.len() {
      bail!("param_lb and param_ub must be non-empty and of equal length");
    }
    if hidden_dim == 0 {
      bail!("hidden_dim must be > 0");
    }
    for (i, (lb, ub)) in param_lb.iter().zip(&param_ub).enumerate() {
      if !lb.is_finite() || !ub.is_finite() {
        bail!("parameter bounds must be finite");
      }
      if ub <= lb {
        bail!("param_ub[{i}] must be greater than param_lb[{i}]");
      }
    }
    if strikes.is_empty() || strikes.iter().any(|k| !k.is_finite() || *k <= 0.0) {
      bail!("strikes must be non-empty, finite and positive");
    }
    let mut sorted = strikes.clone();
    sorted.sort_by(f64::total_cmp);
    if sorted.windows(2).any(|w| w[0] == w[1]) {
      bail!("strikes must be distinct");
    }
    if maturities.is_empty() || maturities.iter().any(|t| !t.is_finite() || *t <= 0.0) {
      bail!("maturities must be non-empty, finite and positive");
    }
    if !maturities.windows(2).all(|w| w[0] < w[1]) {
      bail!("maturities must be strictly ascending");
    }
    Ok(Self {
      model_id: model_id.into(),
      hidden_dim,
      param_lb,
      param_ub,
      strikes,
      maturities,
    })
  }

  pub fn input_dim(&self) -> usize {
    self.param_lb.len()
  }

  pub fn output_dim(&self) -> usize {
    self.strikes.len() * self.maturities.len()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn toy(strikes: Vec<f64>, maturities: Vec<f64>) -> Result<StochVolModelSpec> {
    StochVolModelSpec::new("toy", 8, vec![0.0], vec![1.0], strikes, maturities)
  }

  #[test]
  fn the_spec_carries_the_grid_and_derives_the_dimensions() {
    let spec = StochVolModelSpec::new(
      "toy",
      8,
      vec![0.0, 0.0],
      vec![1.0, 1.0],
      vec![1.1, 0.9, 1.0],
      vec![0.5, 1.0],
    )
    .unwrap();
    assert_eq!((spec.input_dim(), spec.output_dim()), (2, 6));
    assert_eq!(spec.strikes, vec![1.1, 0.9, 1.0]);
  }

  #[test]
  fn a_bad_grid_is_an_error() {
    assert!(toy(vec![], vec![0.5]).is_err());
    assert!(toy(vec![1.0, 1.0], vec![0.5]).is_err());
    assert!(toy(vec![1.0], vec![1.0, 0.5]).is_err());
    assert!(toy(vec![-1.0], vec![0.5]).is_err());
    assert!(toy(vec![f64::NAN], vec![0.5]).is_err());
    assert!(toy(vec![1.0], vec![]).is_err());
  }
}

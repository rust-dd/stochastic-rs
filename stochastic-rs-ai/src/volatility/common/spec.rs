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

#[derive(Clone, Debug)]
pub struct StochVolModelSpec {
  pub model_id: String,
  pub input_dim: usize,
  pub output_dim: usize,
  pub hidden_dim: usize,
  pub param_lb: Vec<f32>,
  pub param_ub: Vec<f32>,
}

impl StochVolModelSpec {
  pub fn new(
    model_id: impl Into<String>,
    input_dim: usize,
    output_dim: usize,
    hidden_dim: usize,
    param_lb: Vec<f32>,
    param_ub: Vec<f32>,
  ) -> Result<Self> {
    if input_dim == 0 {
      bail!("input_dim must be > 0");
    }
    if output_dim == 0 {
      bail!("output_dim must be > 0");
    }
    if hidden_dim == 0 {
      bail!("hidden_dim must be > 0");
    }
    if param_lb.len() != input_dim || param_ub.len() != input_dim {
      bail!("param bounds must match input_dim");
    }
    for i in 0..input_dim {
      if !param_lb[i].is_finite() || !param_ub[i].is_finite() {
        bail!("parameter bounds must be finite");
      }
      if param_ub[i] <= param_lb[i] {
        bail!("param_ub[{i}] must be greater than param_lb[{i}]");
      }
    }
    Ok(Self {
      model_id: model_id.into(),
      input_dim,
      output_dim,
      hidden_dim,
      param_lb,
      param_ub,
    })
  }
}

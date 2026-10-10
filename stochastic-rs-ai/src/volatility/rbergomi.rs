//! Surrogate of the rough Bergomi surface with flat forward variance (`NNRoughBergomi.ipynb`);
//! inputs `[xi0, nu, rho, H]`: forward variance, vol-of-vol, correlation, Hurst exponent.
//! <https://github.com/amuguruza/NN-StochVol-Calibrations> (`RoughBergomi/Flat Forward Variance`)

use std::path::Path;

use anyhow::Result;
use candle_core::Device;
use ndarray::Array2;

use super::common::StochVolModelSpec;
use super::common::StochVolNn;
use super::common::TrainConfig;
use super::common::TrainReport;
use super::grid;

pub const MODEL_ID: &str = "rbergomi";
pub const INPUT_DIM: usize = 4;
pub const OUTPUT_DIM: usize = grid::LEN;
pub const DEFAULT_HIDDEN_DIM: usize = 30;
pub const PARAM_LB: [f32; INPUT_DIM] = [0.01, 0.3, -0.95, 0.025];
pub const PARAM_UB: [f32; INPUT_DIM] = [0.16, 4.0, -0.1, 0.5];

/// `K / S0` of output column `i`, ascending.
pub const STRIKES: [f64; grid::MONEYNESS.len()] = grid::MONEYNESS;

pub struct RBergomiNn {
  inner: StochVolNn,
}

impl RBergomiNn {
  pub fn new(device: &Device) -> Result<Self> {
    Self::with_hidden(device, DEFAULT_HIDDEN_DIM)
  }

  pub fn with_hidden(device: &Device, hidden_dim: usize) -> Result<Self> {
    let spec = StochVolModelSpec::new(
      MODEL_ID,
      INPUT_DIM,
      OUTPUT_DIM,
      hidden_dim,
      PARAM_LB.to_vec(),
      PARAM_UB.to_vec(),
    )?;
    Ok(Self {
      inner: StochVolNn::new(spec, device)?,
    })
  }

  pub fn train(
    &mut self,
    params: &Array2<f32>,
    surfaces: &Array2<f32>,
    config: &TrainConfig,
  ) -> Result<TrainReport> {
    self.inner.train(params, surfaces, config)
  }

  /// The flat surface, [`OUTPUT_DIM`] long: one block per [`grid::MATURITIES`] entry, with columns
  /// in [`STRIKES`] order (ascending).
  pub fn predict_surface(&self, params: &[f32; INPUT_DIM]) -> Result<Vec<f32>> {
    self.inner.predict_surface(params)
  }

  pub fn predict_surfaces(&self, params: &Array2<f32>) -> Result<Array2<f32>> {
    self.inner.predict_surfaces(params)
  }

  /// The prediction as an [`ImpliedVolSurface`](stochastic_rs_quant::vol_surface::ImpliedVolSurface)
  /// with ascending strikes; pass [`STRIKES`] times the spot, [`grid::MATURITIES`] and the forwards.
  #[cfg(feature = "quant")]
  pub fn predict_implied_vol_surface(
    &self,
    params: &[f32; INPUT_DIM],
    strikes: Vec<f64>,
    maturities: Vec<f64>,
    forwards: Vec<f64>,
  ) -> Result<stochastic_rs_quant::vol_surface::ImpliedVolSurface> {
    self
      .inner
      .predict_implied_vol_surface(params, strikes, maturities, forwards)
  }

  pub fn save<P: AsRef<Path>>(&self, dir: P) -> Result<()> {
    self.inner.save(dir)
  }

  pub fn load<P: AsRef<Path>>(dir: P, device: &Device) -> Result<Self> {
    Ok(Self {
      inner: StochVolNn::load(MODEL_ID, dir, device)?,
    })
  }
}

#[cfg(feature = "quant")]
impl crate::calibration::SurrogateModel for RBergomiNn {
  fn nn(&self) -> &StochVolNn {
    &self.inner
  }
}

#[cfg(test)]
mod tests {
  use std::fs;

  use super::*;
  use crate::volatility::common::synthetic_surface_dataset;

  #[test]
  fn train_save_load_roundtrip() -> Result<()> {
    let device = Device::Cpu;
    let (params, surfaces) = synthetic_surface_dataset(&PARAM_LB, &PARAM_UB, 192, OUTPUT_DIM, 13);
    let mut model = RBergomiNn::new(&device)?;
    let cfg = TrainConfig {
      test_ratio: 0.2,
      batch_size: 32,
      epochs: 20,
      learning_rate: 1e-3,
      random_seed: 1234,
      shuffle: true,
    };
    let report = model.train(&params, &surfaces, &cfg)?;
    assert_eq!(report.epochs.len(), cfg.epochs);
    assert!(report.epochs.last().unwrap().val_rmse.is_finite());

    let save_dir = std::env::temp_dir().join(format!(
      "stochastic_rs_rbergomi_nn_{}_{}",
      std::process::id(),
      3003_u64
    ));
    if save_dir.exists() {
      let _ = fs::remove_dir_all(&save_dir);
    }
    model.save(&save_dir)?;
    let loaded = RBergomiNn::load(&save_dir, &device)?;

    let sample = [
      params[[0, 0]],
      params[[0, 1]],
      params[[0, 2]],
      params[[0, 3]],
    ];
    let p1 = model.predict_surface(&sample)?;
    let p2 = loaded.predict_surface(&sample)?;
    let max_diff = p1
      .iter()
      .zip(p2.iter())
      .map(|(a, b)| (a - b).abs())
      .max_by(f32::total_cmp)
      .unwrap();
    assert!(max_diff < 1e-4);

    let _ = fs::remove_dir_all(&save_dir);
    Ok(())
  }
}

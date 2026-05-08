//! Neural-network surrogate for the Heston volatility model.
//!
//! The architecture and scaling conventions follow:
//! - 3 hidden ELU layers (width 30), linear output (88 nodes).
//! - Parameter scaling to `[-1, 1]` with fixed lower/upper bounds.
//! - Output (IV surface) standardization with train-set mean/std.
//!
//! Source:
//! - https://github.com/amuguruza/NN-StochVol-Calibrations
//! - `Heston/NNHeston.ipynb`

use std::path::Path;

use anyhow::Result;
use candle_core::Device;
use ndarray::Array2;

use super::common::StochVolModelSpec;
use super::common::StochVolNn;
use super::common::TrainConfig;
use super::common::TrainReport;

pub const MODEL_ID: &str = "heston";
pub const INPUT_DIM: usize = 5;
pub const OUTPUT_DIM: usize = 88;
pub const DEFAULT_HIDDEN_DIM: usize = 30;
pub const PARAM_LB: [f32; INPUT_DIM] = [0.0001, -0.95, 0.01, 0.01, 1.0];
pub const PARAM_UB: [f32; INPUT_DIM] = [0.04, -0.1, 1.0, 0.2, 10.0];

pub struct HestonNn {
  inner: StochVolNn,
}

impl HestonNn {
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

  /// Predict an implied-volatility surface for the given parameter vector.
  ///
  /// Returns a flat `Vec<f32>` of length [`OUTPUT_DIM`] in row-major order
  /// (maturity-major). To bridge into the rest of the library, feed the
  /// result directly into
  /// `stochastic_rs_quant::vol_surface::ImpliedVolSurface::from_flat_iv_grid`
  /// together with the strike / maturity / forward grid the model was
  /// trained on.
  pub fn predict_surface(&self, params: &[f32; INPUT_DIM]) -> Result<Vec<f32>> {
    self.inner.predict_surface(params)
  }

  pub fn predict_surfaces(&self, params: &Array2<f32>) -> Result<Array2<f32>> {
    self.inner.predict_surfaces(params)
  }

  /// Bridge to `stochastic-rs-quant`: predict and package as
  /// [`ImpliedVolSurface`](stochastic_rs_quant::vol_surface::ImpliedVolSurface).
  ///
  /// `strikes`, `maturities`, `forwards` describe the grid the network was
  /// trained on. Available with the `quant` cargo feature.
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

#[cfg(test)]
mod tests {
  use std::fs;
  use std::path::Path;

  use super::*;
  use crate::volatility::common::load_trainset_gzip_npy;
  use crate::volatility::common::rmse_1d;
  use crate::volatility::common::synthetic_surface_dataset;
  use crate::volatility::common::write_surface_fit_plot_html;

  #[test]
  fn train_save_load_roundtrip() -> Result<()> {
    let device = Device::Cpu;
    let (params, surfaces) = synthetic_surface_dataset(&PARAM_LB, &PARAM_UB, 192, OUTPUT_DIM, 7);
    let mut model = HestonNn::new(&device)?;
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
      "stochastic_rs_heston_nn_{}_{}",
      std::process::id(),
      1001_u64
    ));
    if save_dir.exists() {
      let _ = fs::remove_dir_all(&save_dir);
    }
    model.save(&save_dir)?;
    let loaded = HestonNn::load(&save_dir, &device)?;

    let sample = [
      params[[0, 0]],
      params[[0, 1]],
      params[[0, 2]],
      params[[0, 3]],
      params[[0, 4]],
    ];
    let p1 = model.predict_surface(&sample)?;
    let p2 = loaded.predict_surface(&sample)?;
    let max_diff = p1
      .iter()
      .zip(p2.iter())
      .map(|(a, b)| (a - b).abs())
      .fold(0.0_f32, f32::max);
    assert!(max_diff < 1e-4);

    let _ = fs::remove_dir_all(&save_dir);
    Ok(())
  }

  #[cfg(feature = "quant")]
  #[test]
  fn predict_implied_vol_surface_roundtrip() -> Result<()> {
    let device = Device::Cpu;
    let (params, surfaces) = synthetic_surface_dataset(&PARAM_LB, &PARAM_UB, 64, OUTPUT_DIM, 11);
    let mut model = HestonNn::new(&device)?;
    let cfg = TrainConfig {
      epochs: 5,
      ..TrainConfig::default()
    };
    model.train(&params, &surfaces, &cfg)?;

    // 8 maturities × 11 strikes = OUTPUT_DIM 88.
    let strikes: Vec<f64> = (0..11).map(|i| 0.5 + 0.1 * i as f64).collect();
    let maturities: Vec<f64> = vec![0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0];
    let forwards = vec![1.0; maturities.len()];
    let sample = [
      params[[0, 0]],
      params[[0, 1]],
      params[[0, 2]],
      params[[0, 3]],
      params[[0, 4]],
    ];

    let surface =
      model.predict_implied_vol_surface(&sample, strikes.clone(), maturities.clone(), forwards)?;
    assert_eq!(surface.ivs.dim(), (maturities.len(), strikes.len()));
    assert_eq!(surface.strikes.len(), strikes.len());
    assert_eq!(surface.maturities.len(), maturities.len());
    Ok(())
  }

  #[test]
  fn real_trainset_fit_plot() -> Result<()> {
    let trainset_path = Path::new("src/ai/volatility/HestonTrainSet.txt.gz");
    if !trainset_path.exists() {
      return Ok(());
    }

    let device = Device::Cpu;
    let (params, surfaces) =
      load_trainset_gzip_npy(trainset_path, INPUT_DIM, OUTPUT_DIM, Some(8_000))?;

    let mut model = HestonNn::new(&device)?;
    let cfg = TrainConfig {
      test_ratio: 0.15,
      batch_size: 64,
      epochs: 30,
      learning_rate: 1e-3,
      random_seed: 42,
      shuffle: true,
    };
    let report = model.train(&params, &surfaces, &cfg)?;
    let sample_idx = surfaces.nrows() / 3;
    let sample = [
      params[[sample_idx, 0]],
      params[[sample_idx, 1]],
      params[[sample_idx, 2]],
      params[[sample_idx, 3]],
      params[[sample_idx, 4]],
    ];
    let pred = model.predict_surface(&sample)?;
    let actual = surfaces.row(sample_idx).to_vec();
    let fit_rmse = rmse_1d(&actual, &pred)?;

    let out = Path::new("target/nn_fit_plots/heston_fit.html");
    let strikes = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5];
    let maturities = [0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0];
    write_surface_fit_plot_html(
      out,
      &format!(
        "Heston NN Fit - sample {} - RMSE {:.5}",
        sample_idx, fit_rmse
      ),
      &strikes,
      &maturities,
      &actual,
      &pred,
    )?;
    println!(
      "Heston fit plot written to {} (sample_rmse={:.6}, final_val_rmse={:.6})",
      out.display(),
      fit_rmse,
      report.epochs.last().map(|e| e.val_rmse).unwrap_or(f32::NAN)
    );
    assert!(out.exists());
    Ok(())
  }
}

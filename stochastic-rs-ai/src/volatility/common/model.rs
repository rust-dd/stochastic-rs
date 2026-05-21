use std::fs;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use candle_core::DType;
use candle_core::Device;
use candle_core::Tensor;
use candle_nn::AdamW;
use candle_nn::Module;
use candle_nn::Optimizer;
use candle_nn::ParamsAdamW;
use candle_nn::VarBuilder;
use candle_nn::VarMap;
use ndarray::Array2;
use ndarray::Axis;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use super::dataset::train_test_split_indices;
use super::metadata::parse_metadata;
use super::metadata::parse_usize_field;
use super::metadata::parse_vec_field;
use super::metadata::serialize_metadata;
use super::network::FeedForwardNet;
use super::network::array2_to_tensor;
use super::network::model_rmse;
use super::scaler::BoundedScaler;
use super::scaler::StandardScaler;
use super::spec::EpochMetrics;
use super::spec::StochVolModelSpec;
use super::spec::TrainConfig;
use super::spec::TrainReport;

const META_FILE: &str = "metadata.txt";
const WEIGHTS_FILE: &str = "weights.safetensors";

pub struct StochVolNn {
  spec: StochVolModelSpec,
  device: Device,
  varmap: VarMap,
  model: FeedForwardNet,
  param_scaler: BoundedScaler,
  output_scaler: Option<StandardScaler>,
}

impl StochVolNn {
  pub fn new(spec: StochVolModelSpec, device: &Device) -> Result<Self> {
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = FeedForwardNet::new(vs, spec.input_dim, spec.hidden_dim, spec.output_dim)?;
    let param_scaler = BoundedScaler::new(spec.param_lb.clone(), spec.param_ub.clone());
    Ok(Self {
      spec,
      device: device.clone(),
      varmap,
      model,
      param_scaler,
      output_scaler: None,
    })
  }

  pub fn spec(&self) -> &StochVolModelSpec {
    &self.spec
  }

  pub fn train(
    &mut self,
    params: &Array2<f32>,
    surfaces: &Array2<f32>,
    config: &TrainConfig,
  ) -> Result<TrainReport> {
    if params.nrows() != surfaces.nrows() {
      bail!("params and surfaces must have the same number of rows");
    }
    if params.nrows() < 2 {
      bail!("at least two samples are required for train/test split");
    }
    if params.ncols() != self.spec.input_dim {
      bail!("params width must equal input_dim");
    }
    if surfaces.ncols() != self.spec.output_dim {
      bail!("surfaces width must equal output_dim");
    }
    if config.batch_size == 0 {
      bail!("batch_size must be > 0");
    }
    if !(0.0..1.0).contains(&config.test_ratio) {
      bail!("test_ratio must be in [0,1)");
    }

    let (train_idx, test_idx) =
      train_test_split_indices(params.nrows(), config.test_ratio, config.random_seed);
    let x_train = params.select(Axis(0), &train_idx);
    let y_train = surfaces.select(Axis(0), &train_idx);
    let x_test = params.select(Axis(0), &test_idx);
    let y_test = surfaces.select(Axis(0), &test_idx);

    let x_train_scaled = self.param_scaler.scale_array2(&x_train)?;
    let x_test_scaled = self.param_scaler.scale_array2(&x_test)?;
    let output_scaler = StandardScaler::fit(&y_train)?;
    let y_train_scaled = output_scaler.transform(&y_train)?;
    let y_test_scaled = output_scaler.transform(&y_test)?;
    self.output_scaler = Some(output_scaler);

    let optimizer_params = ParamsAdamW {
      lr: config.learning_rate,
      beta1: 0.9,
      beta2: 0.999,
      eps: 1e-7,
      weight_decay: 0.0,
    };
    let mut opt = AdamW::new(self.varmap.all_vars(), optimizer_params)?;

    let mut report = TrainReport { epochs: Vec::new() };
    let mut order: Vec<usize> = (0..x_train_scaled.nrows()).collect();
    let mut rng = StdRng::seed_from_u64(config.random_seed ^ 0xABCD_1234_EF98_7654);

    for epoch in 1..=config.epochs {
      if config.shuffle {
        order.shuffle(&mut rng);
      }

      for start in (0..x_train_scaled.nrows()).step_by(config.batch_size) {
        let end = (start + config.batch_size).min(x_train_scaled.nrows());
        let batch_idx = &order[start..end];

        let xb = x_train_scaled.select(Axis(0), batch_idx);
        let yb = y_train_scaled.select(Axis(0), batch_idx);
        let xb = array2_to_tensor(&xb, &self.device)?;
        let yb = array2_to_tensor(&yb, &self.device)?;

        let pred = self.model.forward(&xb)?;
        let rmse = candle_nn::loss::mse(&pred, &yb)?.sqrt()?;
        opt.backward_step(&rmse)?;
      }

      let train_rmse = model_rmse(&self.model, &x_train_scaled, &y_train_scaled, &self.device)?;
      let val_rmse = model_rmse(&self.model, &x_test_scaled, &y_test_scaled, &self.device)?;
      report.epochs.push(EpochMetrics {
        epoch,
        train_rmse,
        val_rmse,
      });
    }

    Ok(report)
  }

  pub fn predict_surface(&self, params: &[f32]) -> Result<Vec<f32>> {
    let scaler = self
      .output_scaler
      .as_ref()
      .ok_or_else(|| anyhow!("model is not trained or loaded (missing output scaler)"))?;

    let scaled_params = self.param_scaler.scale_vector(params)?;
    let xt = Tensor::from_slice(&scaled_params, (1, self.spec.input_dim), &self.device)?;
    let yt = self.model.forward(&xt)?;
    let y_scaled = yt.to_vec2::<f32>()?;
    let arr = Array2::from_shape_vec(
      (1, self.spec.output_dim),
      y_scaled.into_iter().flatten().collect(),
    )?;
    let arr = scaler.inverse_transform(&arr)?;
    Ok(arr.row(0).to_vec())
  }

  /// Build an [`ImpliedVolSurface`] by running the network on `params` and
  /// reshaping the flat prediction into the standard `(N_T, N_K)` layout.
  ///
  /// The network's `output_dim` must equal `maturities.len() × strikes.len()`,
  /// and the prediction must already be in the IV (sigma) domain — the
  /// surrogates trained on Romano-Touzi data satisfy both.
  ///
  /// `forwards` carries the per-maturity forward used to compute log-moneyness
  /// and total variance inside the surface struct.
  ///
  /// Available with the `quant` cargo feature.
  #[cfg(feature = "quant")]
  pub fn predict_implied_vol_surface(
    &self,
    params: &[f32],
    strikes: Vec<f64>,
    maturities: Vec<f64>,
    forwards: Vec<f64>,
  ) -> Result<stochastic_rs_quant::vol_surface::ImpliedVolSurface> {
    let n_k = strikes.len();
    let n_t = maturities.len();
    if forwards.len() != n_t {
      bail!("forwards length must match maturities");
    }
    if self.spec.output_dim != n_k * n_t {
      bail!(
        "model output_dim {} does not match strikes×maturities = {} × {} = {}",
        self.spec.output_dim,
        n_k,
        n_t,
        n_k * n_t,
      );
    }
    let pred = self.predict_surface(params)?;
    let ivs =
      Array2::<f64>::from_shape_vec((n_t, n_k), pred.into_iter().map(|v| v as f64).collect())?;
    Ok(
      stochastic_rs_quant::vol_surface::ImpliedVolSurface::from_iv_grid(
        strikes, maturities, forwards, ivs,
      ),
    )
  }

  pub fn predict_surfaces(&self, params: &Array2<f32>) -> Result<Array2<f32>> {
    let scaler = self
      .output_scaler
      .as_ref()
      .ok_or_else(|| anyhow!("model is not trained or loaded (missing output scaler)"))?;

    if params.ncols() != self.spec.input_dim {
      bail!("params width must equal input_dim");
    }

    let x_scaled = self.param_scaler.scale_array2(params)?;
    let xt = array2_to_tensor(&x_scaled, &self.device)?;
    let yt = self.model.forward(&xt)?;
    let y_scaled = yt.to_vec2::<f32>()?;
    let mut flat = Vec::with_capacity(params.nrows() * self.spec.output_dim);
    for row in y_scaled {
      flat.extend_from_slice(&row);
    }
    let arr = Array2::from_shape_vec((params.nrows(), self.spec.output_dim), flat)?;
    scaler.inverse_transform(&arr)
  }

  pub fn save<P: AsRef<Path>>(&self, dir: P) -> Result<()> {
    let scaler = self
      .output_scaler
      .as_ref()
      .ok_or_else(|| anyhow!("cannot save an untrained model (missing output scaler)"))?;
    let dir = dir.as_ref();
    fs::create_dir_all(dir).with_context(|| format!("failed to create directory {dir:?}"))?;

    self
      .varmap
      .save(dir.join(WEIGHTS_FILE))
      .with_context(|| format!("failed to save weights to {:?}", dir.join(WEIGHTS_FILE)))?;

    let meta = serialize_metadata(&self.spec, scaler);
    fs::write(dir.join(META_FILE), meta)
      .with_context(|| format!("failed to write metadata to {:?}", dir.join(META_FILE)))?;
    Ok(())
  }

  pub fn load<P: AsRef<Path>>(expected_model_id: &str, dir: P, device: &Device) -> Result<Self> {
    let dir = dir.as_ref();
    let meta_path = dir.join(META_FILE);
    let content = fs::read_to_string(&meta_path)
      .with_context(|| format!("failed to read metadata from {meta_path:?}"))?;
    let parsed = parse_metadata(&content)?;

    let model_id = parsed
      .get("model_id")
      .ok_or_else(|| anyhow!("missing model_id in metadata"))?;
    if model_id != expected_model_id {
      bail!("metadata model_id '{model_id}' does not match expected '{expected_model_id}'");
    }

    let input_dim = parse_usize_field(&parsed, "input_dim")?;
    let output_dim = parse_usize_field(&parsed, "output_dim")?;
    let hidden_dim = parse_usize_field(&parsed, "hidden_dim")?;
    let lb = parse_vec_field(&parsed, "param_lb")?;
    let ub = parse_vec_field(&parsed, "param_ub")?;
    let mean = parse_vec_field(&parsed, "surface_mean")?;
    let std = parse_vec_field(&parsed, "surface_std")?;

    let spec = StochVolModelSpec::new(model_id.clone(), input_dim, output_dim, hidden_dim, lb, ub)?;
    if mean.len() != output_dim || std.len() != output_dim {
      bail!("surface scaler metadata width does not match output_dim");
    }

    let mut model = Self::new(spec, device)?;
    model.output_scaler = Some(StandardScaler { mean, std });
    model
      .varmap
      .load(dir.join(WEIGHTS_FILE))
      .with_context(|| format!("failed to load weights from {:?}", dir.join(WEIGHTS_FILE)))?;
    Ok(model)
  }
}

use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use candle_core::DType;
use candle_core::Device;
use candle_core::Tensor;
use candle_nn::AdamW;
use candle_nn::Linear;
use candle_nn::Module;
use candle_nn::Optimizer;
use candle_nn::ParamsAdamW;
use candle_nn::VarBuilder;
use candle_nn::VarMap;
use candle_nn::linear;
use flate2::read::GzDecoder;
use ndarray::Array2;
use ndarray::Axis;
use ndarray_npy::ReadNpyExt;
use plotly::Layout;
use plotly::Plot;
use plotly::Scatter;
use plotly::common::DashType;
use plotly::common::Line;
use plotly::common::Mode;
use plotly::common::Title;
use plotly::layout::GridPattern;
use plotly::layout::LayoutGrid;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

const META_FILE: &str = "metadata.txt";
const WEIGHTS_FILE: &str = "weights.safetensors";

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

struct FeedForwardNet {
  dense1: Linear,
  dense2: Linear,
  dense3: Linear,
  out: Linear,
}

impl FeedForwardNet {
  fn new(vs: VarBuilder, input_dim: usize, hidden_dim: usize, output_dim: usize) -> Result<Self> {
    let dense1 = linear(input_dim, hidden_dim, vs.pp("dense_1"))?;
    let dense2 = linear(hidden_dim, hidden_dim, vs.pp("dense_2"))?;
    let dense3 = linear(hidden_dim, hidden_dim, vs.pp("dense_3"))?;
    let out = linear(hidden_dim, output_dim, vs.pp("dense_4"))?;
    Ok(Self {
      dense1,
      dense2,
      dense3,
      out,
    })
  }
}

impl Module for FeedForwardNet {
  fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
    let x = self.dense1.forward(xs)?.elu(1.0)?;
    let x = self.dense2.forward(&x)?.elu(1.0)?;
    let x = self.dense3.forward(&x)?.elu(1.0)?;
    self.out.forward(&x)
  }
}

#[derive(Clone, Debug)]
struct BoundedScaler {
  lb: Vec<f32>,
  ub: Vec<f32>,
}

impl BoundedScaler {
  fn new(lb: Vec<f32>, ub: Vec<f32>) -> Self {
    Self { lb, ub }
  }

  fn scale_array2(&self, x: &Array2<f32>) -> Result<Array2<f32>> {
    if x.ncols() != self.lb.len() {
      bail!("input width mismatch for bounded scaling");
    }
    let mut out = x.clone();
    for i in 0..out.nrows() {
      for j in 0..out.ncols() {
        let center = 0.5 * (self.lb[j] + self.ub[j]);
        let half_range = 0.5 * (self.ub[j] - self.lb[j]);
        out[[i, j]] = (x[[i, j]] - center) / half_range;
      }
    }
    Ok(out)
  }

  fn scale_vector(&self, x: &[f32]) -> Result<Vec<f32>> {
    if x.len() != self.lb.len() {
      bail!("input length mismatch for bounded scaling");
    }
    let mut out = vec![0.0_f32; x.len()];
    for i in 0..x.len() {
      let center = 0.5 * (self.lb[i] + self.ub[i]);
      let half_range = 0.5 * (self.ub[i] - self.lb[i]);
      out[i] = (x[i] - center) / half_range;
    }
    Ok(out)
  }
}

#[derive(Clone, Debug)]
struct StandardScaler {
  mean: Vec<f32>,
  std: Vec<f32>,
}

impl StandardScaler {
  fn fit(data: &Array2<f32>) -> Result<Self> {
    if data.nrows() == 0 || data.ncols() == 0 {
      bail!("cannot fit StandardScaler on empty matrix");
    }
    let rows = data.nrows();
    let cols = data.ncols();
    let mut mean = vec![0.0_f32; cols];
    let mut std = vec![0.0_f32; cols];

    for j in 0..cols {
      let mut s = 0.0_f32;
      for i in 0..rows {
        s += data[[i, j]];
      }
      mean[j] = s / rows as f32;
    }

    for j in 0..cols {
      let mut s2 = 0.0_f32;
      for i in 0..rows {
        let d = data[[i, j]] - mean[j];
        s2 += d * d;
      }
      std[j] = (s2 / rows as f32).sqrt().max(1e-6);
    }

    Ok(Self { mean, std })
  }

  fn transform(&self, data: &Array2<f32>) -> Result<Array2<f32>> {
    if data.ncols() != self.mean.len() {
      bail!("input width mismatch for StandardScaler::transform");
    }
    let mut out = data.clone();
    for i in 0..out.nrows() {
      for j in 0..out.ncols() {
        out[[i, j]] = (data[[i, j]] - self.mean[j]) / self.std[j];
      }
    }
    Ok(out)
  }

  fn inverse_transform(&self, data: &Array2<f32>) -> Result<Array2<f32>> {
    if data.ncols() != self.mean.len() {
      bail!("input width mismatch for StandardScaler::inverse_transform");
    }
    let mut out = data.clone();
    for i in 0..out.nrows() {
      for j in 0..out.ncols() {
        out[[i, j]] = data[[i, j]] * self.std[j] + self.mean[j];
      }
    }
    Ok(out)
  }
}

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

fn serialize_metadata(spec: &StochVolModelSpec, scaler: &StandardScaler) -> String {
  let mut out = String::new();
  out.push_str("version=1\n");
  out.push_str(&format!("model_id={}\n", spec.model_id));
  out.push_str(&format!("input_dim={}\n", spec.input_dim));
  out.push_str(&format!("output_dim={}\n", spec.output_dim));
  out.push_str(&format!("hidden_dim={}\n", spec.hidden_dim));
  out.push_str(&format!("param_lb={}\n", join_f32(&spec.param_lb)));
  out.push_str(&format!("param_ub={}\n", join_f32(&spec.param_ub)));
  out.push_str(&format!("surface_mean={}\n", join_f32(&scaler.mean)));
  out.push_str(&format!("surface_std={}\n", join_f32(&scaler.std)));
  out
}

fn parse_metadata(s: &str) -> Result<HashMap<String, String>> {
  let mut out = HashMap::new();
  for line in s.lines() {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
      continue;
    }
    let (k, v) = line
      .split_once('=')
      .ok_or_else(|| anyhow!("invalid metadata line: {line}"))?;
    out.insert(k.trim().to_string(), v.trim().to_string());
  }
  Ok(out)
}

fn parse_usize_field(map: &HashMap<String, String>, key: &str) -> Result<usize> {
  let raw = map
    .get(key)
    .ok_or_else(|| anyhow!("missing '{key}' in metadata"))?;
  raw
    .parse::<usize>()
    .with_context(|| format!("failed to parse metadata field '{key}'"))
}

fn parse_vec_field(map: &HashMap<String, String>, key: &str) -> Result<Vec<f32>> {
  let raw = map
    .get(key)
    .ok_or_else(|| anyhow!("missing '{key}' in metadata"))?;
  if raw.is_empty() {
    return Ok(Vec::new());
  }
  raw
    .split(',')
    .map(|v| {
      v.parse::<f32>()
        .with_context(|| format!("failed to parse a float in metadata field '{key}'"))
    })
    .collect()
}

fn join_f32(values: &[f32]) -> String {
  values
    .iter()
    .map(|v| format!("{v:.9}"))
    .collect::<Vec<String>>()
    .join(",")
}

fn array2_to_tensor(arr: &Array2<f32>, device: &Device) -> Result<Tensor> {
  let slice = arr
    .as_slice()
    .ok_or_else(|| anyhow!("Array2 must be contiguous"))?;
  Ok(Tensor::from_slice(
    slice,
    (arr.nrows(), arr.ncols()),
    device,
  )?)
}

fn model_rmse(
  model: &FeedForwardNet,
  x: &Array2<f32>,
  y: &Array2<f32>,
  device: &Device,
) -> Result<f32> {
  let xt = array2_to_tensor(x, device)?;
  let yt = array2_to_tensor(y, device)?;
  let pred = model.forward(&xt)?;
  let rmse = candle_nn::loss::mse(&pred, &yt)?.sqrt()?;
  Ok(rmse.to_scalar::<f32>()?)
}

fn train_test_split_indices(n: usize, test_ratio: f32, seed: u64) -> (Vec<usize>, Vec<usize>) {
  let mut idx = (0..n).collect::<Vec<usize>>();
  let mut rng = StdRng::seed_from_u64(seed);
  idx.shuffle(&mut rng);

  let mut n_test = ((n as f32) * test_ratio).round() as usize;
  n_test = n_test.clamp(1, n.saturating_sub(1));
  let test = idx[..n_test].to_vec();
  let train = idx[n_test..].to_vec();
  (train, test)
}

pub fn load_trainset_gzip_npy<P: AsRef<Path>>(
  path: P,
  input_dim: usize,
  output_dim: usize,
  max_rows: Option<usize>,
) -> Result<(Array2<f32>, Array2<f32>)> {
  let path = path.as_ref();
  let file =
    File::open(path).with_context(|| format!("failed to open dataset file at {:?}", path))?;
  let decoder = GzDecoder::new(file);
  let data = Array2::<f64>::read_npy(decoder)
    .with_context(|| format!("failed to decode/load npy data from {:?}", path))?;

  let expected_min_cols = input_dim + output_dim;
  if data.ncols() < expected_min_cols {
    bail!(
      "dataset at {:?} has {} columns, expected at least {} (input_dim + output_dim)",
      path,
      data.ncols(),
      expected_min_cols
    );
  }

  let rows = max_rows.unwrap_or(data.nrows()).min(data.nrows());
  if rows < 2 {
    bail!("dataset at {:?} must contain at least 2 rows", path);
  }

  let params_f64 = data.slice(ndarray::s![0..rows, 0..input_dim]).to_owned();
  let surfaces_f64 = data
    .slice(ndarray::s![0..rows, input_dim..(input_dim + output_dim)])
    .to_owned();

  let params = params_f64.mapv(|v| v as f32);
  let surfaces = surfaces_f64.mapv(|v| v as f32);
  Ok((params, surfaces))
}

pub fn rmse_1d(actual: &[f32], predicted: &[f32]) -> Result<f32> {
  if actual.len() != predicted.len() {
    bail!(
      "rmse_1d length mismatch: {} vs {}",
      actual.len(),
      predicted.len()
    );
  }
  if actual.is_empty() {
    bail!("rmse_1d cannot be computed on empty slices");
  }
  let mse = actual
    .iter()
    .zip(predicted.iter())
    .map(|(a, p)| {
      let d = *a - *p;
      d * d
    })
    .sum::<f32>()
    / (actual.len() as f32);
  Ok(mse.sqrt())
}

pub fn write_surface_fit_plot_html<P: AsRef<Path>>(
  output_html: P,
  title: &str,
  strikes: &[f64],
  maturities: &[f64],
  actual_surface: &[f32],
  predicted_surface: &[f32],
) -> Result<()> {
  if strikes.is_empty() || maturities.is_empty() {
    bail!("strikes and maturities must be non-empty");
  }
  let expected = strikes.len() * maturities.len();
  if actual_surface.len() != expected || predicted_surface.len() != expected {
    bail!(
      "surface length mismatch: expected {}, got actual={} predicted={}",
      expected,
      actual_surface.len(),
      predicted_surface.len()
    );
  }

  let rows = maturities.len().div_ceil(2);
  let cols = 2usize;
  let mut plot = Plot::new();

  for (i, maturity) in maturities.iter().enumerate() {
    let start = i * strikes.len();
    let end = start + strikes.len();
    let actual = actual_surface[start..end]
      .iter()
      .map(|v| *v as f64)
      .collect::<Vec<f64>>();
    let pred = predicted_surface[start..end]
      .iter()
      .map(|v| *v as f64)
      .collect::<Vec<f64>>();

    let axis = i + 1;
    let tr_actual = Scatter::new(strikes.to_vec(), actual)
      .name(format!("Actual T={:.2}", maturity))
      .mode(Mode::Lines)
      .line(Line::new().color("#1f77b4"))
      .x_axis(format!("x{axis}"))
      .y_axis(format!("y{axis}"))
      .show_legend(i == 0);
    let tr_pred = Scatter::new(strikes.to_vec(), pred)
      .name(format!("Pred T={:.2}", maturity))
      .mode(Mode::Lines)
      .line(Line::new().color("#d62728").dash(DashType::Dash))
      .x_axis(format!("x{axis}"))
      .y_axis(format!("y{axis}"))
      .show_legend(i == 0);

    plot.add_trace(tr_actual);
    plot.add_trace(tr_pred);
  }

  let layout = Layout::new()
    .height((rows * 360) as usize)
    .width((cols * 520) as usize)
    .title(Title::from(title))
    .grid(
      LayoutGrid::new()
        .rows(rows)
        .columns(cols)
        .pattern(GridPattern::Independent),
    );
  plot.set_layout(layout);

  let output_html = output_html.as_ref();
  if let Some(parent) = output_html.parent() {
    fs::create_dir_all(parent)
      .with_context(|| format!("failed creating plot output directory {:?}", parent))?;
  }
  plot.write_html(output_html);
  Ok(())
}

#[cfg(test)]
pub(crate) fn synthetic_surface_dataset(
  lb: &[f32],
  ub: &[f32],
  samples: usize,
  output_dim: usize,
  seed: u64,
) -> (Array2<f32>, Array2<f32>) {
  use rand::Rng;

  let dim = lb.len();
  let mut rng = StdRng::seed_from_u64(seed);
  let mut params = Array2::<f32>::zeros((samples, dim));
  let mut surfaces = Array2::<f32>::zeros((samples, output_dim));

  for i in 0..samples {
    for j in 0..dim {
      let u = rng.random::<f32>();
      params[[i, j]] = lb[j] + (ub[j] - lb[j]) * u;
    }
    for k in 0..output_dim {
      let mut v = 0.2 + 0.03 * (k as f32) / (output_dim as f32);
      for j in 0..dim {
        let center = 0.5 * (lb[j] + ub[j]);
        let half = 0.5 * (ub[j] - lb[j]);
        let x = (params[[i, j]] - center) / half;
        let w = 0.08 + 0.02 * ((j + 1) as f32);
        v += w * x * ((k as f32 + 1.0) * (j as f32 + 1.0) * 0.11).sin();
        v += 0.03 * x * x / ((j + 1) as f32);
      }
      surfaces[[i, k]] = v;
    }
  }

  (params, surfaces)
}

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear, AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};

/// Calibration model for the Heston model
pub struct Model {
  linear1: Linear,
  linear2: Linear,
  linear3: Linear,
  output_layer: Linear,
}

impl Model {
  #[must_use = "new is necessary to create a new instance of Model"]
  pub fn new(
    vs: VarBuilder,
    input_dim: usize,
    hidden_size: usize,
    output_dim: usize,
  ) -> Result<Self> {
    let linear1 = linear(input_dim, hidden_size, vs.pp("linear-1"))?;
    let linear2 = linear(hidden_size, hidden_size, vs.pp("linear-2"))?;
    let linear3 = linear(hidden_size, hidden_size, vs.pp("linear-3"))?;
    let output_layer = linear(hidden_size, output_dim, vs.pp("linear-4"))?;

    Ok(Self {
      linear1,
      linear2,
      linear3,
      output_layer,
    })
  }
}

impl Module for Model {
  fn forward(&self, xs: &Tensor) -> Result<Tensor> {
    let xs = self.linear1.forward(&xs)?.elu(2.0)?;
    let xs = self.linear2.forward(&xs)?.elu(2.0)?;
    let xs = self.linear3.forward(&xs)?.elu(2.0)?;
    let xs = self.output_layer.forward(&xs)?;
    Ok(xs)
  }
}

pub struct DataSet {
  pub x_train: Tensor,
  pub y_train: Tensor,
  pub x_test: Tensor,
  pub y_test: Tensor,
}

pub fn train(
  dataset: DataSet,
  device: &Device,
  input_dim: usize,
  hidden_size: usize,
  output_dim: usize,
  batch_size: usize,
  epochs: usize,
) -> Result<Model> {
  let x_train = dataset.x_train.to_device(device)?;
  let y_train = dataset.y_train.to_device(device)?;
  let varmap = VarMap::new();
  let vs = VarBuilder::from_varmap(&varmap, DType::F32, device);
  let model = Model::new(vs, input_dim, hidden_size, output_dim)?;
  let optimizer_params = ParamsAdamW {
    lr: 1e-3,
    beta1: 0.9,
    beta2: 0.999,
    eps: 1e-7,
    weight_decay: 0.0,
  };
  let mut adam = AdamW::new(varmap.all_vars(), optimizer_params)?;
  let x_test = dataset.x_test.to_device(device)?;
  let y_test = dataset.y_test.to_device(device)?;

  let num_batches = (x_train.dim(0)? + batch_size - 1) / batch_size;

  for epoch in 1..=epochs {
    for batch_idx in 0..num_batches {
      let start = batch_idx * batch_size;
      let end = ((start + batch_size).min(x_train.dim(0)?)) as usize;
      let current_batch_size = end - start;

      let x_batch = x_train.narrow(0, start, current_batch_size)?;
      let y_batch = y_train.narrow(0, start, current_batch_size)?;

      let logits = model.forward(&x_batch)?;
      let loss = candle_nn::loss::mse(&logits, &y_batch)?;
      adam.backward_step(&loss)?;
    }

    // Compute and print the loss on the entire training and test set
    let logits = model.forward(&x_train)?;
    let loss = candle_nn::loss::mse(&logits, &y_train)?;

    let test_logits = model.forward(&x_test)?;
    let test_loss = candle_nn::loss::mse(&test_logits, &y_test)?;

    println!(
      "Epoch: {epoch:3} Train MSE: {:8.5} Test MSE: {:8.5}",
      loss.to_scalar::<f32>()?,
      test_loss.to_scalar::<f32>()?
    );
  }

  Ok(model)
}

#[cfg(test)]
mod tests {
  use std::fs::File;

  use super::*;
  use candle_core::Device;
  use flate2::read::GzDecoder;
  use ndarray::{s, stack, Array1, Array2, Axis};
  use ndarray_npy::read_npy;
  use plotly::{
    common::{Mode, Title},
    layout::{GridPattern, LayoutGrid},
    Layout, Plot, Scatter,
  };
  use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
  use tempfile::NamedTempFile;

  #[test]
  fn fit_surface() -> Result<()> {
    // Load the dataset
    let file = File::open("src/ai/volatility/HestonTrainSet.txt.gz")?;
    let mut decoder = GzDecoder::new(file);

    let mut temp_file = NamedTempFile::new()?;
    std::io::copy(&mut decoder, &mut temp_file)?;

    let data: Array2<f64> = read_npy(temp_file.path()).unwrap();

    // Corrected data loading to match the reference dataset
    let parameters = data.slice(s![.., 0..5]).to_owned(); // Parameters (12000, 5)
    let implied_vols = data.slice(s![.., 5..]).to_owned(); // Implied volatilities (12000, 88)

    let strikes = Array1::from(vec![0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]);
    let maturities = Array1::from(vec![0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0]);

    // Data splitting consistent with the reference workflow
    let split_data = train_test_split_for_array2(&[&implied_vols, &parameters], 0.15, Some(42));
    let (x_train, x_test) = &split_data[0]; // Implied volatilities
    let (y_train, y_test) = &split_data[1]; // Parameters

    // Scaling parameters using custom scaling
    let ub = [0.04, -0.1, 1.0, 0.2, 10.0];
    let lb = [0.0001, -0.95, 0.01, 0.01, 1.0];

    fn myscale(x: &Array1<f64>, lb: &[f64; 5], ub: &[f64; 5]) -> Array1<f64> {
      Array1::from(
        (0..5)
          .map(|i| (x[i] - (ub[i] + lb[i]) * 0.5) * 2.0 / (ub[i] - lb[i]))
          .collect::<Vec<f64>>(),
      )
    }

    fn myinverse(x: &Array1<f64>, lb: &[f64; 5], ub: &[f64; 5]) -> Array1<f64> {
      Array1::from(
        (0..5)
          .map(|i| x[i] * (ub[i] - lb[i]) * 0.5 + (ub[i] + lb[i]) * 0.5)
          .collect::<Vec<f64>>(),
      )
    }

    fn scale_parameters(array: &Array2<f64>, lb: &[f64; 5], ub: &[f64; 5]) -> Array2<f64> {
      let scaled_rows: Vec<Array1<f64>> = array
        .axis_iter(Axis(0))
        .map(|row| myscale(&row.to_owned(), lb, ub))
        .collect();

      stack(
        Axis(0),
        &scaled_rows.iter().map(|row| row.view()).collect::<Vec<_>>(),
      )
      .expect("Failed to stack arrays")
    }

    let y_train_scaled = scale_parameters(&y_train, &lb, &ub);
    let y_test_scaled = scale_parameters(&y_test, &lb, &ub);

    // Scaling implied volatilities using StandardScaler
    #[derive(Debug)]
    struct StandardScaler2D {
      mean: Array1<f64>,
      std: Array1<f64>,
    }

    impl StandardScaler2D {
      fn fit(data: &Array2<f64>) -> Self {
        let mean = data.mean_axis(Axis(0)).unwrap();
        let mut std = data.std_axis(Axis(0), 0.0);
        // Handle zero standard deviation
        std = std.mapv(|x| if x == 0.0 { 1e-8 } else { x });
        Self { mean, std }
      }

      fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        (data - &self.mean) / &self.std
      }

      fn inverse_transform(&self, data: &Array2<f64>) -> Array2<f64> {
        data * &self.std + &self.mean
      }

      fn transform1d(&self, data: &Array1<f64>) -> Array1<f64> {
        (data - &self.mean) / &self.std
      }

      fn inverse_transform1d(&self, data: &Array1<f64>) -> Array1<f64> {
        data * &self.std + &self.mean
      }
    }

    let scaler = StandardScaler2D::fit(&x_train);
    let x_train_scaled = scaler.transform(&x_train);
    let x_test_scaled = scaler.transform(&x_test);

    // Prepare the dataset
    let dataset = DataSet {
      x_train: Tensor::from_slice(
        y_train_scaled.mapv(|v| v as f32).as_slice().unwrap(),
        (y_train_scaled.nrows(), 5),
        &Device::Cpu,
      )?,
      y_train: Tensor::from_slice(
        x_train_scaled.mapv(|v| v as f32).as_slice().unwrap(),
        (x_train_scaled.nrows(), 88),
        &Device::Cpu,
      )?,
      x_test: Tensor::from_slice(
        y_test_scaled.mapv(|v| v as f32).as_slice().unwrap(),
        (y_test_scaled.nrows(), 5),
        &Device::Cpu,
      )?,
      y_test: Tensor::from_slice(
        x_test_scaled.mapv(|v| v as f32).as_slice().unwrap(),
        (x_test_scaled.nrows(), 88),
        &Device::Cpu,
      )?,
    };

    // Train the model
    let model = train(
      dataset,
      &Device::Cpu,
      5,   // input_dim (parameters)
      30,  // hidden_size
      88,  // output_dim (implied volatilities)
      32,  // batch_size
      200, // epochs
    )?;

    // Sample index for plotting
    let sample_idx: usize = 1250;

    // Get the sample parameters and inverse-transform
    let y_sample = y_test_scaled.slice(s![sample_idx, ..]).to_owned();

    let y_sample_tensor = Tensor::from_slice(
      y_sample.mapv(|v| v as f32).as_slice().unwrap(),
      (1, y_sample.len()),
      &Device::Cpu,
    )?;

    // Predict implied volatilities
    let y_pred_tensor = model.forward(&y_sample_tensor)?;
    let y_pred_vec = y_pred_tensor.get(0)?.to_vec1::<f32>()?;
    let prediction = Array1::from(y_pred_vec).mapv(|v| v as f64);
    let prediction = scaler.inverse_transform1d(&prediction);

    // Get the actual implied volatilities for comparison
    let x_sample = x_test.slice(s![sample_idx, ..]).to_owned();
    let x_sample = scaler.inverse_transform1d(&x_sample);

    // Plot the results
    plot_results_with_plotly(&strikes, &maturities, &x_sample, &prediction)?;

    Ok(())
  }

  fn plot_results_with_plotly(
    strikes: &Array1<f64>,
    maturities: &Array1<f64>,
    actual: &Array1<f64>,
    prediction: &Array1<f64>,
  ) -> Result<()> {
    let strikes_dim = strikes.len();
    let s0 = 1.0;

    let mut plot = Plot::new();

    // Grid dimensions for subplots
    let rows = 4; // Number of rows
    let columns = 4; // Number of columns

    for (i, &maturity) in maturities.iter().enumerate() {
      let start = i * strikes_dim;
      let end = start + strikes_dim;

      let log_moneyness = strikes.iter().map(|&k| (k / s0).ln()).collect::<Vec<f64>>();

      // Slicing the actual and prediction data for the current maturity
      let actual_slice = actual.slice(s![start..end]).to_vec();
      let prediction_slice = prediction.slice(s![start..end]).to_vec();

      // Ensure dimensions match
      assert_eq!(log_moneyness.len(), actual_slice.len());
      assert_eq!(log_moneyness.len(), prediction_slice.len());

      // Blue line for actual data
      let trace_actual = Scatter::new(log_moneyness.clone(), actual_slice)
        .name(format!("Actual - Maturity {:.2}", maturity))
        .mode(Mode::Lines)
        .line(plotly::common::Line::new().color("blue"))
        .x_axis(format!("x{}", i + 1))
        .y_axis(format!("y{}", i + 1))
        .show_legend(false);

      // Red dashed line for predictions
      let trace_prediction = Scatter::new(log_moneyness.clone(), prediction_slice)
        .name(format!("Prediction - Maturity {:.2}", maturity))
        .mode(Mode::Lines)
        .line(
          plotly::common::Line::new()
            .dash(plotly::common::DashType::Dash)
            .color("red"),
        )
        .x_axis(format!("x{}", i + 1))
        .y_axis(format!("y{}", i + 1))
        .show_legend(false);

      plot.add_trace(trace_actual);
      plot.add_trace(trace_prediction);
    }

    // Layout settings for subplots
    let layout = Layout::new()
      .height(1200)
      .grid(
        LayoutGrid::new()
          .rows(rows)
          .columns(columns)
          .pattern(GridPattern::Independent),
      )
      .title(Title::from("Implied Volatility Surface"))
      .x_axis(plotly::layout::Axis::new().title("Log-moneyness"))
      .y_axis(plotly::layout::Axis::new().title("Implied Volatility"));

    plot.set_layout(layout);
    plot.show();

    Ok(())
  }

  // Utility function for train_test_split
  fn train_test_split_for_array2(
    arrays: &[&Array2<f64>],
    test_size: f64,
    random_seed: Option<u64>,
  ) -> Vec<(Array2<f64>, Array2<f64>)> {
    let n_samples = arrays[0].nrows();
    let indices = (0..n_samples).collect::<Vec<usize>>();

    let mut rng = match random_seed {
      Some(seed) => StdRng::seed_from_u64(seed),
      None => StdRng::from_entropy(),
    };

    let mut shuffled_indices = indices.clone();
    shuffled_indices.shuffle(&mut rng);

    let test_size = (n_samples as f64 * test_size).round() as usize;
    let test_indices = &shuffled_indices[..test_size];
    let train_indices = &shuffled_indices[test_size..];

    let mut result = Vec::new();

    for array in arrays {
      let train_array = array.select(Axis(0), train_indices);
      let test_array = array.select(Axis(0), test_indices);
      result.push((train_array, test_array));
    }

    result
  }
}

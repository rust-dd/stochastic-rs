use std::fs::File;
use std::time::Instant;

use candle_core::DType;
use candle_core::Device;
use candle_core::Module;
use candle_core::Result;
use candle_core::Tensor;
use candle_nn::AdamW;
use candle_nn::Dropout;
use candle_nn::LSTM;
use candle_nn::LSTMConfig;
use candle_nn::LayerNorm;
use candle_nn::LayerNormConfig;
use candle_nn::Optimizer;
use candle_nn::PReLU;
use candle_nn::ParamsAdamW;
use candle_nn::RNN;
use candle_nn::Sequential;
use candle_nn::VarBuilder;
use candle_nn::VarMap;
use candle_nn::layer_norm;
use candle_nn::linear;
use candle_nn::loss::mse;
use candle_nn::lstm;
use candle_nn::prelu;
use candle_nn::seq;
use polars::prelude::*;

use super::fou_lstm_datasets::test_vasicek_2_d;

pub struct Model {
  is_train: bool,
  use_dropout: bool,
  dropout: Dropout,
  prelu: PReLU,
  lstm: Vec<LSTM>,
  layer_norm: LayerNorm,
  mlp: Sequential,
}

impl Model {
  #[must_use = "new is necessary to create a new instance of Model"]
  pub fn new(
    vs: VarBuilder,
    lstm_features: usize,
    hidden_dim: usize,
    out_dim: usize,
    num_lstm_layers: Option<usize>,
    use_dropout: Option<bool>,
    droput_rate: Option<f32>,
  ) -> Result<Self> {
    let dropout = Dropout::new(droput_rate.unwrap_or(0.25));
    let prelu = prelu(None, vs.pp("prelu"))?;
    let mut lstm_layers = Vec::with_capacity(num_lstm_layers.unwrap_or(2));
    for i in 0..num_lstm_layers.unwrap_or(2) {
      lstm_layers.push(lstm(
        if i == 0 { lstm_features } else { hidden_dim },
        hidden_dim,
        LSTMConfig {
          layer_idx: i,
          ..Default::default()
        },
        vs.pp(&format!("lstm-{}", i)),
      )?);
    }
    let layer_n = layer_norm(hidden_dim, LayerNormConfig::default(), vs.pp("layer-norm"))?;
    let mlp = seq()
      .add(linear(hidden_dim, hidden_dim, vs.pp("mpl-linear-1"))?)
      .add_fn(|x| x.relu())
      .add(linear(hidden_dim, hidden_dim / 2, vs.pp("mpl-linear-2"))?)
      .add_fn(|x| x.relu())
      .add(linear(hidden_dim / 2, out_dim, vs.pp("mpl-linear-3"))?);

    Ok(Self {
      is_train: true,
      use_dropout: use_dropout.unwrap_or(true),
      dropout,
      prelu,
      lstm: lstm_layers,
      layer_norm: layer_n,
      mlp,
    })
  }

  pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
    let mut x = x.clone();
    x = self.prelu.forward(&x)?;
    if self.use_dropout {
      x = self.dropout.forward(&x, self.is_train)?;
    }
    for lstm in self.lstm.iter() {
      let states = lstm.seq(&x)?;
      x = lstm
        .states_to_tensor(&states)?
        .reshape(&[x.dims()[0], x.dims()[1], 64])?;
    }
    if self.use_dropout {
      x = self.dropout.forward(&x, self.is_train)?;
    }
    x = self.layer_norm.forward(&x)?;
    let out = self.mlp.forward(&x)?;
    Ok(out)
  }

  pub fn eval(&mut self) {
    self.is_train = false;
  }
}

pub fn test() -> anyhow::Result<()> {
  let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
  let varmap = VarMap::new();
  let vs = VarBuilder::from_varmap(&varmap, DType::F64, &device);

  let epochs = 50_usize;
  let epoch_size = 12_800_usize;
  let lstm_features = 2_usize;
  let hidden_dim = 64_usize;
  let out_dim = 1_usize;
  let batch_size = 64;
  let mut net = Model::new(
    vs,
    lstm_features,
    hidden_dim,
    out_dim,
    Some(3),
    Some(false),
    Some(0.25),
  )
  .unwrap();
  let adamw_params = ParamsAdamW {
    lr: 1e-3,
    beta1: 0.9,
    beta2: 0.999,
    eps: 1e-8,
    weight_decay: 0.01,
  };

  let mut opt = AdamW::new(varmap.all_vars(), adamw_params)?;

  let n: usize = 1600_usize;
  let start = Instant::now();

  for epoch in 0..epochs {
    let (batcher, _) = test_vasicek_2_d(epoch_size, batch_size, n, &device)?;

    'inner: for (batch_idx, batch) in batcher.enumerate() {
      match batch {
        Ok((x, target)) => {
          let inp = net.forward(&x)?;
          let loss = mse(&inp.mean(1)?, &target)?;
          opt.backward_step(&loss)?;
          println!(
            "Epoch: {}, Batch: {}, Loss: {:?}",
            epoch + 1,
            batch_idx + 1,
            loss.to_scalar::<f64>()?
          );
        }
        Err(_) => break 'inner,
      }
    }

    println!("Epoch {} took {:?}", epoch + 1, start.elapsed());
  }

  net.eval();

  // test the model
  let (batcher, hursts) = test_vasicek_2_d(epoch_size, batch_size, n, &device)?;
  let mut theta = Vec::with_capacity(epoch_size);
  let mut est_theta = Vec::with_capacity(epoch_size);

  for batch in batcher {
    match batch {
      Ok((x, target)) => {
        let inp = net.forward(&x)?;
        let inp_vec = inp
          .to_vec2::<f64>()?
          .into_iter()
          .flatten()
          .collect::<Vec<_>>();
        let target_vec = target
          .to_vec2::<f64>()?
          .into_iter()
          .flatten()
          .collect::<Vec<_>>();
        theta.push(target_vec);
        est_theta.push(inp_vec);
      }
      Err(_) => break,
    }
  }

  let theta = theta.into_iter().flatten().collect::<Vec<_>>();
  let est_theta = est_theta.into_iter().flatten().collect::<Vec<_>>();

  let mut dataframe = df!(
      "alpha" => theta,
      "est_alpha" => est_theta,
      "hurst" => hursts
  )?;

  let writer = File::create("vasicek_hurst=0.01..0.99_alpha=-0.5..10.0_init=0.0_slice=300.csv")?;
  let mut csv_writer = CsvWriter::new(writer);
  csv_writer.finish(&mut dataframe)?;

  Ok(())
}

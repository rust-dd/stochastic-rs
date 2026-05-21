use anyhow::Result;
use anyhow::anyhow;
use candle_core::Device;
use candle_core::Tensor;
use candle_nn::Linear;
use candle_nn::Module;
use candle_nn::VarBuilder;
use candle_nn::linear;
use ndarray::Array2;

pub(super) struct FeedForwardNet {
  dense1: Linear,
  dense2: Linear,
  dense3: Linear,
  out: Linear,
}

impl FeedForwardNet {
  pub(super) fn new(
    vs: VarBuilder,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
  ) -> Result<Self> {
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

pub(super) fn array2_to_tensor(arr: &Array2<f32>, device: &Device) -> Result<Tensor> {
  let slice = arr
    .as_slice()
    .ok_or_else(|| anyhow!("Array2 must be contiguous"))?;
  Ok(Tensor::from_slice(
    slice,
    (arr.nrows(), arr.ncols()),
    device,
  )?)
}

pub(super) fn model_rmse(
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

use candle_core::Tensor;
// pub use itransformer::ITransformer;

pub mod fou;
pub mod utils;
pub mod volatility;

#[derive(Clone, Debug)]
pub struct DataSet {
  pub x_train: Tensor,
  pub x_test: Tensor,
  pub y_train: Tensor,
  pub y_test: Tensor,
}

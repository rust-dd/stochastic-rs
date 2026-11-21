use ndarray::Array2;
use ndarray::ArrayView2;
use ndarray::Axis;

/// The data column each vine variable reads, which `pdf`, `cdf` and `sample` translate through.
#[derive(Debug, Clone)]
pub(super) struct VineOrder(Vec<usize>);

impl VineOrder {
  pub(super) fn identity(dim: usize) -> Self {
    Self((0..dim).collect())
  }

  /// Panics unless `order` is a permutation of `0..dim`.
  pub(super) fn new(order: Vec<usize>, dim: usize) -> Self {
    let mut seen = vec![false; dim];
    let permutation = order.len() == dim
      && order
        .iter()
        .all(|&j| j < dim && !std::mem::replace(&mut seen[j], true));
    assert!(
      permutation,
      "order must satisfy `order is a permutation of 0..dim`, got order = {order:?}"
    );
    Self(order)
  }

  pub(super) fn as_slice(&self) -> &[usize] {
    &self.0
  }

  pub(super) fn to_vine(&self, x: ArrayView2<f64>) -> Array2<f64> {
    x.select(Axis(1), &self.0)
  }

  pub(super) fn to_data(&self, sample: Array2<f64>) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros(sample.raw_dim());
    for (k, &j) in self.0.iter().enumerate() {
      out.column_mut(j).assign(&sample.column(k));
    }
    out
  }
}

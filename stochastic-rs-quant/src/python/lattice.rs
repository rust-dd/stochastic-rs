use pyo3::prelude::*;

/// Callable / puttable coupon bond on the one-factor Hull–White trinomial
/// tree. The wrapper holds the tree; the bond is described per `price` call.
#[pyclass(name = "HullWhiteCallableBond", unsendable)]
pub struct PyHullWhiteCallableBond {
  tree: crate::lattice::HullWhiteTree<f64>,
}

#[pymethods]
impl PyHullWhiteCallableBond {
  /// Hull–White tree `dr = a(θ − r) dt + σ dW` from `initial_rate`, built
  /// with `steps` levels up to `horizon` (the bond maturity).
  #[new]
  #[pyo3(signature = (initial_rate, mean_reversion, theta, sigma, horizon, steps))]
  fn new(
    initial_rate: f64,
    mean_reversion: f64,
    theta: f64,
    sigma: f64,
    horizon: f64,
    steps: usize,
  ) -> Self {
    let model = crate::lattice::HullWhiteTreeModel::new(initial_rate, mean_reversion, theta, sigma);
    Self {
      tree: crate::lattice::HullWhiteTree::new(model, horizon, steps),
    }
  }

  /// Prices the bond; `calls` / `puts` are `(time, clean price)` lists.
  /// Returns `(price, straight_price, call_value, put_value)`.
  #[pyo3(signature = (face, coupon_rate, coupon_times, calls=None, puts=None))]
  fn price(
    &self,
    face: f64,
    coupon_rate: f64,
    coupon_times: Vec<f64>,
    calls: Option<Vec<(f64, f64)>>,
    puts: Option<Vec<(f64, f64)>>,
  ) -> (f64, f64, f64, f64) {
    let mut spec = crate::lattice::CallableBondSpec::new(face, coupon_rate, coupon_times);
    if let Some(schedule) = calls {
      spec = spec.with_calls(schedule);
    }
    if let Some(schedule) = puts {
      spec = spec.with_puts(schedule);
    }
    let priced = crate::lattice::price_callable_bond(&self.tree.tree, &self.tree.model, &spec);
    (
      priced.price,
      priced.straight_price,
      priced.call_value,
      priced.put_value,
    )
  }

  /// Zero-coupon bond maturing at the tree horizon.
  fn zero_coupon_bond_price(&self) -> f64 {
    self.tree.zero_coupon_bond_price()
  }
}

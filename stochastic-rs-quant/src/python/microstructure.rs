use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass(name = "AlmgrenChrissPlan", unsendable)]
pub struct PyAlmgrenChrissPlan {
  inner: crate::microstructure::almgren_chriss::AlmgrenChrissPlan<f64>,
}

#[pymethods]
impl PyAlmgrenChrissPlan {
  /// Compute the Almgren-Chriss optimal-execution schedule.
  /// `direction`: "sell" (default) or "buy".
  #[new]
  #[pyo3(signature = (
    total_shares, horizon, n_intervals, volatility, gamma, eta, lambda_,
    epsilon=0.0, direction="sell"
  ))]
  fn new(
    total_shares: f64,
    horizon: f64,
    n_intervals: usize,
    volatility: f64,
    gamma: f64,
    eta: f64,
    lambda_: f64,
    epsilon: f64,
    direction: &str,
  ) -> PyResult<Self> {
    use crate::microstructure::almgren_chriss::AlmgrenChrissParams;
    use crate::microstructure::almgren_chriss::ExecutionDirection;
    use crate::microstructure::almgren_chriss::optimal_execution;
    let dir = match direction.to_ascii_lowercase().as_str() {
      "sell" => ExecutionDirection::Sell,
      "buy" => ExecutionDirection::Buy,
      o => {
        return Err(PyValueError::new_err(format!(
          "direction must be 'sell' or 'buy', got '{o}'"
        )));
      }
    };
    let params = AlmgrenChrissParams {
      total_shares,
      direction: dir,
      horizon,
      n_intervals,
      volatility,
      gamma,
      eta,
      epsilon,
      lambda: lambda_,
    };
    Ok(Self {
      inner: optimal_execution(&params),
    })
  }

  fn inventory<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.inventory.clone().into_pyarray(py)
  }
  fn trades<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.trades.clone().into_pyarray(py)
  }
  fn rates<'py>(&self, py: Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<f64>> {
    use numpy::IntoPyArray;
    self.inner.rates.clone().into_pyarray(py)
  }
  #[getter]
  fn kappa(&self) -> f64 {
    self.inner.kappa
  }
  #[getter]
  fn expected_cost(&self) -> f64 {
    self.inner.expected_cost
  }
  #[getter]
  fn variance(&self) -> f64 {
    self.inner.variance
  }
  fn risk_adjusted_cost(&self, lambda: f64) -> f64 {
    self.inner.risk_adjusted_cost(lambda)
  }
}

#[pyclass(name = "KyleEquilibrium", unsendable)]
pub struct PyKyleEquilibrium {
  inner: crate::microstructure::kyle::KyleEquilibrium<f64>,
}

#[pymethods]
impl PyKyleEquilibrium {
  /// Single-period Kyle (1985) equilibrium.
  #[new]
  fn new(prior_variance: f64, noise_variance: f64) -> Self {
    Self {
      inner: crate::microstructure::kyle::single_period_kyle(prior_variance, noise_variance),
    }
  }

  #[getter]
  fn beta(&self) -> f64 {
    self.inner.beta
  }
  /// Kyle's lambda (price-impact coefficient).
  #[getter]
  fn lambda(&self) -> f64 {
    self.inner.lambda
  }
  #[getter]
  fn posterior_variance(&self) -> f64 {
    self.inner.posterior_variance
  }
  #[getter]
  fn expected_profit(&self) -> f64 {
    self.inner.expected_profit
  }
}

#[pyfunction]
#[pyo3(signature = (prior_variance, noise_variance_per_round, n_periods))]
pub fn multi_period_kyle(
  prior_variance: f64,
  noise_variance_per_round: f64,
  n_periods: usize,
) -> Vec<(f64, f64, f64, f64)> {
  let v = crate::microstructure::kyle::multi_period_kyle(
    prior_variance,
    noise_variance_per_round,
    n_periods,
  );
  v.into_iter()
    .map(|e| (e.beta, e.lambda, e.posterior_variance, e.expected_profit))
    .collect()
}

#[pyfunction]
pub fn roll_spread<'py>(prices: numpy::PyReadonlyArray1<'py, f64>) -> f64 {
  crate::microstructure::spread::roll_spread(prices.as_array())
}

#[pyfunction]
pub fn effective_spread<'py>(
  trade_price: numpy::PyReadonlyArray1<'py, f64>,
  mid: numpy::PyReadonlyArray1<'py, f64>,
) -> f64 {
  crate::microstructure::spread::effective_spread(trade_price.as_array(), mid.as_array())
}

#[pyfunction]
pub fn corwin_schultz_spread<'py>(
  high: numpy::PyReadonlyArray1<'py, f64>,
  low: numpy::PyReadonlyArray1<'py, f64>,
) -> f64 {
  crate::microstructure::spread::corwin_schultz_spread(high.as_array(), low.as_array())
}

#[pyfunction]
#[pyo3(signature = (signed_volumes, kernel="powerlaw", g0=1.0, beta=0.5))]
pub fn propagator_price_impact<'py>(
  signed_volumes: numpy::PyReadonlyArray1<'py, f64>,
  kernel: &str,
  g0: f64,
  beta: f64,
) -> PyResult<f64> {
  use crate::microstructure::impact::ImpactKernel;
  let k = match kernel.to_ascii_lowercase().as_str() {
    "powerlaw" | "power_law" => ImpactKernel::<f64>::PowerLaw,
    "exponential" | "exp" => ImpactKernel::<f64>::Exponential,
    o => {
      return Err(PyValueError::new_err(format!(
        "kernel must be 'powerlaw' or 'exponential', got '{o}'"
      )));
    }
  };
  Ok(crate::microstructure::impact::propagator_price_impact(
    signed_volumes.as_array(),
    k,
    g0,
    beta,
  ))
}

/// Limit order book — supports add / cancel / market-order execution.
#[pyclass(name = "OrderBook", unsendable)]
pub struct PyOrderBook {
  inner: crate::order_book::OrderBook,
}

fn parse_side(s: &str) -> PyResult<crate::order_book::Side> {
  match s.to_ascii_lowercase().as_str() {
    "buy" | "b" | "bid" => Ok(crate::order_book::Side::Buy),
    "sell" | "s" | "ask" | "offer" => Ok(crate::order_book::Side::Sell),
    o => Err(PyValueError::new_err(format!(
      "side must be 'buy' or 'sell', got '{o}'"
    ))),
  }
}

#[pymethods]
impl PyOrderBook {
  #[new]
  fn new() -> Self {
    Self {
      inner: crate::order_book::OrderBook::new(),
    }
  }

  /// Add a limit order. Returns `(order_id, [(price, size, taker_id, maker_id)] of immediate fills)`.
  fn add_order(
    &mut self,
    side: &str,
    price: f64,
    size: f64,
  ) -> PyResult<(u64, Vec<(f64, f64, u64, u64)>)> {
    let s = parse_side(side)?;
    let (id, trades) = self.inner.add_order(s, price, size);
    Ok((
      id,
      trades
        .into_iter()
        .map(|t| (t.price, t.size, t.taker_id, t.maker_id))
        .collect(),
    ))
  }

  /// Execute a market order. Returns `(taker_id, [trades], remaining_unfilled_size)`.
  fn execute_order(
    &mut self,
    side: &str,
    size: f64,
  ) -> PyResult<(u64, Vec<(f64, f64, u64, u64)>, f64)> {
    let s = parse_side(side)?;
    let (id, trades, remaining) = self.inner.execute_order(s, size);
    Ok((
      id,
      trades
        .into_iter()
        .map(|t| (t.price, t.size, t.taker_id, t.maker_id))
        .collect(),
      remaining,
    ))
  }

  fn cancel_order(&mut self, id: u64) -> bool {
    self.inner.cancel_order(id)
  }

  fn best_bid(&self) -> Option<(f64, f64)> {
    self.inner.best_bid()
  }
  fn best_ask(&self) -> Option<(f64, f64)> {
    self.inner.best_ask()
  }
  fn mid(&self) -> Option<f64> {
    self.inner.mid()
  }
  fn spread(&self) -> Option<f64> {
    self.inner.spread()
  }
  /// Returns `(bids, asks)` where each side is `[(price, total_size)]`.
  fn depth(&self) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
    self.inner.depth()
  }
}

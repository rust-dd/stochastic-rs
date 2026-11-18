//! # Loss
//!
//! $$
//! L=\frac1n\sum_{i=1}^n \ell(y_i,\hat y_i)
//! $$
//!
/// Mean Absolute Error: MAE = (1/N) Σ|market_i - model_i|
pub fn mae(market: &[f64], model: &[f64]) -> f64 {
  market
    .iter()
    .zip(model.iter())
    .map(|(m, d)| (m - d).abs())
    .sum::<f64>()
    / market.len() as f64
}

/// Mean Squared Error: MSE = (1/N) Σ(market_i - model_i)²
pub fn mse(market: &[f64], model: &[f64]) -> f64 {
  market
    .iter()
    .zip(model.iter())
    .map(|(m, d)| (m - d).powi(2))
    .sum::<f64>()
    / market.len() as f64
}

/// Root Mean Squared Error: RMSE = √MSE
pub fn rmse(market: &[f64], model: &[f64]) -> f64 {
  mse(market, model).sqrt()
}

/// Mean percentage error `(100/N) Σ (market_i − model_i) / market_i`, NaN when any `market_i` is
/// near zero.
pub fn mpe(market: &[f64], model: &[f64]) -> f64 {
  let sum: f64 = market
    .iter()
    .zip(model.iter())
    .map(|(m, d)| {
      if m.abs() < f64::EPSILON {
        f64::NAN
      } else {
        (m - d) / m
      }
    })
    .sum();
  (sum / market.len() as f64) * 100.0
}

/// Mean relative error `(1/N) Σ (model_i − market_i) / market_i`, NaN when any `market_i` is
/// near zero.
pub fn mre(market: &[f64], model: &[f64]) -> f64 {
  let sum: f64 = market
    .iter()
    .zip(model.iter())
    .map(|(m, d)| {
      if m.abs() < f64::EPSILON {
        f64::NAN
      } else {
        (d - m) / m
      }
    })
    .sum();
  sum / market.len() as f64
}

/// Mean Relative Percentage Error (in %): MRPE = MRE × 100. Returns
/// NaN when any `market_i` ≈ 0 (see [`mre`]).
pub fn mrpe(market: &[f64], model: &[f64]) -> f64 {
  mre(market, model) * 100.0
}

/// Mean absolute percentage error `(100/N) Σ |market_i − model_i| / |market_i|`, NaN when any
/// `market_i` is near zero.
pub fn mape(market: &[f64], model: &[f64]) -> f64 {
  let sum: f64 = market
    .iter()
    .zip(model.iter())
    .map(|(m, d)| {
      if m.abs() < f64::EPSILON {
        f64::NAN
      } else {
        (m - d).abs() / m.abs()
      }
    })
    .sum();
  (sum / market.len() as f64) * 100.0
}

/// `(100/N) Σ ((market_i − model_i)/market_i)²`, a hundred times the mean squared relative error
/// and so in neither % nor %²; NaN when any `market_i` is near zero.
pub fn mspe(market: &[f64], model: &[f64]) -> f64 {
  let sum: f64 = market
    .iter()
    .zip(model.iter())
    .map(|(m, d)| {
      if m.abs() < f64::EPSILON {
        f64::NAN
      } else {
        ((m - d) / m).powi(2)
      }
    })
    .sum();
  (sum / market.len() as f64) * 100.0
}

/// Root mean squared percentage error (in %): `100·√((1/N) Σ ((market_i − model_i)/market_i)²)`,
/// NaN when any `market_i` is near zero.
pub fn rmspe(market: &[f64], model: &[f64]) -> f64 {
  (100.0 * mspe(market, model)).sqrt()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn two_prices_each_missed_by_one_percent_read_one() {
    assert_eq!(rmspe(&[100.0, 200.0], &[101.0, 198.0]), 1.0);
  }
}

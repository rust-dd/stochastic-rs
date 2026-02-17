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

/// Mean Percentage Error (in %): MPE = (100/N) Σ[(market_i - model_i) / market_i]
pub fn mpe(market: &[f64], model: &[f64]) -> f64 {
  let sum: f64 = market
    .iter()
    .zip(model.iter())
    .map(|(m, d)| {
      if m.abs() < f64::EPSILON {
        0.0
      } else {
        (m - d) / m
      }
    })
    .sum();
  (sum / market.len() as f64) * 100.0
}

/// Mean Relative Error: MRE = (1/N) Σ[(model_i - market_i) / market_i]
pub fn mre(market: &[f64], model: &[f64]) -> f64 {
  let sum: f64 = market
    .iter()
    .zip(model.iter())
    .map(|(m, d)| {
      if m.abs() < f64::EPSILON {
        0.0
      } else {
        (d - m) / m
      }
    })
    .sum();
  sum / market.len() as f64
}

/// Mean Relative Percentage Error (in %): MRPE = MRE × 100
pub fn mrpe(market: &[f64], model: &[f64]) -> f64 {
  mre(market, model) * 100.0
}

/// Mean Absolute Percentage Error (in %): MAPE = (100/N) Σ[|market_i - model_i| / |market_i|]
pub fn mape(market: &[f64], model: &[f64]) -> f64 {
  let sum: f64 = market
    .iter()
    .zip(model.iter())
    .map(|(m, d)| {
      if m.abs() < f64::EPSILON {
        0.0
      } else {
        (m - d).abs() / m.abs()
      }
    })
    .sum();
  (sum / market.len() as f64) * 100.0
}

/// Mean Squared Percentage Error (in %): MSPE = (100/N) Σ[((market_i - model_i) / market_i)²]
pub fn mspe(market: &[f64], model: &[f64]) -> f64 {
  let sum: f64 = market
    .iter()
    .zip(model.iter())
    .map(|(m, d)| {
      if m.abs() < f64::EPSILON {
        0.0
      } else {
        ((m - d) / m).powi(2)
      }
    })
    .sum();
  (sum / market.len() as f64) * 100.0
}

/// Root Mean Squared Percentage Error (in %): RMSPE = √MSPE
pub fn rmspe(market: &[f64], model: &[f64]) -> f64 {
  mspe(market, model).sqrt()
}
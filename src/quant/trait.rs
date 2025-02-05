use nalgebra::DVector;

use super::OptionType;

/// Pricer trait.
pub trait PricerExt: TimeExt {
  /// Calculate the price of an option.
  fn calculate_call_put(&self) -> (f64, f64) {
    todo!()
  }

  /// Calculate the price
  fn calculate_price(&self) -> f64 {
    todo!()
  }

  /// Derivatives.
  fn derivatives(&self) -> Vec<f64> {
    todo!()
  }

  /// Calculate the implied volatility using the Newton-Raphson method.
  fn implied_volatility(&self, _c_price: f64, _option_type: OptionType) -> f64 {
    todo!()
  }
}

pub trait TimeExt {
  fn tau(&self) -> Option<f64>;

  fn eval(&self) -> chrono::NaiveDate;

  fn expiration(&self) -> chrono::NaiveDate;

  /// Calculate tau in days.
  fn calculate_tau_in_days(&self) -> f64 {
    if let Some(tau) = self.tau() {
      tau * 365.0
    } else {
      let eval = self.eval();
      let expiration = self.expiration();
      let days = expiration.signed_duration_since(eval).num_days();
      days as f64
    }
  }

  /// Use if tau is None and eval and expiration are Some.
  fn calculate_tau_in_years(&self) -> f64 {
    let eval = self.eval();
    let expiration = self.expiration();
    let days = expiration.signed_duration_since(eval).num_days();
    days as f64 / 365.0
  }
}

/// Calibration Error trait.
pub trait CalibrationLossExt {
  /// Calculate the mean absolute error.
  fn mae(&self, c_market: &DVector<f64>, c_model: &DVector<f64>) -> f64 {
    c_market
      .iter()
      .zip(c_model.iter())
      .map(|(mkt, mdl)| (mkt - mdl).abs())
      .sum::<f64>()
      / c_market.len() as f64
  }

  /// Calculate the mean squared error.
  fn mse(&self, c_market: &DVector<f64>, c_model: &DVector<f64>) -> f64 {
    c_market
      .iter()
      .zip(c_model.iter())
      .map(|(mkt, mdl)| (mkt - mdl).powi(2))
      .sum::<f64>()
      / c_market.len() as f64
  }

  /// Calculate the root mean squared error.
  fn rmse(&self, c_market: &DVector<f64>, c_model: &DVector<f64>) -> f64 {
    self.mse(c_market, c_model).sqrt()
  }

  /// Calculate the mean percentage error (MPE) *in %*.
  fn mpe(&self, c_market: &DVector<f64>, c_model: &DVector<f64>) -> f64 {
    let mpe_ratio = c_market
      .iter()
      .zip(c_model.iter())
      .map(|(mkt, mdl)| (mkt - mdl) / mkt)
      .sum::<f64>()
      / c_market.len() as f64;

    mpe_ratio * 100.0
  }

  /// Calculate the mean absolute percentage error (MAPE) *in %*.
  fn mape(&self, c_market: &DVector<f64>, c_model: &DVector<f64>) -> f64 {
    let mae_val = self.mae(c_market, c_model);
    let mape_ratio = mae_val / c_market.mean();
    mape_ratio * 100.0
  }

  /// Calculate the mean squared percentage error (MSPE) *in %*.
  fn mspe(&self, c_market: &DVector<f64>, c_model: &DVector<f64>) -> f64 {
    let mse_val = self.mse(c_market, c_model);
    let mspe_ratio = mse_val / c_market.mean().powi(2);
    mspe_ratio * 100.0
  }

  /// Calculate the root mean squared percentage error (RMSPE) *in %*.
  fn rmspe(&self, c_market: &DVector<f64>, c_model: &DVector<f64>) -> f64 {
    let rmse_val = self.rmse(c_market, c_model);
    let rmspe_ratio = rmse_val / c_market.mean();
    rmspe_ratio * 100.0
  }
}

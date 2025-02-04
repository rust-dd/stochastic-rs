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

/// Error trait.
pub trait LossExt {
  /// Calculate the mean absolute error.
  fn mae(&self, actual: f64) -> f64;

  /// Calculate the mean squared error.
  fn mse(&self, actual: f64) -> f64;

  /// Calculate the root mean squared error.
  fn rmse(&self, actual: f64) -> f64;

  /// Calculate the mean percentage error.
  fn mpe(&self, actual: f64) -> f64;

  /// Calculate the mean absolute percentage error.
  fn mae_percentage(&self, actual: f64) -> f64;

  /// Calculate the mean squared percentage error.
  fn mse_percentage(&self, actual: f64) -> f64;

  /// Calculate the root mean squared percentage error.
  fn rmse_percentage(&self, actual: f64) -> f64;

  /// Calculate the mean percentage error.
  fn mpe_percentage(&self, actual: f64) -> f64;
}

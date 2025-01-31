use super::OptionType;

/// Pricer trait.
pub trait Pricer: Time {
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

pub trait Time {
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

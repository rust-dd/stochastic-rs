use super::OptionType;

/// Pricer trait.
pub trait PricerExt: TimeExt {
  /// Calculate the call and put price.
  fn calculate_call_put(&self) -> (f64, f64);

  /// Calculate the price.
  fn calculate_price(&self) -> f64;

  /// Derivatives (greeks).
  fn derivatives(&self) -> Vec<f64> {
    vec![]
  }

  /// Calculate the implied volatility.
  fn implied_volatility(&self, _c_price: f64, _option_type: OptionType) -> f64 {
    0.0
  }
}

pub trait TimeExt {
  fn tau(&self) -> Option<f64>;

  fn eval(&self) -> Option<chrono::NaiveDate> {
    None
  }

  fn expiration(&self) -> Option<chrono::NaiveDate> {
    None
  }

  /// Return tau directly, or compute it from eval/expiration dates.
  fn tau_or_from_dates(&self) -> f64 {
    if let Some(tau) = self.tau() {
      return tau;
    }
    match (self.eval(), self.expiration()) {
      (Some(e), Some(x)) => x.signed_duration_since(e).num_days() as f64 / 365.0,
      _ => panic!("either tau or both eval and expiration must be set"),
    }
  }

  /// Calculate tau in days.
  fn calculate_tau_in_days(&self) -> f64 {
    self.tau_or_from_dates() * 365.0
  }

  /// Calculate tau in years.
  fn calculate_tau_in_years(&self) -> f64 {
    self.tau_or_from_dates()
  }
}

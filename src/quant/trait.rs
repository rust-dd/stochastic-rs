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

/// A trait providing various calibration error metrics.
pub trait CalibrationLossExt {
  /// Mean Absolute Error (MAE):
  /// MAE = (1 / N) * Σ |market_i - c_model_i|
  fn mae(&self, c_market: &DVector<f64>, c_model: &DVector<f64>) -> f64 {
    c_market
      .iter()
      .zip(c_model.iter())
      .map(|(mkt, mdl)| (mkt - mdl).abs())
      .sum::<f64>()
      / c_market.len() as f64
  }

  /// Mean Squared Error (MSE):
  /// MSE = (1 / N) * Σ (c_market_i - c_model_i)²
  fn mse(&self, c_market: &DVector<f64>, c_model: &DVector<f64>) -> f64 {
    c_market
      .iter()
      .zip(c_model.iter())
      .map(|(mkt, mdl)| (mkt - mdl).powi(2))
      .sum::<f64>()
      / c_market.len() as f64
  }

  /// Root Mean Squared Error (RMSE):
  /// RMSE = √(MSE)
  fn rmse(&self, c_market: &DVector<f64>, c_model: &DVector<f64>) -> f64 {
    self.mse(c_market, c_model).sqrt()
  }

  /// Mean Percentage Error (MPE) in %:
  /// MPE = (100 / N) * Σ [(c_market_i - c_model_i) / c_market_i]
  /// Note: This includes the sign of the difference.
  fn mpe(&self, c_market: &DVector<f64>, c_model: &DVector<f64>) -> f64 {
    let sum_ratio = c_market
      .iter()
      .zip(c_model.iter())
      .map(|(mkt, mdl)| {
        if mkt.abs() < f64::EPSILON {
          // Handle zero or near-zero c_market value
          0.0
        } else {
          (mkt - mdl) / mkt
        }
      })
      .sum::<f64>();

    (sum_ratio / c_market.len() as f64) * 100.0
  }

  /// Mean Relative Error (MRE):
  /// MRE = (1 / N) * Σ [(c_model_i - c_market_i) / c_market_i]
  /// This is similar to MPE but uses (c_model_i - c_market_i) instead of (c_market_i - c_model_i).
  /// Often used without multiplying by 100 (i.e. in plain ratio form).
  fn mre(&self, c_market: &DVector<f64>, c_model: &DVector<f64>) -> f64 {
    let sum_ratio = c_market
      .iter()
      .zip(c_model.iter())
      .map(|(mkt, mdl)| {
        if mkt.abs() < f64::EPSILON {
          0.0
        } else {
          (mdl - mkt) / mkt
        }
      })
      .sum::<f64>();

    sum_ratio / c_market.len() as f64
  }

  /// Mean Relative Percentage Error (MRPE) in %:
  /// MRPE = (100 / N) * Σ [(c_model_i - c_market_i) / c_market_i]
  /// Essentially MRE * 100.
  fn mrpe(&self, c_market: &DVector<f64>, c_model: &DVector<f64>) -> f64 {
    self.mre(c_market, c_model) * 100.0
  }

  /// Mean Absolute Percentage Error (MAPE) in %:
  /// MAPE = (100 / N) * Σ [|c_market_i - c_model_i| / |c_market_i|]
  fn mape(&self, c_market: &DVector<f64>, c_model: &DVector<f64>) -> f64 {
    let sum_ratio = c_market
      .iter()
      .zip(c_model.iter())
      .map(|(mkt, mdl)| {
        if mkt.abs() < f64::EPSILON {
          0.0
        } else {
          (mkt - mdl).abs() / mkt.abs()
        }
      })
      .sum::<f64>();

    (sum_ratio / c_market.len() as f64) * 100.0
  }

  /// Mean Squared Percentage Error (MSPE) in %:
  /// MSPE = (100 / N) * Σ [((c_market_i - c_model_i) / c_market_i)²]
  fn mspe(&self, c_market: &DVector<f64>, c_model: &DVector<f64>) -> f64 {
    let sum_sq_ratio = c_market
      .iter()
      .zip(c_model.iter())
      .map(|(mkt, mdl)| {
        if mkt.abs() < f64::EPSILON {
          0.0
        } else {
          ((mkt - mdl) / mkt).powi(2)
        }
      })
      .sum::<f64>();

    (sum_sq_ratio / c_market.len() as f64) * 100.0
  }

  /// Root Mean Squared Percentage Error (RMSPE) in %:
  /// RMSPE = √(MSPE)
  fn rmspe(&self, c_market: &DVector<f64>, c_model: &DVector<f64>) -> f64 {
    self.mspe(c_market, c_model).sqrt()
  }
}

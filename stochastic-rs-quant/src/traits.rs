//! # Quant traits — pricing, time, model bridging.

pub use stochastic_rs_distributions::traits::DistributionExt;
pub use stochastic_rs_distributions::traits::DistributionSampler;
pub use stochastic_rs_distributions::traits::FloatExt;
pub use stochastic_rs_distributions::traits::Fn1D;
pub use stochastic_rs_distributions::traits::Fn2D;
pub use stochastic_rs_distributions::traits::SimdFloatExt;
pub use stochastic_rs_stochastic::traits::Malliavin2DExt;
pub use stochastic_rs_stochastic::traits::MalliavinExt;
pub use stochastic_rs_stochastic::traits::ProcessExt;

pub use stochastic_rs_copulas::traits::BivariateExt;
#[cfg(feature = "openblas")]
pub use stochastic_rs_copulas::traits::MultivariateExt;
pub use stochastic_rs_copulas::traits::NCopula2DExt;

use crate::OptionType;

pub trait PricerExt: TimeExt {
  fn calculate_call_put(&self) -> (f64, f64);

  fn calculate_price(&self) -> f64;

  fn derivatives(&self) -> Vec<f64> {
    vec![]
  }

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

  fn tau_or_from_dates(&self) -> f64 {
    if let Some(tau) = self.tau() {
      return tau;
    }
    match (self.eval(), self.expiration()) {
      (Some(e), Some(x)) => crate::calendar::DayCountConvention::Actual365Fixed.year_fraction(e, x),
      _ => panic!("either tau or both eval and expiration must be set"),
    }
  }

  /// Compute `tau` using a specific day count convention.
  ///
  /// If `tau` is set explicitly it is returned as-is. Otherwise the year
  /// fraction is derived from `eval` / `expiration` using the given
  /// [`DayCountConvention`](crate::calendar::DayCountConvention).
  fn tau_with_dcc(&self, dcc: crate::calendar::DayCountConvention) -> f64 {
    if let Some(tau) = self.tau() {
      return tau;
    }
    match (self.eval(), self.expiration()) {
      (Some(e), Some(x)) => dcc.year_fraction(e, x),
      _ => panic!("either tau or both eval and expiration must be set"),
    }
  }

  fn calculate_tau_in_days(&self) -> f64 {
    self.tau_or_from_dates() * 365.0
  }

  fn calculate_tau_in_years(&self) -> f64 {
    self.tau_or_from_dates()
  }
}

/// Trait for calibration results that can produce a [`ModelPricer`].
///
/// Every calibration result implementing this trait can be fed directly into
/// the vol-surface pipeline via [`build_surface_from_calibration`]. Uses an
/// associated type so calling code stays in static dispatch — no `&dyn` or
/// `Box<dyn>` in the user-facing surface.
///
/// Fourier-based models (Heston, Bates, Lévy) embed `r` and `q` in their
/// characteristic function. Non-Fourier models (SABR, HSCM) receive `r`/`q`
/// at pricing time and may ignore them here.
///
/// [`build_surface_from_calibration`]: crate::vol_surface::pipeline::build_surface_from_calibration
pub trait ToModel {
  /// Concrete pricer type produced by this calibration result.
  type Model: ModelPricer;
  fn to_model(&self, r: f64, q: f64) -> Self::Model;
}

/// Trait for models that can price European options at arbitrary (K, T) points.
///
/// Unlike [`PricerExt`], which bundles market data and strike into the pricer,
/// `ModelPricer` separates the model from the pricing query. This enables
/// vectorized pricing across strike/maturity grids for calibration and vol
/// surface construction.
pub trait ModelPricer {
  /// Price a European call option.
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64;

  /// Price a European put via put-call parity.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let call = self.price_call(s, k, r, q, tau);
    call - s * (-q * tau).exp() + k * (-r * tau).exp()
  }

  /// Price a call or put.
  fn price_option(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    match option_type {
      OptionType::Call => self.price_call(s, k, r, q, tau),
      OptionType::Put => self.price_put(s, k, r, q, tau),
    }
  }
}

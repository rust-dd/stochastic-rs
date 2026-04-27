//! # Quant traits — pricing, time, model bridging.

pub use stochastic_rs_distributions::traits::DistributionExt;
pub use stochastic_rs_distributions::traits::DistributionSampler;
pub use stochastic_rs_distributions::traits::FloatExt;
pub use stochastic_rs_distributions::traits::Fn1D;
pub use stochastic_rs_distributions::traits::Fn2D;
pub use stochastic_rs_distributions::traits::SimdFloatExt;
pub use stochastic_rs_stochastic::traits::CurveOutput;
pub use stochastic_rs_stochastic::traits::Malliavin2DExt;
pub use stochastic_rs_stochastic::traits::MalliavinExt;
pub use stochastic_rs_stochastic::traits::MultiDimensional;
pub use stochastic_rs_stochastic::traits::OneDimensional;
pub use stochastic_rs_stochastic::traits::ProcessExt;
pub use stochastic_rs_stochastic::traits::TwoDimensional;

pub use stochastic_rs_copulas::traits::BivariateExt;
#[cfg(feature = "openblas")]
pub use stochastic_rs_copulas::traits::MultivariateExt;

use crate::CalibrationLossScore;
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
/// characteristic function. Non-Fourier models (Sabr, HSCM) receive `r`/`q`
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

/// Common interface for calibrators across the quant library.
///
/// Each calibrator declares an [`InitialGuess`](Calibrator::InitialGuess) type
/// (use `()` if no initial guess is supported) and an [`Output`](Calibrator::Output)
/// type that implements [`CalibrationResult`]. The trait gives a single entry
/// point — `calibrate(initial)` — that returns rich diagnostic information
/// (loss / convergence) regardless of the underlying optimiser.
pub trait Calibrator {
  /// Optional initial-guess type.
  type InitialGuess;
  /// Calibration output. Must implement [`CalibrationResult`].
  type Output: CalibrationResult;

  /// Run calibration. Pass `None` to let the calibrator infer an initial guess.
  fn calibrate(&self, initial: Option<Self::InitialGuess>) -> Self::Output;
}

/// Common interface for calibration results.
///
/// Calibrators expose either a structured [`CalibrationLossScore`] (when LM /
/// trust-region pipelines compute the full metric set) or just a scalar
/// `rmse` / `mae` (when the calibrator runs Nelder-Mead or similar). This
/// trait unifies both: `rmse()` is always available, [`loss_score()`](Self::loss_score)
/// returns `Some(&CalibrationLossScore)` when the richer breakdown is computed.
pub trait CalibrationResult {
  /// Root-mean-square pricing error on the calibration grid.
  fn rmse(&self) -> f64;

  /// Whether the underlying optimiser reported convergence.
  fn converged(&self) -> bool;

  /// Structured loss-metric breakdown when available. `None` for
  /// calibrators that only track scalar `rmse`/`mae`.
  fn loss_score(&self) -> Option<&CalibrationLossScore> {
    None
  }
}

/// Common interface for Greeks reporting.
///
/// Pricers expose Greeks via inherent methods today (`BSMPricer::delta`,
/// `CashOrNothingPricer::delta`, …) — this trait gives generic / heterogeneous
/// code a single dispatch point. Only [`delta`](Self::delta) is required;
/// pricers that don't compute the higher-order Greeks return [`f64::NAN`]
/// from the default impls.
///
/// Pricers may have multiple Greek variants (analytical, Malliavin, finite
/// difference) — the trait exposes the canonical form. For Malliavin /
/// pathwise Greeks call the inherent methods (`malliavin_greeks::*::delta`)
/// directly.
pub trait GreeksExt {
  /// Delta — $\partial V / \partial S$.
  fn delta(&self) -> f64;

  /// Gamma — $\partial^2 V / \partial S^2$. Defaults to NaN when not implemented.
  fn gamma(&self) -> f64 {
    f64::NAN
  }

  /// Vega — $\partial V / \partial \sigma$. Defaults to NaN when not implemented.
  fn vega(&self) -> f64 {
    f64::NAN
  }

  /// Theta — $\partial V / \partial t$. Defaults to NaN when not implemented.
  fn theta(&self) -> f64 {
    f64::NAN
  }

  /// Rho — $\partial V / \partial r$. Defaults to NaN when not implemented.
  fn rho(&self) -> f64 {
    f64::NAN
  }
}

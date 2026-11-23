//! European swaptions under Black-76 and Bachelier on the forward swap rate `s`; `s` and the
//! annuity `A = Σ_j D(t_j) δ_j N_j` come from the underlying swap's valuation.
//!
//! $$
//! V_{\mathrm{payer}} = A\,\bigl[s\,\Phi(d_1) - K\,\Phi(d_2)\bigr],\qquad
//! V_{\mathrm{receiver}} = A\,\bigl[K\,\Phi(-d_2) - s\,\Phi(-d_1)\bigr]
//! $$
//!
//! Reference: Brigo & Mercurio (2006), "Interest Rate Models", 2nd ed., §6.7

use chrono::NaiveDate;

use super::caplet::bachelier_forward_caplet;
use super::caplet::bachelier_forward_floorlet;
use super::caplet::black_forward_caplet;
use super::caplet::black_forward_floorlet;
use super::types::SwaptionDirection;
use super::types::SwaptionValuation;
use super::types::VolatilityQuoteKind;
use super::volatility::VolatilityModel;
use crate::calendar::DayCountConvention;
use crate::cashflows::CurveProvider;
use crate::instruments::VanillaInterestRateSwap;
use crate::traits::RealExt;

/// European swaption struck at `strike` on the forward swap rate of `swap`.
#[derive(Debug, Clone)]
pub struct EuropeanSwaption<T: RealExt, V: VolatilityModel<T>> {
  /// Payoff direction (Payer = call, Receiver = put).
  pub direction: SwaptionDirection,
  /// Fixed strike $K$.
  pub strike: T,
  /// Exercise date.
  pub expiry: NaiveDate,
  /// Underlying forward-starting swap.
  pub swap: VanillaInterestRateSwap<T>,
  /// Volatility model supplying $\sigma(s,K,\tau)$.
  pub vol: V,
}

impl<T: RealExt, V: VolatilityModel<T>> EuropeanSwaption<T, V> {
  /// Build a European swaption.
  pub fn new(
    direction: SwaptionDirection,
    strike: T,
    expiry: NaiveDate,
    swap: VanillaInterestRateSwap<T>,
    vol: V,
  ) -> Self {
    Self {
      direction,
      strike,
      expiry,
      swap,
      vol,
    }
  }

  /// Valuation summary.
  pub fn valuation(
    &self,
    valuation_date: NaiveDate,
    expiry_day_count: DayCountConvention,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> SwaptionValuation<T> {
    let swap_valuation = self
      .swap
      .valuation(valuation_date, discount_day_count, curves);
    let annuity = swap_valuation.annuity;
    let forward_swap_rate = swap_valuation.fair_rate;
    let tau = expiry_day_count.year_fraction(valuation_date, self.expiry);
    let sigma = self
      .vol
      .implied_volatility(forward_swap_rate, self.strike, tau);
    let forward_value = compute_forward_value(
      forward_swap_rate,
      self.strike,
      tau,
      sigma,
      self.direction,
      self.vol.quote_kind(),
      self.vol.displacement(),
    );
    let npv = annuity * forward_value;

    SwaptionValuation {
      npv,
      forward_swap_rate,
      annuity,
      tau,
      volatility: sigma,
      volatility_quote: self.vol.quote_kind(),
    }
  }

  /// Present value of the swaption.
  pub fn npv(
    &self,
    valuation_date: NaiveDate,
    expiry_day_count: DayCountConvention,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> T {
    self
      .valuation(valuation_date, expiry_day_count, discount_day_count, curves)
      .npv
  }
}

fn compute_forward_value<T: RealExt>(
  forward: T,
  strike: T,
  tau: T,
  sigma: T,
  direction: SwaptionDirection,
  quote: VolatilityQuoteKind,
  shift: T,
) -> T {
  match (quote, direction) {
    (VolatilityQuoteKind::Lognormal, SwaptionDirection::Payer) => {
      black_forward_caplet(forward + shift, strike + shift, tau, sigma)
    }
    (VolatilityQuoteKind::Lognormal, SwaptionDirection::Receiver) => {
      black_forward_floorlet(forward + shift, strike + shift, tau, sigma)
    }
    (VolatilityQuoteKind::Normal, SwaptionDirection::Payer) => {
      bachelier_forward_caplet(forward, strike, tau, sigma)
    }
    (VolatilityQuoteKind::Normal, SwaptionDirection::Receiver) => {
      bachelier_forward_floorlet(forward, strike, tau, sigma)
    }
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;
  use crate::calendar::Frequency;
  use crate::calendar::ScheduleBuilder;
  use crate::cashflows::FloatingIndex;
  use crate::cashflows::IborIndex;
  use crate::cashflows::RateTenor;
  use crate::curves::DiscountCurve;
  use crate::curves::InterpolationMethod;
  use crate::curves::MultiCurve;
  use crate::instruments::option::volatility::ShiftedSabrVolatility;

  fn d(y: i32, m: u32, day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, day).unwrap()
  }

  /// A 1y × 2y swaption on flat 3 % curves: Black-76 on the forward and strike displaced by `d`,
  /// the forward values by mpmath with Hagan's A.69a written out.
  #[test]
  fn a_shifted_sabr_vol_prices_the_swaption_on_the_displaced_rates() {
    let flat = || {
      DiscountCurve::from_zero_rates(
        array![0.5, 1.0, 2.0, 5.0].view(),
        array![0.03, 0.03, 0.03, 0.03].view(),
        InterpolationMethod::LogLinearOnDiscountFactors,
      )
    };
    let mut curves = MultiCurve::new(flat());
    curves.add_forecast(RateTenor::ThreeMonths, flat());
    let schedule = |frequency| {
      ScheduleBuilder::new(d(2025, 1, 15), d(2027, 1, 15))
        .frequency(frequency)
        .build()
    };
    let index = IborIndex::new(
      "IBOR-3M",
      RateTenor::ThreeMonths,
      DayCountConvention::Actual360,
    );
    let swap = VanillaInterestRateSwap::new(
      SwaptionDirection::Payer,
      &schedule(Frequency::SemiAnnual),
      &schedule(Frequency::Quarterly),
      1e6,
      0.03,
      DayCountConvention::Thirty360,
      FloatingIndex::Ibor(index),
      0.0,
      DayCountConvention::Actual360,
    );
    let vol = ShiftedSabrVolatility::new(0.03_f64, 0.5, -0.2, 0.4, 0.03);
    for (direction, want) in [
      (SwaptionDirection::Payer, 0.0030874084210704613),
      (SwaptionDirection::Receiver, 0.002863156417450957),
    ] {
      let swaption = EuropeanSwaption::new(direction, 0.03, d(2025, 1, 15), swap.clone(), vol);
      let v = swaption.valuation(
        d(2024, 1, 15),
        DayCountConvention::Actual365Fixed,
        DayCountConvention::Actual365Fixed,
        &curves,
      );
      let black = match direction {
        SwaptionDirection::Payer => black_forward_caplet::<f64>,
        SwaptionDirection::Receiver => black_forward_floorlet::<f64>,
      };
      let shifted = black(v.forward_swap_rate + 0.03, 0.06, v.tau, v.volatility);
      assert_eq!(v.npv, v.annuity * shifted, "{direction}");
      let forward_value = v.npv / v.annuity;
      assert!(
        (forward_value - want).abs() <= 1e-12 * want,
        "{direction}: {forward_value}"
      );
    }
  }
}

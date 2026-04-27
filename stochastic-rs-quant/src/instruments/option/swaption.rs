//! European swaption pricing via Black-76 and Bachelier on the forward swap rate.
//!
//! $$
//! V_{\mathrm{payer}} = A\,\bigl[s\,\Phi(d_1) - K\,\Phi(d_2)\bigr],\qquad
//! V_{\mathrm{receiver}} = A\,\bigl[K\,\Phi(-d_2) - s\,\Phi(-d_1)\bigr]
//! $$
//!
//! The forward swap rate $s$ and the fixed-leg annuity $A = \sum_j D(t_j)\,
//! \delta_j\,N_j$ are both read off the existing curve stack via the
//! [`crate::instruments::VanillaInterestRateSwap`] valuation routine, so
//! the Black measure collapses the discounting into $A$.
//!
//! Reference: Brigo & Mercurio, "Interest Rate Models — Theory and Practice",
//! Springer 2nd ed. (2006), §6.7.

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
use crate::traits::FloatExt;

/// European swaption struck at `strike` on the forward swap rate of `swap`.
#[derive(Debug, Clone)]
pub struct EuropeanSwaption<T: FloatExt, V: VolatilityModel<T>> {
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

impl<T: FloatExt, V: VolatilityModel<T>> EuropeanSwaption<T, V> {
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

fn compute_forward_value<T: FloatExt>(
  forward: T,
  strike: T,
  tau: T,
  sigma: T,
  direction: SwaptionDirection,
  quote: VolatilityQuoteKind,
) -> T {
  match (quote, direction) {
    (VolatilityQuoteKind::Lognormal, SwaptionDirection::Payer) => {
      black_forward_caplet(forward, strike, tau, sigma)
    }
    (VolatilityQuoteKind::Lognormal, SwaptionDirection::Receiver) => {
      black_forward_floorlet(forward, strike, tau, sigma)
    }
    (VolatilityQuoteKind::Normal, SwaptionDirection::Payer) => {
      bachelier_forward_caplet(forward, strike, tau, sigma)
    }
    (VolatilityQuoteKind::Normal, SwaptionDirection::Receiver) => {
      bachelier_forward_floorlet(forward, strike, tau, sigma)
    }
  }
}

//! Cap, Floor, and Collar instruments priced as sums of caplets/floorlets.
//!
//! $$
//! \mathrm{Cap} = \sum_{i=1}^{n} N_i\,\alpha_i\,P(t,t_i)\,
//!   \mathrm{BlackCall}(F_i,K,\tau_i,\sigma_i)
//! $$
//!
//! Reference: Hull, "Options, Futures, and Other Derivatives", 11th ed. (2021),
//! §29.2.

use chrono::NaiveDate;

use super::caplet::caplet_price;
use super::types::CapFloorValuation;
use super::types::CollarValuation;
use super::types::InterestRateOptionKind;
use super::volatility::VolatilityModel;
use crate::quant::calendar::DayCountConvention;
use crate::quant::cashflows::Cashflow;
use crate::quant::cashflows::CurveProvider;
use crate::quant::cashflows::Leg;
use crate::quant::cashflows::RateIndex;
use crate::traits::FloatExt;

/// Interest-rate cap struck at `strike` on the floating coupons of `leg`.
#[derive(Debug, Clone)]
pub struct Cap<T: FloatExt, V: VolatilityModel<T>> {
  /// Fixed strike $K$ applied to every caplet.
  pub strike: T,
  /// Floating leg carrying the underlying coupon fixings.
  pub leg: Leg<T>,
  /// Volatility model supplying $\sigma(F_i,K,\tau_i)$.
  pub vol: V,
}

impl<T: FloatExt, V: VolatilityModel<T>> Cap<T, V> {
  /// Construct a cap.
  pub fn new(strike: T, leg: Leg<T>, vol: V) -> Self {
    Self { strike, leg, vol }
  }

  /// Valuation summary.
  pub fn valuation(
    &self,
    valuation_date: NaiveDate,
    expiry_day_count: DayCountConvention,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> CapFloorValuation<T> {
    price_cap_floor(
      &self.leg,
      self.strike,
      &self.vol,
      InterestRateOptionKind::Cap,
      valuation_date,
      expiry_day_count,
      discount_day_count,
      curves,
    )
  }

  /// Present value of the cap.
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

/// Interest-rate floor struck at `strike` on the floating coupons of `leg`.
#[derive(Debug, Clone)]
pub struct Floor<T: FloatExt, V: VolatilityModel<T>> {
  /// Fixed strike $K$ applied to every floorlet.
  pub strike: T,
  /// Floating leg carrying the underlying coupon fixings.
  pub leg: Leg<T>,
  /// Volatility model supplying $\sigma(F_i,K,\tau_i)$.
  pub vol: V,
}

impl<T: FloatExt, V: VolatilityModel<T>> Floor<T, V> {
  /// Construct a floor.
  pub fn new(strike: T, leg: Leg<T>, vol: V) -> Self {
    Self { strike, leg, vol }
  }

  /// Valuation summary.
  pub fn valuation(
    &self,
    valuation_date: NaiveDate,
    expiry_day_count: DayCountConvention,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> CapFloorValuation<T> {
    price_cap_floor(
      &self.leg,
      self.strike,
      &self.vol,
      InterestRateOptionKind::Floor,
      valuation_date,
      expiry_day_count,
      discount_day_count,
      curves,
    )
  }

  /// Present value of the floor.
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

/// Zero-cost collar constructed from a long cap and a short floor.
#[derive(Debug, Clone)]
pub struct Collar<T: FloatExt, VC: VolatilityModel<T>, VF: VolatilityModel<T>> {
  /// Long cap leg.
  pub cap: Cap<T, VC>,
  /// Short floor leg.
  pub floor: Floor<T, VF>,
}

impl<T: FloatExt, VC: VolatilityModel<T>, VF: VolatilityModel<T>> Collar<T, VC, VF> {
  /// Construct a collar from an existing cap and floor.
  pub fn new(cap: Cap<T, VC>, floor: Floor<T, VF>) -> Self {
    Self { cap, floor }
  }

  /// Valuation summary (long cap minus long floor).
  pub fn valuation(
    &self,
    valuation_date: NaiveDate,
    expiry_day_count: DayCountConvention,
    discount_day_count: DayCountConvention,
    curves: &(impl CurveProvider<T> + ?Sized),
  ) -> CollarValuation<T> {
    let cap = self
      .cap
      .valuation(valuation_date, expiry_day_count, discount_day_count, curves);
    let floor = self
      .floor
      .valuation(valuation_date, expiry_day_count, discount_day_count, curves);
    let npv = cap.npv - floor.npv;
    CollarValuation { npv, cap, floor }
  }

  /// Net present value of the collar.
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

#[allow(clippy::too_many_arguments)]
fn price_cap_floor<T: FloatExt, V: VolatilityModel<T> + ?Sized>(
  leg: &Leg<T>,
  strike: T,
  vol: &V,
  kind: InterestRateOptionKind,
  valuation_date: NaiveDate,
  expiry_day_count: DayCountConvention,
  discount_day_count: DayCountConvention,
  curves: &(impl CurveProvider<T> + ?Sized),
) -> CapFloorValuation<T> {
  let mut total_npv = T::zero();
  let mut total_annuity = T::zero();
  let mut caplet_prices = Vec::with_capacity(leg.cashflows().len());
  let mut forward_rates = Vec::with_capacity(leg.cashflows().len());
  let mut accrual_factors = Vec::with_capacity(leg.cashflows().len());

  for cashflow in leg.cashflows() {
    let Cashflow::Floating(coupon) = cashflow else {
      continue;
    };
    if coupon.period.payment_date < valuation_date {
      continue;
    }
    let tau = expiry_day_count.year_fraction(valuation_date, coupon.period.accrual_start);
    let tau_payment = discount_day_count.year_fraction(valuation_date, coupon.period.payment_date);
    let discount_factor = curves.discount_curve().discount_factor(tau_payment);
    let forward = coupon.observed_rate.unwrap_or_else(|| {
      coupon
        .index
        .forward_rate(curves, valuation_date, &coupon.period)
    }) + coupon.spread;
    let annuity = discount_factor * coupon.notional * coupon.period.accrual_factor;
    total_annuity += annuity;

    let caplet = if tau <= T::zero() || coupon.observed_rate.is_some() {
      let payoff = match kind {
        InterestRateOptionKind::Cap => (forward - strike).max(T::zero()),
        InterestRateOptionKind::Floor => (strike - forward).max(T::zero()),
      };
      coupon.notional * coupon.period.accrual_factor * discount_factor * payoff
    } else {
      caplet_price(
        forward,
        strike,
        tau,
        coupon.notional,
        coupon.period.accrual_factor,
        discount_factor,
        vol,
        kind,
      )
    };

    total_npv += caplet;
    caplet_prices.push(caplet);
    forward_rates.push(forward);
    accrual_factors.push(coupon.period.accrual_factor);
  }

  CapFloorValuation {
    npv: total_npv,
    annuity: total_annuity,
    caplet_prices,
    forward_rates,
    accrual_factors,
  }
}

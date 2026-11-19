//! The exotic contracts: each option's terms, apart from the model that prices it and the query.

use ndarray::Array1;

use crate::OptionType;
use crate::pricing::autocallable::KnockInStyle;
use crate::pricing::barrier::BarrierType;
use crate::pricing::compound::CompoundType;

/// A knock-out pays `rebate` when the barrier is hit; a knock-in pays it at expiry if it never
/// knocked in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BarrierOption {
  pub strike: f64,
  pub barrier: f64,
  pub rebate: f64,
  pub barrier_type: BarrierType,
  pub option_type: OptionType,
}

/// A double knock-out, alive while the spot stays strictly between `lower` and `upper`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DoubleBarrierOption {
  pub strike: f64,
  pub lower: f64,
  pub upper: f64,
  pub option_type: OptionType,
}

/// Floating-strike when `strike` is `None`; a running extreme left `None` is the spot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LookbackOption {
  pub strike: Option<f64>,
  pub running_min: Option<f64>,
  pub running_max: Option<f64>,
  pub option_type: OptionType,
}

/// At `choice_time` the holder takes the call or the put at `strike`, both expiring at the query's
/// `tau`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChooserOption {
  pub strike: f64,
  pub choice_time: f64,
}

/// At `choice_time` the holder takes the call or the put, each with its own strike and maturity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComplexChooserOption {
  pub call_strike: f64,
  pub call_tau: f64,
  pub put_strike: f64,
  pub put_tau: f64,
  pub choice_time: f64,
}

/// The strike is `moneyness` times the spot at `start_time`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForwardStartOption {
  pub moneyness: f64,
  pub start_time: f64,
  pub option_type: OptionType,
}

/// The outer option expires at `outer_tau` on the inner one, which expires at the query's `tau`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompoundOption {
  pub outer_strike: f64,
  pub outer_tau: f64,
  pub inner_strike: f64,
  pub compound_type: CompoundType,
}

/// Each of `periods` equal returns is floored and capped locally, and their sum globally.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CliquetOption {
  pub notional: f64,
  pub periods: usize,
  pub local_floor: Option<f64>,
  pub local_cap: Option<f64>,
  pub global_floor: Option<f64>,
  pub global_cap: Option<f64>,
}

/// The long side receives `notional` times the realised variance less `strike_variance`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VarianceSwapTerms {
  pub strike_variance: f64,
  pub notional: f64,
}

/// The barriers are fractions of the spot at inception; `coupon` is paid per period.
#[derive(Debug, Clone, PartialEq)]
pub struct AutocallableNote {
  pub notional: f64,
  pub observation_times: Array1<f64>,
  pub autocall_barrier: f64,
  pub coupon_barrier: f64,
  pub coupon: f64,
  pub knock_in_barrier: f64,
  pub knock_in_style: KnockInStyle,
  /// Missed coupons accumulate and are paid on the next date that pays one.
  pub memory: bool,
}

/// The volatility the closed-form exotic pricers divide by.
#[track_caller]
pub(crate) fn assert_sigma(sigma: f64) {
  assert!(
    sigma.is_finite() && sigma > 0.0,
    "sigma must satisfy `sigma.is_finite() && sigma > 0.0`, got sigma = {sigma:?}"
  );
}

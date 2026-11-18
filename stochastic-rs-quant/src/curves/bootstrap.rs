//! Yield curve bootstrapping: deposits at the short end, FRAs and futures in the middle, swaps
//! at the long end.
//!
//! $$
//! D(t) = \frac{1}{1 + r\,\delta},\qquad D(t_2) = \frac{D(t_1)}{1 + R\,\delta},\qquad
//! D(t_n) = \frac{1 - S\sum_{i=1}^{n-1}\delta_i\,D(t_i)}{1 + S\,\delta_n}
//! $$
//!
//! Reference: Hagan & West (2006), "Methods for Constructing a Yield Curve", Wilmott Magazine
//! Reference: Ametrano & Bianchetti (2013), SSRN 2219548

mod spec;
mod validate;

pub use spec::BootstrapSpec;
pub use spec::IterativeOptions;

use super::discount_curve::DiscountCurve;
use super::dual_curve;
use super::interpolation::interpolate_discount_factor;
use super::types::BootstrapInstrument;
use super::types::CurvePoint;
use super::types::InterpolationMethod;
use crate::QuantError;
use crate::total_order::ascending;
use crate::traits::RealExt;

/// A curve with one pillar per instrument, by increasing maturity. `InsufficientData` for no
/// instrument, `InvalidQuote` for one that gives no pillar, `Optimizer` for a solve with no root.
pub fn bootstrap<T: RealExt>(spec: BootstrapSpec<'_, T>) -> Result<DiscountCurve<T>, QuantError> {
  let BootstrapSpec {
    instruments,
    method,
    iterative,
    discount,
  } = spec;
  if instruments.is_empty() {
    return Err(QuantError::InsufficientData {
      what: "instruments",
      needed: 1,
      got: 0,
    });
  }
  for (index, instrument) in instruments.iter().enumerate() {
    validate::check(index, instrument, discount.is_some())?;
  }
  let mut order: Vec<usize> = (0..instruments.len()).collect();
  order.sort_by(|&a, &b| ascending(&instruments[a].maturity(), &instruments[b].maturity()));
  if let Some(pair) = order
    .windows(2)
    .find(|w| instruments[w[0]].maturity() == instruments[w[1]].maturity())
  {
    return Err(QuantError::InvalidQuote {
      index: pair[1],
      reason: format!(
        "maturity must differ from quote {}'s, got maturity = {:?} for both",
        pair[0],
        instruments[pair[1]].maturity()
      ),
    });
  }

  let mut points: Vec<CurvePoint<T>> = Vec::with_capacity(order.len());
  for index in order {
    let read = |t: T| pillar_read(&points, t, method, index);
    let instrument = &instruments[index];
    let discount_factor = match instrument {
      BootstrapInstrument::Deposit { maturity, rate } => T::one() / (T::one() + *rate * *maturity),
      BootstrapInstrument::Fra { start, end, rate } => {
        let delta = *end - *start;
        read(*start)? / (T::one() + *rate * delta)
      }
      BootstrapInstrument::Future {
        start,
        end,
        price,
        sigma,
      } => {
        let delta = *end - *start;
        let hundred = T::from_f64_fast(100.0);
        let futures_rate = (hundred - *price) / hundred;
        let convexity_adj = T::from_f64_fast(0.5) * *sigma * *sigma * *start * *end;
        read(*start)? / (T::one() + (futures_rate - convexity_adj) * delta)
      }
      BootstrapInstrument::Swap {
        maturity,
        rate,
        frequency,
      } => {
        let delta = T::one() / T::from_f64_fast(f64::from(frequency.periods_per_year()));
        let payments = validate::whole_payments(index, *maturity, *frequency)?;
        let mut annuity = T::zero();
        for i in 1..payments {
          annuity += delta * read(T::from_f64_fast(i as f64) * delta)?;
        }
        let direct = (T::one() - *rate * annuity) / (T::one() + *rate * delta);
        match iterative {
          Some(options) if !(direct > T::zero() && direct < T::one()) => {
            bisect_swap_df(annuity, delta, *rate, options, index)?
          }
          _ => direct,
        }
      }
      BootstrapInstrument::SwapWithSchedule {
        rate,
        fixed_times,
        float_times,
      } => match discount {
        Some(discount) => dual_curve::swap_node(
          &points,
          *rate,
          fixed_times,
          float_times,
          discount,
          method,
          index,
        )?,
        None => {
          let (mut annuity, mut prev) = (T::zero(), T::zero());
          for t in &fixed_times[..fixed_times.len() - 1] {
            annuity += (*t - prev) * read(*t)?;
            prev = *t;
          }
          (T::one() - *rate * annuity) / (T::one() + *rate * (instrument.maturity() - prev))
        }
      },
    };
    if !(discount_factor.is_finite() && discount_factor > T::zero()) {
      return Err(QuantError::InvalidQuote {
        index,
        reason: format!(
          "the quote implies a discount factor that is not finite and positive, got {discount_factor:?}"
        ),
      });
    }
    points.push(CurvePoint {
      time: instrument.maturity(),
      discount_factor,
    });
  }
  Ok(DiscountCurve::new(points, method))
}

/// The curve built so far at `t`; `InvalidQuote` when `t > 0` is read before any pillar exists.
fn pillar_read<T: RealExt>(
  points: &[CurvePoint<T>],
  t: T,
  method: InterpolationMethod,
  index: usize,
) -> Result<T, QuantError> {
  if t == T::zero() {
    return Ok(T::one());
  }
  if points.is_empty() {
    return Err(QuantError::InvalidQuote {
      index,
      reason: format!("no earlier pillar gives the discount factor at t = {t:?}"),
    });
  }
  Ok(interpolate_discount_factor(points, t, method))
}

/// The last discount factor in `(1e-6, 1)` that reprices `swap_rate`; `Optimizer` if none does.
fn bisect_swap_df<T: RealExt>(
  annuity_known: T,
  delta: T,
  swap_rate: T,
  options: IterativeOptions<T>,
  index: usize,
) -> Result<T, QuantError> {
  let mut lo = T::from_f64_fast(1e-6);
  let mut hi = T::one();
  let half = T::from_f64_fast(0.5);
  for _ in 0..options.max_iterations {
    let mid = half * (lo + hi);
    let err = (T::one() - mid) / (annuity_known + delta * mid) - swap_rate;
    if err.abs() < options.tolerance {
      return Ok(mid);
    }
    if err > T::zero() {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  Err(QuantError::Optimizer {
    what: "bootstrap",
    reason: format!(
      "no discount factor in (1e-6, 1) reprices swap quote {index} within {:?} in {} bisections",
      options.tolerance, options.max_iterations
    ),
  })
}

#[cfg(test)]
mod tests;

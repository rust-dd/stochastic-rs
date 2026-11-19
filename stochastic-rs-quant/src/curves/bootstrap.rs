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

use stochastic_rs_distributions::brent;

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
    // A read past the last pillar is its flat zero rate, which the new node then replaces: such a
    // node is solved on the curve that holds it, so that the built curve reprices the quote.
    let past_last = |t: T| points.last().is_some_and(|last| t > last.time);
    let on_curve = |guess: T, residual: &dyn Fn(&dyn Fn(T) -> T, T) -> T| {
      consistent_node(
        &points,
        instruments[index].maturity(),
        method,
        guess,
        index,
        residual,
      )
    };
    let instrument = &instruments[index];
    let discount_factor = match instrument {
      BootstrapInstrument::Deposit { maturity, rate } => T::one() / (T::one() + *rate * *maturity),
      BootstrapInstrument::Fra { start, end, rate } => {
        let growth = T::one() + *rate * (*end - *start);
        let direct = read(*start)? / growth;
        if past_last(*start) {
          on_curve(direct, &|curve, node| curve(*start) / growth - node)?
        } else {
          direct
        }
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
        let growth = T::one() + (futures_rate - convexity_adj) * delta;
        let direct = read(*start)? / growth;
        if past_last(*start) {
          on_curve(direct, &|curve, node| curve(*start) / growth - node)?
        } else {
          direct
        }
      }
      BootstrapInstrument::Swap {
        maturity,
        rate,
        frequency,
      } => {
        let delta = T::one() / T::from_f64_fast(f64::from(frequency.periods_per_year()));
        let payments = validate::whole_payments(index, *maturity, *frequency)?;
        let payment = |i: usize| T::from_f64_fast(i as f64) * delta;
        let mut annuity = T::zero();
        for i in 1..payments {
          annuity += delta * read(payment(i))?;
        }
        let direct = (T::one() - *rate * annuity) / (T::one() + *rate * delta);
        if payments > 1 && past_last(payment(payments - 1)) {
          let annuity_with = |curve: &dyn Fn(T) -> T, node: T| {
            (1..payments).fold(delta * node, |sum, i| sum + delta * curve(payment(i)))
          };
          let node = on_curve(direct, &|curve, node| {
            T::one() - node - *rate * annuity_with(curve, node)
          })?;
          match iterative {
            Some(options) if !(node > T::zero() && node < T::one()) => {
              let trial = |node: T| with_node(&points, *maturity, node);
              let annuity = |node: T| {
                let trial = trial(node);
                annuity_with(&|t| interpolate_discount_factor(&trial, t, method), node)
              };
              bisect_swap_df(annuity, *rate, options, index)?
            }
            _ => node,
          }
        } else {
          match iterative {
            Some(options) if !(direct > T::zero() && direct < T::one()) => {
              bisect_swap_df(|node| annuity + delta * node, *rate, options, index)?
            }
            _ => direct,
          }
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
          let earlier = &fixed_times[..fixed_times.len() - 1];
          let (mut annuity, mut prev) = (T::zero(), T::zero());
          for t in earlier {
            annuity += (*t - prev) * read(*t)?;
            prev = *t;
          }
          let last_accrual = instrument.maturity() - prev;
          let direct = (T::one() - *rate * annuity) / (T::one() + *rate * last_accrual);
          if earlier.last().is_some_and(|t| past_last(*t)) {
            on_curve(direct, &|curve, node| {
              let (mut annuity, mut prev) = (last_accrual * node, T::zero());
              for t in earlier {
                annuity += (*t - prev) * curve(*t);
                prev = *t;
              }
              T::one() - node - *rate * annuity
            })?
          } else {
            direct
          }
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

/// `points` with a trial node appended at `maturity`.
fn with_node<T: RealExt>(points: &[CurvePoint<T>], maturity: T, node: T) -> Vec<CurvePoint<T>> {
  let mut trial = points.to_vec();
  trial.push(CurvePoint {
    time: maturity,
    discount_factor: node,
  });
  trial
}

/// The node at `maturity` where `residual(curve, node)` is zero, `curve` holding the node, by
/// Brent's `zero` on a bracket doubled around `guess`; `Optimizer` when none holds a sign change.
fn consistent_node<T: RealExt>(
  points: &[CurvePoint<T>],
  maturity: T,
  method: InterpolationMethod,
  guess: T,
  index: usize,
  residual: &dyn Fn(&dyn Fn(T) -> T, T) -> T,
) -> Result<T, QuantError> {
  let mut trial = with_node(points, maturity, guess);
  let slot = trial.len() - 1;
  let mut at = |node: f64| {
    let node = T::from_f64_fast(node);
    trial[slot].discount_factor = node;
    let value = residual(&|t| interpolate_discount_factor(&trial, t, method), node);
    value.to_f64().unwrap_or(f64::NAN)
  };
  let previous = points.last().map_or_else(T::one, |p| p.discount_factor);
  let centre = if guess.is_finite() && guess > T::zero() {
    guess
  } else {
    previous
  };
  let centre = centre.to_f64().unwrap_or(f64::NAN);
  (1..=60)
    .find_map(|k| {
      let width = 2.0_f64.powi(k);
      brent::zero(centre / width, centre * width, f64::MIN_POSITIVE, &mut at)
    })
    .map(T::from_f64_fast)
    .ok_or_else(|| QuantError::Optimizer {
      what: "bootstrap",
      reason: format!("no discount factor reprices quote {index} on the curve that holds its node"),
    })
}

/// The last discount factor in `(1e-6, 1)` that reprices `swap_rate` on the fixed leg's
/// `annuity` at that factor; `Optimizer` if none does.
fn bisect_swap_df<T: RealExt>(
  annuity: impl Fn(T) -> T,
  swap_rate: T,
  options: IterativeOptions<T>,
  index: usize,
) -> Result<T, QuantError> {
  let mut lo = T::from_f64_fast(1e-6);
  let mut hi = T::one();
  let half = T::from_f64_fast(0.5);
  for _ in 0..options.max_iterations {
    let mid = half * (lo + hi);
    let err = (T::one() - mid) / annuity(mid) - swap_rate;
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

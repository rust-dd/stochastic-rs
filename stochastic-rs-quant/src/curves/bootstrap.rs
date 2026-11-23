//! Yield curve bootstrapping: deposits at the short end, FRAs and futures in the middle, swaps
//! at the long end.
//!
//! $$
//! D(t) = \frac{1}{1 + r\,\delta},\qquad D(t_2) = \frac{D(t_1)}{1 + R\,\delta},\qquad
//! D(t_n) = \frac{1 - S\sum_{i=1}^{n-1}\delta_i\,D(t_i)}{1 + S\,\delta_n}
//! $$
//!
//! Reference: Hagan & West (2008), Wilmott Mag., May, 70–81
//! Reference: Ametrano & Bianchetti (2013), SSRN 2219548

mod segment;
mod solve;
mod spec;
mod sweep;
mod validate;

pub use spec::BootstrapSpec;

pub(super) use solve::trial_node;
pub(crate) use validate::whole_periods;

use self::segment::Segment;
use super::discount_curve::DiscountCurve;
use super::dual_curve;
use super::interpolation::interpolate_discount_factor;
use super::types::BootstrapInstrument;
use super::types::CurvePoint;
use super::types::InterpolationMethod;
use crate::QuantError;
use crate::traits::RealExt;

/// A curve with one pillar per instrument that reprices every quote. `InsufficientData` for no
/// instrument, `InvalidQuote` for one that gives no pillar, `Optimizer` when its nodes don't solve.
pub fn bootstrap<T: RealExt>(spec: BootstrapSpec<'_, T>) -> Result<DiscountCurve<T>, QuantError> {
  bootstrap_numbered(spec, None)
}

/// [`bootstrap`], each error naming the instrument at `i` as quote `numbers[i]`.
pub(crate) fn bootstrap_numbered<T: RealExt>(
  spec: BootstrapSpec<'_, T>,
  numbers: Option<&[usize]>,
) -> Result<DiscountCurve<T>, QuantError> {
  let BootstrapSpec {
    instruments,
    method,
    discount,
  } = spec;
  let number = |index: usize| numbers.map_or(index, |numbers| numbers[index]);
  if instruments.is_empty() {
    return Err(QuantError::InsufficientData {
      what: "instruments",
      needed: 1,
      got: 0,
    });
  }
  for (index, instrument) in instruments.iter().enumerate() {
    validate::check(number(index), instrument, discount.is_some())?;
  }
  let mut order: Vec<usize> = (0..instruments.len()).collect();
  order.sort_by(|&a, &b| {
    instruments[a]
      .maturity()
      .nan_last(&instruments[b].maturity())
  });
  if let Some(pair) = order
    .windows(2)
    .find(|w| instruments[w[0]].maturity() == instruments[w[1]].maturity())
  {
    return Err(QuantError::InvalidQuote {
      index: number(pair[1]),
      reason: format!(
        "maturity must differ from quote {}'s, got maturity = {:?} for both",
        number(pair[0]),
        instruments[pair[1]].maturity()
      ),
    });
  }

  let quotes: Vec<(&BootstrapInstrument<T>, usize)> = order
    .iter()
    .map(|&index| (&instruments[index], number(index)))
    .collect();
  let mut points: Vec<CurvePoint<T>> = Vec::with_capacity(quotes.len());
  for &(instrument, quote) in &quotes {
    let discount_factor = quote_node(Place::Next(&points), instrument, discount, method, quote)?;
    points.push(CurvePoint {
      time: instrument.maturity(),
      discount_factor,
    });
  }
  if !segment::is_local(method) {
    sweep::reprice_together(&mut points, &quotes, discount, method)?;
  }
  Ok(DiscountCurve::new(points, method))
}

/// Where a quote's node is solved: past the pillars built so far, or in its slot of the whole
/// curve, every other pillar held.
#[derive(Clone, Copy)]
pub(in crate::curves) enum Place<'a, T: RealExt> {
  Next(&'a [CurvePoint<T>]),
  At(&'a [CurvePoint<T>], usize),
}

/// The node that reprices `instrument` at `place`; `InvalidQuote` unless it is finite and positive.
fn quote_node<T: RealExt>(
  place: Place<'_, T>,
  instrument: &BootstrapInstrument<T>,
  discount: Option<&DiscountCurve<T>>,
  method: InterpolationMethod,
  quote: usize,
) -> Result<T, QuantError> {
  let discount_factor = match (instrument, discount) {
    (
      BootstrapInstrument::SwapWithSchedule {
        rate,
        fixed_times,
        float_times,
      },
      Some(discount),
    ) => dual_curve::swap_node(
      place,
      *rate,
      fixed_times,
      float_times,
      discount,
      method,
      quote,
    )?,
    _ => single_curve_node(place, instrument, method, quote)?,
  };
  if discount_factor.is_finite() && discount_factor > T::zero() {
    Ok(discount_factor)
  } else {
    Err(QuantError::InvalidQuote {
      index: quote,
      reason: format!(
        "the quote implies a discount factor that is not finite and positive, got {discount_factor:?}"
      ),
    })
  }
}

/// A quote's equation in the discount factor `x` at its maturity,
/// `D(start) − x = rate·(Σⱼ aⱼD(tⱼ) + accrual·x)`, on the coupons `(tⱼ, aⱼ)` paid before it.
struct Equation<T, C> {
  start: T,
  rate: T,
  accrual: T,
  coupons: C,
}

impl<T: RealExt, C: Iterator<Item = (T, T)> + Clone> Equation<T, C> {
  /// `D(start) − rate·Σⱼ aⱼD(tⱼ)`: linear in the reads, so their slopes in `x` give its own.
  fn value(&self, start: T, annuity: T) -> T {
    start - self.rate * annuity
  }

  fn slope(&self) -> T {
    T::one() + self.rate * self.accrual
  }
}

/// The equation each single-curve quote prices, solved for its pillar.
fn single_curve_node<T: RealExt>(
  place: Place<'_, T>,
  instrument: &BootstrapInstrument<T>,
  method: InterpolationMethod,
  quote: usize,
) -> Result<T, QuantError> {
  let maturity = instrument.maturity();
  let single = |start: T, rate: T, accrual: T| Equation {
    start,
    rate,
    accrual,
    coupons: std::iter::empty(),
  };
  match instrument {
    BootstrapInstrument::Deposit { rate, .. } => {
      let equation = single(T::zero(), *rate, maturity);
      node(place, maturity, method, quote, equation)
    }
    BootstrapInstrument::Fra { start, end, rate } => {
      let equation = single(*start, *rate, *end - *start);
      node(place, maturity, method, quote, equation)
    }
    BootstrapInstrument::Future {
      start,
      end,
      price,
      sigma,
    } => {
      let hundred = T::from_f64_fast(100.0);
      let futures_rate = (hundred - *price) / hundred;
      let convexity_adj = T::from_f64_fast(0.5) * *sigma * *sigma * *start * *end;
      let equation = single(*start, futures_rate - convexity_adj, *end - *start);
      node(place, maturity, method, quote, equation)
    }
    BootstrapInstrument::Swap {
      rate, frequency, ..
    } => {
      let delta = T::one() / T::from_f64_fast(f64::from(frequency.periods_per_year()));
      let payments = validate::whole_payments(quote, maturity, *frequency)?;
      let equation = Equation {
        start: T::zero(),
        rate: *rate,
        accrual: delta,
        coupons: (1..payments).map(move |i| (T::from_f64_fast(i as f64) * delta, delta)),
      };
      node(place, maturity, method, quote, equation)
    }
    BootstrapInstrument::SwapWithSchedule {
      rate, fixed_times, ..
    } => {
      let earlier = &fixed_times[..fixed_times.len() - 1];
      let previous = |i: usize| if i == 0 { T::zero() } else { earlier[i - 1] };
      let equation = Equation {
        start: T::zero(),
        rate: *rate,
        accrual: maturity - earlier.last().copied().unwrap_or_else(T::zero),
        coupons: (0..earlier.len()).map(move |i| (earlier[i], earlier[i] - previous(i))),
      };
      node(place, maturity, method, quote, equation)
    }
  }
}

/// The node at `maturity` that solves `equation`: its direct formula where every read sits at or
/// before the last pillar, else its root on the curve that holds it, from the direct value.
fn node<T: RealExt, C: Iterator<Item = (T, T)> + Clone>(
  place: Place<'_, T>,
  maturity: T,
  method: InterpolationMethod,
  quote: usize,
  equation: Equation<T, C>,
) -> Result<T, QuantError> {
  let points = match place {
    Place::Next(points) => points,
    Place::At(curve, slot) => return node_at(curve, slot, method, quote, equation),
  };
  let read = |t: T| pillar_read(points, t, method, quote);
  let past = |t: T| points.last().is_some_and(|last| t > last.time);
  let start = read(equation.start)?;
  let (mut annuity, mut coupons, mut known) = (T::zero(), 0, (T::zero(), 0));
  for (t, accrual) in equation.coupons.clone() {
    annuity += accrual * read(t)?;
    coupons += 1;
    if !past(t) {
      known = (annuity, coupons);
    }
  }
  let direct = equation.value(start, annuity) / equation.slope();
  let gapped = past(equation.start) || known.1 < coupons;
  let mut reads = match points.last() {
    Some(last) if gapped => {
      let node = CurvePoint {
        time: maturity,
        discount_factor: direct,
      };
      match Segment::new(last, &node, method) {
        Some(segment) => Reads::Local {
          segment,
          start: (!past(equation.start)).then_some(start),
          known: known.0,
          skip: known.1,
        },
        None => Reads::Whole {
          trial: points.iter().copied().chain([node]).collect(),
          slot: points.len(),
          method,
        },
      }
    }
    _ => Reads::Fixed { start, annuity },
  };
  match points.last() {
    Some(last) if gapped => {
      let guess = if direct.is_finite() && direct > T::zero() {
        direct
      } else {
        last.discount_factor
      };
      let residual = |x: T| {
        let (start, annuity, slopes) = reads.at(&equation, x);
        let slope =
          slopes.map(|(start, annuity)| equation.value(start, annuity) - equation.slope());
        (equation.value(start, annuity) - equation.slope() * x, slope)
      };
      trial_node(guess, residual).ok_or_else(|| no_root(quote))
    }
    _ => Ok(direct),
  }
}

/// `equation`'s root in its slot of the whole `curve`, from the node there.
fn node_at<T: RealExt, C: Iterator<Item = (T, T)> + Clone>(
  curve: &[CurvePoint<T>],
  slot: usize,
  method: InterpolationMethod,
  quote: usize,
  equation: Equation<T, C>,
) -> Result<T, QuantError> {
  let mut reads = Reads::Whole {
    trial: curve.to_vec(),
    slot,
    method,
  };
  let residual = |x: T| {
    let (start, annuity, _) = reads.at(&equation, x);
    (equation.value(start, annuity) - equation.slope() * x, None)
  };
  trial_node(curve[slot].discount_factor, residual).ok_or_else(|| no_root(quote))
}

fn no_root(quote: usize) -> QuantError {
  QuantError::Optimizer {
    what: "bootstrap",
    reason: format!("no discount factor reprices quote {quote} on the curve that holds its node"),
  }
}

/// An equation's reads on the curve that holds its node.
enum Reads<T: RealExt> {
  /// Every read at or before the last pillar.
  Fixed { start: T, annuity: T },
  /// A local method: the reads at or before the last pillar taken once, the rest on its segment.
  Local {
    segment: Segment<T>,
    start: Option<T>,
    known: T,
    skip: usize,
  },
  /// A method whose later pillar reshapes the earlier intervals: every read on the trial curve,
  /// its node at `slot`.
  Whole {
    trial: Vec<CurvePoint<T>>,
    slot: usize,
    method: InterpolationMethod,
  },
}

impl<T: RealExt> Reads<T> {
  /// `D(start)` and `Σⱼ aⱼD(tⱼ)` at node `x`, with their slopes in `x` where the method gives them.
  fn at<C: Iterator<Item = (T, T)> + Clone>(
    &mut self,
    equation: &Equation<T, C>,
    x: T,
  ) -> (T, T, Option<(T, T)>) {
    match self {
      Reads::Fixed { start, annuity } => (*start, *annuity, Some((T::zero(), T::zero()))),
      Reads::Local {
        segment,
        start,
        known,
        skip,
      } => {
        let segment = segment.ending_at(x);
        let (start, start_slope) = match *start {
          Some(start) => (start, T::zero()),
          None => {
            let (d, elasticity) = segment.read(equation.start);
            (d, elasticity * d / x)
          }
        };
        let (mut annuity, mut weighted) = (*known, T::zero());
        for (t, accrual) in equation.coupons.clone().skip(*skip) {
          let (d, elasticity) = segment.read(t);
          annuity += accrual * d;
          weighted += accrual * elasticity * d;
        }
        (start, annuity, Some((start_slope, weighted / x)))
      }
      Reads::Whole {
        trial,
        slot,
        method,
      } => {
        trial[*slot].discount_factor = x;
        let (trial, method) = (&trial[..], *method);
        let read = |t: T| interpolate_discount_factor(trial, t, method);
        let start = read(equation.start);
        let annuity = equation
          .coupons
          .clone()
          .fold(T::zero(), |sum, (t, accrual)| sum + accrual * read(t));
        (start, annuity, None)
      }
    }
  }
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

#[cfg(test)]
mod tests;

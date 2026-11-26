//! What a family declares of its search: each coordinate's size, box and limits, and the
//! parameters that are zero at those limits.

use super::super::certificate::Axis;
use super::super::certificate::Stop;

/// `e`-folds the logarithm of a scale may run from its centre: past `ln(1/ε)` the scale is under
/// the spacing of the doubles at the data's size.
pub(super) const FOLDS: f64 = 36.0;

/// `e`-folds the logarithm of a shape may run: past `ln(1/√ε)` a law is its limit to the rounding
/// of a likelihood, as Student's is the normal's.
pub(super) const SHAPE_FOLDS: f64 = 18.0;

/// What a fit as good at a limit says of the parameter it names.
#[derive(Clone, Copy)]
enum Word {
  Unidentified,
  Bound,
}

/// The parameter that is zero at an end of a coordinate, by how far `d` the coordinate is from
/// the middle of its box toward the other end: a logarithm's `e^d`, a logit's `σ(d)`.
#[derive(Clone, Copy)]
enum Link {
  Log,
  Logit,
}

impl Link {
  fn parameter(self, d: f64) -> f64 {
    match self {
      Self::Log => d.exp(),
      Self::Logit => 1.0 / (1.0 + (-d).exp()),
    }
  }

  fn coordinate(self, parameter: f64) -> f64 {
    match self {
      Self::Log => parameter.ln(),
      Self::Logit => (parameter / (1.0 - parameter)).ln(),
    }
  }
}

/// A limit the family declares along a coordinate: its place, its word, the name it words, a
/// logarithm that rises as this one falls, their product held, and its own parameter's link.
pub(super) struct Limit {
  pub(super) at: f64,
  word: Word,
  names: Option<usize>,
  pub(super) held: Option<usize>,
  /// `None` where the limit is a zero of weights the family declares.
  link: Option<Link>,
}

impl Limit {
  pub(super) fn stop(&self, coordinate: usize) -> Stop {
    let names = self.names.unwrap_or(coordinate);
    match self.word {
      Word::Unidentified => Stop::Unidentified(names),
      Word::Bound => Stop::Bound(names),
    }
  }
}

/// One coordinate of a search: the size it varies on, the crate's own bounds on it, and the
/// limits of the family along it.
pub(crate) struct Coordinate {
  pub(super) scale: f64,
  pub(super) low: f64,
  pub(super) high: f64,
  pub(super) limits: Vec<Limit>,
  /// The others follow it down a valley to its limit, where a simplex stops short.
  pub(super) valley: bool,
}

impl Coordinate {
  fn within(scale: f64, low: f64, high: f64) -> Self {
    Self {
      scale,
      low,
      high,
      limits: Vec::new(),
      valley: false,
    }
  }

  /// A location of a sample centred at `centre` with spread `spread`, held within `1/ε` spreads.
  pub(crate) fn location(centre: f64, spread: f64) -> Self {
    let reach = spread / f64::EPSILON;
    Self::within(spread, centre - reach, centre + reach)
  }

  /// The logarithm of a scale whose natural size is `e^centre`.
  pub(crate) fn log(centre: f64) -> Self {
    Self::within(1.0, centre - FOLDS, centre + FOLDS)
  }

  /// The logarithm of a shape: its far end is the law's limit, which leaves it unidentified,
  /// and its near end the shape's own bound.
  pub(crate) fn shape(centre: f64) -> Self {
    let (low, high) = (centre - SHAPE_FOLDS, centre + SHAPE_FOLDS);
    let limit = |at: f64, word: Word| Limit {
      at,
      word,
      names: None,
      held: None,
      link: Some(Link::Log),
    };
    Self {
      limits: vec![limit(high, Word::Unidentified), limit(low, Word::Bound)],
      valley: true,
      ..Self::within(1.0, low, high)
    }
  }

  /// The logarithm with its low end a limit: the parameter vanishes there, and the logarithm
  /// `orphan` is not identified, the data telling their product alone.
  pub(crate) fn fading(mut self, orphan: usize) -> Self {
    self.limits.push(Limit {
      at: self.low,
      word: Word::Unidentified,
      names: Some(orphan),
      held: Some(orphan),
      link: Some(Link::Log),
    });
    self
  }

  /// The logarithm with its low end a limit where the family loses `orphan`, not identified there.
  pub(crate) fn vanishing(mut self, orphan: usize) -> Self {
    self.limits.push(Limit {
      at: self.low,
      word: Word::Unidentified,
      names: Some(orphan),
      held: None,
      link: Some(Link::Log),
    });
    self
  }

  fn ends(reach: f64, below: usize, above: usize, link: Option<Link>) -> Self {
    let limit = |at: f64, names: usize| Limit {
      at,
      word: Word::Bound,
      names: Some(names),
      held: None,
      link,
    };
    Self {
      limits: vec![limit(-reach, below), limit(reach, above)],
      ..Self::within(1.0, -reach, reach)
    }
  }

  /// A logit, or half of one, within `reach`: each end is the bound of the parameter it names,
  /// where the share or what it leaves is zero.
  pub(crate) fn share(reach: f64, below: usize, above: usize) -> Self {
    Self::ends(reach, below, above, Some(Link::Logit))
  }

  /// A logit that splits a total between weights the family declares: its ends are their zeros.
  pub(crate) fn split(reach: f64, below: usize, above: usize) -> Self {
    Self::ends(reach, below, above, None)
  }

  /// A coordinate of unit size that the objective bounds itself.
  pub(crate) fn free() -> Self {
    Self::within(1.0, f64::NEG_INFINITY, f64::INFINITY)
  }

  /// The limit the coordinate's bound at `value` is, if the family declares one there.
  pub(super) fn limit(&self, value: f64) -> Option<&Limit> {
    self.limits.iter().find(|limit| limit.at == value)
  }
}

/// The reach of a logit: past it the share is `0` or `1` to the spacing of the doubles.
pub(crate) const LOGIT: f64 = FOLDS;

/// A parameter's value at a point of the chart.
type Value<'a> = Box<dyn Fn(&[f64]) -> f64 + 'a>;

/// The point where a parameter is the value given, the rest as its limit's pin holds it; `None`
/// outside the family's region.
type Move<'a> = Box<dyn Fn(&[f64], f64) -> Option<Vec<f64>> + 'a>;

/// A parameter that is zero at a limit. A chart's slope vanishes at a coordinate's end whatever
/// the parameter's does, so the first-order check also steps along the parameter.
pub(crate) struct Parameter<'a> {
  pub(crate) of: Value<'a>,
  pub(crate) moved: Move<'a>,
  pub(crate) axis: Axis,
}

/// The parameter of each limit that is one coordinate's end, from the coordinate's link.
pub(super) fn own(coordinates: &[Coordinate]) -> Vec<Parameter<'_>> {
  let mut parameters = Vec::new();
  for (i, coordinate) in coordinates.iter().enumerate() {
    let middle = 0.5 * (coordinate.low + coordinate.high);
    for limit in &coordinate.limits {
      let Some(link) = limit.link else { continue };
      let inward = (middle - limit.at).signum();
      let parameter = move |value: f64| link.parameter(inward * (value - middle));
      let moved = move |at: &[f64], to: f64| {
        let mut moved = at.to_vec();
        moved[i] = middle + inward * link.coordinate(to);
        if let Some(held) = limit.held {
          moved[held] += at[i] - moved[i];
        }
        Some(moved)
      };
      parameters.push(Parameter {
        of: Box::new(move |at: &[f64]| parameter(at[i])),
        moved: Box::new(moved),
        axis: Axis {
          low: parameter(limit.at),
          high: parameter(2.0 * middle - limit.at),
          scale: 1.0,
        },
      });
    }
  }
  parameters
}

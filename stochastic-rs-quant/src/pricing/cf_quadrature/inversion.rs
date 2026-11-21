//! `∫₀^∞ Re[A − B] du` for a characteristic function's part `A` less a Gaussian control's `B`: a
//! head in `x = ln u` where the kernel is `1/u`, a walk to a phase reach, then each part's tail.

use std::cell::Cell;

use num_complex::Complex64;

use super::MAX_DEPTH;
use super::Panels;
use super::Walk;
use super::integrate_on;
use super::ooura;
use super::refine;
use super::walk;

/// The walk's target per panel; the head's and the near tail's are tighter, being sums of many.
const BODY_TOLERANCE: f64 = 1e-10;
const HEAD_TOLERANCE: f64 = 1e-11;
const NEAR_TOLERANCE: f64 = 1e-12;
/// The head covers `(0, 1]` in panels of `x = ln u` this wide, until three in a row add nothing.
const HEAD_WIDTH: f64 = 2.0;
const MAX_HEAD_PANELS: usize = 400;
/// The radians of `e^{−iuκ}` the walk covers before the tail rule takes over; `|κ|` floored.
const PHASE_BUDGET: f64 = 20.0;
const KAPPA_FLOOR: f64 = 1e-5;
/// `|φ|` past which the walk need not reach, probed at powers of two up to `2⁴⁰`.
const NEGLIGIBLE_CHF: f64 = 1e-12;

/// One tail as integrated: its start, its frequency and the panels of its stretch in `ln u`.
#[derive(Clone, Debug, Default)]
pub(crate) struct TailRule {
  start: f64,
  frequency: f64,
  near: Panels,
}

/// What one inversion walked, so that a replay integrates the same rule at other parameters.
#[derive(Clone, Debug, Default)]
pub(crate) struct Rule {
  head: Panels,
  body: Panels,
  tails: [Option<TailRule>; 2],
}

/// `(reach, scale)`: the walk ends at `min(u*, PHASE_BUDGET/|κ|)`, `u*` the first power of two
/// where `modulus` is negligible; it walks `y = u/scale` so that its panel budget covers the reach.
pub(crate) fn reach_and_scale(modulus: impl Fn(f64) -> f64, kappa: f64) -> (f64, f64) {
  let mut decayed = 1.0_f64;
  while decayed < 1.1e12 {
    if modulus(decayed) <= NEGLIGIBLE_CHF {
      break;
    }
    decayed *= 2.0;
  }
  let reach = decayed.min(PHASE_BUDGET / kappa.abs().max(KAPPA_FLOOR));
  (reach, (reach / 4000.0).max(1.0))
}

/// The first `u = 1e-8·10⁻⁴ⁿ` at which `gap`, a chf's distance from its `u → 0` limit, is under
/// `1e-6`: where a head walked leftwards may stop (a variance of `1e18` puts structure at `1e-9`).
pub(crate) fn head_floor(gap: impl Fn(f64) -> f64) -> f64 {
  let mut floor = 1e-8_f64;
  while floor > 1e-300 {
    if gap(floor) < 1e-6 {
      break;
    }
    floor *= 1e-4;
  }
  floor
}

/// `∫₀^∞ Re[chf_part − control_part]`; `control_part` oscillates at `control_frequency`, and a
/// `head_floor` below 1 says the kernel's `1/u` needs the head in `ln u`, walked at least that far.
pub(crate) struct Inversion<A, B> {
  pub(crate) chf_part: A,
  pub(crate) control_part: B,
  pub(crate) control_frequency: f64,
  pub(crate) head_floor: Option<f64>,
  pub(crate) reach: f64,
  pub(crate) scale: f64,
}

impl<A, B> Inversion<A, B>
where
  A: Fn(f64) -> Complex64,
  B: Fn(f64) -> Complex64,
{
  /// The integral, NaN if any sample is, walked or replayed as `walk` says.
  pub(crate) fn integrate(&self, walk: &mut Walk<'_, Rule>) -> f64 {
    walk.run(|rule| self.adaptive(rule), |rule| self.replay(rule))
  }

  fn real(&self, u: f64) -> f64 {
    ((self.chf_part)(u) - (self.control_part)(u)).re
  }

  fn adaptive(&self, mut rule: Option<&mut Rule>) -> f64 {
    let poisoned = Cell::new(false);
    let watched = |u: f64| {
      let v = self.real(u);
      poisoned.set(poisoned.get() || v.is_nan());
      v
    };
    let (head, start) = match self.head_floor {
      Some(floor) => {
        let leaves = rule.as_deref_mut().map(|r| &mut r.head);
        (self.head_in_log(&watched, floor, leaves), 1.0)
      }
      None => (0.0, 0.0),
    };
    let scaled = |y: f64| self.scale * watched(self.scale * y);
    let leaves = rule.as_deref_mut().map(|r| &mut r.body);
    let (body, end) = walk(
      scaled,
      start / self.scale,
      BODY_TOLERANCE,
      self.reach / self.scale,
      leaves,
    );
    if poisoned.get() || body.is_nan() {
      return f64::NAN;
    }
    let end = (end * self.scale).max(start);
    let mut tails = [
      self.tail_rule(&self.chf_part, end, None),
      self.tail_rule(&self.control_part, end, Some(self.control_frequency)),
    ];
    let chf_tail = tails[0]
      .as_mut()
      .map_or(0.0, |t| self.tail_value(&self.chf_part, t, true));
    let control_tail = tails[1]
      .as_mut()
      .map_or(0.0, |t| self.tail_value(&self.control_part, t, true));
    if let Some(rule) = rule {
      rule.tails = tails;
    }
    let total = head + body + chf_tail - control_tail;
    if poisoned.get() { f64::NAN } else { total }
  }

  fn replay(&self, rule: &Rule) -> f64 {
    let poisoned = Cell::new(false);
    let watched = |u: f64| {
      let v = self.real(u);
      poisoned.set(poisoned.get() || v.is_nan());
      v
    };
    let head = integrate_on(&|x: f64| watched(x.exp()) * x.exp(), &rule.head);
    let body = integrate_on(&|y: f64| self.scale * watched(self.scale * y), &rule.body);
    let mut total = head + body;
    for (slot, tail) in rule.tails.iter().enumerate() {
      let Some(tail) = tail else { continue };
      let mut tail = tail.clone();
      total += if slot == 0 {
        self.tail_value(&self.chf_part, &mut tail, false)
      } else {
        -self.tail_value(&self.control_part, &mut tail, false)
      };
    }
    if poisoned.get() { f64::NAN } else { total }
  }

  /// `∫₀¹` in `x = ln u`, walked leftwards from `x = 0` until, past `floor`, three panels in a row
  /// add nothing.
  fn head_in_log(
    &self,
    f: &impl Fn(f64) -> f64,
    floor: f64,
    mut leaves: Option<&mut Panels>,
  ) -> f64 {
    let g = |x: f64| f(x.exp()) * x.exp();
    let floor = floor.ln();
    let (mut total, mut hi, mut quiet) = (0.0, 0.0, 0);
    for _ in 0..MAX_HEAD_PANELS {
      let (value, _) = refine(
        &g,
        hi - HEAD_WIDTH,
        hi,
        HEAD_TOLERANCE,
        MAX_DEPTH,
        &mut leaves,
      );
      total += value;
      hi -= HEAD_WIDTH;
      quiet = if value.abs() <= HEAD_TOLERANCE {
        quiet + 1
      } else {
        0
      };
      if quiet >= 3 && hi <= floor || value.is_nan() {
        break;
      }
    }
    total
  }

  /// The part's tail from `start`, `None` where it cannot reach the target; a `frequency` of `None`
  /// is read from the part's own phase at `start`.
  fn tail_rule<P>(&self, part: &P, start: f64, frequency: Option<f64>) -> Option<TailRule>
  where
    P: Fn(f64) -> Complex64,
  {
    let at = part(start);
    let frequency = frequency.unwrap_or_else(|| {
      let step = 1e-6 * start.max(1.0);
      (part(start + step) / at).arg() / step
    });
    let span = start.max(1.0 / frequency.abs());
    (at.norm() * span > 1e-3 * BODY_TOLERANCE || at.is_nan()).then(|| TailRule {
      start,
      frequency,
      near: Panels::default(),
    })
  }

  /// `∫_start^∞ Re part`: `[start, 1/|ω|]`, where the phase turns less than a radian, in `ln u`;
  /// the rest by Ooura and Mori's rule at `|ω|`, on the conjugate where `ω < 0`, which keeps `Re`.
  fn tail_value<P>(&self, part: &P, tail: &mut TailRule, record: bool) -> f64
  where
    P: Fn(f64) -> Complex64,
  {
    let omega = tail.frequency;
    if omega.is_nan() {
      return f64::NAN;
    }
    let mut start = tail.start;
    let mut total = 0.0;
    if omega.abs() * start < 1.0 {
      let end = (1.0 / omega.abs()).ln().min(700.0);
      let near = |s: f64| part(s.exp()).re * s.exp();
      total += if record {
        let mut leaves = Some(&mut tail.near);
        refine(
          &near,
          start.ln(),
          end,
          NEAR_TOLERANCE,
          MAX_DEPTH,
          &mut leaves,
        )
        .0
      } else {
        integrate_on(&near, &tail.near)
      };
      start = end.exp();
    }
    if omega == 0.0 {
      return total;
    }
    let (far, _) = if omega > 0.0 {
      ooura::tail(part, start, omega)
    } else {
      ooura::tail(|u| part(u).conj(), start, -omega)
    };
    total + far.re
  }
}

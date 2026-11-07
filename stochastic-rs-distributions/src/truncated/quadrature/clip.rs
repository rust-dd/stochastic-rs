//! The clip: where the log-kernel falls a set depth below its value at the reference point.

use super::Kernel;
use super::panels::Cut;
use super::panels::span;

/// Where `ln k − ln k(c)` falls below `floor` from `from` toward `end`, `end` if never: a doubling probe beside `c`, then
/// bisection beside `c` or, if the bracket lies nearer it, the branch point `end` is nearer, so the cut stays exact.
pub(super) fn clip<K: Kernel>(kernel: &K, c: Cut, from: Cut, end: Cut, floor: f64) -> Cut {
  let below = |origin: Cut, t: f64| {
    let v = kernel.ln_rel(
      origin.plus(t),
      t - span(origin, c),
      origin.shifted(t).branch(),
    );
    v < floor || v.is_nan()
  };
  if end.x.is_finite() && !below(end, 0.0) {
    return end;
  }
  let gap = span(end, from);
  if gap == 0.0 {
    return end;
  }
  let (from_t, end_t, dir) = (span(c, from), span(c, end), -gap.signum());
  let (mut inside, mut outside) = (from_t, end_t);
  // From the kernel's length or its e-fold length at `from`, whichever is shorter, and positive: doubling overflows
  // within 2100 steps.
  let local = 1.0 / kernel.d1(from.x, from_t).abs();
  let mut step = kernel
    .length_scale()
    .min(local)
    .max(f64::from_bits(1))
    .min(f64::MAX);
  for _ in 0..2100 {
    let t = from_t + dir * step;
    if !t.is_finite() || (end_t.is_finite() && dir * (t - end_t) >= 0.0) {
      break;
    }
    if below(c, t) {
      outside = t;
      break;
    }
    inside = t;
    step *= 2.0;
  }
  // An open end the probe overflowed toward is bisected from the largest double.
  if outside.is_infinite() {
    outside = dir * f64::MAX;
  }
  let mut points = kernel.branch_points().into_iter().flatten();
  let near = points.find(|&(s, _)| end.offset(s).abs() < span(end, c).abs());
  let (origin, mut inside, mut outside) = match near {
    Some((s, _)) if (outside - span(c, Cut::at(s))).abs() < outside.abs() => {
      let to_c = span(Cut::at(s), c);
      (
        Cut::at(s),
        if inside == from_t {
          from.offset(s)
        } else {
          to_c + inside
        },
        if outside == end_t {
          end.offset(s)
        } else {
          to_c + outside
        },
      )
    }
    _ => (c, inside, outside),
  };
  // Toward an end at a branch point the halvings run down to the crossing however deep (2⁻¹⁰⁷⁴ at most).
  for _ in 0..1200 {
    if (outside - inside).abs() <= 4.0 * f64::EPSILON * outside.abs() {
      break;
    }
    let m = 0.5 * (inside + outside);
    if below(origin, m) {
      outside = m;
    } else {
      inside = m;
    }
  }
  origin.shifted(inside)
}

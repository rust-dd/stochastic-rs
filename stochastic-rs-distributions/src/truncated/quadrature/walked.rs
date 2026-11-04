//! A walk of `[a, b]` as it was taken: each panel's far end and mass in walk order, and the runs of panels that share an
//! origin, so a point's panel is found by bisection and the walk's running sum re-formed from the same masses.

use super::Sum;
use super::panels::Cut;
use super::panels::Panel;
use super::panels::span;

/// The marks held in place: the median law walks five panels, and a fill that keeps them inline allocates nothing.
const INLINE: usize = 8;

/// A panel's far end, an offset from its run's origin, and its mass.
#[derive(Clone, Copy, Debug, Default)]
struct Mark {
  far: f64,
  mass: f64,
}

/// Panels walked in a row from one origin, each starting where the one before it ends.
#[derive(Clone, Copy, Debug)]
struct Run {
  origin: Cut,
  near: f64,
  graded: bool,
  singular: Option<f64>,
  start: usize,
}

#[derive(Clone, Debug, Default)]
pub(super) struct Walked {
  /// The first `INLINE` marks in place while there are no more, else all of them on the heap.
  inline: [Mark; INLINE],
  spilt: Vec<Mark>,
  count: usize,
  /// The first run inline, as most walks are one, and the others.
  first: Option<Run>,
  rest: Vec<Run>,
  rev: bool,
}

impl Walked {
  pub(super) fn new(rev: bool) -> Self {
    Self {
      rev,
      ..Self::default()
    }
  }

  /// Appends the next panel of the walk and its mass; a panel joins the run only where it rebuilds to the bit.
  pub(super) fn push(&mut self, panel: &Panel, mass: f64) {
    let (near, far) = if self.rev {
      (panel.r, panel.l)
    } else {
      (panel.l, panel.r)
    };
    let joins = self.rest.last().or(self.first.as_ref()).is_some_and(|run| {
      run.origin.same(panel.origin)
        && run.graded == panel.graded
        && run.singular.map(f64::to_bits) == panel.singular.map(f64::to_bits)
        && self
          .marks()
          .last()
          .is_some_and(|m| m.far.to_bits() == near.to_bits())
    });
    debug_assert!(
      self.rev || !joins || far >= near,
      "a forward run's ends ascend"
    );
    if !joins {
      let run = Run {
        origin: panel.origin,
        near,
        graded: panel.graded,
        singular: panel.singular,
        start: self.count,
      };
      match self.first {
        None => self.first = Some(run),
        Some(_) => self.rest.push(run),
      }
    }
    let mark = Mark { far, mass };
    match self.count {
      n if n < INLINE => self.inline[n] = mark,
      INLINE => {
        self.spilt.reserve(4 * INLINE);
        self.spilt.extend_from_slice(&self.inline);
        self.spilt.push(mark);
      }
      _ => self.spilt.push(mark),
    }
    self.count += 1;
  }

  fn marks(&self) -> &[Mark] {
    if self.count <= INLINE {
      &self.inline[..self.count]
    } else {
      &self.spilt
    }
  }

  pub(super) fn len(&self) -> usize {
    self.count
  }

  pub(super) fn mass(&self, i: usize) -> f64 {
    self.marks()[i].mass
  }

  /// The running sum before panel `i`, its additions the walk's own.
  pub(super) fn below(&self, i: usize) -> Sum {
    let mut sum = Sum::default();
    for m in &self.marks()[..i] {
      sum.add(m.mass);
    }
    sum
  }

  /// The forward walk's stop at `at`: the first panel whose far end its offset does not reach, that offset, and the
  /// panel's index. A run's ends ascend from one origin, so the run is bisected.
  pub(super) fn holding(&self, at: Cut) -> Option<(usize, Panel, f64)> {
    debug_assert!(!self.rev);
    for (j, run) in self.first.iter().chain(&self.rest).enumerate() {
      let end = self.end(j);
      let t = span(run.origin, at);
      let i = run.start + self.marks()[run.start..end].partition_point(|m| t >= m.far);
      if i < end {
        return Some((i, self.panel(j, i), t));
      }
    }
    None
  }

  /// The walk's stop at `want`: the first panel whose running sum reaches it, else the last, with the part of `want`
  /// left at it and its mass.
  pub(super) fn reaching(&self, want: f64) -> Option<(Panel, f64, f64)> {
    let mut acc = Sum::default();
    let mut hit = None;
    for (i, m) in self.marks().iter().enumerate() {
      hit = Some((i, (want - acc.hi) - acc.lo, m.mass));
      acc.add(m.mass);
      if acc.value() >= want {
        break;
      }
    }
    let (i, rest, mass) = hit?;
    let run = self.rest.partition_point(|run| run.start <= i);
    Some((self.panel(run, i), rest, mass))
  }

  fn run(&self, j: usize) -> &Run {
    match j {
      0 => self.first.as_ref().expect("a walk with a panel has a run"),
      _ => &self.rest[j - 1],
    }
  }

  /// One past the last panel of run `j`.
  fn end(&self, j: usize) -> usize {
    self.rest.get(j).map_or(self.count, |next| next.start)
  }

  fn panel(&self, run: usize, i: usize) -> Panel {
    let r = self.run(run);
    let marks = self.marks();
    let near = if i == r.start {
      r.near
    } else {
      marks[i - 1].far
    };
    let far = marks[i].far;
    let (l, r_end) = if self.rev { (far, near) } else { (near, far) };
    Panel {
      origin: r.origin,
      l,
      r: r_end,
      graded: r.graded,
      singular: r.singular,
    }
  }
}

//! Composite 64-node Gauss–Legendre quadrature of a truncated law's normalised kernel: the one engine behind the
//! truncated laws' mass, moments, entropy, pdf, cdf and quantile.
//! Reference: Trefethen, L. N. (2008), "Is Gauss Quadrature Better than Clenshaw–Curtis?", *SIAM Review* 50(1), 67-87, DOI 10.1137/060659831 (the Bernstein-ellipse rate that sizes the panels).

use std::ops::ControlFlow;

use panels::GL64_W;
use panels::GL64_X;
use panels::Panel;
use panels::panels;

mod panels;

/// A law's log-kernel on its support, in the law's own coordinate.
pub(super) trait Kernel {
  /// `ln k(x)`: `-∞` outside the support, `+∞` at a singular end.
  fn ln_k(&self, x: f64) -> f64;

  /// `ln k(x) − ln k(c)` from `y = x − c`; on a graded panel `branch` is `(s, x − s)`, exact beside the branch point `s`.
  fn ln_rel(&self, x: f64, y: f64, c: f64, branch: Option<(f64, f64)>) -> f64;

  /// `d ln k / dx` at a regular point.
  fn d1(&self, x: f64) -> f64;

  /// `d² ln k / dx²` at a regular point.
  fn d2(&self, x: f64) -> f64;

  /// The kernel's maximiser over its support (an end when it is monotone).
  fn mode(&self) -> f64;

  /// `(s, p)`: `k ~ |x − s|^p` next to `s`, `p > −1`; the points ascend and lie more than a width apart.
  fn branch_points(&self) -> [Option<(f64, f64)>; 2];

  /// The kernel's own e-fold length: the clip probe's first step and the reference point's distance from a singular end.
  fn length_scale(&self) -> f64;
}

/// Relative kernel below `e^{−CLIP}` is dropped: 1e-18 of the mass near the reference point.
const CLIP: f64 = 41.5;

/// An interval's reference point, clipped ends and normalising sums, cached so a pdf costs one `exp`.
#[derive(Clone, Debug)]
pub(super) struct Truncation<K> {
  kernel: K,
  lo: f64,
  up: f64,
  /// The point every offset is formed from: the clamped mode, a tenth of a scale off a singular end, or the midpoint.
  centre: f64,
  /// The ends moved in to where the kernel is `e^{−CLIP}` below its value at `centre`; a singular end stays.
  a: f64,
  b: f64,
  /// `b − a`, the unit of every sum: no power of the law's own scale leaves the range.
  width: f64,
  /// `∫_a^b k/k(centre)`, in units of `width`.
  m0: f64,
  /// `1/(width·m0)`, so a density costs a product.
  inv_m0: f64,
  /// `∫_lo^a` and `∫_b^up` of `k/k(centre)`, in units of `width`: the cdf adds the first, the quantile both.
  tail_lo: f64,
  tail_up: f64,
}

impl<K: Kernel> Truncation<K> {
  /// `kernel` on `[lo, up]`, `lo < up`, with the mass beyond the clips that the cdf and the quantile read.
  pub(super) fn new(kernel: K, lo: f64, up: f64) -> Self {
    let mut t = Self::mass_only(kernel, lo, up);
    if t.a > lo {
      t.tail_lo = t.integrate(clip(&t.kernel, t.centre, t.a, lo, -2.0 * CLIP), t.a, 0.0);
    }
    if t.b < up {
      t.tail_up = t.integrate(t.b, clip(&t.kernel, t.centre, t.b, up, -2.0 * CLIP), 0.0);
    }
    t
  }

  /// The reference point, the clipped ends and their mass; an infinite end is clipped, a singular one graded.
  pub(super) fn mass_only(kernel: K, lo: f64, up: f64) -> Self {
    let singular = |x: f64| kernel.ln_k(x) == f64::INFINITY;
    let (lo_singular, up_singular) = (singular(lo), singular(up));
    let mut centre = kernel.mode().clamp(lo, up);
    if singular(centre) {
      let exponent = |end: f64| {
        let points = kernel.branch_points().into_iter().flatten();
        points
          .filter(|&(s, _)| s == end)
          .map(|(_, p)| p)
          .sum::<f64>()
      };
      let heavier_lo = !up_singular || (lo_singular && exponent(lo) <= exponent(up));
      let step = (up - lo).min(kernel.length_scale()) / 10.0;
      centre = if heavier_lo { lo + step } else { up - step };
    }
    let a = if lo_singular {
      lo
    } else {
      clip(&kernel, centre, centre, lo, -CLIP)
    };
    let b = if up_singular {
      up
    } else {
      clip(&kernel, centre, centre, up, -CLIP)
    };
    let drop = |e: f64| -kernel.ln_rel(e, e - centre, centre, None);
    // A flat kernel's odd moments about an end lose digits; about the midpoint they do not.
    if !(lo_singular || up_singular) && drop(a).max(drop(b)) < 1.0 {
      centre = 0.5 * (a + b);
    }
    let mut t = Self {
      kernel,
      lo,
      up,
      centre,
      a,
      b,
      width: b - a,
      m0: 0.0,
      inv_m0: 0.0,
      tail_lo: 0.0,
      tail_up: 0.0,
    };
    t.m0 = t.integrate(a, b, 0.0);
    t.inv_m0 = 1.0 / (t.width * t.m0);
    t
  }

  /// The reference point `c`, whose `ln k(c)` the masses are relative to.
  pub(super) fn centre(&self) -> f64 {
    self.centre
  }

  /// `∫_a^b k/k(c)` over the clipped interval.
  pub(super) fn mass(&self) -> f64 {
    self.width * self.m0
  }

  /// `(mean, scale²·variance, skewness, excess kurtosis)` in the kernel's coordinate: moments about the reference point
  /// in units of `width`, the binomial shift by `d = M₁/M₀`, then `scale·width` applied once.
  pub(super) fn moments(&self, scale: f64) -> [f64; 4] {
    let mut m = [0.0; 5];
    let _ = self.walk(self.a, self.b, false, |panel| {
      for (slot, part) in m.iter_mut().zip(self.sums::<5>(&panel, 0.0)) {
        *slot += part;
      }
      ControlFlow::<()>::Continue(())
    });
    let d = m[1] / m[0];
    let (r2, r3, r4) = (m[2] / m[0], m[3] / m[0], m[4] / m[0]);
    let mu2 = r2 - d * d;
    let mu3 = r3 - 3.0 * d * r2 + 2.0 * d * d * d;
    let mu4 = r4 - 4.0 * d * r3 + 6.0 * d * d * r2 - 3.0 * d * d * d * d;
    let unit = scale * self.width;
    [
      self.centre + self.width * d,
      (unit * mu2) * unit,
      mu3 / mu2.powf(1.5),
      mu4 / (mu2 * mu2) - 3.0,
    ]
  }

  /// `−∫ f ln f` as `ln M₀ − E[ln k(X) − ln k(c)]`, so no large `ln k(c)` cancels against `E[ln k(X)]`.
  pub(super) fn entropy(&self) -> f64 {
    let (mut mass, mut log) = (0.0, 0.0);
    let _ = self.walk(self.a, self.b, false, |panel| {
      self.nodes(&panel, |w, _, ln_g| {
        let g = w * ln_g.exp();
        if g > 0.0 {
          mass += g;
          log += g * ln_g;
        }
      });
      ControlFlow::<()>::Continue(())
    });
    (self.width.ln() + mass.ln()) - log / mass
  }

  /// The normalised density `k(x)/∫k` at a point of `[lo, up]`, whose support the law checks at its own bounds.
  pub(super) fn density(&self, x: f64) -> f64 {
    self.ratio(x, 0.0) * self.inv_m0
  }

  /// `∫_lo^x k / ∫k`, integrated up from `lo`: no survival function reads it, so a level near 1 needs absolute digits.
  pub(super) fn cdf(&self, x: f64) -> f64 {
    if x.is_nan() {
      return f64::NAN;
    }
    if x <= self.lo {
      return 0.0;
    }
    if x >= self.b {
      return 1.0;
    }
    if x < self.a {
      return self.below_clip(x);
    }
    ((self.tail_lo + self.integrate(self.a, x, 0.0)) / self.m0).min(1.0)
  }

  /// `cdf⁻¹(p)`: the bounds at 0 and 1; else the panel holding `p`, walked to from the nearer end, then Newton in it.
  pub(super) fn quantile(&self, p: f64) -> f64 {
    if p.is_nan() {
      return f64::NAN;
    }
    if p <= 0.0 {
      return self.lo;
    }
    if p >= 1.0 {
      return self.up;
    }
    self.invert(p)
  }

  fn invert(&self, p: f64) -> f64 {
    if p > 0.5 {
      return self.locate(
        self.a,
        self.b,
        (1.0 - p) * self.m0 - self.tail_up,
        false,
        0.0,
      );
    }
    let want = p * self.m0 - self.tail_lo;
    if want > 0.0 {
      return self.locate(self.a, self.b, want, true, 0.0);
    }
    // `p·M₀` underflowed against an end that was never clipped.
    if self.a <= self.lo {
      return self.lo;
    }
    // Below the clip: from where the kernel is `e^{−CLIP}` below `p` times its reference value, every value taken
    // relative to the kernel there, so no node underflows.
    let start = clip(&self.kernel, self.centre, self.a, self.lo, p.ln() - CLIP);
    let shift = self
      .kernel
      .ln_rel(start, start - self.centre, self.centre, None);
    let want = (p.ln() + self.m0.ln() - shift).exp();
    self.locate(start, self.a, want, true, shift)
  }

  /// The point of `[l, r]` with `want` of the mass beyond the end `from_lo` names, every value relative to `e^{shift}`:
  /// the panels walked from that end, then Newton in the one that holds it.
  fn locate(&self, l: f64, r: f64, want: f64, from_lo: bool, shift: f64) -> f64 {
    let mut acc = 0.0;
    let mut hit = None;
    let _ = self.walk(l, r, !from_lo, |panel| {
      let mass = self.sums::<1>(&panel, shift)[0];
      hit = Some((panel, want - acc, mass));
      acc += mass;
      if acc >= want {
        ControlFlow::Break(())
      } else {
        ControlFlow::Continue(())
      }
    });
    match hit {
      Some((panel, rest, mass)) => self.solve(panel, rest, mass, from_lo, shift),
      None if from_lo => l,
      None => r,
    }
  }

  /// Newton on the panel's partial mass, bisecting whenever a step leaves the bracket, which reaches one ulp past the
  /// panel's rounded ends.
  fn solve(&self, panel: Panel, want: f64, mass: f64, from_lo: bool, shift: f64) -> f64 {
    let (l, r) = (panel.origin + panel.l, panel.origin + panel.r);
    let (mut lb, mut rb) = (l.next_down().max(self.lo), r.next_up().min(self.up));
    let frac = want / mass;
    let frac = if frac.is_finite() {
      frac.clamp(0.0, 1.0)
    } else {
      0.5
    };
    // The start inverts the panel's exponential model, its decay read from the kernel at the two ends.
    let c = self.centre;
    let decay = self.kernel.ln_rel(l, l - c, c, None) - self.kernel.ln_rel(r, r - c, c, None);
    let span = if from_lo { decay } else { -decay };
    let v = if span.is_finite() && span.abs() > 1e-8 {
      let share = frac * -(-span).exp_m1();
      -(-share).ln_1p() / span
    } else {
      frac
    };
    let v = if v.is_finite() {
      v.clamp(0.0, 1.0)
    } else {
      frac
    };
    let mut x = if from_lo {
      l + (r - l) * v
    } else {
      r - (r - l) * v
    };
    let tol = f64::EPSILON * (r - l);
    for _ in 0..100 {
      let f = self.partial(&panel, x, from_lo, shift) - want;
      if f == 0.0 {
        return x;
      }
      if (f > 0.0) == from_lo {
        rb = x;
      } else {
        lb = x;
      }
      let slope = if from_lo {
        self.ratio(x, shift) / self.width
      } else {
        -self.ratio(x, shift) / self.width
      };
      let mut next = x - f / slope;
      if !(lb..=rb).contains(&next) {
        next = 0.5 * (lb + rb);
      }
      if (next - x).abs() <= 4.0 * f64::EPSILON * x.abs() + tol {
        return next;
      }
      x = next;
    }
    x
  }

  /// `panel`'s mass from its exact near end to `x`, one rule on the sub-panel the panel's budgets already cover.
  fn partial(&self, panel: &Panel, x: f64, from_lo: bool, shift: f64) -> f64 {
    let t = x - panel.origin;
    let sub = if from_lo {
      Panel { r: t, ..*panel }
    } else {
      Panel { l: t, ..*panel }
    };
    self.sums::<1>(&sub, shift)[0]
  }

  /// The cdf below the lower clip: the mass from where the kernel is `e^{−CLIP}` below its value at `x`, relative to
  /// that value, which one `exp` restores, so a subnormal answer is rounded once.
  fn below_clip(&self, x: f64) -> f64 {
    let ln_g = self.kernel.ln_rel(x, x - self.centre, self.centre, None);
    if ln_g == f64::NEG_INFINITY {
      return 0.0;
    }
    let start = clip(&self.kernel, self.centre, x, self.lo, ln_g - CLIP);
    let relative = self.integrate(start, x, ln_g);
    (ln_g + (relative / self.m0).ln()).exp().min(1.0)
  }

  /// `k(x)/k(c)` over `e^{shift}`.
  fn ratio(&self, x: f64, shift: f64) -> f64 {
    (self.kernel.ln_rel(x, x - self.centre, self.centre, None) - shift).exp()
  }

  /// `∫_l^r k/k(c)` in units of `width`, over `e^{shift}`, signed.
  fn integrate(&self, l: f64, r: f64, shift: f64) -> f64 {
    if r < l {
      return -self.integrate(r, l, shift);
    }
    let mut sum = 0.0;
    let _ = self.walk(l, r, false, |panel| {
      sum += self.sums::<1>(&panel, shift)[0];
      ControlFlow::<()>::Continue(())
    });
    sum
  }

  fn walk<B>(
    &self,
    l: f64,
    r: f64,
    rev: bool,
    visit: impl FnMut(Panel) -> ControlFlow<B>,
  ) -> ControlFlow<B> {
    panels(&self.kernel, self.centre, l, r, rev, visit)
  }

  /// `Σ_i w_i g(x_i) (y_i/width)^j / e^{shift}`, `j < N`, on one panel.
  fn sums<const N: usize>(&self, panel: &Panel, shift: f64) -> [f64; N] {
    let mut sums = [0.0; N];
    self.nodes(panel, |w, y, ln_g| {
      let (g, z) = (w * (ln_g - shift).exp(), y / self.width);
      let mut zj = 1.0;
      for slot in &mut sums {
        *slot += g * zj;
        zj *= z;
      }
    });
    sums
  }

  /// `f(w, y, ln g)` at each node: weight times half-width over `width`, offset from the centre, `ln k(x) − ln k(c)`.
  fn nodes(&self, panel: &Panel, mut f: impl FnMut(f64, f64, f64)) {
    let half = 0.5 * (panel.r - panel.l);
    let scaled = half / self.width;
    for (node, weight) in GL64_X.iter().zip(&GL64_W) {
      for xi in [-node, *node] {
        let t = panel.l + half * (1.0 + xi);
        let (x, y, branch) = if panel.graded {
          (
            panel.origin + t,
            (panel.origin - self.centre) + t,
            Some((panel.origin, t)),
          )
        } else {
          (self.centre + t, t, None)
        };
        f(
          weight * scaled,
          y,
          self.kernel.ln_rel(x, y, self.centre, branch),
        );
      }
    }
  }
}

/// The point between `from` and `end` where `ln k − ln k(c)` falls below `floor` (a doubling probe, then bisection),
/// or `end` if it never does.
fn clip<K: Kernel>(kernel: &K, c: f64, from: f64, end: f64, floor: f64) -> f64 {
  let below = |x: f64| {
    let v = kernel.ln_rel(x, x - c, c, None);
    v < floor || v.is_nan()
  };
  if end == from || (end.is_finite() && !below(end)) {
    return end;
  }
  let dir = if end > from { 1.0 } else { -1.0 };
  let (mut inside, mut outside) = (from, end);
  let mut step = kernel.length_scale();
  loop {
    let x = from + dir * step;
    if !x.is_finite() || (end.is_finite() && dir * (x - end) >= 0.0) {
      break;
    }
    if below(x) {
      outside = x;
      break;
    }
    inside = x;
    step *= 2.0;
  }
  // Toward an end at 0 the halvings run down to the crossing however deep (2⁻¹⁰⁷⁴ at most), then refine it.
  for _ in 0..1200 {
    if (outside - inside).abs() <= 4.0 * f64::EPSILON * outside.abs() {
      break;
    }
    let m = 0.5 * (inside + outside);
    if below(m) {
      outside = m;
    } else {
      inside = m;
    }
  }
  inside
}

#[cfg(test)]
mod tests;

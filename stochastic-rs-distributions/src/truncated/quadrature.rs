//! Composite 64-node Gauss–Legendre quadrature of a truncated law's normalised kernel, behind all its analytics; a
//! singular end's innermost panel integrates in `u = (t/h)^e`, where `t^{e−1} dt = (h^e/e) du`.
//! Reference: Trefethen, L. N. (2008), "Is Gauss Quadrature Better than Clenshaw–Curtis?", *SIAM Review* 50(1), 67-87, DOI 10.1137/060659831 (the Bernstein-ellipse rate that sizes the panels).
//! Reference: Press, W. H., Teukolsky, S. A., Vetterling, W. T. & Flannery, B. P. (1992), *Numerical Recipes in C*, 2nd ed., Cambridge University Press, §4.4, ISBN 0-521-43108-5 (the change of variable at an integrable power-law end).

use std::ops::ControlFlow;

pub(super) use panels::Cut;
use panels::GL64_W;
use panels::GL64_X;
use panels::Panel;
use panels::panels;
use panels::span;

use crate::traits::RealExt;

mod panels;
mod quantile;
mod singular;

/// A law's log-kernel on its support, in the law's own coordinate, taken relative to a reference point `c`.
pub(super) trait Kernel {
  /// `ln k(x)`: `-∞` outside the support, `+∞` at a singular end.
  fn ln_k(&self, x: f64) -> f64;

  /// Fixes the reference point `c` the relative log-kernel and its derivatives are taken from; `at_mode` when `c` is the
  /// kernel's own mode, where its slope is 0 by construction.
  fn anchor(&mut self, c: Cut, at_mode: bool);

  /// `ln k(x) − ln k(c)` from `y = x − c`; `branch` is `(s, x − s)`, exact beside the point `s`.
  fn ln_rel(&self, x: f64, y: f64, branch: Option<(f64, f64)>) -> f64;

  /// `d ln k / dx` at a regular point `x = c + y`.
  fn d1(&self, x: f64, y: f64) -> f64;

  /// `d² ln k / dx²` at a regular point `x = c + y`.
  fn d2(&self, x: f64, y: f64) -> f64;

  /// The kernel's maximiser over its support (an end when it is monotone), held beside the end it would round onto.
  fn mode(&self) -> Cut;

  /// `(s, e)`: `k ~ |x − s|^{e−1}` next to `s`, `e > 0` exact where `e − 1` would round; the points ascend and lie
  /// more than a width apart.
  fn branch_points(&self) -> [Option<(f64, f64)>; 2];

  /// `ln_rel` less `p·ln|x − s|` at the branch point `s`: the smooth factor, read where `x − s` may round to 0.
  fn ln_rel_regular(&self, x: f64, y: f64, _s: f64) -> f64 {
    self.ln_rel(x, y, None)
  }

  /// The kernel's own e-fold length: the clip probe's first step and the reference point's distance from a singular end.
  fn length_scale(&self) -> f64;

  /// The support's ends, which a bound at or past removes nothing.
  fn support(&self) -> (f64, f64);

  /// The untruncated law's mean, standard deviation, skewness and excess kurtosis.
  fn base_moments(&self) -> [f64; 4];
}

/// `1/(2j + 3)`, `j < 17`: past `(1/9)^17/37` the series `Σ w^{2j}/(2j + 3)` below adds nothing at `|w| ≤ 1/3`.
const ODD: [f64; 17] = [
  1.0 / 3.0,
  1.0 / 5.0,
  1.0 / 7.0,
  1.0 / 9.0,
  1.0 / 11.0,
  1.0 / 13.0,
  1.0 / 15.0,
  1.0 / 17.0,
  1.0 / 19.0,
  1.0 / 21.0,
  1.0 / 23.0,
  1.0 / 25.0,
  1.0 / 27.0,
  1.0 / 29.0,
  1.0 / 31.0,
  1.0 / 33.0,
  1.0 / 35.0,
];

/// `ln(1 + z) − z` to full relative precision: `−wz + 2w³ Σ_j w^{2j}/(2j + 3)`, `w = z/(2 + z)`, by Estrin's scheme
/// for `|w| ≤ 1/3` (from `ln(1 + z) = 2 atanh w`); past that the plain difference loses under two bits.
pub(super) fn ln_1p_mx(z: f64) -> f64 {
  let w = z / (2.0 + z);
  if w.abs() > 1.0 / 3.0 || w.is_nan() {
    return z.ln_1p() - z;
  }
  let x = w * w;
  let (x2, x4) = (x * x, (x * x) * (x * x));
  let pair = |i: usize| x.mul_add(ODD[i + 1], ODD[i]);
  let quad = |i: usize| x2.mul_add(pair(i + 2), pair(i));
  let oct = |i: usize| x4.mul_add(quad(i + 4), quad(i));
  let x8 = x4 * x4;
  let sum = x8.mul_add(x8.mul_add(ODD[16], oct(8)), oct(0));
  2.0 * w * x * sum - w * z
}

/// Relative kernel below `e^{−CLIP}` is dropped: 1e-18 of the mass near the reference point.
const CLIP: f64 = 41.5;

/// An interval's reference point, clipped ends and normalising sums, cached so a pdf costs one `exp`.
#[derive(Clone, Debug)]
pub(super) struct Truncation<K> {
  kernel: K,
  lo: f64,
  up: f64,
  /// The point every offset is formed from: the clamped mode, a tenth of a scale off a singular end, or the midpoint;
  /// beside the end it would round onto, so `ln k(centre)` stays finite.
  centre: Cut,
  /// The point the moments are summed about: the singular branch point the centre steps off, where the mass may sit
  /// closer than any offset from the centre resolves, else the centre.
  frame: Cut,
  /// The ends moved in to where the kernel is `e^{−CLIP}` below its value at `centre`, each an exact offset from the
  /// point it was found from, however much finer than an ulp; a singular end stays.
  a: Cut,
  b: Cut,
  /// `b − a`, the unit of every sum: no power of the law's own scale leaves the range.
  width: f64,
  /// `∫_a^b k/k(centre)`, in units of `width`.
  m0: f64,
  /// `ln(width·m0)`, so a density costs one `exp` and overflows only where its value does.
  ln_norm: f64,
  /// The clipped width is 0: the law lies closer to its reference point than any offset resolves.
  point: bool,
  /// `∫_lo^a` and `∫_b^up` of `k/k(centre)`, in units of `width`: the cdf adds the first, the quantile both.
  tail_lo: f64,
  tail_up: f64,
}

impl<K: Kernel> Truncation<K> {
  /// `kernel` on `[lo, up]`, `lo < up`, with the mass beyond the clips that the cdf and the quantile read.
  pub(super) fn new(kernel: K, lo: f64, up: f64) -> Self {
    let mut t = Self::mass_only(kernel, lo, up);
    if t.a.x > lo {
      t.tail_lo = t.integrate(clip(&t.kernel, t.centre, t.a, lo, -2.0 * CLIP), t.a, 0.0);
    }
    if t.b.x < up {
      t.tail_up = t.integrate(t.b, clip(&t.kernel, t.centre, t.b, up, -2.0 * CLIP), 0.0);
    }
    t
  }

  /// The reference point, the clipped ends and their mass; an infinite end is clipped, a singular one graded.
  pub(super) fn mass_only(mut kernel: K, lo: f64, up: f64) -> Self {
    let step = (up - lo).min(kernel.length_scale()) / 10.0;
    // `(s, e)` of a singular branch point at a point or within `step` of it, where the power law rules the kernel's mass.
    let singular = |x: Cut| {
      let mut points = kernel.branch_points().into_iter().flatten();
      points.find(|&(s, e)| e < 1.0 && x.offset(s).abs() <= step)
    };
    let (lo_singular, up_singular) = (singular(Cut::at(lo)), singular(Cut::at(up)));
    let mode = kernel.mode();
    let mut at_mode = mode.offset(lo) >= 0.0 && mode.offset(up) <= 0.0;
    let mut centre = if mode.offset(lo) < 0.0 {
      Cut::at(lo)
    } else if mode.offset(up) > 0.0 {
      Cut::at(up)
    } else {
      mode
    };
    let mut frame = None;
    if singular(centre).is_some() {
      let heavier_lo = match (lo_singular, up_singular) {
        (Some((_, e)), Some((_, f))) => e <= f,
        (_, None) => true,
        (None, Some(_)) => false,
      };
      let (end, point, side) = if heavier_lo {
        (lo, lo_singular, 1.0)
      } else {
        (up, up_singular, -1.0)
      };
      at_mode = false;
      centre = point.map_or(Cut::at(end + side * step), |(s, _)| {
        frame = Some(Cut::at(s));
        Cut::beside(s, (end - s) + side * step)
      });
    }
    kernel.anchor(centre, at_mode);
    let a = if lo_singular.is_some() {
      Cut::at(lo)
    } else {
      clip(&kernel, centre, centre, lo, -CLIP)
    };
    let b = if up_singular.is_some() {
      Cut::at(up)
    } else {
      clip(&kernel, centre, centre, up, -CLIP)
    };
    let drop = |e: Cut| -kernel.ln_rel(e.x, span(centre, e), e.branch());
    // A flat kernel's odd moments about an end lose digits; about the midpoint they do not.
    if lo_singular.is_none() && up_singular.is_none() && drop(a).max(drop(b)) < 1.0 {
      centre = a.shifted(0.5 * span(a, b));
      kernel.anchor(centre, false);
    }
    let mut t = Self {
      kernel,
      lo,
      up,
      centre,
      frame: frame.unwrap_or(centre),
      a,
      b,
      width: span(a, b),
      m0: 0.0,
      ln_norm: 0.0,
      point: false,
      tail_lo: 0.0,
      tail_up: 0.0,
    };
    t.m0 = t.integrate(a, b, 0.0);
    t.ln_norm = t.width.ln() + t.m0.ln();
    t.point = !(t.width > 0.0 && t.m0 > 0.0 && t.m0.is_finite());
    t
  }

  /// The reference point `c`, whose `ln k(c)` the masses are relative to.
  pub(super) fn centre(&self) -> f64 {
    self.centre.x
  }

  /// `∫_a^b k/k(c)` over the clipped interval.
  pub(super) fn mass(&self) -> f64 {
    self.width * self.m0
  }

  /// `(mean, scale²·variance, skewness, excess kurtosis)`; an interval that cuts neither end removes less than the clip
  /// resolves and reads the base's closed forms, keeping relative digits sums lose (2/√k at k = 1e300).
  pub(super) fn moments(&self, scale: f64) -> [f64; 4] {
    if let Some(c) = self.point() {
      return [c, 0.0, f64::NAN, f64::NAN];
    }
    let (lo, up) = self.kernel.support();
    if (self.lo <= lo || self.a.x > self.lo) && (self.up >= up || self.b.x < self.up) {
      let [mean, sd, skewness, kurtosis] = self.kernel.base_moments();
      let unit = scale * sd;
      return [mean, unit * unit, skewness, kurtosis];
    }
    self.summed_moments(scale)
  }

  /// The moments from sums in units of `width`: the mean's offset `d`, then sums about it, so no binomial shift by a
  /// far reference point cancels.
  fn summed_moments(&self, scale: f64) -> [f64; 4] {
    let mean = self.moment_sums::<2>(0.0);
    let d = mean[1] / mean[0];
    let m = self.moment_sums::<5>(d);
    let e = m[1] / m[0];
    let (r2, r3, r4) = (m[2] / m[0], m[3] / m[0], m[4] / m[0]);
    let mu2 = r2 - e * e;
    let mu3 = r3 - 3.0 * e * r2 + 2.0 * e * e * e;
    let mu4 = r4 - 4.0 * e * r3 + 6.0 * e * e * r2 - 3.0 * e * e * e * e;
    let unit = scale * self.width;
    let (s, t) = self.frame.anchor();
    [
      s + (t + self.width * (d + e)),
      (unit * mu2) * unit,
      (mu3 / mu2) / mu2.sqrt(),
      (mu4 / mu2) / mu2 - 3.0,
    ]
  }

  fn moment_sums<const N: usize>(&self, about: f64) -> [f64; N] {
    let mut m = [0.0; N];
    let _ = self.walk(self.a, self.b, false, |panel| {
      if let Some(e) = panel.singular {
        self.singular_moments(&panel, e, about, &mut m);
      } else {
        self.nodes(&panel, |w, ln_v, offset, _| {
          let (g, z) = (w * ln_v.exp(), offset / self.width - about);
          let mut zj = 1.0;
          for slot in &mut m {
            *slot += g * zj;
            zj *= z;
          }
        });
      }
      ControlFlow::<()>::Continue(())
    });
    m
  }

  /// `−∫ f ln f` as `ln M₀ − E[ln k(X) − ln k(c)]`, so no large `ln k(c)` cancels against `E[ln k(X)]`.
  pub(super) fn entropy(&self) -> f64 {
    if self.point().is_some() {
      return f64::NAN;
    }
    let (mut mass, mut log) = (0.0, 0.0);
    let _ = self.walk(self.a, self.b, false, |panel| {
      self.nodes(&panel, |w, ln_v, _, ln_g| {
        let g = w * ln_v.exp();
        if g > 0.0 {
          mass += g;
          log += g * ln_g;
        }
      });
      if let Some(e) = panel.singular {
        log += self.ln_u_correction(&panel, e);
      }
      ControlFlow::<()>::Continue(())
    });
    (self.width.ln() + mass.ln()) - log / mass
  }

  /// The normalised density `k(x)/∫k` over `e^{ln_scale}` at a point of `[lo, up]`, whose support the law checks at
  /// its own bounds; a law narrower than any offset resolves is its reference point's.
  pub(super) fn density(&self, x: f64, ln_scale: f64) -> f64 {
    if let Some(c) = self.point() {
      return if x == c { f64::INFINITY } else { 0.0 };
    }
    let ln_g = self.kernel.ln_rel(x, span(self.centre, Cut::at(x)), None);
    (ln_g - (self.ln_norm + ln_scale)).exp()
  }

  /// The reference point of a law with no clipped width, whose spread and shape read NaN.
  pub(super) fn point(&self) -> Option<f64> {
    self.point.then_some(self.centre.x)
  }

  /// `∫_lo^x k / ∫k`, integrated up from `lo`: no survival function reads it, so a level near 1 needs absolute digits.
  pub(super) fn cdf(&self, x: f64) -> f64 {
    if x.is_nan() {
      return f64::NAN;
    }
    if x <= self.lo {
      return 0.0;
    }
    if let Some(c) = self.point() {
      return if x < c { 0.0 } else { 1.0 };
    }
    if x >= self.b.x {
      return 1.0;
    }
    if x < self.a.x {
      return self.below_clip(x);
    }
    ((self.tail_lo + self.integrate(self.a, Cut::at(x), 0.0)) / self.m0).min_or_nan(1.0)
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
    if let Some(c) = self.point() {
      return c;
    }
    self.invert(p)
  }

  /// The cdf below the lower clip: the mass from where the kernel is `e^{−CLIP}` below its value at `x`, relative to
  /// that value, which one `exp` restores, so a subnormal answer is rounded once.
  fn below_clip(&self, x: f64) -> f64 {
    let at = Cut::at(x);
    let ln_g = self.kernel.ln_rel(x, span(self.centre, at), None);
    // The kernel rises from `lo` to `x`, so the cdf is at most `k(x)(x − lo)/∫k`: where that underflows it is 0, and
    // an `ln_g` too large to keep its units digit never feeds the walk.
    if ln_g == f64::NEG_INFINITY || ln_g + (x - self.lo).ln() - self.ln_norm < -746.0 {
      return 0.0;
    }
    let start = clip(&self.kernel, self.centre, at, self.lo, ln_g - CLIP);
    let relative = self.integrate(start, at, ln_g);
    (ln_g + (relative / self.m0).ln()).exp().min_or_nan(1.0)
  }

  fn ratio(&self, x: f64, shift: f64) -> f64 {
    (self.kernel.ln_rel(x, span(self.centre, Cut::at(x)), None) - shift).exp()
  }

  /// `∫_l^r k/k(c)` in units of `width`, over `e^{shift}`, signed.
  fn integrate(&self, l: Cut, r: Cut, shift: f64) -> f64 {
    if r.x < l.x {
      return -self.integrate(r, l, shift);
    }
    let mut sum = 0.0;
    let _ = self.walk(l, r, false, |panel| {
      sum += self.panel_mass(&panel, shift);
      ControlFlow::<()>::Continue(())
    });
    sum
  }

  fn walk<B>(
    &self,
    l: Cut,
    r: Cut,
    rev: bool,
    visit: impl FnMut(Panel) -> ControlFlow<B>,
  ) -> ControlFlow<B> {
    panels(&self.kernel, self.centre, l, r, rev, visit)
  }

  fn panel_mass(&self, panel: &Panel, shift: f64) -> f64 {
    let mut mass = 0.0;
    self.nodes(panel, |w, ln_v, _, _| mass += w * (ln_v - shift).exp());
    mass
  }

  /// `f(w, ln v, y, ln g)` at each node: the weight over `width`, the integrand it multiplies, the offset from the
  /// centre and `ln k(x) − ln k(c)`; a singular panel's integrand is in `u`, so `v` is its smooth factor.
  fn nodes(&self, panel: &Panel, mut f: impl FnMut(f64, f64, f64, f64)) {
    if let Some(e) = panel.singular {
      return self.singular_nodes(panel, e, 0, None, f);
    }
    let half = 0.5 * (panel.r - panel.l);
    let scaled = half / self.width;
    let (to_centre, to_frame) = (
      self.centre.offset(panel.origin),
      self.frame.offset(panel.origin),
    );
    for (node, weight) in GL64_X.iter().zip(&GL64_W) {
      for xi in [-node, *node] {
        let t = panel.l + half * (1.0 + xi);
        let branch = panel.graded.then_some((panel.origin, t));
        let ln_g = self.kernel.ln_rel(panel.origin + t, t - to_centre, branch);
        f(weight * scaled, ln_g, t - to_frame, ln_g);
      }
    }
  }
}

/// Where `ln k − ln k(c)` falls below `floor` from `from` toward `end`, `end` if never: a doubling probe beside `c`, then
/// bisection beside `c` or, if the bracket lies nearer it, the branch point `end` is nearer, so the cut stays exact.
fn clip<K: Kernel>(kernel: &K, c: Cut, from: Cut, end: f64, floor: f64) -> Cut {
  let below = |origin: f64, t: f64| {
    let v = kernel.ln_rel(origin + t, t - c.offset(origin), Some((origin, t)));
    v < floor || v.is_nan()
  };
  if end.is_finite() && !below(end, 0.0) {
    return Cut::at(end);
  }
  let gap = from.offset(end);
  if gap == 0.0 {
    return Cut::at(end);
  }
  let (from_t, end_t, dir) = (from.offset(c.x), end - c.x, -gap.signum());
  let (mut inside, mut outside) = (from_t, end_t);
  // From the kernel's length or its e-fold length at `from`, whichever is shorter, and positive: doubling overflows
  // within 2100 steps.
  let local = 1.0 / kernel.d1(from.x, span(c, from)).abs();
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
    if below(c.x, t) {
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
  let near = points.find(|&(s, _)| (end - s).abs() < c.offset(end).abs());
  let (origin, mut inside, mut outside) = match near {
    Some((s, _)) if (outside - (s - c.x)).abs() < outside.abs() => (
      s,
      if inside == from_t {
        from.offset(s)
      } else {
        (c.x - s) + inside
      },
      if outside == end_t {
        end - s
      } else {
        (c.x - s) + outside
      },
    ),
    _ => (c.x, inside, outside),
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
  Cut::beside(origin, inside)
}

#[cfg(test)]
mod tests;

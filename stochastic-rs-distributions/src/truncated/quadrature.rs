//! Composite 64-node Gauss–Legendre quadrature of a truncated law's normalised kernel, behind all its analytics; a
//! singular end's innermost panel integrates in `u = (t/h)^e`, where `t^{e−1} dt = (h^e/e) du`.
//! Reference: Trefethen, L. N. (2008), "Is Gauss Quadrature Better than Clenshaw–Curtis?", *SIAM Review* 50(1), 67-87, DOI 10.1137/060659831 (the Bernstein-ellipse rate that sizes the panels).
//! Reference: Press, W. H., Teukolsky, S. A., Vetterling, W. T. & Flannery, B. P. (1992), *Numerical Recipes in C*, 2nd ed., Cambridge University Press, §4.4, ISBN 0-521-43108-5 (the change of variable at an integrable power-law end).

use std::ops::ControlFlow;

use clip::clip;
pub(super) use panels::Cut;
use panels::GL64_W;
use panels::GL64_X;
use panels::Panel;
use panels::panels;
pub(super) use panels::span;
use sum::Sum;
pub(super) use sum::two_prod;
pub(super) use sum::two_sum;

mod cdf;
mod clip;
mod panels;
mod quantile;
mod singular;
mod sum;

/// A law's log-kernel on its support, in the law's own coordinate, taken relative to a reference point `c`.
pub(super) trait Kernel {
  /// The law maps a quantile walked from a bound or a clip back from the double that point is held beside, so the
  /// engine reports it as `Point::From`; a law reading every point by its value pays nothing for it.
  const FROM_POINT: bool = false;

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

/// `½ − ln 2` correctly rounded (mpmath at 60 and 120 digits).
const HALF_LESS_LN_2: f64 = -0.19314718055994531;

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
/// for `|w| ≤ 1/3`, on `[−3/4, −1/2)` by halving `1 + z`; past that the plain difference loses under two bits.
pub(super) fn ln_1p_mx(z: f64) -> f64 {
  if (-0.75..-0.5).contains(&z) {
    // `ln(1 + z) = −ln 2 + ln(1 + v)`, `v = 2z + 1`: three terms of one sign, both shifts exact.
    return HALF_LESS_LN_2 + (z + 0.5) + ln_1p_mx(2.0 * z + 1.0);
  }
  let w = z / (2.0 + z);
  if w.abs() > 1.0 / 3.0 || w.is_nan() {
    // An offset past the doubles' range from its reference point: the kernel has vanished there.
    return if z == f64::INFINITY {
      f64::NEG_INFINITY
    } else {
      z.ln_1p() - z
    };
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

/// `ln(n/d)` from the quotient while it is normal, else `ln n − ln d`, whose `u·|ln n|` a large exponent multiplies.
#[inline]
pub(super) fn ln_ratio(n: f64, d: f64, ln_d: f64) -> f64 {
  let q = n / d;
  if q.is_normal() { q.ln() } else { n.ln() - ln_d }
}

/// `x·2^p` for any `p`, by normal factors, so `2^p` itself may lie past the doubles (θ above 2¹⁰²³).
pub(super) fn ldexp(mut x: f64, mut p: i32) -> f64 {
  let two = |p: i32| f64::from_bits(((p + 1023) as u64) << 52);
  while p > 1023 {
    x *= two(1023);
    p -= 1023;
  }
  while p < -1022 {
    x *= two(-1022);
    p += 1022;
  }
  x * two(p)
}

/// `x = m·2^e`, `m` in `[1, 2)`, for a positive finite `x`.
pub(super) fn split(x: f64) -> (f64, i32) {
  let scaled = if x < f64::MIN_POSITIVE {
    x * 2f64.powi(64)
  } else {
    x
  };
  let e = ((scaled.to_bits() >> 52) & 0x7ff) as i32 - 1023;
  let e = if x < f64::MIN_POSITIVE { e - 64 } else { e };
  (ldexp(x, -e), e)
}

/// Relative kernel below `e^{−CLIP}` is dropped: 1e-18 of the mass near the reference point.
const CLIP: f64 = 41.5;

/// A position the engine reports: a point, or one beside the branch point `s` whose offset `sign·e^{ln_t}` no double
/// may hold.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Point {
  At(f64),
  /// `s + t`, the offset from the double the walk's origin is held beside kept apart, so a law mapping it back from
  /// that double keeps its digits.
  From {
    s: f64,
    t: f64,
  },
  Beside {
    s: f64,
    sign: f64,
    ln_t: f64,
  },
}

impl Point {
  pub(super) fn value(self) -> f64 {
    match self {
      Self::At(x) => x,
      Self::From { s, t } => s + t,
      Self::Beside { s, sign, ln_t } => s + sign * ln_t.exp(),
    }
  }
}

/// An interval's reference point, clipped ends and normalising sums, cached so a pdf costs one `exp`.
#[derive(Clone, Debug)]
pub(super) struct Truncation<K> {
  kernel: K,
  lo: f64,
  up: f64,
  /// The point every offset is formed from, held exactly: the clamped mode, a tenth of a scale off a singular end, or
  /// the midpoint.
  centre: Cut,
  /// The point the moments are summed about: the singular branch point the centre steps off, where the mass may sit
  /// closer than any offset from the centre resolves, else the centre.
  frame: Cut,
  /// The centre's double and the rest of its exact value, so a point's offset from it is two subtractions.
  near: (f64, f64),
  /// The ends moved in to where the kernel is `e^{−CLIP}` below its value at `centre`, each an exact offset from the
  /// point it was found from, however much finer than an ulp; a singular end stays.
  a: Cut,
  b: Cut,
  /// `b − a`, the unit of every sum: no power of the law's own scale leaves the range.
  width: f64,
  /// `∫_a^b k/k(centre)`, in units of `width`, and the rest of its compensated sum.
  m0: f64,
  m0_rest: f64,
  /// `ln(width·m0)`, so a density costs one `exp` and overflows only where its value does.
  ln_norm: f64,
  /// No normal width: no panel places its nodes, and the law answers in its own scale (`truncated::local`, the beta's
  /// gamma route); the point this engine then reads is for a law neither holds.
  narrow: bool,
  /// `∫_lo^a` and `∫_b^up` of `k/k(centre)`, in units of `width`: the cdf adds the first, the quantile both.
  tail_lo: f64,
  tail_up: f64,
  /// The ends those tails are summed from, `e^{−2·CLIP}` below `centre`: the moments sum out to them, as the first
  /// clip leaves `e^{−CLIP}·CLIP⁴` of a fourth moment, past 1e-13 of it.
  reach: (Cut, Cut),
}

impl<K: Kernel> Truncation<K> {
  /// `kernel` on `[lo, up]`, `lo < up`, with the mass beyond the clips that the cdf and the quantile read.
  pub(super) fn new(kernel: K, lo: f64, up: f64) -> Self {
    let mut t = Self::mass_only(kernel, lo, up);
    if t.a.offset(lo) > 0.0 {
      t.reach.0 = clip(&t.kernel, t.centre, t.a, lo, -2.0 * CLIP);
      t.tail_lo = t.integrate(t.reach.0, t.a, 0.0);
    }
    if t.b.offset(up) < 0.0 {
      t.reach.1 = clip(&t.kernel, t.centre, t.b, up, -2.0 * CLIP);
      t.tail_up = t.integrate(t.b, t.reach.1, 0.0);
    }
    debug_assert!(
      t.narrow || (t.m0 > 0.0 && t.m0.is_finite()),
      "a resolvable width {:e} with mass {:e}",
      t.width,
      t.m0
    );
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
    let (past_lo, past_up) = (mode.offset(lo), mode.offset(up));
    let mut at_mode = past_lo >= 0.0 && past_up <= 0.0;
    let mut centre = if past_lo < 0.0 {
      Cut::at(lo)
    } else if past_up > 0.0 {
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
      near: centre.near(),
      a,
      b,
      width: span(a, b),
      m0: 0.0,
      m0_rest: 0.0,
      ln_norm: 0.0,
      narrow: false,
      tail_lo: 0.0,
      tail_up: 0.0,
      reach: (a, b),
    };
    t.narrow = t.width.is_nan() || t.width < f64::MIN_POSITIVE;
    if !t.narrow {
      let m0 = t.integrate_sum(a, b, 0.0);
      (t.m0, t.m0_rest) = (m0.value(), (m0.hi - m0.value()) + m0.lo);
      t.ln_norm = t.width.ln() + t.m0.ln();
    }
    t
  }

  /// The reference point `c`, whose `ln k(c)` the masses are relative to.
  pub(super) fn centre(&self) -> f64 {
    self.centre.x
  }

  pub(super) fn kernel(&self) -> &K {
    &self.kernel
  }

  /// The clipped ends, outside which the kernel holds less mass than the sums resolve.
  pub(super) fn clipped(&self) -> (f64, f64) {
    (self.a.x, self.b.x)
  }

  /// `∫_a^b k/k(c)` over the clipped interval.
  pub(super) fn mass(&self) -> f64 {
    self.width * self.m0
  }

  /// The reference point of a law with no normal width, which the law answers through its local model.
  pub(super) fn narrow(&self) -> Option<f64> {
    self.narrow.then_some(self.centre.x)
  }

  /// `(mean, scale²·variance, skewness, excess kurtosis)`; an interval that cuts neither end removes less than the clip
  /// resolves and reads the base's closed forms, keeping relative digits sums lose (2/√k at k = 1e300).
  pub(super) fn moments(&self, scale: f64) -> [f64; 4] {
    let (lo, up) = self.kernel.support();
    // By exact offsets: a clip within an ulp of its bound still cuts nothing.
    if (self.lo <= lo || self.a.offset(self.lo) > 0.0)
      && (self.up >= up || self.b.offset(self.up) < 0.0)
    {
      let [mean, sd, skewness, kurtosis] = self.kernel.base_moments();
      let unit = scale * sd;
      return [mean, unit * unit, skewness, kurtosis];
    }
    if self.narrow {
      return [self.centre.x, 0.0, f64::NAN, f64::NAN];
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
    [
      self.frame.plus(self.width * (d + e)),
      (unit * mu2) * unit,
      (mu3 / mu2) / mu2.sqrt(),
      (mu4 / mu2) / mu2 - 3.0,
    ]
  }

  fn moment_sums<const N: usize>(&self, about: f64) -> [f64; N] {
    let mut m = [0.0; N];
    let _ = self.walk_about_centre(|panel| {
      if let Some(e) = panel
        .singular
        .filter(|_| !Self::powers_on_mass_nodes(&panel, N))
      {
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
    if self.narrow {
      return f64::NAN;
    }
    // Relative to `m0`: past 1e154 a mass and its log's own size overflow in one product (`β = 4.9e-165` at 1).
    let shift = self.m0.ln();
    let (mut mass, mut log) = (0.0, 0.0);
    let _ = self.walk_about_centre(|panel| {
      self.nodes(&panel, |w, ln_v, _, ln_g| {
        let g = w * (ln_v - shift).exp();
        if g > 0.0 {
          mass += g;
          log += g * ln_g;
        }
      });
      if let Some(e) = panel.singular {
        log += self.ln_u_correction(&panel, e, shift);
      }
      ControlFlow::<()>::Continue(())
    });
    (self.width.ln() + shift + mass.ln()) - log / mass
  }

  /// The normalised density `k(x)/∫k` over `e^{ln_scale}` at a point of `[lo, up]`, whose support the law checks at
  /// its own bounds.
  pub(super) fn density(&self, x: f64, ln_scale: f64) -> f64 {
    if self.narrow {
      return if x == self.centre.x {
        f64::INFINITY
      } else {
        0.0
      };
    }
    let ln_g = self.kernel.ln_rel(x, self.offset(x), None);
    (ln_g - (self.ln_norm + ln_scale)).exp()
  }

  /// The density at `s + e^{ln_t}` beside the branch point `s` of exponent `e`, an offset no double holds: the power
  /// law's log apart from the kernel's smooth factor.
  pub(super) fn density_beside(&self, s: f64, e: f64, ln_t: f64, ln_scale: f64) -> f64 {
    let t = ln_t.exp();
    let y = t - span(Cut::at(s), self.centre);
    let ln_g = (e - 1.0) * ln_t + self.kernel.ln_rel_regular(s + t, y, s);
    (ln_g - (self.ln_norm + ln_scale)).exp()
  }

  /// `cdf⁻¹(p)`: the bounds at 0 and 1; else the panel holding `p`, walked to from the nearer end, then Newton in it.
  pub(super) fn quantile(&self, p: f64) -> Point {
    if p.is_nan() {
      return Point::At(f64::NAN);
    }
    if p <= 0.0 {
      return Point::At(self.lo);
    }
    if p >= 1.0 {
      return Point::At(self.up);
    }
    if self.narrow {
      return Point::At(self.centre.x);
    }
    self.invert(p)
  }

  /// `x − c` exact where `x` lies within a factor 2 of the centre's double, as `span` forms it.
  #[inline]
  fn offset(&self, x: f64) -> f64 {
    (x - self.near.0) - self.near.1
  }

  /// `∫_l^r k/k(c)` in units of `width`, over `e^{shift}`, signed.
  fn integrate(&self, l: Cut, r: Cut, shift: f64) -> f64 {
    if span(l, r) < 0.0 {
      return -self.integrate(r, l, shift);
    }
    self.integrate_sum(l, r, shift).value()
  }

  /// The sum `integrate` rounds, over `l < r`.
  fn integrate_sum(&self, l: Cut, r: Cut, shift: f64) -> Sum {
    let mut sum = Sum::default();
    let _ = self.walk(l, r, false, |panel| {
      sum.add(self.panel_mass(&panel, shift));
      ControlFlow::<()>::Continue(())
    });
    sum
  }

  /// The masses' walk: each piece from the end the walk starts at.
  fn walk<B>(
    &self,
    l: Cut,
    r: Cut,
    rev: bool,
    visit: impl FnMut(Panel) -> ControlFlow<B>,
  ) -> ControlFlow<B> {
    panels(&self.kernel, self.centre, (l, r), rev, true, visit)
  }

  /// The moments' and entropy's walk: about the centre, where a near-symmetric law's odd powers keep their digits.
  fn walk_about_centre<B>(&self, visit: impl FnMut(Panel) -> ControlFlow<B>) -> ControlFlow<B> {
    panels(&self.kernel, self.centre, self.reach, false, false, visit)
  }

  /// `Σ w·e^{ln v − shift}` over a panel's nodes; on a regular panel its own loop, whose few live values leave the sum
  /// in a register across the kernel's calls.
  fn panel_mass(&self, panel: &Panel, shift: f64) -> f64 {
    let mut mass = 0.0;
    if panel.singular.is_some() || panel.graded {
      self.nodes(panel, |w, ln_v, _, _| mass += w * (ln_v - shift).exp());
      return mass;
    }
    let half = 0.5 * (panel.r - panel.l);
    let (scaled, to_centre) = (half / self.width, span(panel.origin, self.centre));
    for (node, weight) in GL64_X.iter().zip(&GL64_W) {
      for xi in [-node, *node] {
        let t = panel.l + half * (1.0 + xi);
        let ln_g = self.kernel.ln_rel(panel.origin.x + t, t - to_centre, None);
        mass += weight * scaled * (ln_g - shift).exp();
      }
    }
    mass
  }

  /// `f(w, ln v, y, ln g)` at each node: a weight and the log of the integrand it multiplies, in units of `width`, the
  /// offset from the frame and `ln k(x) − ln k(c)`; a singular panel's integrand is in `u`, so `v` is its smooth factor.
  fn nodes(&self, panel: &Panel, mut f: impl FnMut(f64, f64, f64, f64)) {
    if let Some(e) = panel.singular {
      return self.singular_nodes(panel, e, 0, None, f);
    }
    let half = 0.5 * (panel.r - panel.l);
    // Graded, in logs: deep beside a singular end the panel's share of `width` underflows where its kernel ratio
    // overflows.
    let (scaled, ln_scaled) = if panel.graded {
      (1.0, half.ln() - self.width.ln())
    } else {
      (half / self.width, 0.0)
    };
    let (to_centre, to_frame) = (
      span(panel.origin, self.centre),
      span(panel.origin, self.frame),
    );
    for (node, weight) in GL64_X.iter().zip(&GL64_W) {
      for xi in [-node, *node] {
        let t = panel.l + half * (1.0 + xi);
        let branch = panel.graded.then_some((panel.origin.x, t));
        let ln_g = self
          .kernel
          .ln_rel(panel.origin.x + t, t - to_centre, branch);
        f(weight * scaled, ln_g + ln_scaled, t - to_frame, ln_g);
      }
    }
  }
}

#[cfg(test)]
mod tests;

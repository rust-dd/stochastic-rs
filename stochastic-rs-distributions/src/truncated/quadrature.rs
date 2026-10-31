//! Composite 64-node Gauss–Legendre quadrature of a truncated law's normalised kernel, behind all its analytics; a
//! singular end's innermost panel integrates in `u = (t/h)^{p+1}`, where `t^p dt = h^{p+1}/(p+1) du`.
//! Reference: Trefethen, L. N. (2008), "Is Gauss Quadrature Better than Clenshaw–Curtis?", *SIAM Review* 50(1), 67-87, DOI 10.1137/060659831 (the Bernstein-ellipse rate that sizes the panels).

use std::ops::ControlFlow;

use panels::GL64_W;
use panels::GL64_X;
use panels::Panel;
use panels::panels;

mod panels;
mod quantile;

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

  /// `ln_rel` less `p·ln|x − s|` at the branch point `s`: the smooth factor, read where `x − s` may round to 0.
  fn ln_rel_regular(&self, x: f64, y: f64, c: f64, _s: f64) -> f64 {
    self.ln_rel(x, y, c, None)
  }

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
    let step = (up - lo).min(kernel.length_scale()) / 10.0;
    // `p` of a singular branch point at a point or within `step` of it, where the power law rules the kernel's mass.
    let singular = |x: f64| {
      let mut points = kernel.branch_points().into_iter().flatten();
      points
        .find(|&(s, p)| p < 0.0 && (x - s).abs() <= step)
        .map(|(_, p)| p)
    };
    let (lo_singular, up_singular) = (singular(lo), singular(up));
    let mut centre = kernel.mode().clamp(lo, up);
    if singular(centre).is_some() {
      let heavier_lo = match (lo_singular, up_singular) {
        (Some(p), Some(q)) => p <= q,
        (_, None) => true,
        (None, Some(_)) => false,
      };
      centre = if heavier_lo { lo + step } else { up - step };
    }
    let a = if lo_singular.is_some() {
      lo
    } else {
      clip(&kernel, centre, centre, lo, -CLIP)
    };
    let b = if up_singular.is_some() {
      up
    } else {
      clip(&kernel, centre, centre, up, -CLIP)
    };
    let drop = |e: f64| -kernel.ln_rel(e, e - centre, centre, None);
    // A flat kernel's odd moments about an end lose digits; about the midpoint they do not.
    if lo_singular.is_none() && up_singular.is_none() && drop(a).max(drop(b)) < 1.0 {
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

  /// `(mean, scale²·variance, skewness, excess kurtosis)` in the kernel's coordinate, from sums in units of `width`:
  /// the mean's offset `d`, then sums about it, so no binomial shift by a far reference point cancels.
  pub(super) fn moments(&self, scale: f64) -> [f64; 4] {
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
      self.centre + self.width * (d + e),
      (unit * mu2) * unit,
      mu3 / mu2.powf(1.5),
      mu4 / (mu2 * mu2) - 3.0,
    ]
  }

  /// `Σ w v (y/width − about)^j` over the clipped interval, `j < N`.
  fn moment_sums<const N: usize>(&self, about: f64) -> [f64; N] {
    let mut m = [0.0; N];
    let _ = self.walk(self.a, self.b, false, |panel| {
      self.nodes(&panel, |w, ln_v, y, _| {
        let (g, z) = (w * ln_v.exp(), y / self.width - about);
        let mut zj = 1.0;
        for slot in &mut m {
          *slot += g * zj;
          zj *= z;
        }
      });
      ControlFlow::<()>::Continue(())
    });
    m
  }

  /// `−∫ f ln f` as `ln M₀ − E[ln k(X) − ln k(c)]`, so no large `ln k(c)` cancels against `E[ln k(X)]`.
  pub(super) fn entropy(&self) -> f64 {
    let (mut mass, mut log) = (0.0, 0.0);
    let _ = self.walk(self.a, self.b, false, |panel| {
      self.nodes(&panel, |w, ln_v, _, ln_g| {
        let g = w * ln_v.exp();
        if g > 0.0 {
          mass += g;
          log += g * ln_g;
        }
      });
      if let Some(p) = panel.singular {
        log += self.ln_u_correction(&panel, p);
      }
      ControlFlow::<()>::Continue(())
    });
    (self.width.ln() + mass.ln()) - log / mass
  }

  /// On a singular panel `ln g` carries `(p/(p+1)) ln u`, whose endpoint log the rule cannot integrate: the rule's
  /// error on `ln u` against `[u ln u − u]`, times the smooth factor at the near end.
  fn ln_u_correction(&self, panel: &Panel, p: f64) -> f64 {
    let frame = Frame::of(panel, p);
    let u0 = frame.u_near();
    let half = 0.5 * (1.0 - u0);
    let rule = GL64_X
      .iter()
      .zip(&GL64_W)
      .map(|(node, weight)| {
        weight * half * ((u0 + half * (1.0 - node)).ln() + (u0 + half * (1.0 + node)).ln())
      })
      .sum::<f64>();
    let exact = if u0 > 0.0 {
      -1.0 - (u0 * u0.ln() - u0)
    } else {
      -1.0
    };
    let v0 = self.singular_point(panel, &frame, u0).0;
    frame.jacobian() / self.width * (p / frame.e) * v0.exp() * (exact - rule)
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
      sum += self.panel_mass(&panel, shift);
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

  /// `Σ_i w_i v_i / e^{shift}` on one panel: its mass in units of `width`.
  fn panel_mass(&self, panel: &Panel, shift: f64) -> f64 {
    let mut mass = 0.0;
    self.nodes(panel, |w, ln_v, _, _| mass += w * (ln_v - shift).exp());
    mass
  }

  /// `f(w, ln v, y, ln g)` at each node: the weight over `width`, the integrand it multiplies, the offset from the
  /// centre and `ln k(x) − ln k(c)`; a singular panel's integrand is in `u`, so `v` is its smooth factor.
  fn nodes(&self, panel: &Panel, mut f: impl FnMut(f64, f64, f64, f64)) {
    if let Some(p) = panel.singular {
      let u_near = Frame::of(panel, p).u_near();
      return self.singular_nodes(panel, p, (u_near, 1.0), f);
    }
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
        let ln_g = self.kernel.ln_rel(x, y, self.centre, branch);
        f(weight * scaled, ln_g, y, ln_g);
      }
    }
  }

  /// A singular panel's nodes over `u ∈ [u0, u1]`, `|x − s| = h·u^{1/(p+1)}` from its branch point `s`, each weight
  /// times `h^{p+1}/(p+1)`, the Jacobian that `t^p` leaves.
  fn singular_nodes(
    &self,
    panel: &Panel,
    p: f64,
    (u0, u1): (f64, f64),
    mut f: impl FnMut(f64, f64, f64, f64),
  ) {
    let frame = Frame::of(panel, p);
    let jacobian = frame.jacobian() / self.width;
    let half = 0.5 * (u1 - u0);
    for (node, weight) in GL64_X.iter().zip(&GL64_W) {
      for xi in [-node, *node] {
        let (ln_v, y, ln_t) = self.singular_point(panel, &frame, u0 + half * (1.0 + xi));
        f(weight * half * jacobian, ln_v, y, ln_v + p * ln_t);
      }
    }
  }

  /// `(ln v, y, ln|x − s|)` at `u`: `|x − s|` is formed only through its logarithm, so a point next to `s` keeps its
  /// smooth factor however small `u` is.
  fn singular_point(&self, panel: &Panel, frame: &Frame, u: f64) -> (f64, f64, f64) {
    let ln_t = frame.ln_far + u.ln() / frame.e;
    let t = frame.sign * ln_t.exp();
    let (x, y) = (panel.origin + t, (panel.origin - self.centre) + t);
    let ln_v = self.kernel.ln_rel_regular(x, y, self.centre, panel.origin);
    (ln_v, y, ln_t)
  }
}

/// A singular panel seen from its branch point: the side it lies on, its ends' distances and `e = p + 1`.
struct Frame {
  sign: f64,
  near: f64,
  ln_far: f64,
  e: f64,
}

impl Frame {
  fn of(panel: &Panel, p: f64) -> Self {
    let (near, far, sign) = if panel.r > 0.0 {
      (panel.l, panel.r, 1.0)
    } else {
      (-panel.r, -panel.l, -1.0)
    };
    Self {
      sign,
      near,
      ln_far: far.ln(),
      e: p + 1.0,
    }
  }

  /// `u` at the near end, `(near/far)^{p+1}`.
  fn u_near(&self) -> f64 {
    (self.e * (self.near.ln() - self.ln_far)).exp()
  }

  /// `h^{p+1}/(p+1)`, `h` the far end's distance.
  fn jacobian(&self) -> f64 {
    (self.e * self.ln_far - self.e.ln()).exp()
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

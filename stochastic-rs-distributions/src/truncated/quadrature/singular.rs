//! The singular panel: an end with `k ~ |t|^{e−1}` integrates in `u = (|t|/h)^e`, each power of `t` the moments need
//! in its own `u = (|t|/h)^{e+i}`, and the entropy's `ln u` endpoint corrected against its exact integral.

use super::GL64_W;
use super::GL64_X;
use super::Kernel;
use super::Panel;
use super::Point;
use super::Truncation;
use super::ln_ratio;
use super::panels::Cut;
use super::panels::SLOPE_BUDGET;
use super::panels::span;

/// `2⁻¹⁰⁰⁰`: below it `e·ln(far/near)` times the rule's smallest node offset leaves the normal doubles.
const LOG_FORM: f64 = 9.332636185032189e-302;

impl<K: Kernel> Truncation<K> {
  /// A singular panel's entropy sums `(Σ g, Σ g·ln g, Σ g·ln u)`: in `u` `ln g` carries `((e − 1)/e)·ln u`, whose
  /// coefficient passes the doubles below `e ≈ 4.5e-308`, so that sum is kept apart.
  pub(super) fn singular_entropy(&self, panel: &Panel, e: f64, shift: f64) -> (f64, f64, f64) {
    let frame = Frame::of(panel, e);
    let plain = !(frame.reflect || frame.log);
    let (near, far) = (frame.q_near, frame.q_far());
    let (q0, w) = (near.min(far), (far - near).abs());
    let ln_jacobian = frame.ln_jacobian() - self.width.ln();
    let (mut mass, mut log, mut ulog) = (0.0, 0.0, 0.0);
    for (node, weight) in GL64_X.iter().zip(&GL64_W) {
      for xi in [-node, *node] {
        let q = q0 + 0.5 * w * (1.0 + xi);
        let (ln_v, _, ln_t) = self.singular_point(panel.origin, &frame, q);
        let g = weight * 0.5 * w * (ln_v + ln_jacobian - shift).exp();
        if g > 0.0 {
          mass += g;
          if plain {
            log += g * (ln_v + (e - 1.0) * frame.ln_far);
            ulog += g * q.ln();
          } else {
            log += g * (ln_v + (e - 1.0) * ln_t);
          }
        }
      }
    }
    if plain {
      ulog += self.ln_u_correction(panel.origin, &frame, ln_jacobian - shift);
    }
    (mass, log, ulog)
  }

  /// The rule's error on `ln u`'s endpoint log against `[u ln u − u]`, times the smooth factor at the near end and
  /// `e^{ln_scale}`.
  fn ln_u_correction(&self, origin: Cut, frame: &Frame, ln_scale: f64) -> f64 {
    let u0 = frame.q_near;
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
    let v0 = self.singular_point(origin, frame, u0).0;
    let gap = exact - rule;
    gap.signum() * (ln_scale + v0 + gap.abs().ln()).exp()
  }

  /// A singular panel whose near end keeps `(t/far)^{n−1}` within the slope budget sums its `n` powers on the mass's
  /// nodes: a frame per power would set its own branch point at `(near/far)^{e+i}`, beside the panel.
  pub(super) fn powers_on_mass_nodes(panel: &Panel, n: usize) -> bool {
    let (near, far) = if panel.r > 0.0 {
      (panel.l, panel.r)
    } else {
      (-panel.r, -panel.l)
    };
    (near / far).ln() * (n - 1) as f64 >= -SLOPE_BUDGET
  }

  /// A singular panel's share of `Σ g (z − about)^j` from its sums of `(t/width)^i`, each in its own `u = (|t|/h)^{e+i}`:
  /// in `u = (|t|/h)^e` a small `e` turns `t^i` into a power of `u` past what the rule integrates.
  pub(super) fn singular_moments<const N: usize>(
    &self,
    panel: &Panel,
    e: f64,
    about: f64,
    m: &mut [f64; N],
  ) {
    let mut raw = [0.0; N];
    for (i, r) in raw.iter_mut().enumerate() {
      self.singular_nodes(panel, e, i as i32, None, |w, ln_v, _, _| {
        *r += w * (ln_v - self.scale).exp()
      });
    }
    let delta = -span(panel.origin, self.frame) / self.width - about;
    for (j, slot) in m.iter_mut().enumerate() {
      let mut binomial = 1.0;
      for (i, r) in raw.iter().enumerate().take(j + 1) {
        *slot += binomial * r * delta.powi((j - i) as i32);
        binomial *= (j - i) as f64 / (i + 1) as f64;
      }
    }
  }

  /// A singular panel's nodes over `q` (the whole panel when `range` is `None`) for `|t|^{e−1}(t/width)^i`, which
  /// integrates in `u = (|t|/h)^{e+i}`: each weight carries that power's Jacobian `h^{e+i}/(e+i)` over `width^{i+1}`.
  pub(super) fn singular_nodes(
    &self,
    panel: &Panel,
    e: f64,
    i: i32,
    range: Option<(f64, f64)>,
    f: impl FnMut(f64, f64, f64, f64),
  ) {
    let frame = Frame::of(panel, e + f64::from(i));
    self.frame_nodes(panel.origin, &frame, e, i, range, f);
  }

  /// `singular_nodes` over a frame beside the branch point `origin`, whose far end may be given by its log alone;
  /// `range` is a start in `q` and a signed width, so a sliver beside either end keeps its width's digits.
  pub(super) fn frame_nodes(
    &self,
    origin: Cut,
    frame: &Frame,
    e: f64,
    i: i32,
    range: Option<(f64, f64)>,
    mut f: impl FnMut(f64, f64, f64, f64),
  ) {
    let (q0, w) = range.unwrap_or_else(|| {
      let (near, far) = (frame.q_near, frame.q_far());
      (near.min(far), (far - near).abs())
    });
    // In logs, so a Jacobian that overflows alone still meets its smooth factor.
    let ln_jacobian = frame.ln_jacobian() - f64::from(i + 1) * self.width.ln();
    let half = 0.5 * w.abs() * frame.sign.powi(i);
    for (node, weight) in GL64_X.iter().zip(&GL64_W) {
      for xi in [-node, *node] {
        let q = q0 + 0.5 * w * (1.0 + xi);
        let (ln_v, offset, ln_t) = self.singular_point(origin, frame, q);
        f(
          weight * half,
          ln_v + ln_jacobian,
          offset,
          ln_v + (e - 1.0) * ln_t,
        );
      }
    }
  }

  /// `(ln v, x − frame, ln|x − s|)` at `q`: `|x − s|` is formed only through its logarithm, so a point next to `s`
  /// keeps its smooth factor however small `u` is.
  pub(super) fn singular_point(&self, origin: Cut, frame: &Frame, q: f64) -> (f64, f64, f64) {
    let ln_t = frame.ln_t(q);
    let t = frame.sign * ln_t.exp();
    let (x, y) = (origin.plus(t), t - span(origin, self.centre));
    let ln_v = self.kernel.ln_rel_regular(x, y, origin.x);
    (ln_v, t - span(origin, self.frame), ln_t)
  }
}

/// A singular panel seen from its branch point: the side it lies on, its far end's distance and its exponent `e`. It
/// integrates in `q`, which is `u` or, where `u` stays above ½ (a small `e`), `1 − u`, so `u` beside 1 keeps its digits.
pub(super) struct Frame {
  pub(super) sign: f64,
  pub(super) ln_far: f64,
  pub(super) e: f64,
  reflect: bool,
  /// `q = ln(|t|/far)`, the `e → 0` limit of `(u − 1)/e`, where `e·ln(far/near)` passes below 2⁻¹⁰⁰⁰: `u`'s range would
  /// hold no node's digits, and `t^e` is 1 to the last bit.
  pub(super) log: bool,
  pub(super) q_near: f64,
  /// The ends' distances from the branch point and `u` at the near one, which a point beside an end is formed from.
  ends: (f64, f64),
  u_near: f64,
}

impl Frame {
  pub(super) fn of(panel: &Panel, e: f64) -> Self {
    let (near, far, sign) = if panel.r > 0.0 {
      (panel.l, panel.r, 1.0)
    } else {
      (-panel.r, -panel.l, -1.0)
    };
    let ln_far = far.ln();
    // From the exact `near − far` where the ends lie within a factor 2, so a narrow panel's `q` keeps its digits.
    let ln_ratio = if near > 0.5 * far {
      ((near - far) / far).ln_1p()
    } else {
      ln_ratio(near, far, ln_far)
    };
    let ln_u_near = e * ln_ratio;
    let log = near > 0.0 && -ln_u_near < LOG_FORM;
    let reflect = !log && ln_u_near > -std::f64::consts::LN_2;
    Self {
      sign,
      ln_far,
      e,
      reflect,
      log,
      q_near: if log {
        ln_ratio
      } else if reflect {
        -ln_u_near.exp_m1()
      } else {
        ln_u_near.exp()
      },
      ends: (near, far),
      u_near: ln_u_near.exp(),
    }
  }

  /// The panel from the branch point out to `e^{ln_far}`, an offset no double need hold.
  pub(super) fn beside(ln_far: f64, e: f64) -> Self {
    Self {
      sign: 1.0,
      ln_far,
      e,
      reflect: false,
      log: false,
      q_near: 0.0,
      ends: (0.0, f64::NAN),
      u_near: 0.0,
    }
  }

  /// The point `w` along `q` from `qa`, the end a walk started at: from that end's double where it is one, so the
  /// point keeps the digits its distance from the end has, else through its log.
  pub(super) fn point(&self, origin: Cut, from_near: bool, (qa, w): (f64, f64)) -> Point {
    let (base, rate) = match (from_near, self.log) {
      (true, true) => (self.ends.0, w.abs()),
      (false, true) => (self.ends.1, -w.abs()),
      (true, false) => {
        let u = self.u_near;
        (
          if u.is_normal() { self.ends.0 } else { 0.0 },
          (w.abs() / u).ln_1p() / self.e,
        )
      }
      (false, false) => (self.ends.1, (-w.abs()).ln_1p() / self.e),
    };
    // Within a factor 2 of the end its offset from it, `base·(e^r − 1)`, carries the digits; past that `e^r` alone.
    let t = if rate.abs() < std::f64::consts::LN_2 {
      base + base * rate.exp_m1()
    } else {
      base * rate.exp()
    };
    if base.is_normal() && t.is_normal() {
      return Point::At(origin.plus(self.sign * t));
    }
    Point::Beside {
      s: origin.x,
      sign: self.sign,
      ln_t: self.ln_t(qa + w),
    }
  }

  /// `q` at the offset `t` from the branch point, its `ln(|t|/far)` formed as `of` forms the near end's.
  pub(super) fn q_at(&self, t: f64) -> f64 {
    let (n, far) = (t.abs(), self.ends.1);
    let ln_r = if n > 0.5 * far {
      ((n - far) / far).ln_1p()
    } else {
      ln_ratio(n, far, self.ln_far)
    };
    if self.log {
      return ln_r;
    }
    let ln_u = self.e * ln_r;
    if self.reflect {
      -ln_u.exp_m1()
    } else {
      ln_u.exp()
    }
  }

  pub(super) fn q_far(&self) -> f64 {
    if self.reflect || self.log { 0.0 } else { 1.0 }
  }

  pub(super) fn ln_t(&self, q: f64) -> f64 {
    if self.log {
      self.ln_far + q
    } else if self.reflect {
      self.ln_far + (-q).ln_1p() / self.e
    } else {
      self.ln_far + q.ln() / self.e
    }
  }

  /// `ln(dt^e/(e·dq))`'s constant: `far^e/e` in `u`; in `ln t` the density `|t|^e` alone, its `e·q` below 2⁻¹⁰⁰⁰.
  pub(super) fn ln_jacobian(&self) -> f64 {
    if self.log {
      self.e * self.ln_far
    } else {
      self.e * self.ln_far - self.e.ln()
    }
  }
}

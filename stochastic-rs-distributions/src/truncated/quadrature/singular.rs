//! The singular panel: an end with `k ~ |t|^{e−1}` integrates in `u = (|t|/h)^e`, each power of `t` the moments need
//! in its own `u = (|t|/h)^{e+i}`, and the entropy's `ln u` endpoint corrected against its exact integral.

use super::GL64_W;
use super::GL64_X;
use super::Kernel;
use super::Panel;
use super::Truncation;
use super::panels::Cut;
use super::panels::span;

impl<K: Kernel> Truncation<K> {
  /// On a singular panel `ln g` carries `((e − 1)/e) ln u`, whose endpoint log the rule cannot integrate: the rule's
  /// error on `ln u` against `[u ln u − u]`, times the smooth factor at the near end, over `e^{shift}`.
  pub(super) fn ln_u_correction(&self, panel: &Panel, e: f64, shift: f64) -> f64 {
    let frame = Frame::of(panel, e);
    if frame.reflect {
      return 0.0;
    }
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
    let v0 = self.singular_point(panel.origin, &frame, u0).0;
    // In logs: `(e − 1)/e` and the Jacobian each overflow alone for a tiny `e`.
    let gap = exact - rule;
    let ln =
      (1.0 - e).ln() - e.ln() + frame.ln_jacobian() - self.width.ln() + v0 + gap.abs().ln() - shift;
    -gap.signum() * ln.exp()
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
        *r += w * ln_v.exp()
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

  /// `singular_nodes` over a frame beside the branch point `origin`, whose far end may be given by its log alone.
  pub(super) fn frame_nodes(
    &self,
    origin: Cut,
    frame: &Frame,
    e: f64,
    i: i32,
    range: Option<(f64, f64)>,
    mut f: impl FnMut(f64, f64, f64, f64),
  ) {
    let (q0, q1) = range.unwrap_or_else(|| {
      let (near, far) = (frame.q_near, frame.q_far());
      (near.min(far), near.max(far))
    });
    // In logs, so a Jacobian that overflows alone still meets its smooth factor.
    let ln_jacobian = frame.ln_jacobian() - f64::from(i + 1) * self.width.ln();
    let half = 0.5 * (q1 - q0) * frame.sign.powi(i);
    for (node, weight) in GL64_X.iter().zip(&GL64_W) {
      for xi in [-node, *node] {
        let q = q0 + 0.5 * (q1 - q0) * (1.0 + xi);
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
    let ln_t = frame.ln_far + frame.ln_u(q) / frame.e;
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
  pub(super) q_near: f64,
}

impl Frame {
  pub(super) fn of(panel: &Panel, e: f64) -> Self {
    let (near, far, sign) = if panel.r > 0.0 {
      (panel.l, panel.r, 1.0)
    } else {
      (-panel.r, -panel.l, -1.0)
    };
    let ln_far = far.ln();
    let ln_u_near = e * (near.ln() - ln_far);
    let reflect = ln_u_near > -std::f64::consts::LN_2;
    Self {
      sign,
      ln_far,
      e,
      reflect,
      q_near: if reflect {
        -ln_u_near.exp_m1()
      } else {
        ln_u_near.exp()
      },
    }
  }

  /// The panel from the branch point out to `e^{ln_far}`, an offset no double need hold.
  pub(super) fn beside(ln_far: f64, e: f64) -> Self {
    Self {
      sign: 1.0,
      ln_far,
      e,
      reflect: false,
      q_near: 0.0,
    }
  }

  pub(super) fn q_far(&self) -> f64 {
    if self.reflect { 0.0 } else { 1.0 }
  }

  pub(super) fn ln_u(&self, q: f64) -> f64 {
    if self.reflect { (-q).ln_1p() } else { q.ln() }
  }

  pub(super) fn ln_jacobian(&self) -> f64 {
    self.e * self.ln_far - self.e.ln()
  }
}

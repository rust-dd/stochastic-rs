//! The truncated normal's mgf: `e^{t·x_c − ½(c_s − c)(c_s + c)}` times the tilted and plain masses' ratio, `x_c` the
//! tilted reference point in the law's own coordinate and `c_s`, `c` the two reference points in standard units.
//!
//! Reference: Ogita, T., Rump, S. M. & Oishi, S. (2005), "Accurate Sum and Dot Product", *SIAM Journal on Scientific Computing* 26(6), 1955-1988, DOI 10.1137/030601818 (SumK over two-products).

use std::f64::consts::LN_2;

use super::SimdTruncatedNormal;
use super::kernels::TiltedKernel;
use crate::traits::SimdFloatExt;
use crate::truncated::local::Model;
use crate::truncated::quadrature::Cut;
use crate::truncated::quadrature::Terms;
use crate::truncated::quadrature::Truncation;
use crate::truncated::quadrature::accurate_sum;
use crate::truncated::quadrature::ldexp;
use crate::truncated::quadrature::span;
use crate::truncated::quadrature::split;
use crate::truncated::quadrature::two_prod;
use crate::truncated::quadrature::two_sum;

/// A tilt clamped at an end with a rate past this is `e^{−r·y − y²/2}` there, its curvature's share `1/r²` below 2⁻⁵⁴:
/// the exponential's mass, where the quadrature would meet a slope past the doubles.
const STEEP: f64 = 134_217_728.0;

/// The largest `|σt|`, `|α|`, `|β|`, `|t·lo|`, `|t·up|` the plain path takes.
const PLAIN: f64 = 8.0;

/// `σ = m·2^e` and `σ²t·2^{−e−4}` as four doubles summing to it exactly; no square where `|t·2^{e−2}|` passes MAX, `σt`
/// past `4m·MAX`.
#[derive(Clone, Copy)]
struct Scale {
  m: f64,
  e: i32,
  square: Option<[f64; 4]>,
}

impl Scale {
  fn new(sigma: f64, t: f64) -> Self {
    let (m, e) = split(sigma);
    let u = ldexp(t, e - 4);
    let square = (4.0 * u).is_finite().then(|| {
      let (a, a_lo) = two_prod(m, m);
      let ((b, b_lo), (c, c_lo)) = (two_prod(a, u), two_prod(a_lo, u));
      [b, b_lo, c, c_lo]
    });
    Self { m, e, square }
  }
}

/// A finite bound, standardised `z + z_lo` exactly and held beside `held.0` by `held.1`, `x` in the law's coordinate,
/// with the tilted mode's offset `r = 16(q + q_lo)` from it, `q` finite where `r` passes the doubles, and `ln|r|` there.
#[derive(Clone, Copy)]
struct Anchor {
  z: f64,
  z_lo: f64,
  held: (f64, f64),
  x: f64,
  r: f64,
  q: (f64, f64),
  ln_far: f64,
}

/// The plain reference point against an anchor bound: its exact offset where both are held beside one double, else its
/// own double and rest.
#[derive(Clone, Copy)]
enum Centre {
  Beside(f64),
  Apart(f64, f64),
}

/// The exponent's form: from an anchor bound with the plain reference point against it and the tilted reference
/// point's offset `d`, or at the tilted mode, against the anchor the plain reference point is read from.
#[derive(Clone, Copy)]
enum Exponent {
  Bound(Anchor, Centre, f64),
  Mode(Option<(Anchor, Centre)>),
}

impl<T: SimdFloatExt> SimdTruncatedNormal<T> {
  /// The tilted mode is placed by its exact offsets from the bounds, the tilt integrated from the bound it clamps at or
  /// lies nearer, and the exponent summed exactly, save where every term is at most 8; at an infinite `t` the limit.
  pub(super) fn mgf(&self, t: f64) -> Option<f64> {
    if t.is_nan() {
      return Some(f64::NAN);
    }
    if t == 0.0 {
      return Some(1.0);
    }
    let (lo, up) = (self.lower.to_f64()?, self.upper.to_f64()?);
    if t.is_infinite() {
      let end = if t > 0.0 { up } else { lo };
      return Some(if end * t > 0.0 { f64::INFINITY } else { 0.0 });
    }
    let plain = match self.analytics()? {
      Model::Full(plain) => plain,
      Model::Local(l) => return Some(l.mgf(t, (lo, up))),
    };
    let (mu, sigma) = (self.base.mean().to_f64()?, self.base.std_dev().to_f64()?);
    let (alpha, beta) = self.standardised_bounds();
    let (held_lo, held_up) = self.held();
    let width = span(held_lo, held_up);
    let (centre, mass) = (plain.reference(), plain.mass());
    // Part 2's risk 6: where `|σt|`, `|α|`, `|β|`, `|t·lo|`, `|t·up|` (the finite ones) are at most 8, rounded `σt`, bounds
    // without their rests and plain sums move `ln M` by under 2.4e-13 in all.
    let s = sigma * t;
    let small = |v: f64| !v.is_finite() || v.abs() <= PLAIN;
    let rounded = s.abs() <= PLAIN && [alpha, beta, t * lo, t * up].into_iter().all(small);
    let scale = (!rounded).then(|| Scale::new(sigma, t));
    let anchor = |z: f64, x: f64, cut: Cut| {
      let (s0, rest, _) = cut.parts();
      z.is_finite().then(|| match &scale {
        Some(scale) => Anchor::new(scale, sigma, t, x, mu, z, (s0, rest)),
        None => Anchor::rounded(s, x, z, (s0, rest)),
      })
    };
    let (at_lo, at_up) = (anchor(alpha, lo, held_lo), anchor(beta, up, held_up));
    // The plain reference point against an anchor: by its exact offset where both are held beside one double, else as
    // its own double with the rest of the one it is held beside, a bound's own double carrying that bound's.
    let c = |a: &Anchor| {
      let (s, rest, t) = centre.parts();
      if s == a.held.0 {
        return Centre::Beside(span(Cut::held(a.z, a.held.0, a.held.1), centre));
      }
      let own = match (&at_lo, &at_up) {
        (Some(b), _) if b.held == (s, 0.0) => b.z_lo,
        (_, Some(b)) if b.held == (s, 0.0) => b.z_lo,
        _ => 0.0,
      };
      let (c, e) = two_sum(s, rest + t);
      Centre::Apart(c, e + own)
    };
    let clamp = match (at_lo, at_up) {
      (_, Some(a)) if a.r > 0.0 => Some(a),
      (Some(a), _) if a.r < 0.0 => Some(a),
      _ => None,
    };
    if clamp.is_none() && scale.is_some_and(|s| s.square.is_none()) {
      // Past 4·MAX `(σt)²/2 > 8·MAX²` outweighs `|μt|` and every other term, each under `MAX²`, and no bound holds the
      // tilt, whose mass is then at least half the normal's: `+∞`.
      return Some(f64::INFINITY);
    }
    let (exponent, masses) = match clamp {
      Some(a) if a.r.abs() >= STEEP => {
        let rate = a.r.abs();
        // `rW` from `q = r/16`, finite where `r` overflows: past the doubles `e^{−rW}` can still count.
        let reach = -(-a.q.0.abs() * (16.0 * width)).exp_m1();
        let tilted = reach / rate;
        let masses = if tilted >= f64::MIN_POSITIVE {
          ln_quotient(tilted, mass)
        } else {
          reach.ln() - a.ln_r() - mass.ln()
        };
        (Exponent::Bound(a, c(&a), 0.0), masses)
      }
      _ => {
        let nearer = match (at_lo, at_up) {
          (Some(l), Some(u)) => Some(if l.r.abs() < u.r.abs() { l } else { u }),
          (l, u) => l.or(u),
        };
        // At the tilted mode `μt + s²/2 + c²/2`, from the plain reference point's bound where it is one.
        let at_mode = |a: Option<Anchor>| match (at_lo, at_up, a) {
          (Some(b), ..) if centre == held_lo => Exponent::Mode(Some((b, c(&b)))),
          (_, Some(b), _) if centre == held_up => Exponent::Mode(Some((b, c(&b)))),
          (.., Some(a)) if centre.x != 0.0 => Exponent::Mode(Some((a, c(&a)))),
          _ => Exponent::Mode(None),
        };
        match clamp.or(nearer) {
          Some(a) if a.r.is_finite() => {
            let (l, r) = if a.x == lo {
              (0.0, width)
            } else {
              (-width, 0.0)
            };
            let tilted = Truncation::mass_only(TiltedKernel { s: a.r, c: 0.0 }, l, r);
            let d = tilted.centre();
            let exponent = if d == a.r {
              at_mode(Some(a))
            } else {
              Exponent::Bound(a, c(&a), d)
            };
            (exponent, ln_quotient(tilted.mass(), mass))
          }
          // With both ends open the tilted law is the whole normal, as the plain one is.
          None => (Exponent::Mode(None), 0.0),
          // No bound within the doubles' reach of the mode: the tilted law is the whole normal about it.
          Some(a) => {
            let whole = Truncation::mass_only(
              TiltedKernel { s: 0.0, c: 0.0 },
              f64::NEG_INFINITY,
              f64::INFINITY,
            );
            (at_mode(Some(a)), ln_quotient(whole.mass(), mass))
          }
        }
      }
    };
    let ln_m = if rounded {
      let mut sum = Rounded(masses);
      emit(&mut sum, exponent, t, (mu, sigma), Some(s));
      sum.0
    } else {
      let mut terms = Terms::new();
      terms.scaled(masses, 0);
      emit(&mut terms, exponent, t, (mu, sigma), None);
      terms.total()
    };
    Some(ln_m.exp())
  }
}

impl Anchor {
  /// The offset from its numerator `σ²t − (x − μ)`, divided by `16σ` to a double-double; where that passes the doubles
  /// (past `4m·MAX`, or with a far bound beside it) `z/σ` is under a quarter of `t`: `t`'s sign and `ln|r|` from halves.
  fn new(
    &Scale { m, e, square }: &Scale,
    sigma: f64,
    t: f64,
    x: f64,
    mu: f64,
    z: f64,
    held: (f64, f64),
  ) -> Self {
    let ((d, d_lo), scale) = difference(x, mu);
    // The rest `(x − μ)/σ − z` of the rounded bound, by the remainder at `σ`'s scale.
    let half = scale - e - 1;
    let z_lo = ((-z).mul_add(0.5 * m, ldexp(d, half)) + ldexp(d_lo, half)) / (0.5 * m);
    let Some((n, n_lo)) = square
      .map(|sq| numerator(sq, (d, d_lo), scale - e - 4))
      .filter(|(n, _)| n.is_finite())
    else {
      let r = t.signum() * f64::INFINITY;
      let halves = 0.5 * t - 0.5 * z / sigma;
      return Self {
        z,
        z_lo,
        held,
        x,
        r,
        q: ((0.125 * sigma) * halves, 0.0),
        ln_far: sigma.ln() + halves.abs().ln() + LN_2,
      };
    };
    let q = n / m;
    let q_lo = ((-q).mul_add(m, n) + n_lo) / m;
    let r = 16.0 * q;
    Self {
      z,
      z_lo,
      held,
      x,
      r,
      q: (q, q_lo),
      ln_far: if r.is_infinite() {
        n.abs().ln() - m.ln() + 4.0 * LN_2
      } else {
        f64::NAN
      },
    }
  }

  /// The offset `fl(σt) − z` of the plain path, the bound's rest dropped.
  fn rounded(s: f64, x: f64, z: f64, held: (f64, f64)) -> Self {
    let r = s - z;
    Self {
      z,
      z_lo: 0.0,
      held,
      x,
      r,
      q: (r / 16.0, 0.0),
      ln_far: f64::NAN,
    }
  }

  fn ln_r(&self) -> f64 {
    if self.r.is_infinite() {
      self.ln_far
    } else {
      self.r.abs().ln()
    }
  }
}

/// Where an exponent's terms go: exact into the scaled frame, or rounded and summed in plain doubles.
trait Sink {
  /// `f·g·2^p`.
  fn product(&mut self, f: f64, g: f64, p: i32);
}

impl Sink for Terms {
  fn product(&mut self, f: f64, g: f64, p: i32) {
    Terms::product(self, f, g, p);
  }
}

/// The plain path's running sum, whose powers of two lie within ±8 and whose rests are 0.
struct Rounded(f64);

impl Sink for Rounded {
  #[inline]
  fn product(&mut self, f: f64, g: f64, p: i32) {
    if g != 0.0 {
      self.0 += f * g * f64::from_bits(((p + 1023) as u64) << 52);
    }
  }
}

/// The exponent's terms. At the tilted mode it is `μt + (σt)²/2 + c²/2`, exact from `μ`, `σ`, `t` and the reference
/// point, unless `|c|` passes `|r|`: then `t·x + ½(c² − z²) + r²/2`, whose `r` is a double-double.
fn emit(sink: &mut impl Sink, exponent: Exponent, t: f64, (mu, sigma): (f64, f64), s: Option<f64>) {
  match exponent {
    Exponent::Bound(a, c, d) => from_bound(sink, t, a, c, d),
    Exponent::Mode(Some((a, c))) if c.size(&a) > a.r.abs() => from_bound(sink, t, a, c, a.r),
    Exponent::Mode(found) => {
      sink.product(t, mu, 0);
      match s {
        Some(s) => sink.product(s, s, -1),
        None => {
          let ((m, e), (mt, et)) = (split(sigma), split(t.abs()));
          let (h, l) = two_prod(m, mt);
          let p = 2 * (e + et);
          sink.product(h, h, p - 1);
          sink.product(h, l, p);
          sink.product(l, l, p - 1);
        }
      }
      match found {
        Some((a, Centre::Beside(dl))) => {
          for (f, g, p) in [
            (a.z, a.z, -1),
            (a.z, a.z_lo, 0),
            (a.z, dl, 0),
            (a.z_lo, dl, 0),
            (a.z_lo, a.z_lo, -1),
            (dl, dl, -1),
          ] {
            sink.product(f, g, p);
          }
        }
        Some((_, Centre::Apart(c, c_lo))) => {
          sink.product(c, c, -1);
          sink.product(c, c_lo, 0);
          sink.product(c_lo, c_lo, -1);
        }
        None => {}
      }
    }
  }
}

/// `t·x + ½(c² − z²) + d(r − d/2)` from the anchor, `r = 16q`.
fn from_bound(sink: &mut impl Sink, t: f64, a: Anchor, c: Centre, d: f64) {
  sink.product(t, a.x, 0);
  // `½(c² − z²)` as `δz + δ²/2` beside the anchor, where `c` and `z` round apart but their offset does not.
  match c {
    Centre::Beside(dl) => {
      sink.product(dl, a.z, 0);
      sink.product(dl, a.z_lo, 0);
      sink.product(dl, dl, -1);
    }
    Centre::Apart(c, c_lo) => {
      sink.product(-a.z, a.z, -1);
      sink.product(-a.z, a.z_lo, 0);
      sink.product(c, c, -1);
      sink.product(c, c_lo, 0);
    }
  }
  let (q, q_lo) = a.q;
  if d == a.r {
    sink.product(q, q, 7);
    sink.product(q, q_lo, 8);
  } else if d != 0.0 {
    sink.product(d, q, 4);
    sink.product(d, q_lo, 4);
    sink.product(-d, d, -1);
  }
}

impl Centre {
  /// The reference point's size, against an anchor whose `r` a form at the tilted mode trades it for.
  fn size(&self, a: &Anchor) -> f64 {
    match *self {
      Centre::Beside(dl) => (a.z + dl).abs(),
      Centre::Apart(c, _) => c.abs(),
    }
  }
}

/// `ln(n/d)` from the quotient where it is normal, else from the two logs.
fn ln_quotient(n: f64, d: f64) -> f64 {
  let q = n / d;
  if q.is_normal() {
    q.ln()
  } else {
    n.ln() - d.ln()
  }
}

/// `x − μ` as `(d + d_lo)·2^scale`, from halves where it overflows.
fn difference(x: f64, mu: f64) -> ((f64, f64), i32) {
  match two_sum(x, -mu) {
    (d, d_lo) if d.is_finite() => ((d, d_lo), 0),
    _ => (two_sum(0.5 * x, -0.5 * mu), 1),
  }
}

/// `σ²t·2^{−e−4} − (d + d_lo)·2^scale` to a double-double, `(σ²t − (x − μ))·2^{−e−4}`: the square's exact parts and
/// the difference's summed in thrice the precision, so no rounded `σt` meets `x − μ`.
fn numerator([b, b_lo, c, c_lo]: [f64; 4], (d, d_lo): (f64, f64), scale: i32) -> (f64, f64) {
  accurate_sum([b, b_lo, c, c_lo, -ldexp(d, scale), -ldexp(d_lo, scale)])
}

//! The truncated normal's mgf: `e^{t·x_c − ½(c_s − c)(c_s + c)}` times the tilted and plain masses' ratio, `x_c` the
//! tilted reference point in the law's own coordinate and `c_s`, `c` the two reference points in standard units.
//!
//! Reference: Ogita, T., Rump, S. M. & Oishi, S. (2005), "Accurate Sum and Dot Product", *SIAM Journal on Scientific Computing* 26(6), 1955-1988, DOI 10.1137/030601818 (SumK over two-products).

use std::f64::consts::LN_2;

use super::SimdTruncatedNormal;
use super::kernels::TiltedKernel;
use crate::traits::SimdFloatExt;
use crate::truncated::local::Model;
use crate::truncated::local::limit_or;
use crate::truncated::quadrature::Cut;
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

/// `σ = m·2^e` and `σ²t·2^{−e−4}` as four doubles summing to it exactly; no square where `σt` lies past 4·MAX.
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

/// A finite bound, standardised `z + z_lo` exactly, held as `cut` and `x` in the law's coordinate, with the tilted
/// mode's offset `r + r_lo` from it, and `ln|r|` where `r` passes the doubles.
#[derive(Clone, Copy)]
struct Anchor {
  z: f64,
  z_lo: f64,
  cut: Cut,
  x: f64,
  r: f64,
  r_lo: f64,
  ln_far: f64,
}

/// The plain reference point against an anchor bound: its exact offset where both are held beside one double, else its
/// own double and rest.
#[derive(Clone, Copy)]
enum Centre {
  Beside(f64),
  Apart(f64, f64),
}

impl<T: SimdFloatExt> SimdTruncatedNormal<T> {
  /// The tilted mode is placed by its exact offsets from the bounds and the tilt integrated from the bound it clamps at
  /// or lies nearer, so `σt` never meets a bound in a rounded difference; at an infinite `t` the limit.
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
    let scale = Scale::new(sigma, t);
    let anchor = |z: f64, x: f64, cut: Cut| {
      z.is_finite()
        .then(|| Anchor::new(&scale, sigma, t, x, mu, z, cut))
    };
    let (at_lo, at_up) = (anchor(alpha, lo, held_lo), anchor(beta, up, held_up));
    // The plain reference point against an anchor: by its exact offset where both are held beside one double, else as
    // its own double with the rest of the one it is held beside, a bound's own double carrying that bound's.
    let c = |a: &Anchor| {
      let (s, rest, t) = centre.parts();
      if s == a.cut.parts().0 {
        return Centre::Beside(span(a.cut, centre));
      }
      let own = [at_lo, at_up]
        .into_iter()
        .flatten()
        .find(|b| b.cut == Cut::at(s));
      let (c, e) = two_sum(s, rest + t);
      Centre::Apart(c, e + own.map_or(0.0, |b| b.z_lo))
    };
    let clamp = match (at_lo, at_up) {
      (_, Some(a)) if a.r > 0.0 => Some(a),
      (Some(a), _) if a.r < 0.0 => Some(a),
      _ => None,
    };
    if clamp.is_none() && (sigma * t).is_infinite() {
      // Toward an open end the exponent is `σt(μ/σ + σt/2)`, past any double unless the bracket vanishes.
      let lead = mu / sigma + 0.5 * sigma * t;
      return Some(if lead * t > 0.0 { f64::INFINITY } else { 0.0 });
    }
    let (exponent, ln_ratio) = match clamp {
      Some(a) if a.r.abs() >= STEEP => {
        let rate = a.r.abs();
        let reach = -(-rate * width).exp_m1();
        let tilted = reach / rate;
        let ln_ratio = if tilted >= f64::MIN_POSITIVE {
          (tilted / mass).ln()
        } else {
          reach.ln() - a.ln_r() - mass.ln()
        };
        (from_bound(t, a, c(&a), 0.0), ln_ratio)
      }
      _ => {
        let nearer = match (at_lo, at_up) {
          (Some(l), Some(u)) => Some(if l.r.abs() < u.r.abs() { l } else { u }),
          (l, u) => l.or(u),
        };
        // The exponent at the tilted mode, `μt + s²/2 + c²/2`, from the plain reference point's bound where it is one.
        let at_mode = |a: Option<Anchor>| match (at_lo, at_up, a) {
          (Some(b), ..) if centre == held_lo => from_bound(t, b, c(&b), b.r),
          (_, Some(b), _) if centre == held_up => from_bound(t, b, c(&b), b.r),
          (.., Some(a)) if centre.x != 0.0 => from_bound(t, a, c(&a), a.r),
          _ => quadratic(mu, sigma, t, &scale),
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
              from_bound(t, a, c(&a), d)
            };
            (exponent, (tilted.mass() / mass).ln())
          }
          // With both ends open the tilted law is the whole normal, as the plain one is.
          None => return Some(quadratic(mu, sigma, t, &scale).exp()),
          // No bound within the doubles' reach of the mode: the tilted law is the whole normal about it.
          Some(_) => {
            let whole = Truncation::mass_only(
              TiltedKernel { s: 0.0, c: 0.0 },
              f64::NEG_INFINITY,
              f64::INFINITY,
            );
            (at_mode(None), (whole.mass() / mass).ln())
          }
        }
      }
    };
    Some((exponent + ln_ratio).exp())
  }
}

impl Anchor {
  /// The offset from its numerator `σ²t − (x − μ)`, divided by `σ` to a double-double; past 4·MAX, where `z/σ` is under
  /// a quarter of `t`, the sign of `t` and `ln σ + ln|t − z/σ|`.
  fn new(
    &Scale { m, e, square }: &Scale,
    sigma: f64,
    t: f64,
    x: f64,
    mu: f64,
    z: f64,
    cut: Cut,
  ) -> Self {
    let ((d, d_lo), scale) = difference(x, mu);
    // The rest `(x − μ)/σ − z` of the rounded bound, by the remainder at `σ`'s scale.
    let half = scale - e - 1;
    let z_lo = ((-z).mul_add(0.5 * m, ldexp(d, half)) + ldexp(d_lo, half)) / (0.5 * m);
    let Some((n, n_lo)) = square.map(|sq| numerator(sq, (d, d_lo), scale - e - 4)) else {
      return Self {
        z,
        z_lo,
        cut,
        x,
        r: t.signum() * f64::INFINITY,
        r_lo: 0.0,
        ln_far: sigma.ln() + (t - z / sigma).abs().ln(),
      };
    };
    let q = n / m;
    let q_lo = ((-q).mul_add(m, n) + n_lo) / m;
    let r = 16.0 * q;
    Self {
      z,
      z_lo,
      cut,
      x,
      r,
      r_lo: 16.0 * q_lo,
      ln_far: if r.is_infinite() {
        n.abs().ln() - m.ln() + 4.0 * LN_2
      } else {
        f64::NAN
      },
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

/// `t·x − ½(z² − c²) + d(r − d/2)`, the exponent with the tilted reference point `d` from the bound `z`: two-products of
/// the double-double `z`, `c`, `r`, summed in thrice the precision; past the doubles its largest term's limit.
fn from_bound(t: f64, a: Anchor, c: Centre, d: f64) -> f64 {
  let (tx, tx_lo) = two_prod(t, a.x);
  // `½(c² − z²)` as `δz + δ²/2` beside the anchor, where `c` and `z` round apart but their offset does not.
  let [(p, p_lo), (q, q_lo), (rests, _)] = match c {
    Centre::Beside(dl) => [
      two_prod(dl, a.z),
      two_prod(0.5 * dl, dl),
      (dl * a.z_lo, 0.0),
    ],
    Centre::Apart(c, c_lo) => {
      let ((zz, zz_lo), (cc, cc_lo)) = (two_prod(0.5 * a.z, a.z), two_prod(0.5 * c, c));
      [(-zz, -zz_lo), (cc, cc_lo), (c * c_lo - a.z * a.z_lo, 0.0)]
    }
  };
  let (bend, bend_lo) = if d == 0.0 {
    (0.0, 0.0)
  } else if d == a.r {
    let (rr, rr_lo) = two_prod(0.5 * a.r, a.r);
    (rr, rr_lo + a.r * a.r_lo)
  } else {
    (d * (a.r - 0.5 * d), 0.0)
  };
  let (sum, _) = accurate_sum([tx, tx_lo, p, p_lo, q, q_lo, rests, bend, bend_lo]);
  limit_or(sum, || {
    let middle = match c {
      Centre::Beside(dl) => term(1.0, dl, a.z + 0.5 * dl),
      Centre::Apart(c, _) => term(-2.0, 0.5 * a.z - 0.5 * c, 0.5 * a.z + 0.5 * c),
    };
    let ln_bend = if d == a.r {
      2.0 * a.ln_r() - LN_2
    } else {
      bend.abs().ln()
    };
    [term(1.0, t, a.x), middle, (bend.signum(), ln_bend)]
  })
}

/// `μt + (σt)²/2` as `t(μ + σ²t/2)`, the bracket as `numerator` forms it; `σt` rounds only where no cancellation is
/// left: past 4·MAX (a limit) or where `μ/σ` dwarfs it.
fn quadratic(mu: f64, sigma: f64, t: f64, &Scale { e, square, .. }: &Scale) -> f64 {
  let (d, scale) = difference(-mu, mu);
  if let Some((n, _)) = square.map(|sq| numerator(sq, d, scale - e - 4))
    && n.is_finite()
  {
    return scaled_product(t, n, e + 3);
  }
  let s = sigma * t;
  limit_or(mu * t + 0.5 * (s * s), || {
    [term(1.0, t, mu), term(0.5, s, s)]
  })
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

/// `a·b·2^p`, with no product past the doubles before the last scaling.
fn scaled_product(a: f64, b: f64, p: i32) -> f64 {
  if a == 0.0 || b == 0.0 {
    return 0.0;
  }
  let ((ma, ea), (mb, eb)) = (split(a.abs()), split(b.abs()));
  a.signum() * b.signum() * ldexp(ma * mb, ea + eb + p)
}

/// `k·f·g` as the `(sign, ln|·|)` that `limit_or` compares.
fn term(k: f64, f: f64, g: f64) -> (f64, f64) {
  (
    k * f.signum() * g.signum(),
    k.abs().ln() + f.abs().ln() + g.abs().ln(),
  )
}

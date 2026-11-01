//! The truncated normal's mgf: `e^{t·x_c − ½(c_s − c)(c_s + c)}` times the tilted and plain masses' ratio, `x_c` the
//! tilted reference point in the law's own coordinate and `c_s`, `c` the two reference points in standard units.
//!
//! Reference: Shewchuk, J. R. (1997), "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates", *Discrete & Computational Geometry* 18(3), 305-363, DOI 10.1007/PL00009321 (exact sums of two-products).

use std::f64::consts::LN_2;

use super::SimdTruncatedNormal;
use super::kernels::TiltedKernel;
use crate::traits::SimdFloatExt;
use crate::truncated::local::Model;
use crate::truncated::local::limit_or;
use crate::truncated::quadrature::Truncation;
use crate::truncated::quadrature::ldexp;
use crate::truncated::quadrature::split;

/// A tilt clamped at an end with a rate past this is `e^{−r·y − y²/2}` there, its curvature's share `1/r²` below 2⁻⁵⁴:
/// the exponential's mass, where the quadrature would meet a slope past the doubles.
const STEEP: f64 = 134_217_728.0;

/// A finite bound, standardised `z + z_lo` exactly and `x` in the law's coordinate, with the tilted mode's offset
/// `r + r_lo` from it and `ln|r|`.
#[derive(Clone, Copy)]
struct Anchor {
  z: f64,
  z_lo: f64,
  x: f64,
  r: f64,
  r_lo: f64,
  ln_r: f64,
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
    let (alpha, beta) = self.standardised_bounds()?;
    let (c, mass) = (plain.centre(), plain.mass());
    let anchor = |z: f64, x: f64| z.is_finite().then(|| Anchor::new(sigma, t, x, mu, z));
    let (at_lo, at_up) = (anchor(alpha, lo), anchor(beta, up));
    // The plain reference point exactly: a bound with its rest, else the mode or a midpoint the engine chose.
    let c = match (at_lo, at_up) {
      (Some(b), _) if c == alpha => (c, b.z_lo),
      (_, Some(b)) if c == beta => (c, b.z_lo),
      _ => (c, 0.0),
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
        let reach = -(-rate * (beta - alpha)).exp_m1();
        let tilted = reach / rate;
        let ln_ratio = if tilted >= f64::MIN_POSITIVE {
          (tilted / mass).ln()
        } else {
          reach.ln() - a.ln_r - mass.ln()
        };
        (from_bound(t, a, c, 0.0), ln_ratio)
      }
      _ => {
        let nearer = match (at_lo, at_up) {
          (Some(l), Some(u)) => Some(if l.r.abs() < u.r.abs() { l } else { u }),
          (l, u) => l.or(u),
        };
        // The exponent at the tilted mode, `μt + s²/2 + c²/2`, from the plain reference point's bound where it is one.
        let at_mode = |a: Option<Anchor>| match (at_lo, at_up, a) {
          (Some(b), ..) if c.0 == alpha => from_bound(t, b, c, b.r),
          (_, Some(b), _) if c.0 == beta => from_bound(t, b, c, b.r),
          (.., Some(a)) if c.0 != 0.0 => from_bound(t, a, c, a.r),
          _ => quadratic(mu, sigma, t),
        };
        match clamp.or(nearer) {
          Some(a) if a.r.is_finite() => {
            let tilted =
              Truncation::mass_only(TiltedKernel { s: a.r, c: 0.0 }, alpha - a.z, beta - a.z);
            let d = tilted.centre();
            let exponent = if d == a.r {
              at_mode(Some(a))
            } else {
              from_bound(t, a, c, d)
            };
            (exponent, (tilted.mass() / mass).ln())
          }
          // With both ends open the tilted law is the whole normal, as the plain one is.
          None => return Some(quadratic(mu, sigma, t).exp()),
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
  /// The offset from its numerator `σ²t − (x − μ)`, exact in expansion, divided by `σ` to a double-double; past 4·MAX,
  /// where `z/σ` is under a quarter of `t`, the sign of `t` and `ln σ + ln|t − z/σ|`.
  fn new(sigma: f64, t: f64, x: f64, mu: f64, z: f64) -> Self {
    let (m, e) = split(sigma);
    let ((d, d_lo), scale) = difference(x, mu);
    // The rest `(x − μ)/σ − z` of the rounded bound, at `σ`'s scale.
    let (p, p_lo) = two_prod(z, 0.5 * m);
    let (rest, _) = exact_sum([
      ldexp(d, scale - e - 1),
      ldexp(d_lo, scale - e - 1),
      -p,
      -p_lo,
    ]);
    let z_lo = rest / (0.5 * m);
    let Some((n, n_lo)) = numerator(m, ldexp(t, e - 4), (d, d_lo), scale - e - 4) else {
      let (r, ln_r) = (
        t.signum() * f64::INFINITY,
        sigma.ln() + (t - z / sigma).abs().ln(),
      );
      return Self {
        z,
        z_lo,
        x,
        r,
        r_lo: 0.0,
        ln_r,
      };
    };
    let q = n / m;
    let q_lo = ((-q).mul_add(m, n) + n_lo) / m;
    let r = 16.0 * q;
    let ln_r = if r.is_infinite() {
      n.abs().ln() - m.ln() + 4.0 * LN_2
    } else {
      r.abs().ln()
    };
    Self {
      z,
      z_lo,
      x,
      r,
      r_lo: 16.0 * q_lo,
      ln_r,
    }
  }
}

/// `t·x − ½(z² − c²) + d(r − d/2)`, the exponent whose tilted reference point lies `d` from the bound `z`: two-products
/// of the double-double `z`, `c` and `r`, summed exactly; past the doubles the limit its largest term gives.
fn from_bound(t: f64, a: Anchor, (c, c_lo): (f64, f64), d: f64) -> f64 {
  let (tx, tx_lo) = two_prod(t, a.x);
  let ((zz, zz_lo), (cc, cc_lo)) = if c == a.z {
    ((0.0, 0.0), (0.0, 0.0))
  } else {
    (two_prod(0.5 * a.z, a.z), two_prod(0.5 * c, c))
  };
  let rests = if c == a.z {
    0.0
  } else {
    c * c_lo - a.z * a.z_lo
  };
  let ((bend, bend_lo), ln_bend) = if d == 0.0 {
    ((0.0, 0.0), f64::NEG_INFINITY)
  } else if d == a.r {
    let (rr, rr_lo) = two_prod(0.5 * a.r, a.r);
    ((rr, rr_lo + a.r * a.r_lo), 2.0 * a.ln_r - LN_2)
  } else {
    let bend = d * (a.r - 0.5 * d);
    ((bend, 0.0), bend.abs().ln())
  };
  let (h1, h2) = (0.5 * a.z - 0.5 * c, 0.5 * a.z + 0.5 * c);
  let (sum, _) = exact_sum([tx, tx_lo, -zz, -zz_lo, cc, cc_lo, rests, bend, bend_lo]);
  limit_or(
    sum,
    &[
      term(1.0, t, a.x),
      term(-2.0, h1, h2),
      (bend.signum(), ln_bend),
    ],
  )
}

/// `μt + (σt)²/2` as `t(μ + σ²t/2)`, the bracket exact in expansion; `σt` is rounded only where no cancellation is
/// left: past 4·MAX (a limit) or where `μ/σ` dwarfs it.
fn quadratic(mu: f64, sigma: f64, t: f64) -> f64 {
  let (m, e) = split(sigma);
  let (d, scale) = difference(-mu, mu);
  if let Some((n, _)) = numerator(m, ldexp(t, e - 4), d, scale - e - 4)
    && n.is_finite()
  {
    return scaled_product(t, n, e + 3);
  }
  let s = sigma * t;
  limit_or(mu * t + 0.5 * (s * s), &[term(1.0, t, mu), term(0.5, s, s)])
}

/// `x − μ` as `(d + d_lo)·2^scale`, from halves where it overflows.
fn difference(x: f64, mu: f64) -> ((f64, f64), i32) {
  match two_sum(x, -mu) {
    (d, d_lo) if d.is_finite() => ((d, d_lo), 0),
    _ => (two_sum(0.5 * x, -0.5 * mu), 1),
  }
}

/// `m²u − (d + d_lo)·2^scale` to a double-double, `σ = m·2^e` and `u = t·2^{e−4}`: `(σ²t − (x − μ))·2^{−e−4}`, by
/// two-products and an exact sum, so no rounded `σt` meets `x − μ`; `None` where `σt` lies past 4·MAX.
fn numerator(m: f64, u: f64, (d, d_lo): (f64, f64), scale: i32) -> Option<(f64, f64)> {
  if !(4.0 * u).is_finite() {
    return None;
  }
  let (a, a_lo) = two_prod(m, m);
  let (b, b_lo) = two_prod(a, u);
  let (c, c_lo) = two_prod(a_lo, u);
  Some(exact_sum([
    b,
    b_lo,
    c,
    c_lo,
    -ldexp(d, scale),
    -ldexp(d_lo, scale),
  ]))
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

/// `s + e = a + b` exactly.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
  let s = a + b;
  let z = s - a;
  (s, (a - (s - z)) + (b - z))
}

/// `p + e = a·b` exactly while `e` stays a normal double.
fn two_prod(a: f64, b: f64) -> (f64, f64) {
  let p = a * b;
  (p, a.mul_add(b, -p))
}

/// The terms' sum to a double-double: Shewchuk's exact growth of a nonoverlapping expansion, its sum, then the sum of
/// what that left.
fn exact_sum<const N: usize>(terms: [f64; N]) -> (f64, f64) {
  const { assert!(N < 16) };
  let mut parts = [0.0; 16];
  let mut len = 0;
  for x in terms {
    len = grow(&mut parts, len, x);
  }
  let hi = parts[..len].iter().sum::<f64>();
  len = grow(&mut parts, len, -hi);
  (hi, parts[..len].iter().sum())
}

/// Shewchuk's growth by `x` with zero elimination: the parts stay nonoverlapping, least first.
fn grow(parts: &mut [f64; 16], len: usize, x: f64) -> usize {
  let (mut q, mut kept) = (x, 0);
  for i in 0..len {
    let (s, e) = two_sum(q, parts[i]);
    if e != 0.0 {
      parts[kept] = e;
      kept += 1;
    }
    q = s;
  }
  parts[kept] = q;
  kept + 1
}

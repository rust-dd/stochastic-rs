//! The bivariate t cdf to a relative `1e-12`: `P(Y ≤ y | X ≤ x)` from the law of `Y` given `X = √ν sinh t`, integrated
//! in `t` by panels marched out of each feature in its own offsets, the plateaus between features in closed form.
//! Reference: Genz, A. (2004), "Numerical computation of rectangular bivariate and trivariate normal and t probabilities", *Statistics and Computing* 14(3), 251-260, DOI 10.1023/B:STCO.0000035304.20635.31 (the conditional representation).

use std::cell::Cell;
use std::f64::consts::FRAC_2_PI;
use std::f64::consts::LN_2;
use std::f64::consts::PI;

use stochastic_rs_distributions::special::ln_gamma;

use crate::bivariate::conditional::in_unit;
use crate::bivariate::conditional::ln_add_exp;
use crate::bivariate::conditional::sum_minus_one;
use crate::bivariate::gaussian::gaussian_copula_cdf;
use crate::bivariate::integral::March;
use crate::bivariate::integral::NEGLIGIBLE_FALL;
use crate::bivariate::integral::march;
use crate::student_t::Conditional;
use crate::student_t::Feature;
use crate::student_t::Law;
use crate::student_t::NORMAL_ABOVE;
use crate::student_t::ln_cosh;

/// Below this `ν` the copula is its `ν → 0` limit `pM + (1 − p)W` to rounding: the remainder is under `50ν/min(p, 1 − p)`,
/// and `min(p, 1 − p) ≥ 4.7e-9` for every admitted `ρ`.
const LIMIT_NU: f64 = 1e-30;

/// A point `ln H` must settle within before a plateau takes over: a relative `2⁻⁵⁶` on `H`.
const SETTLED: f64 = 1.0 / (1u64 << 56) as f64;

/// What a march into a gap waits for: `H` on its plateau `ln H`, or, below `y`'s tail, the power law alone at the far end
/// and the integrand `NEGLIGIBLE_FALL` under its own sum at the end nearer `|t_y|`.
#[derive(Debug, Clone, Copy)]
enum Facing {
  Plateau(f64),
  BelowTail { near: bool },
}

/// What a march outward from an anchor leaves: up to offset `end` (`±∞` for a tail), where `ln H` is `ln_h_end`, with
/// `H` monotone on either side of `turn`.
#[derive(Debug, Clone, Copy)]
struct Rest {
  end: f64,
  ln_h_end: f64,
  turn: f64,
}

/// A feature of the integrand on `(−∞, t_x]`: `|t|`, its sign, `ν|t|` and `|t| − |t_y|`, the last two exact.
#[derive(Debug, Clone, Copy)]
struct Anchor {
  sign: f64,
  abs_t: f64,
  nu_abs_t: f64,
  d_y: f64,
  /// `|t| < |t_y|` toward the next anchor, the side `y`'s term holds `H` at 1 or below its tail.
  inside_next: bool,
}

/// The bivariate t copula's cdf at one `(ρ, ν)`, its laws and constants hoisted out of the query loop.
#[derive(Debug, Clone)]
pub(crate) struct BivariateT {
  rho: f64,
  nu: f64,
  law: Law,
  wide: Law,
  /// `ln T_{ν+1}(±c₀ρ)`: `H` as `t → −∞` and `t → ∞`.
  ln_h_left: f64,
  ln_h_right: f64,
  /// How far past a feature `H` and the density settle into their plateaus.
  reach: f64,
  /// Where `ν ln_1p(e^{−2|t|})` drops below `2⁻⁵⁶`, so the density is `2^ν e^{−ν|t|}` to rounding.
  power_from: f64,
  /// A march's first panel, under the width of the density's core, `1/√ν`.
  first_step: f64,
  /// `1/(c₀|ρ|)`, the width of `H`'s step at the centre crossing, which a wider first panel's nodes could straddle unseen.
  crossing_width: f64,
  ln_abs_rho: f64,
  /// The `ν → 0` limit's weights `p, 1 − p = (2/π) asin(√((1 ± ρ)/2))`.
  p: f64,
  q: f64,
  /// `ln B(ν/2, 1/2)`, the normaliser of the density `sech^ν t` of `t`, to within `1/(4ν)` past `ν = 40`.
  ln_beta: f64,
}

impl BivariateT {
  pub(crate) fn new(rho: f64, nu: f64) -> Self {
    let c0 = ((nu + 1.0) / ((1.0 - rho) * (1.0 + rho))).sqrt();
    let wide = Law::new(nu + 1.0);
    let (sign, ln_abs) = (rho.signum(), (c0 * rho.abs()).ln());
    Self {
      rho,
      nu,
      law: Law::new(nu),
      ln_h_left: wide.ln_cdf(if rho == 0.0 { 0.0 } else { sign }, ln_abs),
      ln_h_right: wide.ln_cdf(if rho == 0.0 { 0.0 } else { -sign }, ln_abs),
      wide,
      reach: 50.0 + (c0 * (nu + 2.0).sqrt()).ln_1p(),
      power_from: 20.0 + 0.5 * nu.ln().max(0.0),
      first_step: 1.0_f64.min(nu.sqrt().recip()),
      crossing_width: (c0 * rho.abs()).recip(),
      ln_abs_rho: rho.abs().ln(),
      p: FRAC_2_PI * (0.5 * (1.0 + rho)).sqrt().asin(),
      q: FRAC_2_PI * (0.5 * (1.0 - rho)).sqrt().asin(),
      ln_beta: if nu > 40.0 {
        0.5 * (PI.ln() - (0.5 * nu).ln())
      } else {
        ln_gamma(0.5 * nu) + 0.5 * PI.ln() - ln_gamma(0.5 * (nu + 1.0))
      },
    }
  }

  /// `C(u, v)` on the closed square, exact on its edges and inside the Fréchet bounds.
  pub(crate) fn copula_cdf(&self, u: f64, v: f64) -> f64 {
    if !(in_unit(u) && in_unit(v)) {
      return f64::NAN;
    }
    if u == 0.0 || v == 0.0 {
      return 0.0;
    }
    if u == 1.0 {
      return v;
    }
    if v == 1.0 {
      return u;
    }
    let (u, v) = if u <= v { (u, v) } else { (v, u) };
    let floor = sum_minus_one(u, v).max(0.0);
    if self.nu >= NORMAL_ABOVE {
      return gaussian_copula_cdf(u, v, self.rho);
    }
    if self.nu < LIMIT_NU {
      // `u − floor` is `1 − v` past `u + v = 1`, exact; the smaller weight from `asin` of the smaller argument.
      let band = if floor > 0.0 { 1.0 - v } else { u };
      return if self.rho >= 0.0 {
        u - self.q * band
      } else {
        floor + self.p * band
      };
    }
    let ratio = self.ln_conditional(u, v).exp();
    (u * ratio).clamp(floor, u)
  }

  /// `ln P(Y ≤ y | X ≤ x)` at `u ≤ v`: `∫ w H / ∫ w` over `t ≤ t_x`, `w = sech^ν t` the density of `t`, so its constant
  /// cancels, and `H = T_{ν+1}(A)` the law of `Y` given `t`.
  fn ln_conditional(&self, u: f64, v: f64) -> f64 {
    let (fx, fy) = (
      self.law.feature(self.law.quantile(u)),
      self.law.feature(self.law.quantile(v)),
    );
    if !(self.reachable(u, &fx) && self.reachable(v, &fy)) {
      return f64::NAN;
    }
    let cond = Conditional::new(self.rho, self.nu, &fy);
    let gap_x = self.law.abs_t_gap(&fy, &fx);
    let anchors = self.anchors(&fx, &fy, gap_x);
    // Where `A`, and with it `H`, turns: `sinh t = −ρ/sinh t_y`; `H` is monotone on either side of it.
    let turn = if fy.sign == 0.0 {
      f64::NAN
    } else {
      (-self.rho / (fy.sign * fy.abs_t.sinh())).asinh()
    };
    let mut sums = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    let add = |sums: &mut (f64, f64), d: f64, n: f64| {
      *sums = (ln_add_exp(sums.0, d), ln_add_exp(sums.1, n));
    };
    // Every region but the left tail, each anchor's own marches out to the midpoints or into the gaps; a march outward
    // ends once what is left of it is negligible against everything summed so far.
    let mut toward_gap: Vec<(Option<March>, Option<March>)> = Vec::with_capacity(anchors.len());
    for (i, a) in anchors.iter().enumerate() {
      let step = self.step(a, &fy);
      let mut sides = (None, None);
      for (side, other) in [(-1.0, i.checked_sub(1)), (1.0, Some(i + 1))] {
        let Some(b) = other.and_then(|j| anchors.get(j)) else {
          continue;
        };
        let dist = (b.d_y - a.d_y).abs();
        let m = if self.far(dist) {
          let (left, right) = if side < 0.0 { (b, a) } else { (a, b) };
          let near = self.near_is_left(left, right) == (side > 0.0);
          let facing = self.facing(left.inside_next, fy.sign, near);
          self.gap_march(&cond, a, step, side * self.reach, facing, sums)
        } else {
          let end = side * 0.5 * dist;
          let rest = Rest {
            end,
            ln_h_end: self.ln_h(&cond, a, end),
            turn,
          };
          let outer = sums;
          march(
            |delta: f64| self.eval(&cond, a, delta),
            end,
            step,
            &[],
            outer,
            |x, value, own| self.rest_negligible(a, x, value, &rest, totals(outer, own)),
          )
        };
        add(&mut sums, m.ln_g, m.ln_f);
        if side < 0.0 {
          sides.0 = Some(m);
        } else {
          sides.1 = Some(m);
        }
      }
      toward_gap.push(sides);
    }
    // The gaps between far anchors: the density's plateau in closed form, `H` its plateau or, below `y`'s tail, the
    // integrand at the end nearer `|t_y|`, where the march stopped once it fell `NEGLIGIBLE_FALL` below its own sum.
    for i in 0..anchors.len().saturating_sub(1) {
      let (a, b) = (&anchors[i], &anchors[i + 1]);
      if !self.far((b.d_y - a.d_y).abs()) {
        continue;
      }
      let (left, right) = (
        toward_gap[i].1.as_ref().expect("a right march"),
        toward_gap[i + 1].0.as_ref().expect("a left march"),
      );
      let (d_left, d_right) = (self.d_at(a, left.end), self.d_at(b, right.end));
      let (lo_anchor, lo_offset) = if d_left <= d_right {
        (a, left.end)
      } else {
        (b, right.end)
      };
      let ln_band = self.ln_power_band(
        self.nu_abs_t(lo_anchor, lo_offset),
        (d_right - d_left).abs(),
      );
      let ln_h = if !a.inside_next {
        ln_band + self.ln_h_right
      } else if fy.sign > 0.0 {
        ln_band
      } else if self.near_is_left(a, b) {
        left.last.1
      } else {
        right.last.1
      };
      add(&mut sums, ln_band, ln_h);
    }
    // The left tail last, against the rest: it ends on the plateaus, or once what is left of it to `−∞` is negligible.
    let first = &anchors[0];
    let outer = sums;
    let rest = Rest {
      end: f64::NEG_INFINITY,
      ln_h_end: self.ln_h_left,
      turn,
    };
    let negligible = Cell::new(false);
    let tail = march(
      |delta: f64| self.eval(&cond, first, delta),
      -(self.reach + self.power_from),
      self.step(first, &fy),
      &[],
      outer,
      |x, value, own| {
        if self.settled(&cond, first, x, self.ln_h_left) {
          return true;
        }
        let done = self.rest_negligible(first, x, value, &rest, totals(outer, own));
        negligible.set(done);
        done
      },
    );
    add(&mut sums, tail.ln_g, tail.ln_f);
    if !negligible.get() {
      let ln_rest = self.ln_power_mass(self.nu_abs_t(first, tail.end));
      add(&mut sums, ln_rest, ln_rest + self.ln_h_left);
    }
    sums.1 - sums.0
  }

  /// Whether the density can carry the tail beyond the quantile `f` of level `p`: the concave `ψ` bounds that tail's mass by
  /// `e^ψ/(ν tanh|t|)`, so a `ψ` 50 under `ln min(p, 1 − p) B` is not the law's quantile, as `SimdStudentT`'s past `ν ≈ 1e13`.
  fn reachable(&self, p: f64, f: &Feature) -> bool {
    if f.sign == 0.0 {
      return true;
    }
    let psi = if f.abs_t >= 1.0 {
      -f.nu_abs_t + self.nu * (LN_2 - (-2.0 * f.abs_t).exp().ln_1p())
    } else {
      -self.nu * ln_cosh(f.abs_t)
    };
    let tail = if f.sign < 0.0 { p } else { 1.0 - p };
    psi >= tail.ln() + self.ln_beta + (self.nu * f.abs_t.tanh()).ln().min(0.0) - 50.0
  }

  /// A march's first panel: `first_step`, under `1/(c₀|ρ|)` at an anchor within a few such widths of the centre crossing,
  /// `y`'s anchors and an `x` beside them, where `H` steps.
  fn step(&self, a: &Anchor, fy: &Feature) -> f64 {
    let near = fy.sign != 0.0 && a.d_y.abs() <= 8.0 * self.crossing_width + self.ln_abs_rho.abs();
    if near {
      self.first_step.min(0.25 * self.crossing_width)
    } else {
      self.first_step
    }
  }

  /// The features on `(−∞, t_x]` in order: `−|t_y|`, `0`, `|t_y|` where inside, and `t_x`, which ends the domain.
  fn anchors(&self, fx: &Feature, fy: &Feature, gap_x: f64) -> Vec<Anchor> {
    let mut out = Vec::with_capacity(4);
    let at_y = |sign: f64| Anchor {
      sign,
      abs_t: fy.abs_t,
      nu_abs_t: fy.nu_abs_t,
      d_y: 0.0,
      inside_next: sign < 0.0,
    };
    let positive_x = fx.sign > 0.0;
    if fy.sign != 0.0 && (fx.sign >= 0.0 || gap_x < 0.0) {
      out.push(at_y(-1.0));
    }
    if positive_x {
      out.push(Anchor {
        sign: 0.0,
        abs_t: 0.0,
        nu_abs_t: 0.0,
        d_y: -fy.abs_t,
        inside_next: fy.sign != 0.0,
      });
      if fy.sign != 0.0 && gap_x > 0.0 {
        out.push(at_y(1.0));
      }
    }
    out.push(Anchor {
      sign: fx.sign,
      abs_t: fx.abs_t,
      nu_abs_t: fx.nu_abs_t,
      d_y: gap_x,
      inside_next: false,
    });
    out
  }

  /// `(sign t, |t|, ν|t|, |t| − |t_y|)` at offset `delta` from anchor `a`, the last two exact.
  fn point(&self, a: &Anchor, delta: f64) -> (f64, f64, f64, f64) {
    if a.sign != 0.0 && delta.abs() < a.abs_t {
      let dt = a.sign * delta;
      (a.sign, a.abs_t + dt, a.nu_abs_t + self.nu * dt, a.d_y + dt)
    } else {
      let t = a.sign * a.abs_t + delta;
      let sign_t = if t == 0.0 { 0.0 } else { t.signum() };
      (
        sign_t,
        t.abs(),
        self.nu * t.abs(),
        a.d_y + (t.abs() - a.abs_t),
      )
    }
  }

  fn ln_h(&self, cond: &Conditional, a: &Anchor, delta: f64) -> f64 {
    let (sign_t, abs_t, _, d) = self.point(a, delta);
    let (sign_a, ln_a) = cond.argument(sign_t, abs_t, -d);
    self.wide.ln_cdf(sign_a, ln_a)
  }

  /// `(ψ, ψ + ln H)` at offset `delta` from anchor `a`, `ψ = −ν ln cosh t`.
  fn eval(&self, cond: &Conditional, a: &Anchor, delta: f64) -> (f64, f64) {
    let (_, abs_t, nu_abs_t, _) = self.point(a, delta);
    let psi = if abs_t >= 1.0 {
      -nu_abs_t + self.nu * (LN_2 - (-2.0 * abs_t).exp().ln_1p())
    } else {
      -self.nu * ln_cosh(abs_t)
    };
    (psi, psi + self.ln_h(cond, a, delta))
  }

  /// A gap past `|t_y|` holds `H` at `T_{ν+1}(−c₀ρ)`, one inside it at 1 for `y > 0` and below `y`'s tail for `y < 0`.
  fn facing(&self, inside: bool, sign_y: f64, near: bool) -> Facing {
    if !inside {
      Facing::Plateau(self.ln_h_right)
    } else if sign_y > 0.0 {
      Facing::Plateau(0.0)
    } else {
      Facing::BelowTail { near }
    }
  }

  /// Whether `rest`, from offset `x` of `a` and away from `t = 0`, is under `e^{−45}` of `totals`: the concave `ψ` bounds
  /// its density by `e^ψ min(|end − x|, 1/(ν tanh|t|))`, and `H` by `max(H(x), H(end))` where monotone, else by 1.
  fn rest_negligible(
    &self,
    a: &Anchor,
    x: f64,
    (g, f): (f64, f64),
    rest: &Rest,
    totals: (f64, f64),
  ) -> bool {
    let t = a.sign * a.abs_t + x;
    let t_end = a.sign * a.abs_t + rest.end;
    if t == 0.0 || (t_end - t) * t <= 0.0 {
      return false;
    }
    let slope = self.nu * self.abs_t_at(a, x).tanh();
    let ln_bound = g + (rest.end - x).abs().min(slope.recip()).ln();
    let monotone = !(rest.turn > t.min(t_end) && rest.turn < t.max(t_end));
    let ln_h = if monotone {
      (f - g).max(rest.ln_h_end)
    } else {
      0.0
    };
    ln_bound <= totals.0 - NEGLIGIBLE_FALL && ln_bound + ln_h <= totals.1 - NEGLIGIBLE_FALL
  }

  /// A march from `a` into a gap, until the density is a power law and `facing` holds.
  fn gap_march(
    &self,
    cond: &Conditional,
    a: &Anchor,
    step: f64,
    end: f64,
    facing: Facing,
    outer: (f64, f64),
  ) -> March {
    let eval = |delta: f64| self.eval(cond, a, delta);
    march(
      eval,
      end,
      step,
      &[],
      outer,
      |x, (_, f), (_, own_n)| match facing {
        Facing::Plateau(plateau) => self.settled(cond, a, x, plateau),
        Facing::BelowTail { near } => {
          self.abs_t_at(a, x) >= self.power_from && (!near || f <= own_n - NEGLIGIBLE_FALL)
        }
      },
    )
  }

  /// The density a power law at offset `x` from `a` and `ln H` there, formed directly rather than as a difference of the
  /// integrands' exponents, within `SETTLED` of `plateau`.
  fn settled(&self, cond: &Conditional, a: &Anchor, x: f64, plateau: f64) -> bool {
    self.abs_t_at(a, x) >= self.power_from && (self.ln_h(cond, a, x) - plateau).abs() <= SETTLED
  }

  fn far(&self, dist: f64) -> bool {
    dist > 2.0 * self.reach + 50.0
  }

  /// Whether the end of the gap between `a` and `b` nearer `|t_y|` is `a`'s.
  fn near_is_left(&self, a: &Anchor, b: &Anchor) -> bool {
    a.d_y.abs() <= b.d_y.abs()
  }

  fn abs_t_at(&self, a: &Anchor, delta: f64) -> f64 {
    if a.sign != 0.0 && delta.abs() < a.abs_t {
      a.abs_t + a.sign * delta
    } else {
      (a.sign * a.abs_t + delta).abs()
    }
  }

  /// `|t| − |t_y|` at offset `delta` from `a`, exact.
  fn d_at(&self, a: &Anchor, delta: f64) -> f64 {
    a.d_y + (self.abs_t_at(a, delta) - a.abs_t)
  }

  fn nu_abs_t(&self, a: &Anchor, delta: f64) -> f64 {
    a.nu_abs_t + self.nu * (self.abs_t_at(a, delta) - a.abs_t)
  }

  /// `ln ∫_{|t|}^{∞} 2^ν e^{−ν s} ds` from `ν|t|`.
  fn ln_power_mass(&self, nu_abs_t: f64) -> f64 {
    self.nu * LN_2 - nu_abs_t - self.nu.ln()
  }

  /// `ln ∫_{|t|}^{|t| + width} 2^ν e^{−ν s} ds` from `ν|t|`.
  fn ln_power_band(&self, nu_abs_t: f64, width: f64) -> f64 {
    self.ln_power_mass(nu_abs_t) + (-(-self.nu * width).exp_m1()).ln()
  }
}

/// `ln(eᵃ + eᵇ)` of two pairs of log-sums.
fn totals(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
  (ln_add_exp(a.0, b.0), ln_add_exp(a.1, b.1))
}

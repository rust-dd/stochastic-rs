//! The quadrature of the elliptical copulas' cdfs: `ln ∫ e^f` by Gauss–Kronrod panels marched out of an anchor, sized by
//! the variation of `f` and Kronrod's error estimate, so the integral keeps its relative digits wherever it lies.

use std::num::NonZeroUsize;
use std::sync::OnceLock;

use gauss_quad::GaussLegendre;

use crate::bivariate::conditional::ln_add_exp;
use crate::optim::SQRT_MACHEPS;
use crate::optim::localmin;

/// The variation of an exponent one panel may span before it is halved untried; Kronrod's estimate decides the rest.
const PANEL_FALL: f64 = 12.0;

/// Below the maximum by this the rest of the integral is under `e^{−50}` of it, as `f` falls at least linearly.
const TAIL_FALL: f64 = 50.0;

/// A maximum below this leaves nothing an f64 holds.
const NEGLIGIBLE: f64 = -1000.0;

/// The 16-point Gauss–Legendre `(node, weight)` pairs on `[−1, 1]`.
pub(crate) fn rule() -> &'static [(f64, f64)] {
  static RULE: OnceLock<GaussLegendre> = OnceLock::new();
  RULE
    .get_or_init(|| GaussLegendre::new(NonZeroUsize::new(16).expect("16 > 0")))
    .as_node_weight_pairs()
}

/// `(x, f(x))` at the maximum of a concave `f` on `[lo, hi]` (either end may be infinite): doubling uphill steps from
/// `guess` to a bracket, then Brent's `localmin` on `−f`; an end where `f` still rises is the maximum.
fn concave_max(f: &impl Fn(f64) -> f64, lo: f64, hi: f64, guess: f64) -> (f64, f64) {
  let x = guess.clamp(lo, hi);
  let fx = f(x);
  let mut step = 1.0;
  let up = if x + step <= hi && f(x + step) > fx {
    1.0
  } else if x - step >= lo && f(x - step) > fx {
    -1.0
  } else {
    let (a, b) = ((x - step).max(lo), (x + step).min(hi));
    return best(f, a, b, x, fx);
  };
  let (mut prev, mut cur, mut f_cur) = (x, x, fx);
  loop {
    let end = if up > 0.0 { hi } else { lo };
    let next = if up > 0.0 {
      (cur + step).min(end)
    } else {
      (cur - step).max(end)
    };
    let f_next = f(next);
    if f_next <= f_cur {
      let (a, b) = if up > 0.0 { (prev, next) } else { (next, prev) };
      return best(f, a, b, cur, f_cur);
    }
    if next == end {
      return (next, f_next);
    }
    (prev, cur, f_cur) = (cur, next, f_next);
    step *= 2.0;
  }
}

/// `localmin` on `(a, b)` against a known point inside, so a flat `f` keeps its best evaluated point.
fn best(f: &impl Fn(f64) -> f64, a: f64, b: f64, x: f64, fx: f64) -> (f64, f64) {
  if a.is_nan() || b.is_nan() || a >= b {
    return (x, fx);
  }
  let (xm, neg) = localmin(a, b, SQRT_MACHEPS, 1e-10 * (b - a), |t| -f(t));
  if -neg >= fx { (xm, -neg) } else { (x, fx) }
}

/// `ln ∫_lo^hi e^{f(x)} dx` for a concave `f`: `march` out of its maximum on each side until `f` is `TAIL_FALL` below
/// it; `breaks` end panels.
pub(crate) fn ln_integral_concave(
  f: impl Fn(f64) -> f64,
  lo: f64,
  hi: f64,
  guess: f64,
  breaks: &[f64],
) -> f64 {
  if lo.is_nan() || hi.is_nan() || lo >= hi {
    return f64::NEG_INFINITY;
  }
  let (x0, f0) = concave_max(&f, lo, hi, guess);
  // Every integrand here has curvature at least 1, so the integral is under `e^{f0}√(2π)`: zero in f64 below this.
  if f0 < NEGLIGIBLE || f0.is_nan() {
    return f0;
  }
  let eval = |d: f64| {
    let v = f(x0 + d);
    (v, v)
  };
  let breaks: Vec<f64> = breaks.iter().map(|&b| b - x0).collect();
  let side = |end: f64| {
    march(eval, end, 0.5, &breaks, |_, (_, fx), _| {
      fx <= f0 - TAIL_FALL
    })
    .ln_f
  };
  ln_add_exp(side(hi - x0), side(lo - x0))
}

/// QUADPACK's 15-point Kronrod nodes on `[0, 1)` (with their mirrors) and weights, and the weights of the 7-point Gauss
/// rule inside it on the odd nodes: K15 is exact to degree 23, G7 to 13, both checked in mpmath.
/// Reference: Piessens, R., de Doncker-Kapenga, E., Überhuber, C.W., Kahaner, D.K. (1983), "QUADPACK: A Subroutine Package for Automatic Integration", Springer, routine QK15.
const KRONROD_NODES: [f64; 8] = [
  0.991_455_371_120_812_6,
  0.949_107_912_342_758_5,
  0.864_864_423_359_769_1,
  0.741_531_185_599_394_4,
  0.586_087_235_467_691_1,
  0.405_845_151_377_397_2,
  0.207_784_955_007_898_5,
  0.0,
];

const KRONROD_WEIGHTS: [f64; 8] = [
  0.022_935_322_010_529_225,
  0.063_092_092_629_978_55,
  0.104_790_010_322_250_18,
  0.140_653_259_715_525_92,
  0.169_004_726_639_267_9,
  0.190_350_578_064_785_4,
  0.204_432_940_075_298_9,
  0.209_482_141_084_727_83,
];

const GAUSS_WEIGHTS: [f64; 4] = [
  0.129_484_966_168_869_7,
  0.279_705_391_489_276_7,
  0.381_830_050_505_118_9,
  0.417_959_183_673_469_4,
];

/// A panel is kept once K15 and G7 agree to this, relative: for an integrand analytic on the panel, G7's error `r^{−14}`
/// under `1e-9` puts K15's `r^{−23}` under `2e-15`, and the integrands' own rounding noise stays below it.
const KRONROD_AGREEMENT: f64 = 1e-9;

/// What `march` integrated: `ln ∫ e^g` and `ln ∫ e^f` from offset 0 to `end`, and `(g, f)` at `end`.
pub(crate) struct March {
  pub(crate) ln_g: f64,
  pub(crate) ln_f: f64,
  pub(crate) end: f64,
  pub(crate) last: (f64, f64),
}

/// `∫ e^g` and `∫ e^f` of `eval(x) = (g, f)` from offset 0 toward `end`, either sign: Kronrod panels halved while either
/// exponent moves past `PANEL_FALL` across one or K15 and G7 part by more than `KRONROD_AGREEMENT`, doubled after a gentle
/// one, ended at `breaks`; `stop(x, (g, f), (∫ e^g, ∫ e^f) so far)` ends the march at a panel end.
pub(crate) fn march(
  eval: impl Fn(f64) -> (f64, f64),
  end: f64,
  first_step: f64,
  breaks: &[f64],
  mut stop: impl FnMut(f64, (f64, f64), (f64, f64)) -> bool,
) -> March {
  let (mut x, (mut gx, mut fx)) = (0.0, eval(0.0));
  let (mut ln_g, mut ln_f) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
  let mut step = first_step;
  while x != end {
    let mut next = if end > 0.0 {
      (x + step).min(end)
    } else {
      (x - step).max(end)
    };
    for &b in breaks {
      if (b - x) * end > 0.0 && (next - b) * end > 0.0 {
        next = b;
      }
    }
    if next == x {
      break;
    }
    let (gn, f_next) = eval(next);
    let width = (next - x).abs();
    let refinable = width > 1e-300 && width > 8.0 * f64::EPSILON * x.abs();
    // An end where an exponent is −∞ is a zero of the integrand; the panel's variation is read just inside it.
    let inside = if gn == f64::NEG_INFINITY || f_next == f64::NEG_INFINITY {
      eval(x + 0.9 * (next - x))
    } else {
      (gn, f_next)
    };
    let moves = moved(gx, inside.0).max(moved(fx, inside.1));
    if moves > PANEL_FALL && refinable {
      step = 0.5 * width;
      continue;
    }
    let (mid, half) = (0.5 * (x + next), 0.5 * (next - x));
    let mut nodes = [(0.0, 0.0); 15];
    for (i, &t) in KRONROD_NODES.iter().enumerate() {
      nodes[i] = eval(mid + half * t);
      if i < 7 {
        nodes[14 - i] = eval(mid - half * t);
      }
    }
    let (g, f) = (
      kronrod(&nodes, |n| n.0, half),
      kronrod(&nodes, |n| n.1, half),
    );
    if refinable && (unsettled(g, ln_g) || unsettled(f, ln_f)) {
      step = 0.5 * width;
      continue;
    }
    ln_g = ln_add_exp(ln_g, g.0);
    ln_f = ln_add_exp(ln_f, f.0);
    if moves < 0.25 * PANEL_FALL {
      step = step.max(2.0 * width);
    }
    (x, gx, fx) = (next, gn, f_next);
    if stop(x, (gx, fx), (ln_g, ln_f)) {
      break;
    }
  }
  March {
    ln_g,
    ln_f,
    end: x,
    last: (gx, fx),
  }
}

/// A panel `(ln K15, |K15 − G7|/K15)` whose rules part by more than `KRONROD_AGREEMENT`, unless its error is under
/// `e^{−40}` of the integral so far, as at a zero of the integrand, where rounding noise would otherwise halve forever.
fn unsettled(panel: (f64, f64), ln_total: f64) -> bool {
  panel.1 > KRONROD_AGREEMENT && panel.0 + panel.1.ln() > ln_total - 40.0
}

/// `|a − b|`, 0 between two `−∞` and `∞` between `−∞` and a finite value.
fn moved(a: f64, b: f64) -> f64 {
  if a == b { 0.0 } else { (a - b).abs() }
}

/// `(ln K15, |K15 − G7|/K15)` of one exponent over a panel's nodes, which run from the right end to the left.
fn kronrod(nodes: &[(f64, f64); 15], pick: impl Fn(&(f64, f64)) -> f64, half: f64) -> (f64, f64) {
  let peak = nodes.iter().fold(f64::NEG_INFINITY, |m, n| m.max(pick(n)));
  if peak == f64::NEG_INFINITY {
    return (peak, 0.0);
  }
  let term = |i: usize| (pick(&nodes[i]) - peak).exp();
  let (mut k, mut g) = (KRONROD_WEIGHTS[7] * term(7), GAUSS_WEIGHTS[3] * term(7));
  for i in 0..7 {
    let pair = term(i) + term(14 - i);
    k += KRONROD_WEIGHTS[i] * pair;
    if i % 2 == 1 {
      g += GAUSS_WEIGHTS[i / 2] * pair;
    }
  }
  (peak + (k * half.abs()).ln(), (k - g).abs() / k)
}

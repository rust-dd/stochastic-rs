//! The crate's derivative-free solvers: a Nelder–Mead simplex for the two-parameter fits (Nelder & Mead 1965; Lagarias,
//! Reeds, Wright & Wright 1998 parameters), Brent's procedure `zero` for every root and `localmin` for the vine pairs' likelihood.
//!
//! Reference: Brent, R.P. (1973), "Algorithms for Minimization without Derivatives", Prentice-Hall, ch. 4: eqs. (2.9), (3.2)–(3.4) and procedure `zero`, §6, pp. 58–59; ch. 5: eqs. (4.1)–(4.2) and procedure `localmin`, §8, pp. 79–80, with the errata of maths-people.anu.edu.au/~brent/pub/pub011.html.

use stochastic_rs_distributions::RealExt;

/// Brent's `macheps` for rounded binary64, half of `f64::EPSILON`, so his `2 × macheps` is `f64::EPSILON`.
const MACHEPS: f64 = f64::EPSILON / 2.0;

/// Brent's `zero`: a zero of `f` in `[a, b]` within `6 macheps |x| + 2t` (`t > 0`), given `f(a)` and `f(b)` of opposite
/// signs; `None` without a sign change (a NaN end included), or past Brent's evaluation bound.
pub(crate) fn zero(a: f64, b: f64, t: f64, mut f: impl FnMut(f64) -> f64) -> Option<f64> {
  let (mut a, mut b) = (a, b);
  let (mut fa, mut fb) = (f(a), f(b));
  if !((fa <= 0.0 && fb >= 0.0) || (fa >= 0.0 && fb <= 0.0)) {
    return None;
  }
  let bound = evaluation_bound(a, b, t);
  let mut evaluations = 2;
  let (mut c, mut fc) = (a, fa);
  let mut d = b - a;
  let mut e = d;
  loop {
    if fc.abs() < fb.abs() {
      a = b;
      b = c;
      c = a;
      fa = fb;
      fb = fc;
      fc = fa;
    }
    let tol = 2.0 * MACHEPS * b.abs() + t;
    let m = 0.5 * (c - b);
    if !(m.abs() > tol && fb != 0.0) {
      return Some(b);
    }
    if e.abs() < tol || fa.abs() <= fb.abs() {
      d = m;
      e = m;
    } else {
      let mut s = fb / fa;
      let (mut p, mut q) = if a == c {
        (2.0 * m * s, 1.0 - s)
      } else {
        let (q, r) = (fa / fc, fb / fc);
        (
          s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0)),
          (q - 1.0) * (r - 1.0) * (s - 1.0),
        )
      };
      if p > 0.0 {
        q = -q;
      } else {
        p = -p;
      }
      s = e;
      e = d;
      if 2.0 * p < 3.0 * m * q - (tol * q).abs() && p < (0.5 * s * q).abs() {
        d = p / q;
      } else {
        d = m;
        e = m;
      }
    }
    a = b;
    fa = fb;
    b += if d.abs() > tol {
      d
    } else if m > 0.0 {
      tol
    } else {
      -tol
    };
    fb = f(b);
    evaluations += 1;
    if evaluations > bound {
      return None;
    }
    if (fb > 0.0) == (fc > 0.0) {
      c = a;
      fc = fa;
      d = b - a;
      e = d;
    }
  }
}

/// Brent's `√macheps`, the `eps` he recommends for `localmin`: a smooth minimum is located to that relative precision only.
pub(crate) const SQRT_MACHEPS: f64 = 1.053_671_212_772_350_9e-8;

/// Brent's `localmin` with the author's errata: a minimum of `f` on `(a, b)`, never evaluated at `a` or `b`, within `3 tol`,
/// `tol = eps |x| + t`, when `f` is δ-unimodal for a `δ < tol` (`eps ≥ 2 macheps`, `t > 0`); returns `(x, f(x))`.
pub(crate) fn localmin(
  a: f64,
  b: f64,
  eps: f64,
  t: f64,
  mut f: impl FnMut(f64) -> f64,
) -> (f64, f64) {
  let (mut a, mut b) = (a, b);
  let c = 0.5 * (3.0 - 5.0_f64.sqrt());
  let mut x = a + c * (b - a);
  let (mut v, mut w) = (x, x);
  let (mut d, mut e) = (0.0_f64, 0.0_f64);
  let mut fx = f(x);
  let (mut fv, mut fw) = (fx, fx);
  loop {
    let m = 0.5 * (a + b);
    let tol = eps * x.abs() + t;
    let t2 = 2.0 * tol;
    if (x - m).abs() > t2 - 0.5 * (b - a) {
      let (mut p, mut q, mut r) = (0.0, 0.0, 0.0);
      if e.abs() > tol {
        r = (x - w) * (fx - fv);
        q = (x - v) * (fx - fw);
        p = (x - v) * q - (x - w) * r;
        q = 2.0 * (q - r);
        if q > 0.0 {
          p = -p;
        } else {
          q = -q;
        }
        r = e;
        e = d;
      }
      // Brent's errata correct the printed `p < q × (a − x)` to `p > q × (a − x)`, as his FORTRAN has it.
      if p.abs() < (0.5 * q * r).abs() && p > q * (a - x) && p < q * (b - x) {
        d = p / q;
        let u = x + d;
        if u - a < t2 || b - u < t2 {
          d = if x < m { tol } else { -tol };
        }
      } else {
        let far = if x < m { b } else { a };
        e = far - x;
        d = c * e;
      }
      let step = if d.abs() >= tol {
        d
      } else if d > 0.0 {
        tol
      } else {
        -tol
      };
      let u = x + step;
      let fu = f(u);
      if fu <= fx {
        if u < x {
          b = x;
        } else {
          a = x;
        }
        (v, fv, w, fw, x, fx) = (w, fw, x, fx, u, fu);
      } else {
        if u < x {
          a = u;
        } else {
          b = u;
        }
        if fu <= fw || w == x {
          (v, fv, w, fw) = (w, fw, u, fu);
        } else if fu <= fv || v == x || v == w {
          (v, fv) = (u, fu);
        }
      }
    } else {
      return (x, fx);
    }
  }
}

/// Brent's bound `(k + 1)² − 2` (3.4), `k = ⌈log₂(|b − a|/δ_m)⌉` (3.2), `δ_m` the least tolerance `2 macheps |x| + t`
/// (3.3) on the interval; `zero` takes two evaluations when `k ≤ 0`.
fn evaluation_bound(a: f64, b: f64, t: f64) -> usize {
  let nearest = if a.min(b) <= 0.0 && a.max(b) >= 0.0 {
    0.0
  } else {
    a.abs().min(b.abs())
  };
  let ratio = (b - a).abs() / (2.0 * MACHEPS * nearest + t);
  if ratio <= 1.0 {
    return 2;
  }
  // ⌈log₂ ratio⌉ read off the binary exponent, exact where `log2().ceil()` can round across a power of two.
  let bits = ratio.to_bits();
  let k = (bits >> 52) as usize - 1023 + usize::from(bits & ((1 << 52) - 1) != 0);
  (k + 1).pow(2) - 2
}

/// Minimises `f` from `start` over a simplex of initial `steps`; returns the best vertex after
/// `max_iter` iterations or once the simplex spread falls below `tolerance`.
pub(crate) fn nelder_mead(
  f: impl Fn(&[f64]) -> f64,
  start: &[f64],
  steps: &[f64],
  max_iter: usize,
  tolerance: f64,
) -> Vec<f64> {
  let n = start.len();
  assert_eq!(steps.len(), n, "one initial step per coordinate");
  let mut simplex: Vec<Vec<f64>> = (0..=n)
    .map(|i| {
      let mut v = start.to_vec();
      if i > 0 {
        v[i - 1] += steps[i - 1];
      }
      v
    })
    .collect();
  let mut values: Vec<f64> = simplex.iter().map(|v| f(v)).collect();
  let (alpha, gamma, rho, sigma) = (1.0, 2.0, 0.5, 0.5);
  for _ in 0..max_iter {
    let mut order: Vec<usize> = (0..=n).collect();
    order.sort_by(|&a, &b| {
      values[a]
        .partial_cmp(&values[b])
        .unwrap_or(std::cmp::Ordering::Equal)
    });
    let simplex_sorted: Vec<Vec<f64>> = order.iter().map(|&i| simplex[i].clone()).collect();
    let values_sorted: Vec<f64> = order.iter().map(|&i| values[i]).collect();
    simplex = simplex_sorted;
    values = values_sorted;
    let spread = simplex[1..]
      .iter()
      .map(|v| {
        v.iter()
          .zip(&simplex[0])
          .map(|(a, b)| (a - b).abs())
          .fold(0.0, f64::max_or_nan)
      })
      .fold(0.0, f64::max_or_nan);
    if spread < tolerance && (values[n] - values[0]).abs() < tolerance {
      break;
    }
    let centroid: Vec<f64> = (0..n)
      .map(|j| simplex[..n].iter().map(|v| v[j]).sum::<f64>() / n as f64)
      .collect();
    let worst = simplex[n].clone();
    let reflect: Vec<f64> = centroid
      .iter()
      .zip(&worst)
      .map(|(c, w)| c + alpha * (c - w))
      .collect();
    let f_reflect = f(&reflect);
    if f_reflect < values[0] {
      let expand: Vec<f64> = centroid
        .iter()
        .zip(&reflect)
        .map(|(c, r)| c + gamma * (r - c))
        .collect();
      let f_expand = f(&expand);
      if f_expand < f_reflect {
        simplex[n] = expand;
        values[n] = f_expand;
      } else {
        simplex[n] = reflect;
        values[n] = f_reflect;
      }
    } else if f_reflect < values[n - 1] {
      simplex[n] = reflect;
      values[n] = f_reflect;
    } else {
      let contract: Vec<f64> = if f_reflect < values[n] {
        centroid
          .iter()
          .zip(&reflect)
          .map(|(c, r)| c + rho * (r - c))
          .collect()
      } else {
        centroid
          .iter()
          .zip(&worst)
          .map(|(c, w)| c + rho * (w - c))
          .collect()
      };
      let f_contract = f(&contract);
      if f_contract < values[n].min(f_reflect) {
        simplex[n] = contract;
        values[n] = f_contract;
      } else {
        let best = simplex[0].clone();
        for i in 1..=n {
          simplex[i] = simplex[i]
            .iter()
            .zip(&best)
            .map(|(x, b)| b + sigma * (x - b))
            .collect();
          values[i] = f(&simplex[i]);
        }
      }
    }
  }
  let best = (0..=n)
    .min_by(|&a, &b| {
      values[a]
        .partial_cmp(&values[b])
        .unwrap_or(std::cmp::Ordering::Equal)
    })
    .expect("non-empty simplex");
  simplex[best].clone()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn zero_finds_the_fixed_point_of_the_cosine() {
    let root = zero(0.0, 1.0, f64::MIN_POSITIVE, |x| x.cos() - x).unwrap();
    assert!((root - 0.739_085_133_215_160_6).abs() <= 2e-16, "{root}");
  }

  #[test]
  fn zero_needs_a_sign_change_and_returns_an_exact_end() {
    assert_eq!(zero(1.0, 2.0, 1e-12, |x| x), None);
    assert_eq!(zero(0.0, 2.0, 1e-12, |x| x), Some(0.0));
  }

  #[test]
  fn the_bound_on_the_inverse_bracket_is_brents_worst_case() {
    let t = crate::bivariate::conditional::ABSOLUTE_TOLERANCE;
    assert_eq!(evaluation_bound(f64::EPSILON, 1.0, t), 105 * 105 - 2);
    assert_eq!(evaluation_bound(0.0, 2.0, 1.0), 2 * 2 - 2);
    assert_eq!(evaluation_bound(0.0, 4.0, 1.0), 3 * 3 - 2);
    assert_eq!(evaluation_bound(0.0, 4.000000000000001, 1.0), 4 * 4 - 2);
  }

  #[test]
  fn sqrt_macheps_is_the_root_of_macheps() {
    assert_eq!(SQRT_MACHEPS, MACHEPS.sqrt());
  }

  /// Parabolic interpolation lands on a parabola's vertex at once; with the printed `p < q × (a − x)` it took 39 evaluations.
  #[test]
  fn localmin_steps_onto_a_parabolas_vertex() {
    let mut evaluations = 0;
    let (x, fx) = localmin(0.0, 3.0, SQRT_MACHEPS, 1e-10, |x| {
      evaluations += 1;
      (x - 1.5) * (x - 1.5)
    });
    assert!((x - 1.5).abs() <= 1e-12 && fx <= 1e-24, "{x} {fx}");
    assert!(evaluations <= 6, "{evaluations} evaluations");
  }

  /// `x³ − 2x − 5` has its minimum at `√(2/3)` on `(0, 1)`, which `localmin` finds within `3 tol` (Brent ch. 5, §4).
  #[test]
  fn localmin_finds_an_interior_minimum_within_three_tolerances() {
    let t = 1e-10;
    let (x, _) = localmin(0.0, 1.0, SQRT_MACHEPS, t, |x| x * (x * x - 2.0) - 5.0);
    let tol = SQRT_MACHEPS * x.abs() + t;
    assert!((x - (2.0_f64 / 3.0).sqrt()).abs() <= 3.0 * tol, "{x}");
  }

  /// A minimum at an end is approached from inside, to within `2 tol`, and `f` is never evaluated at either end.
  #[test]
  fn localmin_stays_inside_its_interval() {
    let t = 1e-6;
    let mut outside = false;
    let (x, _) = localmin(0.0, 1.0, SQRT_MACHEPS, t, |x| {
      outside |= x <= 0.0 || x >= 1.0;
      x
    });
    assert!(
      !outside && x > 0.0 && x <= 2.0 * (SQRT_MACHEPS * x + t),
      "{x}"
    );
  }

  #[test]
  fn finds_the_minimum_of_a_rosenbrock_valley() {
    let f = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
    let best = nelder_mead(f, &[-1.2, 1.0], &[0.5, 0.5], 5000, 1e-12);
    assert!(
      (best[0] - 1.0).abs() < 1e-4 && (best[1] - 1.0).abs() < 1e-4,
      "{best:?}"
    );
  }
}

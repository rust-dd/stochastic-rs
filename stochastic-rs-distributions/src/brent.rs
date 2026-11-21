//! Brent's procedure `zero`, the root solver of the sibling crates; outside the stability promise.
//!
//! Reference: Brent, R.P. (1973), "Algorithms for Minimization without Derivatives", Prentice-Hall, ch. 4: eqs. (2.9), (3.2)–(3.4) and procedure `zero`, §6, pp. 58–59.

/// Brent's `macheps` for rounded binary64, half of `f64::EPSILON`, so his `2 × macheps` is
/// `f64::EPSILON`.
pub const MACHEPS: f64 = f64::EPSILON / 2.0;

/// A zero of `f` in `[a, b]` within Brent's `6 macheps |x| + 2t` (`t > 0`), at a stop width of
/// `4 macheps |x| + 2t`; `None` for ends of one sign or a NaN end, or past his evaluation bound.
pub fn zero(a: f64, b: f64, t: f64, mut f: impl FnMut(f64) -> f64) -> Option<f64> {
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

/// Brent's bound `(k + 1)² − 2` (3.4), `k = ⌈log₂(|b − a|/δ_m)⌉` (3.2), `δ_m` the least tolerance
/// `2 macheps |x| + t` (3.3) on the interval; `zero` takes two evaluations when `k ≤ 0`.
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
  // ⌈log₂ ratio⌉ read off the binary exponent, exact where `log2().ceil()` can round across a power
  // of two.
  let bits = ratio.to_bits();
  let k = (bits >> 52) as usize - 1023 + usize::from(bits & ((1 << 52) - 1) != 0);
  (k + 1).pow(2) - 2
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
  fn evaluation_bound_takes_brents_worst_case_with_an_exact_log2_ceiling() {
    let t = f64::from_bits(1);
    assert_eq!(evaluation_bound(f64::EPSILON, 1.0, t), 105 * 105 - 2);
    assert_eq!(evaluation_bound(0.0, 2.0, 1.0), 2 * 2 - 2);
    assert_eq!(evaluation_bound(0.0, 4.0, 1.0), 3 * 3 - 2);
    assert_eq!(evaluation_bound(0.0, 4.000000000000001, 1.0), 4 * 4 - 2);
  }

  #[test]
  fn zero_lands_within_its_bound_on_x_to_the_ninth_and_a_bond_residual() {
    let t = 1e-12;
    let root = zero(-1.0, 4.0, t, |x| x.powi(9)).unwrap();
    assert!(root.abs() <= 6.0 * MACHEPS * root.abs() + 2.0 * t, "{root}");
    let residual = |y: f64| {
      (1..=10).map(|k| 5.0 / (1.0 + y).powi(k)).sum::<f64>() + 100.0 / (1.0 + y).powi(10) - 103.0
    };
    let ytm = zero(1e-4, 0.2, t, residual).unwrap();
    // mpmath's root: ten coupons of 5 and a face of 100 priced at 103.
    let exact = 0.046_186_485_721_353_86;
    assert!(
      (ytm - exact).abs() <= 6.0 * MACHEPS * ytm + 2.0 * t,
      "{ytm}"
    );
  }

  #[test]
  fn zero_returns_its_best_iterate_not_the_far_end_of_the_bracket() {
    let root = zero(0.0, 1.0, 1e-3, |x| x - 0.3).unwrap();
    assert!((root - 0.3).abs() <= 4.0 * MACHEPS, "{root}");
  }

  #[test]
  fn tiny_residuals_still_bracket_and_an_unbracketed_pair_is_none() {
    let t = 1e-14;
    let root = zero(0.0, 1.0, t, |x| 1e-170 * (x - 0.3)).unwrap();
    assert!(
      (root - 0.3).abs() <= 6.0 * MACHEPS * root + 2.0 * t,
      "{root}"
    );
    assert_eq!(zero(-1.0, 1.0, 1e-12, |x| x * x + 1.0), None);
  }

  #[test]
  fn a_nan_end_is_not_a_sign_change() {
    let nan_at_one = |slope: f64| {
      move |x: f64| {
        if x == 1.0 {
          f64::NAN
        } else {
          slope * (x - 0.5)
        }
      }
    };
    assert_eq!(zero(0.0, 1.0, 1e-12, nan_at_one(1.0)), None);
    assert_eq!(zero(0.0, 1.0, 1e-12, nan_at_one(-1.0)), None);
  }
}

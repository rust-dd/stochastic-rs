//! Dunnett & Sobel's finite sums for `T_n(h, k; ρ)` at integer `n`, taken where their forward error bound is under the
//! closed forms' budget of the result; the `ν/2` terms recur as Genz's BVTL evaluates them.
//! Reference: Dunnett, C.W., Sobel, M. (1954), "A bivariate generalization of Student's t-distribution", *Biometrika* 41, 153-169; Genz, A. (2004), *Statistics and Computing* 14(3), 251-260, eq. (20); Higham, N.J. (2002), *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, ch. 3-4 (the forward bound).

use std::f64::consts::FRAC_2_PI;
use std::f64::consts::PI;
use std::f64::consts::TAU;

use crate::bivariate::gaussian::CLOSED_FORM_BUDGET;

const U: f64 = f64::EPSILON / 2.0;

/// Whether `n` is an integer whose `m = ⌊n/2⌋` terms leave the recurrence's own `(28m + 20)u` of the terms within the budget.
pub(crate) fn admits(n: f64) -> bool {
  n.fract() == 0.0 && n >= 1.0 && (28.0 * (0.5 * n).floor() + 20.0) * U <= CLOSED_FORM_BUDGET
}

/// `(√x, √(1 − x), sign d, r)` of `x = d²/r²`, `r² = d² + e²`, the incomplete beta argument of one sum.
fn beta_argument(d: f64, e: f64) -> (f64, f64, f64, f64) {
  let r = d.hypot(e);
  (d.abs() / r, e / r, if d < 0.0 { -1.0 } else { 1.0 }, r)
}

/// `T_n(h, k; ρ)` where the forward bound `u[(28m + 20)S + 1.13√(m + 1) Σ (20D/r + π)G]` is within the budget of it:
/// `S` the terms' absolute sum, `G` each sum's `|g|`, `D/r` its angle's conditioning, and `1.13√(m + 1)` the slope of its betas.
pub(crate) fn dunnett_sobel(h: f64, k: f64, rho: f64, n: u32) -> Option<f64> {
  let rn = f64::from(n).sqrt();
  let (rh, rk) = (rn.hypot(h), rn.hypot(k));
  let (sa, ca, sb, cb) = (h / rh, rn / rh, k / rk, rn / rk);
  let om = ((1.0 - rho) * (1.0 + rho)).sqrt();
  let (xk, yk, sk, dist_k) = beta_argument(sb * ca - rho * sa * cb, om * cb);
  let (xh, yh, sh, dist_h) = beta_argument(sa * cb - rho * sb * ca, om * ca);
  let (scale_k, scale_h) = (
    (sb * ca).abs() + (rho * sa * cb).abs(),
    (sa * cb).abs() + (rho * sb * ca).abs(),
  );
  let (c2a, c2b) = (ca * ca, cb * cb);
  let (mut total, mut spread, mut g_h, mut g_k) = (0.0, 0.0, 0.0, 0.0);
  let m = if n.is_multiple_of(2) {
    total += om.atan2(-rho) / TAU;
    spread += total;
    let (mut gh, mut gk) = (0.25 * sa, 0.25 * sb);
    let (mut ik, mut dk) = (FRAC_2_PI * xk.atan2(yk), FRAC_2_PI * xk * yk);
    let (mut ih, mut dh) = (FRAC_2_PI * xh.atan2(yh), FRAC_2_PI * xh * yh);
    for j in 1..=n / 2 {
      let j = f64::from(j);
      total += gh * (1.0 + sk * ik) + gk * (1.0 + sh * ih);
      spread += gh.abs() * (1.0 + ik) + gk.abs() * (1.0 + ih);
      (g_h, g_k) = (g_h + gh.abs(), g_k + gk.abs());
      ik += dk;
      dk *= 2.0 * j * yk * yk / (2.0 * j + 1.0);
      ih += dh;
      dh *= 2.0 * j * yh * yh / (2.0 * j + 1.0);
      gh *= (2.0 * j - 1.0) / (2.0 * j) * c2a;
      gk *= (2.0 * j - 1.0) / (2.0 * j) * c2b;
    }
    n / 2
  } else {
    let (alpha, beta) = (h.atan2(rn), k.atan2(rn));
    let one_minus_z = 2.0 * (0.5 * (alpha - beta)).sin().powi(2) + (1.0 - rho) * ca * cb;
    let one_plus_z = 2.0 * (0.5 * (alpha + beta)).sin().powi(2) + (1.0 + rho) * ca * cb;
    let asin_z = (sa * sb + rho * ca * cb).atan2((one_minus_z * one_plus_z).sqrt());
    total += 0.25 + (alpha + beta + asin_z) / TAU;
    spread += 0.25 + (alpha.abs() + beta.abs() + asin_z.abs()) / TAU;
    let (mut gh, mut gk) = (sa * ca / TAU, sb * cb / TAU);
    let (mut ik, mut dk, mut ih, mut dh) = (xk, xk, xh, xh);
    for j in 1..=(n - 1) / 2 {
      let j = f64::from(j);
      total += gh * (1.0 + sk * ik) + gk * (1.0 + sh * ih);
      spread += gh.abs() * (1.0 + ik) + gk.abs() * (1.0 + ih);
      (g_h, g_k) = (g_h + gh.abs(), g_k + gk.abs());
      dk *= (2.0 * j - 1.0) * yk * yk / (2.0 * j);
      ik += dk;
      dh *= (2.0 * j - 1.0) * yh * yh / (2.0 * j);
      ih += dh;
      gh *= 2.0 * j / (2.0 * j + 1.0) * c2a;
      gk *= 2.0 * j / (2.0 * j + 1.0) * c2b;
    }
    (n - 1) / 2
  };
  let m = f64::from(m);
  let angles = (20.0 * scale_k / dist_k + PI) * g_h + (20.0 * scale_h / dist_h + PI) * g_k;
  let bound = U * ((28.0 * m + 20.0) * spread + 1.13 * (m + 1.0).sqrt() * angles);
  (total > 0.0 && bound <= CLOSED_FORM_BUDGET * total).then_some(total)
}

#[cfg(test)]
mod tests {
  use super::*;

  /// At exact scores the sums, where taken, are within the budget of 40-digit values (checked at 60) from the conditional
  /// integral; a small result whose terms cancel is declined, and so is every `ν` past the recurrence's reach.
  #[test]
  fn the_sums_hold_where_taken() {
    let pins = [
      (
        -0.4322261842161799,
        -3.1761303309193165,
        0.5248430262176583,
        1,
        0.07123057190356022,
      ),
      (
        853454.286536603,
        0.4714682423852714,
        -0.9999988453161978,
        2,
        0.6581331182029092,
      ),
      (
        741.8291001145084,
        0.42814850475866045,
        -0.9999957618069593,
        3,
        0.6512897840763067,
      ),
      (
        -0.8090276834342608,
        1.783986486898753,
        -0.9999999957589499,
        4,
        0.15744142341961218,
      ),
      (
        1.395887028625087,
        0.4548364738262006,
        0.818261193296281,
        5,
        0.6569923907890689,
      ),
      (
        819.9906413190132,
        0.5866268971256592,
        -0.4297029947856713,
        6,
        0.7105794547970418,
      ),
      (
        4.636135698566773,
        -1.9921621211165204,
        -0.9999867429586031,
        7,
        0.0421171371434994,
      ),
      (
        27.155901193356726,
        -0.044801594798991096,
        0.05820329188919138,
        10,
        0.4825737362540084,
      ),
      (
        1.014740313012478,
        0.07585197494237382,
        -0.2826460475117213,
        30,
        0.4190049336464064,
      ),
      (
        10.554318545846368,
        6.413487679259254,
        -0.4451175154881186,
        60,
        0.9999999875990095,
      ),
    ];
    for (h, k, rho, n, want) in pins {
      let got =
        dunnett_sobel(h, k, rho, n).unwrap_or_else(|| panic!("T_{n}({h}, {k}; {rho}) declined"));
      assert!(
        (got - want).abs() <= CLOSED_FORM_BUDGET * want,
        "T_{n}({h}, {k}; {rho}) = {got:e}, want {want:e}"
      );
    }
    assert_eq!(
      dunnett_sobel(
        -0.560495732475435,
        -2.147225891682727,
        -0.5658874826143234,
        10
      ),
      None
    );
    assert!(admits(384.0) && !admits(386.0) && !admits(4.5));
  }
}

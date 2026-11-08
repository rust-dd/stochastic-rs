//! Pins of the incomplete beta from a shape of 100 and of its inverse: mpmath at 60 and 120 digits,
//! `betainc` under a shape of 2e3, the fraction or the kernel's panels above, the limits beside.

use super::*;
use crate::special::beta_ic;

fn rel(got: f64, want: f64) -> f64 {
  ((got - want) / want).abs()
}

/// The quantile's bar, `4e-15·max(1, κ)·max(1, |ln p̃|)` relative to the solved side, `κ =
/// p̃/(q·f(q))`.
fn quantile_bound(p: f64, kappa: f64, side: f64) -> f64 {
  let pt = p.min(1.0 - p);
  4e-15 * kappa.max(1.0) * pt.ln().abs().max(1.0) * side
}

/// Below the switch at a second shape in the thousands and beyond, where the plain `ln Γ` sum lost
/// 3.9e-7; the four cells with a shape under 1e-2 and the bits cell are guards, which the base met.
#[test]
fn large_shapes_keep_their_digits_below_the_switch() {
  for (a, b, x, want) in [
    (
      10.0,
      1778.2794100389228,
      0.005529863073041075,
      0.5277113404837002,
    ),
    (1e4, 0.5, 0.95, 4.340861783560434e-225),
    (1e8, 0.01, 0.999999, 3.879398110133378e-48),
    (1e4, 1e-9, 0.9999999999, 1.323834572425101e-8),
    (1e6, 10.0, 0.99999, 0.4579178290206093),
    (1e-12, 1e8, 1e-300, 0.9999999993282224),
    (1e-9, 1e4, 1e-10, 0.9999999867616542),
    (0.01, 1e8, 1e-10, 0.9603474234749158),
    (300.0, 200.0, 0.6, 0.497571383680068),
    (3000.0, 2000.0, 0.6, 0.4992322120544105),
  ] {
    let got = beta_i(a, b, x);
    assert!(
      rel(got, want) < 4e-14,
      "I_{x}({a:e}, {b:e}) = {got:e}, want {want:e}"
    );
  }
  assert_eq!(beta_i(1e4, 1e-307, 0.999), f64::from_bits(0x0013_7e69_693a));
}

/// Beside the mean at both shapes past 100 the uniform expansion converges where 200 fraction steps
/// read `−0.306`.
#[test]
fn both_shapes_large_converge_beside_the_mean() {
  for (a, b, x, want) in [
    (1e4, 1e4, 0.48, 7.610739113426522e-9),
    (1e6, 1e6, 0.4995, 0.07864957758090163),
    (1e8, 1e8, 0.4999, 0.002338867309689789),
    (1e12, 1e12, 0.5000001, 0.6113512945481627),
  ] {
    let got = beta_i(a, b, x);
    assert!(
      rel(got, want) < 4e-14,
      "I_{x}({a:e}, {b:e}) = {got:e}, want {want:e}"
    );
  }
  for (a, b) in [(1e6, 1e6), (1e8, 1e8), (1e300, 1e300), (f64::MAX, f64::MAX)] {
    let got = beta_i(a, b, 0.5);
    assert!(
      (got - 0.5).abs() <= 8.0 * f64::EPSILON,
      "I_½({a:e}, {b:e}) = {got}"
    );
  }
  assert_eq!(beta_i(1e300, 1e300, 0.5000000000000001), 1.0);
}

/// At a huge second shape past the switch a rounded `1 − x` read `1.0000000277` and `6.2`: every
/// value is in `[0, 1]`, none is NaN, a shape times a log past `MAX` reads its end.
#[test]
fn huge_shapes_stay_in_the_unit_interval() {
  let got = beta_i(0.5, 1e18, 1e-17);
  assert!(
    rel(got, 0.999992255783569) < 1e-12 && got <= 1.0,
    "I_1e-17(0.5, 1e18) = {got}"
  );
  for (a, b, x, want) in [
    (5e-324, 1e18, 5e-17, 1.0),
    (2.0, 1e19, 5e-17, 1.0),
    (1e200, 10.0, 0.5, 0.0),
    (1e300, f64::MAX, 0.3, 1.0),
    (1e300, f64::MAX, 5.551115123125783e-17, 0.0),
    (f64::MAX, 0.5, 0.5, 0.0),
    (f64::MAX, 10.0, 1e-300, 0.0),
    (f64::MAX, 0.9999999999999998, 5e-324, 0.0),
  ] {
    assert_eq!(beta_i(a, b, x), want, "I_{x:e}({a:e}, {b:e})");
  }
  let shapes = [
    5e-324,
    1e-300,
    1e-10,
    0.5,
    1.0,
    2.0,
    10.0,
    100.0,
    1e4,
    1e8,
    1e100,
    1e300,
    f64::MAX,
  ];
  let points = [
    5e-324,
    1e-300,
    1e-10,
    0.3,
    0.5,
    0.7,
    0.9999999999,
    0.9999999999999999,
  ];
  for a in shapes {
    for b in shapes {
      for x in points {
        let v = beta_i(a, b, x);
        assert!((0.0..=1.0).contains(&v), "I_{x:e}({a:e}, {b:e}) = {v}");
      }
    }
  }
}

/// Each route's `ln_ratio` is its value over the prefactor, so the inverse's slope forms no difference of
/// two logarithms of the exponent's size: the fraction on both sides, `asym`, the gamma limit and the series.
#[test]
fn every_route_reads_its_ratio_to_the_prefactor() {
  for (a, b, x) in [
    (1e4, 0.5, 0.95),
    (300.0, 200.0, 0.75),
    (300.0, 200.0, 0.6),
    (1e9, 0.5, 1.0 - 1e-8),
    (0.5, 1e4, 1e-4),
    (1e4, 0.5, 1.0 - 1e-4),
  ] {
    let y = 1.0 - x;
    let (ln_x, ln_y) = large::logs(x, y);
    let s = large::side(a, b, x, y, ln_x, ln_y);
    let want = s.value / s.ln_prefactor.exp();
    let got = s.ln_ratio.exp();
    assert!(
      rel(got, want) < 1e-12,
      "I_{x}({a:e}, {b:e}): ratio {got:e}, want {want:e}"
    );
  }
}

/// A law narrower than its mean's double has every level's root at the mean, read to the bit where Newton on a
/// step wandered by up to 8 % (mpmath's `a/(a + b)` at 60 digits); the last four cells are guards.
#[test]
fn narrow_laws_read_their_mean_to_the_bit() {
  for (a, b, p, want) in [
    (
      4.019196191912396e278,
      6.622884699085498e275,
      0.9999999999999686,
      0.9983548975815724,
    ),
    (
      4.229330893408978e281,
      3.257588714627999e281,
      0.9994451593453486,
      0.5648959939237123,
    ),
    (
      4.406147724036598e154,
      1.278876868892976e164,
      3.061506620330342e-11,
      3.4453259963428655e-10,
    ),
    (
      4.560981577381101e263,
      4.671016195168667e280,
      2.828698299607086e-62,
      9.764431093385254e-18,
    ),
    (1e40, 3e39, 1e-300, 0.7692307692307693),
    (1e40, 3e39, 0.5, 0.7692307692307693),
  ] {
    assert_eq!(beta_i_inv(a, b, p), want, "Beta({a:e}, {b:e}) at {p:e}");
  }
}

/// `beta_ic` reads its plain route from `x` itself where the gamma limit's terms are too large.
#[test]
fn beta_ic_reads_its_plain_route_from_x() {
  let got = beta_ic(1e8, 1e16, 1e-8);
  assert!(rel(got, 0.49994680769616147) < 4e-14, "{got}");
}

/// mpmath roots (Newton on `ln I` at 60 and 120 digits raised), graded by the quantile's bar with κ
/// from the root's density, the complement past ½; the twelve rows in the bulk are guards the clamp met.
#[test]
fn the_inverse_holds_the_bar_in_both_tails() {
  for (a, b, p, want, kappa) in [
    (0.5, 0.5, 1e-20, 2.4674011002723395e-40, 2.0),
    (0.5, 0.5, 1e-10, 2.4674011002723397e-20, 2.0),
    (0.5, 3.0, 1e-10, 2.8444444444444445e-21, 2.0),
    (2.0, 5.0, 1e-300, 2.5819888974716112e-151, 0.5),
    (2.0, 5.0, 1e-20, 2.5819888975605e-11, 0.5),
    (2.0, 5.0, 1e-10, 2.581997786411184e-6, 0.5000017),
    (2.0, 5.0, 0.5, 0.26444998329565994, 0.8141657),
    (2.0, 5.0, 0.9999999999, 0.9930035512206541, 0.2002349),
    (2.0, 5.0, 0.9999999999999999, 0.9995497181989903, 0.200015),
    (0.5, 30.0, 1e-20, 2.6399006310995208e-42, 2.0),
    (0.5, 30.0, 1e-10, 2.6399006310995213e-22, 2.0),
    (0.5, 30.0, 0.5, 0.007616525601879419, 2.323856),
    (0.5, 30.0, 0.9999999999, 0.5046609979043831, 0.03282827),
    (
      0.5,
      30.0,
      0.9999999999999999,
      0.6848547697018144,
      0.03309109,
    ),
    (30.0, 0.5, 1e-300, 1.0788593622087e-10, 0.03333333),
    (30.0, 0.5, 1e-20, 0.2314526681899776, 0.03317366),
    (30.0, 0.5, 1e-10, 0.4953390007501658, 0.03282827),
    (30.0, 0.5, 0.5, 0.9923834743981206, 0.01783555),
    (1e3, 1e3, 1e-300, 0.14764444133469024, 0.001208899),
    (1e3, 1e3, 1e-20, 0.3975320608971773, 0.002907376),
    (1e3, 1e3, 1e-10, 0.429227551117347, 0.003940386),
    (1e3, 1e3, 0.9999999999, 0.5707724487427124, 0.003940386),
    (
      1e3,
      1e3,
      0.9999999999999999,
      0.5910287098569674,
      0.003201013,
    ),
    (3.0, 1e-9, 1e-10, 0.5540314914937051, 0.2622411),
    (5.0, 1e-9, 1e-300, 8.705505629333948e-59, 0.2),
    (1.0, 1e-9, 1e-10, 0.09516258196856461, 0.9508332),
    (2.0, 1e-9, 1e-10, 0.3831831680981665, 0.420091),
    (5.0, 1e-9, 1e-10, 0.7177207036451614, 0.1482182),
  ] {
    let got = beta_i_inv(a, b, p);
    let (err, side) = if p <= 0.5 {
      ((got - want).abs(), want)
    } else {
      (((1.0 - got) - (1.0 - want)).abs(), 1.0 - want)
    };
    let bound = quantile_bound(p, kappa, side);
    assert!(
      err <= bound,
      "Beta({a}, {b}) at {p:e}: {got:e}, want {want:e}, off {err:e} against {bound:e}"
    );
  }
}

/// A root below the subnormals is exactly 0, one whose complement rounds to 1 is 1, and a symmetric
/// law's median is ½ (the medians below 1e-2 are guards): the clamp at `1e-14` is gone.
#[test]
fn the_inverse_reads_the_ends_and_the_symmetric_median_exactly() {
  for (a, b, p) in [
    (0.01, 0.01, 1e-20),
    (0.01, 0.01, 1e-10),
    (0.5, 0.5, 1e-300),
    (0.5, 30.0, 1e-300),
    (5e-324, 0.5132987294888756, 0.9739206006071974),
  ] {
    assert_eq!(beta_i_inv(a, b, p), 0.0, "Beta({a}, {b}) at {p:e}");
  }
  for (a, b, p) in [
    (0.01, 0.01, 0.9999999999),
    (0.5, 0.5, 0.9999999999999999),
    (30.0, 0.5, 0.9999999999),
    (0.5, 1e-310, 0.5),
    (0.5, 1e-310, 1e-10),
    (1.0, 5e-324, 0.3),
    (2.0, 1e-310, 0.9),
    (
      7.442481542949293e199,
      0.04568703665247251,
      2.790842465187859e-200,
    ),
    (
      8.112671229848719e250,
      0.5908267451988093,
      4.163423542827604e-170,
    ),
  ] {
    assert_eq!(beta_i_inv(a, b, p), 1.0, "Beta({a}, {b}) at {p:e}");
  }
  for a in [5e-324, 4e-309, 5.5e-309, 0.01, 0.5, 1e3, 1e8] {
    assert_eq!(beta_i_inv(a, a, 0.5), 0.5, "Beta({a:e}, {a:e})");
  }
}

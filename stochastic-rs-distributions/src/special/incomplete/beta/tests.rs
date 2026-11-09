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

/// Below the switch at a second shape in the thousands and beyond, within 4e-14 of mpmath at 60 and
/// 120 digits; the four cells with a shape under 1e-2 are guards.
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
}

/// Beside the mean at both shapes past 100 the uniform expansion holds 4e-14 of mpmath, and `½` at
/// `x = ½` within 8ε.
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

/// At a huge second shape past the switch every value is in `[0, 1]` and none is NaN; a shape times
/// a log past `MAX` reads its end. The `(2, 1e19)` cell is a guard.
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
    (
      1.8406996318833632e111,
      1.9672203795406507e296,
      9.196176908270334e-186,
      0.0,
    ),
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

/// Each route's `ln_ratio` is its value over the prefactor, so the inverse's slope forms no
/// difference of two large logs: both fractions, `asym`, the gamma limit and the series.
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
    let s = large::side(a, b, x, y, Some((ln_x, ln_y)), None);
    let want = s.value / s.ln_prefactor.exp();
    let got = s.ln_ratio.exp();
    assert!(
      rel(got, want) < 1e-12,
      "I_{x}({a:e}, {b:e}): ratio {got:e}, want {want:e}"
    );
  }
}

/// A law narrower than its mean's double has every root within that double's spacing: `a/(a + b) +
/// σz` to the bit (mpmath at 60 digits); the third to sixth and the last two cells are guards.
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
    (1e34, 1e37, 0.3, 0.000999000999000999),
    (1e33, 3e33, 0.7, 0.25),
    (
      1.8406996318833632e111,
      1.9672203795406507e296,
      1.8063081215370482e-238,
      9.356855241166064e-186,
    ),
    (
      4.09174164464156e49,
      5.102096665990561e219,
      2.4384541642201333e-30,
      8.019725835294729e-171,
    ),
  ] {
    assert_eq!(beta_i_inv(a, b, p), want, "Beta({a:e}, {b:e}) at {p:e}");
  }
}

/// At the largest shape every `x` to `0.999` saturates the tail, `0` with `MAX` first and `1`
/// second, and `(MAX, MAX)` steps at ½; the two-sums of the offset are read from the halves there.
#[test]
fn the_largest_shape_reads_its_ends_at_every_point() {
  for b in [1e-300, 1.3e-47, 0.5, 1.0, 1.3, 3.0, 10.0, 1e3, 1e100] {
    for i in 1..1000 {
      let x = i as f64 / 1000.0;
      assert_eq!(beta_i(f64::MAX, b, x), 0.0, "I_{x}(MAX, {b:e})");
      assert_eq!(beta_i(b, f64::MAX, x), 1.0, "I_{x}({b:e}, MAX)");
    }
  }
  for i in (1..1000).filter(|&i| i != 500) {
    let x = i as f64 / 1000.0;
    let want = if x < 0.5 { 0.0 } else { 1.0 };
    assert_eq!(beta_i(f64::MAX, f64::MAX, x), want, "I_{x}(MAX, MAX)");
  }
}

/// The deep lower tail starts on its power law a few steps from the root, where A&S 26.5.22 lies
/// 190 log units out; the root agrees with `x^a/(aB)` at 50 digits, whose next term is 1e-238.
#[test]
fn the_deep_tail_converges_in_a_few_steps() {
  let (x, steps) = beta_i_inv_steps(1.36, 3000.0, 5e-324);
  assert!(
    steps <= 10 && rel(x, 7.223334001387436e-242) < 3e-12,
    "Beta(1.36, 3000) at 5e-324: {x:e} in {steps} steps"
  );
  for (a, b, p) in [
    (1.25, 300.0, 1e-310),
    (1.6, 1e6, 1e-320),
    (1.0141, 1.5336, 2.0e-173),
    (2.0, 5.0, 1e-300),
    (1e-200, 1e-200, 0.3),
    (f64::MAX, 1.3e-47, 3.9e-157),
  ] {
    let (x, steps) = beta_i_inv_steps(a, b, p);
    assert!(
      x.is_finite() && steps < 40,
      "Beta({a:e}, {b:e}) at {p:e}: {x:e} in {steps} steps"
    );
  }
  assert_eq!(beta_i_inv(f64::MAX, 1.3e-47, 3.9e-157), 1.0);
  assert_eq!(beta_i_inv(1e-200, 1e-200, 0.3), 0.0);
}

/// `beta_ic` reads its plain route from `x` itself where the gamma limit's terms are too large.
#[test]
fn beta_ic_reads_its_plain_route_from_x() {
  let got = beta_ic(1e8, 1e16, 1e-8);
  assert!(rel(got, 0.49994680769616147) < 4e-14, "{got}");
}

/// mpmath roots (Newton on `ln I` at 60 and 120 digits raised), graded by the quantile's bar with κ
/// at the root, the complement past ½; twelve bulk rows are guards, three roots near 1e-240.
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
    (
      5.207052719351951,
      4.800787204511878e240,
      0.17713598640521838,
      6.493710750844055e-241,
      0.3536874,
    ),
    (
      60.74187873563975,
      7.16704305166794e128,
      0.35731990786494083,
      8.038565169319383e-128,
      0.1250988,
    ),
    (
      9.943961394319228,
      9.27896184990215e230,
      0.2174693767038595,
      7.972039692260387e-231,
      0.2588324,
    ),
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
/// law's median is ½; the medians at and below ½ are guards.
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

/// No cap NaN: a tiny shape beside one under 0.1 at a level in the mass at 1 (the side from
/// `ln(t/(t + u))`), a huge shape beside one just above 1 at a tiny level (two guards): 1 in steps.
#[test]
fn admissible_laws_never_reach_the_cap_as_nan() {
  for (a, b, p) in [
    (1e-20, 1e-5, 0.9999999999999999),
    (1e300, 1.01, 1e-300),
    (8.22e192, 1.0045, 5e-324),
  ] {
    let (x, steps) = beta_i_inv_steps(a, b, p);
    assert_eq!(x, 1.0, "Beta({a:e}, {b:e}) at {p:e}");
    assert!(steps <= 10, "Beta({a:e}, {b:e}) at {p:e}: {steps} steps");
  }
}

/// The fraction's step `n·x·(b − n)` overflows at shapes near `MAX`; scaled first, the fraction
/// agrees with the same law at 1e300, its shape corrections being `O(1/a)`.
#[test]
fn the_fraction_scales_its_step_where_it_overflows() {
  let huge = large::fraction(f64::MAX, f64::MAX, 0.4, 0.6, 0.2 * f64::MAX);
  let big = large::fraction(1e300, 1e300, 0.4, 0.6, 0.2e300);
  assert!(
    huge.is_finite() && rel(huge, big) < 1e-12,
    "{huge} against {big}"
  );
}

/// The shapes' sums enter `ln Γ` in two parts: rounded, `ψ(s + o)` times their half ulp put the root
/// 6e-15 off its log at shapes near 16; a subnormal root is read at the double's own log.
#[test]
fn shape_sums_near_16_hold_the_bound() {
  let median = beta_i_inv(0.5, 15.9, 0.5);
  assert!(
    rel(median, 0.014427487626813367) <= 9e-15,
    "Beta(0.5, 15.9) median {median}"
  );
  assert_eq!(beta_i_inv(15.9, 1e300, 5e-324), 3.1363e-320);
}

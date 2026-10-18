use stochastic_rs_copulas::bivariate::fgm::Fgm;
use stochastic_rs_copulas::bivariate::gaussian::GaussianCopula;
use stochastic_rs_copulas::bivariate::independence::Independence;
use stochastic_rs_copulas::bivariate::marshall_olkin::MarshallOlkin;
use stochastic_rs_copulas::bivariate::t_copula::TCopula;
use stochastic_rs_copulas::multivariate::dvine::PairCopula;
use stochastic_rs_copulas::traits::BivariateExt;

use super::GRID;
use super::agrees;
use super::build;
use super::on_grid;

/// `<family> <h|q> <parameters> ; <81 values>`, row-major in the first argument: mpmath at 60 or more digits of the
/// stable forms, each proved equal to `∂_v C` in sympy, `q` by bisection of that `h` in `ln u`.
const REFERENCE: &str = include_str!("../fixtures/extreme_parameters.txt");

/// The 69 t cells `(ρ, ν, kind, cells)`, a cell `9·row + column`, where `SimdStudentT::quantile` misses `1e-12`: past
/// `1 − 1e-12` it stops once its cdf rounds to `p`. Each must keep missing, so a fix in the law fails this test.
const T_LAW_MISSES: [(f64, f64, &str, &[usize]); 6] = [
  (-0.8, 2.5, "h", &[33, 34, 42, 43, 57, 58, 61, 66, 67, 69]),
  (
    -0.8,
    2.5,
    "q",
    &[15, 16, 24, 25, 33, 34, 42, 43, 51, 52, 59],
  ),
  (
    0.5,
    4.0,
    "h",
    &[33, 34, 42, 43, 51, 52, 57, 58, 60, 61, 66, 67, 69, 70],
  ),
  (0.5, 4.0, "q", &[15, 16, 24, 25, 33, 34, 42, 43]),
  (0.99, 30.0, "h", &[33, 34, 42, 43, 51, 52, 60, 61, 69, 70]),
  (
    0.99,
    30.0,
    "q",
    &[
      15, 16, 24, 25, 33, 34, 42, 43, 56, 57, 58, 59, 65, 66, 67, 68,
    ],
  ),
];

#[test]
fn h_and_its_inverse_match_the_mpmath_table() {
  let mut failures = Vec::new();
  for line in REFERENCE.lines() {
    let (head, values) = line.split_once(" ; ").expect("a reference row");
    let mut fields = head.split(' ');
    let (family, kind) = (fields.next().unwrap(), fields.next().unwrap());
    let params = fields.map(|p| p.parse().unwrap()).collect::<Vec<f64>>();
    let want = values.split(' ').map(|x| x.parse().unwrap());
    let got = on_grid(build(family, &params).as_ref(), kind);
    for (i, (got, want)) in got.iter().zip(want).enumerate() {
      let (a, b) = (GRID[i / 9], GRID[i % 9]);
      let inside = got.is_finite() && (0.0..=1.0).contains(got);
      let listed = family == "t"
        && T_LAW_MISSES
          .iter()
          .any(|&(rho, nu, k, cells)| params == [rho, nu] && kind == k && cells.contains(&i));
      if !inside || listed == agrees(*got, want) {
        let note = if listed { " (a T_LAW_MISSES cell)" } else { "" };
        failures.push(format!(
          "{family} {params:?} {kind}({a:e} | {b:e}) = {got:e}, want {want:e}{note}"
        ));
      }
    }
  }
  assert!(
    failures.is_empty(),
    "{} cells:\n{}",
    failures.len(),
    failures.join("\n")
  );
}

/// The families without a tabulated row, and the t copula from `ν ≈ 1e7` up to its normal limit, where `SimdStudentT`'s
/// `beta_i` loses accuracy: finite and in `[0, 1]`.
#[test]
fn every_family_answers_inside_the_unit_interval() {
  let mut copulas: Vec<(String, Box<dyn BivariateExt>)> = Vec::new();
  for rho in [-1.0 + 1e-16, -0.5, 0.0, 0.5, 1.0 - 1e-16] {
    copulas.push((
      format!("gaussian {rho}"),
      Box::new(GaussianCopula::new(rho)),
    ));
  }
  for theta in [-1.0, -0.5, 0.0, 0.5, 1.0] {
    copulas.push((format!("fgm {theta}"), Box::new(Fgm::new(theta))));
  }
  for (alpha, beta) in [(1e-300, 1e-300), (0.3, 0.6), (1.0, 1.0), (1.0, 1e-300)] {
    let name = format!("marshall_olkin {alpha} {beta}");
    copulas.push((name, Box::new(MarshallOlkin::new(alpha, beta))));
  }
  copulas.push(("independence".into(), Box::new(Independence::new())));
  for (rho, nu) in [(0.5, 1e10), (-0.5, 1e15), (0.999999, 1e300)] {
    copulas.push((format!("t {rho} {nu}"), Box::new(TCopula::new(rho, nu))));
  }
  for (name, copula) in copulas {
    for kind in ["h", "q"] {
      let out = on_grid(copula.as_ref(), kind);
      assert!(
        out.iter().all(|x| x.is_finite() && (0.0..=1.0).contains(x)),
        "{name} {kind}: {out}"
      );
    }
  }
}

/// The vine pairs against 50-digit values of their clamped arguments: Clayton past `θ|ln v| ≈ 710`, BB1 and BB7 at the
/// ends of their ranges with their inverses saturating at the bracket, and Frank's density at `θ = 0`.
#[test]
fn vine_pairs_hold_at_strong_dependence() {
  let clayton = |theta| PairCopula::Clayton { theta };
  let bb1 = |theta, delta| PairCopula::Bb1 { theta, delta };
  let bb7 = |theta, delta| PairCopula::Bb7 { theta, delta };
  let h_pins = [
    (clayton(103.0), 0.5, 1e-3, 0.999999999999),
    (clayton(200.0), 0.3, 0.29, 0.998859696596924),
    (clayton(1e300), 0.5, 0.5, 0.5),
    (clayton(1e-300), 0.3, 0.6, 0.3),
    (bb1(100.0, 1.0), 0.5, 0.49, 0.8818101704990862),
    (bb1(1.0, 100.0), 0.5, 0.51, 0.017983983458559526),
    (bb1(1e300, 1e300), 0.3, 0.3, 0.5),
    (bb7(100.0, 1.0), 0.99, 0.99, 0.5034777750283594),
    (bb7(100.0, 100.0), 0.999, 0.9989, 0.999928165127942),
    (bb7(1.0, 1e-300), 0.3, 0.6, 0.3),
  ];
  for (pair, u, v, want) in h_pins {
    let got = pair.h(u, v);
    assert!(
      (got - want).abs() <= 1e-12 * want,
      "{pair:?} h({u} | {v}) = {got}, want {want}"
    );
  }
  // Clayton's closed form to 1e-12; the two-parameter pairs' Brent root within its `6 macheps |x| + 2t`, `t = 2.5e-16`.
  let q_pins = [
    (clayton(200.0), 0.5, 0.01, 0.01000034545197381, 1e-12 * 0.01),
    (
      clayton(103.0),
      1e-6,
      1e-3,
      0.0008756039198499794,
      1e-12 * 1e-3,
    ),
    (bb1(0.8, 1.6), 0.3, 0.6, 0.4540659114878789, 2e-15),
    (bb1(100.0, 1.0), 0.5, 0.49, 0.49006749289180557, 2e-15),
    (bb7(1.7, 0.9), 0.3, 0.6, 0.4057668416272789, 2e-15),
    (bb7(100.0, 1.0), 0.5, 0.99, 0.9899986044733178, 2e-15),
  ];
  for (pair, y, v, want, tolerance) in q_pins {
    let got = pair.h_inverse(y, v);
    assert!(
      (got - want).abs() <= tolerance,
      "{pair:?} h⁻¹({y} | {v}) = {got}, want {want}"
    );
  }
  // `h` at the bracket end already passes the level, so the root lies outside `[1e-12, 1 − 1e-12]`.
  let saturating = [
    (bb1(2.0, 1.0), 0.3, 1e-12, 1e-12),
    (bb1(0.8, 1.6), 0.05, 1e-12, 1e-12),
    (bb7(1.7, 0.9), 0.2, 1e-12, 1e-12),
    (bb7(3.0, 1.0), 0.9, 1.0 - 1e-12, 1.0 - 1e-12),
    (bb1(2.0, 1.0), 1.0 - 1e-12, 1.0 - 1e-12, 1.0 - 1e-12),
  ];
  for (pair, y, v, want) in saturating {
    assert_eq!(pair.h_inverse(y, v), want, "{pair:?} h⁻¹({y} | {v})");
  }
  let frank = PairCopula::Frank { theta: 0.0 };
  assert_eq!(frank.log_density(0.3, 0.6), 0.0);
  assert_eq!(frank.h(0.3, 0.6), 0.3);
}

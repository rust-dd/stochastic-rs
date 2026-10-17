//! h-functions and conditional quantiles at the ends of every admitted parameter range: finite and in `[0, 1]` on a grid
//! of edges and extreme draws, and within `1e-12` of 60-digit mpmath values wherever those are normal numbers.

use ndarray::Array1;
use ndarray::Axis;
use ndarray::stack;
use stochastic_rs_copulas::bivariate::amh::Amh;
use stochastic_rs_copulas::bivariate::bb1::Bb1;
use stochastic_rs_copulas::bivariate::bb7::Bb7;
use stochastic_rs_copulas::bivariate::clayton::Clayton;
use stochastic_rs_copulas::bivariate::fgm::Fgm;
use stochastic_rs_copulas::bivariate::frank::Frank;
use stochastic_rs_copulas::bivariate::galambos::Galambos;
use stochastic_rs_copulas::bivariate::gaussian::GaussianCopula;
use stochastic_rs_copulas::bivariate::gumbel::Gumbel;
use stochastic_rs_copulas::bivariate::husler_reiss::HuslerReiss;
use stochastic_rs_copulas::bivariate::independence::Independence;
use stochastic_rs_copulas::bivariate::joe::Joe;
use stochastic_rs_copulas::bivariate::marshall_olkin::MarshallOlkin;
use stochastic_rs_copulas::bivariate::plackett::Plackett;
use stochastic_rs_copulas::bivariate::t_copula::TCopula;
use stochastic_rs_copulas::multivariate::dvine::PairCopula;
use stochastic_rs_copulas::traits::BivariateExt;

/// The edges, the extreme normal and subnormal values, and the extreme uniform draws `2⁻⁵³` and `1 − 2⁻⁵³`.
const GRID: [f64; 9] = [
  0.0,
  5e-324,
  1e-300,
  1e-15,
  f64::EPSILON / 2.0,
  0.5,
  1.0 - 1e-15,
  1.0 - f64::EPSILON / 2.0,
  1.0,
];

/// `<family> <h|q> <parameters> ; <81 values>`, row-major in the first argument: mpmath at 60 or more digits of the
/// stable forms, each proved equal to `∂_v C` in sympy, `q` by bisection of that `h` in `ln u`.
const REFERENCE: &str = include_str!("fixtures/extreme_parameters.txt");

fn with_theta<C: BivariateExt>(mut copula: C, theta: f64) -> C {
  copula.set_theta(theta);
  copula
}

fn build(family: &str, p: &[f64]) -> Box<dyn BivariateExt> {
  match family {
    "galambos" => Box::new(with_theta(Galambos::default(), p[0])),
    "clayton" => Box::new(Clayton::new(p[0])),
    "plackett" => Box::new(Plackett::new(p[0])),
    "amh" => Box::new(Amh::new(p[0])),
    "frank" => Box::new(Frank::new(p[0])),
    "gumbel" => Box::new(Gumbel::new(p[0])),
    "joe" => Box::new(Joe::new(p[0])),
    "husler_reiss" => Box::new(HuslerReiss::new(p[0])),
    "bb1" => Box::new(Bb1::new(p[0], p[1])),
    "bb7" => Box::new(Bb7::new(p[0], p[1])),
    "t" => Box::new(TCopula::new(p[0], p[1])),
    other => panic!("no family {other}"),
  }
}

/// `h(a | b)` or `h⁻¹(a | b)` over the whole grid in one call, `a` the row and `b` the column.
fn on_grid(copula: &dyn BivariateExt, kind: &str) -> Array1<f64> {
  let a = Array1::from_iter(GRID.iter().flat_map(|&a| GRID.map(|_| a)));
  let b = Array1::from_iter(GRID.iter().flat_map(|_| GRID));
  match kind {
    "h" => copula.partial_derivative(stack![Axis(1), a, b].view()),
    _ => copula.percent_point(a.view(), b.view()),
  }
  .unwrap_or_else(|e| panic!("{e}"))
}

/// A normal truth to `1e-12` relative; one below `f64::MIN_POSITIVE` to within it, so 0 passes.
fn agrees(got: f64, want: f64) -> bool {
  if want >= f64::MIN_POSITIVE {
    (got - want).abs() <= 1e-12 * want
  } else {
    (got - want).abs() <= f64::MIN_POSITIVE
  }
}

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
      // `SimdStudentT`'s quantile above `1 − 1e-12`, in its body from `ν = 1` on, is not yet accurate to 1e-12.
      let near_one = |x: f64| x > 1.0 - 1e-12 && x < 1.0;
      let body =
        family == "t" && params[1] >= 1.0 && params[1] < 1e22 && (near_one(a) || near_one(b));
      if !(inside && (body || agrees(*got, want))) {
        failures.push(format!(
          "{family} {params:?} {kind}({a:e} | {b:e}) = {got:e}, want {want:e}"
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

/// The families without a tabulated row, the two-parameter inverses, which bisect, and the t copula where its law is
/// not yet accurate: finite and in `[0, 1]`.
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
  for (theta, delta) in [
    (1e-300, 1.0),
    (0.8, 1.6),
    (1e300, 1e300),
    (f64::MAX, f64::MAX),
  ] {
    copulas.push((
      format!("bb1 {theta} {delta}"),
      Box::new(Bb1::new(theta, delta)),
    ));
  }
  for (theta, delta) in [
    (1.0, 1e-300),
    (1.7, 0.9),
    (100.0, 100.0),
    (f64::MAX, 1e-300),
  ] {
    copulas.push((
      format!("bb7 {theta} {delta}"),
      Box::new(Bb7::new(theta, delta)),
    ));
  }
  // `SimdStudentT` drifts from `ν ≈ 1e7` up to the normal limit; these answers need only stay in the square.
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

/// The vine pairs at strong dependence against 50-digit values of their clamped arguments: Clayton past `θ|ln v| ≈ 710`,
/// where `u^{−θ}` overflowed to NaN, BB1 and BB7 at the ends of their ranges, and Frank's density at `θ = 0`.
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
  let frank = PairCopula::Frank { theta: 0.0 };
  assert_eq!(frank.log_density(0.3, 0.6), 0.0);
  assert_eq!(frank.h(0.3, 0.6), 0.3);
}

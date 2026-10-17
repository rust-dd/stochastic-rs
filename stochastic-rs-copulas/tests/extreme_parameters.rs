//! h-functions, quantiles, Archimedean generators and nested Archimedean copulas at the ends of every admitted parameter
//! range: finite and in range, and within `1e-12` of mpmath values wherever those are normal numbers.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::Axis;
use ndarray::stack;
use stochastic_rs_copulas::bivariate::BivariateFamily;
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
use stochastic_rs_copulas::correlation::kendall_tau;
use stochastic_rs_copulas::multivariate::dvine::PairCopula;
use stochastic_rs_copulas::multivariate::nac::NacNode;
use stochastic_rs_copulas::multivariate::nac::NestedArchimedean;
use stochastic_rs_copulas::traits::BivariateExt;
use stochastic_rs_copulas::traits::MultivariateExt;
use stochastic_rs_core::simd_rng::Deterministic;

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

/// `<family> <parameters> ; <values on GRID>`: each generator in mpmath at 700 digits, `−ln t` at `θ = 0`.
const GENERATORS: &str = include_str!("fixtures/generator_extremes.txt");

/// `<family> <flat|nested> <θ…> | <u> ; <cdf> <pdf> <log pdf>`: sympy's derivative of `C` in mpmath at 400 and 900 digits.
const NAC: &str = include_str!("fixtures/nac_extremes.txt");

fn archimedean(family: &str, p: &[f64]) -> Box<dyn BivariateExt> {
  match family {
    "clayton" => Box::new(Clayton::new(p[0])),
    "frank" => Box::new(Frank::new(p[0])),
    "joe" => Box::new(Joe::new(p[0])),
    "amh" => Box::new(Amh::new(p[0])),
    "gumbel" => Box::new(Gumbel::new(p[0])),
    "bb1" => Box::new(Bb1::new(p[0], p[1])),
    "bb7" => Box::new(Bb7::new(p[0], p[1])),
    "independence" => Box::new(Independence::new()),
    other => panic!("no family {other}"),
  }
}

/// An infinite truth exactly, a finite one as `agrees` holds it.
fn matches(got: f64, want: f64) -> bool {
  if want.is_infinite() {
    got == want
  } else {
    agrees(got, want)
  }
}

#[test]
fn archimedean_generators_match_the_mpmath_table() {
  let grid = Array1::from(GRID.to_vec());
  let mut failures = Vec::new();
  for line in GENERATORS.lines() {
    let (head, values) = line.split_once(" ; ").expect("a reference row");
    let mut fields = head.split(' ');
    let family = fields.next().unwrap();
    let params = fields.map(|p| p.parse().unwrap()).collect::<Vec<f64>>();
    let got = archimedean(family, &params).generator(grid.view()).unwrap();
    let want = values.split(' ').map(|x| x.parse::<f64>().unwrap());
    for ((t, &got), want) in GRID.iter().zip(&got).zip(want) {
      if !(got >= 0.0 && matches(got, want)) {
        failures.push(format!(
          "{family} {params:?} φ({t:e}) = {got:e}, want {want:e}"
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

/// Every generator, the untabulated AMH and Gumbel too, falls from `φ(0) = ∞` to `φ(1) = 0` without a NaN.
#[test]
fn every_generator_falls_from_infinity_to_zero() {
  let mut grid = Array1::from(GRID.to_vec());
  grid.as_slice_mut().unwrap().sort_by(f64::total_cmp);
  let mut copulas: Vec<(&str, Vec<f64>)> = GENERATORS
    .lines()
    .map(|line| {
      let mut fields = line.split(" ; ").next().unwrap().split(' ');
      let family = fields.next().unwrap();
      (family, fields.map(|p| p.parse().unwrap()).collect())
    })
    .collect();
  for theta in [-1.0, -0.5, 0.0, 0.5, 1.0 - 1e-9, 1.0] {
    copulas.push(("amh", vec![theta]));
  }
  for theta in [1.0, 2.0, 100.0, 1e300, f64::MAX] {
    copulas.push(("gumbel", vec![theta]));
  }
  for (family, params) in copulas {
    let phi = archimedean(family, &params).generator(grid.view()).unwrap();
    let falls = phi.windows(2).into_iter().all(|w| w[0] >= w[1]);
    assert!(
      phi[0] == f64::INFINITY && phi[8] == 0.0 && falls,
      "{family} {params:?}: {phi}"
    );
  }
}

fn nac(family: &str, nested: bool, thetas: &[f64]) -> NestedArchimedean {
  let family = match family {
    "clayton" => BivariateFamily::Clayton,
    _ => BivariateFamily::Gumbel,
  };
  let root = if nested {
    NacNode {
      theta: thetas[0],
      leaves: vec![0],
      children: vec![NacNode::leaf_group(thetas[1], vec![1, 2])],
    }
  } else {
    NacNode::leaf_group(thetas[0], vec![0, 1, 2])
  };
  NestedArchimedean::new(family, root, 3).unwrap()
}

/// Past it the density's terms cancel to within `θ` roundings of `ln c`, so the table holds only `cdf` there.
const DENSITY_UP_TO: f64 = 500.0;

#[test]
fn nested_archimedean_cdf_and_density_match_the_mpmath_table() {
  let mut failures = Vec::new();
  for line in NAC.lines() {
    let (head, values) = line.split_once(" ; ").expect("a reference row");
    let (spec, point) = head.split_once(" | ").unwrap();
    let mut fields = spec.split(' ');
    let (family, shape) = (fields.next().unwrap(), fields.next().unwrap());
    let thetas = fields.map(|p| p.parse().unwrap()).collect::<Vec<f64>>();
    let u = point
      .split(' ')
      .map(|x| x.parse().unwrap())
      .collect::<Vec<f64>>();
    let want = values
      .split(' ')
      .map(|x| x.parse().unwrap())
      .collect::<Vec<f64>>();
    let copula = nac(family, shape == "nested", &thetas);
    let x = Array2::from_shape_vec((1, 3), u).unwrap();
    let cdf = copula.cdf(x.view()).unwrap()[0];
    let mut bad = !matches(cdf, want[0]);
    let (mut pdf, mut log_pdf) = (f64::NAN, f64::NAN);
    if thetas.iter().all(|&t| t <= DENSITY_UP_TO) {
      pdf = copula.pdf(x.view()).unwrap()[0];
      log_pdf = copula.log_pdf(x.view()).unwrap()[0];
      let log_off = (log_pdf - want[2]).abs() > 1e-12 * want[2].abs().max(1.0);
      bad |= !matches(pdf, want[1]) || log_off;
    }
    if bad {
      failures.push(format!(
        "{spec} at {point}: cdf {cdf:e} pdf {pdf:e} log {log_pdf:e}, want {want:?}"
      ));
    }
  }
  assert!(
    failures.is_empty(),
    "{} rows:\n{}",
    failures.len(),
    failures.join("\n")
  );
}

/// Draws from the smallest to the largest `θ` each generator admits: inside the square, uniform margins (√n·KS far from
/// any collapse), and one value per row at the comonotone end.
#[test]
fn nested_archimedean_samples_hold_at_every_theta() {
  let n = 2_000;
  let min_clayton = 5.56268464626801e-309;
  let mut cases: Vec<(&str, bool, Vec<f64>)> = Vec::new();
  for theta in [
    min_clayton,
    1e-300,
    1e-17,
    1e-12,
    1.0,
    200.0,
    1e10,
    1e300,
    f64::MAX,
  ] {
    cases.push(("clayton", false, vec![theta]));
  }
  for pair in [
    [1e-4, 2.0],
    [1.0, 200.0],
    [1.0, 500.0],
    [2.0, 1e300],
    [1.0, f64::MAX],
  ] {
    cases.push(("clayton", true, pair.to_vec()));
  }
  for theta in [1.0, 1.0 + 1e-12, 2.0, 100.0, 1e10, 1e300, f64::MAX] {
    cases.push(("gumbel", false, vec![theta]));
  }
  for pair in [[1.0, 200.0], [1.0, 500.0], [2.0, 1e300], [1.0, f64::MAX]] {
    cases.push(("gumbel", true, pair.to_vec()));
  }
  for (family, nested, thetas) in cases {
    let u = nac(family, nested, &thetas)
      .sample(n, &Deterministic::new(1))
      .unwrap();
    let label = format!("{family} {thetas:?}");
    assert!(
      u.iter().all(|x| x.is_finite() && (0.0..=1.0).contains(x)),
      "{label}"
    );
    for column in u.columns() {
      let mut sorted = column.to_vec();
      sorted.sort_by(f64::total_cmp);
      let ks = sorted
        .iter()
        .enumerate()
        .map(|(i, &x)| ((i + 1) as f64 / n as f64 - x).max(x - i as f64 / n as f64))
        .fold(0.0, f64::max);
      assert!(
        (n as f64).sqrt() * ks < 3.0,
        "{label}: √n·KS {}",
        (n as f64).sqrt() * ks
      );
    }
    if !nested && thetas[0] >= 1e300 {
      let one = u
        .rows()
        .into_iter()
        .all(|r| (r[0] - r[1]).abs() <= 1e-12 && (r[0] - r[2]).abs() <= 1e-12);
      assert!(one, "{label}: a row with distinct values");
    }
  }
}

/// τ̂ of every pair against its youngest common ancestor's `θ/(θ + 2)` (Clayton) or `1 − 1/θ` (Gumbel) within 4 SE of
/// Hoeffding's (1948) variance, ζ₁ from the sample on that pair copula's `cdf`.
#[test]
fn strong_nesting_keeps_kendall_tau() {
  let n = 20_000;
  let cases = [
    ("clayton", false, vec![200.0]),
    ("clayton", false, vec![500.0]),
    ("clayton", true, vec![1.0, 200.0]),
    ("clayton", true, vec![1.0, 500.0]),
    ("gumbel", true, vec![1.0, 200.0]),
    ("gumbel", true, vec![1.0, 500.0]),
  ];
  for (family, nested, thetas) in cases {
    for seed in [3, 5] {
      let u = nac(family, nested, &thetas)
        .sample(n, &Deterministic::new(seed))
        .unwrap();
      let tau_hat = kendall_tau(u.view()).unwrap();
      for (i, j) in [(0, 1), (0, 2), (1, 2)] {
        let theta = if nested && i > 0 {
          thetas[1]
        } else {
          thetas[0]
        };
        let (pair, tau): (Box<dyn BivariateExt>, f64) = match family {
          "clayton" => (Box::new(Clayton::new(theta)), theta / (theta + 2.0)),
          _ => (Box::new(Gumbel::new(theta)), 1.0 - 1.0 / theta),
        };
        let x = stack![Axis(1), u.column(i), u.column(j)];
        let c = pair.cdf(x.view()).unwrap();
        let g =
          Array1::from_iter((0..n).map(|k| 1.0 - 2.0 * x[[k, 0]] - 2.0 * x[[k, 1]] + 4.0 * c[k]));
        let zeta1 = g.var(1.0);
        let m = n as f64;
        let se = (2.0 * (2.0 * (m - 2.0) * zeta1 + 1.0 - tau * tau) / (m * (m - 1.0))).sqrt();
        assert!(
          (tau_hat[[i, j]] - tau).abs() < 4.0 * se,
          "{family} {thetas:?} seed {seed} ({i}, {j}): τ̂ {} vs {tau} ± {}",
          tau_hat[[i, j]],
          4.0 * se
        );
      }
    }
  }
}

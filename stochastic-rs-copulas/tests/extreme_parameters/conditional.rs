use ndarray::array;
use stochastic_rs_copulas::bivariate::bb1::Bb1;
use stochastic_rs_copulas::bivariate::bb7::Bb7;
use stochastic_rs_copulas::bivariate::clayton::Clayton;
use stochastic_rs_copulas::bivariate::frank::Frank;
use stochastic_rs_copulas::bivariate::gaussian::GaussianCopula;
use stochastic_rs_copulas::bivariate::t_copula::TCopula;
use stochastic_rs_copulas::multivariate::dvine::PairCopula;
use stochastic_rs_copulas::traits::BivariateExt;

use super::GRID;
use super::agrees;
use super::build;
use super::on_grid;

/// `<family> <h|q> <parameters> ; <81 values>`, row-major in the first argument: mpmath at 60 or more digits of the
/// stable forms, each proved equal to `∂_v C` in sympy, `q` by bisection of that `h` in `ln u` or its closed form.
const REFERENCE: &str = include_str!("../fixtures/extreme_parameters.txt");

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
      if !inside || !agrees(*got, want) {
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

/// `<ρ> <u> <v> ; <h> <q> <ln c>` past the Gaussian h's rounding switch, where `x ∓ y` needs the exact gap: mpmath at
/// 80 digits from the exact binary inputs, `q` reading `u` as a level.
const GAUSSIAN_GAP: &str = include_str!("../fixtures/gaussian_gap_cells.txt");

#[test]
fn the_gaussian_h_its_inverse_and_density_hold_past_the_rounding_switch() {
  for line in GAUSSIAN_GAP.lines() {
    let (head, values) = line.split_once(" ; ").expect("a reference row");
    let p: Vec<f64> = head.split(' ').map(|x| x.parse().unwrap()).collect();
    let want: Vec<f64> = values.split(' ').map(|x| x.parse().unwrap()).collect();
    let copula = GaussianCopula::new(p[0]);
    let h = copula
      .partial_derivative(array![[p[1], p[2]]].view())
      .unwrap()[0];
    let q = copula
      .percent_point(array![p[1]].view(), array![p[2]].view())
      .unwrap()[0];
    let ln_c = copula.log_pdf(array![[p[1], p[2]]].view()).unwrap()[0];
    assert!(
      agrees(h, want[0]) && agrees(q, want[1]),
      "{line}: h {h:e}, q {q:e}"
    );
    assert!(
      (ln_c - want[2]).abs() <= 1e-12 * want[2].abs().max(1.0),
      "{line}: ln c {ln_c:e}"
    );
  }
}

/// The t copula past its switch to the normal law at `ν = 1e22`, at `ρ = 0.999999`: finite and in `[0, 1]`; below the
/// switch the table carries the t rows.
#[test]
fn the_t_copula_past_the_normal_switch_answers_inside_the_unit_interval() {
  let copula = TCopula::new(0.999999, 1e300);
  for kind in ["h", "q"] {
    let out = on_grid(&copula, kind);
    assert!(
      out.iter().all(|x| x.is_finite() && (0.0..=1.0).contains(x)),
      "t 0.999999 1e300 {kind}: {out}"
    );
  }
}

/// The vine pairs are their families' kernels, at strong dependence and across the open grid: `h` the family's
/// `partial_derivative`, an exact 0 or 1 moved one ulp inside, `h⁻¹` its `percent_point`, `log_density` its `log_pdf`.
#[test]
fn vine_pairs_are_their_bivariate_kernels() {
  let cases: Vec<(PairCopula, Box<dyn BivariateExt>)> = vec![
    (
      PairCopula::Gaussian { rho: 0.0 },
      Box::new(GaussianCopula::new(0.0)),
    ),
    (
      PairCopula::Gaussian { rho: 0.6 },
      Box::new(GaussianCopula::new(0.6)),
    ),
    (
      PairCopula::Gaussian {
        rho: -0.9999999999999999,
      },
      Box::new(GaussianCopula::new(-0.9999999999999999)),
    ),
    (
      PairCopula::Clayton { theta: 0.0 },
      Box::new(Clayton::new(0.0)),
    ),
    (
      PairCopula::Clayton { theta: 1e-300 },
      Box::new(Clayton::new(1e-300)),
    ),
    (
      PairCopula::Clayton { theta: 103.0 },
      Box::new(Clayton::new(103.0)),
    ),
    (
      PairCopula::Clayton { theta: 1e300 },
      Box::new(Clayton::new(1e300)),
    ),
    (PairCopula::Frank { theta: 0.0 }, Box::new(Frank::new(0.0))),
    (
      PairCopula::Frank { theta: -40.0 },
      Box::new(Frank::new(-40.0)),
    ),
    (
      PairCopula::StudentT { rho: 0.5, nu: 4.0 },
      Box::new(TCopula::new(0.5, 4.0)),
    ),
    (
      PairCopula::StudentT { rho: -0.8, nu: 0.5 },
      Box::new(TCopula::new(-0.8, 0.5)),
    ),
    (
      PairCopula::Bb1 {
        theta: 100.0,
        delta: 1.0,
      },
      Box::new(Bb1::new(100.0, 1.0)),
    ),
    (
      PairCopula::Bb1 {
        theta: 1.0,
        delta: 100.0,
      },
      Box::new(Bb1::new(1.0, 100.0)),
    ),
    (
      PairCopula::Bb7 {
        theta: 100.0,
        delta: 100.0,
      },
      Box::new(Bb7::new(100.0, 100.0)),
    ),
    (
      PairCopula::Bb7 {
        theta: 1.0,
        delta: 1e-300,
      },
      Box::new(Bb7::new(1.0, 1e-300)),
    ),
  ];
  let inside = |h: f64| match h {
    0.0 => f64::from_bits(1),
    1.0 => 1.0 - f64::EPSILON / 2.0,
    h => h,
  };
  let same = |a: f64, b: f64| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
  let open = &GRID[1..8];
  let points = open.iter().flat_map(|&u| open.iter().map(move |&v| (u, v)));
  let strong = [(0.5, 1e-3), (0.3, 0.29), (0.999, 0.9989), (1e-6, 1e-3)];
  for (pair, family) in &cases {
    for (u, v) in points.clone().chain(strong) {
      let x = array![[u, v]];
      let h = family.partial_derivative(x.view()).unwrap()[0];
      let q = family
        .percent_point(array![u].view(), array![v].view())
        .unwrap()[0];
      let ln_c = family.log_pdf(x.view()).unwrap()[0];
      assert!(same(pair.h(u, v), inside(h)), "{pair:?} h({u:e} | {v:e})");
      assert!(same(pair.h_inverse(u, v), q), "{pair:?} h⁻¹({u:e} | {v:e})");
      assert!(
        same(pair.log_density(u, v), ln_c),
        "{pair:?} ln c({u:e}, {v:e})"
      );
    }
  }
}

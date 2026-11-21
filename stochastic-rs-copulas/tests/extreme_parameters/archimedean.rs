use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_copulas::bivariate::BivariateFamily;
use stochastic_rs_copulas::bivariate::amh::Amh;
use stochastic_rs_copulas::bivariate::bb1::Bb1;
use stochastic_rs_copulas::bivariate::bb7::Bb7;
use stochastic_rs_copulas::bivariate::clayton::Clayton;
use stochastic_rs_copulas::bivariate::frank::Frank;
use stochastic_rs_copulas::bivariate::gumbel::Gumbel;
use stochastic_rs_copulas::bivariate::independence::Independence;
use stochastic_rs_copulas::bivariate::joe::Joe;
use stochastic_rs_copulas::correlation::kendall_tau;
use stochastic_rs_copulas::multivariate::nac::NacNode;
use stochastic_rs_copulas::multivariate::nac::NestedArchimedean;
use stochastic_rs_copulas::traits::BivariateExt;
use stochastic_rs_copulas::traits::MultivariateExt;
use stochastic_rs_core::simd_rng::Deterministic;

use super::GRID;
use super::matches;

/// `<family> <parameters> ; <values on GRID>`: each generator in mpmath at 700 digits, `−ln t` at `θ = 0`.
const GENERATORS: &str = include_str!("../fixtures/generator_extremes.txt");

/// `<family> <flat|nested> <θ…> | <u> ; <cdf> <pdf> <log pdf>`: sympy's derivative of `C` in mpmath at 400 and 900 digits.
const NAC: &str = include_str!("../fixtures/nac_extremes.txt");

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
  // Clayton at `θ = 1e10` past `x = −θ ln t = ln(f64::MAX)`, at `x ≈ 710` and `731.8`: finite though `expm1(x)` is not.
  let t = Array1::from(vec![0.9999999290000026, 0.9999999268194176]);
  let got = Clayton::new(1e10).generator(t.view()).unwrap();
  for ((t, &got), want) in t
    .iter()
    .zip(&got)
    .zip([2.2339938138387886e298, 6.595423602751007e307])
  {
    if !matches(got, want) {
      failures.push(format!("clayton [1e10] φ({t:e}) = {got:e}, want {want:e}"));
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
    let pdf = copula.pdf(x.view()).unwrap()[0];
    let log_pdf = copula.log_pdf(x.view()).unwrap()[0];
    let log_off = (log_pdf - want[2]).abs() > 1e-12 * want[2].abs().max(1.0);
    bad |= !matches(pdf, want[1]) || log_off;
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
    // A Clayton child under the root `θ = 1e-4` costs about `e·10⁴` proposals a row.
    let n = if nested && thetas[0] < 1e-3 { 200 } else { n };
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
/// Hoeffding's (1948) variance, ζ₁ from the sample on the exact pair law: the NAC's `cdf`, third coordinate at 1.
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
    let copula = nac(family, nested, &thetas);
    for seed in [3, 5] {
      let u = copula.sample(n, &Deterministic::new(seed)).unwrap();
      let tau_hat = kendall_tau(u.view()).unwrap();
      for (i, j) in [(0, 1), (0, 2), (1, 2)] {
        let theta = if nested && i > 0 {
          thetas[1]
        } else {
          thetas[0]
        };
        let tau = match family {
          "clayton" => theta / (theta + 2.0),
          _ => 1.0 - 1.0 / theta,
        };
        let mut x = u.clone();
        x.column_mut(3 - i - j).fill(1.0);
        let c = copula.cdf(x.view()).unwrap();
        let g =
          Array1::from_iter((0..n).map(|k| 1.0 - 2.0 * u[[k, i]] - 2.0 * u[[k, j]] + 4.0 * c[k]));
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

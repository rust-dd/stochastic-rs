use std::time::Duration;
use std::time::Instant;

use ndarray::array;
use stochastic_rs_copulas::bivariate::t_copula::TCopula;
use stochastic_rs_copulas::traits::BivariateExt;

use super::GRID;
use super::build;
use super::matches;
use super::on_grid;

/// `<family> <cdf|log_pdf> <parameters> ; <values>`, row-major in the first argument, `cdf` on `GRID` and `log_pdf` on its
/// open part: mpmath of the stable forms, each proved equal to `C` or to sympy's `∂²C/∂u∂v` (`ln` of it) in sympy.
const DENSITIES: &str = include_str!("../fixtures/extreme_densities.txt");

/// The 84 t cells `(ρ, ν, kind, cells)`, a cell `side · row + column`, where `SimdStudentT::quantile` past `1 − 1e-12` is
/// off by up to 12%; with `−t_ν⁻¹(1 − p)` all pass. Each must keep missing, so a fix fails this test.
const T_LAW_MISSES: [(f64, f64, &str, Misses); 5] = [
  (-0.8, 2.5, "cdf", Misses::Only(&COUNTERMONOTONE_CDF_MISSES)),
  (0.5, 4.0, "cdf", Misses::Only(&CDF_MISSES)),
  (-0.8, 2.5, "log_pdf", Misses::Only(&DENSITY_MISSES)),
  (0.5, 4.0, "log_pdf", Misses::Only(&DENSITY_MISSES)),
  (0.99, 30.0, "log_pdf", Misses::Only(&DENSITY_MISSES)),
];

/// The t rows at large `ν` `(ρ, ν, bound in ms)`, which the table holds for their time and the misses below; a bound is
/// four times a debug evaluation at least, CI's build, the 1e13 row's in the band where the integrations reach their cap.
const WALL_CLOCK: [(f64, f64, u64); 4] = [
  (0.5, 1e8, 500),
  (-0.8, 1e13, 2000),
  (-0.8, 1e15, 500),
  (-0.999999, 9.99e21, 500),
];

/// Where `SimdStudentT`, off the t law by `7e-8` at `ν = 1e8` and not the t law's from `ν ≈ 4.4e12`, puts a large-`ν` cell
/// past `1e-12` or at NaN: 368 cells, each of which must keep missing, so a fix of that law fails this test.
const LARGE_NU_MISSES: [(f64, f64, &str, Misses); 12] = [
  (0.5, 1e8, "cdf", Misses::Only(&[30, 31, 39, 40, 50])),
  (0.5, 1e8, "log_pdf", Misses::AllBut(&UPPER_DENSITY)),
  (
    -0.95,
    4437290359301.342,
    "cdf",
    Misses::Only(&[32, 33, 34, 41, 42, 43, 48, 49, 50, 57, 58, 66, 67]),
  ),
  (
    -0.95,
    4437290359301.342,
    "log_pdf",
    Misses::AllBut(&UPPER_OR_SUBNORMAL_DENSITY),
  ),
  (-0.8, 1e13, "cdf", Misses::Only(&LOWER_CDF)),
  (
    -0.8,
    1e13,
    "log_pdf",
    Misses::AllBut(&UPPER_OR_SUBNORMAL_DENSITY),
  ),
  (-0.8, 4e13, "cdf", Misses::Only(&LOWER_CDF)),
  (
    -0.8,
    4e13,
    "log_pdf",
    Misses::AllBut(&[0, 4, 6, 28, 32, 34, 42, 46, 48]),
  ),
  (-0.8, 1e15, "cdf", Misses::Only(&NARROW_CDF)),
  (-0.8, 1e15, "log_pdf", Misses::AllBut(&[32])),
  (-0.999999, 9.99e21, "cdf", Misses::AllBut(&[])),
  (-0.999999, 9.99e21, "log_pdf", Misses::AllBut(&[32])),
];

/// The cells of a row that miss: those listed, or every other one (the cdf's edges, which are exact, never miss).
#[derive(Clone, Copy)]
enum Misses {
  Only(&'static [usize]),
  AllBut(&'static [usize]),
}

impl Misses {
  fn contains(self, cell: usize) -> bool {
    match self {
      Misses::Only(cells) => cells.contains(&cell),
      Misses::AllBut(cells) => !cells.contains(&cell),
    }
  }
}

/// `C(a, b)` with both levels in `[1e-15, 1 − 2⁻⁵³]`.
const NARROW_CDF: [usize; 25] = [
  30, 31, 32, 33, 34, 39, 40, 41, 42, 43, 48, 49, 50, 51, 52, 57, 58, 59, 60, 61, 66, 67, 68, 69,
  70,
];

/// `NARROW_CDF` with a level under ½, and `(½, ½)`.
const LOWER_CDF: [usize; 17] = [
  30, 31, 32, 33, 34, 39, 40, 41, 42, 43, 48, 49, 50, 57, 58, 66, 67,
];

/// `ln c(a, b)` with both levels in `[½, 1 − 2⁻⁵³]`.
const UPPER_DENSITY: [usize; 9] = [32, 33, 34, 39, 40, 41, 46, 47, 48];

/// `UPPER_DENSITY` and the cells with a level at `5e-324` and the other at `5e-324`, `½`, `1 − 1e-15` or `1 − 2⁻⁵³`,
/// which the t quantile's seed, `ndtri` with its subnormal step, holds within `1e-12`.
const UPPER_OR_SUBNORMAL_DENSITY: [usize; 16] =
  [0, 4, 5, 6, 28, 32, 33, 34, 35, 39, 40, 41, 42, 46, 47, 48];

/// `C(a, b)` with one level in `{1e-15, 2⁻⁵³}` and the other in `{1 − 1e-15, 1 − 2⁻⁵³}`.
const CDF_MISSES: [usize; 8] = [33, 34, 42, 43, 57, 58, 66, 67];

/// `CDF_MISSES` at `ρ < 0` where the scores do not cancel: at `(1e-15, 1 − 1e-15)` and `(2⁻⁵³, 1 − 2⁻⁵³)` the cdf reads the
/// exact `x + y` and the lower quantile alone.
const COUNTERMONOTONE_CDF_MISSES: [usize; 4] = [34, 42, 58, 66];

/// `ln c(a, b)` with a level in `{1 − 1e-15, 1 − 2⁻⁵³}`.
const DENSITY_MISSES: [usize; 24] = [
  5, 6, 12, 13, 19, 20, 26, 27, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
];

/// Within `1e-12 · max(1, |ln c|)`, an infinite truth exactly.
fn log_agrees(got: f64, want: f64) -> bool {
  if want.is_infinite() {
    got == want
  } else {
    (got - want).abs() <= 1e-12 * want.abs().max(1.0)
  }
}

#[test]
fn cdf_and_log_density_match_the_mpmath_table() {
  let mut failures = Vec::new();
  for line in DENSITIES.lines() {
    let (head, values) = line.split_once(" ; ").expect("a reference row");
    let mut fields = head.split(' ');
    let (family, kind) = (fields.next().unwrap(), fields.next().unwrap());
    let params = fields.map(|p| p.parse().unwrap()).collect::<Vec<f64>>();
    let want = values.split(' ').map(|x| x.parse::<f64>().unwrap());
    let got = on_grid(build(family, &params).as_ref(), kind);
    for (i, (&got, want)) in got.iter().zip(want).enumerate() {
      // `C(u, 0) = C(0, v) = 0`, `C(u, 1) = u`, `C(1, v) = v`, exactly.
      let edge = kind == "cdf"
        && [GRID[i / 9], GRID[i % 9]]
          .iter()
          .any(|&x| x == 0.0 || x == 1.0);
      let ok = if kind == "cdf" {
        (0.0..=1.0).contains(&got)
          && if edge {
            got == want
          } else {
            matches(got, want)
          }
      } else {
        log_agrees(got, want)
      };
      let listed = family == "t"
        && !edge
        && T_LAW_MISSES
          .iter()
          .chain(&LARGE_NU_MISSES)
          .any(|&(rho, nu, k, cells)| params == [rho, nu] && kind == k && cells.contains(i));
      if ok == listed {
        let side = if kind == "cdf" { 9 } else { 7 };
        let offset = if kind == "cdf" { 0 } else { 1 };
        let (a, b) = (GRID[i / side + offset], GRID[i % side + offset]);
        let note = if listed { " (a listed miss)" } else { "" };
        failures.push(format!(
          "{family} {params:?} {kind}({a:e}, {b:e}) = {got:e}, want {want:e}{note}"
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

/// The wall-clock rows: at large `ν` every evaluation is bounded in time, whatever noise the law it reads carries, with
/// the cells that took a minute or did not return and two of the band that reach the cap where `C` is the Fréchet floor.
#[test]
fn the_t_cdf_is_time_bounded_at_large_nu() {
  let mut cells: Vec<(f64, f64, f64, f64, u64)> = vec![
    (-0.8, 1e8, 0.5, 0.5, 500),
    (0.99, 1e8, 1e-15, 1e-15, 500),
    (-0.999999, 9.999999999999998e21, 1e-300, 0.5, 500),
    (-0.999999, 1e13, 0.5, 0.7, 2000),
    (-0.999999, 1e13, 0.7, 0.5, 2000),
  ];
  for (rho, nu, bound) in WALL_CLOCK {
    cells.extend(
      GRID
        .iter()
        .flat_map(|&a| GRID.iter().map(move |&b| (rho, nu, a, b, bound))),
    );
  }
  let mut slow = Vec::new();
  for (rho, nu, a, b, bound) in cells {
    let copula = TCopula::new(rho, nu);
    let start = Instant::now();
    let _ = copula.cdf(array![[a, b]].view()).unwrap();
    let elapsed = start.elapsed();
    if elapsed > Duration::from_millis(bound) {
      slow.push(format!(
        "({rho}, {nu:e}) cdf({a:e}, {b:e}) took {elapsed:?}"
      ));
    }
  }
  assert!(slow.is_empty(), "{slow:#?}");
}

/// The copula should be continuous across the switch to the Gaussian at `ν = 1e22`, but below it `SimdStudentT`'s cdf
/// and quantile are not the t law's past `ν ≈ 1e13`: each cell must keep missing, so a fix of that law fails this test.
#[test]
fn the_t_cdf_is_not_yet_continuous_at_the_gaussian_switch() {
  let mut held = Vec::new();
  for (rho, a, b) in [
    (0.5, 0.3, 0.7),
    (-0.8, 1e-15, 0.5),
    (0.99, 0.5, 0.5),
    (-0.999999, 0.2, 0.9),
  ] {
    let x = array![[a, b]];
    let below = TCopula::new(rho, 9.99e21).cdf(x.view()).unwrap()[0];
    let above = TCopula::new(rho, 1e22).cdf(x.view()).unwrap()[0];
    if matches(below, above) {
      held.push(format!(
        "ρ = {rho}: C({a:e}, {b:e}) = {below:e} below, {above:e} above"
      ));
    }
  }
  assert!(held.is_empty(), "{held:#?}");
}

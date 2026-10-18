use super::GRID;
use super::build;
use super::matches;
use super::on_grid;

/// `<family> <cdf|log_pdf> <parameters> ; <values>`, row-major in the first argument, `cdf` on `GRID` and `log_pdf` on its
/// open part: mpmath of the stable forms, each proved equal to `C` or to sympy's `∂²C/∂u∂v` (`ln` of it) in sympy.
const DENSITIES: &str = include_str!("../fixtures/extreme_densities.txt");

/// The 88 t cells `(ρ, ν, kind, cells)`, a cell `side · row + column`, where `SimdStudentT::quantile` past `1 − 1e-12` is
/// off by up to 12% (16c's `T_LAW_MISSES`); with `−t_ν⁻¹(1 − p)` all pass. Each must keep missing, so a fix fails this test.
const T_LAW_MISSES: [(f64, f64, &str, &[usize]); 5] = [
  (-0.8, 2.5, "cdf", &CDF_MISSES),
  (0.5, 4.0, "cdf", &CDF_MISSES),
  (-0.8, 2.5, "log_pdf", &DENSITY_MISSES),
  (0.5, 4.0, "log_pdf", &DENSITY_MISSES),
  (0.99, 30.0, "log_pdf", &DENSITY_MISSES),
];

/// `C(a, b)` with one level in `{1e-15, 2⁻⁵³}` and the other in `{1 − 1e-15, 1 − 2⁻⁵³}`.
const CDF_MISSES: [usize; 8] = [33, 34, 42, 43, 57, 58, 66, 67];

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
      let ok = if kind == "cdf" {
        let (a, b) = (GRID[i / 9], GRID[i % 9]);
        // `C(u, 0) = C(0, v) = 0`, `C(u, 1) = u`, `C(1, v) = v`, exactly.
        let edge = [a, b].iter().any(|&x| x == 0.0 || x == 1.0);
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
        && T_LAW_MISSES
          .iter()
          .any(|&(rho, nu, k, cells)| params == [rho, nu] && kind == k && cells.contains(&i));
      if ok == listed {
        let side = if kind == "cdf" { 9 } else { 7 };
        let offset = if kind == "cdf" { 0 } else { 1 };
        let (a, b) = (GRID[i / side + offset], GRID[i % side + offset]);
        let note = if listed { " (a T_LAW_MISSES cell)" } else { "" };
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

use stochastic_rs_copulas::bivariate::galambos::Galambos;
use stochastic_rs_copulas::bivariate::husler_reiss::HuslerReiss;
use stochastic_rs_copulas::bivariate::plackett::Plackett;
use stochastic_rs_copulas::traits::BivariateExt;

use super::agrees;

/// `<family> <θ> ; <τ> <ln |τ|> <ln(1 − |τ|)>`: mpmath at 40 and 60 digits, rounding alike, of Plackett's closed inner
/// integral of Nelsen's Thm 5.1.3 and of Gudendorf & Segers' `∫ t(1 − t)/A dA'` with its form by parts.
const TABLE: &str = include_str!("../fixtures/kendall_tau.txt");

/// Where the `τ` an f64 holds still fixes `θ` to `1e-9`: past these, one ulp of `τ` near 1 moves `θ` by more.
const RECOVERABLE: [(&str, f64, f64); 3] = [
  ("plackett", 1e-12, 1e14),
  ("galambos", 1e-3, 1e7),
  ("husler_reiss", 0.0266, 1e7),
];

fn family(name: &str) -> Box<dyn BivariateExt> {
  match name {
    "plackett" => Box::new(Plackett::default()),
    "galambos" => Box::new(Galambos::default()),
    _ => Box::new(HuslerReiss::default()),
  }
}

fn rows() -> impl Iterator<Item = (&'static str, f64, f64)> {
  TABLE.lines().map(|line| {
    let (head, values) = line.split_once(" ; ").expect("a reference row");
    let (name, theta) = head.split_once(' ').unwrap();
    let tau = values.split(' ').next().unwrap();
    (name, theta.parse().unwrap(), tau.parse().unwrap())
  })
}

/// `τ` within `1e-12`, and within `1e-12` of itself below `|τ| = 1e-3`, Plackett's either side of `θ = 1` included: the
/// families' whole ranges.
#[test]
fn kendall_tau_matches_the_mpmath_table() {
  let mut failures = Vec::new();
  for (name, theta, want) in rows() {
    let got = family(name).tau_from_theta(theta);
    let close = if want.abs() < 1e-3 {
      agrees(got.abs(), want.abs()) && (got == want || got.signum() == want.signum())
    } else {
      (got - want).abs() <= 1e-12
    };
    if !close {
      failures.push(format!("{name} τ({theta:e}) = {got:e}, want {want:e}"));
    }
  }
  assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn theta_from_tau_recovers_theta_from_the_exact_tau() {
  let mut failures = Vec::new();
  for (name, theta, tau) in rows() {
    let (_, lo, hi) = RECOVERABLE.iter().find(|r| r.0 == name).unwrap();
    if !(*lo <= theta && theta <= *hi) {
      continue;
    }
    match family(name).theta_from_tau(tau) {
      Ok(back) if (back / theta - 1.0).abs() <= 1e-9 => {}
      other => failures.push(format!("{name} θ = {theta:e} from τ = {tau:e}: {other:?}")),
    }
  }
  assert!(failures.is_empty(), "{failures:#?}");
}

/// The Galambos `τ` falls as `(π/4)2^{−1/θ}` toward independence, never negative and 0 under the smallest subnormal.
/// `τ = 1e-300` inverts to mpmath's `θ`, and `τ` there to mpmath's: an ulp of `θ` moves that `τ` by `690` ulps.
#[test]
fn galambos_tau_stays_non_negative_and_relative_toward_independence() {
  let c = Galambos::default();
  let theta = c.theta_from_tau(1e-300).unwrap();
  assert!(
    (theta / 1.003_784_683_550_363e-3 - 1.0).abs() <= 1e-14,
    "θ = {theta:e}"
  );
  assert!(agrees(
    c.tau_from_theta(theta),
    1.000_000_000_001_574_8e-300
  ));
  for theta in [0.0, 1e-300, 1e-5, 1e-4, 9e-4, 9.4e-4, 1e-3] {
    assert!(c.tau_from_theta(theta) >= 0.0, "θ = {theta:e}");
  }
  assert_eq!(c.tau_from_theta(9e-4), 0.0);
}

/// Where τ is tiny its relative accuracy is the rounding of `ln τ`: Galambos at θ = 1e-3, `ln τ = −693.39` to 2 ulps, is
/// `2.7e-13` off mpmath's `7.328123094736333e-302`, the largest relative error while τ is normal.
#[test]
fn a_tiny_tau_holds_its_logarithm_floor() {
  let got = Galambos::default().tau_from_theta(1e-3);
  let want = 7.328123094736333e-302;
  assert!(
    (got / want - 1.0).abs() <= 3e-13,
    "τ(1e-3) = {got:e}, want {want:e}"
  );
}

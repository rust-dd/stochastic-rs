use super::super::RoughHestonParams;
use super::super::scalar::Jet4;
use super::super::scalar::RiccatiScalar;

const GOOD: [f64; 6] = [0.04, 1.0, 0.04, 0.4, -0.65, 0.1];

fn with(index: usize, value: f64) -> RoughHestonParams {
  let mut p = GOOD;
  p[index] = value;
  RoughHestonParams {
    v0: p[0],
    kappa: p[1],
    theta: p[2],
    nu: p[3],
    rho: p[4],
    hurst: p[5],
  }
}

#[test]
fn each_field_off_its_domain_is_named_in_the_crate_s_form() {
  let cases = [
    (0, -0.01, "v0 must satisfy"),
    (0, f64::NAN, "v0 must satisfy"),
    (1, 0.0, "kappa must satisfy"),
    (1, f64::INFINITY, "kappa must satisfy"),
    (2, -1.0, "theta must satisfy"),
    (3, 0.0, "nu must satisfy"),
    (4, 1.5, "rho must satisfy"),
    (4, f64::NAN, "rho must satisfy"),
    (5, 0.0, "hurst must satisfy"),
    (5, 0.5, "hurst must satisfy"),
    (5, f64::NAN, "hurst must satisfy"),
  ];
  for (index, value, words) in cases {
    let reason = with(index, value).violation();
    assert!(
      reason.as_deref().is_some_and(|r| r.starts_with(words)),
      "field {index} at {value}: {reason:?}"
    );
  }
  assert_eq!(with(0, 0.04).violation(), None);
}

#[test]
#[should_panic(expected = "kappa must satisfy `kappa.is_finite() && kappa > 0.0`, got kappa = 0.0")]
fn new_panics_with_the_violation() {
  RoughHestonParams::new(0.04, 0.0, 0.04, 0.4, -0.65, 0.1);
}

fn close(got: Jet4, want: [f64; 5]) {
  for n in 0..5 {
    let scale = want[n].abs().max(1.0);
    assert!(
      (got.0[n] - want[n]).abs() <= 4.0 * f64::EPSILON * scale,
      "order {n}: {:?} vs {want:?}",
      got.0
    );
  }
}

/// The binomial series of `(1+s)^{1/2}`, `1/(1−s)` and `(1+s)²/(1−s)`, exact in f64.
#[test]
fn the_jet_multiplies_divides_and_roots_as_taylor_series_do() {
  let one = Jet4::real(1.0);
  let s = Jet4::VARIABLE;
  close((one + s).sqrt(), [1.0, 0.5, -0.125, 0.0625, -0.0390625]);
  close(one / (one - s), [1.0; 5]);
  close((one + s) * (one + s) / (one - s), [1.0, 3.0, 4.0, 4.0, 4.0]);
  close(
    (one + s).sqrt() * (one + s).sqrt(),
    [1.0, 1.0, 0.0, 0.0, 0.0],
  );
  close(-(s * 2.0) - one, [-1.0, -2.0, 0.0, 0.0, 0.0]);
  assert_eq!(((one + s) * 3.0).re(), 3.0);
  assert_eq!((-(one + s) * 3.0).norm(), 3.0);
  assert_eq!(Jet4::ZERO, Jet4::real(0.0));
}

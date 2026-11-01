//! The normal's mgf summed in one frame scaled below the doubles' range, and its plain path, against mpmath at 60 and
//! 120 digits.

use stochastic_rs_distributions::traits::DistributionExt;
use stochastic_rs_distributions::truncated::SimdTruncatedNormal;

use crate::close;

fn mgf(mu: f64, sigma: f64, lo: f64, up: f64, t: f64) -> f64 {
  SimdTruncatedNormal::<f64>::new(mu, sigma, lo, up)
    .moment_generating_function(t)
    .unwrap()
}

/// Past the doubles the limit comes from the whole exponent: the plain mass's `α²/2` outweighs `t(μ + σ²t/2) < 0` in
/// the first law, and two bounds an ulp apart either side of the balance read 0 and `+∞`.
#[test]
fn past_the_doubles_the_limit_is_the_whole_sums() {
  let inf = f64::INFINITY;
  for (mu, sigma, lo, up, t, want) in [
    (-1.45e308, 1.5, -2.5e307, inf, 1.2e308, inf),
    (1.45e308, 1.5, -inf, 2.5e307, -1.2e308, inf),
    (-1.5e308, 1.2, -2.336730280057996e307, inf, 1.6e308, 0.0),
    (-1.5e308, 1.2, -2.3367302800579957e307, inf, 1.6e308, inf),
  ] {
    assert_eq!(
      mgf(mu, sigma, lo, up, t),
      want,
      "TN({mu:e}, {sigma}, [{lo:e}, {up:e}]).mgf({t:e})"
    );
  }
}

/// `σ²t/2 = −μ` exactly: the exponent's two terms past the doubles cancel to 0, and the mgf is the masses' ratio 2;
/// `μ` an ulp either side leaves `±t·ulp(μ)`, `+∞` and 0.
#[test]
fn an_exactly_cancelling_exponent_keeps_its_finite_value() {
  let (mu, t) = (-1.2640029854500659e308, 1.1235582092889474e308);
  let inf = f64::INFINITY;
  assert_eq!(mgf(mu, 1.5, mu, inf, t), 2.0);
  assert_eq!(mgf(-mu, 1.5, -inf, -mu, -t), 2.0);
  let (above, below) = (mu.next_up(), mu.next_down());
  assert_eq!(mgf(above, 1.5, above, inf, t), inf);
  assert_eq!(mgf(below, 1.5, below, inf, t), 0.0);
}

/// Either side of `B = 8`, the plain path and the exact one; and an exponent either side of `ln MAX`.
#[test]
fn the_plain_path_and_its_neighbours() {
  for (name, got, want) in [
    (
      "TN(0, 1, [−0.5, 0.5]).mgf(8)",
      mgf(0.0, 1.0, -0.5, 0.5, 8.0),
      6.5779839354971035,
    ),
    (
      "TN(0, 1, [−0.5, 0.5]).mgf(8⁺)",
      mgf(0.0, 1.0, -0.5, 0.5, 8.000000000000002),
      6.577983935497108,
    ),
    (
      "TN(0, 1, [−8, 1]).mgf(0.5)",
      mgf(0.0, 1.0, -8.0, 1.0, 0.5),
      0.9312824760690857,
    ),
    (
      "TN(0, 1, [−8⁻, 1]).mgf(0.5)",
      mgf(0.0, 1.0, -8.000000000000002, 1.0, 0.5),
      0.9312824760690857,
    ),
    (
      "TN(−1124.290911, 1.5, [μ, ∞)).mgf(1000)",
      mgf(-1124.290911, 1.5, -1124.290911, f64::INFINITY, 1000.0),
      1.7966764443151313e308,
    ),
  ] {
    close(name, "mgf", got, want, 1e-12, 0.0);
  }
  assert_eq!(
    mgf(-1124.2909102, 1.5, -1124.2909102, f64::INFINITY, 1000.0),
    f64::INFINITY
  );
}

//! Laws narrower than any offset their kernel's coordinate holds, answered in their own scale, and near-two-point betas
//! whose entropy sums pass 1e154 (mpmath, 60 and 120 digits).

use stochastic_rs_distributions::traits::DistributionExt;
use stochastic_rs_distributions::truncated::SimdTruncatedBeta;
use stochastic_rs_distributions::truncated::SimdTruncatedGamma;
use stochastic_rs_distributions::truncated::SimdTruncatedNormal;

use super::close;

fn tn(mu: f64, sigma: f64, lo: f64, up: f64) -> SimdTruncatedNormal<f64> {
  SimdTruncatedNormal::<f64>::new(mu, sigma, lo, up)
}

fn tb(a: f64, b: f64, lo: f64, up: f64) -> SimdTruncatedBeta<f64> {
  SimdTruncatedBeta::<f64>::new(a, b, lo, up)
}

/// Skewness, kurtosis and entropy at 1e-12, kurtosis 1e-11, a skewness of 0 to 1e-14 absolute.
fn shape(name: &str, law: &impl DistributionExt, [skewness, kurtosis, entropy]: [f64; 3]) {
  close(
    name,
    "skewness",
    law.skewness().unwrap(),
    skewness,
    1e-12,
    1e-14,
  );
  close(
    name,
    "kurtosis",
    law.kurtosis().unwrap(),
    kurtosis,
    1e-11,
    0.0,
  );
  close(name, "entropy", law.entropy().unwrap(), entropy, 1e-12, 0.0);
}

/// A truncated exponential from the bound its kernel rises to, or a uniform where the interval is the shorter scale.
#[test]
fn narrow_laws_answer_in_their_own_scale() {
  shape(
    "TN(0, 1e-300, [1e10, 1e20])",
    &tn(0.0, 1e-300, 1e10, 1e20),
    [2.0, 6.0, -1403.576906726368],
  );
  shape(
    "TG(1.708e279, 2.404e115, [0, 2.651e48])",
    &SimdTruncatedGamma::<f64>::new(
      1.7080253438970559e279,
      2.404241670490496e115,
      0.0,
      2.651098534540455e48,
    ),
    [-2.0, 6.0, -530.4575203197925],
  );
  shape(
    "TB(6.637e279, 5.03e-14, [1.14e-114, 1.10e-81])",
    &tb(
      6.636851590359198e279,
      5.025022593506491e-14,
      1.1425672287208008e-114,
      1.102588226096081e-81,
    ),
    [-2.0, 6.0, -829.7256108220108],
  );
  let sliver = tn(0.0, 1.0, -5e-324, 0.0);
  shape(
    "TN(0, 1, [-5e-324, 0])",
    &sliver,
    [0.0, -1.2, -744.4400719213812],
  );
  for t in [1.0, 1e300, -1e300] {
    assert_eq!(
      sliver.moment_generating_function(t),
      Some(1.0),
      "mgf({t:e})"
    );
  }
  let flat = tn(1e300, 1e300, -1.0, 1.0);
  let name = "TN(1e300, 1e300, [-1, 1])";
  close(
    name,
    "variance",
    flat.variance().unwrap(),
    1.0 / 3.0,
    1e-12,
    0.0,
  );
  close(name, "cdf(0)", flat.cdf(0.0).unwrap(), 0.5, 1e-12, 0.0);
  close(name, "pdf(0)", flat.pdf(0.0).unwrap(), 0.5, 1e-12, 0.0);
  let mgf = flat.moment_generating_function(1.0).unwrap();
  close(name, "mgf(1)", mgf, 1.1752011936438014, 1e-12, 0.0);
  assert_eq!(
    tn(0.0, 5e-324, 0.0, f64::INFINITY).pdf(f64::INFINITY),
    Some(0.0)
  );
}

/// A reference point the mode holds exactly: offsets beside a mean of 6.9e-148 resolve an sd of 6.2e-177.
#[test]
fn a_mode_held_to_a_double_double() {
  shape(
    "TB(1.26e58, 1.83e205, [0, 1])",
    &tb(1.261770702601669e58, 1.825920247663797e205, 0.0, 1.0),
    [
      1.7804909726184105e-29,
      4.7552221553634796e-58,
      -404.32186391201253,
    ],
  );
}

/// A minority mass of 3e-162 against a majority beside 1 whose entropy's `g·ln g` passes 1e328 unscaled.
#[test]
fn near_two_point_betas() {
  let a = 3.637648539358857e-25;
  let b = 4.894449240835594e-165;
  for (lo, kurtosis) in [
    (2.7664393147868232e-275, 3.232659835204403e161),
    (1e-100, 8.879514297818518e161),
  ] {
    let law = tb(a, b, lo, 1.0);
    let name = format!("TB(3.6e-25, 4.9e-165, [{lo:e}, 1])");
    close(
      &name,
      "kurtosis",
      law.kurtosis().unwrap(),
      kurtosis,
      1e-11,
      0.0,
    );
    let entropy = law.entropy().unwrap();
    close(
      &name,
      "entropy",
      entropy,
      -2.0431308014326803e164,
      1e-12,
      0.0,
    );
  }
  let law = tb(1e-10, 1e-250, 1e-300, 1.0);
  let name = "TB(1e-10, 1e-250, [1e-300, 1])";
  close(
    name,
    "skewness",
    law.skewness().unwrap(),
    -3.804794399372508e123,
    1e-12,
    0.0,
  );
  close(
    name,
    "kurtosis",
    law.kurtosis().unwrap(),
    1.4479955748088292e247,
    1e-11,
    0.0,
  );
  close(name, "entropy", law.entropy().unwrap(), -1e250, 1e-12, 0.0);
}

/// A mode within a subnormal of a kernel zero is held at the least offset off it, with the slope there: at 1 the
/// exponential of scale `1/f64::MAX` (closed forms, mpmath at 400 and 600 digits), and the far pdf and cdf 0 (M-3).
#[test]
fn modes_within_a_subnormal_of_a_kernel_zero() {
  let near = 1.0 + f64::EPSILON;
  let up = tb(f64::MAX, near, 0.0, 1.0);
  let down = tb(near, f64::MAX, 0.0, 1.0);
  assert_eq!(
    [
      up.pdf(0.0),
      up.pdf(0.5),
      up.cdf(0.5),
      down.pdf(1.0),
      down.pdf(0.5)
    ],
    [Some(0.0); 5]
  );
  assert_eq!((up.mean(), up.median()), (Some(1.0), Some(1.0)));
  for (name, law, mean, median, sign) in [
    ("TB(MAX, 1 + 2⁻⁵²)", &up, 1.0, 1.0, -1.0),
    (
      "TB(1 + 2⁻⁵², MAX)",
      &down,
      5.562684646268003e-309,
      3.855759178904764e-309,
      1.0,
    ),
  ] {
    close(name, "mean", law.mean().unwrap(), mean, 1e-12, 0.0);
    close(name, "median", law.median().unwrap(), median, 1e-12, 0.0);
    shape(
      name,
      law,
      [
        sign * 1.9999999999999998,
        5.999999999999999,
        -708.782712893384,
      ],
    );
  }
}

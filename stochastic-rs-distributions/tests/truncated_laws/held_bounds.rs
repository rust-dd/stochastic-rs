//! The normal's bounds at their own doubles and their width, and answers mapped back from the exact point nearest them,
//! against mpmath at 60 and 120 digits.

use stochastic_rs_distributions::traits::DistributionExt;
use stochastic_rs_distributions::truncated::SimdTruncatedNormal;

use crate::close;

fn tn(mu: f64, sigma: f64, lo: f64, up: f64) -> SimdTruncatedNormal<f64> {
  SimdTruncatedNormal::<f64>::new(mu, sigma, lo, up)
}

/// `(what, value, mpmath)` rows of one law, each to 1e-12; shape numbers to 1e-12 absolute below a size of 1.
fn rows(name: &str, cells: &[(&str, f64, f64)]) {
  for &(what, got, want) in cells {
    let atol = if matches!(what, "skewness" | "kurtosis") {
      1e-12
    } else {
      0.0
    };
    close(name, what, got, want, 1e-12, atol);
  }
}

/// A narrow interval far from `μ` keeps the width `(up − lo)/σ`, which the bounds' own roundings change by an ulp of
/// their place: 46 % of the variance one ulp wide at 1e8 sd, 6 % of the cdf eight ulps wide at 1e7 sd.
#[test]
fn a_narrow_interval_far_from_mu_keeps_its_width() {
  let one = tn(0.3, 0.7, 7e7, 70000000.00000001);
  rows(
    "TN(0.3, 0.7, [7e7, 7e7⁺])",
    &[
      ("variance", one.variance().unwrap(), 1.4960971259498222e-17),
      ("skewness", one.skewness().unwrap(), 0.7218274499042872),
      ("kurtosis", one.kurtosis().unwrap(), -0.4651533068535721),
      (
        "pdf",
        one.pdf(70000000.00000001).unwrap(),
        19293943.19348617,
      ),
    ],
  );
  let eight = tn(-2.5, 3.0, 29999997.5, 29999997.50000003);
  rows(
    "TN(−2.5, 3, [29999997.5, +8 ulps])",
    &[
      ("variance", eight.variance().unwrap(), 7.397836126743365e-17),
      (
        "cdf",
        eight.cdf(29999997.500000007).unwrap(),
        0.2593888690054307,
      ),
      (
        "cdf",
        eight.cdf(29999997.500000022).unwrap(),
        0.7592347109813724,
      ),
    ],
  );
}

/// A mean far from `μ` is its bound plus the engine's offset, not `μ + σz`, which cancels to `ulp(μ)`; the density
/// there reads the point's offset from that bound, its far end open or not.
#[test]
fn a_mean_far_from_mu_maps_back_from_its_bound() {
  let far = tn(-1000.0, 1.1, 0.0, f64::INFINITY);
  rows(
    "TN(−1000, 1.1, [0, ∞))",
    &[
      ("mean", far.mean().unwrap(), 0.0012099970718177156),
      (
        "pdf",
        far.pdf(0.00036300000000000004).unwrap(),
        612.247170852889,
      ),
    ],
  );
  rows(
    "TN(300, 1, (−∞, 0])",
    &[(
      "mean",
      tn(300.0, 1.0, f64::NEG_INFINITY, 0.0).mean().unwrap(),
      -0.0033332592633741475,
    )],
  );
  rows(
    "TN(1e10, 3, [−1, 1])",
    &[(
      "mean",
      tn(1e10, 3.0, -1.0, 1.0).mean().unwrap(),
      0.9999999991,
    )],
  );
}

/// A flat law's mean and middle quantiles are offsets from its bounds' midpoint, the odd part through `sinh` of its
/// slope: about 0 they keep the digits a sum of the kernel's values loses below ε of the width.
#[test]
fn a_flat_law_keeps_its_offset_from_the_midpoint() {
  let far = tn(1e300, 1e300, -1.0, 1.0);
  rows(
    "TN(1e300, 1e300, [−1, 1])",
    &[
      ("mean", far.mean().unwrap(), 3.333333333333333e-301),
      ("median", far.median().unwrap(), 5e-301),
    ],
  );
  let near = tn(1e10, 1e10, -1.0, 1.0);
  rows(
    "TN(1e10, 1e10, [−1, 1])",
    &[
      ("mean", near.mean().unwrap(), 3.3333333333333335e-11),
      ("median", near.median().unwrap(), 5e-11),
      ("variance", near.variance().unwrap(), 0.3333333333333333),
      ("quantile", near.quantile(0.25).unwrap(), -0.4999999999625),
      ("quantile", near.quantile(0.75).unwrap(), 0.5000000000375),
    ],
  );
}

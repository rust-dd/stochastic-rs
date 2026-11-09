//! The log-kernel's slope at the reference point is the kernel's own there, at any reference point
//! and past 2⁵³, against mpmath at 60 and 120 digits.

use stochastic_rs_distributions::traits::DistributionExt;
use stochastic_rs_distributions::truncated::SimdTruncatedBeta;
use stochastic_rs_distributions::truncated::SimdTruncatedGamma;

use crate::close;

/// A bound 1e90 sd past the mode, below ½: an exponential from it of scale `1/|ψ|` (curvature's
/// share 5e-181), whose variance 7.8e-406 underflows; the other law mirrors it at its upper bound.
#[test]
fn a_bound_far_past_a_huge_shapes_mode_reads_its_exponential() {
  let laws: [(&str, [f64; 4], [f64; 6]); 2] = [
    (
      "TB(2.38e214, 3.91e219, [6.1e-6, 1])",
      [
        2.384772331224501e214,
        3.909307525682232e219,
        6.1002051334089114e-6,
        1.0,
      ],
      [
        6.1002051334089114e-6,
        6.1002051334089114e-6,
        0.0,
        2.0,
        6.0,
        -465.3953278032321,
      ],
    ),
    (
      "TB(3.83e151, 1.34e154, [4.6e-4, 2.9e-3])",
      [
        3.834376333253533e151,
        1.3359150907678206e154,
        4.6244185712282477e-4,
        0.0028620099708270737,
      ],
      [
        0.0028620099708270737,
        0.0028620099708270737,
        3.159919265744977e-274,
        -2.0,
        6.0,
        -313.87888450095545,
      ],
    ),
  ];
  for (name, [a, b, lo, up], want) in laws {
    let law = SimdTruncatedBeta::<f64>::new(a, b, lo, up);
    let got = [
      law.mean().unwrap(),
      law.median().unwrap(),
      law.variance().unwrap(),
      law.skewness().unwrap(),
      law.kurtosis().unwrap(),
      law.entropy().unwrap(),
    ];
    for ((what, got), want) in [
      "mean", "median", "variance", "skewness", "kurtosis", "entropy",
    ]
    .into_iter()
    .zip(got)
    .zip(want)
    {
      close(name, what, got, want, 1e-12, 0.0);
    }
  }
}

/// `TB(1e20, 1e21)` with its reference point at `lo`, below ½ and half an sd past the mode, and
/// with its mode, past 2⁵³ in both shapes, inside the interval.
#[test]
fn a_beta_slope_below_one_half_and_past_2_to_the_53() {
  let laws: [(&str, f64, [f64; 5]); 2] = [
    (
      "TB(1e20, 1e21, [m − sd, 1])",
      0.09090909090042307,
      [
        0.09090909090475699,
        0.17814616027995817,
        48276783672.10854,
        4.730926258548098e-23,
        0.5918227546646436,
      ],
    ),
    (
      "TB(1e20, 1e21, [m + sd/2, 1])",
      0.09090909091342483,
      [
        0.09090909091775876,
        0.485783178360724,
        90478169559.03667,
        2.0171325049991717e-23,
        1.1688591486269493,
      ],
    ),
  ];
  for (name, lo, [x, cdf, pdf, variance, skewness]) in laws {
    let law = SimdTruncatedBeta::<f64>::new(1e20, 1e21, lo, 1.0);
    close(name, "cdf", law.cdf(x).unwrap(), cdf, 1e-12, 0.0);
    close(name, "pdf", law.pdf(x).unwrap(), pdf, 1e-12, 0.0);
    close(
      name,
      "variance",
      law.variance().unwrap(),
      variance,
      1e-12,
      0.0,
    );
    close(
      name,
      "skewness",
      law.skewness().unwrap(),
      skewness,
      1e-12,
      0.0,
    );
  }
}

/// `TG(k, 1, [m + sd/4, m + 3sd/4])`, flat, so its reference point is the midpoint held beside
/// `lo`: the slope there with `k − 1` exact.
#[test]
fn a_gamma_slope_at_its_exact_midpoint() {
  let laws: [(f64, [f64; 5]); 3] = [
    (
      1e16,
      [
        1.0000000025e16,
        1.0000000075e16,
        1.000000003e16,
        0.10995305978742326,
        2.1835223401610433e-8,
      ],
    ),
    (
      1e24,
      [
        1.00000000000025e24,
        1.00000000000075e24,
        1.0000000000003e24,
        0.10980651598610687,
        2.1837352716007375e-12,
      ],
    ),
    (
      1e26,
      [
        1.0000000000000251e26,
        1.0000000000000751e26,
        1.0000000000000301e26,
        0.10959543047106644,
        2.184229414377529e-13,
      ],
    ),
  ];
  for (k, [lo, up, x, cdf, pdf]) in laws {
    let law = SimdTruncatedGamma::<f64>::new(k, 1.0, lo, up);
    let name = format!("TG({k:e}, 1, [m + sd/4, m + 3sd/4])");
    close(&name, "cdf", law.cdf(x).unwrap(), cdf, 1e-12, 0.0);
    close(&name, "pdf", law.pdf(x).unwrap(), pdf, 1e-12, 0.0);
  }
}

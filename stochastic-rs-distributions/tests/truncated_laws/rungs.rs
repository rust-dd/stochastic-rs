//! Moments of a shape below 1 with a bound between its singular end's rungs, against mpmath at 60
//! and 120 digits (closed forms in the upper incomplete gamma for gammas, tanh-sinh for betas).

use stochastic_rs_distributions::traits::DistributionExt;
use stochastic_rs_distributions::truncated::SimdTruncatedBeta;
use stochastic_rs_distributions::truncated::SimdTruncatedGamma;

use crate::close;

/// Mean, variance, skewness, excess kurtosis and entropy, each to 1e-12.
fn five(name: &str, law: &impl DistributionExt, want: [f64; 5]) {
  let got = [
    law.mean().unwrap(),
    law.variance().unwrap(),
    law.skewness().unwrap(),
    law.kurtosis().unwrap(),
    law.entropy().unwrap(),
  ];
  for ((what, got), want) in ["mean", "variance", "skewness", "kurtosis", "entropy"]
    .into_iter()
    .zip(got)
    .zip(want)
  {
    close(name, what, got, want, 1e-12, 0.0);
  }
}

/// The singular panel from the bound to the next rung, where a frame per power would put its branch
/// point at `(lo/rung)^{e+i}`; the exponential tail's fourth moment needs sums past the first clip.
#[test]
fn a_gamma_bound_between_two_rungs() {
  let laws: [(&str, [f64; 4], [f64; 5]); 3] = [
    (
      "TG(0.5, 1, [0.4, ∞))",
      [0.5, 1.0, 0.4, f64::INFINITY],
      [
        1.144546229004866,
        0.6646517647799862,
        2.3219952194821802,
        8.272578728634162,
        0.6998607475746312,
      ],
    ),
    (
      "TG(0.05, 1, [0.05, ∞))",
      [0.05, 1.0, 0.05, f64::INFINITY],
      [
        0.40717517677873166,
        0.27960106987181343,
        3.4150632100106986,
        18.229224095245783,
        -0.13797042849959146,
      ],
    ),
    (
      "TG(0.387, 0.00187, [7.69e-4, ∞))",
      [
        0.3868861891419194,
        0.0018654183680914776,
        0.0007687028896898014,
        f64::INFINITY,
      ],
      [
        0.00207252304052343,
        2.104898249888924e-6,
        2.3924009344238995,
        8.83511581442123,
        -5.649438790857098,
      ],
    ),
  ];
  for (name, [k, theta, lo, up], want) in laws {
    five(
      name,
      &SimdTruncatedGamma::<f64>::new(k, theta, lo, up),
      want,
    );
  }
}

/// The beta's singular end at 0, and at 1 through its own branch point.
#[test]
fn a_beta_bound_between_two_rungs_at_either_end() {
  let laws: [(&str, [f64; 4], [f64; 5]); 2] = [
    (
      "TB(0.5, 3, [0.01, 1])",
      [0.5, 3.0, 0.01, 1.0],
      [
        0.17479603263477844,
        0.02796010477389367,
        1.4109492952985025,
        1.6704461140230438,
        -0.827136061307852,
      ],
    ),
    (
      "TB(3, 0.05, [0, 0.99])",
      [3.0, 0.05, 0.0, 0.99],
      [
        0.8904131932523697,
        0.017103952698169764,
        -2.1837039464026335,
        5.350789636164313,
        -1.4082829682668145,
      ],
    ),
  ];
  for (name, [a, b, lo, up], want) in laws {
    five(name, &SimdTruncatedBeta::<f64>::new(a, b, lo, up), want);
  }
}

//! Every `DistributionExt` cell that had a closed form before the `Option` port still answers `Some`,
//! and no cell gained one by accident: 32 laws by 12 methods (`Gbm`'s row is pinned in its own tests).

use num_complex::Complex64;
use stochastic_rs_distributions::DistributionExt;
use stochastic_rs_distributions::alpha_stable::SimdAlphaStable;
use stochastic_rs_distributions::beta::SimdBeta;
use stochastic_rs_distributions::binomial::SimdBinomial;
use stochastic_rs_distributions::cauchy::SimdCauchy;
use stochastic_rs_distributions::chi_square::SimdChiSquared;
use stochastic_rs_distributions::exp::SimdExp;
use stochastic_rs_distributions::gamma::SimdGamma;
use stochastic_rs_distributions::ged::SimdGed;
use stochastic_rs_distributions::generalized_hyperbolic::SimdGeneralizedHyperbolic;
use stochastic_rs_distributions::generalized_inverse_gauss::SimdGig;
use stochastic_rs_distributions::geometric::SimdGeometric;
use stochastic_rs_distributions::gev::SimdGev;
use stochastic_rs_distributions::gpd::SimdGpd;
use stochastic_rs_distributions::hypergeometric::SimdHypergeometric;
use stochastic_rs_distributions::inverse_gauss::SimdInverseGauss;
use stochastic_rs_distributions::johnson_su::SimdJohnsonSu;
use stochastic_rs_distributions::lognormal::SimdLogNormal;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::normal_inverse_gauss::SimdNormalInverseGauss;
use stochastic_rs_distributions::pareto::SimdPareto;
use stochastic_rs_distributions::poisson::SimdPoisson;
use stochastic_rs_distributions::skellam::SimdSkellam;
use stochastic_rs_distributions::skew_t::SimdSkewT;
use stochastic_rs_distributions::studentt::SimdStudentT;
use stochastic_rs_distributions::tempered_stable::SimdTemperedStable;
use stochastic_rs_distributions::truncated::SimdTruncatedBeta;
use stochastic_rs_distributions::truncated::SimdTruncatedExp;
use stochastic_rs_distributions::truncated::SimdTruncatedGamma;
use stochastic_rs_distributions::truncated::SimdTruncatedNormal;
use stochastic_rs_distributions::uniform::SimdUniform;
use stochastic_rs_distributions::variance_gamma::SimdVarianceGamma;
use stochastic_rs_distributions::weibull::SimdWeibull;

const METHODS: [&str; 12] = [
  "cf", "pdf", "cdf", "quantile", "mean", "median", "mode", "variance", "skewness", "kurtosis",
  "entropy", "mgf",
];

/// Which methods answer at a point inside every law's support (x = 0.6, p = 0.3, t = 0.4).
fn answers(law: &dyn DistributionExt) -> [bool; 12] {
  let cf: Option<Complex64> = law.characteristic_function(0.4);
  [
    cf.is_some(),
    law.pdf(0.6).is_some(),
    law.cdf(0.6).is_some(),
    law.quantile(0.3).is_some(),
    law.mean().is_some(),
    law.median().is_some(),
    law.mode().is_some(),
    law.variance().is_some(),
    law.skewness().is_some(),
    law.kurtosis().is_some(),
    law.entropy().is_some(),
    law.moment_generating_function(0.4).is_some(),
  ]
}

fn check(name: &str, law: &dyn DistributionExt, expected: [bool; 12]) {
  let got = answers(law);
  for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
    assert_eq!(
      g,
      e,
      "{name}::{} should be {}",
      METHODS[i],
      if *e { "Some" } else { "None" }
    );
  }
}

const ALL: [bool; 12] = [true; 12];

#[test]
fn every_closed_form_of_the_base_commit_still_answers_and_no_other_does() {
  check(
    "SimdAlphaStable",
    &SimdAlphaStable::<f64>::new(1.5, 0.3, 1.0, 0.0),
    [
      true, false, false, false, true, true, true, true, true, true, false, true,
    ],
  );
  check(
    "SimdBeta",
    &SimdBeta::<f64>::new(2.0, 5.0),
    [
      false, true, true, true, true, true, true, true, true, true, true, false,
    ],
  );
  check(
    "SimdBinomial",
    &SimdBinomial::<u32>::new(10, 0.3),
    [
      true, true, true, true, true, true, true, true, true, true, false, true,
    ],
  );
  check("SimdCauchy", &SimdCauchy::<f64>::new(0.0, 1.0), ALL);
  check("SimdChiSquared", &SimdChiSquared::<f64>::new(3.0), ALL);
  check("SimdExp", &SimdExp::<f64>::new(1.5), ALL);
  check("SimdGamma", &SimdGamma::<f64>::new(2.0, 1.5), ALL);
  check(
    "SimdGed",
    &SimdGed::<f64>::new(0.3, 1.5, 1.3),
    [
      false, true, true, false, false, false, false, false, false, false, false, false,
    ],
  );
  check(
    "SimdGeneralizedHyperbolic",
    &SimdGeneralizedHyperbolic::<f64>::new(1.0, 2.0, 0.5, 1.0, 0.0),
    [
      false, true, false, false, true, false, false, true, true, true, false, true,
    ],
  );
  check("SimdGeometric", &SimdGeometric::<u32>::new(0.3), ALL);
  check(
    "SimdGev",
    &SimdGev::<f64>::new(0.0, 1.0, 0.1),
    [
      false, true, true, true, true, true, true, true, true, true, true, false,
    ],
  );
  check(
    "SimdGig",
    &SimdGig::<f64>::new(0.5, 1.0, 2.0),
    [
      false, true, false, false, true, false, true, true, true, true, false, true,
    ],
  );
  check(
    "SimdGpd",
    &SimdGpd::<f64>::new(0.0, 1.0, 0.1),
    [
      false, true, true, true, true, true, true, true, true, true, true, false,
    ],
  );
  check(
    "SimdHypergeometric",
    &SimdHypergeometric::<u32>::new(20, 7, 12),
    [
      false, true, true, true, true, true, true, true, true, false, false, false,
    ],
  );
  check(
    "SimdInverseGauss",
    &SimdInverseGauss::<f64>::new(1.0, 2.0),
    [
      true, true, true, false, true, true, true, true, true, true, false, true,
    ],
  );
  check(
    "SimdJohnsonSu",
    &SimdJohnsonSu::<f64>::new(0.5, 1.5, 0.0, 1.0),
    [
      false, true, true, true, true, true, false, true, true, true, false, false,
    ],
  );
  check(
    "SimdLogNormal",
    &SimdLogNormal::<f64>::new(0.0, 0.5),
    [
      false, true, true, true, true, true, true, true, true, true, true, false,
    ],
  );
  check("SimdNormal", &SimdNormal::<f64>::new(0.0, 1.0), ALL);
  check(
    "SimdNormalInverseGauss",
    &SimdNormalInverseGauss::<f64>::new(2.0, 0.5, 1.0, 0.0),
    [
      true, true, false, false, true, true, true, true, true, true, false, true,
    ],
  );
  check(
    "SimdPareto",
    &SimdPareto::<f64>::new(1.0, 3.0),
    [
      false, true, true, true, true, true, true, true, true, true, true, true,
    ],
  );
  check("SimdPoisson", &SimdPoisson::<u32>::new(4.0), ALL);
  check(
    "SimdSkellam",
    &SimdSkellam::new(3.0, 1.5),
    [
      false, true, true, false, false, false, false, false, false, false, false, false,
    ],
  );
  check(
    "SimdSkewT",
    &SimdSkewT::<f64>::new(5.0, 0.2),
    [
      false, true, true, true, true, true, true, true, true, true, false, false,
    ],
  );
  check(
    "SimdStudentT",
    &SimdStudentT::<f64>::new(5.0),
    [
      false, true, true, true, true, true, true, true, true, true, true, true,
    ],
  );
  check(
    "SimdTemperedStable",
    &SimdTemperedStable::<f64>::new(0.5, 1.0, 1.0),
    [
      true, false, false, false, true, false, false, true, true, true, false, true,
    ],
  );
  check(
    "SimdTruncatedBeta",
    &SimdTruncatedBeta::<f64>::new(2.0, 3.0, 0.2, 0.7),
    [
      false, true, true, false, false, false, false, false, false, false, false, false,
    ],
  );
  check(
    "SimdTruncatedExp",
    &SimdTruncatedExp::<f64>::new(1.5, 0.5, 2.5),
    [
      false, true, true, false, false, false, false, false, false, false, false, false,
    ],
  );
  check(
    "SimdTruncatedGamma",
    &SimdTruncatedGamma::<f64>::new(2.5, 1.2, 0.8, 4.0),
    [
      false, true, true, false, false, false, false, false, false, false, false, false,
    ],
  );
  check(
    "SimdTruncatedNormal",
    &SimdTruncatedNormal::<f64>::new(0.5, 2.0, -1.0, 2.0),
    [
      false, true, true, false, false, false, false, false, false, false, false, false,
    ],
  );
  check("SimdUniform", &SimdUniform::<f64>::new(0.0, 1.0), ALL);
  check(
    "SimdVarianceGamma",
    &SimdVarianceGamma::<f64>::new(0.2, 0.5, -0.1, 0.0),
    [
      true, true, false, false, true, false, false, true, true, true, false, true,
    ],
  );
  check(
    "SimdWeibull",
    &SimdWeibull::<f64>::new(1.0, 1.5),
    [
      false, true, true, true, true, true, true, true, true, true, true, false,
    ],
  );
}

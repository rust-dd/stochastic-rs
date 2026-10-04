//! Pins which `DistributionExt` methods answer `Some` for each law in `src`, and fails when an
//! implementor has no row; `Gbm` lives in the stochastic crate and pins its row in its own tests.

use std::collections::BTreeSet;
use std::path::Path;

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

const ALL: [bool; 12] = [true; 12];

type Row = (&'static str, Box<dyn DistributionExt>, [bool; 12]);

/// One row per law; `SimdAlphaStable` has two, its median and mode being closed-form only at β = 0.
fn rows() -> Vec<Row> {
  vec![
    (
      "SimdAlphaStable",
      Box::new(SimdAlphaStable::<f64>::new(1.5, 0.3, 1.0, 0.0)),
      [
        true, false, false, false, true, false, false, true, true, true, false, true,
      ],
    ),
    (
      "SimdAlphaStable",
      Box::new(SimdAlphaStable::<f64>::new(1.5, 0.0, 1.0, 0.0)),
      [
        true, false, false, false, true, true, true, true, true, true, false, true,
      ],
    ),
    (
      "SimdBeta",
      Box::new(SimdBeta::<f64>::new(2.0, 5.0)),
      [
        false, true, true, true, true, true, true, true, true, true, true, false,
      ],
    ),
    (
      "SimdBinomial",
      Box::new(SimdBinomial::<u32>::new(10, 0.3)),
      [
        true, true, true, true, true, true, true, true, true, true, false, true,
      ],
    ),
    (
      "SimdCauchy",
      Box::new(SimdCauchy::<f64>::new(0.0, 1.0)),
      ALL,
    ),
    (
      "SimdChiSquared",
      Box::new(SimdChiSquared::<f64>::new(3.0)),
      ALL,
    ),
    ("SimdExp", Box::new(SimdExp::<f64>::new(1.5)), ALL),
    ("SimdGamma", Box::new(SimdGamma::<f64>::new(2.0, 1.5)), ALL),
    (
      "SimdGed",
      Box::new(SimdGed::<f64>::new(0.3, 1.5, 1.3)),
      [
        false, true, true, false, false, false, false, false, false, false, false, false,
      ],
    ),
    (
      "SimdGeneralizedHyperbolic",
      Box::new(SimdGeneralizedHyperbolic::<f64>::new(
        1.0, 2.0, 0.5, 1.0, 0.0,
      )),
      [
        false, true, false, false, true, false, false, true, true, true, false, true,
      ],
    ),
    (
      "SimdGeometric",
      Box::new(SimdGeometric::<u32>::new(0.3)),
      ALL,
    ),
    (
      "SimdGev",
      Box::new(SimdGev::<f64>::new(0.0, 1.0, 0.1)),
      [
        false, true, true, true, true, true, true, true, true, true, true, false,
      ],
    ),
    (
      "SimdGig",
      Box::new(SimdGig::<f64>::new(0.5, 1.0, 2.0)),
      [
        false, true, false, false, true, false, true, true, true, true, false, true,
      ],
    ),
    (
      "SimdGpd",
      Box::new(SimdGpd::<f64>::new(0.0, 1.0, 0.1)),
      [
        false, true, true, true, true, true, true, true, true, true, true, false,
      ],
    ),
    (
      "SimdHypergeometric",
      Box::new(SimdHypergeometric::<u32>::new(20, 7, 12)),
      [
        false, true, true, true, true, true, true, true, true, false, false, false,
      ],
    ),
    (
      "SimdInverseGauss",
      Box::new(SimdInverseGauss::<f64>::new(1.0, 2.0)),
      [
        true, true, true, false, true, false, true, true, true, true, false, true,
      ],
    ),
    (
      "SimdJohnsonSu",
      Box::new(SimdJohnsonSu::<f64>::new(0.5, 1.5, 0.0, 1.0)),
      [
        false, true, true, true, true, true, false, true, true, true, false, false,
      ],
    ),
    (
      "SimdLogNormal",
      Box::new(SimdLogNormal::<f64>::new(0.0, 0.5)),
      [
        false, true, true, true, true, true, true, true, true, true, true, false,
      ],
    ),
    (
      "SimdNormal",
      Box::new(SimdNormal::<f64>::new(0.0, 1.0)),
      ALL,
    ),
    (
      "SimdNormalInverseGauss",
      Box::new(SimdNormalInverseGauss::<f64>::new(2.0, 0.5, 1.0, 0.0)),
      [
        true, true, false, false, true, false, false, true, true, true, false, true,
      ],
    ),
    (
      "SimdPareto",
      Box::new(SimdPareto::<f64>::new(1.0, 3.0)),
      [
        false, true, true, true, true, true, true, true, true, true, true, true,
      ],
    ),
    ("SimdPoisson", Box::new(SimdPoisson::<u32>::new(4.0)), ALL),
    (
      "SimdSkellam",
      Box::new(SimdSkellam::new(3.0, 1.5)),
      [
        false, true, true, false, false, false, false, false, false, false, false, false,
      ],
    ),
    (
      "SimdSkewT",
      Box::new(SimdSkewT::<f64>::new(5.0, 0.2)),
      [
        false, true, true, true, true, true, true, true, true, true, false, false,
      ],
    ),
    (
      "SimdStudentT",
      Box::new(SimdStudentT::<f64>::new(5.0)),
      [
        false, true, true, true, true, true, true, true, true, true, true, true,
      ],
    ),
    (
      "SimdTemperedStable",
      Box::new(SimdTemperedStable::<f64>::new(0.5, 1.0, 1.0)),
      [
        true, false, false, false, true, false, false, true, true, true, false, true,
      ],
    ),
    (
      "SimdTruncatedBeta",
      Box::new(SimdTruncatedBeta::<f64>::new(2.0, 3.0, 0.2, 0.7)),
      [
        false, true, true, false, false, false, false, false, false, false, false, false,
      ],
    ),
    (
      "SimdTruncatedExp",
      Box::new(SimdTruncatedExp::<f64>::new(1.5, 0.5, 2.5)),
      [
        false, true, true, false, false, false, false, false, false, false, false, false,
      ],
    ),
    (
      "SimdTruncatedGamma",
      Box::new(SimdTruncatedGamma::<f64>::new(2.5, 1.2, 0.8, 4.0)),
      [
        false, true, true, false, false, false, false, false, false, false, false, false,
      ],
    ),
    (
      "SimdTruncatedNormal",
      Box::new(SimdTruncatedNormal::<f64>::new(0.5, 2.0, -1.0, 2.0)),
      [
        false, true, true, false, false, false, false, false, false, false, false, false,
      ],
    ),
    (
      "SimdUniform",
      Box::new(SimdUniform::<f64>::new(0.0, 1.0)),
      ALL,
    ),
    (
      "SimdVarianceGamma",
      Box::new(SimdVarianceGamma::<f64>::new(0.2, 0.5, -0.1, 0.0)),
      [
        true, true, false, false, true, false, false, true, true, true, false, true,
      ],
    ),
    (
      "SimdWeibull",
      Box::new(SimdWeibull::<f64>::new(1.0, 1.5)),
      [
        false, true, true, true, true, true, true, true, true, true, true, false,
      ],
    ),
  ]
}

/// Which methods answer at a fixed probe (x = 0.6, p = 0.3, t = 0.4); the probe need not lie in a
/// law's support, since whether a method answers depends on the law and its parameters, not the point.
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

#[test]
fn each_law_answers_exactly_its_closed_forms() {
  for (name, law, expected) in rows() {
    let got = answers(law.as_ref());
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
}

#[test]
fn every_implementor_in_src_has_a_row() {
  let found = implementors(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src"));
  let covered = rows()
    .into_iter()
    .map(|(name, _, _)| name.to_string())
    .collect::<BTreeSet<_>>();
  assert_eq!(
    found, covered,
    "`impl DistributionExt for` types in src vs the rows above"
  );
}

/// The type after `impl … DistributionExt for` in every `.rs` file under `dir`, comment lines skipped.
fn implementors(dir: &Path) -> BTreeSet<String> {
  let mut found = BTreeSet::new();
  for entry in std::fs::read_dir(dir).expect("src is readable") {
    let path = entry.expect("a directory entry").path();
    if path.is_dir() {
      found.extend(implementors(&path));
      continue;
    }
    if path.extension().is_none_or(|e| e != "rs") {
      continue;
    }
    let text = std::fs::read_to_string(&path).expect("a source file");
    let code = text
      .lines()
      .filter(|l| !l.trim_start().starts_with("//"))
      .collect::<Vec<_>>()
      .join("\n");
    let tokens = code.split_whitespace().collect::<Vec<_>>();
    for (i, window) in tokens.windows(3).enumerate() {
      if !window[0].ends_with("DistributionExt") || window[1] != "for" {
        continue;
      }
      let header = tokens[..i]
        .iter()
        .rev()
        .take_while(|t| !t.contains([';', '{', '}']))
        .any(|t| t.starts_with("impl"));
      if header {
        let ty = window[2]
          .chars()
          .take_while(|c| c.is_alphanumeric() || *c == '_')
          .collect::<String>();
        found.insert(ty);
      }
    }
  }
  found
}

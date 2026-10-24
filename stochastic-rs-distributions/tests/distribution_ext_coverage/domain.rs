//! One rule for every law: NaN outside the domain, the support's ends at levels 0 and 1, `+∞` for a divergent mgf.

use stochastic_rs_distributions::DistributionExt;
use stochastic_rs_distributions::alpha_stable::SimdAlphaStable;
use stochastic_rs_distributions::binomial::SimdBinomial;
use stochastic_rs_distributions::cauchy::SimdCauchy;
use stochastic_rs_distributions::chi_square::SimdChiSquared;
use stochastic_rs_distributions::exp::SimdExp;
use stochastic_rs_distributions::gamma::SimdGamma;
use stochastic_rs_distributions::generalized_hyperbolic::SimdGeneralizedHyperbolic;
use stochastic_rs_distributions::generalized_inverse_gauss::SimdGig;
use stochastic_rs_distributions::geometric::SimdGeometric;
use stochastic_rs_distributions::gev::SimdGev;
use stochastic_rs_distributions::gpd::SimdGpd;
use stochastic_rs_distributions::inverse_gauss::SimdInverseGauss;
use stochastic_rs_distributions::normal_inverse_gauss::SimdNormalInverseGauss;
use stochastic_rs_distributions::pareto::SimdPareto;
use stochastic_rs_distributions::skew_t::SimdSkewT;
use stochastic_rs_distributions::studentt::SimdStudentT;
use stochastic_rs_distributions::tempered_stable::SimdTemperedStable;
use stochastic_rs_distributions::truncated::SimdTruncatedBeta;
use stochastic_rs_distributions::truncated::SimdTruncatedExp;
use stochastic_rs_distributions::truncated::SimdTruncatedGamma;
use stochastic_rs_distributions::truncated::SimdTruncatedNormal;
use stochastic_rs_distributions::uniform::SimdUniform;
use stochastic_rs_distributions::variance_gamma::SimdVarianceGamma;

use crate::answers;
use crate::rows;

fn is_nan(v: Option<f64>) -> bool {
  v.is_some_and(f64::is_nan)
}

/// The support of each `rows()` law with a `quantile`, at that row's parameters.
fn support(name: &str) -> Option<(f64, f64)> {
  let (n, p) = (f64::NEG_INFINITY, f64::INFINITY);
  Some(match name {
    "SimdBeta" | "SimdUniform" => (0.0, 1.0),
    "SimdBinomial" => (0.0, 10.0),
    "SimdCauchy" | "SimdGed" | "SimdJohnsonSu" | "SimdNormal" | "SimdSkewT" | "SimdStudentT" => {
      (n, p)
    }
    "SimdChiSquared" | "SimdExp" | "SimdGamma" | "SimdGpd" | "SimdLogNormal" | "SimdPoisson"
    | "SimdWeibull" => (0.0, p),
    "SimdGeometric" | "SimdPareto" => (1.0, p),
    "SimdGev" => (-10.0, p),
    "SimdHypergeometric" => (0.0, 7.0),
    "SimdTruncatedBeta" => (0.2, 0.7),
    "SimdTruncatedExp" => (0.5, 2.5),
    "SimdTruncatedGamma" => (0.8, 4.0),
    "SimdTruncatedNormal" => (-1.0, 2.0),
    _ => return None,
  })
}

type Branch = (&'static str, Box<dyn DistributionExt>, Option<(f64, f64)>);

/// Branches `rows()` does not reach, each with its support where the law has a `quantile`.
fn branches() -> Vec<Branch> {
  let (n, p) = (f64::NEG_INFINITY, f64::INFINITY);
  vec![
    (
      "AlphaStable α = 2",
      Box::new(SimdAlphaStable::<f64>::new(2.0, 0.0, 1.0, 0.0)),
      None,
    ),
    (
      "AlphaStable α = 1, β = 1",
      Box::new(SimdAlphaStable::<f64>::new(1.0, 1.0, 1.0, 0.0)),
      None,
    ),
    (
      "AlphaStable β = −1",
      Box::new(SimdAlphaStable::<f64>::new(1.5, -1.0, 1.0, 0.0)),
      None,
    ),
    (
      "Binomial n = 0",
      Box::new(SimdBinomial::<u32>::new(0, 0.3)),
      Some((0.0, 0.0)),
    ),
    (
      "Binomial p = 0",
      Box::new(SimdBinomial::<u32>::new(10, 0.0)),
      Some((0.0, 0.0)),
    ),
    (
      "Binomial p = 1",
      Box::new(SimdBinomial::<u32>::new(10, 1.0)),
      Some((10.0, 10.0)),
    ),
    (
      "Geometric p = 1",
      Box::new(SimdGeometric::<u64>::new(1.0)),
      Some((1.0, 1.0)),
    ),
    (
      "Gev ξ = 1",
      Box::new(SimdGev::<f64>::new(0.0, 1.0, 1.0)),
      Some((-1.0, p)),
    ),
    (
      "Gev ξ = −1",
      Box::new(SimdGev::<f64>::new(0.0, 1.0, -1.0)),
      Some((n, 1.0)),
    ),
    (
      "Gev ξ = 0",
      Box::new(SimdGev::<f64>::new(0.0, 1.0, 0.0)),
      Some((n, p)),
    ),
    (
      "Gpd ξ = 0",
      Box::new(SimdGpd::<f64>::new(0.0, 1.0, 0.0)),
      Some((0.0, p)),
    ),
    (
      "Gpd ξ = 1",
      Box::new(SimdGpd::<f64>::new(0.0, 1.0, 1.0)),
      Some((0.0, p)),
    ),
    (
      "Gpd ξ = −1",
      Box::new(SimdGpd::<f64>::new(0.0, 1.0, -1.0)),
      Some((0.0, 1.0)),
    ),
    (
      "GH λ = −1",
      Box::new(SimdGeneralizedHyperbolic::<f64>::new(
        -1.0, 2.0, 0.5, 1.5, -0.2,
      )),
      None,
    ),
    (
      "Gig λ = −0.5",
      Box::new(SimdGig::<f64>::new(-0.5, 2.0, 0.5)),
      None,
    ),
    (
      "SkewT λ = −0.9",
      Box::new(SimdSkewT::<f64>::new(5.0, -0.9)),
      Some((n, p)),
    ),
    (
      "SkewT λ = 0.09",
      Box::new(SimdSkewT::<f64>::new(5.0, 0.09)),
      Some((n, p)),
    ),
    (
      "TemperedStable λ = 0",
      Box::new(SimdTemperedStable::<f64>::new(0.6, 0.0, 1.5)),
      None,
    ),
    (
      "TruncatedBeta on [0, 1]",
      Box::new(SimdTruncatedBeta::<f64>::new(2.0, 3.0, 0.0, 1.0)),
      Some((0.0, 1.0)),
    ),
    (
      "TruncatedExp open tail",
      Box::new(SimdTruncatedExp::<f64>::new(1.5, 0.5, p)),
      Some((0.5, p)),
    ),
    (
      "TruncatedGamma open tail",
      Box::new(SimdTruncatedGamma::<f64>::new(2.5, 1.2, 0.8, p)),
      Some((0.8, p)),
    ),
    (
      "TruncatedNormal open tail",
      Box::new(SimdTruncatedNormal::<f64>::new(0.5, 2.0, -1.0, p)),
      Some((-1.0, p)),
    ),
    (
      "Uniform f32",
      Box::new(SimdUniform::<f32>::new(0.1, 0.7)),
      Some((f64::from(0.1_f32), f64::from(0.7_f32))),
    ),
  ]
}

/// A `p` outside `[0, 1]` and a NaN argument answer NaN wherever the law answers at all; `mask` indexes
/// `METHODS` (cf 0, pdf 1, cdf 2, quantile 3, mgf 11).
#[test]
fn out_of_domain_and_nan_arguments_answer_nan() {
  let laws = rows().into_iter().chain(
    branches()
      .into_iter()
      .map(|(name, law, _)| (name, answers(law.as_ref()), law))
      .map(|(name, mask, law)| (name, law, mask)),
  );
  for (name, law, mask) in laws {
    if mask[3] {
      for p in [-0.5, 1.5, f64::NAN] {
        assert!(
          is_nan(law.quantile(p)),
          "{name}: quantile({p}) = {:?}",
          law.quantile(p)
        );
      }
    }
    if mask[1] {
      assert!(
        is_nan(law.pdf(f64::NAN)),
        "{name}: pdf(NaN) = {:?}",
        law.pdf(f64::NAN)
      );
    }
    if mask[2] {
      assert!(
        is_nan(law.cdf(f64::NAN)),
        "{name}: cdf(NaN) = {:?}",
        law.cdf(f64::NAN)
      );
    }
    if mask[11] {
      let m = law.moment_generating_function(f64::NAN);
      assert!(is_nan(m), "{name}: mgf(NaN) = {m:?}");
    }
    if mask[0] {
      let z = law.characteristic_function(f64::NAN).unwrap();
      assert!(z.re.is_nan() && z.im.is_nan(), "{name}: cf(NaN) = {z}");
    }
  }
}

/// `quantile(0)` and `quantile(1)` are exactly the support's ends, for every `rows()` law with a `quantile` and every branch.
#[test]
fn quantile_at_zero_and_one_is_the_support() {
  let mut ends = Vec::new();
  for (name, law, mask) in rows() {
    if mask[3] {
      let s = support(name).unwrap_or_else(|| panic!("{name} has a quantile but no support entry"));
      ends.push((name, law, s));
    }
  }
  ends.extend(
    branches()
      .into_iter()
      .filter_map(|(name, law, s)| s.map(|s| (name, law, s))),
  );
  for (name, law, (lo, hi)) in ends {
    assert_eq!(law.quantile(0.0), Some(lo), "{name}: quantile(0)");
    assert_eq!(law.quantile(1.0), Some(hi), "{name}: quantile(1)");
  }
}

/// `E[e^{tX}] = +∞` answers `+∞`; on the boundary of the GH and GIG regions with λ < 0 the K-limit is finite (mpmath).
#[test]
fn a_divergent_mgf_is_infinite() {
  let cases: Vec<(&str, Box<dyn DistributionExt>, f64)> = vec![
    (
      "AlphaStable",
      Box::new(SimdAlphaStable::<f64>::new(1.5, 0.3, 1.0, 0.0)),
      1.0,
    ),
    (
      "AlphaStable β = 1, heavy side",
      Box::new(SimdAlphaStable::<f64>::new(1.5, 1.0, 1.0, 0.0)),
      1.0,
    ),
    (
      "AlphaStable β = −1, heavy side",
      Box::new(SimdAlphaStable::<f64>::new(0.6, -1.0, 1.0, 0.0)),
      -1.0,
    ),
    ("Cauchy", Box::new(SimdCauchy::<f64>::new(1.0, 0.5)), 1.0),
    ("ChiSquared", Box::new(SimdChiSquared::<f64>::new(6.0)), 1.0),
    ("Exp", Box::new(SimdExp::<f64>::new(1.8)), 3.0),
    ("Gamma", Box::new(SimdGamma::<f64>::new(2.5, 1.5)), 1.0),
    ("Geometric", Box::new(SimdGeometric::<u64>::new(0.15)), 1.0),
    (
      "GH",
      Box::new(SimdGeneralizedHyperbolic::<f64>::new(
        1.0, 2.0, 0.5, 1.5, -0.2,
      )),
      5.0,
    ),
    (
      "GH λ = 0 at the boundary",
      Box::new(SimdGeneralizedHyperbolic::<f64>::new(
        0.0, 2.0, 0.5, 1.5, -0.2,
      )),
      1.5,
    ),
    ("Gig", Box::new(SimdGig::<f64>::new(0.3, 2.0, 0.5)), 1.0),
    (
      "Gig λ > 0 at the boundary",
      Box::new(SimdGig::<f64>::new(0.3, 2.0, 0.5)),
      0.25,
    ),
    (
      "InverseGauss",
      Box::new(SimdInverseGauss::<f64>::new(1.5, 3.0)),
      1.0,
    ),
    (
      "NIG",
      Box::new(SimdNormalInverseGauss::<f64>::new(2.0, 0.5, 1.0, 0.1)),
      3.0,
    ),
    ("Pareto", Box::new(SimdPareto::<f64>::new(1.0, 1.16)), 1.0),
    ("StudentT", Box::new(SimdStudentT::<f64>::new(6.0)), 1.0),
    (
      "TemperedStable",
      Box::new(SimdTemperedStable::<f64>::new(0.6, 2.0, 1.5)),
      3.0,
    ),
    (
      "TruncatedExp",
      Box::new(SimdTruncatedExp::<f64>::new(1.5, 0.5, f64::INFINITY)),
      2.0,
    ),
    (
      "VarianceGamma",
      Box::new(SimdVarianceGamma::<f64>::new(0.2, 0.5, -0.1, 0.05)),
      50.0,
    ),
  ];
  for (name, law, t) in cases {
    assert_eq!(
      law.moment_generating_function(t),
      Some(f64::INFINITY),
      "{name}: mgf({t})"
    );
  }
  let rel = |got: f64, want: f64| (got - want).abs() / want;
  let gh = SimdGeneralizedHyperbolic::<f64>::new(-1.0, 2.0, 0.5, 1.5, -0.2);
  assert!(
    rel(
      gh.moment_generating_function(1.5).unwrap(),
      5.663_931_274_116_102
    ) < 1e-12
  );
  assert_eq!(gh.moment_generating_function(1.6), Some(f64::INFINITY));
  let gig = SimdGig::<f64>::new(-0.5, 2.0, 0.5);
  assert!(
    rel(
      gig.moment_generating_function(0.25).unwrap(),
      std::f64::consts::E
    ) < 1e-12
  );
  assert_eq!(gig.moment_generating_function(0.3), Some(f64::INFINITY));
}

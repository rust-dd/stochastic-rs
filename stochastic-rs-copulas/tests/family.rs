//! One family enum per trait, read through `family()`; the vine fitter's candidates are a subset of it.

use ndarray::Array2;
use stochastic_rs_copulas::CopulaError;
use stochastic_rs_copulas::bivariate::BivariateFamily;
use stochastic_rs_copulas::bivariate::clayton::Clayton;
use stochastic_rs_copulas::bivariate::t_copula::TCopula;
use stochastic_rs_copulas::multivariate::MultivariateFamily;
use stochastic_rs_copulas::multivariate::fit::SelectionCriterion;
use stochastic_rs_copulas::multivariate::fit::VineStructure;
use stochastic_rs_copulas::multivariate::fit::fit_vine;
use stochastic_rs_copulas::multivariate::nac::NacNode;
use stochastic_rs_copulas::multivariate::nac::NestedArchimedean;
use stochastic_rs_copulas::multivariate::rvine::RVine;
use stochastic_rs_copulas::traits::BivariateExt;
use stochastic_rs_copulas::traits::MultivariateExt;

#[test]
fn families_are_one_enum_per_trait() {
  assert_eq!(Clayton::new().family(), BivariateFamily::Clayton);
  assert_eq!(TCopula::new().family(), BivariateFamily::TCopula);
  assert_eq!(BivariateFamily::ALL.len(), 15);
  assert!(
    BivariateFamily::VINE_CANDIDATES
      .iter()
      .all(|f| BivariateFamily::ALL.contains(f))
  );
  assert_eq!(BivariateFamily::Bb1.parameter_count(), 2);
  assert_eq!(BivariateFamily::Independence.parameter_count(), 0);
  let rvine = RVine::independence(3).unwrap();
  assert_eq!(rvine.family(), MultivariateFamily::RVine);
  assert_eq!(rvine.structure(), VineStructure::DVine);
  let nac = NestedArchimedean::new(
    BivariateFamily::Gumbel,
    NacNode::leaf_group(2.0, vec![0, 1]),
    2,
  )
  .unwrap();
  assert_eq!(nac.generator_family(), BivariateFamily::Gumbel);
  assert!(matches!(
    NestedArchimedean::new(
      BivariateFamily::Frank,
      NacNode::leaf_group(2.0, vec![0, 1]),
      2
    ),
    Err(CopulaError::Unsupported(_))
  ));
  let u = Array2::<f64>::from_shape_fn((40, 2), |(i, j)| ((i + 1) as f64 + 0.3 * j as f64) / 41.0);
  assert!(matches!(
    fit_vine(
      u.view(),
      VineStructure::DVine,
      &[BivariateFamily::Gumbel],
      SelectionCriterion::Aic
    ),
    Err(CopulaError::Unsupported(_))
  ));
}

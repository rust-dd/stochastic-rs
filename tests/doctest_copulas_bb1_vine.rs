// docs: copulas#bb1-bb7-two-parameter-tail-asymmetry
//! Backs the BB1 / vine-fitting example on the copulas catalog page.

use stochastic_rs::copulas::bivariate::bb1::Bb1;
use stochastic_rs::copulas::gof::gof_cramer_von_mises;
use stochastic_rs::copulas::gof::pseudo_observations;
use stochastic_rs::copulas::multivariate::fit::PairFamily;
use stochastic_rs::copulas::multivariate::fit::SelectionCriterion;
use stochastic_rs::copulas::multivariate::fit::VineStructure;
use stochastic_rs::copulas::multivariate::fit::fit_vine;
use stochastic_rs::traits::BivariateExt;

#[test]
fn bb1_fit_gof_and_vine_selection() {
  // BB1 with lower tail 2^{-1/(θδ)} and upper tail 2 - 2^{1/δ}: asymmetric tails.
  let truth = Bb1::new(Some(0.8), Some(1.6), None);
  let tails = truth.tail_dependence();
  assert!(tails.lower > 0.5 && tails.upper > 0.4 && tails.lower != tails.upper);
  // Every fit and test runs on pseudo-observations (normalised ranks), as with real data.
  let uv = pseudo_observations(&truth.sample_with_seed(2_000, 42).unwrap()); // shape (2_000, 2)

  // Maximum-likelihood fit of (θ, δ) recovers the generating pair.
  let mut fitted = Bb1::default();
  fitted.fit(&uv).unwrap();
  assert!((fitted.theta.unwrap() - 0.8).abs() < 0.25 && (fitted.delta - 1.6).abs() < 0.25);

  // Parametric-bootstrap Cramér–von Mises test does not reject the true family.
  let gof = gof_cramer_von_mises(&fitted, &uv, 20, 7, |c, x| c.fit(x)).unwrap();
  assert!(gof.p_value > 0.05, "p = {}", gof.p_value);

  // A two-column D-vine fit selects a family on the single edge by AIC.
  let fit = fit_vine(
    &uv,
    VineStructure::DVine,
    &PairFamily::ALL,
    SelectionCriterion::Aic,
  )
  .unwrap();
  assert_eq!(fit.families.len(), 1);
  assert!(fit.aic.is_finite() && fit.log_likelihood > 0.0);
}

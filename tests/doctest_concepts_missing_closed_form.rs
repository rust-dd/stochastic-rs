// docs: concepts/distribution-ext#a-missing-closed-form-is-none-never-zero
//! Backs the "a missing closed form is `None`" example on the DistributionExt concept page.

use stochastic_rs::distributions::ged::SimdGed;
use stochastic_rs::traits::DistributionExt;

#[test]
fn a_law_without_a_closed_form_answers_none() {
  let ged = SimdGed::<f64>::new(0.0, 1.0, 1.5);
  // The GED has a closed-form density but no closed-form characteristic function.
  assert!(ged.pdf(0.0).is_some());
  assert!(ged.characteristic_function(1.0).is_none());
}

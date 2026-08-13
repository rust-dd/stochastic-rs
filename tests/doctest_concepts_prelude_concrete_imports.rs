// docs: concepts/prelude#importing-concrete-types
//! Backs the "importing concrete types" example on the prelude concept
//! page: the prelude ships traits and option-type enums only, so concrete
//! types are always named explicitly.

use stochastic_rs::prelude::*;
use stochastic_rs::quant::calibration::heston::HestonCalibrator;
use stochastic_rs::quant::pricing::heston::HestonPricer;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stochastic::diffusion::ou::Ou;

#[test]
fn prelude_plus_concrete_imports_resolve() {
  let p = Ou::<f64, _>::new(
    2.0,
    0.0,
    1.0,
    64,
    Some(0.0),
    Some(1.0),
    Deterministic::new(1),
  );
  let path = p.sample();
  assert_eq!(path.len(), 64);

  // Concrete quant types resolve too — no calibration run here, just the
  // import surface named in the doc.
  fn _type_check(_: &HestonPricer, _: &HestonCalibrator) {}
}

//! `BsmPricer` at a spot that is not positive.

use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::pricing::bsm::BsmCoc;
use stochastic_rs_quant::pricing::bsm::BsmPricer;
use stochastic_rs_quant::traits::ModelPricer;

use super::assert_all_nan;

const K: f64 = 100.0;
const R: f64 = 0.05;
const Q: f64 = 0.02;

/// Before expiry the closed form reads `ln S`; at expiry a spot that is not positive has a payoff.
#[test]
fn bsm_is_nan_for_a_spot_that_is_not_positive_before_expiry() {
  for carry in [BsmCoc::Merton1973, BsmCoc::Black1976] {
    let bsm = BsmPricer::new(0.2, carry);
    for s in [0.0, -5.0] {
      for option_type in [OptionType::Call, OptionType::Put] {
        let price = bsm.price_option(s, K, R, Q, 1.0, option_type);
        assert!(
          price.is_nan(),
          "{carry:?} {option_type:?} at s = {s}: {price}"
        );
        assert_all_nan(
          bsm.greeks(s, K, R, Q, 1.0, option_type),
          "a spot not positive",
        );
      }
      assert_eq!(bsm.price_call(s, K, R, Q, 0.0), 0.0);
      assert_eq!(bsm.price_put(s, K, R, Q, 0.0), K - s);
    }
    assert!(bsm.price_call(1e-300, K, R, Q, 1.0) == 0.0);
  }
}

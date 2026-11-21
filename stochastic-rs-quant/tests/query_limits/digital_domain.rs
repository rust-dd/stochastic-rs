//! The four digital pricers at a spot or strike that is not positive.

use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::pricing::AssetOrNothingPricer;
use stochastic_rs_quant::pricing::CashOrNothingPricer;
use stochastic_rs_quant::pricing::GapPricer;
use stochastic_rs_quant::pricing::SuperSharePricer;
use stochastic_rs_quant::traits::ModelPricer;

use super::assert_all_nan;

const R: f64 = 0.05;
const Q: f64 = 0.02;
const TYPES: [OptionType; 2] = [OptionType::Call, OptionType::Put];
/// `(s, k)` with one coordinate zero or negative.
const OFF: [(f64, f64); 4] = [(0.0, 100.0), (-5.0, 100.0), (100.0, 0.0), (100.0, -5.0)];

/// Each price and Greek reads `ln(S/K)` through one `d₁`, so none has a value there.
#[test]
fn a_digital_is_nan_before_expiry_for_a_spot_or_strike_that_is_not_positive() {
  let cash = CashOrNothingPricer::new(10.0, 0.2);
  let asset = AssetOrNothingPricer::new(0.2);
  let models: [(&str, &dyn ModelPricer); 4] = [
    ("cash-or-nothing", &cash),
    ("asset-or-nothing", &asset),
    ("gap", &GapPricer::new(95.0, 0.2)),
    ("supershare", &SuperSharePricer::new(110.0, 0.2)),
  ];
  for (s, k) in OFF {
    for (what, model) in models {
      for leg in TYPES {
        let price = model.price_option(s, k, R, Q, 1.0, leg);
        assert!(price.is_nan(), "{what} {leg:?} at ({s}, {k}): {price}");
      }
    }
    for leg in TYPES {
      assert_all_nan(cash.greeks(s, k, R, Q, 1.0, leg), "cash-or-nothing");
      assert_all_nan(asset.greeks(s, k, R, Q, 1.0, leg), "asset-or-nothing");
      let singles = [
        cash.delta(s, k, R, Q, 1.0, leg),
        cash.gamma(s, k, R, Q, 1.0, leg),
        cash.vega(s, k, R, Q, 1.0, leg),
        asset.delta(s, k, R, Q, 1.0, leg),
      ];
      assert!(
        singles.iter().all(|x| x.is_nan()),
        "({s}, {k}): {singles:?}"
      );
    }
  }
}

#[test]
fn a_digital_pays_out_at_expiry_at_a_spot_or_strike_that_is_not_positive() {
  let cash = CashOrNothingPricer::new(10.0, 0.2);
  let asset = AssetOrNothingPricer::new(0.2);
  let gap = GapPricer::new(95.0, 0.2);
  assert_eq!(cash.price_put(0.0, 100.0, R, Q, 0.0), 10.0);
  assert_eq!(cash.price_call(100.0, -5.0, R, Q, 0.0), 10.0);
  assert_eq!(asset.price_call(100.0, 0.0, R, Q, 0.0), 100.0);
  assert_eq!(asset.price_put(-5.0, 100.0, R, Q, 0.0), -5.0);
  assert_eq!(gap.price_put(0.0, 100.0, R, Q, 0.0), 95.0);
  assert_eq!(gap.price_call(100.0, 0.0, R, Q, 0.0), 5.0);
}

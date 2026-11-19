//! The exotic closed forms at the edges of their domains.

use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::pricing::barrier::DoubleBarrierPricer;
use stochastic_rs_quant::pricing::exotic::DoubleBarrierOption;

const R: f64 = 0.05;
const Q: f64 = 0.02;
const SIGMA: f64 = 0.25;

/// A lower bound that is not positive is NaN before expiry, and a corridor narrow against `σ√τ`
/// prices 0 without summing its images.
#[test]
fn the_double_barrier_at_the_edges_of_its_series() {
  let pricer = DoubleBarrierPricer::new(SIGMA);
  let mut option = DoubleBarrierOption {
    strike: 100.0,
    lower: 0.0,
    upper: 130.0,
    option_type: OptionType::Call,
  };
  assert!(pricer.price(&option, 100.0, R, Q, 1.0).is_nan());
  assert_eq!(pricer.price(&option, 110.0, R, Q, 0.0), 10.0);
  option.lower = 99.999;
  option.upper = 100.001;
  assert_eq!(pricer.price(&option, 100.0, R, Q, 1.0), 0.0);
}

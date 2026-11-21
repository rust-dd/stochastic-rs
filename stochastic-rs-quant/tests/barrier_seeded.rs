use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::pricing::barrier::BarrierType;
use stochastic_rs_quant::pricing::barrier::McBarrierPricer;
use stochastic_rs_quant::pricing::exotic::BarrierOption;

const PIN: f64 = 3.833321894842116;

fn price(seed: u64) -> f64 {
  let option = BarrierOption {
    strike: 100.0,
    barrier: 130.0,
    rebate: 0.0,
    barrier_type: BarrierType::UpAndOut,
    option_type: OptionType::Call,
  };
  McBarrierPricer::new(0.2, 4_000, 32)
    .with_seed(Deterministic::new(seed))
    .price(&option, 100.0, 0.03, 0.0, 1.0)
    .mean
}

#[test]
fn a_seeded_price_replays_and_depends_on_the_seed() {
  assert_eq!(price(7).to_bits(), price(7).to_bits());
  assert!((price(7) - price(8)).abs() > 1e-9);
}

#[test]
fn a_seeded_price_is_pinned() {
  assert!(
    (price(7) - PIN).abs() <= 1e-12 * PIN.abs(),
    "price(7) = {:?}",
    price(7)
  );
}

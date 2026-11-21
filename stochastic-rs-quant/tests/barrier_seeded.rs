use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::pricing::barrier::BarrierType;
use stochastic_rs_quant::pricing::barrier::McBarrierPricer;
use stochastic_rs_quant::pricing::exotic::BarrierOption;

const PIN: f64 = 3.7606676940909685;

fn price(seed: u64) -> f64 {
  let option = BarrierOption {
    strike: 100.0,
    barrier: 130.0,
    rebate: 0.0,
    barrier_type: BarrierType::UpAndOut,
    option_type: OptionType::Call,
  };
  McBarrierPricer::new(0.2, 4_000, 32)
    .price_seeded(&option, 100.0, 0.03, 0.0, 1.0, seed)
    .mean
}

#[test]
fn price_seeded_replays_and_depends_on_the_seed() {
  assert!((price(7) - price(7)).abs() <= 1e-12 * price(7).abs());
  assert!((price(7) - price(8)).abs() > 1e-9);
}

#[test]
fn price_seeded_is_pinned() {
  assert!(
    (price(7) - PIN).abs() <= 1e-12 * PIN.abs(),
    "price(7) = {:?}",
    price(7)
  );
}

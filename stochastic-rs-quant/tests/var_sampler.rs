use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_quant::risk::PnlOrLoss;
use stochastic_rs_quant::risk::monte_carlo_var_with_sampler;

const PIN: u64 = 0x4002de256b02d9b8;

fn sampler(seed: u64) -> Seeded<SimdNormal<f64>> {
  SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(seed))
}

fn var(sampler: &mut Seeded<SimdNormal<f64>>) -> f64 {
  monte_carlo_var_with_sampler(sampler, 20_000, 0.99, PnlOrLoss::Loss)
}

#[test]
fn the_sampler_var_replays_and_depends_on_the_seed() {
  assert_eq!(
    var(&mut sampler(7)).to_bits(),
    var(&mut sampler(7)).to_bits()
  );
  assert_ne!(
    var(&mut sampler(7)).to_bits(),
    var(&mut sampler(8)).to_bits()
  );
}

#[test]
fn a_second_call_continues_the_stream() {
  let mut shared = sampler(7);
  let first = var(&mut shared);
  let second = var(&mut shared);
  assert_eq!(first.to_bits(), var(&mut sampler(7)).to_bits());
  assert_ne!(first.to_bits(), second.to_bits());
}

#[test]
fn the_sampler_var_is_pinned() {
  let got = var(&mut sampler(7)).to_bits();
  assert_eq!(got, PIN, "var(7) bits = {got:#018x}");
}

//! Every Monte Carlo estimate carries its error, and a dividend yield reaches the barrier simulation.

use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::pricing::barrier::BarrierPricer;
use stochastic_rs_quant::pricing::barrier::BarrierType;
use stochastic_rs_quant::pricing::barrier::McBarrierPricer;
use stochastic_rs_quant::pricing::bsm::BsmCoc;
use stochastic_rs_quant::pricing::bsm::BsmPricer;
use stochastic_rs_quant::pricing::cliquet::McCliquetPricer;
use stochastic_rs_quant::pricing::exotic::BarrierOption;
use stochastic_rs_quant::pricing::exotic::CliquetOption;
use stochastic_rs_quant::traits::ModelPricer;

fn barrier(barrier: f64, rebate: f64, barrier_type: BarrierType) -> BarrierOption {
  BarrierOption {
    strike: 100.0,
    barrier,
    rebate,
    barrier_type,
    option_type: OptionType::Call,
  }
}

#[test]
fn the_mc_barrier_price_sees_the_dividend_yield() {
  let option = barrier(80.0, 0.0, BarrierType::DownAndOut);
  let mc = McBarrierPricer::new(0.2, 40_000, 252);
  let (s, r, tau) = (100.0, 0.05, 1.0);
  let no_div = mc.price_seeded(&option, s, r, 0.0, tau, 11);
  let with_div = mc.price_seeded(&option, s, r, 0.04, tau, 11);
  assert!(
    with_div.mean < no_div.mean - 3.0 * (with_div.std_err + no_div.std_err),
    "{with_div} vs {no_div}"
  );
  let analytic = BarrierPricer::new(0.2).price(&option, s, r, 0.04, tau);
  assert!(
    (with_div.mean - analytic).abs() < 4.0 * with_div.std_err + 0.15,
    "mc {with_div} vs analytic {analytic} (discrete monitoring bias)"
  );
  assert_eq!(with_div.n_samples, 40_000);
}

/// Monitoring at 252 steps prices as continuous monitoring of a barrier moved away from the spot
/// by `e^{0.5826 σ √Δt}` (Broadie, Glasserman & Kou 1997, Math. Finance 7(4), 325–349).
#[test]
fn the_mc_barrier_pays_the_rebate_the_closed_form_pays() {
  let (sigma, n_steps) = (0.25, 252);
  let mc = McBarrierPricer::new(sigma, 40_000, n_steps);
  let closed = BarrierPricer::new(sigma);
  let shift = (0.5826 * sigma * (1.0 / n_steps as f64).sqrt()).exp();
  for (option, shifted) in [
    (barrier(120.0, 3.0, BarrierType::UpAndOut), 120.0 * shift),
    (barrier(85.0, 3.0, BarrierType::DownAndIn), 85.0 / shift),
  ] {
    let est = mc.price_seeded(&option, 100.0, 0.05, 0.02, 1.0, 5);
    let continuous = BarrierOption {
      barrier: shifted,
      ..option
    };
    let analytic = closed.price(&continuous, 100.0, 0.05, 0.02, 1.0);
    assert!(
      (est.mean - analytic).abs() < 4.0 * est.std_err + 0.05,
      "{:?}: mc {est} vs analytic {analytic}",
      option.barrier_type
    );
  }
}

#[test]
fn a_breached_barrier_has_knocked_at_the_spot() {
  let mc = McBarrierPricer::new(0.2, 4_000, 16);
  let out = mc.price_seeded(
    &barrier(80.0, 3.0, BarrierType::DownAndOut),
    75.0,
    0.05,
    0.0,
    1.0,
    1,
  );
  assert!((out.mean - 3.0).abs() < 1e-12, "{out}");
  let knocked_in = mc.price_seeded(
    &barrier(80.0, 0.0, BarrierType::DownAndIn),
    75.0,
    0.05,
    0.0,
    1.0,
    1,
  );
  let vanilla = BsmPricer::new(0.2, BsmCoc::Merton1973).price_call(75.0, 100.0, 0.05, 0.0, 1.0);
  assert!(
    (knocked_in.mean - vanilla).abs() < 4.0 * knocked_in.std_err + 0.05,
    "{knocked_in}"
  );
}

#[test]
fn the_mc_barrier_and_cliquet_answer_nan_off_their_domain() {
  let mc = McBarrierPricer::new(0.2, 64, 8);
  let option = barrier(130.0, 0.0, BarrierType::UpAndOut);
  for (s, r, tau) in [
    (100.0, 0.05, -1.0),
    (f64::NAN, 0.05, 1.0),
    (100.0, f64::NAN, 1.0),
    (0.0, 0.05, 1.0),
  ] {
    let est = mc.price_seeded(&option, s, r, 0.0, tau, 3);
    assert!(
      est.mean.is_nan() && est.n_samples == 0,
      "s={s} r={r} tau={tau}: {est:?}"
    );
  }
  let at_expiry = mc.price_seeded(&option, 110.0, 0.05, 0.0, 0.0, 3);
  assert_eq!(at_expiry.mean, 10.0);
  let cliquet = CliquetOption {
    notional: 100.0,
    periods: 4,
    local_floor: Some(0.01),
    local_cap: Some(0.05),
    global_floor: None,
    global_cap: Some(0.03),
  };
  let mc = McCliquetPricer::new(0.2, 64);
  assert!(mc.price(&cliquet, 0.03, 0.0, -1.0).mean.is_nan());
  assert!(mc.price(&cliquet, f64::NAN, 0.0, 1.0).mean.is_nan());
  assert!((mc.price(&cliquet, 0.03, 0.0, 0.0).mean - 3.0).abs() < 1e-12);
}

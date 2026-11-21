//! Every Monte Carlo estimate carries its error and is seeded through `S: SeedExt` alike, and a
//! dividend yield reaches the barrier simulation.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::pricing::autocallable::AutocallablePricer;
use stochastic_rs_quant::pricing::autocallable::KnockInStyle;
use stochastic_rs_quant::pricing::barrier::BarrierPricer;
use stochastic_rs_quant::pricing::barrier::BarrierType;
use stochastic_rs_quant::pricing::barrier::McBarrierPricer;
use stochastic_rs_quant::pricing::basket::BasketAverageType;
use stochastic_rs_quant::pricing::basket::McBasketPricer;
use stochastic_rs_quant::pricing::bermudan::generate_gbm_paths;
use stochastic_rs_quant::pricing::bsm::BsmCoc;
use stochastic_rs_quant::pricing::bsm::BsmPricer;
use stochastic_rs_quant::pricing::cliquet::McCliquetPricer;
use stochastic_rs_quant::pricing::exotic::AutocallableNote;
use stochastic_rs_quant::pricing::exotic::BarrierOption;
use stochastic_rs_quant::pricing::exotic::CliquetOption;
use stochastic_rs_quant::pricing::rainbow::McRainbowPricer;
use stochastic_rs_quant::pricing::rainbow::RainbowPayoff;
use stochastic_rs_quant::pricing::spread::McSpreadPricer;
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
  let no_div = mc
    .with_seed(Deterministic::new(11))
    .price(&option, s, r, 0.0, tau);
  let with_div = mc
    .with_seed(Deterministic::new(11))
    .price(&option, s, r, 0.04, tau);
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
    let est = mc
      .with_seed(Deterministic::new(5))
      .price(&option, 100.0, 0.05, 0.02, 1.0);
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
  let mc = McBarrierPricer::new(0.2, 4_000, 16).with_seed(Deterministic::new(1));
  let out = mc.price(
    &barrier(80.0, 3.0, BarrierType::DownAndOut),
    75.0,
    0.05,
    0.0,
    1.0,
  );
  assert!((out.mean - 3.0).abs() < 1e-12, "{out}");
  let knocked_in = mc.price(
    &barrier(80.0, 0.0, BarrierType::DownAndIn),
    75.0,
    0.05,
    0.0,
    1.0,
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
    let est = mc
      .with_seed(Deterministic::new(3))
      .price(&option, s, r, 0.0, tau);
    assert!(
      est.mean.is_nan() && est.n_samples == 0,
      "s={s} r={r} tau={tau}: {est:?}"
    );
  }
  let at_expiry = mc
    .with_seed(Deterministic::new(3))
    .price(&option, 110.0, 0.05, 0.0, 0.0);
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

fn reproduces(name: &str, run: impl Fn() -> (f64, f64) + Sync) {
  let one = rayon::ThreadPoolBuilder::new()
    .num_threads(1)
    .build()
    .unwrap();
  let four = rayon::ThreadPoolBuilder::new()
    .num_threads(4)
    .build()
    .unwrap();
  let (first, second) = (one.install(&run), four.install(&run));
  assert_eq!(
    first.0.to_bits(),
    second.0.to_bits(),
    "{name}: the mean depends on the pool"
  );
  assert_eq!(
    first.1.to_bits(),
    second.1.to_bits(),
    "{name}: the error depends on the pool"
  );
  assert_eq!(first, run(), "{name}: one seed, one stream");
}

fn pair(e: stochastic_rs_stochastic::mc::McEstimate<f64>) -> (f64, f64) {
  (e.mean, e.std_err)
}

#[test]
fn every_monte_carlo_estimator_is_seeded_the_same_way() {
  let option = barrier(80.0, 0.0, BarrierType::DownAndOut);
  let a = McBarrierPricer::new(0.2, 5_000, 50).with_seed(Deterministic::new(11));
  let c = McBarrierPricer::new(0.2, 5_000, 50).with_seed(Deterministic::new(12));
  reproduces("barrier", || pair(a.price(&option, 100.0, 0.05, 0.0, 1.0)));
  assert_ne!(
    a.price(&option, 100.0, 0.05, 0.0, 1.0).mean,
    c.price(&option, 100.0, 0.05, 0.0, 1.0).mean
  );

  let cliquet = CliquetOption {
    notional: 1.0,
    periods: 4,
    local_floor: Some(0.0),
    local_cap: Some(0.05),
    global_floor: None,
    global_cap: None,
  };
  let seeded = McCliquetPricer::new(0.2, 2_000).with_seed(Deterministic::new(5));
  reproduces("cliquet", || pair(seeded.price(&cliquet, 0.03, 0.0, 1.0)));

  let note = AutocallableNote {
    notional: 100.0,
    observation_times: array![0.5, 1.0],
    autocall_barrier: 1.0,
    coupon_barrier: 0.8,
    coupon: 0.04,
    knock_in_barrier: 0.6,
    knock_in_style: KnockInStyle::EuropeanAtMaturity,
    memory: true,
  };
  let auto = AutocallablePricer::new(0.25, 2_000, 12).with_seed(Deterministic::new(3));
  reproduces("autocallable", || pair(auto.price(&note, 100.0, 0.02, 0.0)));

  let rho = Array2::from_shape_fn((2, 2), |(i, j)| if i == j { 1.0 } else { 0.4 });
  let spots = Array1::from_elem(2, 100.0);
  let yields = Array1::zeros(2);
  let rainbow = McRainbowPricer::new(
    RainbowPayoff::CallOnMax,
    array![0.2, 0.3],
    rho.clone(),
    4_000,
  )
  .with_seed(Deterministic::new(7));
  reproduces("rainbow", || {
    pair(rainbow.price(spots.view(), 100.0, 0.03, yields.view(), 1.0))
  });
  let basket = McBasketPricer::new(
    array![0.5, 0.5],
    BasketAverageType::Arithmetic,
    array![0.2, 0.3],
    rho,
    4_000,
  )
  .with_seed(Deterministic::new(8));
  reproduces("basket", || {
    pair(basket.price_call(spots.view(), 100.0, 0.03, yields.view(), 1.0))
  });
  let spread = McSpreadPricer::new(0.3, 0.2, 0.5, 4_000).with_seed(Deterministic::new(9));
  reproduces("spread", || {
    pair(spread.price_call(110.0, 100.0, 5.0, 0.03, 0.0, 0.0, 1.0))
  });

  let paths = || generate_gbm_paths(100.0, 0.05, 0.0, 0.2, 1.0, 500, 20, &Deterministic::new(4));
  assert_eq!(paths(), paths());
  let shared = Deterministic::new(4);
  let first = generate_gbm_paths(100.0, 0.05, 0.0, 0.2, 1.0, 500, 20, &shared);
  assert_ne!(
    first,
    generate_gbm_paths(100.0, 0.05, 0.0, 0.2, 1.0, 500, 20, &shared)
  );
}

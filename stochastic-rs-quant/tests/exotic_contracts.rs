//! The exotic pricers hold the model; a contract holds the terms, and the query travels.

use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::pricing::barrier::BarrierPricer;
use stochastic_rs_quant::pricing::barrier::BarrierType;
use stochastic_rs_quant::pricing::barrier::DoubleBarrierPricer;
use stochastic_rs_quant::pricing::bsm::BsmCoc;
use stochastic_rs_quant::pricing::bsm::BsmPricer;
use stochastic_rs_quant::pricing::chooser::ChooserPricer;
use stochastic_rs_quant::pricing::cliquet::CliquetPricer;
use stochastic_rs_quant::pricing::compound::CompoundPricer;
use stochastic_rs_quant::pricing::compound::CompoundType;
use stochastic_rs_quant::pricing::exotic::BarrierOption;
use stochastic_rs_quant::pricing::exotic::ChooserOption;
use stochastic_rs_quant::pricing::exotic::CliquetOption;
use stochastic_rs_quant::pricing::exotic::ComplexChooserOption;
use stochastic_rs_quant::pricing::exotic::CompoundOption;
use stochastic_rs_quant::pricing::exotic::DoubleBarrierOption;
use stochastic_rs_quant::pricing::exotic::ForwardStartOption;
use stochastic_rs_quant::pricing::exotic::LookbackOption;
use stochastic_rs_quant::pricing::exotic::VarianceSwapTerms;
use stochastic_rs_quant::pricing::lookback::LookbackPricer;
use stochastic_rs_quant::pricing::variance_swap::VarianceSwapPricer;
use stochastic_rs_quant::traits::ModelPricer;

const S: f64 = 100.0;
const R: f64 = 0.05;
const Q: f64 = 0.02;
const SIGMA: f64 = 0.25;

const CORRIDOR: DoubleBarrierOption = DoubleBarrierOption {
  strike: 100.0,
  lower: 70.0,
  upper: 130.0,
  option_type: OptionType::Call,
};

fn barrier(barrier_type: BarrierType, option_type: OptionType, rebate: f64) -> BarrierOption {
  let barrier = match barrier_type {
    BarrierType::UpAndIn | BarrierType::UpAndOut => 130.0,
    BarrierType::DownAndIn | BarrierType::DownAndOut => 80.0,
  };
  BarrierOption {
    strike: 100.0,
    barrier,
    rebate,
    barrier_type,
    option_type,
  }
}

fn lookback(
  strike: Option<f64>,
  running_min: Option<f64>,
  running_max: Option<f64>,
  option_type: OptionType,
) -> LookbackOption {
  LookbackOption {
    strike,
    running_min,
    running_max,
    option_type,
  }
}

#[test]
fn an_exotic_pricer_holds_the_model_the_contract_holds_the_terms_and_the_query_travels() {
  let tau = 1.0;
  let vanilla = BsmPricer::new(SIGMA, BsmCoc::Merton1973);
  let pricer = BarrierPricer::new(SIGMA);
  let up_out = barrier(BarrierType::UpAndOut, OptionType::Call, 0.0);
  let up_in = BarrierOption {
    barrier_type: BarrierType::UpAndIn,
    ..up_out
  };
  let out_plus_in = pricer.price(&up_out, S, R, Q, tau) + pricer.price(&up_in, S, R, Q, tau);
  let call = vanilla.price_call(S, 100.0, R, Q, tau);
  assert!((out_plus_in - call).abs() < 1e-10, "{out_plus_in}");
  assert!(pricer.price(&up_out, 125.0, R, Q, tau) < pricer.price(&up_out, S, R, Q, tau));

  let floating = lookback(None, Some(S), None, OptionType::Call);
  let at_the_low = LookbackPricer::new(SIGMA).price(&floating, S, R, Q, tau);
  assert!(at_the_low > vanilla.price_call(S, S, R, Q, tau));
  let chooser = ChooserOption {
    strike: 100.0,
    choice_time: 0.5,
  };
  let chooser = ChooserPricer::new(SIGMA).price_simple(&chooser, S, R, Q, tau);
  assert!(chooser > call.max(vanilla.price_put(S, 100.0, R, Q, tau)));
}

#[test]
fn the_barrier_prices_keep_their_values() {
  use BarrierType::*;
  use OptionType::*;
  let pricer = BarrierPricer::new(SIGMA);
  for (barrier_type, option_type, plain, with_rebate) in [
    (UpAndIn, Put, 0.14288692243022005, 2.162078727572185),
    (UpAndOut, Put, 8.083950125023785, 8.93788312309951),
    (DownAndIn, Call, 0.3854876045521558, 2.1726177751007674),
    (DownAndIn, Put, 7.055231729522353, 8.842361900070964),
    (DownAndOut, Call, 10.738274323505967, 11.83254742187439),
    (DownAndOut, Put, 1.1716053179316517, 2.2658784163000742),
  ] {
    let tag = format!("{barrier_type:?} {option_type:?}");
    let plain_option = barrier(barrier_type, option_type, 0.0);
    assert_eq!(pricer.price(&plain_option, S, R, Q, 1.0), plain, "{tag}");
    let rebate_option = barrier(barrier_type, option_type, 3.0);
    assert_eq!(
      pricer.price(&rebate_option, S, R, Q, 1.0),
      with_rebate,
      "{tag}"
    );
  }
}

/// QuantLib 1.43's `AnalyticBarrierEngine` at the same queries, on either side of the strike.
#[test]
fn the_barrier_prices_match_quantlib_on_either_side_of_the_strike() {
  use BarrierType::*;
  use OptionType::*;
  let pricer = BarrierPricer::new(SIGMA);
  for (barrier_type, option_type, s, h, rebate, quantlib) in [
    (UpAndIn, Call, 100.0, 130.0, 0.0, 8.99025449535496),
    (UpAndIn, Call, 100.0, 130.0, 3.0, 11.009446300496926),
    (UpAndIn, Put, 100.0, 130.0, 0.0, 0.14288692243021436),
    (UpAndIn, Put, 100.0, 130.0, 3.0, 2.162078727572179),
    (UpAndOut, Call, 100.0, 130.0, 0.0, 2.133507432703177),
    (UpAndOut, Call, 100.0, 130.0, 3.0, 2.987440430778903),
    (UpAndOut, Put, 100.0, 130.0, 0.0, 8.083950125023785),
    (UpAndOut, Put, 100.0, 130.0, 3.0, 8.93788312309951),
    (DownAndIn, Call, 100.0, 80.0, 0.0, 0.3854876045521596),
    (DownAndIn, Call, 100.0, 80.0, 3.0, 2.172617775100772),
    (DownAndIn, Put, 100.0, 80.0, 0.0, 7.055231729522349),
    (DownAndIn, Put, 100.0, 80.0, 3.0, 8.842361900070962),
    (DownAndOut, Call, 100.0, 80.0, 0.0, 10.73827432350598),
    (DownAndOut, Call, 100.0, 80.0, 3.0, 11.832547421874402),
    (DownAndOut, Put, 100.0, 80.0, 0.0, 1.1716053179316501),
    (DownAndOut, Put, 100.0, 80.0, 3.0, 2.2658784163000725),
    (UpAndIn, Call, 85.0, 90.0, 0.0, 4.182205922933827),
    (UpAndIn, Call, 85.0, 90.0, 3.0, 4.700960551162889),
    (UpAndIn, Put, 85.0, 90.0, 0.0, 10.186397171875639),
    (UpAndIn, Put, 85.0, 90.0, 3.0, 10.7051518001047),
    (UpAndOut, Call, 85.0, 90.0, 0.0, 0.0),
    (UpAndOut, Call, 85.0, 90.0, 3.0, 2.4346478236860243),
    (UpAndOut, Put, 85.0, 90.0, 0.0, 5.801863970055376),
    (UpAndOut, Put, 85.0, 90.0, 3.0, 8.2365117937414),
    (DownAndIn, Call, 120.0, 110.0, 0.0, 11.599214600577708),
    (DownAndIn, Call, 120.0, 110.0, 3.0, 12.372360115828785),
    (DownAndIn, Put, 120.0, 110.0, 0.0, 2.898197848493051),
    (DownAndIn, Put, 120.0, 110.0, 3.0, 3.6713433637441284),
    (DownAndOut, Call, 120.0, 110.0, 0.0, 13.799881594654599),
    (DownAndOut, Call, 120.0, 110.0, 3.0, 15.961366390480972),
    (DownAndOut, Put, 120.0, 110.0, 0.0, 0.0),
    (DownAndOut, Put, 120.0, 110.0, 3.0, 2.1614847958263743),
  ] {
    let option = BarrierOption {
      strike: 100.0,
      barrier: h,
      rebate,
      barrier_type,
      option_type,
    };
    let price = pricer.price(&option, s, R, Q, 1.0);
    let tag = format!("{barrier_type:?} {option_type:?} s={s} h={h} rebate={rebate}");
    assert!(
      (price - quantlib).abs() <= 1e-12 * quantlib.abs().max(1.0),
      "{tag}: {price}"
    );
  }
}

#[test]
fn the_double_barrier_keeps_its_values_and_holds_its_series_length() {
  let pricer = DoubleBarrierPricer::new(SIGMA);
  assert_eq!(pricer.terms, 5);
  assert_eq!(pricer.price(&CORRIDOR, 90.0, R, Q, 1.0), 5.52676147834582);
  assert_eq!(pricer.price(&CORRIDOR, S, R, Q, 1.0), 10.416829363785439);
  assert_eq!(
    pricer.price(&CORRIDOR, 110.0, R, Q, 0.25),
    11.973631651189004
  );
  let put = DoubleBarrierOption {
    option_type: OptionType::Put,
    ..CORRIDOR
  };
  assert_eq!(pricer.price(&put, 90.0, R, Q, 1.0), 12.946402490967737);
  assert_eq!(pricer.price(&put, S, R, Q, 0.25), 2.419613886597333);
  let fifty = pricer.with_terms(50);
  assert_eq!(fifty.price(&CORRIDOR, S, R, Q, 1.0), 10.408603719549937);
  let one = pricer.with_terms(0);
  assert_eq!(one.price(&CORRIDOR, S, R, Q, 1.0), 10.880443789902783);
}

#[test]
fn one_lookback_pricer_prices_both_contracts() {
  use OptionType::*;
  let pricer = LookbackPricer::new(SIGMA);
  for (option, pinned) in [
    (lookback(None, None, None, Call), 19.29808796226954),
    (lookback(None, Some(90.0), None, Call), 20.83910394711078),
    (lookback(None, None, None, Put), 19.418793165628067),
    (lookback(None, None, Some(110.0), Put), 20.878121871030842),
    (lookback(Some(90.0), None, None, Call), 31.828012291239318),
    (
      lookback(Some(105.0), None, Some(110.0), Call),
      19.018899629131397,
    ),
    (lookback(Some(110.0), None, None, Put), 25.91345732667257),
    (
      lookback(Some(95.0), Some(90.0), None, Put),
      13.18603194400309,
    ),
  ] {
    assert_eq!(pricer.price(&option, S, R, Q, 1.0), pinned, "{option:?}");
  }
  let driftless = lookback(None, None, None, Call);
  assert_eq!(pricer.price(&driftless, S, R, R, 1.0), 12.730787441362034);
}

#[test]
fn one_chooser_pricer_prices_the_three_contracts() {
  let haug = ChooserOption {
    strike: 50.0,
    choice_time: 0.25,
  };
  let haug_price = ChooserPricer::new(0.25).price_simple(&haug, 50.0, 0.08, 0.0, 0.5);
  assert_eq!(haug_price, 6.10707749816234);
  let pricer = ChooserPricer::new(0.3);
  let simple = ChooserOption {
    strike: 95.0,
    choice_time: 0.4,
  };
  assert_eq!(
    pricer.price_simple(&simple, S, R, Q, 1.0),
    19.398120558726195
  );
  let complex = ComplexChooserOption {
    call_strike: 105.0,
    call_tau: 1.0,
    put_strike: 95.0,
    put_tau: 0.75,
    choice_time: 0.4,
  };
  assert_eq!(pricer.price_complex(&complex, S, R, Q), 15.285108413576715);
  let same = ComplexChooserOption {
    call_strike: 95.0,
    put_tau: 1.0,
    ..complex
  };
  assert_eq!(pricer.price_complex(&same, S, R, Q), 19.398120558726188);
  let forward_start = ForwardStartOption {
    moneyness: 1.1,
    start_time: 0.25,
    option_type: OptionType::Call,
  };
  let pricer = ChooserPricer::new(SIGMA);
  assert_eq!(
    pricer.price_forward_start(&forward_start, S, R, Q, 1.0),
    5.5564185611989085
  );
  let forward_start = ForwardStartOption {
    option_type: OptionType::Put,
    ..forward_start
  };
  assert_eq!(
    pricer.price_forward_start(&forward_start, S, R, Q, 1.0),
    12.959502437827553
  );
}

#[test]
fn the_compound_and_cliquet_prices_keep_their_values() {
  let pricer = CompoundPricer::new(SIGMA);
  for (compound_type, pinned) in [
    (CompoundType::CallOnCall, 6.5922904220493415),
    (CompoundType::CallOnPut, 3.856686630328778),
    (CompoundType::PutOnCall, 0.40641749646062775),
    (CompoundType::PutOnPut, 0.5677385853441723),
  ] {
    let option = CompoundOption {
      outer_strike: 5.0,
      outer_tau: 0.25,
      inner_strike: 100.0,
      compound_type,
    };
    let price = pricer.price(&option, S, R, Q, 1.0);
    assert_eq!(price, pinned, "{compound_type:?}");
  }

  let pricer = CliquetPricer::new(SIGMA);
  for (local_floor, local_cap, pinned) in [
    (None, None, 2.8859742821983847),
    (Some(0.0), None, 34.69196090198439),
    (None, Some(0.04), -13.683474058824492),
    (Some(-0.01), Some(0.04), 12.670379765513067),
  ] {
    let option = CliquetOption {
      notional: 100.0,
      periods: 12,
      local_floor,
      local_cap,
      global_floor: None,
      global_cap: None,
    };
    assert_eq!(pricer.price(&option, 0.04, 0.01, 1.0), pinned);
  }
}

#[test]
fn the_variance_swap_functions_take_the_query() {
  assert_eq!(
    VarianceSwapPricer::fair_strike_bsm(0.2),
    0.04000000000000001
  );
  assert_eq!(
    VarianceSwapPricer::fair_strike_heston(0.04, 1.5, 0.06, 0.5),
    0.04592977473976039
  );
  let strikes = [60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0];
  let otm = [0.02, 0.11, 0.48, 1.57, 5.2, 2.31, 0.83, 0.25, 0.07];
  assert_eq!(
    VarianceSwapPricer::fair_strike_replication(&strikes, &otm, S, R, Q, 0.5),
    0.043994906076447834
  );
  let terms = VarianceSwapTerms {
    strike_variance: 0.04,
    notional: 100_000.0,
  };
  assert!((VarianceSwapPricer::pnl(0.06, &terms) - 2_000.0).abs() < 1e-9);
}

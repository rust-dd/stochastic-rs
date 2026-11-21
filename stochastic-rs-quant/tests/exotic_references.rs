//! The analytic exotics against QuantLib 1.43's engines, and mpmath where QuantLib's is inexact.

use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::pricing::barrier::DoubleBarrierPricer;
use stochastic_rs_quant::pricing::bsm::BsmCoc;
use stochastic_rs_quant::pricing::bsm::BsmPricer;
use stochastic_rs_quant::pricing::chooser::ChooserPricer;
use stochastic_rs_quant::pricing::cliquet::CliquetPricer;
use stochastic_rs_quant::pricing::compound::CompoundPricer;
use stochastic_rs_quant::pricing::compound::CompoundType;
use stochastic_rs_quant::pricing::exotic::ChooserOption;
use stochastic_rs_quant::pricing::exotic::CliquetOption;
use stochastic_rs_quant::pricing::exotic::ComplexChooserOption;
use stochastic_rs_quant::pricing::exotic::CompoundOption;
use stochastic_rs_quant::pricing::exotic::DoubleBarrierOption;
use stochastic_rs_quant::pricing::exotic::ForwardStartOption;
use stochastic_rs_quant::pricing::exotic::LookbackOption;
use stochastic_rs_quant::pricing::heston::HestonParams;
use stochastic_rs_quant::pricing::lookback::LookbackPricer;
use stochastic_rs_quant::pricing::variance_swap::VolatilitySwapPricer;
use stochastic_rs_quant::traits::ModelPricer;

const TABLE: &str = include_str!("fixtures/exotic_references.txt");

/// Each row of one kind: its words after the kind, and its reference.
fn rows(kind: &str) -> Vec<(Vec<&'static str>, f64)> {
  let picked = TABLE
    .lines()
    .filter(|line| !line.starts_with('#'))
    .filter_map(|line| {
      let (case, reference) = line.split_once(" | ")?;
      let mut words = case.split_whitespace();
      (words.next()? == kind).then(|| (words.collect(), reference.parse().unwrap()))
    })
    .collect::<Vec<_>>();
  assert!(!picked.is_empty(), "no {kind} rows");
  picked
}

fn number(words: &[&str], i: usize) -> f64 {
  words[i].parse().unwrap()
}

fn option_type(word: &str) -> OptionType {
  if word == "call" {
    OptionType::Call
  } else {
    OptionType::Put
  }
}

/// Within `tolerance` of `max(|reference|, 1)`; each test's doc states its reference's own error.
fn assert_close(got: f64, reference: f64, tolerance: f64, case: &[&str]) {
  let close = (got - reference).abs() <= tolerance * reference.abs().max(1.0);
  assert!(close, "{case:?}: {got} against {reference}");
}

fn double_barrier(words: &[&str]) -> (f64, f64) {
  let x = |i| number(words, i);
  let option = DoubleBarrierOption {
    strike: x(2),
    lower: x(3),
    upper: x(4),
    option_type: option_type(words[0]),
  };
  let knock_out = DoubleBarrierPricer::new(x(7)).price(&option, x(1), x(5), x(6), x(8));
  let vanilla = BsmPricer::new(x(7), BsmCoc::Merton1973);
  let call_or_put = vanilla.price_option(x(1), x(2), x(5), x(6), x(8), option.option_type);
  (knock_out, call_or_put - knock_out)
}

/// QuantLib's knock-out (series 20) is within 2.1e-12 of the 40-digit series here, so 1e-11; its
/// knock-in is the vanilla less the knock-out, the parity the crate's pair keeps.
#[test]
fn the_double_barrier_matches_quantlib_and_its_knock_in_by_parity() {
  for (words, quantlib) in rows("dbko") {
    assert_close(double_barrier(&words).0, quantlib, 1e-11, &words);
  }
  for (words, quantlib) in rows("dbki") {
    assert_close(double_barrier(&words).1, quantlib, 1e-11, &words);
  }
}

/// A strike outside the corridor, and `sigma = 0.005` against a carry of 0.08, where QuantLib's
/// powers overflow and it answers 0; the reference is the series in mpmath.
#[test]
fn the_double_barrier_matches_the_series_where_quantlib_does_not_reach() {
  for (words, mpmath) in rows("dbx") {
    assert_close(double_barrier(&words).0, mpmath, 1e-12, &words);
  }
}

/// `sigma` at 1e-9 and 1e-10 with the forward within `σ√τ` of a bound, against the series in mpmath
/// at 80 digits; one ulp of the spot moves these prices by up to `2.4e-16/σ`, so twice that.
#[test]
fn the_double_barrier_keeps_its_digits_as_sigma_vanishes_by_a_bound() {
  for (words, mpmath) in rows("dbs") {
    let tolerance = 4.8e-16 / number(&words, 7);
    assert_close(double_barrier(&words).0, mpmath, tolerance, &words);
  }
}

/// QuantLib's continuous engines where `r != q`, which they divide by, and at `r = q` the formula's
/// `b -> 0` limit in mpmath; the crate is within 2.6e-14 of either.
#[test]
fn the_lookbacks_match_quantlib_and_their_driftless_limit() {
  for (words, reference) in rows("lbfl") {
    let x = |i| number(&words, i);
    let call = words[0] == "call";
    let option = LookbackOption {
      strike: None,
      running_min: call.then_some(x(2)),
      running_max: (!call).then_some(x(2)),
      option_type: option_type(words[0]),
    };
    let price = LookbackPricer::new(x(5)).price(&option, x(1), x(3), x(4), x(6));
    assert_close(price, reference, 1e-13, &words);
  }
  for (words, quantlib) in rows("lbfx") {
    let x = |i| number(&words, i);
    let call = words[0] == "call";
    let option = LookbackOption {
      strike: Some(x(2)),
      running_min: (!call).then_some(x(3)),
      running_max: call.then_some(x(3)),
      option_type: option_type(words[0]),
    };
    let price = LookbackPricer::new(x(6)).price(&option, x(1), x(4), x(5), x(7));
    assert_close(price, quantlib, 1e-13, &words);
  }
}

/// QuantLib for the simple chooser; mpmath for the complex one, since QuantLib 1.43's engine takes
/// each leg's remaining time as `T - 2t` and its bivariate normal to six digits.
#[test]
fn the_choosers_match_quantlib_and_the_complex_one_mpmath() {
  for (words, quantlib) in rows("chs") {
    let x = |i| number(&words, i);
    let option = ChooserOption {
      strike: x(1),
      choice_time: x(5),
    };
    let price = ChooserPricer::new(x(4)).price_simple(&option, x(0), x(2), x(3), x(6));
    assert_close(price, quantlib, 1e-13, &words);
  }
  for (words, mpmath) in rows("chc") {
    let x = |i| number(&words, i);
    let option = ComplexChooserOption {
      call_strike: x(1),
      call_tau: x(2),
      put_strike: x(3),
      put_tau: x(4),
      choice_time: x(5),
    };
    let price = ChooserPricer::new(x(8)).price_complex(&option, x(0), x(6), x(7));
    assert_close(price, mpmath, 1e-13, &words);
  }
}

/// Geske in mpmath, since QuantLib 1.43's engine reads Drezner's (1978) six-digit bivariate normal
/// and is off by up to 2.2e-5 here; the crate is within 1.7e-14.
#[test]
fn the_compound_matches_geske_in_mpmath() {
  for (words, mpmath) in rows("cmp") {
    let x = |i| number(&words, i);
    let compound_type = match words[0] {
      "coc" => CompoundType::CallOnCall,
      "cop" => CompoundType::CallOnPut,
      "poc" => CompoundType::PutOnCall,
      _ => CompoundType::PutOnPut,
    };
    let option = CompoundOption {
      outer_strike: x(2),
      outer_tau: x(3),
      inner_strike: x(4),
      compound_type,
    };
    let price = CompoundPricer::new(x(7)).price(&option, x(1), x(5), x(6), x(8));
    assert_close(price, mpmath, 1e-13, &words);
  }
}

/// QuantLib's `ForwardEuropeanEngine`, and its `AnalyticPerformanceEngine` for one period, which a
/// cliquet floored at `f` pays as `f` plus the forward start struck at `1 + f`.
#[test]
fn the_forward_start_and_a_one_period_cliquet_match_quantlib() {
  for (words, quantlib) in rows("fws") {
    let x = |i| number(&words, i);
    let option = ForwardStartOption {
      moneyness: x(2),
      start_time: x(3),
      option_type: option_type(words[0]),
    };
    let price = ChooserPricer::new(x(6)).price_forward_start(&option, x(1), x(4), x(5), x(7));
    assert_close(price, quantlib, 1e-13, &words);
  }
  for (words, quantlib) in rows("clq1") {
    let x = |i| number(&words, i);
    let option = CliquetOption {
      notional: 1.0,
      periods: 1,
      local_floor: Some(x(0)),
      local_cap: None,
      global_floor: None,
      global_cap: None,
    };
    let price = CliquetPricer::new(x(3)).price(&option, x(1), x(2), x(4));
    assert_close(price, quantlib, 1e-13, &words);
  }
}

/// Brockhaus–Long's convexity strike with the CIR `Var(V)`, in mpmath at 40 digits.
#[test]
fn the_heston_vol_strike_matches_brockhaus_long_in_mpmath() {
  for (words, mpmath) in rows("vs") {
    let x = |i| number(&words, i);
    let params = HestonParams {
      v0: x(0),
      kappa: x(1),
      theta: x(2),
      sigma: x(3),
      rho: -0.7,
    };
    let strike = VolatilitySwapPricer::fair_strike_heston(&params, x(4));
    assert_close(strike, mpmath, 1e-13, &words);
  }
}

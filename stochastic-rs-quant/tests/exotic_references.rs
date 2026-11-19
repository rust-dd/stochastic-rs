//! The analytic exotics against QuantLib 1.43's engines, and mpmath where QuantLib's is inexact.

use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::pricing::barrier::DoubleBarrierPricer;
use stochastic_rs_quant::pricing::bsm::BsmCoc;
use stochastic_rs_quant::pricing::bsm::BsmPricer;
use stochastic_rs_quant::pricing::exotic::DoubleBarrierOption;
use stochastic_rs_quant::pricing::exotic::LookbackOption;
use stochastic_rs_quant::pricing::lookback::LookbackPricer;
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

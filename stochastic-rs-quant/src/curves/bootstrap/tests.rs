//! The single-curve bootstrap on its smallest inputs.

use super::*;
use crate::calendar::Frequency;

#[test]
fn a_single_deposit_is_one_pillar() {
  let inst: Vec<BootstrapInstrument<f64>> = vec![BootstrapInstrument::Deposit {
    maturity: 1.0,
    rate: 0.05,
  }];
  let curve = bootstrap(BootstrapSpec::new(
    &inst,
    InterpolationMethod::LinearOnZeroRates,
  ))
  .unwrap();
  assert_eq!(curve.len(), 1);
  assert_eq!(curve.discount_factor(1.0), 1.0 / 1.05);
}

/// A zero par rate is a root at one, on the edge of `(0, 1)`: the node is `1` exactly.
#[test]
fn a_zero_par_rate_prices_its_node_at_one() {
  let inst: Vec<BootstrapInstrument<f64>> = vec![
    BootstrapInstrument::Deposit {
      maturity: 0.5,
      rate: 0.04,
    },
    BootstrapInstrument::Swap {
      maturity: 1.0,
      rate: 0.0,
      frequency: Frequency::SemiAnnual,
    },
  ];
  let spec = BootstrapSpec::new(&inst, InterpolationMethod::LinearOnZeroRates);
  assert_eq!(bootstrap(spec).unwrap().discount_factor(1.0), 1.0);
}

const LOG_LINEAR: InterpolationMethod = InterpolationMethod::LogLinearOnDiscountFactors;
const LINEAR_ZERO: InterpolationMethod = InterpolationMethod::LinearOnZeroRates;

fn six_month_deposit(rate: f64) -> BootstrapInstrument<f64> {
  BootstrapInstrument::Deposit {
    maturity: 0.5,
    rate,
  }
}

fn annual_swap(maturity: f64, rate: f64) -> BootstrapInstrument<f64> {
  BootstrapInstrument::Swap {
    maturity,
    rate,
    frequency: Frequency::Annual,
  }
}

fn built(inst: &[BootstrapInstrument<f64>], method: InterpolationMethod) -> DiscountCurve<f64> {
  bootstrap(BootstrapSpec::new(inst, method)).unwrap()
}

/// The node by mpmath on the same interpolation, a 6M deposit and one annual swap; the deposit's
/// flat zero rate would miss the par rate by 1 to 118 bp.
#[test]
fn a_swap_with_payments_past_the_last_pillar_reprices_on_the_built_curve() {
  let rows = [
    (0.03, 5.0, 0.04, LOG_LINEAR, 0.8215788082556382),
    (0.03, 5.0, 0.04, LINEAR_ZERO, 0.8205191839738211),
    (0.03, 10.0, 0.04, LOG_LINEAR, 0.6749343861014856),
    (0.03, 10.0, 0.04, LINEAR_ZERO, 0.6708761735232842),
    (0.01, 10.0, 0.05, LOG_LINEAR, 0.610934988718196),
    (0.01, 10.0, 0.05, LINEAR_ZERO, 0.591267015972375),
    (0.05, 5.0, 0.02, LOG_LINEAR, 0.9063259439727934),
    (0.05, 5.0, 0.02, LINEAR_ZERO, 0.9080919453093967),
    (0.03, 2.0, 0.04, LOG_LINEAR, 0.9244411548013117),
    (0.03, 2.0, 0.04, LINEAR_ZERO, 0.9243242075920116),
    (-0.002, 5.0, -0.001, LOG_LINEAR, 1.0050161522500663),
    (-0.002, 5.0, -0.001, LINEAR_ZERO, 1.0050195048883834),
  ];
  for (deposit, maturity, rate, method, node) in rows {
    let curve = built(
      &[six_month_deposit(deposit), annual_swap(maturity, rate)],
      method,
    );
    let (got, par) = (
      curve.discount_factor(maturity),
      curve.par_rate(maturity, Frequency::Annual),
    );
    let row = format!("{deposit}, {maturity}y at {rate}, {method:?}");
    assert!((got - node).abs() < 1e-14, "{row}: D = {got}");
    assert!((par - rate).abs() < 1e-14, "{row}: par = {par}");
  }
}

#[test]
fn every_interpolation_reprices_a_swap_past_the_last_pillar() {
  let methods = [
    LOG_LINEAR,
    LINEAR_ZERO,
    InterpolationMethod::CubicSplineOnZeroRates,
    InterpolationMethod::MonotoneConvex,
  ];
  for method in methods {
    let inst = [
      six_month_deposit(0.03),
      annual_swap(1.0, 0.032),
      annual_swap(4.0, 0.04),
    ];
    let par = built(&inst, method).par_rate(4.0, Frequency::Annual);
    assert!((par - 0.04).abs() < 1e-14, "{method:?}: par = {par}");
  }
}

#[test]
fn a_schedule_swap_past_the_last_pillar_takes_the_same_node() {
  let schedule = BootstrapInstrument::SwapWithSchedule {
    rate: 0.04,
    fixed_times: vec![1.0, 2.0, 3.0, 4.0, 5.0],
    float_times: vec![5.0],
  };
  for method in [LOG_LINEAR, LINEAR_ZERO] {
    let by_schedule = built(&[six_month_deposit(0.03), schedule.clone()], method);
    let by_frequency = built(&[six_month_deposit(0.03), annual_swap(5.0, 0.04)], method);
    let (a, b) = (
      by_schedule.discount_factor(5.0),
      by_frequency.discount_factor(5.0),
    );
    assert!((a - b).abs() < 1e-15, "{method:?}: {a} against {b}");
    let par = by_schedule.par_rate(5.0, Frequency::Annual);
    assert!((par - 0.04).abs() < 1e-14, "{method:?}: par = {par}");
  }
}

/// The node by mpmath: `D(0.5) / 1.02²` log-linearly, a root on the zero rates.
#[test]
fn an_fra_or_a_future_starting_past_the_last_pillar_reprices_on_the_built_curve() {
  let fra = BootstrapInstrument::Fra {
    start: 1.0,
    end: 1.5,
    rate: 0.04,
  };
  let future = BootstrapInstrument::Future {
    start: 1.0,
    end: 1.5,
    price: 96.0,
    sigma: 0.0,
  };
  for (method, node) in [
    (LOG_LINEAR, 0.946964316490626),
    (LINEAR_ZERO, 0.949293875524884),
  ] {
    for quote in [fra.clone(), future.clone()] {
      let curve = built(&[six_month_deposit(0.03), quote], method);
      let (got, forward) = (
        curve.discount_factor(1.5),
        curve.simple_forward_rate(1.0, 1.5),
      );
      assert!((got - node).abs() < 1e-14, "{method:?}: D = {got}");
      assert!((forward - 0.04).abs() < 1e-13, "{method:?}: F = {forward}");
    }
  }
}

#[test]
fn a_ladder_with_a_pillar_at_every_payment_keeps_the_direct_formula() {
  let inst = [
    BootstrapInstrument::Deposit {
      maturity: 1.0,
      rate: 0.03,
    },
    BootstrapInstrument::Fra {
      start: 1.0,
      end: 1.5,
      rate: 0.033,
    },
    annual_swap(2.0, 0.035),
    annual_swap(3.0, 0.04),
  ];
  let d1 = 1.0 / 1.03;
  let d2 = (1.0 - 0.035 * d1) / (1.0 + 0.035);
  let d3 = (1.0 - 0.04 * (d1 + d2)) / (1.0 + 0.04);
  for method in [LOG_LINEAR, LINEAR_ZERO, InterpolationMethod::MonotoneConvex] {
    let curve = built(&inst, method);
    let nodes: Vec<f64> = curve.points().iter().map(|p| p.discount_factor).collect();
    assert_eq!(nodes[..3], [d1, d1 / (1.0 + 0.033 * 0.5), d2]);
    // `D(2)` is read at the last pillar through `exp` and `ln`.
    assert!(
      (nodes[3] - d3).abs() <= 1e-15,
      "{method:?}: {} against {d3}",
      nodes[3]
    );
  }
}

#[test]
fn a_segment_reads_what_the_curve_reads_between_its_pillars() {
  let p0 = CurvePoint {
    time: 0.75,
    discount_factor: 0.981,
  };
  for method in [LOG_LINEAR, LINEAR_ZERO] {
    for x in [0.893, 0.91, 1.02] {
      let p1 = CurvePoint {
        time: 3.5,
        discount_factor: x,
      };
      let start = CurvePoint {
        discount_factor: 0.5,
        ..p1
      };
      let segment = Segment::new(&p0, &start, method).unwrap().ending_at(x);
      for k in 1..100 {
        let t = 0.75 + 2.75 * k as f64 / 100.0;
        let curve = interpolate_discount_factor(&[p0, p1], t, method);
        assert_eq!(segment.read(t).0, curve, "{method:?}, x = {x}, t = {t}");
      }
    }
  }
}

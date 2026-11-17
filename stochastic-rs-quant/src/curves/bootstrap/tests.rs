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

#[test]
fn the_iterative_solve_bisects_where_the_direct_root_leaves_the_unit_interval() {
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
  let direct = bootstrap(spec).unwrap().discount_factor(1.0);
  let bisected = bootstrap(spec.iterative(IterativeOptions::default()))
    .unwrap()
    .discount_factor(1.0);
  assert_eq!(
    direct, 1.0,
    "a zero par rate is a root on the bracket's edge"
  );
  assert!(bisected < 1.0 && 1.0 - bisected < 2e-12, "{bisected:e}");
}

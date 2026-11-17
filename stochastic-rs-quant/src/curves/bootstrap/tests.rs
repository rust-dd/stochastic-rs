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
fn the_iterative_solve_matches_the_direct_one() {
  let inst: Vec<BootstrapInstrument<f64>> = vec![
    BootstrapInstrument::Deposit {
      maturity: 0.25,
      rate: 0.04,
    },
    BootstrapInstrument::Swap {
      maturity: 1.0,
      rate: 0.045,
      frequency: Frequency::SemiAnnual,
    },
  ];
  let spec = BootstrapSpec::new(&inst, InterpolationMethod::LinearOnZeroRates);
  let options = IterativeOptions {
    tolerance: 1e-10,
    max_iterations: 50,
  };
  let direct = bootstrap(spec).unwrap();
  let iterative = bootstrap(spec.iterative(options)).unwrap();
  assert_eq!(iterative.len(), 2);
  assert_eq!(iterative.discount_factor(1.0), direct.discount_factor(1.0));
}

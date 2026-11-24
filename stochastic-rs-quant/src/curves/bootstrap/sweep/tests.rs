//! The sweep against the first pass on the review's fuzz ladders: 2 to 30 bp of noise over
//! 1y–30y and out to 50y pillars. Whatever the first pass builds, the sweep builds and reprices.

use super::super::bootstrap;
use super::super::first_pass;
use crate::calendar::Frequency;
use crate::curves::bootstrap::BootstrapSpec;
use crate::curves::discount_curve::DiscountCurve;
use crate::curves::types::BootstrapInstrument;
use crate::curves::types::InterpolationMethod;
use crate::traits::RealExt;

const GLOBAL: [InterpolationMethod; 2] = [
  InterpolationMethod::CubicSplineOnZeroRates,
  InterpolationMethod::MonotoneConvex,
];

/// The review's generator, seeded as it was, so case `i` is its ladder `i`.
struct Lcg(u64);

impl Lcg {
  fn uniform(&mut self) -> f64 {
    self.0 = self
      .0
      .wrapping_mul(6_364_136_223_846_793_005)
      .wrapping_add(1_442_695_040_888_963_407);
    (self.0 >> 11) as f64 / (1_u64 << 53) as f64
  }
}

/// Case `case`'s ladder; `None` for the uneven pillars and the 100 bp noise, which draw their
/// numbers all the same.
fn ladder(rng: &mut Lcg, case: usize) -> Option<Vec<BootstrapInstrument<f64>>> {
  let kind = case % 6;
  let frequency = if rng.uniform() < 0.5 {
    Frequency::SemiAnnual
  } else {
    Frequency::Annual
  };
  let maturities = match kind {
    0 => vec![2.0, 5.0, 10.0, 30.0],
    1 => (1..=30).map(f64::from).collect(),
    2 => vec![1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0],
    3 => vec![
      1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0,
    ],
    4 => vec![1.0, 1.5, 2.0, 29.5, 30.0],
    _ => (1..=10).map(|i| f64::from(i) * 5.0).collect::<Vec<_>>(),
  };
  let level = 0.08 * rng.uniform();
  let noise = [0.0002, 0.001, 0.003, 0.01][(rng.uniform() * 4.0) as usize % 4];
  let deposit = level + noise * (rng.uniform() - 0.5);
  let mut quotes = vec![BootstrapInstrument::Deposit {
    maturity: 0.5,
    rate: deposit,
  }];
  let mut rate = level;
  for maturity in maturities {
    rate += noise * (rng.uniform() - 0.5) * 2.0;
    quotes.push(BootstrapInstrument::Swap {
      maturity,
      rate,
      frequency,
    });
  }
  (kind != 4 && noise < 0.01).then_some(quotes)
}

/// The quote `curve` implies for `quote`, by the quote's own definition.
fn implied(curve: &DiscountCurve<f64>, quote: &BootstrapInstrument<f64>) -> (f64, f64) {
  match *quote {
    BootstrapInstrument::Deposit { maturity, rate } => (
      (1.0 / curve.discount_factor(maturity) - 1.0) / maturity,
      rate,
    ),
    BootstrapInstrument::Swap {
      maturity,
      rate,
      frequency,
    } => (curve.par_rate(maturity, frequency), rate),
    _ => unreachable!("the fuzz ladders hold deposits and swaps"),
  }
}

#[test]
fn the_sweep_builds_and_reprices_every_ladder_the_first_pass_builds() {
  let mut failures = Vec::new();
  for method in GLOBAL {
    let mut rng = Lcg(7);
    for case in 0..1800 {
      let Some(quotes) = ladder(&mut rng, case) else {
        continue;
      };
      let spec = BootstrapSpec::new(&quotes, method);
      if first_pass(spec, None).is_err() {
        continue;
      }
      match bootstrap(spec) {
        Ok(curve) => {
          for quote in &quotes {
            let (implied, rate) = implied(&curve, quote);
            let miss = (implied - rate).abs();
            if miss.is_nan() || miss >= 1e-13 {
              failures.push(format!(
                "{method:?} case {case}: {quote:?} implies {implied:e}"
              ));
            }
          }
        }
        Err(error) => failures.push(format!("{method:?} case {case}: {error}")),
      }
    }
  }
  assert!(
    failures.is_empty(),
    "{} failures: {:#?}",
    failures.len(),
    &failures[..failures.len().min(8)]
  );
}

/// 50y quarterly and 20y monthly ladders: 200 and 240 coupons a residual, the most rounding a
/// re-solve carries.
#[test]
fn long_quarterly_and_monthly_ladders_settle_and_reprice() {
  let mut rng = Lcg(11);
  for method in GLOBAL {
    for (frequency, maturities) in [
      (
        Frequency::Quarterly,
        vec![1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0],
      ),
      (
        Frequency::Monthly,
        vec![1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0],
      ),
    ] {
      for _ in 0..4 {
        let mut rate = 0.08 * rng.uniform();
        let mut quotes = vec![BootstrapInstrument::Deposit {
          maturity: 0.25,
          rate,
        }];
        for &maturity in &maturities {
          rate += 0.003 * (rng.uniform() - 0.5) * 2.0;
          quotes.push(BootstrapInstrument::Swap {
            maturity,
            rate,
            frequency,
          });
        }
        let curve = bootstrap(BootstrapSpec::new(&quotes, method)).unwrap();
        for quote in &quotes {
          let (implied, rate) = implied(&curve, quote);
          assert!(
            (implied - rate).abs() < 1e-13,
            "{method:?} {quote:?}: {implied:e}"
          );
        }
      }
    }
  }
}

/// The same rule in `f32`, where the node solve and the sweep stop at `f32`'s own epsilon.
#[test]
fn an_f32_ladder_settles_wherever_its_first_pass_builds() {
  let mut worst = 0.0_f32;
  let mut built = 0;
  for method in GLOBAL {
    let mut rng = Lcg(7);
    for case in 0..240 {
      let Some(quotes) = ladder(&mut rng, case) else {
        continue;
      };
      let quotes = quotes
        .iter()
        .map(|quote| match *quote {
          BootstrapInstrument::Deposit { maturity, rate } => BootstrapInstrument::Deposit {
            maturity: maturity as f32,
            rate: rate as f32,
          },
          BootstrapInstrument::Swap {
            maturity,
            rate,
            frequency,
          } => BootstrapInstrument::Swap {
            maturity: maturity as f32,
            rate: rate as f32,
            frequency,
          },
          _ => unreachable!("the fuzz ladders hold deposits and swaps"),
        })
        .collect::<Vec<_>>();
      let spec = BootstrapSpec::new(&quotes, method);
      if first_pass(spec, None).is_err() {
        continue;
      }
      let curve = bootstrap(spec).unwrap_or_else(|e| panic!("{method:?} case {case}: {e}"));
      built += 1;
      for quote in &quotes {
        let (implied, rate) = match *quote {
          BootstrapInstrument::Deposit { maturity, rate } => (
            (1.0 / curve.discount_factor(maturity) - 1.0) / maturity,
            rate,
          ),
          BootstrapInstrument::Swap {
            maturity,
            rate,
            frequency,
          } => (curve.par_rate(maturity, frequency), rate),
          _ => unreachable!("the fuzz ladders hold deposits and swaps"),
        };
        worst = worst.max_or_nan((implied - rate).abs());
      }
    }
  }
  println!("f32: {built} curves, worst miss {worst:e}");
  assert!(worst < 1e-6, "{worst:e}");
}

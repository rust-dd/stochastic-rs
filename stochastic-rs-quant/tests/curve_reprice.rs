//! Every quote a bootstrap takes reprices on the curve it returns, under every interpolation: each
//! rate re-derived here from `discount_factor` by the instrument's own definition.

use stochastic_rs_quant::calendar::Frequency;
use stochastic_rs_quant::curves::BootstrapInstrument;
use stochastic_rs_quant::curves::BootstrapSpec;
use stochastic_rs_quant::curves::CurvePoint;
use stochastic_rs_quant::curves::DiscountCurve;
use stochastic_rs_quant::curves::InterpolationMethod;
use stochastic_rs_quant::curves::bootstrap;

const METHODS: [InterpolationMethod; 4] = [
  InterpolationMethod::LogLinearOnDiscountFactors,
  InterpolationMethod::LinearOnZeroRates,
  InterpolationMethod::CubicSplineOnZeroRates,
  InterpolationMethod::MonotoneConvex,
];

fn swap(maturity: f64, rate: f64, frequency: Frequency) -> BootstrapInstrument<f64> {
  BootstrapInstrument::Swap {
    maturity,
    rate,
    frequency,
  }
}

/// The `market` bench's seven quotes at their day-count year fractions.
fn bench_ladder() -> Vec<BootstrapInstrument<f64>> {
  let semi = Frequency::SemiAnnual;
  vec![
    BootstrapInstrument::Deposit {
      maturity: 31.0 / 360.0,
      rate: 0.03,
    },
    BootstrapInstrument::Fra {
      start: 31.0 / 360.0,
      end: 181.0 / 360.0,
      rate: 0.032,
    },
    BootstrapInstrument::Fra {
      start: 181.0 / 360.0,
      end: 365.0 / 360.0,
      rate: 0.035,
    },
    swap(730.0 / 365.0, 0.038, semi),
    swap(1826.0 / 365.0, 0.04, semi),
    swap(3652.0 / 365.0, 0.042, semi),
    swap(10957.0 / 365.0, 0.043, semi),
  ]
}

fn steep_ladder() -> Vec<BootstrapInstrument<f64>> {
  let semi = Frequency::SemiAnnual;
  vec![
    BootstrapInstrument::Deposit {
      maturity: 0.5,
      rate: 0.01,
    },
    swap(2.0, 0.03, semi),
    swap(10.0, 0.05, semi),
    swap(30.0, 0.055, semi),
  ]
}

fn futures_ladder() -> Vec<BootstrapInstrument<f64>> {
  let future = |start: f64, price: f64| BootstrapInstrument::Future {
    start,
    end: start + 0.25,
    price,
    sigma: 0.01,
  };
  vec![
    BootstrapInstrument::Deposit {
      maturity: 0.25,
      rate: 0.03,
    },
    future(0.5, 96.8),
    future(1.0, 96.5),
    swap(3.0, 0.036, Frequency::Annual),
  ]
}

/// The quote's rate, a future's as `(100 − price) / 100`.
fn quoted(quote: &BootstrapInstrument<f64>) -> f64 {
  match quote {
    BootstrapInstrument::Deposit { rate, .. }
    | BootstrapInstrument::Fra { rate, .. }
    | BootstrapInstrument::Swap { rate, .. }
    | BootstrapInstrument::SwapWithSchedule { rate, .. } => *rate,
    BootstrapInstrument::Future { price, .. } => (100.0 - price) / 100.0,
    _ => unreachable!("{quote:?}"),
  }
}

/// The rate `curve` implies for `quote`; a swap pays `1/f` at each whole period and at maturity.
fn implied(curve: &DiscountCurve<f64>, quote: &BootstrapInstrument<f64>) -> f64 {
  let d = |t: f64| curve.discount_factor(t);
  let forward = |start: f64, end: f64| (d(start) / d(end) - 1.0) / (end - start);
  match quote {
    BootstrapInstrument::Deposit { maturity, .. } => (1.0 / d(*maturity) - 1.0) / maturity,
    BootstrapInstrument::Fra { start, end, .. } => forward(*start, *end),
    BootstrapInstrument::Future {
      start, end, sigma, ..
    } => forward(*start, *end) + 0.5 * sigma * sigma * start * end,
    BootstrapInstrument::Swap {
      maturity,
      frequency,
      ..
    } => {
      let delta = 1.0 / f64::from(frequency.periods_per_year());
      let payments = (maturity / delta).round() as usize;
      let coupons: f64 = (1..payments).map(|i| delta * d(i as f64 * delta)).sum();
      (1.0 - d(*maturity)) / (coupons + delta * d(*maturity))
    }
    BootstrapInstrument::SwapWithSchedule { fixed_times, .. } => {
      let (mut annuity, mut previous) = (0.0, 0.0);
      for &t in fixed_times {
        annuity += (t - previous) * d(t);
        previous = t;
      }
      (1.0 - d(previous)) / annuity
    }
    _ => unreachable!("{quote:?}"),
  }
}

/// The largest `|implied − quote|` over the ladder, in basis points.
fn worst_miss_bp(quotes: &[BootstrapInstrument<f64>], method: InterpolationMethod) -> f64 {
  let curve = bootstrap(BootstrapSpec::new(quotes, method)).unwrap();
  quotes
    .iter()
    .map(|quote| (implied(&curve, quote) - quoted(quote)).abs() * 1e4)
    .fold(0.0, f64::max)
}

/// Under cubic spline and monotone convex a later pillar reshapes the earlier intervals, so the
/// nodes are solved together; the local methods solve them one by one.
#[test]
fn every_quote_reprices_on_the_curve_it_built() {
  for (name, quotes) in [
    ("bench", bench_ladder()),
    ("steep", steep_ladder()),
    ("futures", futures_ladder()),
  ] {
    for method in METHODS {
      let miss = worst_miss_bp(&quotes, method);
      assert!(miss < 1e-9, "{name}, {method:?}: {miss:e} bp");
    }
  }
}

/// A 3M forecast curve 50 bp over a flat 2 % OIS curve: each swap's floating leg on the forecast
/// curve, discounted on the OIS curve, equals its fixed leg.
#[test]
fn every_forecast_swap_reprices_on_the_curve_it_built() {
  let ois = DiscountCurve::new(
    [0.25, 1.0, 5.0, 10.0]
      .map(|t: f64| CurvePoint {
        time: t,
        discount_factor: (-0.02 * t).exp(),
      })
      .to_vec(),
    InterpolationMethod::LogLinearOnDiscountFactors,
  );
  let mut quotes = vec![BootstrapInstrument::Deposit {
    maturity: 0.25,
    rate: 0.031,
  }];
  for (years, rate) in [(1_usize, 0.0262), (2, 0.0271), (3, 0.0278), (5, 0.0289)] {
    quotes.push(BootstrapInstrument::SwapWithSchedule {
      rate,
      fixed_times: (1..=years).map(|i| i as f64).collect(),
      float_times: (1..=4 * years).map(|i| 0.25 * i as f64).collect(),
    });
  }
  for method in METHODS {
    let spec = BootstrapSpec::new(&quotes, method).discounting_with(&ois);
    let forecast = bootstrap(spec).unwrap();
    for quote in &quotes[1..] {
      let BootstrapInstrument::SwapWithSchedule {
        rate,
        fixed_times,
        float_times,
      } = quote
      else {
        unreachable!()
      };
      let (mut floating, mut previous) = (0.0, 1.0);
      for &t in float_times {
        let p = forecast.discount_factor(t);
        floating += ois.discount_factor(t) * (previous / p - 1.0);
        previous = p;
      }
      let (mut annuity, mut start) = (0.0, 0.0);
      for &t in fixed_times {
        annuity += (t - start) * ois.discount_factor(t);
        start = t;
      }
      let miss = (floating / annuity - rate).abs() * 1e4;
      assert!(miss < 1e-9, "{method:?}, {quote:?}: {miss:e} bp");
    }
  }
}

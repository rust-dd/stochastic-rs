//! Comparison tests for the yield curve construction module.
//!
//! Discount factor / zero rate / forward rate identities verified analytically.
//! Nelson-Siegel values verified against the `nelson_siegel_svensson` Python library (luphord).
//! Bootstrapping values verified against CFA / AnalystPrep worked examples.
//! Par rate verified against manual derivation.

use ndarray::array;
use stochastic_rs::quant::curves::Compounding;
use stochastic_rs::quant::curves::CurvePoint;
use stochastic_rs::quant::curves::DiscountCurve;
use stochastic_rs::quant::curves::Instrument;
use stochastic_rs::quant::curves::InterpolationMethod;
use stochastic_rs::quant::curves::MultiCurve;
use stochastic_rs::quant::curves::NelsonSiegel;
use stochastic_rs::quant::curves::bootstrap;

const TOL: f64 = 1e-6;

fn approx(a: f64, b: f64, tol: f64) {
  assert!(
    (a - b).abs() < tol,
    "expected {b:.10}, got {a:.10}, diff = {:.2e}",
    (a - b).abs()
  );
}

// Compounding conversions

#[test]
fn continuous_compounding_roundtrip() {
  let rate = 0.05_f64;
  let tau = 2.0;
  let df = Compounding::Continuous.discount_factor(rate, tau);
  approx(df, (-0.10_f64).exp(), 1e-15);
  let r_back = Compounding::Continuous.zero_rate(df, tau);
  approx(r_back, rate, 1e-15);
}

#[test]
fn simple_compounding_roundtrip() {
  let rate = 0.05_f64;
  let tau = 0.25;
  let df = Compounding::Simple.discount_factor(rate, tau);
  approx(df, 1.0 / (1.0 + 0.05 * 0.25), 1e-15);
  let r_back = Compounding::Simple.zero_rate(df, tau);
  approx(r_back, rate, 1e-15);
}

#[test]
fn semiannual_compounding_roundtrip() {
  let rate = 0.06_f64;
  let tau = 3.0;
  let df = Compounding::Periodic(2).discount_factor(rate, tau);
  let expected = (1.0 + 0.03_f64).powf(-6.0);
  approx(df, expected, 1e-12);
  let r_back = Compounding::Periodic(2).zero_rate(df, tau);
  approx(r_back, rate, 1e-12);
}

// Discount factor ↔ zero rate ↔ forward rate identities

#[test]
fn discount_to_zero_rate() {
  let curve = DiscountCurve::new(
    vec![
      CurvePoint {
        time: 0.0,
        discount_factor: 1.0,
      },
      CurvePoint {
        time: 1.0,
        discount_factor: 0.95,
      },
      CurvePoint {
        time: 2.0,
        discount_factor: 0.90,
      },
    ],
    InterpolationMethod::LinearOnZeroRates,
  );
  // r(1) = -ln(0.95)/1 = 0.051293
  approx(curve.zero_rate(1.0), -0.95_f64.ln(), TOL);
  // r(2) = -ln(0.90)/2 = 0.052680
  approx(curve.zero_rate(2.0), -0.90_f64.ln() / 2.0, TOL);
}

#[test]
fn forward_rate_from_discount_factors() {
  let curve = DiscountCurve::new(
    vec![
      CurvePoint {
        time: 0.0,
        discount_factor: 1.0,
      },
      CurvePoint {
        time: 1.0,
        discount_factor: 0.95,
      },
      CurvePoint {
        time: 2.0,
        discount_factor: 0.90,
      },
    ],
    InterpolationMethod::LinearOnZeroRates,
  );
  // f(1,2) = -[ln(0.90) - ln(0.95)] / 1 = ln(0.95/0.90) = 0.054067
  let expected = (0.95_f64 / 0.90).ln();
  approx(curve.forward_rate(1.0, 2.0), expected, TOL);
}

#[test]
fn simple_forward_rate_from_discount_factors() {
  let curve = DiscountCurve::new(
    vec![
      CurvePoint {
        time: 0.0,
        discount_factor: 1.0,
      },
      CurvePoint {
        time: 0.5,
        discount_factor: 0.98,
      },
      CurvePoint {
        time: 1.0,
        discount_factor: 0.95,
      },
    ],
    InterpolationMethod::LinearOnZeroRates,
  );
  // F(0.5,1.0) = (D(0.5)/D(1.0) - 1) / 0.5 = (0.98/0.95 - 1) / 0.5
  let expected = (0.98 / 0.95 - 1.0) / 0.5;
  approx(curve.simple_forward_rate(0.5, 1.0), expected, TOL);
}

#[test]
fn zero_rate_to_discount_factor_identity() {
  // D(T) = exp(-r*T), r(T) = -ln(D)/T
  let r = 0.05_f64;
  for t in [0.25, 0.5, 1.0, 2.0, 5.0, 10.0] {
    let df = (-r * t).exp();
    let r_back = -df.ln() / t;
    approx(r_back, r, 1e-14);
  }
}

// Nelson-Siegel model
// Reference: luphord/nelson_siegel_svensson Python library

#[test]
fn nelson_siegel_zero_rates_match_python_reference() {
  let ns = NelsonSiegel::new(
    0.04201739383636799,
    -0.031829031569430594,
    -0.026797319779108236,
    1.7170972656534174,
  );

  // y(0) approaches beta0 + beta1 = 0.010188...
  // y(1) = 0.012547870204470839
  // y(2) = 0.015748555528558850
  // y(3) = 0.018979558041460460
  approx(ns.zero_rate(1.0), 0.012547870204470839, 1e-10);
  approx(ns.zero_rate(2.0), 0.015_748_555_528_558_85, 1e-10);
  approx(ns.zero_rate(3.0), 0.018_979_558_041_460_46, 1e-10);
}

#[test]
fn nelson_siegel_short_rate_is_beta0_plus_beta1() {
  let ns = NelsonSiegel::<f64>::new(0.06, -0.02, 0.01, 1.5);
  // As tau → 0: y(tau) → beta0 + beta1
  let short = ns.zero_rate(1e-8);
  approx(short, 0.06 + (-0.02), 1e-6);
}

#[test]
fn nelson_siegel_long_rate_is_beta0() {
  let ns = NelsonSiegel::<f64>::new(0.06, -0.02, 0.01, 1.5);
  // As tau → ∞: y(tau) → beta0
  let long = ns.zero_rate(1000.0);
  approx(long, 0.06, 1e-4);
}

#[test]
fn nelson_siegel_forward_rate_at_zero_is_beta0_plus_beta1() {
  let ns = NelsonSiegel::<f64>::new(0.06, -0.02, 0.01, 1.5);
  // f(0) = beta0 + beta1
  let f0 = ns.forward_rate(0.0);
  approx(f0, 0.04, 1e-10);
}

#[test]
fn nelson_siegel_discount_factor_consistency() {
  let ns = NelsonSiegel::new(
    0.04201739383636799,
    -0.031829031569430594,
    -0.026797319779108236,
    1.7170972656534174,
  );
  for tau in [0.5, 1.0, 2.0, 5.0, 10.0] {
    let r = ns.zero_rate(tau);
    let df = ns.discount_factor(tau);
    approx(df, (-r * tau).exp(), 1e-14);
  }
}

// Bootstrapping from deposits and swaps
// Reference: CFA / AnalystPrep worked examples + analytic formulas

#[test]
fn bootstrap_deposits_simple_compounding() {
  // D(T) = 1 / (1 + r * T) for deposits
  let instruments = vec![
    Instrument::Deposit {
      maturity: 1.0 / 12.0,
      rate: 0.05,
    },
    Instrument::Deposit {
      maturity: 0.25,
      rate: 0.052,
    },
    Instrument::Deposit {
      maturity: 0.50,
      rate: 0.054,
    },
    Instrument::Deposit {
      maturity: 1.00,
      rate: 0.056,
    },
  ];
  let curve = bootstrap(&instruments, InterpolationMethod::LinearOnZeroRates);

  approx(
    curve.discount_factor(1.0 / 12.0),
    1.0 / (1.0 + 0.05 / 12.0),
    TOL,
  );
  approx(curve.discount_factor(0.25), 1.0 / (1.0 + 0.052 * 0.25), TOL);
  approx(curve.discount_factor(0.50), 1.0 / (1.0 + 0.054 * 0.50), TOL);
  approx(curve.discount_factor(1.00), 1.0 / (1.0 + 0.056 * 1.00), TOL);
}

#[test]
fn bootstrap_deposit_plus_swap() {
  // D(1) from deposit, D(2) from 2Y swap
  // D(1) = 1/(1+0.056) = 0.946970
  // D(2) = (1 - 0.058 * D(1)) / (1 + 0.058) = (1 - 0.054924) / 1.058
  let instruments = vec![
    Instrument::Deposit {
      maturity: 1.0,
      rate: 0.056,
    },
    Instrument::Swap {
      maturity: 2.0,
      rate: 0.058,
      frequency: 1,
    },
  ];
  let curve = bootstrap(&instruments, InterpolationMethod::LinearOnZeroRates);

  let d1 = 1.0 / (1.0 + 0.056);
  let d2 = (1.0 - 0.058 * d1) / (1.0 + 0.058);

  approx(curve.discount_factor(1.0), d1, TOL);
  approx(curve.discount_factor(2.0), d2, TOL);
}

#[test]
fn bootstrap_fra() {
  // Deposit gives D(0.25), FRA gives D(0.50)
  // D(0.25) = 1/(1 + 0.05 * 0.25)
  // D(0.50) = D(0.25) / (1 + fra_rate * 0.25)
  let instruments = vec![
    Instrument::Deposit {
      maturity: 0.25,
      rate: 0.05,
    },
    Instrument::Fra {
      start: 0.25,
      end: 0.50,
      rate: 0.055,
    },
  ];
  let curve = bootstrap(&instruments, InterpolationMethod::LinearOnZeroRates);

  let d_3m = 1.0 / (1.0 + 0.05 * 0.25);
  let d_6m = d_3m / (1.0 + 0.055 * 0.25);

  approx(curve.discount_factor(0.25), d_3m, TOL);
  approx(curve.discount_factor(0.50), d_6m, TOL);
}

#[test]
fn bootstrap_futures_with_convexity_adjustment() {
  // Future price = 94.5, sigma = 0.01
  // Futures rate = (100 - 94.5)/100 = 0.055
  // Convexity adj = 0.5 * 0.01^2 * 0.25 * 0.50 = 0.00000625
  // FRA rate = 0.055 - 0.00000625
  let instruments = vec![
    Instrument::Deposit {
      maturity: 0.25,
      rate: 0.05,
    },
    Instrument::Future {
      start: 0.25,
      end: 0.50,
      price: 94.5,
      sigma: 0.01,
    },
  ];
  let curve = bootstrap(&instruments, InterpolationMethod::LinearOnZeroRates);

  let d_3m = 1.0 / (1.0 + 0.05 * 0.25);
  let fra_rate = 0.055 - 0.5 * 0.0001 * 0.25 * 0.50;
  let d_6m = d_3m / (1.0 + fra_rate * 0.25);

  approx(curve.discount_factor(0.50), d_6m, TOL);
}

// Par rate from discount factors

#[test]
fn par_rate_semiannual() {
  // D(0.5) = 0.975, D(1.0) = 0.950
  // par = 2 * (1 - 0.950) / (0.975 + 0.950) = 0.100 / 1.925 = 0.051948
  let curve = DiscountCurve::new(
    vec![
      CurvePoint {
        time: 0.0,
        discount_factor: 1.0,
      },
      CurvePoint {
        time: 0.5,
        discount_factor: 0.975,
      },
      CurvePoint {
        time: 1.0,
        discount_factor: 0.950,
      },
    ],
    InterpolationMethod::LinearOnZeroRates,
  );
  let par = curve.par_rate(1.0, 2);
  approx(par, 2.0 * (1.0 - 0.950) / (0.975 + 0.950), 1e-4);
}

#[test]
fn par_rate_annual() {
  // D(1) = 0.95, D(2) = 0.90
  // par = (1 - 0.90) / (0.95 + 0.90) = 0.10 / 1.85 = 0.054054
  let curve = DiscountCurve::new(
    vec![
      CurvePoint {
        time: 0.0,
        discount_factor: 1.0,
      },
      CurvePoint {
        time: 1.0,
        discount_factor: 0.95,
      },
      CurvePoint {
        time: 2.0,
        discount_factor: 0.90,
      },
    ],
    InterpolationMethod::LinearOnZeroRates,
  );
  let par = curve.par_rate(2.0, 1);
  approx(par, (1.0 - 0.90) / (0.95 + 0.90), 1e-4);
}

// Interpolation methods

#[test]
fn log_linear_preserves_constant_forward() {
  // Log-linear on DF implies piecewise constant forward rates.
  // If r is constant, D(t) = exp(-r*t), and log-linear interpolation is exact.
  let r = 0.05_f64;
  let pts = vec![
    CurvePoint {
      time: 0.0,
      discount_factor: 1.0,
    },
    CurvePoint {
      time: 1.0,
      discount_factor: (-r).exp(),
    },
    CurvePoint {
      time: 2.0,
      discount_factor: (-2.0 * r).exp(),
    },
  ];
  let curve = DiscountCurve::new(pts, InterpolationMethod::LogLinearOnDiscountFactors);

  for t in [0.3, 0.5, 0.7, 1.2, 1.5, 1.8] {
    approx(curve.discount_factor(t), (-r * t).exp(), 1e-10);
  }
}

#[test]
fn linear_on_zero_rates_interpolates_correctly() {
  let pts = vec![
    CurvePoint {
      time: 0.0,
      discount_factor: 1.0,
    },
    CurvePoint {
      time: 1.0,
      discount_factor: (-0.04_f64).exp(),
    },
    CurvePoint {
      time: 2.0,
      discount_factor: (-0.10_f64).exp(),
    },
  ];
  let curve = DiscountCurve::new(pts, InterpolationMethod::LinearOnZeroRates);

  // r(1) = 0.04, r(2) = 0.05, so at t=1.5: r = 0.045
  let r_at_1_5 = curve.zero_rate(1.5);
  approx(r_at_1_5, 0.045, 1e-4);
}

// DiscountCurve construction from zero rates

#[test]
fn from_zero_rates_consistency() {
  let times = array![0.5, 1.0, 2.0, 5.0];
  let rates = array![0.03, 0.04, 0.05, 0.06];
  let curve =
    DiscountCurve::from_zero_rates(&times, &rates, InterpolationMethod::LinearOnZeroRates);

  for i in 0..times.len() {
    let df = (-(rates[i] as f64) * (times[i] as f64)).exp();
    approx(curve.discount_factor(times[i]), df, 1e-12);
  }
}

// Multi-curve framework

#[test]
fn multi_curve_fair_swap_rate_single_curve() {
  // When discount == forecast, fair swap rate should equal par rate
  let pts = vec![
    CurvePoint {
      time: 0.0,
      discount_factor: 1.0,
    },
    CurvePoint {
      time: 0.5,
      discount_factor: 0.975,
    },
    CurvePoint {
      time: 1.0,
      discount_factor: 0.950,
    },
  ];
  let curve = DiscountCurve::new(pts.clone(), InterpolationMethod::LinearOnZeroRates);
  let forecast = DiscountCurve::new(pts, InterpolationMethod::LinearOnZeroRates);

  let mut mc = MultiCurve::new(curve);
  mc.add_forecast("6M", forecast);

  let schedule = array![0.0, 0.5, 1.0];
  let fair_rate = mc.fair_swap_rate("6M", &schedule).unwrap();
  let par_rate = mc.discount.par_rate(1.0, 2);

  approx(fair_rate, par_rate, 1e-4);
}

#[test]
fn multi_curve_basis_spread() {
  let ois_pts = vec![
    CurvePoint {
      time: 0.0,
      discount_factor: 1.0,
    },
    CurvePoint {
      time: 1.0,
      discount_factor: 0.96,
    },
  ];
  let forecast_pts = vec![
    CurvePoint {
      time: 0.0,
      discount_factor: 1.0,
    },
    CurvePoint {
      time: 1.0,
      discount_factor: 0.955,
    },
  ];

  let ois = DiscountCurve::new(ois_pts, InterpolationMethod::LinearOnZeroRates);
  let forecast = DiscountCurve::new(forecast_pts, InterpolationMethod::LinearOnZeroRates);

  let mut mc = MultiCurve::new(ois);
  mc.add_forecast("3M", forecast);

  let spread = mc.basis_spread("3M", 0.0, 1.0).unwrap();
  // Forecast forward > OIS forward, so spread > 0
  assert!(spread > 0.0);

  let fwd_forecast = mc.projected_forward("3M", 0.0, 1.0).unwrap();
  let fwd_ois = mc.discount.simple_forward_rate(0.0, 1.0);
  approx(spread, fwd_forecast - fwd_ois, 1e-12);
}

#[test]
fn multi_curve_pv_floating_leg() {
  let ois_pts = vec![
    CurvePoint {
      time: 0.0,
      discount_factor: 1.0,
    },
    CurvePoint {
      time: 0.5,
      discount_factor: 0.98,
    },
    CurvePoint {
      time: 1.0,
      discount_factor: 0.96,
    },
  ];
  let forecast_pts = vec![
    CurvePoint {
      time: 0.0,
      discount_factor: 1.0,
    },
    CurvePoint {
      time: 0.5,
      discount_factor: 0.979,
    },
    CurvePoint {
      time: 1.0,
      discount_factor: 0.957,
    },
  ];

  let ois = DiscountCurve::new(ois_pts, InterpolationMethod::LinearOnZeroRates);
  let forecast = DiscountCurve::new(forecast_pts, InterpolationMethod::LinearOnZeroRates);

  let mut mc = MultiCurve::new(ois);
  mc.add_forecast("6M", forecast);

  let schedule = array![0.0, 0.5, 1.0];
  let notional = 1_000_000.0;
  let pv = mc.pv_floating_leg("6M", &schedule, notional).unwrap();

  // Manual: PV = D_ois(0.5) * 0.5 * F_fc(0,0.5) * N + D_ois(1) * 0.5 * F_fc(0.5,1) * N
  let f1 = (1.0 / 0.979 - 1.0) / 0.5;
  let f2 = (0.979 / 0.957 - 1.0) / 0.5;
  let expected = 0.98 * 0.5 * f1 * notional + 0.96 * 0.5 * f2 * notional;
  approx(pv, expected, 1.0);
}

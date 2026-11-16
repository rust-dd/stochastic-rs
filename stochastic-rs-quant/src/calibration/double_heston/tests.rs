use num_complex::Complex64;

use super::loss::double_heston_call_price;
use super::loss::double_heston_cf;
use super::*;
use crate::LossMetric;
use crate::OptionQuote;
use crate::OptionType;
use crate::QuantError;
use crate::calibration::CalibrationHistory;
use crate::calibration::CalibrationOptions;
use crate::pricing::fourier::FourierModelExt;
use crate::pricing::heston::HestonParams;
use crate::traits::Calibrator;
use crate::traits::ToModel;

// Reference Heston (v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.7)
// S=100, r=0.05, q=0, T=1.0. A double Heston with v1_0+v2_0 = 0.04 and
// factor-1 degenerate (v1_0 ≈ 0) collapses onto the single Heston answer.
const HESTON_REF: [f64; 9] = [
  25.095178, 20.976171, 17.106937, 13.548230, 10.361869, 7.604362, 5.317953, 3.519953, 2.193310,
];
const STRIKES: [f64; 9] = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0];

#[test]
fn double_heston_reduces_to_heston_when_one_factor_vanishes() {
  // Factor 2 has v2_0=0 and theta2≈0 so its contribution is negligible.
  let p = DoubleHestonParams {
    first: HestonParams {
      v0: 0.04,
      kappa: 1.5,
      theta: 0.04,
      sigma: 0.3,
      rho: -0.7,
    },
    second: HestonParams {
      v0: 1e-6,
      kappa: 5.0,
      theta: 1e-6,
      sigma: 0.01,
      rho: 0.0,
    },
  };
  for (i, &k) in STRIKES.iter().enumerate() {
    let price = double_heston_call_price(&p, 100.0, k, 0.05, 0.0, 1.0);
    assert!(
      (price - HESTON_REF[i]).abs() < 0.25,
      "Double Heston (1 active factor) K={k}: got {price:.6}, expected {:.6}",
      HESTON_REF[i]
    );
  }
}

#[test]
fn double_heston_two_factors_produces_sensible_smile() {
  // Standard Christoffersen 2-factor split: one fast-mean-reverting factor,
  // one slow-mean-reverting factor. Prices should be positive, finite,
  // monotonic in strike, and satisfy basic bounds.
  let p = DoubleHestonParams {
    first: HestonParams {
      v0: 0.02,
      kappa: 3.0,
      theta: 0.02,
      sigma: 0.4,
      rho: -0.6,
    },
    second: HestonParams {
      v0: 0.02,
      kappa: 0.5,
      theta: 0.03,
      sigma: 0.2,
      rho: -0.3,
    },
  };
  let mut prev = f64::INFINITY;
  for &k in STRIKES.iter() {
    let price = double_heston_call_price(&p, 100.0, k, 0.05, 0.0, 1.0);
    // Intrinsic value lower bound: max(S e^{-qT} - K e^{-rT}, 0)
    let intrinsic = (100.0_f64 - k * (-0.05_f64 * 1.0).exp()).max(0.0);
    assert!(
      price.is_finite() && price >= intrinsic - 1e-6 && price <= 100.0 + 1e-6,
      "Double Heston K={k}: got {price:.6} outside [{intrinsic:.6}, 100]"
    );
    assert!(
      price < prev + 1e-6,
      "Double Heston prices should be monotone decreasing in strike: {prev:.6} → {price:.6} at K={k}"
    );
    prev = price;
  }
}

fn heston_ref_quotes() -> Vec<OptionQuote> {
  STRIKES
    .iter()
    .zip(HESTON_REF)
    .map(|(&k, price)| OptionQuote::new(k, 1.0, price, OptionType::Call))
    .collect()
}

const START: DoubleHestonParams = DoubleHestonParams {
  first: HestonParams {
    v0: 0.03,
    kappa: 2.0,
    theta: 0.03,
    sigma: 0.25,
    rho: -0.5,
  },
  second: HestonParams {
    v0: 0.01,
    kappa: 0.8,
    theta: 0.02,
    sigma: 0.15,
    rho: -0.4,
  },
};

#[test]
fn double_heston_calibrate_to_heston_surface() {
  // Calibrate to Heston prices: we expect the RMSE to be small.
  let result = DoubleHestonCalibrator::new(100.0, 0.05, 0.0, heston_ref_quotes())
    .with_start(START)
    .calibrate(None)
    .unwrap();
  assert!(
    result.loss.get(LossMetric::Rmse) < 0.6,
    "Double Heston calibration RMSE={:.6}",
    result.loss.get(LossMetric::Rmse)
  );
  // Verify the calibrated price is close to the reference
  let p_out = result.params;
  for (i, &k) in STRIKES.iter().enumerate() {
    let price = double_heston_call_price(&p_out, 100.0, k, 0.05, 0.0, 1.0);
    assert!(
      (price - HESTON_REF[i]).abs() < 1.5,
      "Calibrated price K={k}: got {price:.4}, ref {:.4}",
      HESTON_REF[i]
    );
  }
}

/// The trail opens at the projected start and each later entry is an accepted step, so the
/// objective falls along it to the fit.
#[test]
fn the_trail_opens_at_the_start_and_records_market_minus_model() {
  let quotes = heston_ref_quotes();
  let fit = DoubleHestonCalibrator::new(100.0, 0.05, 0.0, quotes.clone())
    .with_start(START)
    .with_options(
      CalibrationOptions::default()
        .with_record_history(true)
        .with_max_iterations(8),
    )
    .calibrate(None)
    .unwrap();
  assert!(fit.history.len() > 1, "{} entries", fit.history.len());
  assert_eq!(fit.history[0].params, START.projected());
  assert_eq!(
    fit.history.last().map(|entry| entry.params),
    Some(fit.params)
  );
  let objective =
    |entry: &CalibrationHistory<DoubleHestonParams>| entry.residuals.dot(&entry.residuals);
  assert!(
    fit
      .history
      .windows(2)
      .all(|pair| objective(&pair[1]) < objective(&pair[0]))
  );
  for entry in [&fit.history[0], fit.history.last().unwrap()] {
    for (quote, residual) in quotes.iter().zip(&entry.residuals) {
      let model = double_heston_call_price(&entry.params, 100.0, quote.strike, 0.05, 0.0, 1.0);
      assert_eq!(*residual, quote.price - model);
    }
  }
}

#[test]
fn a_start_outside_the_heston_domain_is_an_error_before_the_fit() {
  let calibrator = DoubleHestonCalibrator::new(100.0, 0.05, 0.0, heston_ref_quotes());
  let nan_rho = DoubleHestonParams {
    second: HestonParams {
      rho: f64::NAN,
      ..START.second
    },
    ..START
  };
  match calibrator.calibrate(Some(nan_rho)) {
    Err(QuantError::InvalidInput {
      what: "start",
      reason,
    }) => assert!(
      reason.starts_with("second factor: rho must satisfy"),
      "{reason}"
    ),
    other => panic!("{other:?}"),
  }
  let mut field = calibrator;
  field.start = Some(DoubleHestonParams {
    first: HestonParams {
      v0: -0.01,
      ..START.first
    },
    ..START
  });
  assert!(matches!(
    field.calibrate(None),
    Err(QuantError::InvalidInput { what: "start", .. })
  ));
}

#[test]
#[should_panic(expected = "sigma must satisfy `sigma.is_finite() && sigma > 0.0`, got sigma = 0.0")]
fn with_start_panics_outside_the_heston_domain() {
  let _ = DoubleHestonCalibrator::new(100.0, 0.05, 0.0, heston_ref_quotes()).with_start(
    DoubleHestonParams {
      first: HestonParams {
        sigma: 0.0,
        ..START.first
      },
      ..START
    },
  );
}

#[test]
fn double_heston_params_to_model() {
  let p = DoubleHestonParams {
    first: HestonParams {
      v0: 0.02,
      kappa: 3.0,
      theta: 0.02,
      sigma: 0.3,
      rho: -0.6,
    },
    second: HestonParams {
      v0: 0.02,
      kappa: 0.5,
      theta: 0.02,
      sigma: 0.15,
      rho: -0.3,
    },
  };
  let model = p.to_model();
  for u in [Complex64::new(0.7, 0.0), Complex64::new(1.3, -0.5)] {
    assert_eq!(model.chf(1.0, u), double_heston_cf(&p, 0.0, 0.0, 1.0, u));
  }
}

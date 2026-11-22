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
  // The fit descends along Feller's boundary to any budget; 100 iterations show the rmse falls.
  let result = DoubleHestonCalibrator::new(100.0, 0.05, 0.0, heston_ref_quotes())
    .with_start(START)
    .with_options(CalibrationOptions::default().with_max_iterations(100))
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
  // The start returns from each σ's share within a rounding.
  let mut opened = fit.history[0].params;
  for (factor, start) in [
    (&mut opened.first, START.first),
    (&mut opened.second, START.second),
  ] {
    assert!((factor.sigma - start.sigma).abs() < 1e-15, "{factor:?}");
    factor.sigma = start.sigma;
  }
  assert_eq!(opened, START.projected());
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
#[should_panic(
  expected = "second factor: sigma must satisfy `sigma.is_finite() && sigma > 0.0`, got sigma = 0.0"
)]
fn with_start_panics_naming_the_factor_outside_the_heston_domain() {
  let _ = DoubleHestonCalibrator::new(100.0, 0.05, 0.0, heston_ref_quotes()).with_start(
    DoubleHestonParams {
      second: HestonParams {
        sigma: 0.0,
        ..START.second
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

#[test]
fn the_jacobian_is_the_slope_of_the_residuals_across_the_box() {
  use super::loss::DoubleHestonProblem;
  use super::params::FOLD;
  use crate::calibration::least_squares::LeastSquaresProblem;
  use crate::calibration::least_squares::checks::beside_points;
  use crate::calibration::least_squares::checks::beside_reference;
  use crate::calibration::least_squares::checks::box_points;
  use crate::calibration::least_squares::checks::jacobian_error;
  let calibrator = DoubleHestonCalibrator::new(100.0, 0.05, 0.0, heston_ref_quotes());
  let mut problem = DoubleHestonProblem::new(&calibrator, START);
  let far = ndarray::array![0.2, 12.0, 0.3, 0.9, 0.6, 0.15, 6.0, 0.2, 0.7, -0.8];
  for base in [problem.params(), far] {
    for (label, point) in box_points(base.view(), &FOLD, &[true; 10]) {
      let error = jacobian_error(&mut problem, point.view(), |_, x| 2e-3 * x.abs().max(1e-2));
      assert!(error < 2e-2, "{label} of {base}: {error:e}");
    }
    for (label, point) in beside_points(base.view(), &FOLD) {
      let error = jacobian_error(&mut problem, point.view(), beside_reference);
      assert!(error < 2e-2, "{label} of {base}: {error:e}");
    }
  }
}

/// The first factor's `ρ` meets its lower bound while the cost still falls across it: the fit
/// goes on along the bound.
#[test]
fn the_default_start_fits_the_heston_reference_along_the_rho_bound() {
  let fit = DoubleHestonCalibrator::new(100.0, 0.05, 0.0, heston_ref_quotes())
    .with_options(CalibrationOptions::default().with_max_iterations(200))
    .calibrate(None)
    .unwrap();
  let rmse = fit.loss.get(LossMetric::Rmse);
  assert!(rmse < 1e-3, "{rmse:e}: {fit:?}");
}

/// Where a raw step leaves `ρ` unmoved no residual moves and the trust region closes: the fold has
/// no such strip.
#[test]
fn no_raw_step_leaves_a_correlation_where_it_was() {
  let rho = |raw: f64| {
    let mut x = ndarray::Array1::from(START);
    x[4] = raw;
    DoubleHestonParams::from(x).folded().first.rho
  };
  for raw in (0..=60).map(|i| -1.0003 + 1e-5 * f64::from(i)) {
    let (at, beside) = (rho(raw), rho(raw + 1e-6));
    assert!(
      at != beside && at.abs() <= 0.9999,
      "{raw}: {at} and {beside}"
    );
  }
}

/// On the surface fit the first `ρ` is held at −0.9999: there its folded value is the bound.
#[test]
fn a_correlation_held_at_its_bound_is_on_it_not_beside_it() {
  use crate::calibration::least_squares::HELD;
  HELD.take();
  DoubleHestonCalibrator::new(100.0, 0.05, 0.0, heston_ref_quotes())
    .with_start(START)
    .with_options(CalibrationOptions::default().with_max_iterations(400))
    .calibrate(None)
    .unwrap();
  let held = HELD.take().into_iter().filter(|(_, set)| set[4]);
  let gaps = held.map(|(x, _)| DoubleHestonParams::from(x).folded().first.rho + 0.9999);
  let gaps = gaps.collect::<Vec<_>>();
  assert!(
    !gaps.is_empty() && gaps.iter().all(|gap| *gap == 0.0),
    "{gaps:?}"
  );
}

/// A restart from a minimum stays there: the default fit, `converged` or at its budget as the
/// build's sine and cosine round, is not improved tenfold from its own point.
#[test]
fn the_default_fit_is_not_improved_tenfold_by_a_restart() {
  let calibrator = || DoubleHestonCalibrator::new(100.0, 0.05, 0.0, heston_ref_quotes());
  let fit = calibrator().calibrate(None).unwrap();
  let again = calibrator().with_start(fit.params).calibrate(None).unwrap();
  let rmse = |fit: &Calibrated<DoubleHestonParams>| fit.loss.get(LossMetric::Rmse);
  let (first, second) = (rmse(&fit), rmse(&again));
  assert!(
    10.0 * second >= first,
    "{first:e} ({:?}), {second:e} from it",
    fit.message
  );
}

/// Each factor has its own slots: the second's `ρ` reflected at its bound, and its `κ` at the top.
#[test]
fn the_room_of_each_coordinate_is_its_distance_to_its_bounds() {
  use super::params::room_in_box;
  let base = ndarray::Array1::from(START);
  let rooms = |j: usize, value: f64| {
    let mut x = base.clone();
    x[j] = value;
    let room = room_in_box(x.view());
    (room.down[j], room.up[j])
  };
  let near = |(down, up): (f64, f64), want: (f64, f64)| {
    (down - want.0).abs() < 1e-12 && (up - want.1).abs() < 1e-12
  };
  for (j, value, want) in [
    (4, -0.5, (0.4999, 1.4999)),
    (9, -0.99995, (1.99975, 5e-5)),
    (9, -1.00005, (1.99965, 1.5e-4)),
    (9, 0.9995, (1.9994, 4e-4)),
    (6, 19.99, (19.89, 0.01)),
    (5, 0.0005, (0.2485, 0.0005)),
  ] {
    assert!(
      near(rooms(j, value), want),
      "{j} at {value}: {:?}",
      rooms(j, value)
    );
  }
}

/// A pull acts on each factor's `(v0, κ, θ, σ, ρ)`: σ's row runs through its share and its
/// cap's `κ` and `θ`, and a reflected coordinate's slope is −1.
#[test]
fn the_natural_jacobian_matches_central_differences() {
  use super::loss::DoubleHestonProblem;
  use crate::calibration::least_squares::LeastSquaresProblem;
  use crate::calibration::least_squares::checks::natural_jacobian_by_differences;
  use crate::calibration::least_squares::checks::worst_relative_error;
  let calibrator = DoubleHestonCalibrator::new(100.0, 0.05, 0.0, heston_ref_quotes());
  let mut problem = DoubleHestonProblem::new(&calibrator, START);
  for x in [
    ndarray::array![0.03, 2.0, 0.03, 0.6, -0.5, 0.01, 0.8, 0.02, 0.3, -0.4],
    ndarray::array![0.03, 20.5, 0.3, 0.4, -0.5, 0.26, 0.8, 0.02, 1.2, -0.4],
  ] {
    problem.set_params(x.view());
    let analytic = problem.natural_jacobian();
    let numeric = natural_jacobian_by_differences(&mut problem, x.view(), 1e-6);
    let worst = worst_relative_error(analytic.view(), numeric.view());
    assert!(worst < 1e-6, "{x}: {worst:e}");
  }
}

/// A start inside Feller's condition is its own model point, and a factor that breaks it starts
/// on its cap.
#[test]
fn a_start_maps_to_its_shares_and_back() {
  use super::loss::DoubleHestonProblem;
  use crate::calibration::least_squares::LeastSquaresProblem;
  let calibrator = DoubleHestonCalibrator::new(100.0, 0.05, 0.0, heston_ref_quotes());
  let inside = DoubleHestonProblem::new(&calibrator, START).effective_params();
  assert!((inside.first.sigma - START.first.sigma).abs() < 1e-15);
  assert!((inside.second.sigma - START.second.sigma).abs() < 1e-15);
  let breaking = DoubleHestonParams {
    second: HestonParams {
      sigma: 0.9,
      ..START.second
    },
    ..START
  };
  let capped = DoubleHestonProblem::new(&calibrator, breaking);
  assert_eq!(capped.params()[8], 1.0);
  let (capped, want) = (
    capped.effective_params().second,
    breaking.projected().second,
  );
  assert!((capped.sigma - want.sigma).abs() < 1e-15, "{capped:?}");
  assert!(capped.sigma * capped.sigma <= 2.0 * capped.kappa * capped.theta);
}

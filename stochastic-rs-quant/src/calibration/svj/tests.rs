use super::SvjCalibrator;
use super::SvjParams;
use super::loss::bates_call_price;
use crate::LossMetric;
use crate::OptionQuote;
use crate::OptionType;
use crate::QuantError;
use crate::calibration::CalibrationHistory;
use crate::calibration::CalibrationOptions;
use crate::pricing::fourier::BatesFourier;
use crate::traits::CalibrationResult;
use crate::traits::Calibrator;
use crate::traits::ToModel;

const HESTON_REF: [f64; 9] = [
  25.095178, 20.976171, 17.106937, 13.548230, 10.361869, 7.604362, 5.317953, 3.519953, 2.193310,
];
const STRIKES: [f64; 9] = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0];
const START: SvjParams = SvjParams {
  v0: 0.06,
  kappa: 2.0,
  theta: 0.06,
  sigma: 0.4,
  rho: -0.5,
  lambda: 0.1,
  mu_j: 0.0,
  sigma_j: 0.1,
};

fn heston_ref_quotes() -> Vec<OptionQuote> {
  STRIKES
    .iter()
    .zip(HESTON_REF)
    .map(|(&k, price)| OptionQuote::new(k, 1.0, price, OptionType::Call))
    .collect()
}

#[test]
fn bates_pricer_matches_heston_reference() {
  let p = SvjParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
    lambda: 0.0,
    mu_j: 0.0,
    sigma_j: 0.01,
  };
  for (i, &k) in STRIKES.iter().enumerate() {
    let price = bates_call_price(&p, 100.0, k, 0.05, 0.0, 1.0);
    assert!(
      (price - HESTON_REF[i]).abs() < 0.1,
      "Heston K={k}: got {price:.6}, expected {:.6}",
      HESTON_REF[i]
    );
  }
}

#[test]
fn svj_calibrate_recovers_heston_prices() {
  // The fit descends along Feller's boundary to any budget; 100 iterations show the rmse falls.
  let result = SvjCalibrator::new(100.0, 0.05, 0.0, heston_ref_quotes())
    .with_start(START)
    .with_options(CalibrationOptions::default().with_max_iterations(100))
    .calibrate(None)
    .unwrap();
  assert!(
    result.loss.get(LossMetric::Rmse) < 0.5,
    "SVJ→Heston RMSE={:.6}",
    result.loss.get(LossMetric::Rmse)
  );
}

/// At `T = 5`, `ρ = −0.9` the original Heston (1993) form crosses its logarithm's branch cut; the
/// Little Heston Trap form of Albrecher et al. (2007) keeps the SVJ call finite and bounded.
#[test]
fn svj_bates_little_trap_long_maturity_high_rho() {
  let p = SvjParams {
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.9,
    lambda: 0.3,
    mu_j: -0.05,
    sigma_j: 0.15,
  };
  let call = bates_call_price(&p, 100.0, 100.0, 0.05, 0.0, 5.0);
  assert!(
    call.is_finite() && call > 0.0 && call < 100.0,
    "SVJ bates_cf Trap form: finite positive bounded call required at T=5y, ρ=-0.9, got {call}"
  );
}

/// The trail opens at the projected start and each later entry is an accepted step, so the
/// objective falls along it to the fit.
#[test]
fn the_trail_opens_at_the_start_and_records_market_minus_model() {
  let quotes = heston_ref_quotes();
  let fit = SvjCalibrator::new(100.0, 0.05, 0.0, quotes.clone())
    .with_start(START)
    .with_options(
      CalibrationOptions::default()
        .with_record_history(true)
        .with_max_iterations(8),
    )
    .calibrate(None)
    .unwrap();
  assert!(fit.history.len() > 1, "{} entries", fit.history.len());
  // The start returns from σ's share within a rounding.
  let opened = fit.history[0].params;
  assert!((opened.sigma - START.sigma).abs() < 1e-15, "{opened:?}");
  let exact = SvjParams {
    sigma: START.sigma,
    ..opened
  };
  assert_eq!(exact, START.projected());
  assert_eq!(
    fit.history.last().map(|entry| entry.params),
    Some(fit.params)
  );
  let objective = |entry: &CalibrationHistory<SvjParams>| entry.residuals.dot(&entry.residuals);
  assert!(
    fit
      .history
      .windows(2)
      .all(|pair| objective(&pair[1]) < objective(&pair[0]))
  );
  for entry in [&fit.history[0], fit.history.last().unwrap()] {
    for (quote, residual) in quotes.iter().zip(&entry.residuals) {
      let model = bates_call_price(&entry.params, 100.0, quote.strike, 0.05, 0.0, quote.tau);
      assert_eq!(*residual, quote.price - model);
    }
  }
}

#[test]
fn a_start_outside_the_bates_domain_is_an_error() {
  let calibrator = SvjCalibrator::new(100.0, 0.05, 0.0, heston_ref_quotes());
  for start in [
    SvjParams {
      sigma: -0.3,
      ..START
    },
    SvjParams {
      rho: f64::NAN,
      ..START
    },
  ] {
    let err = calibrator.calibrate(Some(start)).unwrap_err();
    assert!(
      matches!(err, QuantError::InvalidInput { what: "start", .. }),
      "{err}"
    );
  }
  let mut field = calibrator;
  field.start = Some(SvjParams {
    lambda: -1.0,
    ..START
  });
  assert!(matches!(
    field.calibrate(None),
    Err(QuantError::InvalidInput { what: "start", .. })
  ));
}

#[test]
#[should_panic(expected = "rho must satisfy `(-1.0..=1.0).contains(&rho)`, got rho = 1.5")]
fn with_start_panics_outside_the_bates_domain() {
  let _ = SvjCalibrator::new(100.0, 0.05, 0.0, heston_ref_quotes())
    .with_start(SvjParams { rho: 1.5, ..START });
}

#[test]
#[should_panic(expected = "sigma_j must satisfy `sigma_j.is_finite() && sigma_j >= 0.0`")]
fn bates_fourier_new_panics_outside_the_domain() {
  let _ = BatesFourier::new(SvjParams {
    sigma_j: -0.1,
    ..START
  });
}

#[test]
fn the_fit_converts_to_the_bates_model_of_its_parameters() {
  let fit = SvjCalibrator::new(100.0, 0.05, 0.0, heston_ref_quotes())
    .with_start(START)
    .with_options(CalibrationOptions::default().with_max_iterations(3))
    .calibrate(None)
    .unwrap();
  assert_eq!(fit.to_model().params(), fit.params);
}

/// Twelve half-year calls no Bates model reprices, and a start that breaks Feller's condition.
fn market_fit() -> SvjCalibrator {
  let k = [
    80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0,
  ];
  let c_market = [
    21.5, 17.9, 14.2, 11.0, 8.2, 6.0, 4.3, 3.1, 2.2, 1.6, 1.2, 0.9,
  ];
  let quotes = k
    .iter()
    .zip(c_market)
    .map(|(&k, price)| OptionQuote::new(k, 0.5, price, OptionType::Call))
    .collect();
  SvjCalibrator::new(100.0, 0.01, 0.0, quotes).with_start(SvjParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.5,
    rho: -0.7,
    lambda: 0.5,
    mu_j: -0.05,
    sigma_j: 0.1,
  })
}

#[test]
fn test_svj_calibrate() {
  let result = market_fit()
    .with_options(CalibrationOptions::default().with_record_history(true))
    .calibrate(None)
    .unwrap();
  assert!(result.rmse().is_finite() && !result.history.is_empty());
}

/// Each column within 2e-2 of `‖D‖` on the box and 1e-2 beside a bound, or within what its
/// residuals' rounding explains: Heston's `κθ/σ²` terms cancel to `1e-9` at σ ≈ 0.01.
#[test]
fn the_jacobian_is_the_slope_of_the_residuals_across_the_box() {
  use super::loss::SvjProblem;
  use super::params::*;
  use crate::calibration::least_squares::LeastSquaresProblem;
  use crate::calibration::least_squares::checks::assert_slope;
  use crate::calibration::least_squares::checks::beside_points;
  use crate::calibration::least_squares::checks::beside_reference;
  use crate::calibration::least_squares::checks::box_points;
  let calibrator = SvjCalibrator::new(100.0, 0.05, 0.0, heston_ref_quotes());
  let mut problem = SvjProblem::new(&calibrator, START);
  let bounds = BOX;
  let far = ndarray::array![0.2, 12.0, 0.3, 0.9, 0.6, 4.0, -0.3, 0.5];
  let on_the_box = |_: usize, x: f64| 2e-3 * x.abs().max(1e-2);
  for base in [problem.params(), far] {
    for (label, point) in box_points(base.view(), &bounds, &[true; 8]) {
      let label = format!("{label} of {base}");
      assert_slope(&mut problem, &label, point.view(), on_the_box, 2e-2);
    }
    for (label, point) in beside_points(base.view(), &bounds) {
      let label = format!("{label} of {base}");
      assert_slope(&mut problem, &label, point.view(), beside_reference, 1e-2);
    }
  }
}

/// `λ` beside 0 and reflected below it, `μ_J` at the top of its range, `ρ` reflected at its bound.
#[test]
fn the_room_of_each_coordinate_is_its_distance_to_its_bounds() {
  use super::params::room_in_box;
  let base = ndarray::array![0.04, 1.5, 0.04, 0.3, -0.5, 0.5, -0.05, 0.1];
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
    (5, 5e-6, (5e-6, 10.0 - 5e-6)),
    (5, -5e-6, (10.0 - 5e-6, 5e-6)),
    (6, 0.4999, (0.9999, 1e-4)),
    (6, 0.5001, (1e-4, 0.9999)),
    (4, -0.99995, (1.99975, 5e-5)),
    (4, 0.9995, (1.9994, 4e-4)),
  ] {
    assert!(
      near(rooms(j, value), want),
      "{j} at {value}: {:?}",
      rooms(j, value)
    );
  }
}

/// The market fit starts on Feller's cap, `w = 1`, and its cost falls across it: at each point a
/// solve steps from with the share held, its folded value is the bound itself.
#[test]
fn a_volatility_held_at_fellers_cap_is_on_it_not_beside_it() {
  use crate::calibration::least_squares::HELD;
  let options = CalibrationOptions::default().with_max_iterations(150);
  HELD.take();
  market_fit().with_options(options).calibrate(None).unwrap();
  let held = HELD.take().into_iter().filter(|(_, set)| set[3]);
  let gaps = held.map(|(x, _)| SvjParams::from(x).folded().sigma - 1.0);
  let gaps = gaps.collect::<Vec<_>>();
  assert!(
    !gaps.is_empty() && gaps.iter().all(|gap| *gap == 0.0),
    "{gaps:?}"
  );
}

/// The default start breaks Feller's condition, so it begins on the cap, σ's upper bound: the fit
/// leaves it for the reference's `σ = 0.3` under a cap of `0.35`.
#[test]
fn the_default_start_leaves_fellers_cap_for_the_heston_reference() {
  let fit = SvjCalibrator::new(100.0, 0.05, 0.0, heston_ref_quotes())
    .with_options(CalibrationOptions::default().with_max_iterations(200))
    .calibrate(None)
    .unwrap();
  let (rmse, p) = (fit.loss.get(LossMetric::Rmse), fit.params);
  let cap = (2.0 * p.kappa * p.theta).sqrt();
  assert!(rmse < 5e-5 && p.sigma < 0.99 * cap, "{rmse:e}: {p:?}");
}

/// A pull acts on `(v0, κ, θ, σ, …)`: σ's row runs through its share and the cap's `κ` and `θ`,
/// and a reflected coordinate's slope is −1.
#[test]
fn the_natural_jacobian_matches_central_differences() {
  use super::loss::SvjProblem;
  use crate::calibration::least_squares::LeastSquaresProblem;
  use crate::calibration::least_squares::checks::natural_jacobian_by_differences;
  use crate::calibration::least_squares::checks::worst_relative_error;
  let calibrator = SvjCalibrator::new(100.0, 0.05, 0.0, heston_ref_quotes());
  let mut problem = SvjProblem::new(&calibrator, START);
  for x in [
    ndarray::array![0.04, 1.5, 0.04, 0.6, -0.5, 0.5, -0.05, 0.1],
    ndarray::array![0.04, 20.5, 0.3, 0.4, -0.5, -0.2, 0.6, 0.1],
    ndarray::array![0.04, 1.5, 0.04, 1.3, -0.5, 0.5, -0.05, 0.1],
  ] {
    problem.set_params(x.view());
    let analytic = problem.natural_jacobian();
    let numeric = natural_jacobian_by_differences(&mut problem, x.view(), 1e-6);
    let worst = worst_relative_error(analytic.view(), numeric.view());
    assert!(worst < 1e-6, "{x}: {worst:e}");
  }
}

/// A start inside Feller's condition is its own model point, and one that breaks it is the cap.
#[test]
fn a_start_maps_to_its_share_and_back() {
  use super::loss::SvjProblem;
  use crate::calibration::least_squares::LeastSquaresProblem;
  let calibrator = SvjCalibrator::new(100.0, 0.05, 0.0, heston_ref_quotes());
  let inside = SvjProblem::new(&calibrator, START);
  assert!((inside.effective_params().sigma - START.sigma).abs() < 1e-15);
  let breaking = SvjParams {
    sigma: 0.9,
    ..START
  };
  let capped = SvjProblem::new(&calibrator, breaking);
  assert_eq!(capped.params()[3], 1.0);
  let (capped, want) = (capped.effective_params(), breaking.projected());
  assert!((capped.sigma - want.sigma).abs() < 1e-15, "{capped:?}");
  assert!(capped.sigma * capped.sigma <= 2.0 * capped.kappa * capped.theta);
}

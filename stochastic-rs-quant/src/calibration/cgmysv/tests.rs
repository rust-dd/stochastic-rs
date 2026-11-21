use super::*;
use crate::LossMetric;
use crate::calibration::CalibrationHistory;
use crate::calibration::least_squares::checks::natural_jacobian_by_differences;
use crate::calibration::least_squares::checks::worst_relative_error;
use crate::pricing::cf_quadrature::Walk;
use crate::traits::Calibrator;

fn known_params() -> CgmysvParams {
  CgmysvParams {
    alpha: 0.5184,
    lambda_plus: 25.4592,
    lambda_minus: 4.6040,
    kappa: 1.0029,
    eta: 0.0711,
    zeta: 0.3443,
    rho: -2.0283,
    v0: 0.01115,
  }
}

fn problem(calibrator: &CgmysvCalibrator) -> CgmysvProblem<'_> {
  CgmysvProblem::new(calibrator, default_params())
}

fn synthetic(s: f64, r: f64, q: f64, tau: f64) -> Vec<OptionQuote> {
  (2400..=2600)
    .step_by(25)
    .map(|k| {
      let k = k as f64;
      let price = fourier_call_price(&known_params(), s, k, r, q, tau, &mut Walk::Adaptive);
      OptionQuote::new(k, tau, price, OptionType::Call)
    })
    .collect()
}

/// Generate synthetic market data from known params, calibrate back, check recovery.
#[test]
fn round_trip_calibration() {
  let (s, r, q, tau) = (2488.11, 0.01213, 0.01884, 28.0 / 365.0);
  let result = CgmysvCalibrator::new(s, r, q, synthetic(s, r, q, tau))
    .calibrate(None)
    .unwrap();
  assert!(result.converged, "CGMYSV exhausted its evaluation budget");
  // The reference solver recovers these synthetic quotes to numerical precision.
  assert!(
    result.loss.get(LossMetric::Rmse) < 1e-6,
    "RMSE = {:.4}, should be small for synthetic data",
    result.loss.get(LossMetric::Rmse)
  );
}

/// Fourier pricing sanity: call price positive and decreasing in K.
#[test]
fn fourier_call_monotone() {
  let p = known_params();
  let s = 2488.11;
  let r = 0.01213;
  let q = 0.01884;
  let tau = 28.0 / 365.0;

  let c1 = fourier_call_price(&p, s, 2400.0, r, q, tau, &mut Walk::Adaptive);
  let c2 = fourier_call_price(&p, s, 2500.0, r, q, tau, &mut Walk::Adaptive);
  let c3 = fourier_call_price(&p, s, 2600.0, r, q, tau, &mut Walk::Adaptive);

  assert!(c1 > c2 && c2 > c3, "calls must decrease in K");
  assert!(c1 > 0.0 && c3 > 0.0, "calls must be positive");
}

#[test]
fn optimizer_coordinates_preserve_both_signs_of_rho() {
  let calibrator = CgmysvCalibrator::new(100.0, 0.03, 0.0, Vec::new());
  let mut problem = problem(&calibrator);
  for rho in [-4.0, 0.0, 4.0] {
    problem.params = default_params();
    problem.params[6] = rho;
    let physical = problem.params.clone();
    let coordinates = LeastSquaresProblem::params(&problem);
    problem.set_params(coordinates.view());
    for (&actual, &expected) in problem.params.iter().zip(&physical) {
      assert!((actual - expected).abs() < 1e-12);
    }
  }
  for upper in [false, true] {
    problem.params = param_bounds()
      .map(|(lo, hi)| if upper { hi } else { lo })
      .to_vec();
    let coordinates = LeastSquaresProblem::params(&problem);
    assert!(coordinates.iter().all(|v| v.is_finite()));
    problem.set_params(coordinates.view());
    for (&value, (lo, hi)) in problem.params.iter().zip(param_bounds()) {
      assert!(value > lo && value < hi);
    }
  }
}

#[test]
fn optimizer_jacobian_matches_finite_differences() {
  let quotes = [95.0, 100.0, 105.0].map(|k| OptionQuote::new(k, 0.5, 0.0, OptionType::Call));
  let calibrator = CgmysvCalibrator::new(100.0, 0.03, 0.0, quotes.to_vec());
  let mut problem = problem(&calibrator);
  let coordinates = LeastSquaresProblem::params(&problem);
  problem.set_params(coordinates.view());
  let jacobian = problem.jacobian().unwrap();
  let step = 1e-5;
  for j in 0..N_PARAMS {
    let mut shifted = coordinates.clone();
    shifted[j] += step;
    problem.set_params(shifted.view());
    let up = problem.residuals().unwrap();
    shifted[j] -= 2.0 * step;
    problem.set_params(shifted.view());
    let down = problem.residuals().unwrap();
    for i in 0..3 {
      let numeric = (up[i] - down[i]) / (2.0 * step);
      let relative = (jacobian[(i, j)] - numeric).abs() / (1.0 + numeric.abs());
      assert!(
        relative < 1e-5,
        "row {i}, column {j}: {} vs {numeric}",
        jacobian[(i, j)]
      );
    }
  }
}

/// The trail opens at the start, through the logistic round trip, and records market − model,
/// the negation of the fit's residual.
#[test]
fn the_trail_opens_at_the_start_and_records_market_minus_model() {
  let (s, r, q, tau) = (2488.11, 0.01213, 0.01884, 28.0 / 365.0);
  let quotes = synthetic(s, r, q, tau);
  let start = CgmysvParams {
    alpha: 0.6,
    lambda_plus: 20.0,
    lambda_minus: 5.0,
    kappa: 1.2,
    eta: 0.06,
    zeta: 0.3,
    rho: -1.5,
    v0: 0.012,
  };
  let fit = CgmysvCalibrator::new(s, r, q, quotes.clone())
    .with_start(start.clone())
    .with_options(
      CalibrationOptions::default()
        .with_record_history(true)
        .with_max_iterations(12),
    )
    .calibrate(None)
    .unwrap();
  assert!(fit.history.len() > 1, "{} entries", fit.history.len());
  for (first, start) in to_vec(&fit.history[0].params)
    .into_iter()
    .zip(to_vec(&start))
  {
    assert!(
      (first - start).abs() <= 1e-12 * start.abs(),
      "{first} against {start}"
    );
  }
  let last = fit.history.last().unwrap();
  assert_eq!(to_vec(&last.params), to_vec(&fit.params));
  let objective = |entry: &CalibrationHistory<CgmysvParams>| entry.residuals.dot(&entry.residuals);
  assert!(
    fit
      .history
      .windows(2)
      .all(|pair| objective(&pair[1]) < objective(&pair[0]))
  );
  for entry in [&fit.history[0], last] {
    for (quote, residual) in quotes.iter().zip(&entry.residuals) {
      let model = fourier_call_price(
        &entry.params,
        s,
        quote.strike,
        r,
        q,
        tau,
        &mut Walk::Adaptive,
      );
      assert_eq!(*residual, quote.price - model);
    }
  }
}

#[test]
fn a_start_with_a_nan_is_an_error_and_with_start_panics() {
  let quotes = [95.0, 100.0, 105.0].map(|k| OptionQuote::new(k, 0.5, 5.0, OptionType::Call));
  let calibrator = CgmysvCalibrator::new(100.0, 0.03, 0.0, quotes.to_vec());
  let bad = CgmysvParams {
    zeta: f64::NAN,
    ..known_params()
  };
  assert!(matches!(
    calibrator.calibrate(Some(bad.clone())),
    Err(QuantError::InvalidInput { what: "start", .. })
  ));
  let panic = std::panic::catch_unwind(|| calibrator.clone().with_start(bad));
  assert!(panic.is_err());
}

#[test]
fn a_put_is_priced_by_parity_and_keeps_a_nan() {
  let p = known_params();
  let (s, k, r, q, t) = (100.0, 95.0, 0.03, 0.01, 0.5);
  let call = fourier_call_price(&p, s, k, r, q, t, &mut Walk::Adaptive);
  let put = fourier_option_price(&p, s, k, r, q, t, OptionType::Put, &mut Walk::Adaptive);
  assert_eq!(put, call - s * (-q * t).exp() + k * (-r * t).exp());
  assert!(
    fourier_option_price(
      &p,
      f64::NAN,
      k,
      r,
      q,
      t,
      OptionType::Put,
      &mut Walk::Adaptive
    )
    .is_nan()
  );
}

#[test]
fn the_natural_jacobian_matches_central_differences() {
  let quotes = vec![OptionQuote::new(100.0, 0.5, 5.0, OptionType::Call)];
  let calibrator = CgmysvCalibrator::new(100.0, 0.02, 0.0, quotes);
  let mut problem = CgmysvProblem::new(&calibrator, to_vec(&known_params()));
  let x = problem.params();
  problem.set_params(x.view());
  let analytic = problem.natural_jacobian();
  let numeric = natural_jacobian_by_differences(&mut problem, x.view(), 1e-6);
  let worst = worst_relative_error(analytic.view(), numeric.view());
  assert!(worst < 1e-6, "{worst:e}");
}

/// A point beside which a walk that stops on its envelope moves the Lewis price of K = 2525 by
/// 3.3e-5 within `1e-6` of η, in `(α, λ₊, λ₋, κ, η, ζ, ρ, v₀)`.
const STALLED: [f64; 8] = [
  0.591203583519947,
  22.88796593498391,
  3.5756432849914654,
  1.7507022596640776,
  0.07989722351382043,
  0.39842728458201776,
  -1.7672945556843604,
  0.009934101706378984,
];

/// A step of `2e-3·|θ|` in each natural parameter reads every column within 2e-2.
#[test]
fn the_jacobian_is_the_slope_of_the_residuals_above_the_quadratures_jumps() {
  use crate::calibration::least_squares::checks::jacobian_error;
  let (s, r, q, tau) = (2488.11, 0.01213, 0.01884, 28.0 / 365.0);
  let calibrator = CgmysvCalibrator::new(s, r, q, synthetic(s, r, q, tau));
  let mut problem = problem(&calibrator);
  for natural in [STALLED.to_vec(), to_vec(&known_params()), default_params()] {
    problem.params = natural.clone();
    let x = problem.params();
    problem.set_params(x.view());
    let factors = problem.logistic_factors().collect::<Vec<_>>();
    let step = |j: usize, _| 2e-3 * natural[j].abs() / factors[j];
    let error = jacobian_error(&mut problem, x.view(), step);
    assert!(error < 2e-2, "{natural:?}: {error:e}");
  }
}

/// `(K, half-width in η)` about `STALLED`: where a walk that stops on its envelope jumps by 3.3e-5
/// and 9.9e-5.
const WINDOWS: [(f64, f64); 2] = [(2525.0, 1.5e-6), (2550.0, 8e-5)];

/// Across each window a step of the price differs from its neighbours' mean by rounding alone.
#[test]
fn the_price_is_continuous_in_eta_where_an_envelope_stop_would_move() {
  let (s, r, q, tau) = (2488.11, 0.01213, 0.01884, 28.0 / 365.0);
  for (k, half) in WINDOWS {
    let price = |i: i32| {
      let mut natural = STALLED;
      natural[4] += half * (f64::from(i) / 30.0 - 1.0);
      let params = to_cgmysv_params(&natural);
      fourier_call_price(&params, s, k, r, q, tau, &mut Walk::Adaptive)
    };
    let prices = (0..=60).map(price).collect::<Vec<_>>();
    let steps = prices.windows(2).map(|w| w[1] - w[0]).collect::<Vec<_>>();
    let jump = steps
      .windows(3)
      .map(|w| (w[1] - 0.5 * (w[0] + w[2])).abs())
      .fold(0.0, f64::max);
    assert!(jump < 1e-10, "K = {k}: a jump of {jump:e}");
  }
}

/// Each coordinate saturated to within 1e-3 of an end of its logistic map.
#[test]
fn the_jacobian_is_the_slope_of_the_residuals_beside_each_bound() {
  use crate::calibration::least_squares::checks::jacobian_error;
  let quotes = vec![OptionQuote::new(100.0, 0.5, 5.0, OptionType::Call)];
  let calibrator = CgmysvCalibrator::new(100.0, 0.02, 0.0, quotes);
  let mut problem = problem(&calibrator);
  let start = problem.params();
  for coordinate in 0..N_PARAMS {
    for end in [-6.9, 6.9] {
      let mut point = start.clone();
      point[coordinate] = end;
      problem.set_params(point.view());
      // The model has no price where `E[e^L]` is infinite, at the low end of `λ₊`.
      if (coordinate, end) == (1, -6.9) {
        assert!(problem.residuals().unwrap().iter().all(|r| r.is_nan()));
        continue;
      }
      let error = jacobian_error(&mut problem, point.view(), |_, _| 1e-2);
      assert!(error < 2e-2, "coordinate {coordinate} at {end}: {error:e}");
    }
  }
}

/// On the rule each price's own walk took, a difference integrates one fixed rule: at `STALLED`,
/// `J(h)` is `J(2h)`.
#[test]
fn the_jacobian_at_twice_its_step_is_the_same_where_a_walk_jumps() {
  use crate::calibration::least_squares::checks::step_doubling_error;
  let (s, r, q, tau) = (2488.11, 0.01213, 0.01884, 28.0 / 365.0);
  let calibrator = CgmysvCalibrator::new(s, r, q, synthetic(s, r, q, tau));
  let mut problem = problem(&calibrator);
  problem.params = STALLED.to_vec();
  let error = step_doubling_error(|scale| problem.difference_jacobian(scale));
  assert!(error < 1e-4, "{error:e}");
}

/// On `ρ`'s upper bound, `1e-6` of the box from it, the quotes pull `ρ` inward: the stop re-seats
/// it `0.005` of the box inside and names it, where the first-order check alone passes.
#[test]
fn a_stop_on_a_bound_the_cost_pulls_away_from_is_reseated() {
  let (s, r, q, tau) = (2488.11, 0.01213, 0.01884, 28.0 / 365.0);
  let calibrator = CgmysvCalibrator::new(s, r, q, synthetic(s, r, q, tau));
  let mut natural = to_vec(&known_params());
  natural[6] = 5.0 - 1e-5;
  let mut problem = CgmysvProblem::new(&calibrator, natural.clone());
  let (start, first) = problem.reseated(None).expect("ρ is pulled inward");
  assert_eq!(first, 6);
  assert!((start[6] - (5.0 - 0.005 * 10.0)).abs() < 1e-12, "{start:?}");
  let others = (0..N_PARAMS).filter(|&j| j != 6);
  assert!(
    others
      .into_iter()
      .all(|j| (start[j] - natural[j]).abs() < 1e-12 * natural[j].abs())
  );
  let mut inside = to_vec(&known_params());
  inside[6] = 0.0;
  assert!(
    CgmysvProblem::new(&calibrator, inside)
      .reseated(None)
      .is_none()
  );
}

/// From `ρ₀ = 5` the steps throw `ρ` onto `−5`, where its logistic slope rounds to 0 while the
/// quotes pull it inward: unseated, the fit ends there `converged` inside the 150 iterations.
#[test]
fn a_coordinate_rounded_onto_its_bound_is_still_reseated() {
  let (s, r, q, tau) = (2488.11, 0.01213, 0.01884, 28.0 / 365.0);
  let calibrator = CgmysvCalibrator::new(s, r, q, synthetic(s, r, q, tau))
    .with_options(CalibrationOptions::default().with_max_iterations(150));
  let mut start = default_params();
  start[6] = 5.0;
  let fit = calibrator
    .calibrate(Some(to_cgmysv_params(&start)))
    .unwrap();
  let on_bound = fit.params.rho + 5.0 < 1e-3 * 10.0;
  assert!(
    !(fit.converged && on_bound),
    "{:?}: {:?}",
    fit.message,
    fit.params
  );
}

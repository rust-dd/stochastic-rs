use super::*;
use basin::CostFunction;
use basin::Gradient;

use crate::LossMetric;
use crate::QuantError;
use crate::pricing::sabr::SabrParams;
use crate::traits::Calibrator;

fn bounded_problem() -> objective::SabrSmileProblem {
  objective::SabrSmileProblem {
    s: 1.0,
    r_d: 0.02,
    r_f: 0.01,
    tau: 0.5,
    beta: 1.0,
    sigma_atm: 0.2,
    sigma_rr: 0.01,
    sigma_bf: 0.002,
    bounds_lo: vec![0.5, 0.5, 0.5, 0.5, 0.01, -0.99],
    bounds_hi: vec![2.0, 2.0, 2.0, 2.0, 10.0, 0.99],
  }
}

#[test]
fn bounded_sabr_objective_clamps_strikes_and_model_parameters() {
  let problem = bounded_problem();
  let outside = vec![3.0, -0.1, 3.0, -0.1, 0.6, 1.2];
  let boundary = vec![2.0, 0.5, 2.0, 0.5, 0.6, 0.99];
  assert_eq!(
    problem.cost(&outside).unwrap(),
    problem.cost(&boundary).unwrap()
  );
  assert_eq!(
    problem.gradient(&outside).unwrap(),
    problem.gradient(&boundary).unwrap()
  );
}

/// A NaN coordinate prices nothing: the cost and every partial are NaN, not a bound's value.
#[test]
fn a_nan_coordinate_costs_nan_and_has_a_nan_gradient() {
  let problem = bounded_problem();
  for axis in 0..objective::NVARS {
    let mut x = vec![1.0, 1.0, 1.0, 1.0, 0.6, 0.1];
    x[axis] = f64::NAN;
    assert!(problem.cost(&x).unwrap().is_nan(), "axis {axis}");
    let gradient = problem.gradient(&x).unwrap();
    assert!(
      gradient.iter().all(|g| g.is_nan()),
      "axis {axis}: {gradient:?}"
    );
  }
}

#[test]
fn bounded_sabr_gradient_retains_the_derivative_at_the_upper_bound() {
  let problem = bounded_problem();
  let x = vec![2.0, 0.9, 1.1, 0.9, 0.6, 0.5];
  let mut inside = x.clone();
  inside[0] -= 1e-6;
  let expected = (problem.cost(&x).unwrap() - problem.cost(&inside).unwrap()) / (x[0] - inside[0]);
  let actual = problem.gradient(&x).unwrap()[0];
  assert!(expected.abs() > 1e-6);
  assert!((actual - expected).abs() < 1e-4 * expected.abs());
}

#[test]
fn bounded_sabr_low_spot_calibration_returns_consistent_results() {
  let quotes = SabrSmileQuotes {
    tau: 0.5,
    sigma_atm: 0.2,
    sigma_rr: 0.01,
    sigma_bf: 0.002,
  };
  for iterations in [0, 2] {
    let calibrator = SabrSmileCalibrator::new(0.05, 0.02, 0.01, 1.0, quotes)
      .with_strike_bounds(0.02, 0.08)
      .with_basin_hopping_iters(iterations, iterations);
    let result = calibrator.calibrate(None).unwrap();
    let k = result.diagnostics;
    let x = vec![
      k.k_rr_call,
      k.k_rr_put,
      k.k_bf_call,
      k.k_bf_put,
      result.params.nu,
      result.params.rho,
    ];
    let mut problem = bounded_problem();
    problem.s = calibrator.s();
    problem.bounds_lo[..4].fill(calibrator.strike_lo);
    problem.bounds_hi[..4].fill(calibrator.strike_hi);
    for i in 0..x.len() {
      assert!((problem.bounds_lo[i]..=problem.bounds_hi[i]).contains(&x[i]));
    }
    let objective = problem.cost(&x).unwrap();
    assert!(objective.is_finite());
    assert_eq!(result.message, Some(format!("objective {objective:e}")));
    assert_eq!(result.converged, objective < calibrator.success_tol);
  }
}

/// With no hop the fit stays at its start and reads back the `(ν, ρ)` it was seeded with.
#[test]
fn a_start_seeds_nu_and_rho() {
  let quotes = SabrSmileQuotes {
    tau: 0.5,
    sigma_atm: 0.2,
    sigma_rr: 0.01,
    sigma_bf: 0.002,
  };
  let start = SabrParams {
    alpha: 0.2,
    beta: 0.5,
    rho: -0.4,
    nu: 1.2,
  };
  let fit = SabrSmileCalibrator::new(1.0, 0.02, 0.01, 1.0, quotes)
    .with_basin_hopping_iters(0, 0)
    .calibrate(Some(start))
    .unwrap();
  assert_eq!(
    (fit.params.nu, fit.params.rho, fit.params.beta),
    (1.2, -0.4, 1.0)
  );
}

fn norm_pdf(x: f64) -> f64 {
  (-0.5 * x * x).exp() / std::f64::consts::TAU.sqrt()
}

/// `rmse()` is the root-mean-square of the ATM, RR and BF quote errors in vol, the BF one being the
/// strangle-premium mismatch over the strangle's vega at `σ_ATM + σ_BF`.
#[test]
fn rmse_is_the_root_mean_square_of_the_three_quote_errors_in_vol() {
  use crate::pricing::sabr::bs_price_fx;
  use crate::pricing::sabr::forward_fx;
  use crate::pricing::sabr::hagan_implied_vol;
  use crate::pricing::sabr::model_price_hagan;
  use crate::traits::CalibrationResult;

  let quotes = SabrSmileQuotes {
    tau: 0.5,
    sigma_atm: 0.2,
    sigma_rr: 0.01,
    sigma_bf: 0.002,
  };
  let (s, r_d, r_f) = (1.0, 0.02, 0.01);
  let fit = SabrSmileCalibrator::new(s, r_d, r_f, 1.0, quotes)
    .with_basin_hopping_iters(0, 0)
    .calibrate(None)
    .unwrap();
  let (k, p) = (fit.diagnostics, fit.params);
  let f = forward_fx(s, quotes.tau, r_d, r_f);
  let vol = |strike: f64| hagan_implied_vol(strike, f, quotes.tau, &p);
  let sigma_ref = quotes.sigma_atm + quotes.sigma_bf;
  let (mc, _) = model_price_hagan(s, k.k_bf_call, r_d, r_f, quotes.tau, &p);
  let (_, mp) = model_price_hagan(s, k.k_bf_put, r_d, r_f, quotes.tau, &p);
  let (bc, _) = bs_price_fx(s, k.k_bf_call, r_d, r_f, quotes.tau, sigma_ref);
  let (_, bp) = bs_price_fx(s, k.k_bf_put, r_d, r_f, quotes.tau, sigma_ref);
  let vega = |strike: f64| {
    let root = quotes.tau.sqrt();
    let d1 = ((f / strike).ln() + 0.5 * sigma_ref * sigma_ref * quotes.tau) / (sigma_ref * root);
    s * (-r_f * quotes.tau).exp() * norm_pdf(d1) * root
  };
  let bf_error = ((mc + mp) - (bc + bp)) / (vega(k.k_bf_call) + vega(k.k_bf_put));
  let atm_error = vol(f) - quotes.sigma_atm;
  let rr_error = vol(k.k_rr_call) - vol(k.k_rr_put) - quotes.sigma_rr;
  assert!(bf_error.abs() > 1e-6, "the case must exercise the BF term");
  let mse = (atm_error.powi(2) + rr_error.powi(2) + bf_error.powi(2)) / 3.0;
  assert!(
    (fit.rmse() - mse.sqrt()).abs() < 1e-12,
    "{} vs {}",
    fit.rmse(),
    mse.sqrt()
  );
  assert!((fit.loss.get(LossMetric::Mse) - mse).abs() < 1e-12);
}

#[test]
fn a_start_outside_the_search_box_is_an_error_not_a_clamp() {
  let quotes = SabrSmileQuotes {
    tau: 0.5,
    sigma_atm: 0.2,
    sigma_rr: 0.01,
    sigma_bf: 0.002,
  };
  let calibrator = SabrSmileCalibrator::new(1.0, 0.02, 0.01, 1.0, quotes);
  for (field, nu, rho) in [
    ("rho", 0.5, 5.0),
    ("rho", 0.5, -0.995),
    ("nu", 20.0, 0.0),
    ("nu", 0.0, 0.0),
  ] {
    let start = SabrParams {
      alpha: 0.2,
      beta: 1.0,
      rho,
      nu,
    };
    let err = calibrator.calibrate(Some(start)).unwrap_err();
    assert!(
      matches!(&err, QuantError::InvalidInput { what: "start", reason } if reason.starts_with(field)),
      "{err}"
    );
  }
}

#[test]
fn bad_quotes_and_a_non_finite_start_are_errors() {
  let quotes = SabrSmileQuotes {
    tau: 0.5,
    sigma_atm: 0.2,
    sigma_rr: 0.01,
    sigma_bf: 0.002,
  };
  for (field, bad) in [
    ("tau", SabrSmileQuotes { tau: 0.0, ..quotes }),
    (
      "sigma_atm",
      SabrSmileQuotes {
        sigma_atm: f64::NAN,
        ..quotes
      },
    ),
    (
      "sigma_rr",
      SabrSmileQuotes {
        sigma_rr: f64::INFINITY,
        ..quotes
      },
    ),
  ] {
    let err = SabrSmileCalibrator::new(1.0, 0.02, 0.01, 1.0, bad)
      .calibrate(None)
      .unwrap_err();
    assert!(
      matches!(&err, QuantError::InvalidInput { what: "smile quotes", reason } if reason.starts_with(field)),
      "{err}"
    );
  }
  let start = SabrParams {
    alpha: 0.2,
    beta: 1.0,
    rho: f64::NAN,
    nu: 0.5,
  };
  let err = SabrSmileCalibrator::new(1.0, 0.02, 0.01, 1.0, quotes)
    .calibrate(Some(start))
    .unwrap_err();
  assert!(
    matches!(err, QuantError::InvalidInput { what: "start", .. }),
    "{err}"
  );
}

/// The strangle is quoted at `σ_ATM + σ_BF`, so a quote putting it at or below 0 is `InvalidInput`.
#[test]
fn a_non_positive_strangle_vol_is_an_input_error() {
  let quotes = SabrSmileQuotes {
    tau: 0.5,
    sigma_atm: 0.2,
    sigma_rr: 0.01,
    sigma_bf: -0.2,
  };
  let err = SabrSmileCalibrator::new(1.0, 0.02, 0.01, 1.0, quotes)
    .with_basin_hopping_iters(0, 0)
    .calibrate(None)
    .unwrap_err();
  assert_eq!(
    err,
    QuantError::InvalidInput {
      what: "smile quotes",
      reason: "sigma_atm + sigma_bf must satisfy `sigma_atm + sigma_bf > 0.0`, got sigma_atm + \
               sigma_bf = 0.0"
        .to_string(),
    }
  );
}

/// Strikes held 50 % above the forward at `σ√τ ≈ 0.006` underflow the strangle's vega to zero:
/// `Numerical`, not a NaN BF error inside `Ok`.
#[test]
fn an_underflowing_strangle_vega_is_a_numerical_error() {
  let quotes = SabrSmileQuotes {
    tau: 1e-3,
    sigma_atm: 0.2,
    sigma_rr: 0.01,
    sigma_bf: 0.002,
  };
  let err = SabrSmileCalibrator::new(1.0, 0.02, 0.01, 1.0, quotes)
    .with_strike_bounds(1.5, 2.0)
    .with_basin_hopping_iters(0, 0)
    .calibrate(None)
    .unwrap_err();
  assert!(
    matches!(&err, QuantError::Numerical { what: "strangle vega", reason } if reason.starts_with("it is 0.0 at strikes 1.5 and 1.5")),
    "{err}"
  );
}

#[test]
#[should_panic(expected = "beta must satisfy `(0.0..=1.0).contains(&beta)`, got beta = 1.5")]
fn a_beta_outside_the_unit_interval_panics_in_new() {
  let quotes = SabrSmileQuotes {
    tau: 0.5,
    sigma_atm: 0.2,
    sigma_rr: 0.01,
    sigma_bf: 0.002,
  };
  let _ = SabrSmileCalibrator::new(1.0, 0.02, 0.01, 1.5, quotes);
}

#[test]
fn test_sabr_smile_calibrate() {
  let r_usd = 0.022_f64;
  let r_brl = 0.065_f64;
  let s = 3.724_f64;
  let beta = 1.0;

  let cases: [(&str, SabrSmileQuotes); 8] = [
    (
      "ON",
      SabrSmileQuotes {
        tau: 1.0 / 365.0,
        sigma_atm: 20.98 / 100.0,
        sigma_rr: 1.2 / 100.0,
        sigma_bf: 0.15 / 100.0,
      },
    ),
    (
      "1W",
      SabrSmileQuotes {
        tau: 7.0 / 365.0,
        sigma_atm: 13.91 / 100.0,
        sigma_rr: 1.3 / 100.0,
        sigma_bf: 0.20 / 100.0,
      },
    ),
    (
      "2W",
      SabrSmileQuotes {
        tau: 14.0 / 365.0,
        sigma_atm: 13.75 / 100.0,
        sigma_rr: 1.4 / 100.0,
        sigma_bf: 0.20 / 100.0,
      },
    ),
    (
      "1M",
      SabrSmileQuotes {
        tau: 30.0 / 365.0,
        sigma_atm: 14.24 / 100.0,
        sigma_rr: 1.5 / 100.0,
        sigma_bf: 0.22 / 100.0,
      },
    ),
    (
      "2M",
      SabrSmileQuotes {
        tau: 60.0 / 365.0,
        sigma_atm: 13.84 / 100.0,
        sigma_rr: 1.75 / 100.0,
        sigma_bf: 0.27 / 100.0,
      },
    ),
    (
      "3M",
      SabrSmileQuotes {
        tau: 90.0 / 365.0,
        sigma_atm: 13.82 / 100.0,
        sigma_rr: 2.0 / 100.0,
        sigma_bf: 0.32 / 100.0,
      },
    ),
    (
      "6M",
      SabrSmileQuotes {
        tau: 180.0 / 365.0,
        sigma_atm: 13.82 / 100.0,
        sigma_rr: 2.4 / 100.0,
        sigma_bf: 0.43 / 100.0,
      },
    ),
    (
      "1Y",
      SabrSmileQuotes {
        tau: 1.0,
        sigma_atm: 13.94 / 100.0,
        sigma_rr: 2.9 / 100.0,
        sigma_bf: 0.55 / 100.0,
      },
    ),
  ];

  let results = SabrSmileCalibrator::calibrate_many(s, r_brl, r_usd, beta, &cases).unwrap();
  assert_eq!(results.len(), cases.len());
  for ((label, _), fit) in cases.iter().zip(&results) {
    assert!(fit.converged, "tenor {label}: {:?}", fit.message);
  }
}

#[test]
fn a_strike_for_an_uncomputable_delta_is_nan_not_a_bracket_edge() {
  let params = crate::pricing::sabr::SabrParams {
    alpha: 0.2,
    beta: 1.0,
    rho: -0.3,
    nu: 0.5,
  };
  assert!(strike_for_delta(100.0, 0.02, 0.01, -1.0, params, 0.25, 1.0).is_nan());
  assert!(strike_for_delta(100.0, 0.02, 0.01, 1.0, params, 0.25, 1.0).is_finite());
}

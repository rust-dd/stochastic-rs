use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_distributions::special::norm_cdf;

use super::calibrator::clean_local_vol;
use super::*;
use crate::calibration::heston::HestonParams;
use crate::pricing::fourier::HestonFourier;
use crate::pricing::heston::HestonPricer;
use crate::pricing::slv::FokkerPlanckMethod;
use crate::pricing::slv::ParticleMethod;
use crate::traits::CalibrationResult;
use crate::traits::Calibrator;
use crate::traits::ModelPricer;
use crate::traits::ToModel;

const S: f64 = 100.0;
const R: f64 = 0.02;
const Q: f64 = 0.005;

fn heston() -> HestonParams {
  HestonParams {
    v0: 0.04,
    kappa: 2.0,
    theta: 0.05,
    sigma: 0.4,
    rho: -0.6,
  }
}

/// A Heston vanilla surface on a strike grid fine enough for the Dupire
/// stencil and a maturity grid dense enough for its time derivative.
fn heston_surface(strikes: &[f64], maturities: &[f64]) -> Array2<f64> {
  let p = heston();
  let model = HestonFourier {
    v0: p.v0,
    kappa: p.kappa,
    theta: p.theta,
    sigma: p.sigma,
    rho: p.rho,
    r: R,
    q: Q,
  };
  Array2::from_shape_fn((maturities.len(), strikes.len()), |(j, i)| {
    model.price_call(S, strikes[i], R, Q, maturities[j])
  })
}

fn fine_strikes() -> Vec<f64> {
  Array1::linspace(60.0, 150.0, 91).to_vec()
}

fn dense_maturities() -> Vec<f64> {
  (1..=10).map(|k| 0.1 * k as f64).collect()
}

fn method() -> ParticleMethod {
  ParticleMethod::default()
    .with_particles(20_000)
    .with_steps_per_year(100)
    .with_seed(2026)
}

fn calibrator(eta: f64) -> HestonSlvCalibrator {
  let (strikes, maturities) = (fine_strikes(), dense_maturities());
  let calls = heston_surface(&strikes, &maturities);
  HestonSlvCalibrator::new(S, R, Q, strikes, maturities, calls)
    .with_mixing(eta)
    .with_heston_params(heston())
    .with_particle_method(method())
}

/// Gyöngy: under the parameters that generated the surface, with the full
/// vol-of-vol, the Heston model already reproduces its own Dupire local
/// volatility, so the leverage is `1` wherever the cloud and the Dupire
/// stencil are both well resolved — and the cloud reprices the surface.
#[test]
fn a_heston_surface_under_its_own_parameters_gives_unit_leverage_and_reprices() {
  let result = calibrator(1.0).calibrate(None).unwrap();
  assert!(result.converged());
  assert!(result.heston.is_none(), "the parameters were pinned");
  let lev = result.leverage();
  let mut central = Vec::new();
  for (j, &t) in lev.times().iter().enumerate() {
    if t < 0.2 {
      continue;
    }
    for (i, &k) in lev.spots().iter().enumerate() {
      if (85.0..=115.0).contains(&k) {
        central.push((lev.values()[[j, i]] - 1.0).abs());
      }
    }
  }
  central.sort_by(f64::total_cmp);
  let median = central[central.len() / 2];
  let worst = central[central.len() - 1];
  assert!(
    median < 0.05,
    "median |L - 1| over the central strikes is {median}"
  );
  assert!(
    worst < 0.25,
    "worst |L - 1| over the central strikes is {worst}"
  );
  assert!(
    result.rmse() < 0.25,
    "repricing rmse {} on a spot of 100",
    result.rmse()
  );
  assert!(
    result.max_error() < 1.0,
    "worst repricing error {}",
    result.max_error()
  );
}

/// The same Gyöngy check through the forward Kolmogorov equation: the
/// finite-volume leverage is unit in the centre and its marginal reprices
/// the surface, with no Monte Carlo noise in either.
#[test]
fn the_fokker_planck_route_gives_unit_leverage_and_reprices() {
  let result = calibrator(1.0)
    .with_fokker_planck(
      FokkerPlanckMethod::default()
        .with_nodes(161, 80)
        .with_steps_per_year(100),
    )
    .calibrate(None)
    .unwrap();
  assert!(result.converged());
  let lev = result.leverage();
  let mut central = Vec::new();
  for (j, &t) in lev.times().iter().enumerate() {
    if t < 0.2 {
      continue;
    }
    for (i, &k) in lev.spots().iter().enumerate() {
      if (85.0..=115.0).contains(&k) {
        central.push((lev.values()[[j, i]] - 1.0).abs());
      }
    }
  }
  central.sort_by(f64::total_cmp);
  let median = central[central.len() / 2];
  let worst = central[central.len() - 1];
  assert!(
    median < 0.05,
    "median |L - 1| over the central strikes is {median}"
  );
  assert!(
    worst < 0.25,
    "worst |L - 1| over the central strikes is {worst}"
  );
  assert!(result.rmse() < 0.25, "repricing rmse {}", result.rmse());
  assert!(
    result.max_error() < 1.0,
    "worst repricing error {}",
    result.max_error()
  );
}

/// Half the vol-of-vol moves the leverage away from one and the surface is
/// still reproduced: the mixing fraction is free.
#[test]
fn half_mixing_still_reprices_the_surface() {
  let result = calibrator(0.5).calibrate(None).unwrap();
  assert!(result.converged());
  assert_eq!(result.slv_params().eta, 0.5);
  assert!(result.rmse() < 0.25, "repricing rmse {}", result.rmse());
  let lev = result.leverage();
  let atm_late = lev.interpolate(100.0, 1.0);
  assert!(
    (0.5..1.5).contains(&atm_late),
    "the leverage stays a moderate correction, got {atm_late}"
  );
}

/// A flat local volatility under a variance that never moves is
/// Black–Scholes: `L ≡ 1` exactly and the cloud reprices Black calls to its
/// Monte Carlo error.
#[test]
fn a_supplied_local_vol_bypasses_dupire() {
  let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
  let maturities = vec![0.5, 1.0];
  let vol = 0.2;
  let black = |k: f64, tau: f64| {
    let d1 = ((S / k).ln() + (R - Q + 0.5 * vol * vol) * tau) / (vol * tau.sqrt());
    let d2 = d1 - vol * tau.sqrt();
    S * (-Q * tau).exp() * norm_cdf(d1) - k * (-R * tau).exp() * norm_cdf(d2)
  };
  let calls = Array2::from_shape_fn((2, 5), |(j, i)| black(strikes[i], maturities[j]));
  let pinned = HestonParams {
    v0: vol * vol,
    kappa: 1.0,
    theta: vol * vol,
    sigma: 0.3,
    rho: -0.5,
  };
  let result = HestonSlvCalibrator::new(S, R, Q, strikes, maturities, calls)
    .with_mixing(0.0)
    .with_heston_params(pinned)
    .with_local_vol(Array2::from_elem((2, 5), vol))
    .with_particle_method(method())
    .calibrate(None)
    .unwrap();
  assert!(result.converged());
  assert!(
    result
      .leverage()
      .values()
      .iter()
      .all(|l| (l - 1.0).abs() < 1e-12),
    "flat local vol at sqrt(v0) is unit leverage"
  );
  assert!(result.rmse() < 0.15, "rmse {}", result.rmse());
}

#[test]
fn the_heston_fit_runs_when_the_parameters_are_not_pinned() {
  let strikes = Array1::linspace(80.0, 120.0, 9).to_vec();
  let maturities = vec![0.25, 0.5, 1.0];
  let calls = heston_surface(&strikes, &maturities);
  let guess = HestonParams {
    v0: 0.05,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.5,
    rho: -0.5,
  };
  let result = HestonSlvCalibrator::new(S, R, Q, strikes, maturities, calls)
    .with_particle_method(method().with_particles(2_000))
    .calibrate(Some(guess))
    .unwrap();
  let fit = result.heston.as_ref().expect("the parameters were fitted");
  let p = &fit.params;
  assert!(p.kappa > 0.0 && p.theta > 0.0 && p.sigma > 0.0 && p.rho.abs() < 1.0);
  assert!((p.v0 - 0.04).abs() < 0.02, "v0 = {}", p.v0);
  assert_eq!(result.slv_params().kappa, p.kappa);
  assert!(result.rmse().is_finite());
}

#[test]
fn to_model_prices_at_the_calibration_rates() {
  let result = calibrator(1.0)
    .with_particle_method(method().with_particles(2_000))
    .calibrate(None)
    .unwrap();
  let pricer = result
    .to_model(R, Q)
    .with_paths(4_000)
    .with_steps_per_year(50);
  let c = pricer.price_call(S, 100.0, R, Q, 0.5);
  assert!(c.is_finite() && c > 0.0);
  let exact =
    HestonPricer::new(0.04, -0.6, 2.0, 0.05, 0.4, Some(0.0)).price_call(S, 100.0, R, Q, 0.5);
  assert!((c - exact).abs() < 1.0, "slv {c} vs heston {exact}");
  let via_trait = ToModel::to_model(&result, R, Q);
  assert_eq!(via_trait.calibration_rates, Some((R, Q)));
}

#[test]
#[should_panic(expected = "calibrated at r=0.02, q=0.005 but a pricer at r=0.03, q=0.005")]
fn to_model_rejects_other_rates() {
  let result = calibrator(1.0)
    .with_particle_method(method().with_particles(500))
    .calibrate(None)
    .unwrap();
  let _ = result.to_model(0.03, Q);
}

#[test]
fn bad_inputs_are_errors() {
  let strikes = vec![90.0, 100.0, 110.0];
  let maturities = vec![0.5, 1.0];
  let calls = Array2::from_elem((2, 3), 5.0);
  let base = || {
    HestonSlvCalibrator::new(S, R, Q, strikes.clone(), maturities.clone(), calls.clone())
      .with_heston_params(heston())
      .with_particle_method(method().with_particles(10))
  };
  assert!(base().with_mixing(1.5).calibrate(None).is_err());
  assert!(base().with_mixing(-0.1).calibrate(None).is_err());
  assert!(
    base()
      .with_local_vol(Array2::zeros((3, 3)))
      .calibrate(None)
      .is_err()
  );
  let mut two_strikes = base();
  two_strikes.strikes = vec![90.0, 100.0];
  two_strikes.calls = Array2::from_elem((2, 2), 5.0);
  assert!(two_strikes.calibrate(None).is_err());
  let mut descending = base();
  descending.strikes = vec![110.0, 100.0, 90.0];
  assert!(descending.calibrate(None).is_err());
  let mut shape = base();
  shape.calls = Array2::from_elem((3, 3), 5.0);
  assert!(shape.calibrate(None).is_err());
  let mut one_maturity = base();
  one_maturity.maturities = vec![0.5];
  one_maturity.calls = Array2::from_elem((1, 3), 5.0);
  assert!(
    one_maturity.calibrate(None).is_err(),
    "Dupire's time derivative needs two maturities"
  );
  assert!(
    one_maturity
      .with_local_vol(Array2::from_elem((1, 3), 0.2))
      .calibrate(None)
      .is_ok(),
    "a supplied local volatility needs no time derivative"
  );
  let err = base().calibrate(None).unwrap_err();
  assert!(
    err.to_string().contains("no admissible cell"),
    "a flat call slice has no admissible Dupire cell, got: {err}"
  );
}

#[test]
fn dupire_cleaning_fills_the_boundary_and_the_holes_and_refuses_an_empty_row() {
  let nan = f64::NAN;
  let raw = Array2::from_shape_vec(
    (2, 6),
    vec![nan, 0.2, nan, nan, 0.26, nan, nan, nan, 0.3, 0.31, nan, nan],
  )
  .unwrap();
  let cleaned = clean_local_vol(raw, &[0.5, 1.0]).unwrap();
  let row0 = cleaned.row(0).to_vec();
  assert_eq!(row0, vec![0.2, 0.2, 0.22, 0.24, 0.26, 0.26]);
  assert_eq!(
    cleaned.row(1).to_vec(),
    vec![0.3, 0.3, 0.3, 0.31, 0.31, 0.31]
  );
  let empty = Array2::from_elem((1, 3), nan);
  let err = clean_local_vol(empty, &[0.75]).unwrap_err();
  assert!(err.to_string().contains("0.75"));
}

// docs: quant#heston-slv-leverage-calibration-by-the-particle-method
//! Backs the Heston SLV calibration example on the quant catalog page: a
//! vanilla call surface in, a leverage-calibrated model and its Monte Carlo
//! pricer out.

use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs::quant::calibration::heston::HestonParams;
use stochastic_rs::quant::calibration::heston_slv::HestonSlvCalibrator;
use stochastic_rs::quant::pricing::fourier::HestonFourier;
use stochastic_rs::quant::pricing::slv::ParticleMethod;
use stochastic_rs::traits::CalibrationResult;
use stochastic_rs::traits::Calibrator;
use stochastic_rs::traits::ModelPricer;

#[test]
fn heston_slv_calibrates_a_vanilla_surface_and_prices_under_half_mixing() {
  let (s, r, q) = (100.0, 0.02, 0.0);
  let heston = HestonParams {
    v0: 0.04,
    kappa: 2.0,
    theta: 0.05,
    sigma: 0.4,
    rho: -0.6,
  };
  // The vanilla surface — generated from a Heston model here, market quotes
  // in practice: `calls[[j, i]]` is the call at `strikes[i]`, `maturities[j]`.
  let strikes = Array1::linspace(70.0, 140.0, 36).to_vec();
  let maturities = vec![0.25, 0.5, 0.75, 1.0];
  let model = HestonFourier {
    v0: heston.v0,
    kappa: heston.kappa,
    theta: heston.theta,
    sigma: heston.sigma,
    rho: heston.rho,
    r,
    q,
  };
  let calls = Array2::from_shape_fn((maturities.len(), strikes.len()), |(j, i)| {
    model.price_call(s, strikes[i], r, q, maturities[j])
  });

  // Half the vol-of-vol; the leverage makes up the difference so the surface
  // is reproduced. The Heston fit is pinned here — leave it out to have
  // `HestonCalibrator` fit the same quotes.
  let result = HestonSlvCalibrator::new(s, r, q, strikes, maturities, calls)
    .with_mixing(0.5)
    .with_heston_params(heston)
    .with_particle_method(
      ParticleMethod::default()
        .with_particles(5_000)
        .with_steps_per_year(50)
        .with_seed(7),
    )
    .calibrate(None)
    .unwrap();
  assert!(result.converged());
  assert!(result.rmse() < 1.0, "in-sample repricing rmse {}", result.rmse());
  assert!(result.leverage().covers(100.0, 0.5));

  // The pricer is anchored to the calibration rates and reprices the market.
  let pricer = result.to_model(r, q).with_paths(4_000).with_steps_per_year(50);
  let call = pricer.price_call(s, 100.0, r, q, 0.5);
  let market = model.price_call(s, 100.0, r, q, 0.5);
  assert!((call - market).abs() < 1.0, "slv {call} vs market {market}");
}

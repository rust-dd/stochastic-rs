// docs: tutorials/svi-volatility-surface
//! Fits raw SVI to one maturity of a Heston smile and checks the fit for butterfly arbitrage.

use stochastic_rs::quant::pricing::heston::HestonPricer;
use stochastic_rs::quant::vol_surface::ModelSurface;

#[test]
fn raw_svi_fits_a_heston_smile() {
  // The smile to fit: Black implied vols of a Heston model at nine strikes, one year out.
  let model = HestonPricer::new(0.04, -0.5, 2.0, 0.04, 0.3, None);
  let strikes = (0..9).map(|i| 80.0 + 5.0 * i as f64).collect::<Vec<_>>();
  let surface = model.vol_surface(100.0, 0.02, 0.0, &strikes, &[1.0]);
  let smile = surface.smile_slice(0); // log-forward moneyness k and total variance w = iv^2 t

  let svi = smile.fit_svi(None);
  assert!(svi.is_admissible());
  for (&k, &w) in smile.log_moneyness.iter().zip(&smile.total_variance) {
    assert!((svi.total_variance(k) - w).abs() < 1e-5);
  }

  // Durrleman's g(k) stays non-negative on a wide grid: no butterfly arbitrage.
  let grid = (-100..=100).map(|i| 0.01 * i as f64).collect::<Vec<_>>();
  assert!(svi.is_butterfly_arb_free(&grid));

  // Jump-wings view of the slice: half the ATM slope w'(0), negative for a downward skew.
  assert!(svi.jump_wings(1.0).psi_t < 0.0);
}

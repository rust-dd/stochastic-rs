// docs: quant#heston-pde-by-adi-finite-differences
//! Backs the Heston ADI example on the quant catalog page.

use stochastic_rs::quant::pricing::heston::HestonPricer;
use stochastic_rs::quant::pricing::heston_adi::AdiScheme;
use stochastic_rs::quant::pricing::heston_adi::HestonAdiPricer;
use stochastic_rs::traits::ModelPricer;

#[test]
fn adi_solver_reprices_the_semi_analytic_heston_call() {
  // Case 1 of in 't Hout & Foulon: κ = 1.5, η = 0.04, σ = 0.3, ρ = −0.9, r_d = 2.5 %, one year.
  let (v0, kappa, eta, sigma, rho) = (0.04, 1.5, 0.04, 0.3, -0.9);
  let (s, k, r_d, r_f, tau) = (100.0, 100.0, 0.025, 0.0, 1.0);

  // Model + numerics state; the query travels as arguments (r = r_d, q = r_f).
  let pde = HestonAdiPricer::new(v0, kappa, eta, sigma, rho)
    .with_grid(100, 50, 50)
    .with_scheme(AdiScheme::ModifiedCraigSneyd);
  let adi = pde.price_call(s, k, r_d, r_f, tau);

  // Semi-analytic reference through the crate's Heston pricer.
  let analytic = HestonPricer::new(v0, rho, kappa, eta, sigma, Some(0.0))
    .call_put(s, k, r_d, r_f, tau)
    .0;
  assert!(
    (adi - analytic).abs() / analytic < 1e-2,
    "adi {adi} vs analytic {analytic}"
  );

  // The same grid prices a down-and-out call by moving the lower boundary to the barrier.
  let knocked = pde.with_barrier(90.0).price_call(s, k, r_d, r_f, tau);
  assert!(knocked > 0.0 && knocked < adi);
}

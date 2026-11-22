//! The Fourier quadrature accepts a panel on the tanh-sinh rule's last level difference: the cells
//! the squared difference passed unresolved, against Lewis's integral in mpmath at 30 digits.

use stochastic_rs_quant::pricing::heston::HestonParams;
use stochastic_rs_quant::pricing::heston::HestonPricer;
use stochastic_rs_quant::traits::ModelPricer;

const fn heston(v0: f64, kappa: f64, theta: f64, sigma: f64, rho: f64) -> HestonParams {
  HestonParams {
    v0,
    kappa,
    theta,
    sigma,
    rho,
  }
}

const BASE: HestonParams = heston(0.04, 1.5, 0.04, 0.3, -0.7);
const LOW_V0: HestonParams = heston(0.0025, 1.5, 0.04, 0.3, -0.7);
const WIDE: HestonParams = heston(0.5, 3.0, 0.5, 8.0, -0.99);

fn call(params: HestonParams, s: f64, k: f64, tau: f64) -> f64 {
  HestonPricer::new(params).price_call(s, k, 0.05, 0.0, tau)
}

/// `(params, s, k, τ, call)` at `r = 0.05`, `q = 0`. The squared difference left the first seven
/// `1.8e-5`, `2.3e-6`, `3.4e-7`, `4.3e-7`, `6.3e-8`, `6.8e-5`, `1.3e-5` off, and the last `3.1e-9`.
const CELLS: [(HestonParams, f64, f64, f64, f64); 8] = [
  (BASE, 73.5, 100.0, 0.25, 7.156_161_445_821_137e-5),
  (BASE, 86.0, 100.0, 0.1, 3.425_079_336_968_455_6e-3),
  (BASE, 90.0, 100.0, 1.0, 4.388_581_382_492_856),
  (BASE, 100.0, 118.25, 0.02, 4.817_873_950_783_985e-14),
  (BASE, 100.0, 142.5, 0.02, 0.0),
  (BASE, 100.0, 80.75, 0.5, 21.773_933_738_370_95),
  (LOW_V0, 111.5, 100.0, 1.0, 17.849_705_327_826_66),
  (LOW_V0, 71.0, 100.0, 0.25, 3.222_487_123_179_181_5e-12),
];

#[test]
fn heston_cells_the_squared_difference_passed_unresolved_match_mpmath() {
  for (params, s, k, tau, want) in CELLS {
    let got = call(params, s, k, tau);
    assert!(
      (got - want).abs() < 5e-9,
      "s = {s}, k = {k}, tau = {tau}: {got} vs {want}"
    );
  }
}

/// `σ = 8`, `ρ = −0.99` far out of the money 16 hours from expiry, where mpmath's value is below
/// `1e-28`: the squared difference priced it `0.0344`.
#[test]
fn a_vol_of_vol_of_8_far_from_the_money_prices_zero() {
  let got = call(WIDE, 16.46, 100.0, 1.8e-3);
  assert!(got.abs() < 5e-9, "{got}");
}

/// The one cell of a 3,208-strike grid left above `1e-8`, at `1.4e-8`; a halves check on each
/// panel's first acceptance clears it at 2.6 times the evaluations.
#[test]
fn the_cell_at_k_89_stays_within_2e_8() {
  let got = call(BASE, 100.0, 89.0, 0.05);
  assert!((got - 11.237_958_473_906_275).abs() < 2e-8, "{got}");
}

use super::DoubleHestonParams;
use super::loss::double_heston_call_price;
use crate::pricing::heston::HestonParams;
use crate::pricing::heston::HestonPricer;

#[test]
fn vanished_factor_matches_heston_at_short_maturity() {
  let params = DoubleHestonParams {
    first: HestonParams {
      v0: 0.04,
      kappa: 1.5,
      theta: 0.04,
      sigma: 0.3,
      rho: -0.7,
    },
    second: HestonParams {
      v0: 0.0,
      kappa: 1.0,
      theta: 0.0,
      sigma: 0.2,
      rho: 0.0,
    },
  };
  for (k, tau) in [(95.0, 0.01), (100.0, 0.005), (105.0, 0.01)] {
    let double_heston = double_heston_call_price(&params, 100.0, k, 0.05, 0.0, tau);
    let heston = HestonPricer::new(params.first)
      .call_put(100.0, k, 0.05, 0.0, tau)
      .0;

    assert!(
      (double_heston - heston).abs() < 2e-3,
      "K={k}, τ={tau}: one-factor Double Heston={double_heston}, Heston={heston}"
    );
  }
}

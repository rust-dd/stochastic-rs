use super::DoubleHestonParams;
use super::loss::double_heston_call_price;
use crate::pricing::heston::HestonPricer;

#[test]
fn vanished_factor_matches_heston_at_short_maturity() {
  let params = DoubleHestonParams {
    v1_0: 0.04,
    kappa1: 1.5,
    theta1: 0.04,
    sigma1: 0.3,
    rho1: -0.7,
    v2_0: 0.0,
    kappa2: 1.0,
    theta2: 0.0,
    sigma2: 0.2,
    rho2: 0.0,
  };
  for (k, tau) in [(95.0, 0.01), (100.0, 0.005), (105.0, 0.01)] {
    let double_heston = double_heston_call_price(&params, 100.0, k, 0.05, 0.0, tau);
    let heston = HestonPricer::new(
      params.v1_0,
      params.rho1,
      params.kappa1,
      params.theta1,
      params.sigma1,
      Some(0.0),
    )
    .call_put(100.0, k, 0.05, 0.0, tau)
    .0;

    assert!(
      (double_heston - heston).abs() < 2e-3,
      "K={k}, τ={tau}: one-factor Double Heston={double_heston}, Heston={heston}"
    );
  }
}

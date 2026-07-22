use super::SVJParams;
use super::loss::bates_call_price;
use crate::pricing::heston::HestonPricer;
use crate::traits::PricerExt;

#[test]
fn zero_jump_svj_matches_heston_at_short_maturity() {
  let params = SVJParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma_v: 0.3,
    rho: -0.7,
    lambda: 0.0,
    mu_j: 0.0,
    sigma_j: 0.1,
  };
  for (k, tau) in [(95.0, 0.01), (100.0, 0.005), (105.0, 0.01)] {
    let svj = bates_call_price(&params, 100.0, k, 0.05, 0.0, tau);
    let heston = HestonPricer::new(
      100.0,
      params.v0,
      k,
      0.05,
      Some(0.0),
      params.rho,
      params.kappa,
      params.theta,
      params.sigma_v,
      Some(0.0),
      Some(tau),
      None,
      None,
    )
    .calculate_call_put()
    .0;

    assert!(
      (svj - heston).abs() < 2e-3,
      "K={k}, τ={tau}: zero-jump SVJ={svj}, Heston={heston}"
    );
  }
}

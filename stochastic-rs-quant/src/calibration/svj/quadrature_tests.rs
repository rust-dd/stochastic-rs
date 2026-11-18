use super::SvjParams;
use super::loss::bates_call_price;
use crate::pricing::heston::HestonParams;
use crate::pricing::heston::HestonPricer;
use crate::traits::ModelPricer;

#[test]
fn zero_jump_svj_matches_heston_at_short_maturity() {
  let params = SvjParams {
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
    let heston = HestonPricer::new(HestonParams {
      v0: params.v0,
      kappa: params.kappa,
      theta: params.theta,
      sigma: params.sigma_v,
      rho: params.rho,
    })
    .price_call(100.0, k, 0.05, 0.0, tau);

    assert!(
      (svj - heston).abs() < 2e-3,
      "K={k}, τ={tau}: zero-jump SVJ={svj}, Heston={heston}"
    );
  }
}

use super::LevyModelType;
use super::loss::fourier_call_price;
use crate::pricing::BsmCoc;
use crate::pricing::BsmPricer;
use crate::pricing::cf_quadrature::Walk;
use crate::traits::ModelPricer;

#[test]
fn zero_jump_merton_matches_bsm_at_short_maturity() {
  let tau = 0.005;
  let params = [0.2, 0.0, 0.0, 0.1];
  let merton = fourier_call_price(
    LevyModelType::MertonJd,
    &params,
    100.0,
    100.0,
    0.05,
    0.0,
    tau,
    &mut Walk::Adaptive,
  );
  let bsm = BsmPricer::new(params[0], BsmCoc::Merton1973).price_call(100.0, 100.0, 0.05, 0.0, tau);

  assert!(
    (merton - 0.5767009444428197).abs() < 1e-8,
    "Merton={merton}, expected=0.5767009444428197"
  );
  assert!((merton - bsm).abs() < 3e-6, "Merton={merton}, BSM={bsm}");
}

#[test]
fn variance_gamma_short_maturity_matches_converged_reference() {
  let price = fourier_call_price(
    LevyModelType::VarianceGamma,
    &[0.2, -0.1, 0.5],
    100.0,
    100.0,
    0.05,
    0.0,
    0.02,
    &mut Walk::Adaptive,
  );
  assert!(
    (price - 0.5243235081).abs() < 2e-3,
    "short-dated Variance Gamma price={price}"
  );
}

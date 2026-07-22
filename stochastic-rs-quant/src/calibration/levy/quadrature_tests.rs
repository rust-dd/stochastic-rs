use super::LevyModelType;
use super::loss::fourier_call_price;
use crate::OptionType;
use crate::pricing::BSMCoc;
use crate::pricing::BSMPricer;
use crate::traits::PricerExt;

#[test]
fn zero_jump_merton_matches_bsm_at_short_maturity() {
  let tau = 0.005;
  let params = [0.2, 0.0, 0.0, 0.1];
  let merton = fourier_call_price(
    LevyModelType::MertonJD,
    &params,
    100.0,
    100.0,
    0.05,
    0.0,
    tau,
  );
  let bsm = BSMPricer::new(
    100.0,
    params[0],
    100.0,
    0.05,
    None,
    None,
    Some(0.0),
    Some(tau),
    None,
    None,
    OptionType::Call,
    BSMCoc::Merton1973,
  )
  .calculate_call_put()
  .0;

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
  );
  assert!(
    (price - 0.5243235081).abs() < 2e-3,
    "short-dated Variance Gamma price={price}"
  );
}

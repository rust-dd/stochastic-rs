//! The t copula cdfs at `ν` from 0.01 to 0.25 against the χ² mixture `E_S[Φ_R(√(S/ν) t_ν⁻¹(u))]`, `S ~ χ²_ν`, integrated
//! over `ln S`: at `d = 2` by mpmath at 30 and 45 digits, at `d = 3` by adaptive quadrature at two tolerances.

use ndarray::array;
use stochastic_rs_copulas::bivariate::t_copula::TCopula;
use stochastic_rs_copulas::multivariate::t::TMultivariate;
use stochastic_rs_copulas::traits::BivariateExt;
use stochastic_rs_copulas::traits::MultivariateExt;

/// `(ν, ρ, u, v, C)`, the two precisions agreeing to 27 digits; at `ν = 4` the mixture is the bivariate t cdf to `4e-16`.
const BIVARIATE: [(f64, f64, f64, f64, f64); 25] = [
  (0.25, 0.5, 0.3, 0.7, 0.22090860310133786),
  (0.25, 0.5, 0.1, 0.2, 0.06877357223769079),
  (0.25, -0.8, 0.5, 0.5, 0.10241638234956671),
  (0.25, 0.5, 0.9, 0.95, 0.8843866673716592),
  (0.25, -0.8, 0.001, 0.5, 0.00016963931716272533),
  (0.1, 0.5, 0.3, 0.7, 0.2091680162120005),
  (0.1, 0.5, 0.1, 0.2, 0.06770866001605544),
  (0.1, -0.8, 0.5, 0.5, 0.10241638234956671),
  (0.1, 0.5, 0.9, 0.95, 0.8838543300080139),
  (0.1, -0.8, 0.001, 0.5, 0.0001897701969526671),
  (0.05, 0.5, 0.3, 0.7, 0.2047120566915809),
  (0.05, 0.5, 0.1, 0.2, 0.06719687292774162),
  (0.05, -0.8, 0.5, 0.5, 0.10241638234956671),
  (0.05, 0.5, 0.9, 0.95, 0.8835984364638708),
  (0.05, -0.8, 0.001, 0.5, 0.00019712341774374368),
  (0.02, 0.5, 0.3, 0.7, 0.20191666725689525),
  (0.02, 0.5, 0.1, 0.2, 0.06688071068693367),
  (0.02, -0.8, 0.5, 0.5, 0.10241638234956671),
  (0.02, 0.5, 0.9, 0.95, 0.8834403553434669),
  (0.02, -0.8, 0.001, 0.5, 0.00020170482500510146),
  (0.01, 0.5, 0.3, 0.7, 0.20096374034731607),
  (0.01, 0.5, 0.1, 0.2, 0.06677402067548562),
  (0.01, -0.8, 0.5, 0.5, 0.10241638234956671),
  (0.01, 0.5, 0.9, 0.95, 0.8833870103377428),
  (0.01, -0.8, 0.001, 0.5, 0.0002032613010000381),
];

/// `(ν, u, C)` at the correlation `[[1, 0.7, 0.3], [0.7, 1, 0.5], [0.3, 0.5, 1]]`, the two tolerances agreeing to `4e-16`.
const TRIVARIATE: [(f64, [f64; 3], f64); 15] = [
  (0.25, [0.3, 0.5, 0.7], 0.17276136691166671),
  (0.25, [0.1, 0.4, 0.8], 0.053615682383079344),
  (0.25, [0.6, 0.6, 0.6], 0.37293083961439705),
  (0.1, [0.3, 0.5, 0.7], 0.16070621203539365),
  (0.1, [0.1, 0.4, 0.8], 0.05166990913310035),
  (0.1, [0.6, 0.6, 0.6], 0.38717951382490357),
  (0.05, [0.3, 0.5, 0.7], 0.15623316250132524),
  (0.05, [0.1, 0.4, 0.8], 0.05110451321868584),
  (0.05, [0.6, 0.6, 0.6], 0.39442467764447253),
  (0.02, [0.3, 0.5, 0.7], 0.1534590067729726),
  (0.02, [0.1, 0.4, 0.8], 0.050758136657602926),
  (0.02, [0.6, 0.6, 0.6], 0.39900087499064446),
  (0.01, [0.3, 0.5, 0.7], 0.15251872551800147),
  (0.01, [0.1, 0.4, 0.8], 0.050641197034968055),
  (0.01, [0.6, 0.6, 0.6], 0.40054323527261826),
];

/// To a relative `1e-12`, as the bivariate table holds it, and the multivariate copula at `d = 2` bit for bit.
#[test]
fn the_bivariate_t_cdf_holds_its_digits_at_small_nu() {
  for (nu, rho, u, v, want) in BIVARIATE {
    let q = array![[u, v]];
    let got = TCopula::new(rho, nu).cdf(q.view()).unwrap()[0];
    let pair = array![[1.0, rho], [rho, 1.0]];
    let mv = TMultivariate::new(pair, nu).cdf(q.view()).unwrap()[0];
    assert_eq!(mv, got, "ν = {nu}, ρ = {rho}, ({u}, {v})");
    assert!(
      (got - want).abs() <= 1e-12 * want,
      "ν = {nu}, ρ = {rho}: C({u}, {v}) = {got:e}, want {want:e}"
    );
  }
}

/// Within Genz–Bretz's absolute envelope `2.84e-4`, which their χ scale holds once its quantile does.
#[test]
fn the_trivariate_t_cdf_holds_the_genz_envelope_at_small_nu() {
  let corr = array![[1.0, 0.7, 0.3], [0.7, 1.0, 0.5], [0.3, 0.5, 1.0]];
  for (nu, u, want) in TRIVARIATE {
    let got = TMultivariate::new(corr.clone(), nu)
      .cdf(array![[u[0], u[1], u[2]]].view())
      .unwrap()[0];
    assert!(
      (got - want).abs() <= 2.84e-4,
      "ν = {nu}: C({u:?}) = {got:e}, want {want:e}"
    );
  }
}

use std::f64::consts::PI;

use super::*;
use crate::pricing::bsm::BsmCoc;
use crate::pricing::bsm::BsmPricer;
use crate::pricing::heston::HestonParams;
use crate::traits::ModelPricer;

const TOL_FOURIER: f64 = 0.15;
const TOL_TIGHT: f64 = 0.05;

/// Short-dated Heston (τ = 0.005 ATM, τ = 0.01 at K = 105), where a fixed `φ_max = 100` truncates
/// the tail: Gil-Pelaez and Lewis match the converged inversion and each other.
#[test]
fn fourier_pricers_match_converged_short_dated() {
  let model = HestonFourier::new(HestonParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.30,
    rho: -0.7,
  });
  // (K, τ, the call by Lewis quadrature at 30 digits, mpmath)
  let cases = [
    (100.0, 0.005, 0.576_580_780_722_825_6),
    (105.0, 0.01, 0.002_967_139_824_184_934_6),
    (95.0, 0.01, 5.052_617_475_497_76),
  ];
  for (k, t, expected) in cases {
    let gp = GilPelaezPricer::price_call(&model, 100.0, k, 0.05, 0.0, t);
    let lw = LewisPricer::price_call(&model, 100.0, k, 0.05, 0.0, t);
    assert!(
      (gp - expected).abs() < 1e-11,
      "Gil-Pelaez at K={k}, τ={t}: got {gp}, converged {expected}"
    );
    assert!(
      (lw - expected).abs() < 1e-12,
      "Lewis at K={k}, τ={t}: got {lw}, converged {expected}"
    );
    assert!(
      (gp - lw).abs() < 1e-11,
      "Gil-Pelaez {gp} and Lewis {lw} must agree"
    );
  }
}

#[test]
fn carr_madan_bsm_reference() {
  let model = BsmFourier { sigma: 0.15 };
  let pricer = CarrMadanPricer::default();
  let price = pricer.price_call(&model, 100.0, 100.0, 0.05, 0.01, 1.0);
  let expected = 7.94871378854164;
  assert!(
    (price - expected).abs() < TOL_FOURIER,
    "Carr-Madan BSM: got={price}, expected={expected}"
  );
}

/// An out-of-grid strike prices to NaN, not `0.0`, so a calibration past the FFT grid cannot fit
/// a silent zero.
#[test]
fn carr_madan_out_of_grid_returns_nan_not_zero() {
  let model = BsmFourier { sigma: 0.15 };
  let pricer = CarrMadanPricer::default();
  let s = 100.0;
  let r = 0.05;
  let t = 1.0;

  let p_itm = pricer.price_call(&model, s, 1e-9, r, 0.0, t);
  assert!(
    p_itm.is_nan(),
    "Out-of-grid deep-ITM must return NaN to flag truncation, got {p_itm}"
  );

  let p_otm = pricer.price_call(&model, s, 1e9, r, 0.0, t);
  assert!(
    p_otm.is_nan(),
    "Out-of-grid deep-OTM must return NaN to flag truncation, got {p_otm}"
  );

  let p_atm = pricer.price_call(&model, s, 100.0, r, 0.0, t);
  assert!(p_atm.is_finite() && p_atm > 0.0);

  assert!(!pricer.strike_in_grid(&model, s, 1e-9, r, 0.0, t));
  assert!(!pricer.strike_in_grid(&model, s, 1e9, r, 0.0, t));
  assert!(pricer.strike_in_grid(&model, s, 100.0, r, 0.0, t));
}

#[test]
fn lewis_bsm_reference() {
  let model = BsmFourier { sigma: 0.15 };
  let price = LewisPricer::price_call(&model, 100.0, 100.0, 0.05, 0.01, 1.0);
  let expected = 7.94871378854164;
  assert!(
    (price - expected).abs() < TOL_FOURIER,
    "Lewis BSM: got={price}, expected={expected}"
  );
}

#[test]
fn gil_pelaez_bsm_reference() {
  let model = BsmFourier { sigma: 0.15 };
  let price = GilPelaezPricer::price_call(&model, 100.0, 100.0, 0.05, 0.01, 1.0);
  let expected = 7.94871378854164;
  assert!(
    (price - expected).abs() < TOL_FOURIER,
    "Gil-Pelaez BSM: got={price}, expected={expected}"
  );
}

#[test]
fn carr_madan_vg_reference() {
  let model = VarianceGammaFourier {
    sigma: 0.2,
    theta: 0.1,
    nu: 0.85,
  };
  let pricer = CarrMadanPricer::default();
  let price = pricer.price_call(&model, 100.0, 100.0, 0.05, 0.01, 1.0);
  let expected = 10.13935062748614;
  assert!(
    (price - expected).abs() < TOL_FOURIER,
    "Carr-Madan Vg: got={price}, expected={expected}"
  );
}

#[test]
fn carr_madan_cgmy_reference() {
  let model = CgmyFourier {
    c: 0.02,
    g: 5.0,
    m: 15.0,
    y: 1.2,
  };
  let pricer = CarrMadanPricer::default();
  let price = pricer.price_call(&model, 100.0, 100.0, 0.05, 0.01, 1.0);
  let expected = 5.80222163947386;
  assert!(
    (price - expected).abs() < TOL_FOURIER,
    "Carr-Madan Cgmy: got={price}, expected={expected}"
  );
}

#[test]
fn carr_madan_merton_reference() {
  let model = MertonJdFourier {
    sigma: 0.12,
    lambda: 0.4,
    mu_j: -0.12,
    sigma_j: 0.18,
  };
  let pricer = CarrMadanPricer::default();
  let price = pricer.price_call(&model, 100.0, 100.0, 0.05, 0.01, 1.0);
  let expected = 8.675684635426279;
  assert!(
    (price - expected).abs() < TOL_FOURIER,
    "Carr-Madan MJD: got={price}, expected={expected}"
  );
}

#[test]
fn carr_madan_kou_reference() {
  let model = KouFourier {
    sigma: 0.15,
    lambda: 3.0,
    p_up: 0.2,
    eta1: 25.0,
    eta2: 10.0,
  };
  let pricer = CarrMadanPricer::default();
  let price = pricer.price_call(&model, 100.0, 100.0, 0.05, 0.01, 1.0);
  let expected = 11.92430307601936;
  assert!(
    (price - expected).abs() < TOL_FOURIER,
    "Carr-Madan Kou: got={price}, expected={expected}"
  );
}

#[test]
fn carr_madan_heston_vs_analytical() {
  let model = HestonFourier::new(HestonParams {
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.1,
    rho: -0.6,
  });
  let pricer = CarrMadanPricer::default();
  let price = pricer.price_call(&model, 100.0, 100.0, 0.05, 0.0, 1.0);
  assert!(
    (price - 10.474).abs() < 0.2,
    "Carr-Madan Heston: got={price}, expected≈10.474"
  );
}

/// A range inside the default grid leaves `eta` at 0.25, where the trapezoid rule's image is
/// `Se^{−2πα/η} = 6.5e-7` on a spot of 100.
#[test]
fn carr_madan_cumulant_sized_matches_black_scholes() {
  let model = BsmFourier { sigma: 0.2 };
  let closed = BsmPricer::new(0.2, BsmCoc::Merton1973);
  for tau in [0.1, 0.5, 1.0, 5.0] {
    let pricer = CarrMadanPricer::cumulant_sized(&model, tau, 100.0, 12.0);
    assert_eq!((pricer.n, pricer.eta), (4096, 0.25), "tau = {tau}");
    for k in [80.0, 100.0, 120.0] {
      let price = pricer.price_call(&model, 100.0, k, 0.05, 0.01, tau);
      let want = closed.price_call(100.0, k, 0.05, 0.01, tau);
      assert!(
        (price - want).abs() < 1e-6,
        "tau = {tau}, K = {k}: {price} vs {want}"
      );
    }
  }
}

/// A range past the default grid's `4π` refines `eta` to cover it.
#[test]
fn carr_madan_cumulant_sized_widens_the_grid_for_a_long_maturity() {
  let model = BsmFourier { sigma: 0.2 };
  let closed = BsmPricer::new(0.2, BsmCoc::Merton1973);
  let pricer = CarrMadanPricer::cumulant_sized(&model, 25.0, 100.0, 12.0);
  let half_width = 100.0_f64.ln() + model.cumulants(25.0).truncation_half_width(12.0);
  assert!(half_width > 4.0 * PI && (pricer.eta - PI / half_width).abs() < 1e-15);
  for k in [80.0, 100.0, 120.0] {
    let price = pricer.price_call(&model, 100.0, k, 0.05, 0.01, 25.0);
    let want = closed.price_call(100.0, k, 0.05, 0.01, 25.0);
    assert!((price - want).abs() < 5e-4, "K = {k}: {price} vs {want}");
  }
}

/// Lord–Kahl (2010) §3.2: the grid spans at least `|ln s| + truncation_half_width(12)` either
/// side of `ln K = 0`, so it brackets `ln s`.
#[test]
fn carr_madan_cumulant_sized_grid_brackets_log_spot() {
  let model = HestonFourier::new(HestonParams {
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
  });
  let s = 100.0_f64;
  let t = 1.0_f64;
  let pricer = CarrMadanPricer::cumulant_sized(&model, t, s, 12.0);
  let (log_strikes, _) = pricer.price_call_surface(&model, s, 0.05, 0.0, t);
  let half_width = (log_strikes[log_strikes.len() - 1] - log_strikes[0]) / 2.0;
  let expected_half_width = s.ln().abs() + model.cumulants(t).truncation_half_width(12.0);
  assert!(
    half_width > expected_half_width,
    "FFT half-width {half_width} against |ln s| + truncation_half_width(12) = {expected_half_width}"
  );
  let ln_s = s.ln();
  assert!(
    log_strikes[0] <= ln_s && ln_s <= log_strikes[log_strikes.len() - 1],
    "ln(s)={ln_s} must be inside the FFT grid [{}, {}]",
    log_strikes[0],
    log_strikes[log_strikes.len() - 1]
  );
}

/// Lord–Kahl §3.2: `cumulant_sized` prices a long-maturity, high-|ρ| Heston ATM call finite,
/// positive and close to Gil-Pelaez, where a fixed-η grid truncates the skewed wings.
#[test]
fn carr_madan_cumulant_sized_finite_positive_long_maturity_skew() {
  let model = HestonFourier::new(HestonParams {
    v0: 0.04,
    kappa: 1.0,
    theta: 0.04,
    sigma: 0.5,
    rho: -0.85,
  });
  let s = 100.0_f64;
  let t = 5.0_f64;
  let r = 0.05_f64;
  let k = 100.0_f64;

  let reference = GilPelaezPricer::price_call(&model, s, k, r, 0.0, t);

  let cum = CarrMadanPricer::cumulant_sized(&model, t, s, 12.0).price_call(&model, s, k, r, 0.0, t);

  assert!(
    cum.is_finite() && cum > 0.0 && cum < s,
    "cumulant_sized price must be finite, positive, and bounded by S at T=5y, ρ=-0.85, got {cum}"
  );
  assert!(
    (cum - reference).abs() < 5.0,
    "cumulant_sized must track Gil-Pelaez within 5.0 at T=5y, ρ=-0.85: cum={cum}, ref={reference}"
  );
}

#[test]
fn hkde_zero_jumps_equals_heston() {
  let heston = HestonFourier::new(HestonParams {
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.1,
    rho: -0.6,
  });
  let hkde = HkdeFourier::new(HkdeParams {
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.1,
    rho: -0.6,
    lambda: 0.0,
    p_up: 0.5,
    eta1: 10.0,
    eta2: 10.0,
  });
  let pricer = CarrMadanPricer::default();
  let heston_price = pricer.price_call(&heston, 100.0, 100.0, 0.05, 0.0, 1.0);
  let hkde_price = pricer.price_call(&hkde, 100.0, 100.0, 0.05, 0.0, 1.0);
  assert!(
    (heston_price - hkde_price).abs() < TOL_TIGHT,
    "Hkde(lambda=0) vs Heston: hkde={hkde_price}, heston={heston_price}"
  );
}

#[test]
fn hkde_prices_positive() {
  let model = HkdeFourier::new(HkdeParams {
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
    lambda: 3.0,
    p_up: 0.4,
    eta1: 10.0,
    eta2: 5.0,
  });
  let pricer = CarrMadanPricer::default();
  let price = pricer.price_call(&model, 100.0, 100.0, 0.05, 0.01, 1.0);
  assert!(
    price > 0.0,
    "Hkde call price should be positive, got={price}"
  );
}

#[test]
fn hkde_call_decreases_with_strike() {
  let model = HkdeFourier::new(HkdeParams {
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
    lambda: 3.0,
    p_up: 0.4,
    eta1: 10.0,
    eta2: 5.0,
  });
  let pricer = CarrMadanPricer::default();
  let p1 = pricer.price_call(&model, 100.0, 90.0, 0.05, 0.01, 1.0);
  let p2 = pricer.price_call(&model, 100.0, 100.0, 0.05, 0.01, 1.0);
  let p3 = pricer.price_call(&model, 100.0, 110.0, 0.05, 0.01, 1.0);
  assert!(
    p1 > p2 && p2 > p3,
    "Hkde call should decrease with strike: p(90)={p1}, p(100)={p2}, p(110)={p3}"
  );
}

#[test]
fn hkde_lewis_vs_carr_madan() {
  let model = HkdeFourier::new(HkdeParams {
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
    lambda: 1.0,
    p_up: 0.4,
    eta1: 10.0,
    eta2: 5.0,
  });
  let cm_price = CarrMadanPricer::default().price_call(&model, 100.0, 100.0, 0.05, 0.01, 1.0);
  let lw_price = LewisPricer::price_call(&model, 100.0, 100.0, 0.05, 0.01, 1.0);
  assert!(
    (cm_price - lw_price).abs() < TOL_FOURIER,
    "Hkde Lewis vs Carr-Madan: lewis={lw_price}, cm={cm_price}"
  );
}

/// At `T = 5`, `ρ = −0.9` Heston's (1993) form of the chf crosses a branch cut; the Little-Trap
/// form (Albrecher et al. 2007) prices finite and bounded.
#[test]
fn heston_fourier_little_trap_long_maturity_high_rho() {
  let model = HestonFourier::new(HestonParams {
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.9,
  });
  let call = LewisPricer::price_call(&model, 100.0, 100.0, 0.05, 0.0, 5.0);
  assert!(
    call.is_finite() && call > 0.0 && call < 100.0,
    "HestonFourier Trap form: finite positive bounded call required at T=5y, ρ=-0.9, got {call}"
  );
}

/// The same stress on both factors of `DoubleHestonFourier`, each on the principal log branch.
#[test]
fn double_heston_fourier_little_trap_long_maturity_high_rho() {
  let model = DoubleHestonFourier::new(
    HestonParams::new(0.04, 2.0, 0.04, 0.3, -0.9),
    HestonParams::new(0.02, 1.0, 0.03, 0.25, -0.85),
  );
  let call = LewisPricer::price_call(&model, 100.0, 100.0, 0.05, 0.0, 5.0);
  assert!(
    call.is_finite() && call > 0.0 && call < 100.0,
    "DoubleHestonFourier Trap form: finite positive bounded call required at T=5y, ρ_j=-0.9/-0.85, got {call}"
  );
}

/// Same long-maturity / high-|ρ| stress for `HkdeFourier::chf` — the Heston
/// diffusion part must stay on the principal log-branch under the Kou jump CF.
#[test]
fn hkde_fourier_little_trap_long_maturity_high_rho() {
  let model = HkdeFourier::new(HkdeParams {
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.9,
    lambda: 1.5,
    p_up: 0.4,
    eta1: 10.0,
    eta2: 5.0,
  });
  let call = LewisPricer::price_call(&model, 100.0, 100.0, 0.05, 0.0, 5.0);
  assert!(
    call.is_finite() && call > 0.0 && call < 100.0,
    "HkdeFourier Trap form: finite positive bounded call required at T=5y, ρ=-0.9, got {call}"
  );
}

/// The same stress on `BatesFourier`, whose Heston part stays on the principal branch under jumps.
#[test]
fn bates_fourier_little_trap_long_maturity_high_rho() {
  let model = BatesFourier::new(SvjParams {
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.9,
    lambda: 0.3,
    mu_j: -0.05,
    sigma_j: 0.15,
  });
  let call = LewisPricer::price_call(&model, 100.0, 100.0, 0.05, 0.0, 5.0);
  assert!(
    call.is_finite() && call > 0.0 && call < 100.0,
    "BatesFourier Trap form: finite positive bounded call required at T=5y, ρ=-0.9, got {call}"
  );
}

#[test]
#[should_panic(expected = "s must satisfy `s.is_finite() && s > 0.0`")]
fn cumulant_sizing_rejects_a_non_positive_spot_instead_of_a_default_grid() {
  let model = crate::pricing::fourier::BsmFourier { sigma: 0.2 };
  let _ = CarrMadanPricer::cumulant_sized(&model, 1.0, -100.0, 12.0);
}

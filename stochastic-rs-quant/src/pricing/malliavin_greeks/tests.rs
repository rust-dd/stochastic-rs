use super::*;

#[test]
fn gbm_delta_positive_for_call() {
  let greeks = GbmMalliavinGreeks {
    s0: 100.0,
    sigma: 0.2,
    r: 0.05,
    q: 0.0,
    tau: 1.0,
    k: 100.0,
    n_paths: 50_000,
    n_steps: 252,
  };
  let delta = greeks.delta();
  assert!(
    delta > 0.0 && delta < 1.0,
    "Call delta should be in (0,1), got {delta}"
  );
}

#[test]
fn gbm_gamma_positive() {
  let greeks = GbmMalliavinGreeks {
    s0: 100.0,
    sigma: 0.2,
    r: 0.05,
    q: 0.0,
    tau: 1.0,
    k: 100.0,
    n_paths: 50_000,
    n_steps: 252,
  };
  let gamma = greeks.gamma();
  assert!(gamma > 0.0, "Call gamma should be > 0, got {gamma}");
}

#[test]
fn gbm_vega_positive() {
  let greeks = GbmMalliavinGreeks {
    s0: 100.0,
    sigma: 0.2,
    r: 0.05,
    q: 0.0,
    tau: 1.0,
    k: 100.0,
    n_paths: 50_000,
    n_steps: 252,
  };
  let vega = greeks.vega();
  assert!(vega > 0.0, "Call vega should be > 0, got {vega}");
}

#[test]
fn gbm_all_greeks_consistent() {
  let greeks = GbmMalliavinGreeks {
    s0: 100.0,
    sigma: 0.2,
    r: 0.05,
    q: 0.0,
    tau: 1.0,
    k: 100.0,
    n_paths: 100_000,
    n_steps: 252,
  };
  let g = greeks.all_greeks();
  // BS analytical: Delta ~ 0.64, Gamma ~ 0.019, Vega ~ 37.5, Rho ~ 53.2
  assert!(g.delta > 0.3 && g.delta < 0.9, "delta={}", g.delta);
  assert!(g.gamma > 0.005 && g.gamma < 0.05, "gamma={}", g.gamma);
  assert!(g.vega > 10.0 && g.vega < 60.0, "vega={}", g.vega);
}

#[test]
fn heston_delta_positive_for_call() {
  let greeks = HestonMalliavinGreeks {
    s0: 100.0,
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    xi: 0.3,
    rho: -0.7,
    r: 0.05,
    tau: 1.0,
    k: 100.0,
    n_paths: 50_000,
    n_steps: 252,
  };
  let delta = greeks.delta();
  assert!(
    delta > 0.0 && delta < 1.0,
    "Heston Malliavin delta should be in (0,1), got {delta}"
  );
}

#[test]
fn heston_delta_pathwise_positive() {
  let greeks = HestonMalliavinGreeks {
    s0: 100.0,
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    xi: 0.3,
    rho: -0.7,
    r: 0.05,
    tau: 1.0,
    k: 100.0,
    n_paths: 50_000,
    n_steps: 252,
  };
  let delta = greeks.delta_pathwise();
  assert!(
    delta > 0.3 && delta < 0.9,
    "Heston pathwise delta should be ~0.6 for ATM, got {delta}"
  );
}

#[test]
fn heston_delta_el_khatib_gbm_limit() {
  let greeks = HestonMalliavinGreeks {
    s0: 100.0,
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    xi: 0.0,
    rho: -0.7,
    r: 0.05,
    tau: 1.0,
    k: 100.0,
    n_paths: 20_000,
    n_steps: 64,
  };
  let delta = greeks.delta_el_khatib_with_seed(0x5EED);
  assert!(
    delta > 0.3 && delta < 0.9,
    "El-Khatib delta should reduce to the Gbm Malliavin range, got {delta}"
  );
}

#[test]
fn heston_delta_el_khatib_full_kernel_finite() {
  let greeks = HestonMalliavinGreeks {
    s0: 100.0,
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    xi: 0.15,
    rho: -0.4,
    r: 0.05,
    tau: 1.0,
    k: 100.0,
    n_paths: 2_000,
    n_steps: 32,
  };
  let delta = greeks.delta_el_khatib_with_seed(0xDEC0DE);
  assert!(
    delta.is_finite() && delta.abs() < 5.0,
    "El-Khatib delta should stay finite under the full kernel, got {delta}"
  );
}

#[test]
fn heston_gamma_positive() {
  let greeks = HestonMalliavinGreeks {
    s0: 100.0,
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    xi: 0.3,
    rho: -0.7,
    r: 0.05,
    tau: 1.0,
    k: 100.0,
    n_paths: 100_000,
    n_steps: 252,
  };
  let gamma = greeks.gamma();
  assert!(
    gamma > 0.0 && gamma < 0.1,
    "Heston gamma should be positive and reasonable, got {gamma}"
  );
}

#[test]
fn heston_vega_v0_positive() {
  let greeks = HestonMalliavinGreeks {
    s0: 100.0,
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    xi: 0.3,
    rho: -0.7,
    r: 0.05,
    tau: 1.0,
    k: 100.0,
    n_paths: 50_000,
    n_steps: 252,
  };
  let vega = greeks.vega_v0();
  assert!(vega > 0.0, "Heston vega_v0 should be > 0, got {vega}");
}

#[test]
fn heston_malliavin_vs_pathwise_consistent() {
  let greeks = HestonMalliavinGreeks {
    s0: 100.0,
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    xi: 0.1, // low vol-of-vol so zeroth-order approx is good
    rho: -0.3,
    r: 0.05,
    tau: 1.0,
    k: 100.0,
    n_paths: 200_000,
    n_steps: 252,
  };
  let mall = greeks.delta();
  let path = greeks.delta_pathwise();
  let rel_err = ((mall - path) / path).abs();
  assert!(
    rel_err < 0.15,
    "Malliavin and pathwise delta should be close for low xi, got mall={mall} path={path} rel_err={rel_err}"
  );
}

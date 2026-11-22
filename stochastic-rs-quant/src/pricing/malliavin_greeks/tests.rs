use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::Unseeded;

use super::*;
use crate::pricing::heston::HestonParams;

#[test]
fn gbm_delta_positive_for_call() {
  let greeks = GbmMalliavinGreeks {
    s: 100.0,
    sigma: 0.2,
    r: 0.05,
    q: 0.0,
    tau: 1.0,
    k: 100.0,
    n_paths: 50_000,
    n_steps: 252,
    seed: Unseeded,
  };
  let delta = greeks.delta().mean;
  assert!(
    delta > 0.0 && delta < 1.0,
    "Call delta should be in (0,1), got {delta}"
  );
}

#[test]
fn gbm_gamma_positive() {
  let greeks = GbmMalliavinGreeks {
    s: 100.0,
    sigma: 0.2,
    r: 0.05,
    q: 0.0,
    tau: 1.0,
    k: 100.0,
    n_paths: 50_000,
    n_steps: 252,
    seed: Unseeded,
  };
  let gamma = greeks.gamma().mean;
  assert!(gamma > 0.0, "Call gamma should be > 0, got {gamma}");
}

#[test]
fn gbm_vega_positive() {
  let greeks = GbmMalliavinGreeks {
    s: 100.0,
    sigma: 0.2,
    r: 0.05,
    q: 0.0,
    tau: 1.0,
    k: 100.0,
    n_paths: 50_000,
    n_steps: 252,
    seed: Unseeded,
  };
  let vega = greeks.vega().mean;
  assert!(vega > 0.0, "Call vega should be > 0, got {vega}");
}

#[test]
fn gbm_all_greeks_consistent() {
  let greeks = GbmMalliavinGreeks {
    s: 100.0,
    sigma: 0.2,
    r: 0.05,
    q: 0.0,
    tau: 1.0,
    k: 100.0,
    n_paths: 100_000,
    n_steps: 252,
    seed: Unseeded,
  };
  let g = greeks.all_greeks();
  // BS analytical: Delta ~ 0.64, Gamma ~ 0.019, Vega ~ 37.5, Rho ~ 53.2
  assert!(g.delta > 0.3 && g.delta < 0.9, "delta={}", g.delta);
  assert!(g.gamma > 0.005 && g.gamma < 0.05, "gamma={}", g.gamma);
  assert!(g.vega > 10.0 && g.vega < 60.0, "vega={}", g.vega);
}

#[test]
fn heston_delta_positive_for_call() {
  let greeks = HestonMalliavinGreeks::new(
    HestonParams {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma: 0.3,
      rho: -0.7,
    },
    100.0,
    100.0,
    0.05,
    1.0,
    50_000,
    252,
  );
  let delta = greeks.delta().mean;
  assert!(
    delta > 0.0 && delta < 1.0,
    "Heston Malliavin delta should be in (0,1), got {delta}"
  );
}

#[test]
fn heston_delta_pathwise_positive() {
  let greeks = HestonMalliavinGreeks::new(
    HestonParams {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma: 0.3,
      rho: -0.7,
    },
    100.0,
    100.0,
    0.05,
    1.0,
    50_000,
    252,
  );
  let delta = greeks.delta_pathwise().mean;
  assert!(
    delta > 0.3 && delta < 0.9,
    "Heston pathwise delta should be ~0.6 for ATM, got {delta}"
  );
}

#[test]
fn heston_delta_el_khatib_gbm_limit() {
  let greeks = HestonMalliavinGreeks::new(
    HestonParams {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma: 1e-300,
      rho: -0.7,
    },
    100.0,
    100.0,
    0.05,
    1.0,
    20_000,
    64,
  );
  let delta = greeks
    .with_seed(Deterministic::new(0x5EED))
    .delta_el_khatib()
    .mean;
  assert!(
    delta > 0.3 && delta < 0.9,
    "El-Khatib delta should reduce to the Gbm Malliavin range, got {delta}"
  );
}

#[test]
fn heston_delta_el_khatib_full_kernel_finite() {
  let greeks = HestonMalliavinGreeks::new(
    HestonParams {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma: 0.15,
      rho: -0.4,
    },
    100.0,
    100.0,
    0.05,
    1.0,
    2_000,
    32,
  );
  let delta = greeks
    .with_seed(Deterministic::new(0xDEC0DE))
    .delta_el_khatib()
    .mean;
  assert!(
    delta.is_finite() && delta.abs() < 5.0,
    "El-Khatib delta should stay finite under the full kernel, got {delta}"
  );
}

#[test]
fn heston_gamma_positive() {
  let greeks = HestonMalliavinGreeks::new(
    HestonParams {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma: 0.3,
      rho: -0.7,
    },
    100.0,
    100.0,
    0.05,
    1.0,
    100_000,
    252,
  );
  let gamma = greeks.gamma().mean;
  assert!(
    gamma > 0.0 && gamma < 0.1,
    "Heston gamma should be positive and reasonable, got {gamma}"
  );
}

#[test]
fn heston_vega_v0_positive() {
  let greeks = HestonMalliavinGreeks::new(
    HestonParams {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma: 0.3,
      rho: -0.7,
    },
    100.0,
    100.0,
    0.05,
    1.0,
    50_000,
    252,
  );
  let vega = greeks.vega_v0().mean;
  assert!(vega > 0.0, "Heston vega_v0 should be > 0, got {vega}");
}

#[test]
fn heston_malliavin_vs_pathwise_consistent() {
  let greeks = HestonMalliavinGreeks::new(
    HestonParams {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      // low vol-of-vol so zeroth-order approx is good
      sigma: 0.1,
      rho: -0.3,
    },
    100.0,
    100.0,
    0.05,
    1.0,
    200_000,
    252,
  );
  let mall = greeks.delta().mean;
  let path = greeks.delta_pathwise().mean;
  let rel_err = ((mall - path) / path).abs();
  assert!(
    rel_err < 0.15,
    "Malliavin and pathwise delta should be close for low xi, got mall={mall} path={path} rel_err={rel_err}"
  );
}

#[test]
fn gbm_greeks_are_nan_for_a_degenerate_spot_or_horizon() {
  let base = GbmMalliavinGreeks {
    s: 100.0,
    sigma: 0.2,
    r: 0.05,
    q: 0.0,
    tau: 1.0,
    k: 100.0,
    n_paths: 64,
    n_steps: 16,
    seed: Unseeded,
  };
  let degenerate = [
    GbmMalliavinGreeks {
      s: 0.0,
      ..base.clone()
    },
    GbmMalliavinGreeks {
      s: f64::NAN,
      ..base.clone()
    },
    GbmMalliavinGreeks {
      tau: 0.0,
      ..base.clone()
    },
    GbmMalliavinGreeks {
      q: f64::NAN,
      ..base
    },
  ];
  for g in degenerate {
    let all = g.all_greeks();
    let estimates = [
      g.delta().mean,
      g.gamma().mean,
      g.vega().mean,
      g.rho_greek().mean,
      all.delta,
      all.gamma,
      all.vega,
      all.rho,
    ];
    assert!(estimates.iter().all(|x| x.is_nan()), "{g:?}: {estimates:?}");
  }
}

#[test]
fn heston_greeks_are_nan_for_a_degenerate_spot_or_horizon() {
  let base = HestonMalliavinGreeks::new(
    HestonParams {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma: 0.3,
      rho: -0.7,
    },
    100.0,
    100.0,
    0.05,
    1.0,
    64,
    16,
  );
  let degenerate = [
    HestonMalliavinGreeks {
      s: -1.0,
      ..base.clone()
    },
    HestonMalliavinGreeks {
      tau: 0.0,
      ..base.clone()
    },
    HestonMalliavinGreeks {
      r: f64::NAN,
      ..base
    },
  ];
  for h in degenerate {
    let all = crate::traits::GreeksExt::greeks(&h);
    let estimates = [
      h.delta().mean,
      h.delta_pathwise().mean,
      h.gamma().mean,
      h.vega_v0().mean,
      h.clone()
        .with_seed(Deterministic::new(7))
        .delta_el_khatib()
        .mean,
      all.delta,
      all.gamma,
      all.vega,
    ];
    assert!(estimates.iter().all(|x| x.is_nan()), "{h:?}: {estimates:?}");
    assert!(h.simulate().0.iter().all(|x| x.is_nan()));
  }
}

#[test]
#[should_panic(expected = "sigma must satisfy `sigma.is_finite() && sigma > 0.0`, got sigma = 0.0")]
fn the_heston_greeks_refuse_a_model_outside_heston_params() {
  let params = HestonParams {
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.0,
    rho: -0.7,
  };
  let _ = HestonMalliavinGreeks::new(params, 100.0, 100.0, 0.05, 1.0, 64, 16);
}

#[test]
fn every_malliavin_greek_carries_its_standard_error() {
  let gbm = GbmMalliavinGreeks {
    s: 100.0,
    sigma: 0.2,
    r: 0.05,
    q: 0.0,
    tau: 1.0,
    k: 100.0,
    n_paths: 2_000,
    n_steps: 32,
    seed: Unseeded,
  };
  let params = HestonParams {
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
  };
  let heston = HestonMalliavinGreeks::new(params, 100.0, 100.0, 0.05, 1.0, 1_000, 16);
  let estimates = [
    gbm.delta(),
    gbm.gamma(),
    gbm.vega(),
    gbm.rho_greek(),
    heston.delta(),
    heston.delta_pathwise(),
    heston.gamma(),
    heston.vega_v0(),
    heston
      .clone()
      .with_seed(Deterministic::new(5))
      .delta_el_khatib(),
  ];
  for estimate in estimates {
    assert!(
      estimate.std_err > 0.0 && estimate.std_err.is_finite(),
      "{estimate:?}"
    );
    assert!(
      estimate.n_samples == 2_000 || estimate.n_samples == 1_000,
      "{estimate:?}"
    );
  }
}

#[test]
#[should_panic(expected = "n_steps must satisfy `n_steps >= 2`, got n_steps = 0")]
fn the_gbm_greeks_refuse_a_grid_without_a_step() {
  GbmMalliavinGreeks::new(0.2, 100.0, 100.0, 0.05, 0.0, 1.0, 100, 0);
}

#[test]
#[should_panic(expected = "n_paths must satisfy `n_paths >= 1`, got n_paths = 0")]
fn the_gbm_greeks_refuse_an_empty_sample() {
  GbmMalliavinGreeks::new(0.2, 100.0, 100.0, 0.05, 0.0, 1.0, 0, 16);
}

#[test]
fn the_gbm_greeks_constructor_builds_the_literal() {
  let built = GbmMalliavinGreeks::new(0.2, 100.0, 105.0, 0.05, 0.01, 1.0, 64, 16);
  assert_eq!(
    (built.sigma, built.s, built.k, built.r, built.q, built.tau),
    (0.2, 100.0, 105.0, 0.05, 0.01, 1.0)
  );
  assert_eq!((built.n_paths, built.n_steps), (64, 16));
}

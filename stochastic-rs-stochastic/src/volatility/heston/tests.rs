use stochastic_rs_core::simd_rng::Deterministic;

use super::*;
use crate::traits::ProcessExt;

#[test]
#[should_panic(
  expected = "v0 must satisfy `v0.is_none_or(|v| v >= T::zero())`, got v0 = Some(-0.1)"
)]
fn negative_initial_variance_panics() {
  let _ = Heston::new(
    Some(100.0_f64),
    Some(-0.1),
    1.0,
    0.04,
    0.3,
    -0.5,
    0.0,
    8,
    Some(1.0),
    HestonPow::Sqrt,
    VarianceBoundary::Floor,
    Unseeded,
  );
}

#[test]
fn variance_path_stays_non_negative() {
  let p = Heston::new(
    Some(100.0_f64),
    Some(0.04),
    1.5,
    0.04,
    0.5,
    -0.7,
    0.0,
    128,
    Some(1.0),
    HestonPow::Sqrt,
    VarianceBoundary::Floor,
    Unseeded,
  );
  let [_s, v] = p.sample();
  assert!(v.iter().all(|x| *x >= 0.0));
}

/// Andersen QE: variance stays non-negative even with the Feller condition
/// violated (2κθ = 0.16 < ξ² = 0.25), the simulated E[V_T] matches the exact
/// CIR mean θ + (v0−θ)e^{−κT}, and the driftless asset is a martingale,
/// E[S_T] ≈ S_0. Pinned seed; tolerances cover the MC error plus the small
/// uncorrected-martingale bias of the plain QE asset scheme (§4.3 of
/// Andersen has an optional exact correction not applied here).
#[test]
fn qe_variance_mean_and_asset_martingale() {
  let (s0, v0, kappa, theta, sigma, rho, mu) = (100.0_f64, 0.04, 2.0, 0.04, 0.5, -0.7, 0.0);
  let (n, t, m) = (64usize, 1.0_f64, 30_000usize);
  let model = Heston::new(
    Some(s0),
    Some(v0),
    kappa,
    theta,
    sigma,
    rho,
    mu,
    n,
    Some(t),
    HestonPow::Sqrt,
    VarianceBoundary::Floor,
    Deterministic::new(20_240_601),
  )
  .qe();

  let mut sum_s = 0.0;
  let mut sum_v = 0.0;
  let mut nonneg = true;
  for _ in 0..m {
    let [s, v] = model.sample();
    sum_s += s[n - 1];
    sum_v += v[n - 1];
    if v.iter().any(|x| *x < 0.0) {
      nonneg = false;
    }
  }
  let mean_s = sum_s / m as f64;
  let mean_v = sum_v / m as f64;
  let v_exact = theta + (v0 - theta) * (-kappa * t).exp();

  assert!(
    nonneg,
    "QE variance must stay non-negative (Feller violated here)"
  );
  assert!(
    (mean_v - v_exact).abs() / v_exact < 0.05,
    "QE E[V_T] = {mean_v}, exact CIR mean = {v_exact}"
  );
  assert!(
    (mean_s - s0).abs() / s0 < 0.025,
    "QE asset not ~martingale: E[S_T] = {mean_s}, S_0 = {s0}"
  );
}

/// QE is a square-root (CIR) scheme; it must reject the 3/2 variance.
#[test]
#[should_panic(
  expected = "pow must satisfy `matches!(pow, HestonPow::Sqrt)`, got pow = ThreeHalves"
)]
fn qe_rejects_three_halves() {
  let _ = Heston::new(
    Some(100.0_f64),
    Some(0.04),
    2.0,
    0.04,
    0.5,
    -0.7,
    0.0,
    16,
    Some(1.0),
    HestonPow::ThreeHalves,
    VarianceBoundary::Floor,
    Unseeded,
  )
  .qe()
  .sample();
}

fn qe_switch(kappa: f64, sigma: f64, s0: f64) -> Heston<f64, AndersenQe> {
  Heston::new(
    Some(s0),
    Some(0.04),
    kappa,
    0.04,
    sigma,
    -0.7,
    0.0,
    16,
    Some(1.0),
    HestonPow::Sqrt,
    VarianceBoundary::Floor,
    Unseeded,
  )
  .qe()
}

#[test]
#[should_panic(expected = "kappa must satisfy `kappa > T::zero()`, got kappa = 0.0")]
fn qe_switch_rejects_a_zero_mean_reversion() {
  let _ = qe_switch(0.0, 0.5, 100.0);
}

#[test]
#[should_panic(expected = "sigma must satisfy `sigma > T::zero()`, got sigma = 0.0")]
fn qe_switch_rejects_a_zero_vol_of_vol() {
  let _ = qe_switch(2.0, 0.0, 100.0);
}

#[test]
#[should_panic(expected = "s0 must satisfy `s0 > T::zero()`, got s0 = 0.0")]
fn qe_switch_rejects_a_non_positive_spot() {
  let _ = qe_switch(2.0, 0.5, 0.0);
}

#[test]
#[should_panic(expected = "sigma must satisfy `sigma > T::zero()`, got sigma = 0.0")]
fn qe_rechecks_its_parameters_where_the_scheme_runs() {
  let _ = qe_switch(2.0, 0.5, 100.0).with_sigma(0.0).sample();
}

#[test]
fn the_boundary_selects_the_step() {
  let make = |b| {
    Heston::<f64, Euler, Deterministic>::new(
      Some(100.0),
      Some(0.0),
      2.0,
      0.04,
      2.0,
      -0.7,
      0.03,
      65,
      Some(1.0),
      HestonPow::Sqrt,
      b,
      Deterministic::new(3),
    )
    .sample()
  };
  let [_, floor] = make(VarianceBoundary::Floor);
  let [_, reflect] = make(VarianceBoundary::Reflect);
  assert!(floor.iter().all(|v| *v >= 0.0) && reflect.iter().all(|v| *v >= 0.0));
  let split = floor.iter().zip(reflect.iter()).position(|(f, r)| f != r);
  let split = split.expect("v0 = 0 under a vol-of-vol of 2 crosses zero, where the two steps part");
  assert_eq!(floor[split], 0.0);
  assert!(reflect[split] > 0.0);
}

/// A generic re-tag type-checks only if `qe` keeps the handle it was given.
#[test]
fn qe_keeps_the_backend_handle() {
  fn retag<B: crate::euler::EulerBackend<f64>>(
    h: Heston<f64, Euler, Unseeded, B>,
  ) -> Heston<f64, AndersenQe, Unseeded, B> {
    h.qe()
  }
  assert_eq!(retag(Heston::default().on::<Cpu>()).backend(), &Cpu);
}

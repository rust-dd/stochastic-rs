use stochastic_rs_core::simd_rng::Unseeded;

use super::*;
use crate::euler::families::host_report;
use crate::noise::cgns::Cgns;
use crate::params::Drift;
use crate::params::VarianceBoundary;
use crate::traits::PathSampler;
use crate::volatility::HestonPow;
use crate::volatility::heston::Heston;
use crate::volatility::heston::HestonScheme;
use crate::volatility::heston_log::HestonLog;

/// Outside Feller's condition, `2κθ = 0.16 < σ² = 0.36`, on a monthly grid over five years.
const V0: f64 = 0.05;
const KAPPA: f64 = 0.4;
const THETA: f64 = 0.2;
const SIGMA: f64 = 0.6;
const N: usize = 61;
const T: f64 = 5.0;

/// The unit correlation factors to the unit matrix, so `dW_v` is the second shock itself.
fn sub_feller() -> HestonHullWhite2F<f64, Euler, Deterministic> {
  let model = paper(N, 1).with_horizon(Some(T)).with_rho(Array2::eye(4));
  let model = model.with_v0(V0).with_kappa(KAPPA).with_theta(THETA);
  model.with_sigma(SIGMA)
}

/// Fed their `dW_v`, the hybrid reports `HestonLog`'s full-truncation variance bit for bit below
/// zero too; `Heston`'s step takes `powf(0.5)`, an ulp off `sqrt` at 0.1 % of arguments.
#[test]
fn the_euler_variance_is_hestons_full_truncation_path_bit_for_bit() {
  let (rho, boundary) = (-0.7, VarianceBoundary::FullTruncation);
  let heston = Heston::<f64, Euler, _>::new(
    1.0,
    V0,
    KAPPA,
    THETA,
    SIGMA,
    rho,
    0.0,
    N,
    Some(T),
    HestonPow::Sqrt,
    boundary,
    Deterministic::new(0),
  );
  let log = |seed| {
    let drift = Drift::Mu(0.0);
    let seed = Deterministic::new(seed);
    HestonLog::new(
      drift,
      KAPPA,
      THETA,
      SIGMA,
      rho,
      N,
      1.0,
      V0,
      Some(T),
      boundary,
      seed,
    )
  };
  let model = sub_feller();
  let dt = model.dt();
  let (mut below_zero, mut worst) = (0, 0.0_f64);
  for seed in 0..16 {
    let [_, v_log] = log(seed).sampler().sample();
    let [_, v] = <Euler as HestonScheme>::simulate(&heston, &Deterministic::new(seed));
    let noise = Cgns::new(rho, N - 1, Some(T), Unseeded);
    let [_, dw_v] = noise.sample_impl(&Deterministic::new(seed));
    let mut state = [0.0, V0, 0.03, 0.0];
    for i in 1..N {
      (state, _) = model.euler_step(0.033, dt, state, [0.0, dw_v[i - 1], 0.0, 0.0]);
      below_zero += usize::from(state[1] < 0.0);
      let reported = state[1].max(0.0);
      assert_eq!(
        reported.to_bits(),
        v_log[i].to_bits(),
        "seed {seed}, point {i}"
      );
      worst = worst.max_or_nan((reported - v[i]).abs());
    }
  }
  assert!(below_zero > 0, "no carried variance went below zero");
  assert!(worst < 1e-12, "{worst:e} from `Heston`'s path");
}

fn family_report(state: [f64; 4], params: &[f64]) -> [f64; 4] {
  let mut out = [0.0; 4];
  host_report(
    Family::HestonHullWhite2F,
    &state,
    params,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    &mut out,
  );
  out
}

/// A carried variance below zero steps as on the host, and the family reports its positive part
/// and `S₀ exp(·)`, which is `S₀` itself at the start.
#[test]
fn the_euler_device_family_carries_a_signed_variance_and_reports_its_positive_part() {
  let model = paper(33, 1).with_rho(rho_full()).with_s0(97.3);
  let (state, dz) = ([0.2, -0.013, 0.02, 0.01], [0.21, -0.4, 0.1, -0.05]);
  let (host, _) = model.euler_step(0.033, model.dt(), state, dz);
  let spec = Euler::euler_spec(&model);
  let device = family_step(Family::HestonHullWhite2F, state, spec, 0.033, 0.0, dz);
  assert_eq!(device, host);
  assert!(host[1] > state[1], "a truncated variance drifts up by κθΔ");
  let (_, params) = Euler::euler_spec(&model).encode();
  let reported = family_report(state, &params);
  assert_eq!(reported, [97.3 * state[0].exp(), 0.0, state[2], state[3]]);
  assert_eq!(family_report([0.0, 0.2, 0.03, 0.0], &params)[0], 97.3);
}

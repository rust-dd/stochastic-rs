//! `E[v_T]` of the Euler variance step against the CIR mean `θ + (v₀ − θ)e^{−κT}`, Feller violated.
//! Reference: Lord, Koekkoek & van Dijk (2010), 10(2), 177–194, DOI 10.1080/14697680802392496

use ndarray::Array1;
use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stochastic::correlation::heston_stoch_corr::HestonStochCorr;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::interest::cir_pp::CirPlusPlus;
use stochastic_rs_stochastic::interest::wu_zhang::WuZhang;
use stochastic_rs_stochastic::jump::bates::Bates;
use stochastic_rs_stochastic::params::Drift;
use stochastic_rs_stochastic::params::VarianceBoundary;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::HestonPow;
use stochastic_rs_stochastic::volatility::double_heston::DoubleHeston;
use stochastic_rs_stochastic::volatility::heston::Heston;
use stochastic_rs_stochastic::volatility::heston_log::HestonLog;
use stochastic_rs_stochastic::volatility::heston_slv::HestonSlv;
use stochastic_rs_stochastic::volatility::heston2d::Heston2D;
use stochastic_rs_stochastic::volatility::hkde::Hkde;
use stochastic_rs_stochastic::volatility::multifactor_heston::MultifactorHeston;

const KAPPA: f64 = 0.4;
const THETA: f64 = 0.2;
const SIGMA: f64 = 0.6;
const V0: f64 = 0.05;
const T: f64 = 5.0;
const RHO: f64 = -0.7;
const PATHS: usize = 20_000;
const SEEDS: [u64; 3] = [2718, 999, 42];
/// Grid points of a monthly grid and of 100 steps a year over `T`.
const MONTHLY: usize = 61;
const FINE: usize = 501;

const FULL_TRUNCATION: VarianceBoundary = VarianceBoundary::FullTruncation;
const ABSORPTION: VarianceBoundary = VarianceBoundary::Absorption;
const REFLECT: VarianceBoundary = VarianceBoundary::Reflect;

/// The relative bias of the mean terminal variance per seed; `terminal` draws `PATHS` of them.
fn bias(terminal: impl Fn(Deterministic) -> Vec<f64>) -> [f64; 3] {
  let exact = THETA + (V0 - THETA) * (-KAPPA * T).exp();
  SEEDS.map(|seed| {
    let draws = terminal(Deterministic::new(seed));
    draws.iter().sum::<f64>() / draws.len() as f64 / exact - 1.0
  })
}

fn smallest(bias: [f64; 3]) -> f64 {
  bias.into_iter().fold(f64::INFINITY, |a, b| a.min(b.abs()))
}

fn largest(bias: [f64; 3]) -> f64 {
  bias.into_iter().fold(f64::NEG_INFINITY, f64::max)
}

fn heston(boundary: VarianceBoundary, n: usize) -> [f64; 3] {
  let pow = HestonPow::Sqrt;
  bias(|seed| {
    Heston::new(
      100.0,
      V0,
      KAPPA,
      THETA,
      SIGMA,
      RHO,
      0.0,
      n,
      Some(T),
      pow,
      boundary,
      seed,
    )
    .sample_map(PATHS, |[_, v]| v[n - 1])
  })
}

fn heston_log(boundary: VarianceBoundary) -> [f64; 3] {
  let n = MONTHLY;
  bias(|seed| {
    HestonLog::new(
      Drift::Mu(0.0),
      KAPPA,
      THETA,
      SIGMA,
      RHO,
      n,
      100.0,
      V0,
      Some(T),
      boundary,
      seed,
    )
    .sample_map(PATHS, |[_, v]| v[n - 1])
  })
}

fn heston2d(boundary: VarianceBoundary) -> [f64; 3] {
  let n = MONTHLY;
  bias(|seed| {
    Heston2D::new(
      [4.6; 2],
      [V0; 2],
      [0.0; 2],
      [THETA; 2],
      [KAPPA; 2],
      [SIGMA; 2],
      [0.0; 6],
      n,
      Some(T),
      boundary,
      seed,
    )
    .sample_map(PATHS, |p| 0.5 * (p.v1[n - 1] + p.v2[n - 1]))
  })
}

fn double_heston(boundary: VarianceBoundary) -> [f64; 3] {
  let n = MONTHLY;
  bias(|seed| {
    DoubleHeston::new(
      100.0,
      V0,
      V0,
      KAPPA,
      THETA,
      SIGMA,
      RHO,
      KAPPA,
      THETA,
      SIGMA,
      RHO,
      0.0,
      n,
      Some(T),
      boundary,
      seed,
    )
    .sample_map(PATHS, |p| 0.5 * (p.v1[n - 1] + p.v2[n - 1]))
  })
}

fn bates(boundary: VarianceBoundary) -> [f64; 3] {
  let n = MONTHLY;
  let jumps = SimdNormal::new(0.0, 0.05);
  bias(|seed| {
    Bates::new(
      Drift::Mu(0.0),
      1.0,
      0.0,
      KAPPA * THETA,
      KAPPA,
      SIGMA,
      RHO,
      jumps,
      n,
      100.0,
      V0,
      Some(T),
      boundary,
      seed,
    )
    .sample_map(PATHS, |[_, v]| v[n - 1])
  })
}

fn hkde(boundary: VarianceBoundary) -> [f64; 3] {
  let n = MONTHLY;
  bias(|seed| {
    Hkde::new(
      0.0,
      KAPPA,
      THETA,
      SIGMA,
      RHO,
      V0,
      1.0,
      0.4,
      10.0,
      5.0,
      n,
      100.0,
      Some(T),
      boundary,
      seed,
    )
    .sample_map(PATHS, |[_, v]| v[n - 1])
  })
}

fn cir(boundary: VarianceBoundary) -> [f64; 3] {
  let n = MONTHLY;
  bias(|seed| {
    Cir::new(KAPPA, THETA, SIGMA, n, Some(V0), Some(T), boundary, seed)
      .sample_map(PATHS, |x| x[n - 1])
  })
}

fn cir_plus_plus(boundary: VarianceBoundary) -> [f64; 3] {
  let n = MONTHLY;
  let phi = (|_| 0.0) as fn(f64) -> f64;
  bias(|seed| {
    CirPlusPlus::new(
      KAPPA,
      THETA,
      SIGMA,
      phi,
      n,
      Some(V0),
      Some(T),
      boundary,
      seed,
    )
    .sample_map(PATHS, |x| x[n - 1])
  })
}

#[test]
fn the_default_boundary_keeps_the_monthly_variance_mean_within_two_percent() {
  let bias = heston(VarianceBoundary::default(), MONTHLY);
  assert!(
    smallest(bias) < 0.02,
    "every seed is off by 2 % or more: {bias:?}"
  );
}

/// At two million paths (standard error 0.11 %): full truncation −0.02 % (fine) and +0.46 %
/// (monthly), absorption +3.98 % and +11.08 %, reflection +7.66 % and +22.15 %.
#[test]
fn full_truncation_is_the_low_bias_scheme_and_absorption_and_reflection_are_not() {
  for (n, truncated, absorbed, reflected) in [(FINE, 0.02, 0.03, 0.06), (MONTHLY, 0.02, 0.07, 0.15)]
  {
    let bias = heston(FULL_TRUNCATION, n);
    assert!(
      smallest(bias) < truncated,
      "full truncation, n = {n}: {bias:?}"
    );
    let bias = heston(ABSORPTION, n);
    assert!(largest(bias) > absorbed, "absorption, n = {n}: {bias:?}");
    let bias = heston(REFLECT, n);
    assert!(largest(bias) > reflected, "reflection, n = {n}: {bias:?}");
  }
}

#[test]
fn every_process_with_a_boundary_has_the_three_biases_on_a_monthly_grid() {
  type Biases = fn(VarianceBoundary) -> [f64; 3];
  let processes: [(&str, Biases); 7] = [
    ("HestonLog", heston_log),
    ("Heston2D", heston2d),
    ("DoubleHeston", double_heston),
    ("Bates", bates),
    ("Hkde", hkde),
    ("Cir", cir),
    ("CirPlusPlus", cir_plus_plus),
  ];
  for (name, biases) in processes {
    let bias = biases(FULL_TRUNCATION);
    assert!(smallest(bias) < 0.02, "{name}, full truncation: {bias:?}");
    let bias = biases(ABSORPTION);
    assert!(largest(bias) > 0.07, "{name}, absorption: {bias:?}");
    let bias = biases(REFLECT);
    assert!(largest(bias) > 0.15, "{name}, reflection: {bias:?}");
  }
}

#[test]
fn the_processes_without_a_boundary_step_under_full_truncation() {
  let n = MONTHLY;
  let slv = bias(|seed| {
    let leverage = (|_, _| 1.0) as fn(f64, f64) -> f64;
    HestonSlv::new(
      100.0,
      V0,
      KAPPA,
      THETA,
      SIGMA,
      RHO,
      0.0,
      1.0,
      leverage,
      n,
      Some(T),
      seed,
    )
    .sample_map(PATHS, |[_, v]| v[n - 1])
  });
  let multifactor = bias(|seed| {
    MultifactorHeston::new(
      100.0,
      [V0; 2],
      [KAPPA; 2],
      [THETA; 2],
      [SIGMA; 2],
      [RHO; 2],
      0.0,
      n,
      Some(T),
      seed,
    )
    .sample_map(PATHS, |p| 0.5 * (p.v[0][n - 1] + p.v[1][n - 1]))
  });
  let stochastic_correlation = bias(|seed| {
    HestonStochCorr::new(
      0.0,
      100.0,
      V0,
      KAPPA,
      THETA,
      SIGMA,
      RHO,
      1.0,
      -0.5,
      0.2,
      0.1,
      n,
      Some(T),
      seed,
    )
    .sample_map(PATHS, |p| p.v[n - 1])
  });
  let wu_zhang = bias(|seed| {
    let one = |x: f64| -> Array1<f64> { array![x] };
    WuZhang::new(
      one(THETA),
      one(KAPPA),
      one(SIGMA),
      one(0.1),
      one(0.03),
      one(V0),
      1,
      Some(T),
      n,
      seed,
    )
    .sample_map(PATHS, |rows| rows[[1, n - 1]])
  });
  for (name, bias) in [
    ("HestonSlv", slv),
    ("MultifactorHeston", multifactor),
    ("HestonStochCorr", stochastic_correlation),
    ("WuZhang", wu_zhang),
  ] {
    assert!(smallest(bias) < 0.02, "{name}: {bias:?}");
  }
}

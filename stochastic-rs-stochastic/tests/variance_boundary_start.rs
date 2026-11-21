//! A square-root state started below zero: the path reports `f₃(x₀)` at its first point and the
//! first step leaves `f₁(x₀)`, for each boundary of Lord, Koekkoek & van Dijk (2010).

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::correlation::heston_stoch_corr::HestonStochCorr;
use stochastic_rs_stochastic::diffusion::bessel::Bessel;
use stochastic_rs_stochastic::diffusion::bessel::SquaredBessel;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::diffusion::fcir::Fcir;
use stochastic_rs_stochastic::diffusion::feller_logistic::FellerLogistic;
use stochastic_rs_stochastic::interest::cir_2f::Cir2F;
use stochastic_rs_stochastic::interest::cir_pp::CirPlusPlus;
use stochastic_rs_stochastic::params::VarianceBoundary;
use stochastic_rs_stochastic::traits::ProcessExt;

const FT: VarianceBoundary = VarianceBoundary::FullTruncation;
const ABSORPTION: VarianceBoundary = VarianceBoundary::Absorption;
const REFLECT: VarianceBoundary = VarianceBoundary::Reflect;
const BOUNDARIES: [VarianceBoundary; 3] = [FT, ABSORPTION, REFLECT];
const THETA: f64 = 0.4;
const MU: f64 = 0.2;
const SIGMA: f64 = 0.6;
const N: usize = 61;
const T: f64 = 5.0;
const BELOW: f64 = -0.01;

type Build = fn(f64, VarianceBoundary) -> Array1<f64>;

fn cir(x0: f64, boundary: VarianceBoundary) -> Array1<f64> {
  let seed = Deterministic::new(42);
  Cir::new(THETA, MU, SIGMA, N, Some(x0), Some(T), boundary, seed).sample()
}

fn fcir(x0: f64, boundary: VarianceBoundary) -> Array1<f64> {
  let seed = Deterministic::new(42);
  Fcir::new(0.7, THETA, MU, SIGMA, N, Some(x0), Some(T), boundary, seed).sample()
}

fn cir_pp(x0: f64, boundary: VarianceBoundary) -> Array1<f64> {
  let (phi, seed) = ((|_| 0.0) as fn(f64) -> f64, Deterministic::new(42));
  CirPlusPlus::new(THETA, MU, SIGMA, phi, N, Some(x0), Some(T), boundary, seed).sample()
}

fn squared_bessel(x0: f64, boundary: VarianceBoundary) -> Array1<f64> {
  SquaredBessel::new(0.5, N, Some(x0), Some(T), boundary, Deterministic::new(42)).sample()
}

fn feller_logistic(x0: f64, boundary: VarianceBoundary) -> Array1<f64> {
  let seed = Deterministic::new(42);
  FellerLogistic::new(THETA, MU, SIGMA, N, Some(x0), Some(T), boundary, seed).sample()
}

fn takers() -> [(&'static str, Build); 5] {
  [
    ("Cir", cir),
    ("Fcir", fcir),
    ("CirPlusPlus", cir_pp),
    ("SquaredBessel", squared_bessel),
    ("FellerLogistic", feller_logistic),
  ]
}

fn bits(path: &Array1<f64>) -> Vec<u64> {
  path.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn a_negative_start_is_reported_as_f3_at_the_first_point() {
  for (name, build) in takers() {
    for boundary in BOUNDARIES {
      let want = if boundary == REFLECT { -BELOW } else { 0.0 };
      assert_eq!(build(BELOW, boundary)[0], want, "{name} under {boundary:?}");
    }
  }
}

#[test]
fn absorption_and_reflection_leave_a_negative_start_from_f1() {
  for (name, build) in takers() {
    for (boundary, entered) in [(ABSORPTION, 0.0), (REFLECT, -BELOW)] {
      assert_eq!(
        bits(&build(BELOW, boundary)),
        bits(&build(entered, boundary)),
        "{name} under {boundary:?}"
      );
    }
  }
}

/// `f₁ = x` and `f₂ = f₃ = x⁺ = 0`: the first step from `x₀ < 0` has no noise and no pull to zero.
#[test]
fn full_truncation_carries_the_negative_start_into_the_first_step() {
  let dt = T / (N - 1) as f64;
  let x0 = -0.001;
  for (name, build) in [("Cir", cir as Build), ("CirPlusPlus", cir_pp)] {
    let path = build(x0, FT);
    assert_eq!(path[1], x0 + THETA * MU * dt, "{name}");
    assert_ne!(bits(&path), bits(&build(0.0, FT)), "{name}");
  }
  assert_eq!(squared_bessel(x0, FT)[1], x0 + 0.5 * dt, "SquaredBessel");
}

#[test]
fn two_factor_cir_reports_the_sum_of_its_factors_f3() {
  let factor = |x0: f64, boundary| {
    let seed = Deterministic::new(1);
    Cir::new(THETA, MU, SIGMA, N, Some(x0), Some(T), boundary, seed)
  };
  let phi = (|_| 0.0) as fn(f64) -> f64;
  for (x, y, want) in [
    (FT, ABSORPTION, 0.0),
    (FT, REFLECT, 0.02),
    (REFLECT, REFLECT, 0.03),
  ] {
    let seed = Deterministic::new(42);
    let path = Cir2F::new(factor(BELOW, x), factor(-0.02, y), phi, seed).sample();
    assert_eq!(path[0], want, "{x:?} and {y:?}");
  }
}

#[test]
fn bessel_reports_the_radius_of_a_negative_start() {
  for boundary in BOUNDARIES {
    let path = Bessel::new(
      0.5,
      N,
      Some(-0.3),
      Some(T),
      boundary,
      Deterministic::new(42),
    )
    .sample();
    assert_eq!(path[0], 0.3, "{boundary:?}");
  }
}

#[test]
fn stochastic_correlation_heston_reports_the_positive_part_of_a_negative_variance() {
  let seed = Deterministic::new(42);
  let (rho0, kappa_r, mu_r, sigma_r, rho2) = (-0.5, 1.0, -0.5, 0.2, 0.1);
  let process = HestonStochCorr::new(
    0.0,
    100.0,
    BELOW,
    THETA,
    MU,
    SIGMA,
    rho0,
    kappa_r,
    mu_r,
    sigma_r,
    rho2,
    N,
    Some(T),
    seed,
  );
  assert_eq!(process.sample().v[0], 0.0);
}

//! Exact anchors of `HestonHullWhite2F`, one per leg: Heston's closed form with the rates frozen,
//! and the Gaussian rate leg's own law on the grid for `∫r`, the bond and Black on the forward.
//!
//! Reference: Brigo & Mercurio (2006), 2nd ed., §4.2.3-4.2.5, DOI 10.1007/978-3-540-34604-3

use stochastic_rs::ndarray::Array2;
use stochastic_rs::ndarray::array;
use stochastic_rs::quant::pricing::bsm::BsmCoc;
use stochastic_rs::quant::pricing::bsm::BsmPricer;
use stochastic_rs::quant::pricing::heston::HestonParams;
use stochastic_rs::quant::pricing::heston::HestonPricer;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stochastic::volatility::heston_hull_white_2f::HestonHullWhite2F;
use stochastic_rs::stochastic::volatility::heston_hull_white_2f::HestonHullWhite2FPaths;
use stochastic_rs::stochastic::volatility::heston_hull_white_2f::HybridScheme;
use stochastic_rs::traits::ModelPricer;
use stochastic_rs::traits::ProcessExt;
use stochastic_rs::traits::RealExt;

/// Each statistic is asserted on the best of these seeds, each power check on every one.
const SEEDS: [u64; 3] = [1993, 2006, 1973];
/// Standard errors a statistic may sit from its reference.
const Z_MAX: f64 = 3.0;
/// Standard errors by which a statistic must reject a reference with the tested ingredient changed.
const Z_POWER: f64 = 5.0;
const FLAT_RATE: f64 = 0.03;

/// Slower mean reversion and larger volatilities than a calibrated curve gives, so the rate
/// terms are a large share of what is measured.
const A: f64 = 0.3;
const B: f64 = 0.1;
const SIGMA1: f64 = 0.02;
const SIGMA2: f64 = 0.01;
const RHO_RU: f64 = -0.4;
const RHO_XR: f64 = 0.4;
const RHO_XU: f64 = 0.3;

/// Holds the mean short rate at [`FLAT_RATE`] from `r₀ = FLAT_RATE`, so `E ∫₀ᵀ r = FLAT_RATE · T`.
fn flat_mean_theta(_t: f64) -> f64 {
  A * FLAT_RATE
}

/// `B_c(τ) = (1 − e^{−cτ})/c`.
fn b_fn(c: f64, tau: f64) -> f64 {
  (1.0 - (-c * tau).exp()) / c
}

/// `∫₀ᵀ B_c(τ) B_d(τ) dτ`.
fn b_product_integral(c: f64, d: f64, t: f64) -> f64 {
  (t - b_fn(c, t) - b_fn(d, t) + b_fn(c + d, t)) / (c * d)
}

/// `Var ∫₀ᵀ r` in continuous time: `V(0, T)` of eq. 4.10 through §4.2.5's map to G2++.
fn var_int_r(rho_ru: f64, t: f64) -> f64 {
  let k = SIGMA2 / (A - B);
  let (jaa, jab, jbb) = (
    b_product_integral(A, A, t),
    b_product_integral(A, B, t),
    b_product_integral(B, B, t),
  );
  SIGMA1 * SIGMA1 * jaa + k * k * (jbb - 2.0 * jab + jaa) + 2.0 * rho_ru * SIGMA1 * k * (jab - jaa)
}

/// `Cov(W^x_T, ∫₀ᵀ r)` in continuous time.
fn cov_wx_int_r(rho_xr: f64, rho_xu: f64, t: f64) -> f64 {
  let k = SIGMA2 / (A - B);
  let (ia, ib) = ((t - b_fn(A, t)) / A, (t - b_fn(B, t)) / B);
  rho_xr * SIGMA1 * ia + rho_xu * k * (ib - ia)
}

/// `(Var I, Cov(W^x_T, I))` of the trapezoidal `I = Σ ½(r_{i−1} + r_i)dt` under the Euler rate
/// recursion: exact on the grid, since `I` is linear in the Gaussian shocks.
fn grid_moments(rho_xr: f64, rho_xu: f64, rho_ru: f64, steps: usize, dt: f64) -> (f64, f64) {
  let (mut g_r, mut g_u) = (0.5 * dt, 0.0);
  let (mut var, mut cov) = (0.0, 0.0);
  for _ in 0..steps {
    var += dt
      * (SIGMA1 * SIGMA1 * g_r * g_r
        + 2.0 * rho_ru * SIGMA1 * SIGMA2 * g_r * g_u
        + SIGMA2 * SIGMA2 * g_u * g_u);
    cov += dt * (rho_xr * SIGMA1 * g_r + rho_xu * SIGMA2 * g_u);
    (g_r, g_u) = (dt + (1.0 - A * dt) * g_r, dt * g_r + (1.0 - B * dt) * g_u);
  }
  (var, cov)
}

fn rho(rho_xv: f64, rho_xr: f64, rho_xu: f64, rho_ru: f64) -> Array2<f64> {
  array![
    [1.0, rho_xv, rho_xr, rho_xu],
    [rho_xv, 1.0, 0.0, 0.0],
    [rho_xr, 0.0, 1.0, rho_ru],
    [rho_xu, 0.0, rho_ru, 1.0],
  ]
}

fn mean_and_se(values: impl Iterator<Item = f64>) -> (f64, f64) {
  let values = values.collect::<Vec<_>>();
  let n = values.len() as f64;
  let mean = values.iter().sum::<f64>() / n;
  let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
  (mean, (var / n).sqrt())
}

/// The sample variance's distance from `reference`, in its standard error under a Gaussian law.
fn variance_z(values: impl Iterator<Item = f64>, reference: f64) -> f64 {
  let values = values.collect::<Vec<_>>();
  let n = values.len() as f64;
  let mean = values.iter().sum::<f64>() / n;
  let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
  (var - reference) / (reference * (2.0 / (n - 1.0)).sqrt())
}

/// Discounted put prices at `strikes` from `(S_T, ∫₀ᵀ r)` pairs, with their standard errors.
fn put_prices(terminal: &[(f64, f64)], strikes: &[f64]) -> Vec<(f64, f64)> {
  strikes
    .iter()
    .map(|k| {
      mean_and_se(
        terminal
          .iter()
          .map(|(s, int_r)| (k - s).max(0.0) * (-int_r).exp()),
      )
    })
    .collect()
}

/// `min` over the seeds of a statistic, NaN if any seed gave one.
fn best(z: impl IntoIterator<Item = f64>) -> f64 {
  f64::min_or_nan_fold(z)
}

/// The grid's law tends to the continuous one at first order in the step: at 100 steps a year
/// its variance of `∫r` sits 0.38 % above eq. 4.10 at one year.
#[test]
fn the_rate_leg_on_the_grid_converges_to_brigo_mercurio() {
  for tau in [1.0_f64, 5.0, 10.0] {
    let gaps = [100_usize, 200].map(|per_year| {
      let steps = tau as usize * per_year;
      let (var, cov) = grid_moments(RHO_XR, RHO_XU, RHO_RU, steps, 1.0 / per_year as f64);
      (
        var / var_int_r(RHO_RU, tau) - 1.0,
        cov / cov_wx_int_r(RHO_XR, RHO_XU, tau) - 1.0,
      )
    });
    let [(var_coarse, cov_coarse), (var_fine, cov_fine)] = gaps;
    assert!(
      var_coarse.abs() < 4e-3 && cov_coarse.abs() < 4e-4,
      "T = {tau}"
    );
    for ratio in [var_coarse / var_fine, cov_coarse / cov_fine] {
      assert!(
        (ratio - 2.0).abs() < 0.05,
        "T = {tau}: halving gave {ratio}"
      );
    }
  }
}

type Euler = stochastic_rs::stochastic::volatility::heston::Euler;

/// A sampler of the hybrid's paths, under either variance step.
trait Hybrid: ProcessExt<f64, Output = HestonHullWhite2FPaths<f64>> {}

impl<Sch: HybridScheme> Hybrid for HestonHullWhite2F<f64, Sch, Deterministic> where
  Self: ProcessExt<f64, Output = HestonHullWhite2FPaths<f64>>
{
}

/// `(S, ∫r)` at grid point `at` over `paths` paths.
fn terminal(model: &impl Hybrid, paths: usize, at: usize) -> Vec<(f64, f64)> {
  model.sample_map(paths, |path| (path.s[at], path.integrated_rate[at]))
}

const HESTON_RHO_XV: f64 = -0.7;

/// `κ = 1.5`, `θ = v₀ = 0.04`, `σ = 0.3`, `T = 2` on 100 steps a year: at zero rate volatility
/// and `θ_r = a r₀` the short rate never moves and the model is Heston.
fn heston_model(seed: u64) -> HestonHullWhite2F<f64, Euler, Deterministic> {
  HestonHullWhite2F::new(
    1.0,
    0.04,
    1.5,
    0.04,
    0.3,
    flat_mean_theta as fn(f64) -> f64,
    A,
    0.0,
    0.0,
    B,
    rho(HESTON_RHO_XV, RHO_XR, RHO_XU, RHO_RU),
    Some(FLAT_RATE),
    201,
    Some(2.0),
    Deterministic::new(seed),
  )
}

/// The worst put's `|z|` against Heston's closed form, and against it with `ρ_xv` flipped.
fn pure_heston_z(model: &impl Hybrid) -> (f64, f64) {
  const STRIKES: [f64; 3] = [0.8, 1.0, 1.2];
  let prices = put_prices(&terminal(model, 20_000, 200), &STRIKES);
  let worst_z = |rho_ref: f64| {
    let closed = HestonPricer::new(HestonParams::new(0.04, 1.5, 0.04, 0.3, rho_ref));
    f64::max_or_nan_fold(STRIKES.iter().zip(&prices).map(|(&k, &(price, se))| {
      ((price - closed.price_put(1.0, k, FLAT_RATE, 0.0, 2.0)) / se).abs()
    }))
  };
  (worst_z(HESTON_RHO_XV), worst_z(-HESTON_RHO_XV))
}

fn assert_pure_heston(what: &str, z: [(f64, f64); 3]) {
  println!("{what}: {z:.2?}");
  let closest = best(z.map(|(z, _)| z));
  assert!(
    closest < Z_MAX,
    "{what}: every seed has a put {closest:.2} SE or more off"
  );
  for (_, flipped) in z {
    assert!(
      flipped > Z_POWER,
      "{what}: ρ_xv flipped is only {flipped:.2} SE away"
    );
  }
}

#[test]
fn zero_rate_volatility_prices_the_heston_closed_form() {
  assert_pure_heston(
    "Euler",
    SEEDS.map(|seed| pure_heston_z(&heston_model(seed))),
  );
  assert_pure_heston(
    "QE",
    SEEDS.map(|seed| pure_heston_z(&heston_model(seed).qe())),
  );
}

/// Steps a year of the rate-leg cases; their references are the grid's own exact moments.
const PER_YEAR: usize = 10;

fn rate_model(
  theta: f64,
  sigma: f64,
  rho_xv: f64,
  years: usize,
  seed: u64,
) -> HestonHullWhite2F<f64, Euler, Deterministic> {
  HestonHullWhite2F::new(
    1.0,
    theta,
    0.8,
    theta,
    sigma,
    flat_mean_theta as fn(f64) -> f64,
    A,
    SIGMA1,
    SIGMA2,
    B,
    rho(rho_xv, RHO_XR, RHO_XU, RHO_RU),
    Some(FLAT_RATE),
    years * PER_YEAR + 1,
    Some(years as f64),
    Deterministic::new(seed),
  )
}

/// `∫₀ᵀ r` at 1, 5 and 10 years: the worst `|z|` of its mean, its variance and the bond price
/// `exp(−FLAT_RATE·T + ½ Var)`, then the two power statistics.
fn rate_leg_z(model: &impl Hybrid) -> (f64, f64, f64) {
  const YEARS: [usize; 3] = [1, 5, 10];
  let dt = 1.0 / PER_YEAR as f64;
  let integrals = model.sample_map(40_000, |path| {
    YEARS.map(|years| path.integrated_rate[years * PER_YEAR])
  });
  let mut worst = vec![];
  let mut flipped = vec![];
  let mut no_convexity = f64::NAN;
  for (m, years) in YEARS.into_iter().enumerate() {
    let tau = years as f64;
    let int_r = || integrals.iter().map(|row| row[m]);
    let moments = |rho_ru: f64| grid_moments(RHO_XR, RHO_XU, rho_ru, years * PER_YEAR, dt).0;
    let v = moments(RHO_RU);
    let (mean, mean_se) = mean_and_se(int_r());
    let (bond, bond_se) = mean_and_se(int_r().map(|x| (-x).exp()));
    worst.push(((mean - FLAT_RATE * tau) / mean_se).abs());
    worst.push(variance_z(int_r(), v).abs());
    worst.push(((bond - (-FLAT_RATE * tau + 0.5 * v).exp()) / bond_se).abs());
    flipped.push(variance_z(int_r(), moments(-RHO_RU)).abs());
    no_convexity = ((bond - (-FLAT_RATE * tau).exp()) / bond_se).abs();
  }
  (
    f64::max_or_nan_fold(worst),
    f64::min_or_nan_fold(flipped),
    no_convexity,
  )
}

fn assert_rate_leg(what: &str, z: [(f64, f64, f64); 3]) {
  println!("{what}: {z:.2?}");
  let closest = best(z.map(|(z, _, _)| z));
  assert!(
    closest < Z_MAX,
    "{what}: every seed has a moment {closest:.2} SE or more off"
  );
  for (_, flipped, no_convexity) in z {
    assert!(
      flipped > Z_POWER,
      "{what}: ρ_ru flipped is only {flipped:.2} SE away"
    );
    assert!(
      no_convexity > Z_POWER,
      "{what}: the ten-year bond without convexity is only {no_convexity:.2} SE away"
    );
  }
}

#[test]
fn integrated_rate_has_its_grid_moments_and_bond_price() {
  let model = |seed| rate_model(0.2, 0.2, -0.3, 10, seed);
  assert_rate_leg("Euler", SEEDS.map(|seed| rate_leg_z(&model(seed))));
  assert_rate_leg("QE", SEEDS.map(|seed| rate_leg_z(&model(seed).qe())));
}

/// `v₀ = θ = 0.04`, `ρ_xv = 0`, `T = 5`: with the variance frozen `log S_T` is Gaussian with
/// variance `θT + Var ∫r + 2√θ Cov(W^x_T, ∫r)` and a put is Black's on the forward `S₀/P(0, T)`.
fn frozen_variance_z(model: &impl Hybrid, theta: f64) -> (f64, f64) {
  const STRIKES: [f64; 3] = [0.8, 1.0, 1.3];
  let (tau, steps) = (5.0, 5 * PER_YEAR);
  let terminal = terminal(model, 50_000, steps);
  let prices = put_prices(&terminal, &STRIKES);
  let moments = |rho_xr, rho_xu| grid_moments(rho_xr, rho_xu, RHO_RU, steps, tau / steps as f64);
  let total_variance = |rho_xr: f64, rho_xu: f64| {
    let (v, cov) = moments(rho_xr, rho_xu);
    theta * tau + v + 2.0 * theta.sqrt() * cov
  };
  let bond_rate = FLAT_RATE - 0.5 * moments(RHO_XR, RHO_XU).0 / tau;
  let log_spot = || terminal.iter().map(|(s, _)| s.ln());
  let worst_price_z = |rho_xr: f64, rho_xu: f64| {
    let volatility = (total_variance(rho_xr, rho_xu) / tau).sqrt();
    let black = BsmPricer::new(volatility, BsmCoc::Merton1973);
    f64::max_or_nan_fold(STRIKES.iter().zip(&prices).map(|(&k, &(price, se))| {
      ((price - black.price_put(1.0, k, bond_rate, 0.0, tau)) / se).abs()
    }))
  };
  let worst = worst_price_z(RHO_XR, RHO_XU)
    .max_or_nan(variance_z(log_spot(), total_variance(RHO_XR, RHO_XU)).abs());
  let missing = [(0.0, 0.0), (RHO_XR, 0.0), (0.0, RHO_XU)];
  let power = f64::min_or_nan_fold(
    missing
      .into_iter()
      .map(|(rho_xr, rho_xu)| variance_z(log_spot(), total_variance(rho_xr, rho_xu)).abs())
      .chain([worst_price_z(0.0, 0.0)]),
  );
  (worst, power)
}

fn assert_frozen_variance(what: &str, z: [(f64, f64); 3]) {
  println!("{what}: {z:.2?}");
  let closest = best(z.map(|(z, _)| z));
  assert!(
    closest < Z_MAX,
    "{what}: every seed is {closest:.2} SE or more from Black"
  );
  for (_, power) in z {
    assert!(
      power > Z_POWER,
      "{what}: an equity-rate correlation removed is only {power:.2} SE away"
    );
  }
}

/// QE needs `σ > 0` and runs at `σ = 0.01`: the variance then moves by 4 % of its level, which
/// changes `Var log S_T` by under `1e-4` of its value, far inside the standard error here.
#[test]
fn frozen_variance_prices_black_on_the_forward_under_two_factor_rates() {
  let theta = 0.04;
  let model = |sigma, seed| rate_model(theta, sigma, 0.0, 5, seed);
  let euler = SEEDS.map(|seed| frozen_variance_z(&model(0.0, seed), theta));
  assert_frozen_variance("Euler", euler);
  let qe = SEEDS.map(|seed| frozen_variance_z(&model(0.01, seed).qe(), theta));
  assert_frozen_variance("QE", qe);
}

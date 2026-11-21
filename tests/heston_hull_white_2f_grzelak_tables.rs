//! The H-G2++ hybrid against Grzelak, Oosterlee and van Weeren (2011), tables 3 and 4: implied
//! volatilities of European calls simulated with [`HestonHullWhite2F`] under the quadratic-exponential
//! variance step, against the paper's own Monte Carlo column (standard deviations in parentheses there).
//!
//! Reference: Grzelak L. A., Oosterlee C. W. & van Weeren S. (2011), *The affine Heston model with
//! correlated Gaussian interest rates for pricing hybrid derivatives*, Quantitative Finance 11(11),
//! 1647-1663, DOI: 10.1080/14697688.2011.615216, experiments 4.2 and 4.3.
//!
//! Each cell's implied volatility is inverted from the put, whose payoff is bounded, so the estimator's
//! variance stays small at twenty years where a call's does not; the volatility is the same. Run with
//! `--nocapture` to print every cell. Pinned seed; the tolerance is in standard errors of the difference of
//! the two Monte Carlo estimates, the paper's standard deviation and this run's standard error in quadrature.
//!
//! This is the comparison with the paper's own example and not the process's correctness anchor: at the
//! paper's rate volatilities a Heston price without the rate leg already sits 0.2 to 0.5 volatility
//! points from table 3's cells, so the tables say little about that leg. The exact anchors, one per leg
//! of the model, are in `heston_hull_white_2f_closed_forms.rs`.

use stochastic_rs::ndarray::array;
use stochastic_rs::quant::OptionType;
use stochastic_rs::quant::pricing::bsm::BsmCoc;
use stochastic_rs::quant::pricing::bsm::BsmPricer;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stochastic::volatility::heston_hull_white_2f::HestonHullWhite2F;
use stochastic_rs::traits::ProcessExt;

/// Rates of experiment 4.2: `a = 1.1`, `σ₁ = 0.01`, `b = 0.8`, `σ₂ = 0.015`, `ρ_ru = −0.4`, a flat 3 % curve.
const A: f64 = 1.1;
const SIGMA1: f64 = 0.01;
const B: f64 = 0.8;
const SIGMA2: f64 = 0.015;
const RHO_RU: f64 = -0.4;
const FLAT_RATE: f64 = 0.03;

/// Maturities of the tables and their position on a monthly grid over twenty years.
const MATURITIES: [(f64, usize); 4] = [(1.0, 12), (5.0, 60), (10.0, 120), (20.0, 240)];
const STEPS_PER_YEAR: usize = 12;
const PATHS: usize = 100_000;
const SEED: u64 = 2011_1647;
/// Standard errors of the difference a cell may be off by.
const Z_MAX: f64 = 3.0;

/// Table 3's five-year row from its second strike on is printed but not asserted: it rises from 44.59
/// to 45.07 and then falls to 27.52 across strikes over which the paper's own 1, 10 and 20-year rows and
/// table 4's 5-year row move by under one point, and the closed-form Heston of this workspace without the
/// rate leg gives 44.1, 43.8, 43.5 and 43.4 there, so the printed values cannot be the stated model.
/// The authors' preprint (SSRN abstract 1434829, version of 20 October 2010, table 4.2) prints the same row.
const HELD_OUT: [(usize, usize); 4] = [(1, 1), (1, 2), (1, 3), (1, 4)];

/// Drift target fitting the flat curve: `θ_r(t) = a f + ½ (g'(t) + a g(t))` with `g(τ)` the integrand
/// of `Var ∫₀ᵀ r`, from `r(t) = m(t) + σ₁ ∫ e^{−a(t−s)} dW₁ + σ₂/(a−b) ∫ (e^{−b(t−s)} − e^{−a(t−s)}) dW₂`.
fn flat_curve_theta(t: f64) -> f64 {
  let c = SIGMA2 / (A - B);
  let (ea, eb) = ((-A * t).exp(), (-B * t).exp());
  let (ba, bb) = ((1.0 - ea) / A, (1.0 - eb) / B);
  let g = SIGMA1 * SIGMA1 * ba * ba
    + c * c * (bb - ba) * (bb - ba)
    + 2.0 * RHO_RU * SIGMA1 * c * ba * (bb - ba);
  let dg = 2.0 * SIGMA1 * SIGMA1 * ba * ea
    + 2.0 * c * c * (bb - ba) * (eb - ea)
    + 2.0 * RHO_RU * SIGMA1 * c * (ea * (bb - ba) + ba * (eb - ea));
  A * FLAT_RATE + 0.5 * (dg + A * g)
}

/// `(strike, implied volatility in percent, its standard deviation in percent)` per maturity, transcribed.
type Table = [[(f64, f64, f64); 5]; 4];

/// Table 3: Feller satisfied, `κ = 0.8`, `θ = 0.2`, `σ = 0.2`.
const TABLE_3: Table = [
  [
    (0.8869, 44.81, 0.19),
    (0.9324, 44.67, 0.23),
    (1.0305, 44.40, 0.30),
    (1.1388, 44.16, 0.38),
    (1.1972, 44.04, 0.42),
  ],
  [
    (0.8308, 44.59, 0.11),
    (0.9290, 45.07, 0.12),
    (1.1618, 37.89, 0.15),
    (1.4530, 30.86, 0.23),
    (1.6248, 27.52, 0.25),
  ],
  [
    (0.8400, 44.57, 0.09),
    (0.9839, 44.44, 0.13),
    (1.3499, 44.22, 0.25),
    (1.8519, 44.00, 0.40),
    (2.1692, 43.90, 0.48),
  ],
  [
    (0.9316, 44.55, 0.18),
    (1.1651, 44.46, 0.22),
    (1.8221, 44.31, 0.38),
    (2.8497, 44.16, 0.45),
    (3.5638, 44.08, 0.52),
  ],
];

/// Table 4: Feller violated, `κ = 0.4`, `θ = 0.2`, `σ = 0.6`.
const TABLE_4: Table = [
  [
    (0.8869, 43.12, 0.15),
    (0.9324, 42.53, 0.16),
    (1.0305, 41.48, 0.16),
    (1.1388, 40.71, 0.20),
    (1.1972, 40.44, 0.26),
  ],
  [
    (0.8308, 40.29, 0.08),
    (0.9290, 39.59, 0.09),
    (1.1618, 38.40, 0.13),
    (1.4530, 37.59, 0.17),
    (1.6248, 37.33, 0.17),
  ],
  [
    (0.8400, 39.82, 0.14),
    (0.9839, 39.22, 0.17),
    (1.3499, 38.17, 0.23),
    (1.8519, 37.37, 0.35),
    (2.1692, 37.09, 0.40),
  ],
  [
    (0.9316, 39.71, 0.06),
    (1.1651, 39.24, 0.06),
    (1.8221, 38.40, 0.15),
    (2.8497, 37.73, 0.30),
    (3.5638, 37.48, 0.41),
  ],
];

/// The model of experiments 4.2 and 4.3 on a monthly grid to twenty years: `S₀ = 1`, `v₀ = θ`,
/// `r₀ = 0.03`, correlations `ρ_xv = −0.3`, `ρ_xr = 0.35`, `ρ_xu = 0.08`, `ρ_ru = −0.4`.
fn model(
  kappa: f64,
  theta: f64,
  sigma: f64,
) -> HestonHullWhite2F<
  f64,
  stochastic_rs::stochastic::volatility::heston_hull_white_2f::AndersenQe,
  Deterministic,
> {
  let rho = array![
    [1.0, -0.3, 0.35, 0.08],
    [-0.3, 1.0, 0.0, 0.0],
    [0.35, 0.0, 1.0, RHO_RU],
    [0.08, 0.0, RHO_RU, 1.0],
  ];
  HestonHullWhite2F::new(
    1.0,
    theta,
    kappa,
    theta,
    sigma,
    flat_curve_theta as fn(f64) -> f64,
    A,
    SIGMA1,
    SIGMA2,
    B,
    rho,
    Some(FLAT_RATE),
    20 * STEPS_PER_YEAR + 1,
    Some(20.0),
    Deterministic::new(SEED),
  )
  .qe()
}

/// Spot and the trapezoidal `∫₀ᵀ r` at each table maturity, the latter from the rate row itself so
/// the comparison does not rest on the process's own integrated-rate row.
fn terminal_states(kappa: f64, theta: f64, sigma: f64) -> Vec<[(f64, f64); 4]> {
  let dt = 1.0 / STEPS_PER_YEAR as f64;
  model(kappa, theta, sigma).sample_map(PATHS, |path| {
    let mut out = [(0.0, 0.0); 4];
    let mut int_r = 0.0;
    let mut next = 0;
    for i in 1..path.r.len() {
      int_r += 0.5 * (path.r[i - 1] + path.r[i]) * dt;
      if i == MATURITIES[next].1 {
        out[next] = (path.s[i], int_r);
        next += 1;
        if next == MATURITIES.len() {
          break;
        }
      }
    }
    out
  })
}

fn mean_and_se(values: impl Iterator<Item = f64>) -> (f64, f64) {
  let values = values.collect::<Vec<_>>();
  let n = values.len() as f64;
  let mean = values.iter().sum::<f64>() / n;
  let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
  (mean, (var / n).sqrt())
}

fn compare(
  name: &str,
  kappa: f64,
  theta: f64,
  sigma: f64,
  table: &Table,
  held_out: &[(usize, usize)],
) {
  let states = terminal_states(kappa, theta, sigma);
  let black = BsmPricer::new(0.2, BsmCoc::Merton1973);
  let mut worst = 0.0_f64;
  println!("{name}: T, K, implied vol (paper, this run), z");
  for (m, &(tau, _)) in MATURITIES.iter().enumerate() {
    let (discount, discount_se) = mean_and_se(states.iter().map(|s| (-s[m].1).exp()));
    let z_discount = (discount - (-FLAT_RATE * tau).exp()) / discount_se;
    assert!(
      z_discount.abs() < 5.0,
      "{name}, T = {tau}: discount factor {discount} ± {discount_se}, the curve says {}",
      (-FLAT_RATE * tau).exp()
    );
    for (j, &(strike, paper_iv, paper_sd)) in table[m].iter().enumerate() {
      let (price, price_se) = mean_and_se(
        states
          .iter()
          .map(|s| (strike - s[m].0).max(0.0) * (-s[m].1).exp()),
      );
      let iv = black.implied_volatility(price, 1.0, strike, FLAT_RATE, 0.0, tau, OptionType::Put);
      assert!(
        iv.is_finite(),
        "{name}, T = {tau}, K = {strike}: price {price} has no implied volatility"
      );
      let vega = BsmPricer::new(iv, BsmCoc::Merton1973).vega(1.0, strike, FLAT_RATE, 0.0, tau);
      let iv_se = 100.0 * price_se / vega;
      let z = (100.0 * iv - paper_iv) / (paper_sd * paper_sd + iv_se * iv_se).sqrt();
      let held = held_out.contains(&(m, j));
      println!(
        "{tau:>5}  {strike:.4}  {paper_iv:.2} ({paper_sd:.2})  {:.2} ({iv_se:.2})  z = {z:+.2}{}",
        100.0 * iv,
        if held { "  (held out)" } else { "" }
      );
      if !held {
        worst = worst.max(z.abs());
      }
    }
  }
  assert!(
    worst < Z_MAX,
    "{name}: a cell is {worst:.2} standard errors off, above {Z_MAX}"
  );
}

#[test]
fn table_3_feller_satisfied() {
  compare("table 3", 0.8, 0.2, 0.2, &TABLE_3, &HELD_OUT);
}

#[test]
fn table_4_feller_violated() {
  compare("table 4", 0.4, 0.2, 0.6, &TABLE_4, &[]);
}

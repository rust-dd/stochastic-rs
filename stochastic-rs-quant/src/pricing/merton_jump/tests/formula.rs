//! The per-term volatility of the Poisson series and the properties that fix it:
//!
//! ```text
//! σ_n = √(d² + z²·n/τ)
//! ```
//!
//! The diffusion runs all of `τ` whatever the jump count, so `σ_0 = d`, the diffusive volatility.
//! Reference: Merton (1976), J. Financ. Econ. 3, 125–144, eq. (18)
//! Reference: Haug (2007), The Complete Guide to Option Pricing Formulas, 2nd ed., §6.9.1

use super::*;

/// At `gamma = 0` the jumps have zero size, so `σ_n = d = sigma` and the series is Black-Scholes at
/// `sigma` for every `lambda`.
#[test]
fn merton_gamma_zero_is_black_scholes() {
  let bs = BsmPricer::new(0.2, BsmCoc::Bsm1973);
  let (want_c, want_p) = (
    bs.price_call(S, K, R, Q, TAU),
    bs.price_put(S, K, R, Q, TAU),
  );
  for &lambda in &[0.1, 0.5, 2.0, 10.0] {
    let m = Merton1976Pricer::new(0.2, lambda, 0.0, 60, BsmCoc::Bsm1973);
    let CallPut { call, put } = m.price_call_put(S, K, R, Q, TAU);
    assert!(
      (call - want_c).abs() < 1e-12,
      "lambda={lambda}: call {call}, Black-Scholes {want_c}"
    );
    assert!(
      (put - want_p).abs() < 1e-12,
      "lambda={lambda}: put {put}, Black-Scholes {want_p}"
    );
  }
}

/// `σ_0` is the diffusive volatility, not `0`: the no-jump term still diffuses.
#[test]
fn merton_term_vol_at_zero_jumps_is_the_diffusive_volatility() {
  let m = merton(0.5, 0.4, 10);
  let d = m.diffusive_std();
  assert!(d > 0.0, "diffusive vol should be positive: {d}");
  assert_eq!(m.term_vol(0, TAU), d);
  // and it does not depend on tau, unlike every later term
  assert_eq!(m.term_vol(0, 3.0), d);
  assert!(m.term_vol(1, TAU) > m.term_vol(0, TAU));
}

/// The Poisson mean of the `n`-conditional variance is the total `d²τ + λτz² = σ²τ`, so `sigma` is
/// the model's total volatility.
#[test]
fn merton_conditional_variance_averages_to_the_total() {
  for &(lambda, gamma) in &[(0.5, 0.4), (0.5, 0.3), (2.0, 0.6), (0.25, 0.8)] {
    let m = Merton1976Pricer::new(0.2, lambda, gamma, 40, BsmCoc::Bsm1973);
    let mean_var = (0..m.m)
      .map(|n| m.poisson_weight(n, TAU) * m.term_vol(n, TAU).powi(2) * TAU)
      .sum::<f64>();
    let total = m.sigma * m.sigma * TAU;
    assert!(
      (mean_var - total).abs() < 1e-15,
      "lambda={lambda} gamma={gamma}: mean conditional variance {mean_var}, total {total}"
    );
  }
}

/// At fixed `gamma` and growing `lambda` (`z² = γσ²/λ`) the model tends to Black-Scholes at `sigma`
/// as `O(1/λ)`: quadrupling `lambda` cuts the gap at least threefold.
#[test]
fn merton_dense_small_jumps_tend_to_black_scholes() {
  let bs = BsmPricer::new(0.2, BsmCoc::Bsm1973).price_call(S, K, R, Q, TAU);
  let gap = |lambda: f64, m: usize| {
    (Merton1976Pricer::new(0.2, lambda, 0.4, m, BsmCoc::Bsm1973).price_call(S, K, R, Q, TAU) - bs)
      .abs()
  };
  let coarse = gap(50.0, 120);
  let fine = gap(200.0, 340);
  assert!(coarse < 1e-2, "lambda=50 gap {coarse} is not small");
  assert!(
    fine < coarse / 3.0,
    "quadrupling lambda must cut the gap by 3x: {coarse} -> {fine}"
  );
}

/// Haug's two printed examples at his parameterisation (`υ = sigma`, `γ = gamma`, `b = r`, so
/// [`BsmCoc::Bsm1973`]), to his printed four and six decimals.
/// Reference: Haug (2007), The Complete Guide to Option Pricing Formulas, 2nd ed., §6.9.1
#[test]
fn merton_matches_haugs_published_examples() {
  let first = Merton1976Pricer::new(0.25, 3.0, 0.40, 50, BsmCoc::Bsm1973);
  let got = first.price_call(45.0, 55.0, 0.10, 0.0, 0.25);
  assert!((got - 0.2417).abs() < 1e-4, "Haug 0.2417, got {got}");

  let second = Merton1976Pricer::new(0.25, 1.0, 0.25, 50, BsmCoc::Bsm1973);
  let got = second.price_call(100.0, 80.0, 0.08, 0.0, 0.25);
  assert!((got - 21.735476).abs() < 1e-6, "Haug 21.735476, got {got}");
}

/// The series against a Merton (1976) reference sharing no code with the pricer: its own Poisson
/// weights, `σ_n` from the public fields, and a hand-written Black-Scholes call per term.
#[test]
fn merton_matches_a_hand_written_haug_series() {
  use stochastic_rs_distributions::special::norm_cdf;

  let m = merton(0.5, 0.4, 10);
  let jump_var = m.sigma * m.sigma * m.gamma / m.lambda;
  let diffusive_var = m.sigma * m.sigma - m.lambda * jump_var;
  let lt = m.lambda * TAU;
  let disc = (-R * TAU).exp();

  let mut weight = (-lt).exp();
  let mut want = 0.0;
  for n in 0..m.m {
    if n > 0 {
      weight *= lt / n as f64;
    }
    let sigma = (diffusive_var + jump_var * n as f64 / TAU).sqrt();
    let sd = sigma * TAU.sqrt();
    // Bsm1973 carries at b = r, so the forward is S and the carry factor is 1.
    let d1 = ((S / K).ln() + R * TAU) / sd + 0.5 * sd;
    let d2 = d1 - sd;
    want += weight * (S * norm_cdf(d1) - K * disc * norm_cdf(d2));
  }

  let got = m.price_call(S, K, R, Q, TAU);
  assert!((got - want).abs() < TOL, "got {got}, hand-written {want}");
}

//! The per-term volatility of the Poisson series, and the three properties
//! that fix it.
//!
//! Conditional on `n` jumps, the log-return over the option's life is the
//! diffusion plus `n` i.i.d. jump sizes, so its variance is `d²τ + n·z²`:
//! the diffusion runs for the whole of `τ` however many jumps land in it.
//! A Black-Scholes term consumes that as `σ_n²τ`, so
//!
//! ```text
//! σ_n = √(d² + z²·n/τ)
//! ```
//!
//! which is Merton (1976) eq. (18) and the formula Haug prints in §6.9.1.
//! In particular `σ_0 = d`, the diffusive volatility.
//!
//! Every test below fails loudly against the `√((d² + z²)·n/τ)` this crate
//! used to compute. All but the last two are self-adjudicating — they need
//! no reference value at all, only a property the model must have.

use super::*;

/// Zero-size jumps are not jumps: at `gamma = 0` the whole variance is
/// diffusive, `z = 0`, and `σ_n = d = v` for every term, so the Poisson
/// series collapses to plain Black-Scholes at `v` **whatever `lambda`
/// is** — a jump that never moves the price cannot change what the option
/// is worth.
///
/// The decisive discriminator, and it needs no reference value: the
/// superseded `√((d² + z²)·n/τ)` gives `σ_n = v√(n/τ)`, which prices the
/// same option at 0.34 at `λ = 0.1` and at 16.20 at `λ = 10` — a
/// four-and-a-half-fold spread across a parameter the payoff does not
/// depend on, against a Black-Scholes value of 4.58.
#[test]
fn merton_gamma_zero_is_black_scholes() {
  let bs = BSMPricer::new(0.2, BSMCoc::Bsm1973);
  let (want_c, want_p) = bs.call_put(S, K, R, Q, TAU);
  for &lambda in &[0.1, 0.5, 2.0, 10.0] {
    let m = Merton1976Pricer::new(0.2, lambda, 0.0, 60, BSMCoc::Bsm1973);
    let (call, put) = m.call_put(S, K, R, Q, TAU);
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

/// `σ_0` is the diffusive volatility, not `0`. The `n = 0` term is the
/// no-*jump* term, not the no-*diffusion* term: the Brownian part is still
/// running.
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

/// Averaging the `n`-conditional variance over `N ~ Poisson(λτ)` returns
/// the total variance the caller asked for: `d²τ + λτ·z² = v²τ`.
///
/// This is the identity `diffusive_std`'s `v² − λz²` subtraction exists to
/// arrange, and it is what makes the `v` field the *total* volatility its
/// name claims. Under the superseded formula the same average came to
/// `(d² + z²)λτ`, which misses `v²τ` by −30 % to +40 % depending on
/// `lambda` — so `v` was not the model's volatility at all.
#[test]
fn merton_conditional_variance_averages_to_the_total() {
  for &(lambda, gamma) in &[(0.5, 0.4), (0.5, 0.3), (2.0, 0.6), (0.25, 0.8)] {
    let m = Merton1976Pricer::new(0.2, lambda, gamma, 40, BSMCoc::Bsm1973);
    let mean_var = (0..m.m)
      .map(|n| m.poisson_weight(n, TAU) * m.term_vol(n, TAU).powi(2) * TAU)
      .sum::<f64>();
    let total = m.v * m.v * TAU;
    assert!(
      (mean_var - total).abs() < 1e-15,
      "lambda={lambda} gamma={gamma}: mean conditional variance {mean_var}, total {total}"
    );
  }
}

/// Many tiny jumps are a second Brownian motion: holding the jump share
/// `gamma` fixed while `lambda` grows shrinks each jump as `z² = γv²/λ`,
/// so by the central limit theorem the compound Poisson part converges to
/// a diffusion of variance rate `γv²` and the model tends to
/// Black-Scholes at the total `v`.
///
/// The approach is `O(1/λ)` — the jump part's excess kurtosis decays at
/// that rate — so this pins the *rate* rather than a limit: quadrupling
/// `lambda` must cut the gap by at least three. The superseded formula
/// diverges instead of converging (`σ_n² ≈ (1−γ)v²λ`), so it fails on the
/// magnitude before it fails on the rate.
#[test]
fn merton_dense_small_jumps_tend_to_black_scholes() {
  let bs = BSMPricer::new(0.2, BSMCoc::Bsm1973)
    .call_put(S, K, R, Q, TAU)
    .0;
  let gap = |lambda: f64, m: usize| {
    (Merton1976Pricer::new(0.2, lambda, 0.4, m, BSMCoc::Bsm1973)
      .call_put(S, K, R, Q, TAU)
      .0
      - bs)
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

/// The two worked examples Haug prints for this formula in *The Complete
/// Guide to Option Pricing Formulas* (jump-diffusion section, §6.9.1 in the
/// 2nd edition), reproduced at his own parameterisation — his `υ` is this
/// pricer's total volatility `v`, his `γ` its jump-variance share `gamma`,
/// and his terms carry at `b = r`, which is [`BSMCoc::Bsm1973`].
///
/// This is the one pin in the crate whose expected value comes from
/// published literature rather than from an internal identity or a
/// reference the crate's own authors wrote. It is also the sharpest
/// discriminator on record: the superseded `σ_n` prices the first example
/// at `0.9316` against Haug's `0.2417`, and the second at `22.0857` against
/// his `21.735476`.
///
/// Tolerances are set by Haug's printed precision — four decimals for the
/// first, six for the second — not by the crate's accuracy, which is
/// tighter.
#[test]
fn merton_matches_haugs_published_examples() {
  let first = Merton1976Pricer::new(0.25, 3.0, 0.40, 50, BSMCoc::Bsm1973);
  let got = first.price_call(45.0, 55.0, 0.10, 0.0, 0.25);
  assert!((got - 0.2417).abs() < 1e-4, "Haug 0.2417, got {got}");

  let second = Merton1976Pricer::new(0.25, 1.0, 0.25, 50, BSMCoc::Bsm1973);
  let got = second.price_call(100.0, 80.0, 0.08, 0.0, 0.25);
  assert!((got - 21.735476).abs() < 1e-6, "Haug 21.735476, got {got}");
}

/// The whole series against a hand-written Merton (1976) reference that
/// shares no code with the pricer: the Poisson weight is accumulated
/// independently, `σ_n` is rebuilt from the public `v`/`lambda`/`gamma`
/// fields, and each term is a hand-written Black-Scholes call rather than
/// a `BSMPricer` call.
#[test]
fn merton_matches_a_hand_written_haug_series() {
  use stochastic_rs_distributions::special::norm_cdf;

  let m = merton(0.5, 0.4, 10);
  let jump_var = m.v * m.v * m.gamma / m.lambda;
  let diffusive_var = m.v * m.v - m.lambda * jump_var;
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

//! Kyle (1985) strategic-trading equilibria with asymmetric information.
//!
//! Single-period: an insider observes the terminal value $\tilde v$ and submits
//! a market order $x$ to a competitive risk-neutral market maker who only sees
//! the aggregate order flow $y = x + \tilde u$ with noise $\tilde u \sim
//! \mathcal N(0, \sigma_u^2)$. The pricing rule $p = p_0 + \lambda y$ and
//! insider's strategy $x = \beta(\tilde v - p_0)$ form a Bayesian-Nash
//! equilibrium with
//!
//! $$
//! \beta = \sqrt{\sigma_u^2 / \Sigma_0},\qquad
//! \lambda = \tfrac{1}{2}\sqrt{\Sigma_0 / \sigma_u^2},\qquad
//! \mathbb E[\pi] = \tfrac{1}{2}\sqrt{\Sigma_0\,\sigma_u^2}.
//! $$
//!
//! Multi-period: the same structure with $\Sigma_n$, $\beta_n$, $\lambda_n$
//! determined by the Kyle 1985 / Cetin-Larsen 2023 backward recursion
//! (Theorem 2.1, eq. 2.4). Letting $\gamma_n := \alpha_n \lambda_n$ with
//! terminal $\gamma_N = 0$ and $\Delta = 1$, the equilibrium is solved by:
//!
//! 1. Backward induction in $\gamma_n$: $\gamma_n \in (0, 1/2)$ is the unique
//!    root of $8\gamma^2(1-\gamma)(1 - 2\gamma_{n+1}) = 1 - 2\gamma$ for the
//!    given $\gamma_{n+1}$. (Derivation: combine eq. 2.4's λ, β, α formulas
//!    after substituting $\beta_n^2 \Sigma_{n-1} \Delta + \sigma_u^2 = \beta_n
//!    \Sigma_{n-1}/\lambda_n$ and eliminating β.)
//! 2. Forward induction in Σ: $\Sigma_n = \Sigma_{n-1} / (2(1-\gamma_n))$.
//! 3. Per-period $(\lambda_n, \beta_n)$ from explicit formulas:
//!    $\lambda_n^2 = (1-2\gamma_n)\Sigma_{n-1}/(4(1-\gamma_n)^2 \sigma_u^2)$,
//!    $\beta_n = (1-2\gamma_n)/(2\lambda_n(1-\gamma_n))$.
//!
//! References:
//! - Kyle, "Continuous Auctions and Insider Trading", Econometrica, 53(6),
//!   1315-1335 (1985). DOI: 10.2307/1913210
//! - Cetin & Larsen, "Is Kyle's equilibrium model stable?", arXiv:2307.09392
//!   (2023), Theorem 2.1.

use crate::traits::FloatExt;

/// Equilibrium quantities of a single round of trading.
#[derive(Debug, Clone, Copy)]
pub struct KyleEquilibrium<T: FloatExt> {
  /// Insider trading intensity $\beta$.
  pub beta: T,
  /// Price-impact (Kyle's lambda) $\lambda$.
  pub lambda: T,
  /// Posterior variance of $\tilde v$ after the round.
  pub posterior_variance: T,
  /// Insider's expected profit over the round.
  pub expected_profit: T,
}

/// Solve the single-period Kyle equilibrium.
pub fn single_period_kyle<T: FloatExt>(prior_variance: T, noise_variance: T) -> KyleEquilibrium<T> {
  assert!(
    prior_variance > T::zero(),
    "prior_variance must be positive"
  );
  assert!(
    noise_variance > T::zero(),
    "noise_variance must be positive"
  );
  let beta = (noise_variance / prior_variance).sqrt();
  let lambda = T::from_f64_fast(0.5) * (prior_variance / noise_variance).sqrt();
  let expected_profit = T::from_f64_fast(0.5) * (prior_variance * noise_variance).sqrt();
  let posterior_variance = T::from_f64_fast(0.5) * prior_variance;
  KyleEquilibrium {
    beta,
    lambda,
    posterior_variance,
    expected_profit,
  }
}

/// Multi-period Kyle equilibrium with `n_periods` rounds and i.i.d.
/// per-period noise variance `noise_variance_per_round`.
///
/// Returns the per-round equilibrium $(\beta_n, \lambda_n, \Sigma_n,
/// \mathbb E[\pi_n])$ in chronological order, found by the Cetin-Larsen 2023
/// Theorem 2.1 backward recursion described in the module-level docs.
///
/// At `n_periods = 1` this matches [`single_period_kyle`] exactly (catches
/// the previous broken `α_{n-1} = (1-√α_n)/2` shortcut which produced
/// $\beta\lambda = 0.25$ instead of the correct Kyle product $0.5$).
pub fn multi_period_kyle<T: FloatExt>(
  prior_variance: T,
  noise_variance_per_round: T,
  n_periods: usize,
) -> Vec<KyleEquilibrium<T>> {
  assert!(
    prior_variance > T::zero(),
    "prior_variance must be positive"
  );
  assert!(
    noise_variance_per_round > T::zero(),
    "noise_variance_per_round must be positive"
  );
  assert!(n_periods >= 1, "n_periods must be at least 1");

  let gammas = backward_gamma_sequence::<T>(n_periods);
  let two = T::from_f64_fast(2.0);
  let four = T::from_f64_fast(4.0);

  let mut sigma = prior_variance;
  let mut out = Vec::with_capacity(n_periods);
  for &gamma in &gammas {
    let one_minus_gamma = T::one() - gamma;
    let one_minus_two_gamma = T::one() - two * gamma;

    // λ_n² = (1 - 2γ_n) Σ_{n-1} / (4 (1-γ_n)² σ_u²) — Kyle/Cetin-Larsen with Δ=1.
    let lambda_sq = one_minus_two_gamma * sigma
      / (four * one_minus_gamma * one_minus_gamma * noise_variance_per_round);
    let lambda = lambda_sq.sqrt();

    // β_n = (1 - 2γ_n) / (2 λ_n (1-γ_n)).
    let beta = one_minus_two_gamma / (two * lambda * one_minus_gamma);

    // Σ_n = Σ_{n-1} / (2(1-γ_n)).
    let sigma_next = sigma / (two * one_minus_gamma);

    // Per-round expected profit: E[π_n] = β_n Σ_{n-1} (1 - λ_n β_n).
    // At terminal (γ=0) this collapses to ½√(Σ_{n-1}σ_u²) — the single-period value.
    let expected_profit = beta * sigma * (T::one() - lambda * beta);

    out.push(KyleEquilibrium {
      beta,
      lambda,
      posterior_variance: sigma_next,
      expected_profit,
    });
    sigma = sigma_next;
  }
  out
}

/// Solve $\gamma_n \in (0, 1/2)$ from the cubic
/// $8\gamma^2(1-\gamma)(1 - 2\gamma_{\text{next}}) = 1 - 2\gamma$.
///
/// At $\gamma = 0$ the LHS is 0 and the RHS is $1$, so the residual is negative;
/// at $\gamma = 1/2$ the LHS is $(1 - 2\gamma_{\text{next}}) \in (0, 1]$ and the
/// RHS is 0, so the residual is positive. Bisection on this monotone-in-sign
/// bracket converges to the unique physical root.
fn solve_gamma_cubic<T: FloatExt>(g_next: T) -> T {
  let half = T::from_f64_fast(0.5);
  let two = T::from_f64_fast(2.0);
  let eight = T::from_f64_fast(8.0);
  let coef = eight * (T::one() - two * g_next);
  let f = |g: T| coef * g * g * (T::one() - g) - (T::one() - two * g);

  let mut lo = T::from_f64_fast(1e-15);
  let mut hi = half - T::from_f64_fast(1e-15);
  for _ in 0..120 {
    let mid = (lo + hi) * half;
    if f(mid) < T::zero() {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  (lo + hi) * half
}

fn backward_gamma_sequence<T: FloatExt>(n_periods: usize) -> Vec<T> {
  // gammas[i] holds γ_{i+1}; terminal γ_N = 0 sits at index n_periods-1.
  let mut gammas = vec![T::zero(); n_periods];
  if n_periods >= 2 {
    for i in (0..n_periods - 1).rev() {
      gammas[i] = solve_gamma_cubic(gammas[i + 1]);
    }
  }
  gammas
}

#[cfg(test)]
mod tests {
  use super::*;

  fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
  }

  #[test]
  fn single_period_satisfies_beta_lambda_half() {
    let eq = single_period_kyle(0.04_f64, 1.0);
    assert!(approx(eq.beta * eq.lambda, 0.5, 1e-12));
  }

  #[test]
  fn single_period_lambda_scales_correctly() {
    let eq = single_period_kyle(1.0_f64, 1.0);
    assert!(approx(eq.lambda, 0.5, 1e-12));
    assert!(approx(eq.beta, 1.0, 1e-12));
    assert!(approx(eq.expected_profit, 0.5, 1e-12));
  }

  #[test]
  fn single_period_posterior_halves_prior() {
    let eq = single_period_kyle(2.5_f64, 0.1);
    assert!(approx(eq.posterior_variance, 1.25, 1e-12));
  }

  #[test]
  fn multi_period_returns_one_per_round() {
    let eqs = multi_period_kyle(1.0_f64, 1.0, 5);
    assert_eq!(eqs.len(), 5);
    for eq in &eqs {
      assert!(eq.lambda > 0.0);
      assert!(eq.beta > 0.0);
      assert!(eq.posterior_variance > 0.0);
    }
  }

  #[test]
  fn multi_period_posterior_decreases() {
    let eqs = multi_period_kyle(1.0_f64, 1.0, 8);
    let mut last = f64::INFINITY;
    for eq in &eqs {
      assert!(eq.posterior_variance < last);
      last = eq.posterior_variance;
    }
  }

  /// Decisive analytic-benchmark test: the multi-period recursion at N=1 must
  /// reproduce the single-period closed form exactly. The previous buggy
  /// `(1 - √α_n)/2` recursion gave `λ = 1.0, β = 0.25` at N=1 instead of the
  /// canonical `λ = 0.5, β = 1.0` (off by factor of 2 in λ and 4 in β).
  #[test]
  fn multi_period_one_round_matches_single_period() {
    let prior = 0.04_f64;
    let noise = 1.0;
    let single = single_period_kyle(prior, noise);
    let multi = multi_period_kyle(prior, noise, 1);
    assert_eq!(multi.len(), 1);
    assert!(
      approx(multi[0].lambda, single.lambda, 1e-12),
      "lambda mismatch: multi={} single={}",
      multi[0].lambda,
      single.lambda
    );
    assert!(
      approx(multi[0].beta, single.beta, 1e-12),
      "beta mismatch: multi={} single={}",
      multi[0].beta,
      single.beta
    );
    assert!(
      approx(multi[0].posterior_variance, single.posterior_variance, 1e-12),
      "posterior_variance mismatch: multi={} single={}",
      multi[0].posterior_variance,
      single.posterior_variance
    );
    assert!(
      approx(multi[0].expected_profit, single.expected_profit, 1e-12),
      "expected_profit mismatch: multi={} single={}",
      multi[0].expected_profit,
      single.expected_profit
    );
  }

  /// Two-period analytic benchmark from the Cetin-Larsen 2023 Theorem 2.1
  /// recursion with $\Sigma_0 = \sigma_u^2 = 1$. The terminal γ_N = 0 leads to
  /// γ_1 = unique root of $8\gamma^3 - 8\gamma^2 - 2\gamma + 1 = 0$ in
  /// (0, 1/2), which is γ_1 ≈ 0.27750. The remaining quantities follow:
  /// λ_1 ≈ 0.4617, β_1 ≈ 0.6669, Σ_1 ≈ 0.6920;
  /// λ_2 ≈ 0.4159, β_2 ≈ 1.2022, Σ_2 ≈ 0.3460.
  /// At each round β_n λ_n is below 1/2 (the static second-order condition),
  /// and at the terminal round β_N λ_N = 1/2 exactly.
  #[test]
  fn multi_period_two_round_matches_canonical() {
    let eqs = multi_period_kyle(1.0_f64, 1.0, 2);
    assert_eq!(eqs.len(), 2);

    assert!(
      approx(eqs[0].lambda, 0.4617, 1e-3),
      "λ_1 = {}",
      eqs[0].lambda
    );
    assert!(
      approx(eqs[0].beta, 0.6669, 1e-3),
      "β_1 = {}",
      eqs[0].beta
    );
    assert!(
      approx(eqs[0].posterior_variance, 0.6920, 1e-3),
      "Σ_1 = {}",
      eqs[0].posterior_variance
    );

    assert!(
      approx(eqs[1].lambda, 0.4159, 1e-3),
      "λ_2 = {}",
      eqs[1].lambda
    );
    assert!(
      approx(eqs[1].beta, 1.2022, 1e-3),
      "β_2 = {}",
      eqs[1].beta
    );
    assert!(
      approx(eqs[1].posterior_variance, 0.3460, 1e-3),
      "Σ_2 = {}",
      eqs[1].posterior_variance
    );

    // Terminal round: β_N λ_N = 1/2 exactly.
    assert!(
      approx(eqs[1].lambda * eqs[1].beta, 0.5, 1e-12),
      "terminal β_N·λ_N must equal 1/2, got {}",
      eqs[1].lambda * eqs[1].beta
    );
  }
}

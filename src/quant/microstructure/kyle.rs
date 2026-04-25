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
//! computed by backward induction.
//!
//! Reference: Kyle, "Continuous Auctions and Insider Trading", Econometrica,
//! 53(6), 1315-1335 (1985). DOI: 10.2307/1913210

use crate::traits::FloatExt;

/// Equilibrium quantities of a single round of trading.
#[derive(Debug, Clone, Copy)]
pub struct KyleEquilibrium<T: FloatExt> {
  /// Insider trading intensity $\beta$.
  pub beta: T,
  /// Price-impact (Kyle's lambda) $\lambda$.
  pub lambda: T,
  /// Posterior variance of $\tilde v$ after the round (one-period only).
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
/// \mathbb E[\pi_n])$ in chronological order, found by backward induction on
/// the recursion
///
/// $$
/// \alpha_{n-1} = \tfrac{1 - \sqrt{\alpha_n}}{2},\qquad
/// \lambda_n^2 = \tfrac{\alpha_n\,\Sigma_{n-1}}{(1-\alpha_n)\,\sigma_u^2},\qquad
/// \beta_n = \tfrac{1-\alpha_n}{2\lambda_n},\qquad
/// \Sigma_n = (1-\alpha_n)\Sigma_{n-1},
/// $$
///
/// with terminal condition $\alpha_{N-1} = 1/2$.
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

  let alphas = backward_alpha_sequence::<T>(n_periods);
  let mut sigma = prior_variance;
  let mut out = Vec::with_capacity(n_periods);
  for &alpha_n in &alphas {
    let one_minus_alpha = T::one() - alpha_n;
    let lambda = (alpha_n * sigma / (one_minus_alpha * noise_variance_per_round)).sqrt();
    let beta = one_minus_alpha / (T::from_f64_fast(2.0) * lambda);
    let sigma_next = one_minus_alpha * sigma;
    let expected_profit =
      lambda * noise_variance_per_round + (sigma - sigma_next) / (T::from_f64_fast(4.0) * lambda);
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

fn backward_alpha_sequence<T: FloatExt>(n_periods: usize) -> Vec<T> {
  let mut alphas = vec![T::zero(); n_periods];
  alphas[n_periods - 1] = T::from_f64_fast(0.5);
  for n in (0..n_periods - 1).rev() {
    let next = alphas[n + 1];
    alphas[n] = (T::one() - next.sqrt()) / T::from_f64_fast(2.0);
  }
  alphas
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
}

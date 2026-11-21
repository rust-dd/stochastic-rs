//! # Variance / volatility swap pricing
//!
//! Static-replication (Demeterfi–Derman–Kamal–Zou 1999) and stochastic-vol
//! closed-form fair strikes (Brockhaus–Long 2000 for Heston, Bernard–Cui
//! 2013 discrete corrections).
//!
//! Continuous Demeterfi formula (with continuous dividend yield $q$ and
//! forward $F = S_0 e^{(r-q)T}$, ATM cutoff $K_0$):
//!
//! $$
//! K_{\text{var}}=\frac{2}{T}\!\left[
//!   (r-q)T - \!\!\left(\!\frac{F}{K_0}\!-\!1\!\right)\! -
//!   \ln\frac{K_0}{S_0} +
//!   e^{rT}\!\!\int_0^{K_0}\!\!\frac{P(K)}{K^2}\,dK +
//!   e^{rT}\!\!\int_{K_0}^{\infty}\!\!\frac{C(K)}{K^2}\,dK
//! \right]
//! $$
//!
//! Heston closed-form (continuous monitoring):
//!
//! $$
//! K_{\text{var}}^{\text{Heston}}=\theta+(V_0-\theta)\,\frac{1-e^{-\kappa T}}{\kappa T}
//! $$
//!
//! Source:
//! - Demeterfi, K., Derman, E., Kamal, M. & Zou, J. (1999),
//!   "More Than You Ever Wanted to Know About Volatility Swaps", Goldman Sachs.
//! - Brockhaus, O. & Long, D. (2000), "Volatility Swaps Made Simple", Risk 13.
//! - Swishchuk, A. & Wang, Z. (2017), arXiv:1712.02735, eqs. 10–13: Brockhaus–Long for Heston.
//! - Bernard, C. & Cui, Z. (2013), "Prices and Asymptotics for Discrete
//!   Variance Swaps", arXiv:1305.7092.
//! - Carr, P. & Madan, D. (1998), "Towards a Theory of Volatility Trading",
//!   in *Volatility: New Estimation Techniques for Pricing Derivatives*.

use crate::pricing::exotic::VarianceSwapTerms;
use crate::pricing::heston::HestonParams;
use crate::total_order::ascending;

mod discrete;

/// Variance-swap fair strikes, realised variance and P&L; each takes the query it needs.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct VarianceSwapPricer;

impl VarianceSwapPricer {
  /// Black–Scholes fair strike: $K_{\text{var}} = \sigma^2$.
  pub fn fair_strike_bsm(sigma: f64) -> f64 {
    sigma * sigma
  }

  /// Demeterfi–Derman–Kamal–Zou fair strike from an ascending OTM strip, `K_0` nearest the forward,
  /// floored at 0 unless NaN, NaN unless `s, tau > 0`; panics on bad lengths (order: debug only).
  pub fn fair_strike_replication(
    strikes: &[f64],
    otm_prices: &[f64],
    s: f64,
    r: f64,
    q: f64,
    tau: f64,
  ) -> f64 {
    assert_eq!(
      strikes.len(),
      otm_prices.len(),
      "strikes / prices length mismatch"
    );
    let n = strikes.len();
    assert!(
      n >= 2,
      "static replication needs at least 2 strikes (got {n})"
    );
    if !(s > 0.0 && tau > 0.0) {
      return f64::NAN;
    }
    debug_assert!(
      strikes.windows(2).all(|w| w[0] <= w[1]),
      "strikes must be sorted ascending"
    );
    debug_assert!(
      strikes.iter().all(|k| k.is_finite()),
      "strikes must be finite (no NaN)"
    );

    let fwd = s * ((r - q) * tau).exp();
    let disc = (r * tau).exp();

    let k0_idx = strikes
      .iter()
      .enumerate()
      .min_by(|(_, a), (_, b)| ascending(&(*a - fwd).abs(), &(*b - fwd).abs()))
      .map(|(i, _)| i)
      .unwrap_or(0);
    let k0 = strikes[k0_idx];

    let mut integral = 0.0;
    for i in 0..n {
      let dk = if i == 0 {
        strikes[1] - strikes[0]
      } else if i == n - 1 {
        strikes[n - 1] - strikes[n - 2]
      } else {
        0.5 * (strikes[i + 1] - strikes[i - 1])
      };
      integral += dk * otm_prices[i] / (strikes[i] * strikes[i]);
    }

    let drift = (r - q) * tau;
    let fair = (2.0 / tau) * (drift - (fwd / k0 - 1.0) - (k0 / s).ln() + disc * integral);
    if fair.is_nan() { fair } else { fair.max(0.0) }
  }

  /// Brockhaus–Long continuous-monitoring fair strike, reading `v0`, `kappa` and `theta`: `v0` at
  /// `tau = 0`, at `kappa = 0` and at `v0 = theta`, and NaN for a negative or NaN `tau`.
  pub fn fair_strike_heston(params: &HestonParams, tau: f64) -> f64 {
    let HestonParams {
      v0, kappa, theta, ..
    } = *params;
    if tau.is_nan() || tau < 0.0 {
      return f64::NAN;
    }
    let x = if kappa == 0.0 { 0.0 } else { kappa * tau };
    if tau == 0.0 || (v0 == theta && !x.is_nan()) {
      return v0;
    }
    theta + (v0 - theta) * discrete::mean_reversion_factor(x)
  }

  /// [`fair_strike_heston`](Self::fair_strike_heston) plus Bernard–Cui's first-order correction
  /// for `n_obs` observations (Proposition 6.1); `n_obs = 0` is the continuous limit.
  pub fn fair_strike_heston_discrete(
    params: &HestonParams,
    n_obs: usize,
    r: f64,
    q: f64,
    tau: f64,
  ) -> f64 {
    let HestonParams {
      v0,
      kappa,
      theta,
      sigma,
      rho,
    } = *params;
    let cont = Self::fair_strike_heston(params, tau);
    if n_obs == 0 {
      return cont;
    }
    let dt = tau / n_obs as f64;
    let mean_square = discrete::mean_square_variance(v0, kappa, theta, sigma, tau);
    let drift = r - q;
    let bias = dt * (drift * drift - drift * cont + 0.25 * mean_square - 0.5 * rho * sigma * cont);
    cont + bias
  }

  /// Realised variance $\frac{1}{N\Delta t}\sum_{i=1}^N (\ln S_i/S_{i-1})^2$ of a path sampled
  /// every `dt` years (`1.0 / 252.0` on trading days); panics on fewer than two prices.
  pub fn realized_variance(prices: &[f64], dt: f64) -> f64 {
    assert!(
      prices.len() >= 2,
      "realized variance needs at least 2 prices (got {})",
      prices.len()
    );
    let n = prices.len() - 1;
    let mut rv = 0.0;
    for i in 1..=n {
      let lr = (prices[i] / prices[i - 1]).ln();
      rv += lr * lr;
    }
    rv / (n as f64 * dt)
  }

  /// P&L of a long variance-swap position, $N \times (\hat\sigma^2 - K_{\text{var}})$.
  pub fn pnl(realized_var: f64, terms: &VarianceSwapTerms) -> f64 {
    terms.notional * (realized_var - terms.strike_variance)
  }
}

/// Log-contract replication weights $\frac{2}{T}\frac{\Delta K_i}{K_i^2}$ (Demeterfi et al. eq. 28)
/// in `strikes` order; panics on fewer than two strikes, all NaN unless `maturity > 0`.
pub fn replication_weights(strikes: &[f64], maturity: f64) -> Vec<f64> {
  let n = strikes.len();
  assert!(
    n >= 2,
    "static replication needs at least 2 strikes (got {n})"
  );
  if maturity.is_nan() || maturity <= 0.0 {
    return vec![f64::NAN; n];
  }

  let mut w = vec![0.0; n];
  for i in 0..n {
    let dk = if i == 0 {
      strikes[1] - strikes[0]
    } else if i == n - 1 {
      strikes[n - 1] - strikes[n - 2]
    } else {
      0.5 * (strikes[i + 1] - strikes[i - 1])
    };
    w[i] = (2.0 / maturity) * dk / (strikes[i] * strikes[i]);
  }
  w
}

/// Volatility-swap fair strike with convexity correction.
///
/// Naive: $K_{\text{vol}} \approx \sqrt{K_{\text{var}}}$. With variance-of-variance
/// the convex Jensen correction lowers the strike:
///
/// $$
/// K_{\text{vol}} \approx \sqrt{K_{\text{var}}} - \frac{\text{Var}(V)}{8\,K_{\text{var}}^{3/2}}
/// $$
pub struct VolatilitySwapPricer;

impl VolatilitySwapPricer {
  /// Black–Scholes vol strike: $K_{\text{vol}} = \sigma$.
  pub fn fair_strike_bsm(sigma: f64) -> f64 {
    sigma
  }

  /// Convexity-adjusted vol strike from the variance strike and the variance of variance; panics
  /// unless `k_var > 0`, since the Jensen term divides by $K_{\text{var}}^{3/2}$.
  pub fn fair_strike_from_var(k_var: f64, var_of_var: f64) -> f64 {
    assert!(
      k_var > 0.0,
      "variance strike k_var must be strictly positive (got {k_var})"
    );
    k_var.sqrt() - var_of_var / (8.0 * k_var.powf(1.5))
  }

  /// Brockhaus–Long's `√K − Var(V)/(8K^{3/2})`, within 0.5 % of `E√V` while `Var(V) ≤ K²/4`; NaN
  /// past `Var(V) = 8K²`, where it turns negative, and for a negative variance strike `K`.
  pub fn fair_strike_heston(params: &HestonParams, tau: f64) -> f64 {
    let k_var = VarianceSwapPricer::fair_strike_heston(params, tau);
    if k_var.is_nan() || k_var < 0.0 {
      return f64::NAN;
    }
    if tau == 0.0 || k_var == 0.0 {
      return k_var.sqrt();
    }
    let k_vol =
      Self::fair_strike_from_var(k_var, discrete::realized_variance_variance(params, tau));
    if k_vol < 0.0 { f64::NAN } else { k_vol }
  }
}

#[cfg(test)]
mod tests;

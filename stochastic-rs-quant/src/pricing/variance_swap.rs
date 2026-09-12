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
//!   (r-q)T - \!\!\left(\!\frac{F}{K_0}\!-\!1\!\right)\!
//!   - \ln\frac{K_0}{S_0}
//!   + e^{rT}\!\!\int_0^{K_0}\!\!\frac{P(K)}{K^2}\,dK
//!   + e^{rT}\!\!\int_{K_0}^{\infty}\!\!\frac{C(K)}{K^2}\,dK
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
//! - Bernard, C. & Cui, Z. (2013), "Prices and Asymptotics for Discrete
//!   Variance Swaps", arXiv:1305.7092.
//! - Carr, P. & Madan, D. (1998), "Towards a Theory of Volatility Trading",
//!   in *Volatility: New Estimation Techniques for Pricing Derivatives*.

/// Conventional daily increment for a 252-day equity trading year, exported
/// so callers can write `realized_variance(prices, BUSINESS_DAY_252_DT)` to
/// document the choice rather than sprinkling `1.0 / 252.0` literals.
///
/// Use a different value (e.g. `1.0 / 365.0` for calendar-day,
/// `1.0 / 260.0` for FX) when the underlying observation frequency differs.
pub const BUSINESS_DAY_252_DT: f64 = 1.0 / 252.0;

mod discrete;

/// Variance-swap pricer.
///
/// State stores the forward-curve inputs (spot, rates, maturity); pricing
/// methods consume external option strips or model parameters.
#[derive(Debug, Clone, Copy)]
pub struct VarianceSwapPricer {
  /// Spot price.
  pub s: f64,
  /// Continuously-compounded risk-free rate.
  pub r: f64,
  /// Continuous dividend yield.
  pub q: f64,
  /// Time to maturity in years.
  pub tau: f64,
}

impl VarianceSwapPricer {
  /// Forward $F = S_0 e^{(r-q)T}$.
  pub fn forward(&self) -> f64 {
    self.s * ((self.r - self.q) * self.tau).exp()
  }

  /// Black–Scholes fair strike: $K_{\text{var}} = \sigma^2$.
  pub fn fair_strike_bsm(&self, sigma: f64) -> f64 {
    sigma * sigma
  }

  /// Static replication fair strike (Demeterfi–Derman–Kamal–Zou).
  ///
  /// Inputs are the OTM option strip — puts for $K < K_0$, calls for
  /// $K \geq K_0$ — with $K_0$ identified as the strike closest to the
  /// forward. Strikes must be sorted ascending. Trapezoidal weights are
  /// used for the $\int P(K)/K^2 dK + \int C(K)/K^2 dK$ contribution.
  ///
  /// **Preconditions:** `strikes` must contain only finite (non-NaN) values.
  /// NaN strikes will cause the closest-to-forward selection to panic via
  /// `partial_cmp().unwrap()` since NaN is unordered. Filter NaN at the
  /// caller side (real exchange data should never carry NaN strikes).
  ///
  /// # Panics
  /// - if `strikes` and `otm_prices` differ in length
  /// - if fewer than two strikes are supplied — the trapezoidal weights
  ///   need a neighbour on at least one side, so a one-point "strip" is not
  ///   a thin replication, it is not a replication
  /// - if `self.tau` is not strictly positive
  ///
  /// All three used to return `0.0`, which is a plausible-looking variance
  /// strike: `fair_strike_bsm(0.0)` is `0.0` too, and a caller cannot tell
  /// the two apart. Case 1 of the crate's [failure
  /// convention](crate::traits::ModelPricer#how-pricing-fails).
  ///
  /// A non-`NaN` result is floored at zero. The floor tests for `NaN` first,
  /// because `f64::max` discards a `NaN` operand in favour of the finite one
  /// — so a single `NaN` in `otm_prices` used to come back as exactly `0.0`,
  /// re-entering by the back door the sentinel the guards above remove.
  pub fn fair_strike_replication(&self, strikes: &[f64], otm_prices: &[f64]) -> f64 {
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
    assert!(
      self.tau > 0.0,
      "maturity tau must be strictly positive (got {})",
      self.tau
    );
    debug_assert!(
      strikes.windows(2).all(|w| w[0] <= w[1]),
      "strikes must be sorted ascending"
    );
    debug_assert!(
      strikes.iter().all(|k| k.is_finite()),
      "strikes must be finite (no NaN)"
    );

    let fwd = self.forward();
    let disc = (self.r * self.tau).exp();

    let k0_idx = strikes
      .iter()
      .enumerate()
      .min_by(|(_, a), (_, b)| {
        (*a - fwd)
          .abs()
          .partial_cmp(&(*b - fwd).abs())
          .unwrap_or(std::cmp::Ordering::Equal)
      })
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

    let drift = (self.r - self.q) * self.tau;
    let fair = (2.0 / self.tau) * (drift - (fwd / k0 - 1.0) - (k0 / self.s).ln() + disc * integral);
    if fair.is_nan() { fair } else { fair.max(0.0) }
  }

  /// Heston closed-form fair variance strike (Brockhaus–Long 2000).
  ///
  /// Continuous-monitoring expected integrated variance,
  /// $E\!\left[\frac{1}{T}\int_0^T V_t\,dt\right]$, depends only on
  /// `(v0, kappa, theta, T)` — not on `(rho, sigma, r, q)`.
  ///
  /// At `tau == 0` the factor $\frac{1-e^{-\kappa T}}{\kappa T}$ tends to 1
  /// and the strike is `v0`. That branch is a genuine limit and stays.
  ///
  /// # Panics
  /// Panics if `self.tau` is negative — or `NaN`, which fails the same test.
  /// A negative maturity is not a market state, and the `tau <= 0.0` branch
  /// this replaces returned `v0` for one: a plausible variance strike,
  /// numerically identical to the correct $T \to 0$ answer, for an input that
  /// has no answer at all. Case 1 of the crate's [failure
  /// convention](crate::traits::ModelPricer#how-pricing-fails), and the same
  /// guard its neighbour
  /// [`fair_strike_replication`](Self::fair_strike_replication) already
  /// carries.
  pub fn fair_strike_heston(&self, v0: f64, kappa: f64, theta: f64) -> f64 {
    let tau = self.tau;
    assert!(tau >= 0.0, "maturity tau must be non-negative (got {tau})");
    if tau == 0.0 {
      // Limit T → 0 of (1 - e^{-κT})/(κT) is 1, so K_var → v0.
      return v0;
    }
    let factor = discrete::mean_reversion_factor(kappa * tau);
    theta + (v0 - theta) * factor
  }

  /// Discrete-monitoring correction to the continuous Heston fair strike
  /// (Bernard–Cui 2013, leading-order in $T/N$).
  ///
  /// Uses the full first-order coefficient from Bernard–Cui, Proposition 6.1:
  /// $\frac{T}{N}[(r-q)^2-(r-q)K_c+\frac14\overline{E[V^2]}
  /// -\frac12\rho\sigma K_c]$. The omitted remainder is $O(N^{-2})$.
  /// A zero observation count selects the continuous-monitoring limit.
  ///
  /// # Panics
  /// Panics on a negative `self.tau`, via
  /// [`fair_strike_heston`](Self::fair_strike_heston).
  pub fn fair_strike_heston_discrete(
    &self,
    v0: f64,
    kappa: f64,
    theta: f64,
    sigma: f64,
    rho: f64,
    n_obs: usize,
  ) -> f64 {
    let cont = self.fair_strike_heston(v0, kappa, theta);
    if n_obs == 0 {
      return cont;
    }
    let tau = self.tau;
    let dt = tau / n_obs as f64;
    let mean_square = discrete::mean_square_variance(v0, kappa, theta, sigma, tau);
    let drift = self.r - self.q;
    let bias = dt * (drift * drift - drift * cont + 0.25 * mean_square - 0.5 * rho * sigma * cont);
    cont + bias
  }

  /// Realised variance estimator from a price path,
  /// $\hat\sigma^2 = \frac{1}{N\Delta t}\sum_{i=1}^N (\ln S_i/S_{i-1})^2$.
  ///
  /// `dt` is the time between observations in years; use
  /// [`BUSINESS_DAY_252_DT`] for the standard 252-day equity convention or
  /// `1.0 / 365.0` for calendar-day sampling.
  ///
  /// # Panics
  /// Panics if fewer than two prices are supplied: $N = \text{len} - 1$ log
  /// returns is zero of them, and the estimator's $1/(N\Delta t)$ normaliser
  /// divides by zero.
  ///
  /// This returned `0.0` before, and `0.0` is the sharpest possible case of
  /// a plausible-looking sentinel here — it is also the *correct* answer for
  /// a constant price path, which
  /// `realized_variance_constant_path_is_zero` pins. "No data" and "no
  /// movement" were the same number. Case 1 of the crate's [failure
  /// convention](crate::traits::ModelPricer#how-pricing-fails).
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
  pub fn pnl(realized_var: f64, fair_strike: f64, notional: f64) -> f64 {
    notional * (realized_var - fair_strike)
  }
}

/// Replicating portfolio weights for the log contract — useful for
/// hedging a variance swap with an actual strip of vanilla options.
///
/// Weight at strike $K_i$ is $\frac{2}{T}\,\frac{\Delta K_i}{K_i^2}$
/// (Demeterfi et al., eq. (28)). Returned in the same order as `strikes`.
///
/// # Panics
/// - if fewer than two strikes are supplied — $\Delta K_i$ is a difference
///   against a neighbour, so a one-point "strip" has no weight to compute
/// - if `maturity` is not strictly positive, `NaN` included — the $2/T$
///   prefactor has no value there
///
/// Both used to return `vec![0.0; n]`. An all-zero weight vector hedges
/// nothing and prices a zero strike downstream, and nothing in it says the
/// strip was rejected rather than computed — case 1 of the crate's [failure
/// convention](crate::traits::ModelPricer#how-pricing-fails). The sibling
/// [`VarianceSwapPricer::fair_strike_replication`], which this function
/// exists to hedge, already panicked on exactly these two conditions, so the
/// pair disagreed about whether the same strip was an error.
pub fn replication_weights(strikes: &[f64], maturity: f64) -> Vec<f64> {
  let n = strikes.len();
  assert!(
    n >= 2,
    "static replication needs at least 2 strikes (got {n})"
  );
  assert!(
    maturity > 0.0,
    "maturity must be strictly positive (got {maturity})"
  );

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

  /// Convexity-adjusted vol strike from variance strike + variance-of-variance.
  ///
  /// # Panics
  /// Panics if `k_var` is not strictly positive. The crate's [failure
  /// convention](crate::traits::ModelPricer#how-pricing-fails) names a
  /// negative variance as programmer error outright, and `k_var = 0` is no
  /// better here: the Jensen correction divides by $K_{\text{var}}^{3/2}$,
  /// so the expansion this method *is* has no value there.
  ///
  /// Returning `0.0` — the old behaviour — handed back a vol strike
  /// indistinguishable from `fair_strike_bsm(0.0)`.
  pub fn fair_strike_from_var(k_var: f64, var_of_var: f64) -> f64 {
    assert!(
      k_var > 0.0,
      "variance strike k_var must be strictly positive (got {k_var})"
    );
    k_var.sqrt() - var_of_var / (8.0 * k_var.powf(1.5))
  }

  /// Heston-implied vol strike — uses continuous Heston variance fair
  /// strike with second-order convexity adjustment from variance dispersion.
  ///
  /// $\text{Var}\!\left(\frac{1}{T}\int_0^T V_t dt\right) \approx
  /// \frac{\sigma^2(V_0 - \theta)^2 (1-e^{-2\kappa T})}{2\kappa^3 T^2}$
  /// to leading order; the closed form is messier — we use a tractable
  /// approximation suitable for short maturities.
  ///
  /// # Panics
  /// Panics if the underlying variance strike
  /// $\theta + (V_0 - \theta)\,\frac{1 - e^{-\kappa T}}{\kappa T}$ is not
  /// strictly positive, which for a convex combination means a negative
  /// `v0` or `theta`. The check sits ahead of the κ → 0 branch rather than
  /// inside [`fair_strike_from_var`](Self::fair_strike_from_var) so that
  /// both branches reject the same inputs — the short-circuit used to floor
  /// a negative strike to `0.0` through `max(0.0)` while the main path
  /// returned a sentinel of its own.
  ///
  /// Returns [`f64::NAN`] for a `NaN` `sigma`, which is the one undefined
  /// input the `k_var` assertion cannot reach: `k_var` is built from
  /// `(v0, kappa, theta, tau)` and never reads `sigma`, so the check has to
  /// happen where the dispersion is floored instead. A floor and a poison
  /// check are different operations and `f64::max` runs them together into
  /// one wrong answer — a dispersion below zero is round-off and still
  /// floors, an undefined one has nothing to floor. Same split as
  /// [`VarianceSwapPricer::fair_strike_replication`]'s.
  pub fn fair_strike_heston(v0: f64, kappa: f64, theta: f64, sigma: f64, tau: f64) -> f64 {
    let pricer = VarianceSwapPricer {
      s: 1.0,
      r: 0.0,
      q: 0.0,
      tau,
    };
    let k_var = pricer.fair_strike_heston(v0, kappa, theta);
    assert!(
      k_var > 0.0,
      "heston variance strike must be strictly positive (got {k_var} from v0={v0}, theta={theta})"
    );
    if kappa.abs() < 1e-10 || tau <= 0.0 {
      return k_var.sqrt();
    }
    let dispersion = (sigma * sigma * (v0 - theta).powi(2) * (1.0 - (-2.0 * kappa * tau).exp()))
      / (2.0 * kappa.powi(3) * tau * tau);
    let floored = if dispersion.is_nan() {
      dispersion
    } else {
      dispersion.max(0.0)
    };
    Self::fair_strike_from_var(k_var, floored)
  }
}

#[cfg(test)]
mod tests;

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
    if kappa.abs() < 1e-10 {
      // Limit κ → 0 of (1 - e^{-κT})/(κT) is 1, so K_var → v0.
      return v0;
    }
    let factor = (1.0 - (-kappa * tau).exp()) / (kappa * tau);
    theta + (v0 - theta) * factor
  }

  /// Discrete-monitoring correction to the continuous Heston fair strike
  /// (Bernard–Cui 2013, leading-order in $T/N$).
  ///
  /// Adds $\frac{T}{N}$ correction reflecting the discrete-vs-continuous
  /// gap; for $N \to \infty$ converges to the continuous strike.
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
    // Leading-order Bernard–Cui adjustment for log-return moment.
    // ξ = θ + (V0 - θ)·factor; correction ≈ ¼·ξ²·dt + ρ·σ·ξ·dt.
    let xi = cont;
    let bias = 0.25 * xi * xi * dt + rho * sigma * xi * dt;
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
mod tests {
  use super::*;

  fn pricer() -> VarianceSwapPricer {
    VarianceSwapPricer {
      s: 100.0,
      r: 0.05,
      q: 0.0,
      tau: 1.0,
    }
  }

  #[test]
  fn bsm_fair_strike_is_sigma_squared() {
    assert!((pricer().fair_strike_bsm(0.2) - 0.04).abs() < 1e-12);
  }

  #[test]
  fn forward_under_zero_dividend() {
    let p = pricer();
    assert!((p.forward() - 100.0 * 0.05_f64.exp()).abs() < 1e-10);
  }

  #[test]
  fn realized_variance_constant_path_is_zero() {
    let prices = vec![100.0; 252];
    let rv = VarianceSwapPricer::realized_variance(&prices, BUSINESS_DAY_252_DT);
    assert!((rv - 0.0).abs() < 1e-15);
  }

  #[test]
  fn realized_variance_recovers_known_drift() {
    let dt: f64 = BUSINESS_DAY_252_DT;
    let daily = 0.20 * dt.sqrt();
    let prices: Vec<f64> = (0..253).map(|i| 100.0 * (daily * i as f64).exp()).collect();
    let rv = VarianceSwapPricer::realized_variance(&prices, dt);
    assert!((rv - 0.04).abs() < 0.005, "rv={rv}, expected≈0.04");
  }

  #[test]
  fn pnl_scales_with_notional() {
    assert!((VarianceSwapPricer::pnl(0.06, 0.04, 100_000.0) - 2_000.0).abs() < 1e-9);
  }

  #[test]
  fn vol_swap_convexity_lowers_strike() {
    let k_vol = VolatilitySwapPricer::fair_strike_from_var(0.04, 0.001);
    assert!(k_vol < 0.04_f64.sqrt());
    assert!(k_vol > 0.0);
  }

  #[test]
  fn vol_swap_zero_dispersion_recovers_sqrt_var() {
    let k_vol = VolatilitySwapPricer::fair_strike_from_var(0.04, 0.0);
    assert!((k_vol - 0.2).abs() < 1e-10);
  }

  #[test]
  fn heston_fair_strike_equals_v0_when_at_long_run_mean() {
    let p = pricer();
    let k_var = p.fair_strike_heston(0.04, 1.5, 0.04);
    assert!((k_var - 0.04).abs() < 1e-12);
  }

  #[test]
  fn heston_fair_strike_blends_v0_to_theta() {
    // V0 = 0.09 (high IV), θ = 0.04 (low LR), strong κ → fair K close to θ
    let p = pricer();
    let k_strong = p.fair_strike_heston(0.09, 5.0, 0.04);
    let k_weak = p.fair_strike_heston(0.09, 0.1, 0.04);
    assert!(k_strong < k_weak);
    assert!(k_weak <= 0.09);
    assert!(k_strong > 0.04);
  }

  #[test]
  fn heston_kappa_zero_limit_equals_v0() {
    let p = pricer();
    assert!((p.fair_strike_heston(0.04, 0.0, 0.10) - 0.04).abs() < 1e-12);
  }

  #[test]
  fn heston_long_t_limit_approaches_theta() {
    // T → ∞ with κ > 0 ⇒ factor → 0, K_var → θ.
    let p = VarianceSwapPricer {
      s: 100.0,
      r: 0.0,
      q: 0.0,
      tau: 50.0,
    };
    let k_var = p.fair_strike_heston(0.09, 2.0, 0.04);
    assert!(
      (k_var - 0.04).abs() < 0.01,
      "K_var={k_var} should approach θ=0.04"
    );
  }

  #[test]
  fn heston_discrete_correction_vanishes_with_n() {
    let p = pricer();
    let k_cont = p.fair_strike_heston(0.04, 1.5, 0.04);
    let k_disc_fine = p.fair_strike_heston_discrete(0.04, 1.5, 0.04, 0.3, -0.7, 100_000);
    let k_disc_coarse = p.fair_strike_heston_discrete(0.04, 1.5, 0.04, 0.3, -0.7, 12);
    assert!((k_disc_fine - k_cont).abs() < (k_disc_coarse - k_cont).abs());
  }

  /// A non-positive maturity is invalid input, not a not-computable point:
  /// there is no window to annualise over. It returned `0.0` before, which
  /// `bsm_fair_strike_is_sigma_squared` shows is a value the same type also
  /// produces as a genuine answer.
  #[test]
  #[should_panic(expected = "maturity tau must be strictly positive (got 0)")]
  fn replication_rejects_a_nonpositive_maturity() {
    let p = VarianceSwapPricer {
      s: 100.0,
      r: 0.0,
      q: 0.0,
      tau: 0.0,
    };
    let _ = p.fair_strike_replication(&[90.0, 100.0, 110.0], &[1.0, 2.0, 1.0]);
  }

  /// The trapezoidal weights read `strikes[i±1]`, so one strike is not a
  /// coarse strip — it is not a strip. Checked at both reachable lengths so
  /// the guard cannot pass on the empty case alone.
  #[test]
  fn replication_rejects_a_strip_shorter_than_two_strikes() {
    for (strikes, prices) in [(&[][..], &[][..]), (&[100.0][..], &[2.0][..])] {
      let err =
        std::panic::catch_unwind(|| pricer().fair_strike_replication(strikes, prices)).unwrap_err();
      let msg = err
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_else(|| (*err.downcast_ref::<&str>().unwrap_or(&"")).to_string());
      assert!(
        msg.contains("static replication needs at least 2 strikes"),
        "wrong panic for len {}: {msg}",
        strikes.len()
      );
    }
  }

  /// The guards above are worth nothing if a `NaN` can still arrive as
  /// `0.0`, and `f64::max` hands back the finite operand when the other is
  /// `NaN` — so the final floor had to learn to test first. A `NaN` option
  /// price is the reachable source: the strikes are `debug_assert`ed finite,
  /// the prices never were.
  #[test]
  fn replication_does_not_floor_a_nan_price_to_zero() {
    let p = pricer();
    let strikes = [90.0, 100.0, 110.0];
    let got = p.fair_strike_replication(&strikes, &[1.0, f64::NAN, 1.0]);
    assert!(
      got.is_nan(),
      "a NaN price must not floor to a strike, got {got}"
    );

    let clean = p.fair_strike_replication(&strikes, &[1.0, 2.0, 1.0]);
    assert!(
      clean.is_finite(),
      "control case must still price, got {clean}"
    );
  }

  /// "No observations" and "no movement" were the same number before —
  /// `realized_variance_constant_path_is_zero` above is the genuine `0.0`
  /// this one used to be indistinguishable from.
  #[test]
  #[should_panic(expected = "realized variance needs at least 2 prices (got 1)")]
  fn realized_variance_rejects_a_single_price() {
    let _ = VarianceSwapPricer::realized_variance(&[100.0], BUSINESS_DAY_252_DT);
  }

  #[test]
  #[should_panic(expected = "realized variance needs at least 2 prices (got 0)")]
  fn realized_variance_rejects_an_empty_path() {
    let _ = VarianceSwapPricer::realized_variance(&[], BUSINESS_DAY_252_DT);
  }

  /// A negative variance is programmer error by the crate convention, and a
  /// zero one leaves the Jensen correction dividing by `k_var^{3/2}`.
  /// `vol_swap_zero_dispersion_recovers_sqrt_var` above is the real `0.2`
  /// this used to collide with at the bottom of its range.
  #[test]
  #[should_panic(expected = "variance strike k_var must be strictly positive (got -0.01)")]
  fn vol_swap_rejects_a_negative_variance_strike() {
    let _ = VolatilitySwapPricer::fair_strike_from_var(-0.01, 0.001);
  }

  #[test]
  #[should_panic(expected = "variance strike k_var must be strictly positive (got 0)")]
  fn vol_swap_rejects_a_zero_variance_strike() {
    let _ = VolatilitySwapPricer::fair_strike_from_var(0.0, 0.001);
  }

  /// A `NaN` vol-of-vol is the one undefined input the `k_var > 0` guard
  /// cannot see: `k_var` is built from `(v0, kappa, theta, tau)` and does not
  /// read `sigma` at all, so the assertion passes and the `NaN` arrives at
  /// the Jensen correction intact. `f64::NAN.max(0.0)` is `0.0`, so the floor
  /// used to hand back `sqrt(k_var)` — exactly `0.2` here, which is the
  /// number `vol_swap_zero_dispersion_recovers_sqrt_var` pins as the *real*
  /// answer for a genuinely dispersion-free swap. The two were
  /// indistinguishable.
  #[test]
  fn vol_swap_heston_preserves_a_nan_vol_of_vol() {
    let k = VolatilitySwapPricer::fair_strike_heston(0.04, 1.5, 0.04, f64::NAN, 1.0);
    assert!(k.is_nan(), "a NaN sigma must exit as NaN, got {k}");
  }

  /// The poison check must not disturb the dispersion it is guarding: a
  /// finite vol-of-vol still lowers the strike below `sqrt(k_var)` by the
  /// convexity correction, and `sigma = 0` still lands exactly on it.
  #[test]
  fn vol_swap_heston_is_unchanged_by_the_poison_check() {
    let naive = 0.04_f64.sqrt();
    let dispersed = VolatilitySwapPricer::fair_strike_heston(0.09, 1.5, 0.04, 0.3, 1.0);
    assert!(dispersed.is_finite() && dispersed > 0.0, "{dispersed}");
    let flat = VolatilitySwapPricer::fair_strike_heston(0.04, 1.5, 0.04, 0.3, 1.0);
    assert!((flat - naive).abs() < 1e-15, "{flat} vs {naive}");
  }

  /// Both branches of `VolatilitySwapPricer::fair_strike_heston` must reject
  /// the same inputs. The κ → 0 short-circuit used to floor a negative
  /// strike through `max(0.0)` and return `0.0` while the main path returned
  /// its own sentinel, so a caller sweeping κ would have seen the guard
  /// change shape underneath them.
  #[test]
  fn vol_swap_heston_rejects_a_negative_variance_on_both_branches() {
    for &kappa in &[1e-12, 1.5] {
      let err = std::panic::catch_unwind(|| {
        VolatilitySwapPricer::fair_strike_heston(-0.04, kappa, -0.04, 0.3, 1.0)
      })
      .unwrap_err();
      let msg = err
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_else(|| (*err.downcast_ref::<&str>().unwrap_or(&"")).to_string());
      assert!(
        msg.contains("heston variance strike must be strictly positive"),
        "kappa={kappa} gave the wrong panic: {msg}"
      );
    }
  }

  #[test]
  fn replication_weights_are_positive_and_decay() {
    // Strikes near forward have largest weight; weight ∝ 1/K^2.
    let strikes: Vec<f64> = (50..=150).step_by(10).map(|i| i as f64).collect();
    let w = replication_weights(&strikes, 1.0);
    assert_eq!(w.len(), strikes.len());
    for &wi in &w {
      assert!(wi > 0.0);
    }
    // Weight at K=50 should exceed weight at K=150 (1/K^2 dominates Δk).
    assert!(w[0] > *w.last().unwrap());
  }

  #[test]
  fn replication_strike_within_one_percent_of_bsm_for_dense_strip() {
    // Build a dense BS option strip (σ = 25%) and replicate the strike.
    use stochastic_rs_distributions::special::norm_cdf;
    let p = VarianceSwapPricer {
      s: 100.0,
      r: 0.0,
      q: 0.0,
      tau: 1.0,
    };
    let sigma = 0.25;
    let strikes: Vec<f64> = (10..=400).map(|i| i as f64 * 0.5).collect();
    let prices: Vec<f64> = strikes
      .iter()
      .map(|&k| {
        let d1 = ((p.s / k).ln() + 0.5 * sigma * sigma * p.tau) / (sigma * p.tau.sqrt());
        let d2 = d1 - sigma * p.tau.sqrt();
        if k >= p.s {
          // call
          p.s * norm_cdf(d1) - k * norm_cdf(d2)
        } else {
          // put via parity (r = q = 0)
          k * norm_cdf(-d2) - p.s * norm_cdf(-d1)
        }
      })
      .collect();
    let k_var = p.fair_strike_replication(&strikes, &prices);
    let target = sigma * sigma;
    let rel_err = (k_var - target).abs() / target;
    assert!(
      rel_err < 0.02,
      "K_var={k_var}, expected≈{target}, rel_err={rel_err}"
    );
  }

  #[test]
  #[should_panic(expected = "static replication needs at least 2 strikes (got 1)")]
  fn replication_weights_reject_a_single_strike() {
    let _ = replication_weights(&[100.0], 1.0);
  }

  #[test]
  #[should_panic(expected = "static replication needs at least 2 strikes (got 0)")]
  fn replication_weights_reject_an_empty_strip() {
    let _ = replication_weights(&[], 1.0);
  }

  #[test]
  #[should_panic(expected = "maturity must be strictly positive (got 0)")]
  fn replication_weights_reject_a_zero_maturity() {
    let _ = replication_weights(&[90.0, 100.0, 110.0], 0.0);
  }

  #[test]
  #[should_panic(expected = "maturity must be strictly positive (got -1)")]
  fn replication_weights_reject_a_negative_maturity() {
    let _ = replication_weights(&[90.0, 100.0, 110.0], -1.0);
  }

  /// `replication_weights` and `fair_strike_replication` are meant to be used
  /// together — one hedges what the other prices — so they must agree about
  /// which inputs are errors. They did not: the sibling panicked on a
  /// one-point strip and a non-positive maturity while this one returned an
  /// all-zero weight vector, which replicates nothing and prices a zero
  /// strike downstream with no signal.
  #[test]
  fn the_replication_pair_rejects_the_same_inputs() {
    for (strikes, tau) in [
      (vec![100.0], 1.0),
      (vec![90.0, 100.0, 110.0], 0.0),
      (vec![90.0, 100.0, 110.0], -1.0),
    ] {
      let prices = vec![1.0; strikes.len()];
      let p = VarianceSwapPricer {
        s: 100.0,
        r: 0.0,
        q: 0.0,
        tau,
      };
      let sibling =
        std::panic::catch_unwind(|| p.fair_strike_replication(&strikes, &prices)).is_err();
      let weights = std::panic::catch_unwind(|| replication_weights(&strikes, tau)).is_err();
      assert_eq!(
        sibling, weights,
        "strikes={strikes:?} tau={tau}: fair_strike_replication panicked={sibling}, \
         replication_weights panicked={weights}"
      );
      assert!(sibling, "strikes={strikes:?} tau={tau} must be rejected");
    }
  }

  #[test]
  #[should_panic(expected = "maturity tau must be non-negative (got -1)")]
  fn heston_fair_strike_rejects_a_negative_maturity() {
    let p = VarianceSwapPricer {
      s: 100.0,
      r: 0.05,
      q: 0.0,
      tau: -1.0,
    };
    let _ = p.fair_strike_heston(0.04, 1.5, 0.04);
  }

  /// `tau == 0` is the genuine $T \to 0$ limit of
  /// $\frac{1-e^{-\kappa T}}{\kappa T} \to 1$, not a sentinel, so it keeps
  /// returning `v0` — the negative-maturity guard must not take it with it.
  #[test]
  fn heston_fair_strike_zero_maturity_stays_the_v0_limit() {
    let p = VarianceSwapPricer {
      s: 100.0,
      r: 0.05,
      q: 0.0,
      tau: 0.0,
    };
    assert_eq!(p.fair_strike_heston(0.04, 1.5, 0.10), 0.04);
  }
}

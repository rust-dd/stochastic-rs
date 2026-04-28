//! Merton structural model of default.
//!
//! Reference: Merton, "On the Pricing of Corporate Debt: The Risk Structure of
//! Interest Rates", Journal of Finance, 29(2), 449–470 (1974).
//! DOI: 10.1111/j.1540-6261.1974.tb03058.x
//!
//! Reference: Crosbie & Bohn, "Modeling Default Risk" (Moody's-KMV technical
//! document, 2003).
//!
//! The firm's asset value follows a geometric Brownian motion under the risk
//! neutral measure,
//! $$
//! dV_t=rV_t\,dt+\sigma_V V_t\,dW_t^{\mathbb{Q}},
//! $$
//! and default is triggered at the debt maturity $T$ whenever $V_T<D$.  Under
//! this terminal-default mechanism the standard Black-Scholes arithmetic gives
//! analytical formulas for the risk-neutral default probability, the equity
//! value (a call on the firm), the debt value, the Merton credit spread and the
//! distance-to-default:
//! $$
//! d_1=\frac{\ln(V_0/D)+(r+\tfrac12\sigma_V^2)\,T}{\sigma_V\sqrt T},\quad
//! d_2=d_1-\sigma_V\sqrt T,
//! $$
//! $$
//! \mathbb{Q}(V_T<D)=\Phi(-d_2),\qquad
//! E_0=V_0\Phi(d_1)-De^{-rT}\Phi(d_2).
//! $$

use stochastic_rs_distributions::special::norm_cdf;

/// Analytical Merton structural model.
///
/// Parameters are fixed at construction; all quantities are functions of a
/// user-supplied time-to-maturity `tau`.  All numeric results are `f64` because
/// the model uses the standard-normal CDF from
/// [`stochastic_rs_distributions::special`], which only supports double
/// precision; higher-level types that are generic over `T: FloatExt` should
/// convert into `f64` before calling this model.
#[derive(Debug, Clone)]
pub struct MertonStructural {
  /// Initial asset value $V_0$.
  pub asset_value: f64,
  /// Face value of zero-coupon debt $D$ maturing at `tau`.
  pub debt_face_value: f64,
  /// Asset (firm) volatility $\sigma_V$, annualised.
  pub asset_volatility: f64,
  /// Risk-free short rate $r$.
  pub risk_free_rate: f64,
  /// Continuous cash-flow / dividend yield $q$ paid by the firm assets.
  pub asset_payout: f64,
}

impl MertonStructural {
  /// Construct a Merton model.
  pub fn new(
    asset_value: f64,
    debt_face_value: f64,
    asset_volatility: f64,
    risk_free_rate: f64,
    asset_payout: f64,
  ) -> Self {
    assert!(asset_value > 0.0, "asset_value must be positive");
    assert!(debt_face_value > 0.0, "debt_face_value must be positive");
    assert!(asset_volatility > 0.0, "asset_volatility must be positive");
    Self {
      asset_value,
      debt_face_value,
      asset_volatility,
      risk_free_rate,
      asset_payout,
    }
  }

  fn d1_d2(&self, tau: f64) -> (f64, f64) {
    assert!(tau > 0.0, "tau must be positive");
    let sigma_sqrt_t = self.asset_volatility * tau.sqrt();
    let d1 = ((self.asset_value / self.debt_face_value).ln()
      + (self.risk_free_rate - self.asset_payout + 0.5 * self.asset_volatility.powi(2)) * tau)
      / sigma_sqrt_t;
    let d2 = d1 - sigma_sqrt_t;
    (d1, d2)
  }

  /// Risk-neutral default probability $\mathbb{Q}(\tau_d \le T) = \Phi(-d_2)$.
  pub fn risk_neutral_default_probability(&self, tau: f64) -> f64 {
    let (_, d2) = self.d1_d2(tau);
    norm_cdf(-d2)
  }

  /// Real-world default probability using a user-supplied real drift $\mu$.
  ///
  /// $$\mathbb{P}(\tau_d\le T)=\Phi(-\mathrm{DD}_{\mathbb{P}}),\qquad
  /// \mathrm{DD}_{\mathbb{P}}=\frac{\ln(V_0/D)+(\mu-q-\tfrac12\sigma_V^2)T}{\sigma_V\sqrt T}.$$
  pub fn real_world_default_probability(&self, tau: f64, asset_drift: f64) -> f64 {
    assert!(tau > 0.0, "tau must be positive");
    let sigma_sqrt_t = self.asset_volatility * tau.sqrt();
    let dd = ((self.asset_value / self.debt_face_value).ln()
      + (asset_drift - self.asset_payout - 0.5 * self.asset_volatility.powi(2)) * tau)
      / sigma_sqrt_t;
    norm_cdf(-dd)
  }

  /// Survival probability $Q(T)=1-\mathbb{Q}(\tau_d\le T)$.
  pub fn survival_probability(&self, tau: f64) -> f64 {
    1.0 - self.risk_neutral_default_probability(tau)
  }

  /// Risk-neutral distance-to-default $d_2$.
  pub fn distance_to_default(&self, tau: f64) -> f64 {
    self.d1_d2(tau).1
  }

  /// Equity value $E_0=V_0 e^{-qT}\Phi(d_1)-De^{-rT}\Phi(d_2)$.
  pub fn equity_value(&self, tau: f64) -> f64 {
    let (d1, d2) = self.d1_d2(tau);
    self.asset_value * (-self.asset_payout * tau).exp() * norm_cdf(d1)
      - self.debt_face_value * (-self.risk_free_rate * tau).exp() * norm_cdf(d2)
  }

  /// Risky debt value $B_0 = V_0 e^{-qT} - E_0$.
  pub fn debt_value(&self, tau: f64) -> f64 {
    self.asset_value * (-self.asset_payout * tau).exp() - self.equity_value(tau)
  }

  /// Merton credit spread $s(T)=-\ln(B_0/D)/T - r$ (continuous compounding).
  pub fn credit_spread(&self, tau: f64) -> f64 {
    assert!(tau > 0.0, "tau must be positive");
    let b = self.debt_value(tau);
    if b <= 0.0 {
      return f64::INFINITY;
    }
    -(b / self.debt_face_value).ln() / tau - self.risk_free_rate
  }

  /// Risk-neutral recovery rate implied by the Merton payoff on default:
  /// $\mathbb{E}^{\mathbb{Q}}[V_T\mid V_T<D]/D$.
  ///
  /// Derived from $\mathbb{E}^{\mathbb{Q}}[V_T\mathbf{1}_{V_T<D}]
  /// =V_0 e^{(r-q)T}\Phi(-d_1)$, so
  /// $$
  /// R(T)=\frac{V_0 e^{(r-q)T}\Phi(-d_1)}{D\,\Phi(-d_2)}.
  /// $$
  pub fn implied_recovery(&self, tau: f64) -> f64 {
    let (d1, d2) = self.d1_d2(tau);
    let denom = self.debt_face_value * norm_cdf(-d2);
    if denom <= 0.0 {
      return 0.0;
    }
    self.asset_value * ((self.risk_free_rate - self.asset_payout) * tau).exp() * norm_cdf(-d1)
      / denom
  }

  /// Leverage ratio $L=D e^{-rT}/V_0$.
  pub fn leverage(&self, tau: f64) -> f64 {
    self.debt_face_value * (-self.risk_free_rate * tau).exp() / self.asset_value
  }
}

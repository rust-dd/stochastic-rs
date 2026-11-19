//! Merton's structural model: the firm defaults at the debt's maturity `T` when its assets, a
//! geometric Brownian motion paying out `q`, end below the face value `D`.
//!
//! $$
//! d_{1,2}=\frac{\ln(V_0/D)+(r-q\pm\tfrac12\sigma_V^2)\,T}{\sigma_V\sqrt T},\qquad
//! \mathbb{Q}(V_T<D)=\Phi(-d_2),\qquad
//! E_0=V_0e^{-qT}\Phi(d_1)-De^{-rT}\Phi(d_2)
//! $$
//!
//! Reference: Merton (1974), J. Finance 29(2), 449–470, DOI: 10.1111/j.1540-6261.1974.tb03058.x
//! Reference: Crosbie & Bohn (2003), "Modeling Default Risk", Moody's KMV

use stochastic_rs_distributions::special::ln_norm_cdf;
use stochastic_rs_distributions::special::norm_cdf;

use crate::traits::RealExt;

/// Analytical Merton structural model in `f64`. A query at `tau = 0`, the debt's maturity, is
/// the payoff, or NaN where none exists; one at a negative or NaN `tau` is NaN.
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

/// Where a query stands against the debt's maturity.
enum Horizon {
  Live,
  /// `tau = 0`, with whether the firm is in default, $V_0 < D$.
  Expiry {
    defaulted: bool,
  },
  /// A negative or NaN `tau`, or a NaN asset or face value at `tau = 0`.
  Undefined,
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

  fn horizon(&self, tau: f64) -> Horizon {
    if tau > 0.0 {
      Horizon::Live
    } else if tau == 0.0 && !(self.asset_value - self.debt_face_value).is_nan() {
      Horizon::Expiry {
        defaulted: self.asset_value < self.debt_face_value,
      }
    } else {
      Horizon::Undefined
    }
  }

  /// `(d₁, d₂)` for `tau > 0`.
  fn d1_d2(&self, tau: f64) -> (f64, f64) {
    let sigma_sqrt_t = self.asset_volatility * tau.sqrt();
    let d1 = ((self.asset_value / self.debt_face_value).ln()
      + (self.risk_free_rate - self.asset_payout + 0.5 * self.asset_volatility.powi(2)) * tau)
      / sigma_sqrt_t;
    let d2 = d1 - sigma_sqrt_t;
    (d1, d2)
  }

  /// Risk-neutral default probability $\mathbb{Q}(\tau_d \le T) = \Phi(-d_2)$; at `tau = 0`,
  /// one if $V_0 < D$ and zero otherwise.
  pub fn risk_neutral_default_probability(&self, tau: f64) -> f64 {
    match self.horizon(tau) {
      Horizon::Live => norm_cdf(-self.d1_d2(tau).1),
      Horizon::Expiry { defaulted } => f64::from(u8::from(defaulted)),
      Horizon::Undefined => f64::NAN,
    }
  }

  /// Real-world default probability under the asset drift $\mu$; the risk-neutral one's value at
  /// `tau = 0`.
  ///
  /// $$\mathbb{P}(\tau_d\le T)=\Phi(-\mathrm{DD}_{\mathbb{P}}),\qquad
  /// \mathrm{DD}_{\mathbb{P}}=\frac{\ln(V_0/D)+(\mu-q-\tfrac12\sigma_V^2)T}{\sigma_V\sqrt T}.$$
  pub fn real_world_default_probability(&self, tau: f64, asset_drift: f64) -> f64 {
    match self.horizon(tau) {
      Horizon::Live => {
        let sigma_sqrt_t = self.asset_volatility * tau.sqrt();
        let dd = ((self.asset_value / self.debt_face_value).ln()
          + (asset_drift - self.asset_payout - 0.5 * self.asset_volatility.powi(2)) * tau)
          / sigma_sqrt_t;
        norm_cdf(-dd)
      }
      Horizon::Expiry { defaulted } => f64::from(u8::from(defaulted)),
      Horizon::Undefined => f64::NAN,
    }
  }

  /// Survival probability $Q(T)=1-\mathbb{Q}(\tau_d\le T)$.
  pub fn survival_probability(&self, tau: f64) -> f64 {
    1.0 - self.risk_neutral_default_probability(tau)
  }

  /// Risk-neutral distance-to-default $d_2$; NaN at `tau = 0`, where it has no value.
  pub fn distance_to_default(&self, tau: f64) -> f64 {
    match self.horizon(tau) {
      Horizon::Live => self.d1_d2(tau).1,
      _ => f64::NAN,
    }
  }

  /// Equity value $E_0=V_0 e^{-qT}\Phi(d_1)-De^{-rT}\Phi(d_2)$; $(V_0-D)^+$ at `tau = 0`.
  pub fn equity_value(&self, tau: f64) -> f64 {
    match self.horizon(tau) {
      Horizon::Live => {
        let (d1, d2) = self.d1_d2(tau);
        self.asset_value * (-self.asset_payout * tau).exp() * norm_cdf(d1)
          - self.debt_face_value * (-self.risk_free_rate * tau).exp() * norm_cdf(d2)
      }
      Horizon::Expiry { .. } => (self.asset_value - self.debt_face_value).max_or_nan(0.0),
      Horizon::Undefined => f64::NAN,
    }
  }

  /// Risky debt value $B_0=V_0e^{-qT}\Phi(-d_1)+De^{-rT}\Phi(d_2)$, the assets less the equity
  /// without their cancellation; $\min(V_0, D)$ at `tau = 0`.
  pub fn debt_value(&self, tau: f64) -> f64 {
    match self.horizon(tau) {
      Horizon::Live => {
        let (d1, d2) = self.d1_d2(tau);
        self.asset_value * (-self.asset_payout * tau).exp() * norm_cdf(-d1)
          + self.debt_face_value * (-self.risk_free_rate * tau).exp() * norm_cdf(d2)
      }
      Horizon::Expiry { .. } => self.asset_value.min_or_nan(self.debt_face_value),
      Horizon::Undefined => f64::NAN,
    }
  }

  /// Merton credit spread $s(T)=-\ln(B_0/D)/T - r$ (continuous compounding); NaN at `tau = 0`.
  pub fn credit_spread(&self, tau: f64) -> f64 {
    match self.horizon(tau) {
      Horizon::Live => {
        -(self.debt_value(tau) / self.debt_face_value).ln() / tau - self.risk_free_rate
      }
      _ => f64::NAN,
    }
  }

  /// Recovery rate $\mathbb{E}^{\mathbb{Q}}[V_T\mid V_T<D]/D$, the tails' ratio taken in
  /// logarithms; at `tau = 0`, $V_0/D$ in default and NaN out of it.
  ///
  /// $$
  /// R(T)=\frac{V_0 e^{(r-q)T}\Phi(-d_1)}{D\,\Phi(-d_2)}.
  /// $$
  pub fn implied_recovery(&self, tau: f64) -> f64 {
    let cover = self.asset_value / self.debt_face_value;
    match self.horizon(tau) {
      Horizon::Live => {
        let (d1, d2) = self.d1_d2(tau);
        let carry = (self.risk_free_rate - self.asset_payout) * tau;
        cover * (carry + ln_norm_cdf(-d1) - ln_norm_cdf(-d2)).exp()
      }
      Horizon::Expiry { defaulted: true } => cover,
      _ => f64::NAN,
    }
  }

  /// Leverage ratio $L=D e^{-rT}/V_0$.
  pub fn leverage(&self, tau: f64) -> f64 {
    match self.horizon(tau) {
      Horizon::Undefined => f64::NAN,
      _ => self.debt_face_value * (-self.risk_free_rate * tau).exp() / self.asset_value,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!(
      (got - want).abs() <= rel * want.abs(),
      "{what}: got {got:e}, want {want:e}"
    );
  }

  /// mpmath at 50 digits; `Φ(−d₂)` underflows from `d₂ ≈ 38` on. The last column is the
  /// tolerance: `d₂` itself carries `1e-16 · d₂²` into the logarithms.
  #[test]
  fn implied_recovery_holds_where_the_default_probability_underflows() {
    let rows = [
      (100.0, 80.0, 0.25, 0.03, 1.0, 0.87744960459366647, 1e-14),
      (50.0, 80.0, 0.3, 0.05, 1.0, 0.62551638459102471, 1e-14),
      (100.0, 10.0, 0.1, 0.03, 1.0, 0.99573763004076692, 1e-13),
      (100.0, 1.0, 0.1, 0.03, 1.0, 0.99784690598165948, 1e-12),
      (100.0, 0.01, 0.05, 0.03, 1.0, 0.99972949962690805, 1e-11),
      (100.0, 50.0, 0.01, 0.03, 0.1, 0.9999856359175033, 1e-11),
      (100.0, 1e-20, 0.2, 0.04, 1.0, 0.99921133236491839, 1e-11),
    ];
    for (v, d, sigma, r, tau, want, rel) in rows {
      let got = MertonStructural::new(v, d, sigma, r, 0.0).implied_recovery(tau);
      close(got, want, rel, &format!("V = {v}, D = {d}"));
    }
  }

  /// mpmath at 50 digits, with a payout in the second row.
  #[test]
  fn the_live_queries_match_the_closed_forms() {
    let rows = [
      (
        100.0,
        0.0,
        1.0,
        0.22348430668853513,
        26.462085709671792,
        73.537914290328208,
        0.034225520770423176,
      ),
      (
        100.0,
        0.02,
        2.0,
        0.324466956570756,
        28.966124415136839,
        67.112819500095482,
        0.037825779136018597,
      ),
      (
        50.0,
        0.0,
        1.0,
        0.93943069379150957,
        0.67315370147359232,
        49.326846298526408,
        0.43355815217275251,
      ),
    ];
    for (v, q, tau, pd, equity, debt, spread) in rows {
      let m = MertonStructural::new(v, 80.0, 0.3, 0.05, q);
      close(m.risk_neutral_default_probability(tau), pd, 1e-14, "pd");
      close(m.survival_probability(tau), 1.0 - pd, 1e-14, "survival");
      close(m.equity_value(tau), equity, 1e-13, "equity");
      close(m.debt_value(tau), debt, 1e-14, "debt");
      close(m.credit_spread(tau), spread, 1e-13, "spread");
    }
  }

  /// A firm that cannot default: the debt is `D e^{-rT}` and the spread zero, where `V − E` would
  /// cancel to zero.
  #[test]
  fn the_debt_of_a_riskless_firm_is_the_discounted_face_value() {
    let m = MertonStructural::new(100.0, 1e-20, 0.2, 0.04, 0.0);
    close(m.debt_value(1.0), 9.6078943915232321e-21, 1e-15, "debt");
    assert!(
      m.credit_spread(1.0).abs() < 1e-15,
      "{}",
      m.credit_spread(1.0)
    );
  }

  #[test]
  fn at_expiry_the_queries_are_the_payoff() {
    let solvent = MertonStructural::new(100.0, 80.0, 0.3, 0.05, 0.02);
    assert_eq!(solvent.equity_value(0.0), 20.0);
    assert_eq!(solvent.debt_value(0.0), 80.0);
    assert_eq!(solvent.risk_neutral_default_probability(0.0), 0.0);
    assert_eq!(solvent.real_world_default_probability(0.0, 0.08), 0.0);
    assert_eq!(solvent.survival_probability(0.0), 1.0);
    assert_eq!(solvent.leverage(0.0), 0.8);
    assert!(solvent.implied_recovery(0.0).is_nan());
    assert!(solvent.distance_to_default(0.0).is_nan());
    assert!(solvent.credit_spread(0.0).is_nan());

    let defaulted = MertonStructural::new(50.0, 80.0, 0.3, 0.05, 0.02);
    assert_eq!(defaulted.equity_value(0.0), 0.0);
    assert_eq!(defaulted.debt_value(0.0), 50.0);
    assert_eq!(defaulted.risk_neutral_default_probability(0.0), 1.0);
    assert_eq!(defaulted.real_world_default_probability(0.0, 0.08), 1.0);
    assert_eq!(defaulted.survival_probability(0.0), 0.0);
    assert_eq!(defaulted.implied_recovery(0.0), 0.625);

    let at_par = MertonStructural::new(80.0, 80.0, 0.3, 0.05, 0.0);
    assert_eq!(at_par.risk_neutral_default_probability(0.0), 0.0);
    assert!(at_par.implied_recovery(0.0).is_nan());
  }

  #[test]
  fn a_negative_or_nan_tau_is_nan_in_every_query() {
    let m = MertonStructural::new(100.0, 80.0, 0.3, 0.05, 0.0);
    for tau in [-1.0, f64::NAN] {
      let answers = [
        m.risk_neutral_default_probability(tau),
        m.real_world_default_probability(tau, 0.08),
        m.survival_probability(tau),
        m.distance_to_default(tau),
        m.equity_value(tau),
        m.debt_value(tau),
        m.credit_spread(tau),
        m.implied_recovery(tau),
        m.leverage(tau),
      ];
      assert!(
        answers.iter().all(|x| x.is_nan()),
        "tau = {tau}: {answers:?}"
      );
    }
  }

  #[test]
  fn a_nan_asset_value_is_nan_at_expiry_too() {
    let mut m = MertonStructural::new(100.0, 80.0, 0.3, 0.05, 0.0);
    m.asset_value = f64::NAN;
    let answers = [
      m.risk_neutral_default_probability(0.0),
      m.equity_value(0.0),
      m.debt_value(0.0),
      m.implied_recovery(0.0),
      m.leverage(0.0),
    ];
    assert!(answers.iter().all(|x| x.is_nan()), "{answers:?}");
  }
}

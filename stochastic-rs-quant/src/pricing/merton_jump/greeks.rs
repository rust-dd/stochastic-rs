//! Merton (1976) Greeks: Poisson sums of Black–Scholes Greeks, through `σ_n = σ√(1 − γ + γn/(λτ))`.
//! `gamma` and the Greeks in `σ` take no `option_type`: parity's spread is `σ`-free, linear in `S`.

use stochastic_rs_distributions::special::norm_pdf;

use super::Merton1976Pricer;
use super::TermRegime;
use crate::OptionType;
use crate::pricing::bsm::BsmPricer;
use crate::traits::Greeks;

impl Merton1976Pricer {
  const H_TAU: f64 = 1e-5;

  fn h_s(s: f64) -> f64 {
    s.abs() * 1e-4
  }

  fn volatility_series(&self, tau: f64, f: impl Fn(&BsmPricer, f64) -> f64) -> f64 {
    let unit_volatility = Self {
      sigma: 1.0,
      ..*self
    };
    (0..self.m)
      .map(|n| {
        let weight = self.poisson_weight(n, tau);
        if weight == 0.0 {
          return 0.0;
        }
        weight * f(&self.term_bsm(n, tau), unit_volatility.term_vol(n, tau))
      })
      .sum()
  }

  /// Poisson-weighted spot and rate Greeks. Each caller supplies its own
  /// zero-volatility limit; the no-jump case uses the base Black-Scholes model.
  fn greek_series(&self, tau: f64, greek: impl Fn(&BsmPricer) -> f64) -> f64 {
    if self.lambda <= 0.0 {
      return greek(&self.base_bsm());
    }
    (0..self.m)
      .map(|n| self.poisson_weight(n, tau) * greek(&self.term_bsm(n, tau)))
      .sum()
  }

  /// Conditional prices used by the maturity differences in theta and charm, at the price's own
  /// zero-volatility limits.
  pub(super) fn series_price(
    &self,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
    option_type: OptionType,
  ) -> f64 {
    self.greek_series(tau, |bsm| {
      let term = Merton1976Pricer::term_call_put(bsm, s, k, r, q, tau);
      match option_type {
        OptionType::Call => term.call,
        OptionType::Put => term.put,
      }
    })
  }

  /// Delta `∂V/∂S`; a `σ_n = 0` term at its forward adds the limit `±½e^{(b−r)τ}`, written in `r`
  /// and `τ` so a non-finite rate propagates.
  pub fn delta(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    if tau == 0.0 {
      return f64::NAN;
    }
    self.greek_series(tau, |bsm| {
      match Merton1976Pricer::term_regime(bsm, s, k, r, q, tau) {
        TermRegime::AtTheForward => {
          let half_carry = 0.5 * ((bsm.b(r, q) - r) * tau).exp();
          match option_type {
            OptionType::Call => half_carry,
            OptionType::Put => -half_carry,
          }
        }
        TermRegime::Saturated | TermRegime::Ordinary => bsm.delta(s, k, r, q, tau, option_type),
      }
    })
  }

  /// Gamma $\partial^2 V/\partial S^2$: a degenerate term contributes `+∞` at its forward, where
  /// `Γ` diverges like `1/σ`, and `0` away from it.
  pub fn gamma(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    if tau == 0.0 {
      return f64::NAN;
    }
    self.greek_series(tau, |bsm| {
      match Merton1976Pricer::term_regime(bsm, s, k, r, q, tau) {
        TermRegime::AtTheForward => {
          ((bsm.b(r, q) - r) * tau).exp() * norm_pdf(0.0) / (s * bsm.sigma * tau.sqrt())
        }
        TermRegime::Saturated => ((bsm.b(r, q) - r) * tau).exp() * 0.0,
        TermRegime::Ordinary => bsm.gamma(s, k, r, q, tau),
      }
    })
  }

  /// Rho `∂V/∂r`; a `σ_n = 0` term at its forward adds `±½Kτe^{−rτ}`, the one limit on the
  /// discount factor rather than the carry factor.
  pub fn rho(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    if tau == 0.0 {
      return f64::NAN;
    }
    self.greek_series(tau, |bsm| {
      match Merton1976Pricer::term_regime(bsm, s, k, r, q, tau) {
        TermRegime::AtTheForward => {
          let half = 0.5 * k * tau * (-r * tau).exp();
          match option_type {
            OptionType::Call => half,
            OptionType::Put => -half,
          }
        }
        TermRegime::Saturated | TermRegime::Ordinary => bsm.rho(s, k, r, q, tau, option_type),
      }
    })
  }

  /// Vega `∂V/∂σ`, one value for call and put (parity's spread is free of `σ`); a `σ_n = 0` term
  /// adds its right limit.
  pub fn vega(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    if tau == 0.0 {
      return f64::NAN;
    }
    if self.lambda <= 0.0 {
      return self.base_bsm().vega(s, k, r, q, tau);
    }
    self.volatility_series(tau, |bsm, scale| {
      scale
        * match Self::term_regime(bsm, s, k, r, q, tau) {
          TermRegime::AtTheForward => {
            s * ((bsm.b(r, q) - r) * tau).exp() * norm_pdf(0.0) * tau.sqrt()
          }
          TermRegime::Saturated => ((bsm.b(r, q) - r) * tau).exp() * 0.0,
          TermRegime::Ordinary => bsm.vega(s, k, r, q, tau),
        }
    })
  }

  /// Theta `∂V/∂t` (calendar) by a centred `τ` difference; `NaN` at `λ > 0` for a non-finite `tau`
  /// or `tau ≤ 1e-5` (the bump), [`BsmPricer`]'s answer at `λ ≤ 0`.
  pub fn theta(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    if tau == 0.0 {
      return f64::NAN;
    }
    if self.lambda <= 0.0 {
      return self.base_bsm().theta(s, k, r, q, tau, option_type);
    }
    let h = Self::H_TAU;
    if !(tau.is_finite() && tau > h) {
      return f64::NAN;
    }
    -(self.series_price(s, k, r, q, tau + h, option_type)
      - self.series_price(s, k, r, q, tau - h, option_type))
      / (2.0 * h)
  }

  /// Vanna `∂²V/∂S∂σ`, one value for call and put, since parity's spread is free of `σ`.
  pub fn vanna(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    if tau == 0.0 {
      return f64::NAN;
    }
    if self.lambda <= 0.0 {
      return self.base_bsm().vanna(s, k, r, q, tau);
    }
    self.volatility_series(tau, |bsm, scale| {
      scale
        * match Self::term_regime(bsm, s, k, r, q, tau) {
          TermRegime::AtTheForward => {
            0.5 * ((bsm.b(r, q) - r) * tau).exp() * norm_pdf(0.0) * tau.sqrt()
          }
          TermRegime::Saturated => ((bsm.b(r, q) - r) * tau).exp() * 0.0,
          TermRegime::Ordinary => bsm.vanna(s, k, r, q, tau),
        }
    })
  }

  /// Charm `∂²V/∂S∂t` (delta decay) by a centred `τ` difference; `NaN` at `λ > 0` for a non-finite
  /// `tau` or `tau ≤ 1e-5`, [`BsmPricer`]'s answer at `λ ≤ 0`.
  pub fn charm(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    if tau == 0.0 {
      return f64::NAN;
    }
    if self.lambda <= 0.0 {
      return self.base_bsm().charm(s, k, r, q, tau, option_type);
    }
    let ht = Self::H_TAU;
    if !(tau.is_finite() && tau > ht) {
      return f64::NAN;
    }
    let hs = Self::h_s(s);
    -(self.series_price(s + hs, k, r, q, tau + ht, option_type)
      - self.series_price(s + hs, k, r, q, tau - ht, option_type)
      - self.series_price(s - hs, k, r, q, tau + ht, option_type)
      + self.series_price(s - hs, k, r, q, tau - ht, option_type))
      / (4.0 * hs * ht)
  }

  /// Volga `∂²V/∂σ²`, one value for call and put, since parity's spread is free of `σ`.
  pub fn volga(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    if tau == 0.0 {
      return f64::NAN;
    }
    if self.lambda <= 0.0 {
      return self.base_bsm().volga(s, k, r, q, tau);
    }
    self.volatility_series(tau, |bsm, scale| {
      scale
        * scale
        * match Self::term_regime(bsm, s, k, r, q, tau) {
          TermRegime::AtTheForward | TermRegime::Saturated => ((bsm.b(r, q) - r) * tau).exp() * 0.0,
          TermRegime::Ordinary => bsm.volga(s, k, r, q, tau),
        }
    })
  }

  /// Veta `∂²V/∂σ∂t` (vega decay), one value for call and put; `NaN` at `λ > 0` for a non-finite
  /// `tau` or `tau ≤ 1e-5`, [`BsmPricer`]'s answer at `λ ≤ 0`.
  pub fn veta(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    if tau == 0.0 {
      return f64::NAN;
    }
    if self.lambda <= 0.0 {
      return self.base_bsm().veta(s, k, r, q, tau);
    }
    let ht = Self::H_TAU;
    if !(tau.is_finite() && tau > ht) {
      return f64::NAN;
    }
    -(self.vega(s, k, r, q, tau + ht) - self.vega(s, k, r, q, tau - ht)) / (2.0 * ht)
  }

  /// Every Greek at one query point.
  pub fn greeks(
    &self,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
    option_type: OptionType,
  ) -> Greeks {
    if tau == 0.0 {
      return Greeks::nan();
    }
    Greeks {
      delta: self.delta(s, k, r, q, tau, option_type),
      gamma: self.gamma(s, k, r, q, tau),
      vega: self.vega(s, k, r, q, tau),
      theta: self.theta(s, k, r, q, tau, option_type),
      rho: self.rho(s, k, r, q, tau, option_type),
      vanna: self.vanna(s, k, r, q, tau),
      charm: self.charm(s, k, r, q, tau, option_type),
      volga: self.volga(s, k, r, q, tau),
      veta: self.veta(s, k, r, q, tau),
    }
  }
}

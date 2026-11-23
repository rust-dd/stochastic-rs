use stochastic_rs_distributions::special::norm_cdf;
use stochastic_rs_distributions::special::norm_pdf;

use super::pricer::BsmPricer;
use crate::OptionType;
use crate::pricing::digital::expiry_indicator;
use crate::traits::Greeks;
use crate::traits::ModelPricer;

/// Closed-form Greeks at the `(s, k, r, q, tau)` query of [`BsmPricer`]'s prices, each but `zeta`
/// NaN at `tau = 0`, where an expired option has a value and no sensitivity.
impl BsmPricer {
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

  /// Delta — $\partial V/\partial S$.
  pub fn delta(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    if tau == 0.0 {
      return f64::NAN;
    }
    let (d1, _) = self.d1_d2(s, k, r, q, tau);
    let exp_bt = ((self.b(r, q) - r) * tau).exp();

    if option_type == OptionType::Call {
      exp_bt * norm_cdf(d1)
    } else {
      -exp_bt * norm_cdf(-d1)
    }
  }

  /// Gamma — $\partial^2 V/\partial S^2$.
  pub fn gamma(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    if tau == 0.0 {
      return f64::NAN;
    }
    let (d1, _) = self.d1_d2(s, k, r, q, tau);

    ((self.b(r, q) - r) * tau).exp() * norm_pdf(d1) / (s * self.sigma * tau.sqrt())
  }

  /// `S·Γ/100`, the change in delta for a 1 % move in spot.
  pub fn gamma_percent(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.gamma(s, k, r, q, tau) * s / 100.0
  }

  /// Theta — $\partial V/\partial t$ (calendar convention).
  pub fn theta(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    if tau == 0.0 {
      return f64::NAN;
    }
    let (d1, d2) = self.d1_d2(s, k, r, q, tau);
    let b = self.b(r, q);

    let exp_bt = ((b - r) * tau).exp();
    let exp_rt = (-r * tau).exp();
    let pdf_d1 = norm_pdf(d1);

    let first_term = -s * exp_bt * pdf_d1 * self.sigma / (2.0 * tau.sqrt());

    if option_type == OptionType::Call {
      let second_term = -(b - r) * s * exp_bt * norm_cdf(d1);
      let third_term = -r * k * exp_rt * norm_cdf(d2);
      first_term + second_term + third_term
    } else {
      let second_term = (b - r) * s * exp_bt * norm_cdf(-d1);
      let third_term = -r * k * exp_rt * norm_cdf(-d2);
      first_term + second_term + third_term
    }
  }

  /// Vega — $\partial V/\partial \sigma$.
  pub fn vega(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    if tau == 0.0 {
      return f64::NAN;
    }
    let (d1, _) = self.d1_d2(s, k, r, q, tau);

    s * ((self.b(r, q) - r) * tau).exp() * norm_pdf(d1) * tau.sqrt()
  }

  /// Rho — $\partial V/\partial r$.
  pub fn rho(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    if tau == 0.0 {
      return f64::NAN;
    }
    let (_, d2) = self.d1_d2(s, k, r, q, tau);

    let exp_rt = (-r * tau).exp();

    if option_type == OptionType::Call {
      k * tau * exp_rt * norm_cdf(d2)
    } else {
      -k * tau * exp_rt * norm_cdf(-d2)
    }
  }

  /// Vomma / volga — $\partial^2 V/\partial \sigma^2$.
  pub fn volga(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let (d1, d2) = self.d1_d2(s, k, r, q, tau);

    self.vega(s, k, r, q, tau) * d1 * d2 / self.sigma
  }

  /// Charm — $\partial^2 V/\partial S\partial t$ (delta decay).
  pub fn charm(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    if tau == 0.0 {
      return f64::NAN;
    }
    let v = self.sigma;
    let b = self.b(r, q);
    let (d1, d2) = self.d1_d2(s, k, r, q, tau);

    let exp_bt = ((b - r) * tau).exp();
    let pdf_d1 = norm_pdf(d1);
    let sqrt_tau = tau.sqrt();

    match option_type {
      OptionType::Call => {
        exp_bt * (pdf_d1 * ((b / (v * sqrt_tau)) - (d2 / (2.0 * tau))) + (b - r) * norm_cdf(d1))
      }
      OptionType::Put => {
        exp_bt * (pdf_d1 * ((b / (v * sqrt_tau)) - (d2 / (2.0 * tau))) - (b - r) * norm_cdf(-d1))
      }
    }
  }

  /// Vanna — $\partial^2 V/\partial S\partial \sigma$.
  pub fn vanna(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    if tau == 0.0 {
      return f64::NAN;
    }
    let (d1, d2) = self.d1_d2(s, k, r, q, tau);

    -((self.b(r, q) - r) * tau).exp() * norm_pdf(d1) * d2 / self.sigma
  }

  /// Zomma — $\partial \Gamma/\partial \sigma$.
  pub fn zomma(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let (d1, d2) = self.d1_d2(s, k, r, q, tau);

    self.gamma(s, k, r, q, tau) * (d1 * d2 - 1.0) / self.sigma
  }

  /// Zomma per 1% of spot.
  pub fn zomma_percent(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.zomma(s, k, r, q, tau) * s / 100.0
  }

  /// Speed — $\partial^3 V/\partial S^3$.
  pub fn speed(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let (d1, _) = self.d1_d2(s, k, r, q, tau);

    -self.gamma(s, k, r, q, tau) * (1.0 + d1 / (self.sigma * tau.sqrt())) / s
  }

  /// Color — $\partial \Gamma/\partial t$.
  pub fn color(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let (d1, d2) = self.d1_d2(s, k, r, q, tau);
    let b = self.b(r, q);

    self.gamma(s, k, r, q, tau)
      * (r - b + b * d1 / (self.sigma * tau.sqrt()) + (1.0 - d1 * d2) / (2.0 * tau))
  }

  /// Ultima — $\partial^3 V/\partial \sigma^3$,
  /// $$\mathrm{Vega}\,(d_1^2 d_2^2 - d_1^2 - d_2^2 - d_1 d_2)/\sigma^2.$$
  ///
  /// Reference: Haug (2007), *The Complete Guide to Option Pricing Formulas*, 2nd ed., DvommaDvol.
  pub fn ultima(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let (d1, d2) = self.d1_d2(s, k, r, q, tau);

    self.vega(s, k, r, q, tau) / self.sigma.powi(2)
      * (d1 * d1 * d2 * d2 - d1 * d1 - d2 * d2 - d1 * d2)
  }

  /// Veta / DvegaDtime — $\partial^2 V/\partial \sigma \partial t$.
  pub fn veta(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let (d1, d2) = self.d1_d2(s, k, r, q, tau);
    let b = self.b(r, q);

    self.vega(s, k, r, q, tau)
      * (r - b + b * d1 / (self.sigma * tau.sqrt()) - (d1 * d2 + 1.0) / (2.0 * tau))
  }

  /// Lambda (elasticity) $\Delta\,S/V$ of the given option.
  pub fn lambda(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    let price = self.price_option(s, k, r, q, tau, option_type);
    self.delta(s, k, r, q, tau, option_type) * s / price
  }

  /// Phi — $\partial V/\partial q$ (dividend / carry rho).
  pub fn phi(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    if tau == 0.0 {
      return f64::NAN;
    }
    let (d1, _) = self.d1_d2(s, k, r, q, tau);

    let exp_bt = ((self.b(r, q) - r) * tau).exp();

    if option_type == OptionType::Call {
      -tau * s * exp_bt * norm_cdf(d1)
    } else {
      tau * s * exp_bt * norm_cdf(-d1)
    }
  }

  /// Zeta — the in-the-money probability of Haug (2007), *The Complete Guide to Option Pricing
  /// Formulas*: `N(d₂)` (call) or `N(−d₂)` (put), and `1{s > k}` or `1{s < k}` at `tau = 0`.
  pub fn zeta(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    if tau == 0.0 && self.sigma.is_finite() {
      return expiry_indicator(option_type, s * (self.b(r, q) * tau).exp(), k);
    }
    let (_, d2) = self.d1_d2(s, k, r, q, tau);

    if option_type == OptionType::Call {
      norm_cdf(d2)
    } else {
      norm_cdf(-d2)
    }
  }

  /// Strike delta — $\partial V/\partial K$.
  pub fn strike_delta(
    &self,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
    option_type: OptionType,
  ) -> f64 {
    if tau == 0.0 {
      return f64::NAN;
    }
    let (_, d2) = self.d1_d2(s, k, r, q, tau);

    let exp_rt = (-r * tau).exp();

    if option_type == OptionType::Call {
      -exp_rt * norm_cdf(d2)
    } else {
      exp_rt * norm_cdf(-d2)
    }
  }

  /// Strike gamma — $\partial^2 V/\partial K^2$.
  pub fn strike_gamma(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    if tau == 0.0 {
      return f64::NAN;
    }
    let (_, d2) = self.d1_d2(s, k, r, q, tau);

    norm_pdf(d2) * (-r * tau).exp() / (k * self.sigma * tau.sqrt())
  }
}

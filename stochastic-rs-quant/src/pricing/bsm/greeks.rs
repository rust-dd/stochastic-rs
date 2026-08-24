use stochastic_rs_distributions::special::norm_cdf;
use stochastic_rs_distributions::special::norm_pdf;

use super::pricer::BSMPricer;
use crate::OptionType;
use crate::traits::Greeks;

/// Closed-form Greeks, evaluated at the same `(s, k, r, q, tau)` query
/// [`BSMPricer`]'s pricing methods take. Only the Greeks whose value
/// actually differs between a call and a put take `option_type`.
///
/// These are inherent methods and not a
/// [`GreeksExt`](crate::traits::GreeksExt) impl: that trait's accessors
/// take no arguments, so only a type that already carries a query can
/// implement it. The query-carrying types built on this model
/// (`AnalyticBSEngine`, `Merton1976Pricer`) implement it and delegate here.
impl BSMPricer {
  /// Every Greek at one query point, in a [`Greeks`] aggregate.
  ///
  /// This is what the removed `GreeksExt` impl's `greeks()` provided, and
  /// it is the **only** place the aggregate's two renamed members are
  /// mapped: `Greeks::volga` is [`vomma`](Self::vomma) and `Greeks::veta`
  /// is [`dvega_dtime`](Self::dvega_dtime). Callers that need the whole set
  /// (`AnalyticBSEngine`, the `mc_greeks_demo` example) go through here
  /// rather than re-deriving the mapping in a struct literal — see
  /// `bsm_greeks_aggregate_matches_accessors`.
  pub fn greeks(
    &self,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
    option_type: OptionType,
  ) -> Greeks {
    Greeks {
      delta: self.delta(s, k, r, q, tau, option_type),
      gamma: self.gamma(s, k, r, q, tau),
      vega: self.vega(s, k, r, q, tau),
      theta: self.theta(s, k, r, q, tau, option_type),
      rho: self.rho(s, k, r, q, tau, option_type),
      vanna: self.vanna(s, k, r, q, tau),
      charm: self.charm(s, k, r, q, tau, option_type),
      volga: self.vomma(s, k, r, q, tau),
      veta: self.dvega_dtime(s, k, r, q, tau),
    }
  }

  /// Delta — $\partial V/\partial S$.
  pub fn delta(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    let (d1, _) = self.d1_d2(s, k, r, q, tau);
    let exp_bt = ((self.b(r, q) - r) * tau).exp();

    if option_type == OptionType::Call {
      exp_bt * norm_cdf(d1)
    } else {
      exp_bt * (norm_cdf(d1) - 1.0)
    }
  }

  /// Gamma — $\partial^2 V/\partial S^2$.
  pub fn gamma(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let (d1, _) = self.d1_d2(s, k, r, q, tau);

    ((self.b(r, q) - r) * tau).exp() * norm_pdf(d1) / (s * self.v * tau.sqrt())
  }

  /// Gamma per 1% of spot.
  pub fn gamma_percent(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.gamma(s, k, r, q, tau) / s * 100.0
  }

  /// Theta — $\partial V/\partial t$ (calendar convention).
  pub fn theta(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    let (d1, d2) = self.d1_d2(s, k, r, q, tau);
    let b = self.b(r, q);

    let exp_bt = ((b - r) * tau).exp();
    let exp_rt = (-r * tau).exp();
    let pdf_d1 = norm_pdf(d1);

    let first_term = -s * exp_bt * pdf_d1 * self.v / (2.0 * tau.sqrt());

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
    let (d1, _) = self.d1_d2(s, k, r, q, tau);

    s * ((self.b(r, q) - r) * tau).exp() * norm_pdf(d1) * tau.sqrt()
  }

  /// Rho — $\partial V/\partial r$.
  pub fn rho(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    let (_, d2) = self.d1_d2(s, k, r, q, tau);

    let exp_rt = (-r * tau).exp();

    if option_type == OptionType::Call {
      k * tau * exp_rt * norm_cdf(d2)
    } else {
      -k * tau * exp_rt * norm_cdf(-d2)
    }
  }

  /// Vomma / volga — $\partial^2 V/\partial \sigma^2$.
  pub fn vomma(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let (d1, d2) = self.d1_d2(s, k, r, q, tau);

    self.vega(s, k, r, q, tau) * d1 * d2 / self.v
  }

  /// Charm — $\partial^2 V/\partial S\partial t$ (delta decay).
  pub fn charm(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    let v = self.v;
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
    let (d1, d2) = self.d1_d2(s, k, r, q, tau);

    -((self.b(r, q) - r) * tau).exp() * norm_pdf(d1) * d2 / self.v
  }

  /// Zomma — $\partial \Gamma/\partial \sigma$.
  pub fn zomma(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let (d1, d2) = self.d1_d2(s, k, r, q, tau);

    self.gamma(s, k, r, q, tau) * (d1 * d2 - 1.0) / self.v
  }

  /// Zomma per 1% of spot.
  pub fn zomma_percent(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.zomma(s, k, r, q, tau) * s / 100.0
  }

  /// Speed — $\partial^3 V/\partial S^3$.
  pub fn speed(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let (d1, _) = self.d1_d2(s, k, r, q, tau);

    -self.gamma(s, k, r, q, tau) * (1.0 + d1 / (self.v * tau.sqrt())) / s
  }

  /// Color — $\partial \Gamma/\partial t$.
  pub fn color(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let (d1, d2) = self.d1_d2(s, k, r, q, tau);
    let b = self.b(r, q);

    self.gamma(s, k, r, q, tau)
      * (r - b + b * d1 / (self.v * tau.sqrt()) + (1.0 - d1 * d2) / (2.0 * tau))
  }

  /// Ultima — $\partial^3 V/\partial \sigma^3$.
  pub fn ultima(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let (d1, d2) = self.d1_d2(s, k, r, q, tau);

    -self.vomma(s, k, r, q, tau) / self.v * (d1 * d2 - (d1 / d2) + (d2 / d1) - 1.0)
  }

  /// Veta / DvegaDtime — $\partial^2 V/\partial \sigma \partial t$.
  pub fn dvega_dtime(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let (d1, d2) = self.d1_d2(s, k, r, q, tau);
    let b = self.b(r, q);

    self.vega(s, k, r, q, tau)
      * (r - b + b * d1 / (self.v * tau.sqrt()) - (d1 * d2 + 1.0) / (2.0 * tau))
  }

  /// Lambda (elasticity) against the call and against the put, both using
  /// the delta of `option_type` — preserved verbatim from the pre-query
  /// method, which likewise divided a single delta by both prices.
  pub fn lambda(
    &self,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
    option_type: OptionType,
  ) -> (f64, f64) {
    let (call, put) = self.call_put(s, k, r, q, tau);
    let delta = self.delta(s, k, r, q, tau, option_type);
    (delta * s / call, delta * s / put)
  }

  /// Phi — $\partial V/\partial q$ (dividend / carry rho).
  pub fn phi(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    let (d1, _) = self.d1_d2(s, k, r, q, tau);

    let exp_bt = ((self.b(r, q) - r) * tau).exp();

    if option_type == OptionType::Call {
      -tau * s * exp_bt * norm_cdf(d1)
    } else {
      tau * s * exp_bt * norm_cdf(-d1)
    }
  }

  /// Zeta — risk-neutral probability of finishing in the money.
  pub fn zeta(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    let (_, d2) = self.d1_d2(s, k, r, q, tau);

    if option_type == OptionType::Call {
      norm_cdf(d2)
    } else {
      -norm_cdf(-d2)
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
    let (_, d2) = self.d1_d2(s, k, r, q, tau);

    norm_pdf(d2) * (-r * tau).exp() / (k * self.v * tau.sqrt())
  }
}

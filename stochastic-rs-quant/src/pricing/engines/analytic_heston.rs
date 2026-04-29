//! Closed-form Heston engine for [`EuropeanOption`].
//!
//! Wraps [`HestonPricer`] behind reactive market handles. Heston Greeks
//! are produced by central finite differences against the analytic
//! characteristic-function call price.

use std::sync::Arc;

use crate::OptionType;
use crate::instruments::equity::EuropeanOption;
use crate::market::Handle;
use crate::market::Quote;
use crate::market::SimpleQuote;
use crate::pricing::HestonPricer;
use crate::traits::Greeks;
use crate::traits::PricerExt;
use crate::traits::PricingEngine;
use crate::traits::StandardResult;

/// Heston model parameters that are calibrated, not market-quoted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HestonStaticParams {
  /// Initial variance.
  pub v0: f64,
  /// Mean-reversion speed.
  pub kappa: f64,
  /// Long-run variance.
  pub theta: f64,
  /// Volatility of variance.
  pub sigma: f64,
  /// Spot/variance correlation.
  pub rho: f64,
  /// Optional market price of vol risk.
  pub lambda: Option<f64>,
}

impl HestonStaticParams {
  pub const fn new(v0: f64, kappa: f64, theta: f64, sigma: f64, rho: f64) -> Self {
    Self {
      v0,
      kappa,
      theta,
      sigma,
      rho,
      lambda: None,
    }
  }
}

/// Analytic Heston engine.
#[derive(Clone)]
pub struct AnalyticHestonEngine {
  pub spot: Handle<SimpleQuote<f64>>,
  pub risk_free: Handle<SimpleQuote<f64>>,
  pub dividend_yield: Handle<SimpleQuote<f64>>,
  pub params: HestonStaticParams,
  /// Relative bump used for finite-difference Greeks (default 1e-3).
  pub bump: f64,
}

impl AnalyticHestonEngine {
  pub fn new(
    spot: Handle<SimpleQuote<f64>>,
    risk_free: Handle<SimpleQuote<f64>>,
    dividend_yield: Handle<SimpleQuote<f64>>,
    params: HestonStaticParams,
  ) -> Self {
    Self {
      spot,
      risk_free,
      dividend_yield,
      params,
      bump: 1e-3,
    }
  }

  /// Wrap scalars in fresh handles. Useful in tests / one-shot pricing.
  pub fn with_constants(s: f64, r: f64, q: f64, params: HestonStaticParams) -> Self {
    Self::new(
      Handle::new(Arc::new(SimpleQuote::new(s))),
      Handle::new(Arc::new(SimpleQuote::new(r))),
      Handle::new(Arc::new(SimpleQuote::new(q))),
      params,
    )
  }

  fn read_quote(handle: &Handle<SimpleQuote<f64>>, default: f64) -> f64 {
    handle.current().map(|q| q.value()).unwrap_or(default)
  }

  fn build_pricer(
    &self,
    s_override: Option<f64>,
    v0_override: Option<f64>,
    tau_override: Option<f64>,
    opt: &EuropeanOption,
  ) -> HestonPricer {
    HestonPricer {
      s: s_override.unwrap_or_else(|| Self::read_quote(&self.spot, 0.0)),
      v0: v0_override.unwrap_or(self.params.v0),
      k: opt.strike,
      r: Self::read_quote(&self.risk_free, 0.0),
      q: Some(Self::read_quote(&self.dividend_yield, 0.0)),
      rho: self.params.rho,
      kappa: self.params.kappa,
      theta: self.params.theta,
      sigma: self.params.sigma,
      lambda: self.params.lambda,
      tau: tau_override.or(opt.tau),
      eval: opt.eval,
      expiration: opt.expiry,
    }
  }

  fn price_at(
    &self,
    s_override: Option<f64>,
    v0_override: Option<f64>,
    tau_override: Option<f64>,
    opt: &EuropeanOption,
  ) -> f64 {
    let pricer = self.build_pricer(s_override, v0_override, tau_override, opt);
    let (call, put) = pricer.calculate_call_put();
    match opt.option_type {
      OptionType::Call => call,
      OptionType::Put => put,
    }
  }

  fn finite_diff_greeks(&self, opt: &EuropeanOption) -> Greeks {
    let s = Self::read_quote(&self.spot, 0.0);
    let h_s = (s.abs().max(1.0)) * self.bump;
    let p0 = self.price_at(None, None, None, opt);
    let p_up = self.price_at(Some(s + h_s), None, None, opt);
    let p_dn = self.price_at(Some(s - h_s), None, None, opt);
    let delta = (p_up - p_dn) / (2.0 * h_s);
    let gamma = (p_up - 2.0 * p0 + p_dn) / (h_s * h_s);

    let v0 = self.params.v0;
    let h_v = v0.abs().max(1e-3) * self.bump;
    let p_v_up = self.price_at(None, Some(v0 + h_v), None, opt);
    let p_v_dn = self.price_at(None, Some((v0 - h_v).max(1e-12)), None, opt);
    let dv_dv0 = (p_v_up - p_v_dn) / (2.0 * h_v);
    // Vega = ∂P/∂σ ≈ ∂P/∂v0 · 2 √v0 (chain rule via σ = √v).
    let vega = if v0 > 0.0 {
      dv_dv0 * 2.0 * v0.sqrt()
    } else {
      f64::NAN
    };

    let tau = opt.tau.unwrap_or(f64::NAN);
    let theta = if tau.is_finite() && tau > self.bump {
      let h_t = tau * self.bump;
      let p_t_dn = self.price_at(None, None, Some(tau - h_t), opt);
      // θ = -∂P/∂t = ∂P/∂(T-t) sign convention.
      -(p_t_dn - p0) / h_t
    } else {
      f64::NAN
    };

    Greeks {
      delta,
      gamma,
      vega,
      theta,
      ..Greeks::nan()
    }
  }
}

impl PricingEngine<EuropeanOption> for AnalyticHestonEngine {
  type Result = StandardResult;

  fn calculate(&self, opt: &EuropeanOption) -> StandardResult {
    let pricer = self.build_pricer(None, None, None, opt);
    let (call, put) = pricer.calculate_call_put();
    let npv = match opt.option_type {
      OptionType::Call => call,
      OptionType::Put => put,
    };
    let greeks = self.finite_diff_greeks(opt);
    StandardResult::with_greeks(npv, greeks)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::pricing::engines::AnalyticBSEngine;
  use crate::traits::PricingResult;

  #[test]
  fn heston_call_atm_positive() {
    let opt = EuropeanOption::new_tau(100.0, OptionType::Call, 1.0);
    let params = HestonStaticParams::new(0.04, 1.5, 0.04, 0.3, -0.7);
    let engine = AnalyticHestonEngine::with_constants(100.0, 0.05, 0.0, params);
    let r = engine.calculate(&opt);
    assert!(r.npv() > 0.0);
    let g = r.greeks().unwrap();
    assert!(g.delta > 0.0 && g.delta < 1.0);
    assert!(g.gamma > 0.0);
    assert!(g.vega > 0.0);
  }

  #[test]
  fn heston_zero_vol_of_vol_collapses_to_bs() {
    // sigma → 0 with v0 = theta freezes variance at v0 → Heston ≈ BS(σ=√v0).
    let opt = EuropeanOption::new_tau(100.0, OptionType::Call, 1.0);
    let v0 = 0.04;
    let params = HestonStaticParams::new(v0, 1.0, v0, 1e-4, 0.0);
    let heston = AnalyticHestonEngine::with_constants(100.0, 0.05, 0.0, params);
    let bs = AnalyticBSEngine::with_constants(100.0, v0.sqrt(), 0.05, 0.0);
    let p_h = heston.calculate(&opt).npv();
    let p_b = bs.calculate(&opt).npv();
    assert!((p_h - p_b).abs() < 0.05, "heston={p_h}, bs={p_b}");
  }

  #[test]
  fn heston_put_call_parity() {
    let call = EuropeanOption::new_tau(100.0, OptionType::Call, 1.0);
    let put = EuropeanOption::new_tau(100.0, OptionType::Put, 1.0);
    let params = HestonStaticParams::new(0.04, 1.5, 0.04, 0.3, -0.7);
    let engine = AnalyticHestonEngine::with_constants(100.0, 0.05, 0.02, params);
    let c = engine.calculate(&call).npv();
    let p = engine.calculate(&put).npv();
    let parity = 100.0 * (-0.02_f64).exp() - 100.0 * (-0.05_f64).exp();
    assert!((c - p - parity).abs() < 1e-2);
  }
}

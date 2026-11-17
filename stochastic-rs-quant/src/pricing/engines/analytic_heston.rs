//! Closed-form Heston engine for [`EuropeanOption`].
//!
//! Wraps [`HestonPricer`] behind reactive market handles. Heston Greeks
//! are produced by central finite differences against the analytic
//! characteristic-function call price.

use std::sync::Arc;

use crate::instruments::equity::EuropeanOption;
use crate::market::Handle;
use crate::market::Quote;
use crate::market::SimpleQuote;
use crate::pricing::heston::HestonParams;
use crate::pricing::heston::HestonPricer;
use crate::traits::Greeks;
use crate::traits::ModelPricer;
use crate::traits::PricingEngine;
use crate::traits::StandardResult;
use crate::traits::TimeExt;

/// [`PricingEngine`] pricing a European option with the semi-closed-form Heston formula.
#[derive(Clone)]
pub struct AnalyticHestonEngine {
  pub s: Handle<SimpleQuote<f64>>,
  pub r: Handle<SimpleQuote<f64>>,
  pub dividend_yield: Handle<SimpleQuote<f64>>,
  pub params: HestonParams,
  /// Heston's market price of volatility risk `λ`; `None` is `λ = 0`.
  pub lambda: Option<f64>,
  /// Relative bump used for finite-difference Greeks (default 1e-3).
  pub bump: f64,
}

impl AnalyticHestonEngine {
  /// Panics unless `params` passes [`HestonParams::new`].
  pub fn new(
    s: Handle<SimpleQuote<f64>>,
    r: Handle<SimpleQuote<f64>>,
    dividend_yield: Handle<SimpleQuote<f64>>,
    params: HestonParams,
  ) -> Self {
    Self {
      s,
      r,
      dividend_yield,
      params: params.checked(),
      lambda: None,
      bump: 1e-3,
    }
  }

  /// Wraps the scalars in fresh handles; panics as [`new`](Self::new) does.
  pub fn with_constants(s: f64, r: f64, q: f64, params: HestonParams) -> Self {
    Self::new(
      Handle::new(Arc::new(SimpleQuote::new(s))),
      Handle::new(Arc::new(SimpleQuote::new(r))),
      Handle::new(Arc::new(SimpleQuote::new(q))),
      params,
    )
  }

  /// Current value of a market handle, or [`f64::NAN`] when the handle is
  /// unlinked — the same missing-data answer as
  /// [`AnalyticBsEngine::read_quote`](crate::pricing::engines::AnalyticBsEngine)
  /// and as [`TimeExt::tau_or_from_dates`] gives for a missing maturity.
  /// Reading an unset handle as `0.0` priced at a spot or rate the caller
  /// never supplied.
  fn read_quote(handle: &Handle<SimpleQuote<f64>>) -> f64 {
    handle.current().map(|q| q.value()).unwrap_or(f64::NAN)
  }

  /// The model carries no query, so the engine resolves one from its own
  /// market handles and the instrument — `EuropeanOption` implements
  /// [`TimeExt`](crate::traits::TimeExt), so it resolves its own τ.
  fn model_and_query(
    &self,
    s_override: Option<f64>,
    v0_override: Option<f64>,
    tau_override: Option<f64>,
    opt: &EuropeanOption,
  ) -> (HestonPricer, f64, f64, f64, f64, f64) {
    let mut model = HestonPricer::new(HestonParams {
      v0: v0_override.unwrap_or(self.params.v0),
      ..self.params
    });
    if let Some(lambda) = self.lambda {
      model = model.with_vol_risk_premium(lambda);
    }
    let s = s_override.unwrap_or_else(|| Self::read_quote(&self.s));
    let r = Self::read_quote(&self.r);
    let q = Self::read_quote(&self.dividend_yield);
    let tau = tau_override.unwrap_or_else(|| opt.tau_or_from_dates());
    (model, s, opt.strike, r, q, tau)
  }

  fn price_at(
    &self,
    s_override: Option<f64>,
    v0_override: Option<f64>,
    tau_override: Option<f64>,
    opt: &EuropeanOption,
  ) -> f64 {
    let (model, s, k, r, q, tau) = self.model_and_query(s_override, v0_override, tau_override, opt);
    model.price_option(s, k, r, q, tau, opt.option_type)
  }

  /// Bump-and-revalue delta, gamma, vega and theta: all NaN at `tau = 0`, vega at `v0 <= 0`, theta
  /// unless `tau` is finite and above `bump`; the other five fields stay NaN.
  fn finite_diff_greeks(&self, opt: &EuropeanOption) -> Greeks {
    let tau = opt.tau_or_from_dates();
    if tau == 0.0 {
      return Greeks::nan();
    }
    let s = Self::read_quote(&self.s);
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

    let theta = if tau.is_finite() && tau > self.bump {
      let h_t = tau * self.bump;
      let p_t_dn = self.price_at(None, None, Some(tau - h_t), opt);
      // Calendar convention θ = ∂P/∂t, matching `GreeksExt::theta`'s own
      // doc and every other in-tree Greeks impl (`BsmPricer`,
      // `Merton1976Pricer`, `HestonPricer`, `AnalyticBsEngine`). Since
      // τ = T-t, ∂P/∂t = -∂P/∂τ ≈ -(p0 - p_t_dn)/h_t = (p_t_dn - p0)/h_t.
      (p_t_dn - p0) / h_t
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
    let npv = self.price_at(None, None, None, opt);
    let greeks = self.finite_diff_greeks(opt);
    StandardResult::with_greeks(npv, greeks)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::OptionType;
  use crate::pricing::engines::AnalyticBsEngine;
  use crate::traits::PricingResult;

  const P: HestonParams = HestonParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
  };

  #[test]
  fn heston_call_atm_positive() {
    let opt = EuropeanOption::new_tau(100.0, OptionType::Call, 1.0);
    let params = P;
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
    let params = HestonParams {
      v0,
      kappa: 1.0,
      theta: v0,
      sigma: 1e-4,
      rho: 0.0,
    };
    let heston = AnalyticHestonEngine::with_constants(100.0, 0.05, 0.0, params);
    let bs = AnalyticBsEngine::with_constants(100.0, v0.sqrt(), 0.05, 0.0);
    let p_h = heston.calculate(&opt).npv();
    let p_b = bs.calculate(&opt).npv();
    assert!((p_h - p_b).abs() < 0.05, "heston={p_h}, bs={p_b}");
  }

  /// Each unlinked handle poisons the NPV and every Greek; the delta and gamma step stays finite
  /// (`f64::max` drops the NaN), so the NaN arrives through the re-valuation.
  #[test]
  fn every_unlinked_handle_poisons_npv_and_greeks() {
    let opt = EuropeanOption::new_tau(100.0, OptionType::Call, 1.0);
    let params = P;
    let linked = |v: f64| Handle::new(Arc::new(SimpleQuote::new(v)));
    let quotes = [100.0, 0.05, 0.0];

    for missing in 0..3 {
      let mut h = [linked(quotes[0]), linked(quotes[1]), linked(quotes[2])];
      h[missing] = Handle::empty();
      let [s, r, q] = h;
      let res = AnalyticHestonEngine::new(s, r, q, params).calculate(&opt);
      assert!(res.npv().is_nan(), "handle {missing}: npv {}", res.npv());
      let g = res.greeks().unwrap();
      assert!(g.delta.is_nan(), "handle {missing}: delta {}", g.delta);
      assert!(g.gamma.is_nan(), "handle {missing}: gamma {}", g.gamma);
      assert!(g.vega.is_nan(), "handle {missing}: vega {}", g.vega);
      assert!(g.theta.is_nan(), "handle {missing}: theta {}", g.theta);
    }
  }

  #[test]
  fn an_expired_option_npvs_its_payoff_and_has_no_greek() {
    let engine = AnalyticHestonEngine::with_constants(110.0, 0.05, 0.0, P);
    let res = engine.calculate(&EuropeanOption::new_tau(100.0, OptionType::Call, 0.0));
    assert_eq!(res.npv(), 10.0);
    let greeks = res.greeks().unwrap().as_array();
    assert!(greeks.iter().all(|g| g.is_nan()), "{greeks:?}");
  }

  #[test]
  fn heston_put_call_parity() {
    let call = EuropeanOption::new_tau(100.0, OptionType::Call, 1.0);
    let put = EuropeanOption::new_tau(100.0, OptionType::Put, 1.0);
    let params = P;
    let engine = AnalyticHestonEngine::with_constants(100.0, 0.05, 0.02, params);
    let c = engine.calculate(&call).npv();
    let p = engine.calculate(&put).npv();
    let parity = 100.0 * (-0.02_f64).exp() - 100.0 * (-0.05_f64).exp();
    assert!((c - p - parity).abs() < 1e-2);
  }

  /// A literal handed to the engine meets `HestonParams::new`'s checks, by name; `κ < 0`, zero
  /// variances and `|ρ| = 1` still construct.
  #[test]
  fn new_checks_a_literal_as_heston_params_new_does() {
    let cases = [
      (
        HestonParams { v0: -0.01, ..P },
        "v0 must satisfy `v0.is_finite() && v0 >= 0.0`, got v0 = -0.01",
      ),
      (
        HestonParams { theta: -0.04, ..P },
        "theta must satisfy `theta.is_finite() && theta >= 0.0`, got theta = -0.04",
      ),
      (
        HestonParams { sigma: 0.0, ..P },
        "sigma must satisfy `sigma.is_finite() && sigma > 0.0`, got sigma = 0.0",
      ),
      (
        HestonParams { rho: -1.5, ..P },
        "rho must satisfy `(-1.0..=1.0).contains(&rho)`, got rho = -1.5",
      ),
    ];
    for (bad, expected) in cases {
      let payload = std::panic::catch_unwind(|| {
        AnalyticHestonEngine::with_constants(100.0, 0.05, 0.0, bad);
      })
      .expect_err(expected);
      let message = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .unwrap_or_default();
      assert_eq!(message, expected);
    }
    let edges = HestonParams {
      v0: 0.0,
      kappa: -1.5,
      theta: 0.0,
      sigma: 1e-12,
      rho: -1.0,
    };
    assert_eq!(
      AnalyticHestonEngine::with_constants(100.0, 0.05, 0.0, edges).params,
      edges
    );
  }
}

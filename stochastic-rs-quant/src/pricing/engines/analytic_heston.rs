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
use crate::pricing::HestonPricer;
use crate::traits::Greeks;
use crate::traits::ModelPricer;
use crate::traits::PricingEngine;
use crate::traits::StandardResult;
use crate::traits::TimeExt;

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
  /// Validating constructor, rejecting exactly what
  /// [`HestonPricer::new`] rejects.
  ///
  /// These parameters feed that constructor inside
  /// `AnalyticHestonEngine::model_and_query`, so an invalid one was
  /// already rejected — but at *pricing* time, naming a type the caller
  /// never mentioned, and only if they went on to call
  /// [`PricingEngine::calculate`]. Nothing about which values are valid
  /// changes here; only the layer that says so.
  ///
  /// Note the argument order differs from the inner constructor's
  /// (`v0, rho, kappa, theta, sigma, lambda`), which is why each check is
  /// matched to its field by name rather than by position.
  ///
  /// The fields are `pub`, so this is the front door and not a wall: a
  /// struct literal still reaches them, and the inner constructor stays the
  /// second line of defence. Its messages carry the `HestonPricer::new`
  /// prefix and these carry `HestonStaticParams::new`, so neither is a
  /// substring of the other.
  ///
  /// # Panics
  /// - if `v0` or `theta` is negative or `NaN` — a variance cannot be either
  /// - if `sigma` is not strictly positive — the characteristic function
  ///   divides by $\sigma^2$
  /// - if `rho` is outside `[-1, 1]` or `NaN` — not a correlation
  ///
  /// `kappa` is unconstrained for the same reason it is on
  /// [`HestonPricer::new`]: a non-positive mean-reversion rate is a
  /// non-stationary but well-defined affine model.
  pub fn new(v0: f64, kappa: f64, theta: f64, sigma: f64, rho: f64) -> Self {
    assert!(
      v0 >= 0.0,
      "HestonStaticParams::new: v0 must be a non-negative variance (got {v0})"
    );
    assert!(
      theta >= 0.0,
      "HestonStaticParams::new: theta must be a non-negative variance (got {theta})"
    );
    assert!(
      sigma > 0.0,
      "HestonStaticParams::new: sigma must be strictly positive (got {sigma})"
    );
    assert!(
      (-1.0..=1.0).contains(&rho),
      "HestonStaticParams::new: rho must be in [-1, 1] (got {rho})"
    );
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
  pub s: Handle<SimpleQuote<f64>>,
  pub r: Handle<SimpleQuote<f64>>,
  pub dividend_yield: Handle<SimpleQuote<f64>>,
  pub params: HestonStaticParams,
  /// Relative bump used for finite-difference Greeks (default 1e-3).
  pub bump: f64,
}

impl AnalyticHestonEngine {
  pub fn new(
    s: Handle<SimpleQuote<f64>>,
    r: Handle<SimpleQuote<f64>>,
    dividend_yield: Handle<SimpleQuote<f64>>,
    params: HestonStaticParams,
  ) -> Self {
    Self {
      s,
      r,
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

  /// Current value of a market handle, or [`f64::NAN`] when the handle is
  /// unlinked — the same missing-data answer as
  /// [`AnalyticBSEngine::read_quote`](crate::pricing::engines::AnalyticBSEngine)
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
    let model = HestonPricer::new(
      v0_override.unwrap_or(self.params.v0),
      self.params.rho,
      self.params.kappa,
      self.params.theta,
      self.params.sigma,
      self.params.lambda,
    );
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

  /// Bump-and-revalue Greeks, with `vanna`/`charm`/`volga`/`veta` left at
  /// [`Greeks::nan`] because this engine does not compute them.
  ///
  /// `vega` is additionally `NaN` at `v0 <= 0`, where the `σ = √v0` chain
  /// rule is undefined, and `theta` at a `tau` that is non-finite or not
  /// larger than `bump`. Both are case 2 of the crate's [failure
  /// convention](crate::traits::ModelPricer#how-pricing-fails), as is the
  /// all-`NaN` struct an unlinked market handle produces — see
  /// [`read_quote`](Self::read_quote).
  ///
  /// Unlike [`HestonPricer`](crate::pricing::heston::HestonPricer)'s own
  /// volatility-space Greeks, a *negative* `v0` is not rejected here: this
  /// method fills one member of a struct whose others are finite and usable,
  /// so panicking would take down the whole
  /// [`PricingEngine::calculate`](crate::traits::PricingEngine::calculate)
  /// call to report a single degenerate field.
  fn finite_diff_greeks(&self, opt: &EuropeanOption) -> Greeks {
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

    let tau = opt.tau_or_from_dates();
    let theta = if tau.is_finite() && tau > self.bump {
      let h_t = tau * self.bump;
      let p_t_dn = self.price_at(None, None, Some(tau - h_t), opt);
      // Calendar convention θ = ∂P/∂t, matching `GreeksExt::theta`'s own
      // doc and every other in-tree Greeks impl (`BSMPricer`,
      // `Merton1976Pricer`, `HestonPricer`, `AnalyticBSEngine`). Since
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

  /// Same missing-data convention as [`AnalyticBSEngine`], checked handle by
  /// handle. Before the fix an unlinked spot read as `0.0`, and the Heston
  /// characteristic-function integral at `s = 0` returned a finite NPV for a
  /// spot the caller never supplied. `delta` and `gamma` are additionally
  /// worth naming here: they bump `s` by `s.abs().max(1.0) * bump`, and
  /// `f64::max` discards a `NaN` operand, so the step size stays finite —
  /// only the re-valuation at `NaN ± h` carries the poison through.
  #[test]
  fn every_unlinked_handle_poisons_npv_and_greeks() {
    let opt = EuropeanOption::new_tau(100.0, OptionType::Call, 1.0);
    let params = HestonStaticParams::new(0.04, 1.5, 0.04, 0.3, -0.7);
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

  /// `HestonStaticParams::new` validates the same parameters
  /// [`HestonPricer::new`] does, at the layer the caller actually supplied
  /// them.
  ///
  /// Before this the struct built happily and the rejection arrived from
  /// `model_and_query`'s inner `HestonPricer::new` — at *pricing* time,
  /// naming a type the caller never mentioned, and only if they went on to
  /// call `calculate`. The parameters are identical; only the layer moved.
  ///
  /// The fields are `pub`, so this is a front door and not a wall: a struct
  /// literal still reaches them, which is why the inner constructor stays
  /// the second line of defence. Its messages carry the `HestonPricer::new`
  /// prefix and these carry `HestonStaticParams::new`, so neither is a
  /// substring of the other and an `expected` anchor cannot be satisfied by
  /// the wrong guard firing.
  mod construction_validation {
    use super::*;

    #[test]
    #[should_panic(
      expected = "HestonStaticParams::new: v0 must be a non-negative variance (got -0.01)"
    )]
    fn new_rejects_negative_v0() {
      let _ = HestonStaticParams::new(-0.01, 1.5, 0.04, 0.3, -0.7);
    }

    #[test]
    #[should_panic(
      expected = "HestonStaticParams::new: theta must be a non-negative variance (got -0.04)"
    )]
    fn new_rejects_negative_theta() {
      let _ = HestonStaticParams::new(0.04, 1.5, -0.04, 0.3, -0.7);
    }

    #[test]
    #[should_panic(expected = "HestonStaticParams::new: sigma must be strictly positive (got 0)")]
    fn new_rejects_zero_sigma() {
      let _ = HestonStaticParams::new(0.04, 1.5, 0.04, 0.0, -0.7);
    }

    #[test]
    #[should_panic(expected = "HestonStaticParams::new: rho must be in [-1, 1] (got -1.5)")]
    fn new_rejects_out_of_range_rho() {
      let _ = HestonStaticParams::new(0.04, 1.5, 0.04, 0.3, -1.5);
    }

    /// The argument order differs from [`HestonPricer::new`]'s — this one
    /// is `(v0, kappa, theta, sigma, rho)`, that one is
    /// `(v0, rho, kappa, theta, sigma, lambda)` — so a guard copied
    /// positionally rather than by name would check the wrong field. This
    /// pins each message against the value that actually offended.
    #[test]
    fn each_message_names_the_field_the_caller_passed() {
      let cases: [(&str, fn()); 4] = [
        ("v0", || {
          let _ = HestonStaticParams::new(-0.5, 1.5, 0.04, 0.3, -0.7);
        }),
        ("theta", || {
          let _ = HestonStaticParams::new(0.04, 1.5, -0.5, 0.3, -0.7);
        }),
        ("sigma", || {
          let _ = HestonStaticParams::new(0.04, 1.5, 0.04, -0.5, -0.7);
        }),
        ("rho", || {
          let _ = HestonStaticParams::new(0.04, 1.5, 0.04, 0.3, -1.5);
        }),
      ];
      for (field, build) in cases {
        let err = std::panic::catch_unwind(build).expect_err("must reject");
        let msg = err.downcast_ref::<String>().cloned().unwrap_or_else(|| {
          err
            .downcast_ref::<&str>()
            .copied()
            .unwrap_or("")
            .to_string()
        });
        assert!(
          msg.contains(&format!("HestonStaticParams::new: {field} ")),
          "{field}: wrong message {msg}"
        );
      }
    }

    /// `kappa` stays unconstrained for the same reason it does on
    /// [`HestonPricer::new`] — a non-positive mean-reversion rate is a
    /// non-stationary but well-defined affine model — and the admissible
    /// zero variances and unit correlations must still construct.
    #[test]
    fn new_accepts_what_the_inner_constructor_accepts() {
      assert_eq!(
        HestonStaticParams::new(0.04, -1.5, 0.04, 0.3, -0.7).kappa,
        -1.5
      );
      let edges = HestonStaticParams::new(0.0, 1.5, 0.0, 1e-12, -1.0);
      assert_eq!(edges.v0, 0.0);
      assert_eq!(edges.theta, 0.0);
      assert_eq!(edges.rho, -1.0);
      assert_eq!(HestonStaticParams::new(0.04, 1.5, 0.04, 0.3, 1.0).rho, 1.0);
    }
  }
}

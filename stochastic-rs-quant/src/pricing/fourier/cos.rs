//! Generic Fourier-cosine (COS) pricing engine.
//!
//! Reference:
//! - Fang, F. & Oosterlee, C.W. (2008), "A Novel Pricing Method for European
//!   Options Based on Fourier-Cosine Series Expansions", SIAM Journal on
//!   Scientific Computing 31(2), 826-848, DOI: 10.1137/080718061.
//!
//! Unlike [`crate::pricing::regime_switching::CosPricer`] (hardcoded to
//! [`crate::pricing::regime_switching::RegimeSwitchingModel`]), [`CosEngine`]
//! prices any [`FourierModelExt`] model — the same trait consumed by
//! [`super::CarrMadanPricer`], [`super::GilPelaezPricer`] and
//! [`super::LewisPricer`].
//!
//! # Convention adaptation
//!
//! Fang-Oosterlee's own derivation works directly in the state variable
//! $y=\ln(S_T/K)$: the process starts at $x=\ln(S_0/K)$, so their cumulants
//! $c_1,c_2,c_4$ and characteristic function $\varphi(u;x)=\phi(u)e^{iux}$
//! are already expressed relative to the strike. [`FourierModelExt::chf`]
//! and [`super::Cumulants`] are instead expressed relative to the *spot*:
//! `chf` is the characteristic function of $\ln(S_T/S_0)$ (see the trait
//! doc) and `cumulants` are cumulants of that same log-return, so one model
//! instance can price any strike without rebuilding its `chf`.
//!
//! This engine bridges the two conventions with the deterministic shift
//! $x=\ln(S_0/K)$: the truncation range folds $x$ into the first cumulant
//! ($\tilde c_1=x+c_1$; $c_2,c_4$ are shift-invariant so they need no
//! adjustment), and the per-frequency characteristic function is
//! $\varphi_y(u)=\mathrm{chf}(t,u)\cdot e^{iux}$ rather than Fang-Oosterlee's
//! $e^{-iu\ln K}$ (their formula assumes a `chf` that is already the
//! characteristic function of $\ln S_T$ itself, not of $\ln(S_T/S_0)$).
//! With that substitution the payoff coefficients and pricing sum below are
//! exactly Fang-Oosterlee (2008) Eq. (22)-(23) and Eq. (29)-(30)/(32),
//! evaluated in $y=\ln(S_T/K)$-space.

use std::f64::consts::PI;

use num_complex::Complex64;

use super::FourierModelExt;
use crate::OptionType;

/// Fang-Oosterlee (2008) Fourier-cosine (COS) pricer, generic over any
/// [`FourierModelExt`] model.
///
/// `n` is the number of cosine expansion terms and `l` is the truncation
/// half-width in cumulant-standard-deviations (Fang-Oosterlee's $L$).
/// [`Default`] uses `n=256, l=10.0`, the paper's own §5.1 configuration.
///
/// # Truncation accuracy depends on `model.cumulants()`
///
/// The `[a, b]` truncation range — and therefore the entire expansion — is
/// sized from `model.cumulants(t)`, not from `model.chf` directly. If a
/// [`FourierModelExt`] implementor's `cumulants()` understates its
/// log-return's true variance/kurtosis, `l=10` can be too narrow to cover
/// the density's mass. The failure mode is silent: [`Self::price`] returns
/// a finite, plausible-looking but **wrong** price, not a panic or `NaN`.
/// This was not hypothetical: `HestonFourier::cumulants()` used to
/// understate `c2` (an earlier version of its formula omitted the `v0`
/// terms entirely, by 36-400× for common stochastic-volatility
/// parameters), which made `Default` converge to the wrong price for that
/// model — fixed by deriving the correct closed form directly from
/// `HestonFourier::chf` (see that method's doc). When a model's
/// `cumulants()` accuracy is unknown, cross-check the price under a couple
/// of `l` increases and confirm it stabilizes, the way
/// `cos_heston_matches_quadrature` does against an independent Gil-Pelaez
/// quadrature over the same `chf`, rather than trusting `Default` blindly.
#[derive(Debug, Clone)]
pub struct CosEngine {
  /// Number of cosine expansion terms.
  pub n: usize,
  /// Truncation half-width parameter (Fang-Oosterlee's `L`).
  pub l: f64,
}

impl Default for CosEngine {
  fn default() -> Self {
    Self { n: 256, l: 10.0 }
  }
}

impl CosEngine {
  /// Build a COS engine with `n` expansion terms and half-width `l`.
  ///
  /// # Panics
  ///
  /// Panics if `l <= 0.0` or `n < 2`: a non-positive truncation half-width
  /// or fewer than 2 expansion terms cannot span a meaningful density
  /// support, and would otherwise silently produce finite-looking garbage
  /// instead of a price.
  pub fn new(n: usize, l: f64) -> Self {
    assert!(l > 0.0 && n >= 2, "CosEngine requires l > 0 and n >= 2");
    Self { n, l }
  }

  /// Price a European call or put under `model`'s risk-neutral dynamics.
  ///
  /// `r` discounts the terminal payoff (`e^{-rt}`), matching
  /// [`super::GilPelaezPricer::price_call`] /
  /// [`super::LewisPricer::price_call`]. `q` is accepted for signature
  /// parity with those pricers but is not read here: the COS sum
  /// integrates the discounted payoff directly against `model.chf`, so any
  /// risk-neutral drift adjustment (dividend yield, jump compensator, …)
  /// must already be embedded in `model` — as it is for every in-tree
  /// [`FourierModelExt`] implementor.
  ///
  /// Returns [`f64::NAN`] when the truncation range degenerates
  /// (`c2 + sqrt(c4)` not finite and positive, e.g. at `t=0`) — the crate
  /// convention is `NaN` for an undefined result, never a silent `0.0`.
  /// `NaN` also propagates from a `model.chf` blow-up further down (e.g.
  /// characteristic-function overflow for extreme parameters): the finite
  /// sum is clamped at `0.0` from below (the truncated cosine series can
  /// leave machine-epsilon-scale negative noise on a deep out-of-the-money
  /// price whose true value is ~0, and a price can never be negative), but
  /// that clamp checks for `NaN` explicitly first rather than using
  /// `f64::max` directly — `f64::max` returns its non-`NaN` operand when
  /// one side is `NaN`, which would otherwise turn the blow-up into a
  /// silent `0.0` instead of the `NaN` this convention requires.
  pub fn price<M: FourierModelExt>(
    &self,
    model: &M,
    s0: f64,
    k: f64,
    r: f64,
    _q: f64,
    t: f64,
    option_type: OptionType,
  ) -> f64 {
    let cumulants = model.cumulants(t);
    let width_arg = cumulants.c2 + cumulants.c4.sqrt();
    if !(width_arg.is_finite() && width_arg > 0.0) {
      return f64::NAN;
    }

    let x = (s0 / k).ln();
    let half_width = self.l * width_arg.sqrt();
    let a = x + cumulants.c1 - half_width;
    let b = x + cumulants.c1 + half_width;
    let bma = b - a;

    let i_unit = Complex64::i();
    let mut price = 0.0;
    for k_idx in 0..self.n {
      let kf = k_idx as f64;
      let u = kf * PI / bma;

      let phi_y = model.chf(t, Complex64::new(u, 0.0)) * (i_unit * u * x).exp();
      let f_k = (phi_y * (-i_unit * u * a).exp()).re;

      let v_k = match option_type {
        OptionType::Call => call_coefficient(kf, a, b, bma, k),
        OptionType::Put => put_coefficient(kf, a, bma, k),
      };

      let weight = if k_idx == 0 { 0.5 } else { 1.0 };
      price += weight * f_k * v_k;
    }

    let discounted = (-r * t).exp() * price;
    if discounted.is_nan() {
      return f64::NAN;
    }
    discounted.max(0.0)
  }
}

/// Call payoff coefficient $V_k=\frac{2}{b-a}K(\chi_k(0,b)-\psi_k(0,b))$,
/// Fang-Oosterlee (2008) Eq. (29).
fn call_coefficient(kf: f64, a: f64, b: f64, bma: f64, strike: f64) -> f64 {
  let (chi, psi) = chi_psi(kf, a, bma, 0.0, b);
  (2.0 / bma) * strike * (chi - psi)
}

/// Put payoff coefficient $V_k=\frac{2}{b-a}K(\psi_k(a,0)-\chi_k(a,0))$,
/// Fang-Oosterlee (2008) Eq. (30).
fn put_coefficient(kf: f64, a: f64, bma: f64, strike: f64) -> f64 {
  let (chi, psi) = chi_psi(kf, a, bma, a, 0.0);
  (2.0 / bma) * strike * (psi - chi)
}

/// Fang-Oosterlee (2008) Eq. (22)-(23): cosine-series coefficients of $e^y$
/// (`chi`) and of the constant function `1` (`psi`) on `[c, d] ⊆ [a, b]`.
fn chi_psi(kf: f64, a: f64, bma: f64, c: f64, d: f64) -> (f64, f64) {
  if kf.abs() < 1e-14 {
    return (d.exp() - c.exp(), d - c);
  }

  let w = kf * PI / bma;
  let (sin_d, cos_d) = (w * (d - a)).sin_cos();
  let (sin_c, cos_c) = (w * (c - a)).sin_cos();

  let chi =
    (cos_d * d.exp() - cos_c * c.exp() + w * (sin_d * d.exp() - sin_c * c.exp())) / (1.0 + w * w);
  let psi = (sin_d - sin_c) / w;

  (chi, psi)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::pricing::bsm::BSMCoc;
  use crate::pricing::bsm::BSMPricer;
  use crate::pricing::fourier::BSMFourier;
  use crate::pricing::fourier::GilPelaezPricer;
  use crate::pricing::fourier::HestonFourier;
  use crate::pricing::fourier::KouFourier;
  use crate::pricing::heston::HestonPricer;
  use crate::traits::ModelPricer;

  /// Reference: Fang & Oosterlee (2008) §5.1. Verifies COS against the
  /// in-tree analytic BSM pricer to `3e-5` — not machine precision: that
  /// gap is the reference [`BSMPricer`]'s own `erf` approximation
  /// (Abramowitz & Stegun 7.1.26, ~1.5e-7 relative error, documented in
  /// `stochastic-rs-distributions/src/special.rs`), propagated through
  /// `S`/`K·disc` (~O(100)). COS's own truncation error at N=256, L=10 is
  /// independently machine-level — see `cos_converges_in_n`, and the `//`
  /// comment below for the cross-check that isolated the `erf` floor.
  #[test]
  fn cos_bsm_matches_analytic() {
    let model = BSMFourier {
      sigma: 0.25,
      r: 0.05,
      q: 0.0,
    };
    let expected = BSMPricer::new(0.25, BSMCoc::Bsm1973).price_call(100.0, 110.0, 0.05, 0.0, 1.0);
    let price = CosEngine::default().price(&model, 100.0, 110.0, 0.05, 0.0, 1.0, OptionType::Call);
    // `BSMPricer` goes through `norm_cdf`, whose `erf` (stochastic-rs-distributions
    // special.rs) is documented at "relative error ~1.5e-7" (Abramowitz & Stegun
    // 7.1.26). Scaled by S/K·disc (~O(100)) that is a ~1e-5 price-level floor,
    // independent of N/L: COS itself is machine-accurate here (cross-checked in
    // Python against a full-precision `erf` reference — diff 3e-14 already at
    // N=64), so 1e-8 would be testing `norm_cdf`'s approximation, not the COS
    // engine. `cos_converges_in_n` below is what pins the engine's own N-convergence.
    assert!(
      (price - expected).abs() < 3e-5,
      "COS BSM: got={price}, expected={expected}"
    );
  }

  #[test]
  fn cos_heston_matches_quadrature() {
    let model = HestonFourier {
      v0: 0.04,
      kappa: 1.5,
      theta: 0.04,
      sigma: 0.3,
      rho: -0.7,
      r: 0.05,
      q: 0.0,
    };
    let reference =
      HestonPricer::new(0.04, -0.7, 1.5, 0.04, 0.3, None).price_call(100.0, 100.0, 0.05, 0.0, 1.0);
    // History: `HestonFourier::cumulants` used to understate `c2` for this
    // parameter set (missing `v0` terms — see `cumulants`'s doc), so
    // `CosEngine::default`'s `L=10` truncation was too narrow and this test
    // used `CosEngine::new(256, 40.0)` to compensate. Now that `c2` is
    // correct, `L=40` is instead too *wide* for `N=256` — COS needs
    // `[a,b]` sized to the true density's support at a given `N`; an
    // unnecessarily wide range under-resolves the density and the price
    // degrades (empirically: `L=40,N=256` misses `reference` by 6.8e-4
    // once `c2` is correct, worse than `Default` below). `Default` is once
    // again the right choice, which was this test's original intent.
    let price = CosEngine::default().price(&model, 100.0, 100.0, 0.05, 0.0, 1.0, OptionType::Call);
    // Cross-check against an independent Gil-Pelaez quadrature over the
    // same `chf`: the two agree to ~1.25e-8, confirming COS(256, L=10) has
    // itself saturated (its own truncation is not the source of the gap to
    // `HestonPricer` below).
    let gil_pelaez = GilPelaezPricer::price_call(&model, 100.0, 100.0, 0.05, 0.0, 1.0);
    assert!(
      (price - gil_pelaez).abs() < 1e-6,
      "COS vs Gil-Pelaez (same chf): cos={price}, gil_pelaez={gil_pelaez}"
    );
    // The remaining ~1.28e-5 gap to `HestonPricer` is the two pricers'
    // independent characteristic-function/quadrature implementations
    // agreeing to their own tolerance floor, not a COS truncation error —
    // the identical floor `CosEngine::new(256, 40.0)` reached under the
    // old (understated) `c2`, before this fix.
    assert!(
      (price - reference).abs() < 2e-5,
      "COS Heston: got={price}, expected={reference}"
    );
  }

  #[test]
  fn cos_kou_matches_gil_pelaez() {
    let model = KouFourier {
      sigma: 0.15,
      lambda: 3.0,
      p_up: 0.2,
      eta1: 25.0,
      eta2: 10.0,
      r: 0.05,
      q: 0.01,
    };
    let reference = GilPelaezPricer::price_call(&model, 100.0, 100.0, 0.05, 0.01, 1.0);
    let price = CosEngine::default().price(&model, 100.0, 100.0, 0.05, 0.01, 1.0, OptionType::Call);
    assert!(
      (price - reference).abs() < 1e-4,
      "COS Kou: got={price}, expected={reference}"
    );
  }

  #[test]
  fn cos_put_call_parity() {
    let model = BSMFourier {
      sigma: 0.2,
      r: 0.05,
      q: 0.03,
    };
    let engine = CosEngine::default();
    let call = engine.price(&model, 100.0, 95.0, 0.05, 0.03, 0.5, OptionType::Call);
    let put = engine.price(&model, 100.0, 95.0, 0.05, 0.03, 0.5, OptionType::Put);
    let parity = 100.0 * (-0.03_f64 * 0.5).exp() - 95.0 * (-0.05_f64 * 0.5).exp();
    assert!(
      (call - put - parity).abs() < 1e-8,
      "put-call parity: call={call}, put={put}, parity={parity}"
    );
  }

  #[test]
  fn cos_converges_in_n() {
    let model = BSMFourier {
      sigma: 0.25,
      r: 0.05,
      q: 0.0,
    };
    let expected = BSMPricer::new(0.25, BSMCoc::Bsm1973).price_call(100.0, 110.0, 0.05, 0.0, 1.0);
    let err_small =
      (CosEngine::new(64, 10.0).price(&model, 100.0, 110.0, 0.05, 0.0, 1.0, OptionType::Call)
        - expected)
        .abs();
    let err_large =
      (CosEngine::new(512, 10.0).price(&model, 100.0, 110.0, 0.05, 0.0, 1.0, OptionType::Call)
        - expected)
        .abs();
    assert!(
      err_small >= err_large,
      "convergence in N: err(64)={err_small}, err(512)={err_large}"
    );
  }

  /// Degenerate maturity `t=0` collapses `c2` (and `c4`) to zero, so the
  /// truncation half-width is zero — the crate convention is `NaN`, never a
  /// silently-wrong `0.0` (see `carr_madan_out_of_grid_returns_nan_not_zero`
  /// in `super::super::tests` for the same convention elsewhere in this module).
  #[test]
  fn cos_invalid_cumulants_gives_nan() {
    let model = BSMFourier {
      sigma: 0.25,
      r: 0.05,
      q: 0.0,
    };
    let price = CosEngine::default().price(&model, 100.0, 100.0, 0.05, 0.0, 0.0, OptionType::Call);
    assert!(
      price.is_nan(),
      "degenerate t=0 cumulants must give NaN, got {price}"
    );
  }

  #[test]
  #[should_panic(expected = "CosEngine requires l > 0 and n >= 2")]
  fn cos_engine_new_rejects_nonpositive_l() {
    CosEngine::new(256, 0.0);
  }

  #[test]
  #[should_panic(expected = "CosEngine requires l > 0 and n >= 2")]
  fn cos_engine_new_rejects_too_few_terms() {
    CosEngine::new(1, 10.0);
  }

  /// A characteristic function that returns `NaN` (standing in for a
  /// user-supplied model overflowing its own `chf`) must exit `price` as
  /// `NaN`, not the `0.0` `f64::max` would silently produce by picking its
  /// non-`NaN` operand — see [`CosEngine::price`]'s doc.
  #[test]
  fn cos_price_preserves_nan_from_chf_blowup() {
    struct NanModel;
    impl FourierModelExt for NanModel {
      fn chf(&self, _t: f64, _xi: Complex64) -> Complex64 {
        Complex64::new(f64::NAN, f64::NAN)
      }
      fn cumulants(&self, _t: f64) -> super::super::Cumulants {
        super::super::Cumulants {
          c1: 0.0,
          c2: 0.04,
          c4: 0.0,
        }
      }
    }
    let price =
      CosEngine::default().price(&NanModel, 100.0, 100.0, 0.05, 0.0, 1.0, OptionType::Call);
    assert!(
      price.is_nan(),
      "NaN chf must propagate to a NaN price, got {price}"
    );
  }
}

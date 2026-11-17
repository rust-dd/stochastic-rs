//! Fang–Oosterlee COS pricing of any [`FourierModelExt`] model in `y = ln(S_T/K)` from
//! `x = ln(F_T/K)`: `chf(t, u)·e^{iux}` is their `φ(u; x)`, and Eqs. (22)–(23), (29)–(30) follow.
//!
//! Reference: Fang, F. & Oosterlee, C.W. (2008), "A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions", SIAM J. Sci. Comput. 31(2), 826–848, DOI 10.1137/080718061

use std::f64::consts::PI;

use num_complex::Complex64;

use super::FourierModelExt;
use crate::OptionType;

/// Fang–Oosterlee (2008) COS pricer for any [`FourierModelExt`]: `n` terms (default 256) over `l`
/// (default 10, §5.1) cumulant standard deviations, so understated `cumulants()` misprice silently.
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

  /// `r` discounts and `r − q` carries the forward the drift-free `chf` is centred on; a call is
  /// the put plus `S e^{−qt} − K e^{−rt}`; NaN where `truncation_half_width` or `chf` is.
  pub fn price<M: FourierModelExt>(
    &self,
    model: &M,
    s0: f64,
    k: f64,
    r: f64,
    q: f64,
    t: f64,
    option_type: OptionType,
  ) -> f64 {
    let cumulants = model.cumulants(t);
    let half_width = cumulants.truncation_half_width(self.l);
    if !half_width.is_finite() {
      return f64::NAN;
    }

    let x = (s0 / k).ln() + (r - q) * t;
    let a = x + cumulants.c1 - half_width;
    let b = x + cumulants.c1 + half_width;
    let bma = b - a;

    let i_unit = Complex64::i();
    let mut put = 0.0;
    for k_idx in 0..self.n {
      let kf = k_idx as f64;
      let u = kf * PI / bma;

      let phi_y = model.chf(t, Complex64::new(u, 0.0)) * (i_unit * u * x).exp();
      let f_k = (phi_y * (-i_unit * u * a).exp()).re;

      let weight = if k_idx == 0 { 0.5 } else { 1.0 };
      put += weight * f_k * put_coefficient(kf, a, b, bma, k);
    }

    let put = (-r * t).exp() * put;
    // The call's coefficients grow like `e^b`, the put's stay below `K`: Le Floc'h (2020), §3.
    let price = match option_type {
      OptionType::Call => put + s0 * (-q * t).exp() - k * (-r * t).exp(),
      OptionType::Put => put,
    };
    if price.is_nan() {
      return f64::NAN;
    }
    price.max(0.0)
  }
}

/// Put payoff coefficient $V_k$ on $[a,b]\cap(-\infty,0]$, Fang-Oosterlee (2008) Eq. (30)
/// for $a<0<b$.
fn put_coefficient(kf: f64, a: f64, b: f64, bma: f64, strike: f64) -> f64 {
  if a >= 0.0 {
    return 0.0;
  }
  let (chi, psi) = chi_psi(kf, a, bma, a, b.min(0.0));
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
  use crate::pricing::bsm::BsmCoc;
  use crate::pricing::bsm::BsmPricer;
  use crate::pricing::fourier::BsmFourier;
  use crate::pricing::fourier::GilPelaezPricer;
  use crate::pricing::fourier::HestonFourier;
  use crate::pricing::fourier::KouFourier;
  use crate::pricing::heston::HestonParams;
  use crate::pricing::heston::HestonPricer;
  use crate::traits::ModelPricer;

  /// Fang & Oosterlee (2008) §5.1: COS at `N = 256, L = 10` reproduces the closed-form BSM call.
  #[test]
  fn cos_bsm_matches_analytic() {
    let model = BsmFourier { sigma: 0.25 };
    let expected = BsmPricer::new(0.25, BsmCoc::Bsm1973).price_call(100.0, 110.0, 0.05, 0.0, 1.0);
    let price = CosEngine::default().price(&model, 100.0, 110.0, 0.05, 0.0, 1.0, OptionType::Call);
    assert!(
      (price - expected).abs() < 1e-10,
      "COS BSM: got={price}, expected={expected}"
    );
  }

  #[test]
  fn cos_heston_matches_quadrature() {
    let model = HestonFourier::new(HestonParams {
      v0: 0.04,
      kappa: 1.5,
      theta: 0.04,
      sigma: 0.3,
      rho: -0.7,
    });
    let reference = HestonPricer::new(HestonParams {
      v0: 0.04,
      kappa: 1.5,
      theta: 0.04,
      sigma: 0.3,
      rho: -0.7,
    })
    .price_call(100.0, 100.0, 0.05, 0.0, 1.0);
    // A wider `L` under-resolves the density at `N = 256` (`L = 40` misses by 1.5e-4).
    let price = CosEngine::default().price(&model, 100.0, 100.0, 0.05, 0.0, 1.0, OptionType::Call);
    let gil_pelaez = GilPelaezPricer::price_call(&model, 100.0, 100.0, 0.05, 0.0, 1.0);
    assert!(
      (price - gil_pelaez).abs() < 1e-6,
      "COS vs Gil-Pelaez (same chf): cos={price}, gil_pelaez={gil_pelaez}"
    );
    assert!(
      (price - reference).abs() < 1e-6,
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
    let model = BsmFourier { sigma: 0.2 };
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
    let model = BsmFourier { sigma: 0.25 };
    let expected = BsmPricer::new(0.25, BsmCoc::Bsm1973).price_call(100.0, 110.0, 0.05, 0.0, 1.0);
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
    let model = BsmFourier { sigma: 0.25 };
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
          c4: Some(0.0),
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

  /// Le Floc'h (2020), arXiv:2005.13248, §3: a call's coefficients grow like `e^b`, so over a wide
  /// range a call stays at the closed form only through the put and parity.
  #[test]
  fn cos_call_holds_over_a_wide_truncation_range() {
    let model = BsmFourier { sigma: 0.2 };
    let bsm = BsmPricer::new(0.2, BsmCoc::Merton1973);
    let engine = CosEngine::new(4096, 100.0);
    for k in [80.0, 100.0, 120.0] {
      let call = engine.price(&model, 100.0, k, 0.05, 0.01, 1.0, OptionType::Call);
      let want = bsm.price_call(100.0, k, 0.05, 0.01, 1.0);
      assert!((call - want).abs() < 1e-10, "K={k}: {call} vs {want}");
    }
  }

  /// Where 0 falls outside [a, b] the payoff coefficients must clamp to the payoff's support.
  #[test]
  fn cos_matches_closed_form_at_wing_strikes() {
    let model = BsmFourier { sigma: 0.2 };
    let bsm = BsmPricer::new(0.2, BsmCoc::Merton1973);
    let engine = CosEngine::default();
    for k in [40.0, 250.0] {
      let call = engine.price(&model, 100.0, k, 0.05, 0.01, 0.05, OptionType::Call);
      let put = engine.price(&model, 100.0, k, 0.05, 0.01, 0.05, OptionType::Put);
      let (want_c, want_p) = (
        bsm.price_call(100.0, k, 0.05, 0.01, 0.05),
        bsm.price_put(100.0, k, 0.05, 0.01, 0.05),
      );
      assert!(
        (call - want_c).abs() < 1e-9,
        "K={k} call: {call} vs {want_c}"
      );
      assert!((put - want_p).abs() < 1e-9, "K={k} put: {put} vs {want_p}");
    }
  }
}

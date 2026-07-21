//! # Fourier
//!
//! $$
//! C(K)=\frac{e^{-\alpha k}}{\pi}\int_0^\infty e^{-ivk}\,\psi_T(v)\,dv,\quad
//! \psi_T(v)=\frac{e^{-rT}\,\phi_T(v-(1+\alpha)i)}{\alpha^2+\alpha-v^2+i(2\alpha+1)v}
//! $$
//!
//! Source:
//! - Carr, P. & Madan, D. (1999), "Option valuation using the fast Fourier transform"
//! - Lewis, A. (2001), "A Simple Option Formula for General Jump-Diffusion and Other
//!   Exponential Lévy Processes"
//! - Gil-Pelaez, J. (1951), "Note on the inversion theorem"

use num_complex::Complex64;

mod bsm;
mod frft;
mod heston;
mod hybrid;
mod levy;
mod pricer;

pub use bsm::BSMFourier;
pub use frft::FrftCarrMadanPricer;
pub use frft::frft;
pub use heston::DoubleHestonFourier;
pub use heston::HestonFourier;
pub use hybrid::BatesFourier;
pub use hybrid::HKDEFourier;
pub use levy::CGMYFourier;
pub use levy::KouFourier;
pub use levy::MertonJDFourier;
pub use levy::NigFourier;
pub use levy::VarianceGammaFourier;
pub use pricer::CarrMadanPricer;
pub use pricer::GilPelaezPricer;
pub use pricer::LewisPricer;

/// Cumulants of the log-price distribution.
#[derive(Debug, Clone)]
pub struct Cumulants {
  /// First cumulant (mean of log-return).
  pub c1: f64,
  /// Second cumulant (variance of log-return).
  pub c2: f64,
  /// Fourth cumulant.
  pub c4: f64,
}

/// Trait for models that expose a characteristic function of the log-price,
/// enabling pricing via Fourier inversion.
pub trait FourierModelExt {
  /// Characteristic function of the log-price $\ln(S_T/S_0)$:
  /// $\phi_T(\xi) = E[e^{i\xi \ln(S_T/S_0)}]$.
  ///
  /// Must already include the risk-neutral drift, i.e.\ the expectation is
  /// under the risk-neutral measure $\mathbb{Q}$.
  fn chf(&self, t: f64, xi: Complex64) -> Complex64;

  /// Cumulants of the log-return for maturity `t`.
  fn cumulants(&self, t: f64) -> Cumulants;
}

/// Blanket implementation: every [`FourierModelExt`] model automatically
/// implements [`crate::traits::ModelPricer`] via Gil-Pelaez quadrature.
impl<T: FourierModelExt> crate::traits::ModelPricer for T {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    GilPelaezPricer::price_call(self, s, k, r, q, tau)
  }
}

/// Compute $\Gamma(-Y)$ via the reflection formula.
pub fn gamma_neg_y_pub(y: f64) -> f64 {
  gamma_neg_y(y)
}

pub(crate) fn gamma_neg_y(y: f64) -> f64 {
  if y.abs() < 1e-8 {
    return 1e15;
  }
  if y < 0.0 {
    return stochastic_rs_distributions::special::gamma(-y);
  }
  if (y - 1.0).abs() < 1e-8 {
    return stochastic_rs_distributions::special::gamma(-0.999);
  }
  let g = stochastic_rs_distributions::special::gamma(y);
  let sin_val = (std::f64::consts::PI * y).sin();
  if sin_val.abs() < 1e-8 || g.abs() < 1e-8 {
    return 1e15;
  }
  -std::f64::consts::PI / (y * sin_val * g)
}

#[cfg(test)]
mod tests {
  use super::*;

  const TOL_FOURIER: f64 = 0.15;
  const TOL_TIGHT: f64 = 0.05;

  /// Gil-Pelaez and Lewis both integrated to a fixed `φ_max = 100`, which
  /// truncates the slowly-decaying tail of the short-dated Heston integrand.
  /// Pre-fix at τ=0.005/ATM the price was 0.488 vs a converged 0.577, and at
  /// τ=0.01/K=105 it was 0.0100 vs 0.0030 (over 3× too large). Both must now
  /// match the converged inversion and agree with each other.
  #[test]
  fn fourier_pricers_match_converged_short_dated() {
    let model = HestonFourier {
      v0: 0.04,
      kappa: 1.5,
      theta: 0.04,
      sigma: 0.30,
      rho: -0.7,
      r: 0.05,
      q: 0.0,
    };
    // (K, τ, converged call)
    let cases = [
      (100.0, 0.005, 0.576581),
      (105.0, 0.01, 0.002966),
      (95.0, 0.01, 5.052617),
    ];
    for (k, t, expected) in cases {
      let gp = GilPelaezPricer::price_call(&model, 100.0, k, 0.05, 0.0, t);
      let lw = LewisPricer::price_call(&model, 100.0, k, 0.05, 0.0, t);
      assert!(
        (gp - expected).abs() < 1e-3,
        "Gil-Pelaez at K={k}, τ={t}: got {gp}, converged {expected}"
      );
      assert!(
        (lw - expected).abs() < 1e-3,
        "Lewis at K={k}, τ={t}: got {lw}, converged {expected}"
      );
      assert!(
        (gp - lw).abs() < 1e-3,
        "Gil-Pelaez {gp} and Lewis {lw} must agree"
      );
    }
  }

  #[test]
  fn carr_madan_bsm_reference() {
    let model = BSMFourier {
      sigma: 0.15,
      r: 0.05,
      q: 0.01,
    };
    let pricer = CarrMadanPricer::default();
    let price = pricer.price_call(&model, 100.0, 100.0, 0.05, 1.0);
    let expected = 7.94871378854164;
    assert!(
      (price - expected).abs() < TOL_FOURIER,
      "Carr-Madan BSM: got={price}, expected={expected}"
    );
  }

  /// Regression: out-of-grid strikes must NOT silently return `0.0`. The
  /// rc.0 behavior was a hard `0.0` fallback whenever `ln(k)` fell outside
  /// `[log_strikes.first(), log_strikes.last()]` — that produced silent-zero
  /// residuals at the wings of any calibration that stretched past the FFT
  /// grid. rc.1 returns `f64::NAN` instead, so calibration objectives are
  /// poisoned (forcing the user to detect / fix) rather than silently fitting.
  #[test]
  fn carr_madan_out_of_grid_returns_nan_not_zero() {
    let model = BSMFourier {
      sigma: 0.15,
      r: 0.05,
      q: 0.0,
    };
    let pricer = CarrMadanPricer::default();
    let s = 100.0;
    let r = 0.05;
    let t = 1.0;

    let p_itm = pricer.price_call(&model, s, 1e-9, r, t);
    assert!(
      p_itm.is_nan(),
      "Out-of-grid deep-ITM must return NaN to flag truncation, got {p_itm}"
    );

    let p_otm = pricer.price_call(&model, s, 1e9, r, t);
    assert!(
      p_otm.is_nan(),
      "Out-of-grid deep-OTM must return NaN to flag truncation, got {p_otm}"
    );

    let p_atm = pricer.price_call(&model, s, 100.0, r, t);
    assert!(p_atm.is_finite() && p_atm > 0.0);

    assert!(!pricer.strike_in_grid(&model, s, 1e-9, r, t));
    assert!(!pricer.strike_in_grid(&model, s, 1e9, r, t));
    assert!(pricer.strike_in_grid(&model, s, 100.0, r, t));
  }

  #[test]
  fn lewis_bsm_reference() {
    let model = BSMFourier {
      sigma: 0.15,
      r: 0.05,
      q: 0.01,
    };
    let price = LewisPricer::price_call(&model, 100.0, 100.0, 0.05, 0.01, 1.0);
    let expected = 7.94871378854164;
    assert!(
      (price - expected).abs() < TOL_FOURIER,
      "Lewis BSM: got={price}, expected={expected}"
    );
  }

  #[test]
  fn gil_pelaez_bsm_reference() {
    let model = BSMFourier {
      sigma: 0.15,
      r: 0.05,
      q: 0.01,
    };
    let price = GilPelaezPricer::price_call(&model, 100.0, 100.0, 0.05, 0.01, 1.0);
    let expected = 7.94871378854164;
    assert!(
      (price - expected).abs() < TOL_FOURIER,
      "Gil-Pelaez BSM: got={price}, expected={expected}"
    );
  }

  #[test]
  fn carr_madan_vg_reference() {
    let model = VarianceGammaFourier {
      sigma: 0.2,
      theta: 0.1,
      nu: 0.85,
      r: 0.05,
      q: 0.01,
    };
    let pricer = CarrMadanPricer::default();
    let price = pricer.price_call(&model, 100.0, 100.0, 0.05, 1.0);
    let expected = 10.13935062748614;
    assert!(
      (price - expected).abs() < TOL_FOURIER,
      "Carr-Madan Vg: got={price}, expected={expected}"
    );
  }

  #[test]
  fn carr_madan_cgmy_reference() {
    let model = CGMYFourier {
      c: 0.02,
      g: 5.0,
      m: 15.0,
      y: 1.2,
      r: 0.05,
      q: 0.01,
    };
    let pricer = CarrMadanPricer::default();
    let price = pricer.price_call(&model, 100.0, 100.0, 0.05, 1.0);
    let expected = 5.80222163947386;
    assert!(
      (price - expected).abs() < TOL_FOURIER,
      "Carr-Madan Cgmy: got={price}, expected={expected}"
    );
  }

  #[test]
  fn carr_madan_merton_reference() {
    let model = MertonJDFourier {
      sigma: 0.12,
      lambda: 0.4,
      mu_j: -0.12,
      sigma_j: 0.18,
      r: 0.05,
      q: 0.01,
    };
    let pricer = CarrMadanPricer::default();
    let price = pricer.price_call(&model, 100.0, 100.0, 0.05, 1.0);
    let expected = 8.675684635426279;
    assert!(
      (price - expected).abs() < TOL_FOURIER,
      "Carr-Madan MJD: got={price}, expected={expected}"
    );
  }

  #[test]
  fn carr_madan_kou_reference() {
    let model = KouFourier {
      sigma: 0.15,
      lambda: 3.0,
      p_up: 0.2,
      eta1: 25.0,
      eta2: 10.0,
      r: 0.05,
      q: 0.01,
    };
    let pricer = CarrMadanPricer::default();
    let price = pricer.price_call(&model, 100.0, 100.0, 0.05, 1.0);
    let expected = 11.92430307601936;
    assert!(
      (price - expected).abs() < TOL_FOURIER,
      "Carr-Madan Kou: got={price}, expected={expected}"
    );
  }

  #[test]
  fn carr_madan_heston_vs_analytical() {
    let model = HestonFourier {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma: 0.1,
      rho: -0.6,
      r: 0.05,
      q: 0.0,
    };
    let pricer = CarrMadanPricer::default();
    let price = pricer.price_call(&model, 100.0, 100.0, 0.05, 1.0);
    assert!(
      (price - 10.474).abs() < 0.2,
      "Carr-Madan Heston: got={price}, expected≈10.474"
    );
  }

  /// Lord-Kahl 2007 cumulant-based grid sizing should produce a pricer that
  /// agrees with the default symmetric grid for moderate parameters, while
  /// remaining configurable for skewed/long-maturity regimes.
  #[test]
  fn carr_madan_cumulant_sized_constructor() {
    let model = HestonFourier {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma: 0.3,
      rho: -0.7,
      r: 0.05,
      q: 0.0,
    };
    let cum_pricer = CarrMadanPricer::cumulant_sized(&model, 1.0, 100.0, 12.0);
    assert!(cum_pricer.n.is_power_of_two(), "n must be power of two");
    assert!(cum_pricer.n >= 4096, "n must be at least default 4096");
    let price = cum_pricer.price_call(&model, 100.0, 100.0, 0.05, 1.0);
    assert!(
      price.is_finite() && price > 0.0,
      "cumulant_sized Carr-Madan must produce positive finite price, got {price}"
    );
    let default_pricer = CarrMadanPricer::default();
    let default_price = default_pricer.price_call(&model, 100.0, 100.0, 0.05, 1.0);
    assert!(
      (price - default_price).abs() < 1.0,
      "cumulant_sized vs default Heston ATM should agree to within 1.0: cum={price}, def={default_price}"
    );
  }

  /// Lord-Kahl 2010 §3.2 cumulant-based half-width sizing: `cumulant_sized`
  /// must produce a grid that brackets `ln S_0` plus a buffer scaled by
  /// $L_{\text{factor}}\sqrt{|c_2|+\sqrt{|c_4|}}$. For `s = 100, c2 = v0·T = 0.04`
  /// and `c4` small, the half-width is `ln(100) + 12·sqrt(0.04+ε) ≈ 7.05`,
  /// covering ln(K) for K in roughly `[exp(-7), exp(+7)] ≈ [0.9e-3, 1100]`.
  #[test]
  fn carr_madan_cumulant_sized_grid_brackets_log_spot() {
    let model = HestonFourier {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma: 0.3,
      rho: -0.7,
      r: 0.05,
      q: 0.0,
    };
    let s = 100.0_f64;
    let t = 1.0_f64;
    let pricer = CarrMadanPricer::cumulant_sized(&model, t, s, 12.0);
    let (log_strikes, _) = pricer.price_call_surface(&model, s, 0.05, t);
    let half_width = (log_strikes[log_strikes.len() - 1] - log_strikes[0]) / 2.0;
    let cumulants = model.cumulants(t);
    let c4_term = if cumulants.c4.is_finite() && cumulants.c4 >= 0.0 {
      cumulants.c4.sqrt()
    } else {
      0.0
    };
    let expected_buffer = 12.0 * (cumulants.c2.abs() + c4_term).sqrt();
    let expected_half_width = s.ln().abs() + expected_buffer;
    assert!(
      (half_width - expected_half_width).abs() < 0.1,
      "FFT half-width {half_width} must match Lord-Kahl rule |ln(s)|+L_factor·sqrt(|c2|+sqrt|c4|) = {expected_half_width}"
    );
    let ln_s = s.ln();
    assert!(
      log_strikes[0] <= ln_s && ln_s <= log_strikes[log_strikes.len() - 1],
      "ln(s)={ln_s} must be inside the FFT grid [{}, {}]",
      log_strikes[0],
      log_strikes[log_strikes.len() - 1]
    );
  }

  /// Lord-Kahl §3.2 regression: a long-maturity / high-|ρ| / high-skew Heston
  /// must produce a finite, positive, accurate ATM price under the
  /// `cumulant_sized` constructor (the meaningful audit fix). The reference
  /// is the Gil-Pelaez quadrature pricer (independent of the FFT grid). We
  /// use a moderately tight tolerance — both Carr-Madan variants converge
  /// to the same analytic price, the audit's concern was that the legacy
  /// fixed-η grid silently truncates the wings on skewed / long-dated
  /// distributions.
  #[test]
  fn carr_madan_cumulant_sized_finite_positive_long_maturity_skew() {
    let model = HestonFourier {
      v0: 0.04,
      kappa: 1.0,
      theta: 0.04,
      sigma: 0.5,
      rho: -0.85,
      r: 0.05,
      q: 0.0,
    };
    let s = 100.0_f64;
    let t = 5.0_f64;
    let r = 0.05_f64;
    let k = 100.0_f64;

    let reference = GilPelaezPricer::price_call(&model, s, k, r, 0.0, t);

    let cum = CarrMadanPricer::cumulant_sized(&model, t, s, 12.0).price_call(&model, s, k, r, t);

    assert!(
      cum.is_finite() && cum > 0.0 && cum < s,
      "cumulant_sized price must be finite, positive, and bounded by S at T=5y, ρ=-0.85, got {cum}"
    );
    assert!(
      (cum - reference).abs() < 5.0,
      "cumulant_sized must track Gil-Pelaez within 5.0 at T=5y, ρ=-0.85: cum={cum}, ref={reference}"
    );
  }

  #[test]
  fn barrier_down_and_out_call_vs_haug() {
    use crate::OptionType;
    use crate::pricing::barrier::BarrierPricer;
    use crate::pricing::barrier::BarrierType;
    let p = BarrierPricer {
      s: 100.0,
      k: 100.0,
      h: 90.0,
      r: 0.05,
      q: 0.0,
      sigma: 0.2,
      t: 1.0,
      rebate: 0.0,
      barrier_type: BarrierType::DownAndOut,
      option_type: OptionType::Call,
    };
    let price = p.price();
    let expected = 8.66547165824566;
    assert!(
      (price - expected).abs() < TOL_TIGHT,
      "Barrier DO call: got={price}, expected={expected}"
    );
  }

  #[test]
  fn hkde_zero_jumps_equals_heston() {
    let heston = HestonFourier {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma: 0.1,
      rho: -0.6,
      r: 0.05,
      q: 0.0,
    };
    let hkde = HKDEFourier {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma_v: 0.1,
      rho: -0.6,
      r: 0.05,
      q: 0.0,
      lam: 0.0,
      p_up: 0.5,
      eta1: 10.0,
      eta2: 10.0,
    };
    let pricer = CarrMadanPricer::default();
    let heston_price = pricer.price_call(&heston, 100.0, 100.0, 0.05, 1.0);
    let hkde_price = pricer.price_call(&hkde, 100.0, 100.0, 0.05, 1.0);
    assert!(
      (heston_price - hkde_price).abs() < TOL_TIGHT,
      "Hkde(lam=0) vs Heston: hkde={hkde_price}, heston={heston_price}"
    );
  }

  #[test]
  fn hkde_prices_positive() {
    let model = HKDEFourier {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma_v: 0.3,
      rho: -0.7,
      r: 0.05,
      q: 0.01,
      lam: 3.0,
      p_up: 0.4,
      eta1: 10.0,
      eta2: 5.0,
    };
    let pricer = CarrMadanPricer::default();
    let price = pricer.price_call(&model, 100.0, 100.0, 0.05, 1.0);
    assert!(
      price > 0.0,
      "Hkde call price should be positive, got={price}"
    );
  }

  #[test]
  fn hkde_call_decreases_with_strike() {
    let model = HKDEFourier {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma_v: 0.3,
      rho: -0.7,
      r: 0.05,
      q: 0.01,
      lam: 3.0,
      p_up: 0.4,
      eta1: 10.0,
      eta2: 5.0,
    };
    let pricer = CarrMadanPricer::default();
    let p1 = pricer.price_call(&model, 100.0, 90.0, 0.05, 1.0);
    let p2 = pricer.price_call(&model, 100.0, 100.0, 0.05, 1.0);
    let p3 = pricer.price_call(&model, 100.0, 110.0, 0.05, 1.0);
    assert!(
      p1 > p2 && p2 > p3,
      "Hkde call should decrease with strike: p(90)={p1}, p(100)={p2}, p(110)={p3}"
    );
  }

  #[test]
  fn hkde_lewis_vs_carr_madan() {
    let model = HKDEFourier {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma_v: 0.3,
      rho: -0.7,
      r: 0.05,
      q: 0.01,
      lam: 1.0,
      p_up: 0.4,
      eta1: 10.0,
      eta2: 5.0,
    };
    let cm_price = CarrMadanPricer::default().price_call(&model, 100.0, 100.0, 0.05, 1.0);
    let lw_price = LewisPricer::price_call(&model, 100.0, 100.0, 0.05, 0.01, 1.0);
    assert!(
      (cm_price - lw_price).abs() < TOL_FOURIER,
      "Hkde Lewis vs Carr-Madan: lewis={lw_price}, cm={cm_price}"
    );
  }

  /// Long-maturity / high-|ρ| regression: `HestonFourier::chf` must use the
  /// Albrecher-Mayer-Schoutens-Tistaert (2007) "Little Heston Trap" form
  /// (`g̃ = 1/g_original`, `exp(-d·t)`). Original Heston (1993) form develops a
  /// branch-cut discontinuity at T = 5y, ρ = -0.9; the Trap form does not.
  #[test]
  fn heston_fourier_little_trap_long_maturity_high_rho() {
    let model = HestonFourier {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma: 0.3,
      rho: -0.9,
      r: 0.05,
      q: 0.0,
    };
    let call = LewisPricer::price_call(&model, 100.0, 100.0, 0.05, 0.0, 5.0);
    assert!(
      call.is_finite() && call > 0.0 && call < 100.0,
      "HestonFourier Trap form: finite positive bounded call required at T=5y, ρ=-0.9, got {call}"
    );
  }

  /// Same long-maturity / high-|ρ| stress for `DoubleHestonFourier::factor_cd` —
  /// both factors must stay on the principal log-branch.
  #[test]
  fn double_heston_fourier_little_trap_long_maturity_high_rho() {
    let model = DoubleHestonFourier {
      v1_0: 0.04,
      kappa1: 2.0,
      theta1: 0.04,
      sigma1: 0.3,
      rho1: -0.9,
      v2_0: 0.02,
      kappa2: 1.0,
      theta2: 0.03,
      sigma2: 0.25,
      rho2: -0.85,
      r: 0.05,
      q: 0.0,
    };
    let call = LewisPricer::price_call(&model, 100.0, 100.0, 0.05, 0.0, 5.0);
    assert!(
      call.is_finite() && call > 0.0 && call < 100.0,
      "DoubleHestonFourier Trap form: finite positive bounded call required at T=5y, ρ_j=-0.9/-0.85, got {call}"
    );
  }

  /// Same long-maturity / high-|ρ| stress for `HKDEFourier::chf` — the Heston
  /// diffusion part must stay on the principal log-branch under the Kou jump CF.
  #[test]
  fn hkde_fourier_little_trap_long_maturity_high_rho() {
    let model = HKDEFourier {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma_v: 0.3,
      rho: -0.9,
      r: 0.05,
      q: 0.0,
      lam: 1.5,
      p_up: 0.4,
      eta1: 10.0,
      eta2: 5.0,
    };
    let call = LewisPricer::price_call(&model, 100.0, 100.0, 0.05, 0.0, 5.0);
    assert!(
      call.is_finite() && call > 0.0 && call < 100.0,
      "HKDEFourier Trap form: finite positive bounded call required at T=5y, ρ=-0.9, got {call}"
    );
  }

  /// Same long-maturity / high-|ρ| stress for `BatesFourier::chf` — the Heston
  /// diffusion part must stay on the principal log-branch under the Merton
  /// log-normal jump CF.
  #[test]
  fn bates_fourier_little_trap_long_maturity_high_rho() {
    let model = BatesFourier {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma_v: 0.3,
      rho: -0.9,
      lambda: 0.3,
      mu_j: -0.05,
      sigma_j: 0.15,
      r: 0.05,
      q: 0.0,
    };
    let call = LewisPricer::price_call(&model, 100.0, 100.0, 0.05, 0.0, 5.0);
    assert!(
      call.is_finite() && call > 0.0 && call < 100.0,
      "BatesFourier Trap form: finite positive bounded call required at T=5y, ρ=-0.9, got {call}"
    );
  }
}

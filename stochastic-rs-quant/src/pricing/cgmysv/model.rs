//! CGMYSV model characteristic function and Fourier integration support.
//!
//! $$
//! \phi_{L_t}(u) = \Phi_t\!\bigl(\psi_{\mathrm{stdCGMY}}(u),\;\rho\,u,\;v_0\bigr)
//! $$
//!
//! Reference: Kim, Y. S. (2021), arXiv:2101.11001, Sections 2–3.

use std::f64::consts::PI;

use num_complex::Complex64;
use stochastic_rs_distributions::special::gamma;

use crate::pricing::fourier::Cumulants;
use crate::pricing::fourier::FourierModelExt;

/// Parameters for the CGMYSV process (Eq. 4).
///
/// Combines a standard Cgmy jump process $Z_t$ with a Cir stochastic
/// time change $\mathcal{V}_t = \int_0^t v_s\,ds$.
#[derive(Debug, Clone)]
pub struct CgmysvParams {
  /// Activity parameter $\alpha \in (0,2)$ (Y in Cgmy notation).
  pub alpha: f64,
  /// Positive tempering $\lambda_+ > 0$ (G in Cgmy notation).
  pub lambda_plus: f64,
  /// Negative tempering $\lambda_- > 0$ (M in Cgmy notation).
  pub lambda_minus: f64,
  /// Cir mean reversion rate $\kappa > 0$.
  pub kappa: f64,
  /// Cir long-run mean $\eta > 0$.
  pub eta: f64,
  /// Cir volatility of volatility $\zeta > 0$.
  pub zeta: f64,
  /// Coupling parameter $\rho$.
  pub rho: f64,
  /// Initial variance $v_0 > 0$.
  pub v0: f64,
}

/// CGMYSV model for Fourier-based option pricing.
///
/// Implements [`FourierModelExt`] to enable pricing via Carr–Madan FFT,
/// Gil–Pelaez quadrature, or Lewis formula.
#[derive(Debug, Clone)]
pub struct CgmysvModel {
  pub params: CgmysvParams,
}

impl CgmysvParams {
  /// Standard Cgmy normalising constant
  /// $C = \bigl(\Gamma(2-\alpha)(\lambda_+^{\alpha-2}+\lambda_-^{\alpha-2})\bigr)^{-1}$.
  pub fn norm_const(&self) -> f64 {
    let g2a = gamma(2.0 - self.alpha);
    1.0
      / (g2a * (self.lambda_plus.powf(self.alpha - 2.0) + self.lambda_minus.powf(self.alpha - 2.0)))
  }

  /// Lévy symbol of the standard Cgmy distribution $\psi_{\mathrm{stdCGMY}}(u)$ (Eq. 3).
  ///
  /// $$
  /// \psi(u) = \frac{\lambda_+^{\alpha-1}-\lambda_-^{\alpha-1}}{(\alpha-1)\,D}\,iu +
  ///   \frac{(\lambda_+-iu)^\alpha - \lambda_+^\alpha + (\lambda_-+iu)^\alpha - \lambda_-^\alpha}
  ///   {\alpha(\alpha-1)\,D}
  /// $$
  pub fn psi_std_cgmy(&self, u: Complex64) -> Complex64 {
    let a = self.alpha;
    let lp = self.lambda_plus;
    let lm = self.lambda_minus;
    let i = Complex64::i();
    let denom_base = lp.powf(a - 2.0) + lm.powf(a - 2.0);

    let drift = (lp.powf(a - 1.0) - lm.powf(a - 1.0)) / ((a - 1.0) * denom_base) * i * u;

    let jumps = ((Complex64::from(lp) - i * u).powf(a) - lp.powf(a)
      + (Complex64::from(lm) + i * u).powf(a)
      - lm.powf(a))
      / (a * (a - 1.0) * denom_base);

    drift + jumps
  }

  /// CIR joint chf $E[e^{ia\mathcal V_t + ib v_t} \mid v_0 = x]$ (Eq. 2) as one exponential over
  /// `e^{−γt}`, which neither overflows at a small `ζ` nor leaves the principal log branch.
  /// Albrecher, Mayer, Schoutens & Tistaert (2007), "The Little Heston Trap", Wilmott, 83–92.
  pub fn cir_joint_chf(&self, t: f64, a: Complex64, b: Complex64, x: f64) -> Complex64 {
    let kappa = self.kappa;
    let zeta2 = self.zeta * self.zeta;
    let i = Complex64::i();

    let gamma = (Complex64::from(kappa * kappa) - 2.0 * zeta2 * i * a).sqrt();
    // κ − γ rationalised: the difference itself cancels to rounding as ζ → 0.
    let kappa_less_gamma = 2.0 * zeta2 * i * a / (kappa + gamma);
    let kappa_b = kappa - i * b * zeta2;
    let g = (kappa_less_gamma - i * b * zeta2) / (kappa_b + gamma);
    let e = (-gamma * t).exp();

    let ln_a =
      kappa * self.eta / zeta2 * (kappa_less_gamma * t - 2.0 * ((1.0 - g * e) / (1.0 - g)).ln());
    let b_num = i * b * (gamma * (1.0 + e) - kappa * (1.0 - e)) + 2.0 * i * a * (1.0 - e);
    let b_val = b_num / ((kappa_b + gamma) * (1.0 - g * e));

    (ln_a + b_val * x).exp()
  }

  /// Characteristic function of $L_t$ (Eq. 5).
  ///
  /// $\phi_{L_t}(u) = \Phi_t(-i\psi(u),\,\rho u,\,v_0)$
  pub fn chf_l(&self, t: f64, u: Complex64) -> Complex64 {
    let psi = self.psi_std_cgmy(u);
    let a = -Complex64::i() * psi;
    let b = Complex64::from(self.rho) * u;
    self.cir_joint_chf(t, a, b, self.v0)
  }

  /// Convexity correction $\omega(t) = \ln E[e^{L_t}]$; NaN where that moment is infinite, at
  /// `λ₊ ≤ 1` or from the CIR factor's explosion time on.
  pub fn omega(&self, t: f64) -> f64 {
    let u = -Complex64::i();
    let psi = self.psi_std_cgmy(u);
    if !(t < self.explosion_time(psi.re)) {
      return f64::NAN;
    }
    let phi = self.cir_joint_chf(
      t,
      -Complex64::i() * psi,
      Complex64::from(self.rho) * u,
      self.v0,
    );
    phi.ln().re
  }

  /// First `t` with `E[e^{L_t}] = ∞`, `psi` being `ψ(−i)`: 0 at `λ₊ ≤ 1`, else where the Riccati
  /// `B' = ½ζ²B² − κB + psi`, `B(0) = ρ` blows up, `∞` if it never does.
  /// Andersen & Piterbarg (2007), Finance Stoch. 11(1), 29–50.
  fn explosion_time(&self, psi: f64) -> f64 {
    if !(self.lambda_plus > 1.0) {
      return 0.0;
    }
    let zeta2 = self.zeta * self.zeta;
    let disc = self.kappa * self.kappa - 2.0 * zeta2 * psi;
    let kappa_b = self.kappa - self.rho * zeta2;
    if disc < 0.0 {
      let w = (-disc).sqrt();
      return (PI + 2.0 * (kappa_b / w).atan()) / w;
    }
    let gamma = disc.sqrt();
    if kappa_b + gamma >= 0.0 {
      f64::INFINITY
    } else {
      ((kappa_b - gamma) / (kappa_b + gamma)).ln() / gamma
    }
  }
}

impl FourierModelExt for CgmysvModel {
  fn chf(&self, t: f64, xi: Complex64) -> Complex64 {
    let omega = self.params.omega(t);
    let i = Complex64::i();
    (-i * xi * omega).exp() * self.params.chf_l(t, xi)
  }

  fn cumulants(&self, t: f64) -> Cumulants {
    let h = 1e-4;
    let f = |u: f64| self.chf(t, Complex64::new(u, 0.0)).ln();

    let f0 = f(0.0);
    let fp = f(h);
    let fm = f(-h);
    let f2p = f(2.0 * h);
    let f2m = f(-2.0 * h);

    Cumulants {
      c1: (fp - fm).im / (2.0 * h),
      c2: -(fp - 2.0 * f0 + fm).re / (h * h),
      c4: Some((f2p - 4.0 * fp + 6.0 * f0 - 4.0 * fm + f2m).re / h.powi(4)),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::pricing::fourier::LewisPricer;

  fn paper() -> CgmysvParams {
    CgmysvParams {
      alpha: 0.5184,
      lambda_plus: 25.4592,
      lambda_minus: 4.6040,
      kappa: 1.0029,
      eta: 0.0711,
      zeta: 0.3443,
      rho: -2.0283,
      v0: 0.01115,
    }
  }

  /// A calibration iterate where `κ²ηt/ζ² ≈ 811` overflows `exp` and `base^(−2κη/ζ²)` underflows.
  fn small_zeta() -> CgmysvModel {
    CgmysvModel {
      params: CgmysvParams {
        alpha: 0.026903895236514812,
        lambda_plus: 99.94654878990875,
        lambda_minus: 43.88327435265859,
        kappa: 2.679354497451188,
        eta: 0.08193038626442775,
        zeta: 0.01904131284985643,
        rho: -2.638349514477035,
        v0: 0.00841920903908928,
      },
    }
  }

  /// mpmath at 40 digits; the transform's Riccati ODE, integrated by RK4, agrees to 1e-18.
  #[test]
  fn a_small_zeta_chf_and_price_match_mpmath() {
    let model = small_zeta();
    for (xi, want) in [
      (
        0.5,
        Complex64::new(0.9973982291261041, -0.0050978901672321395),
      ),
      (
        5.0,
        Complex64::new(0.7722420717963142, -0.02818955767339433),
      ),
    ] {
      let got = model.chf(0.5, Complex64::new(xi, 0.0));
      assert!((got - want).norm() < 1e-12, "chf at {xi}: {got} vs {want}");
    }
    let got = model.chf(0.5, Complex64::new(3.0, -0.5));
    let want = Complex64::new(0.910175885026733, 0.002887352173093535);
    assert!(
      (got - want).norm() < 1e-12,
      "chf at 3 − i/2: {got} vs {want}"
    );
    // Lewis's ∫₀^{1e-8} head, 1.26e-6 here, is inside the tolerance.
    let call = LewisPricer::price_call(&model, 100.0, 100.0, 0.03, 0.01, 0.5);
    assert!((call - 6.1426157634621505).abs() < 2e-6, "{call}");
  }

  /// `E[e^{L_t}]` is infinite at `λ₊ ≤ 1` and past the CIR factor's blow-up (mpmath, which the
  /// Riccati ODE confirms), where no drift-free chf exists.
  #[test]
  fn the_chf_is_nan_once_the_exponential_moment_explodes() {
    for (params, blow_up) in [
      (
        CgmysvParams {
          zeta: 2.0,
          ..paper()
        },
        3.6432800209919963,
      ),
      (
        CgmysvParams {
          zeta: 5.0,
          ..paper()
        },
        1.3059860516179807,
      ),
      (
        CgmysvParams {
          zeta: 1.0,
          rho: 3.0,
          ..paper()
        },
        1.0092675415357096,
      ),
    ] {
      let model = CgmysvModel { params };
      let before = model.chf(0.99 * blow_up, Complex64::new(1.0, 0.0));
      assert!(before.is_finite(), "{before} before {blow_up}");
      let after = model.chf(1.01 * blow_up, Complex64::new(1.0, 0.0));
      assert!(after.is_nan(), "{after} after {blow_up}");
    }
    let heavy = CgmysvModel {
      params: CgmysvParams {
        lambda_plus: 0.5,
        ..paper()
      },
    };
    assert!(heavy.chf(0.5, Complex64::new(1.0, 0.0)).is_nan());
    assert!(
      CgmysvModel { params: paper() }
        .chf(30.0, Complex64::new(1.0, 0.0))
        .is_finite()
    );
  }

  /// At `ζ = 2` a principal-branch power of `cosh + κ'/γ·sinh` leaves the branch the ODE follows.
  #[test]
  fn a_large_zeta_chf_stays_on_the_branch_of_its_riccati_ode() {
    let params = CgmysvParams {
      zeta: 2.0,
      ..paper()
    };
    let got = params.chf_l(0.5, Complex64::new(194.25, 0.0));
    let want = Complex64::new(0.3270656077906372, 0.1708596916589531);
    assert!((got - want).norm() < 1e-12, "{got} vs {want}");
  }
}

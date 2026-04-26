//! CGMYSV model characteristic function and Fourier integration support.
//!
//! $$
//! \phi_{L_t}(u) = \Phi_t\!\bigl(\psi_{\mathrm{stdCGMY}}(u),\;\rho\,u,\;v_0\bigr)
//! $$
//!
//! Reference: Kim, Y. S. (2021), arXiv:2101.11001, Sections 2–3.

use num_complex::Complex64;
use scilib::math::basic::gamma;

use crate::pricing::fourier::Cumulants;
use crate::pricing::fourier::FourierModelExt;

/// Parameters for the CGMYSV process (Eq. 4).
///
/// Combines a standard CGMY jump process $Z_t$ with a CIR stochastic
/// time change $\mathcal{V}_t = \int_0^t v_s\,ds$.
#[derive(Debug, Clone)]
pub struct CgmysvParams {
  /// Activity parameter $\alpha \in (0,2)$ (Y in CGMY notation).
  pub alpha: f64,
  /// Positive tempering $\lambda_+ > 0$ (G in CGMY notation).
  pub lambda_plus: f64,
  /// Negative tempering $\lambda_- > 0$ (M in CGMY notation).
  pub lambda_minus: f64,
  /// CIR mean reversion rate $\kappa > 0$.
  pub kappa: f64,
  /// CIR long-run mean $\eta > 0$.
  pub eta: f64,
  /// CIR volatility of volatility $\zeta > 0$.
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
  /// Risk-free rate.
  pub r: f64,
  /// Continuous dividend yield.
  pub q: f64,
}

impl CgmysvParams {
  /// Standard CGMY normalising constant
  /// $C = \bigl(\Gamma(2-\alpha)(\lambda_+^{\alpha-2}+\lambda_-^{\alpha-2})\bigr)^{-1}$.
  pub fn norm_const(&self) -> f64 {
    let g2a = gamma(2.0 - self.alpha);
    1.0
      / (g2a * (self.lambda_plus.powf(self.alpha - 2.0) + self.lambda_minus.powf(self.alpha - 2.0)))
  }

  /// Lévy symbol of the standard CGMY distribution $\psi_{\mathrm{stdCGMY}}(u)$ (Eq. 3).
  ///
  /// $$
  /// \psi(u) = \frac{\lambda_+^{\alpha-1}-\lambda_-^{\alpha-1}}{(\alpha-1)\,D}\,iu
  /// + \frac{(\lambda_+-iu)^\alpha - \lambda_+^\alpha + (\lambda_-+iu)^\alpha - \lambda_-^\alpha}
  ///        {\alpha(\alpha-1)\,D}
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

  /// CIR joint characteristic function $\Phi_t(a,b,x) = E[\exp(a\mathcal V_t + ib\,v_t)\mid v_0=x]$ (Eq. 2).
  ///
  /// Following Kim (2021) / Lamberton–Lapeyre (1996) Prop. 6.2.5 verbatim:
  /// $\gamma = \sqrt{\kappa^2 - 2\zeta^2 i a}$.
  pub fn cir_joint_chf(&self, t: f64, a: Complex64, b: Complex64, x: f64) -> Complex64 {
    let kappa = self.kappa;
    let eta = self.eta;
    let zeta = self.zeta;
    let zeta2 = zeta * zeta;
    let i = Complex64::i();

    let gamma = (Complex64::from(kappa * kappa) - 2.0 * zeta2 * i * a).sqrt();
    let half_gt = gamma * t / 2.0;
    let ch = half_gt.cosh();
    let sh = half_gt.sinh();

    let base = ch + (Complex64::from(kappa) - i * b * zeta2) / gamma * sh;
    let exp_power = 2.0 * kappa * eta / zeta2;
    let a_val = (kappa * kappa * eta * t / zeta2).exp() * base.powf(-exp_power);

    let b_num = i * b * (gamma * ch - Complex64::from(kappa) * sh) + 2.0 * i * a * sh;
    let b_denom = gamma * ch + (Complex64::from(kappa) - i * b * zeta2) * sh;
    let b_val = b_num / b_denom;

    a_val * (b_val * x).exp()
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

  /// Convexity correction $\omega(t) = \ln E[\exp(L_t)] = \mathrm{Re}\!\bigl(\ln\phi_{L_t}(-i)\bigr)$.
  pub fn omega(&self, t: f64) -> f64 {
    self.chf_l(t, -Complex64::i()).ln().re
  }
}

impl FourierModelExt for CgmysvModel {
  fn chf(&self, t: f64, xi: Complex64) -> Complex64 {
    let omega = self.params.omega(t);
    let i = Complex64::i();
    (i * xi * ((self.r - self.q) * t - omega)).exp() * self.params.chf_l(t, xi)
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
      c4: (f2p - 4.0 * fp + 6.0 * f0 - 4.0 * fm + f2m).re / h.powi(4),
    }
  }
}

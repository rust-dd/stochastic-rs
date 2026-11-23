//! Characteristic-function pricing of any [`FourierModelExt`]: Carr–Madan, Lewis, Gil–Pelaez, COS
//! and the fractional FFT.
//!
//! $$
//! C(K)=\frac{e^{-\alpha k}}{\pi}\int_0^\infty e^{-ivk}\,\psi_T(v)\,dv,\quad
//! \psi_T(v)=\frac{e^{-rT}\,\phi_T(v-(1+\alpha)i)}{\alpha^2+\alpha-v^2+i(2\alpha+1)v}
//! $$
//!
//! Reference: Carr & Madan (1999), J. Comput. Finance 2(4), 61–73
//! Reference: Lewis (2001), SSRN 282110, DOI 10.2139/ssrn.282110
//! Reference: Gil-Pelaez (1951), Biometrika 38(3–4), 481–482, DOI 10.1093/biomet/38.3-4.481

use num_complex::Complex64;

mod bsm;
mod cos;
mod cumulants;
mod frft;
mod heston;
mod hybrid;
mod levy;
mod pricer;
mod quadrature;
mod rough_heston;

pub use bsm::BsmFourier;
pub use cos::CosEngine;
pub use cumulants::Cumulants;
pub(crate) use cumulants::heston_cumulants;
pub(crate) use cumulants::mean_integrated_variance;
pub use frft::FrftCarrMadanPricer;
pub use heston::DoubleHestonFourier;
pub use heston::HestonFourier;
pub(crate) use heston::factor_cd;
pub(crate) use heston::riccati_cd;
pub use hybrid::BatesFourier;
pub use hybrid::HkdeFourier;
pub use hybrid::HkdeParams;
pub use hybrid::SvjParams;
pub use levy::CgmyFourier;
pub use levy::KouFourier;
pub use levy::MertonJdFourier;
pub use levy::NigFourier;
pub use levy::VarianceGammaFourier;
pub use pricer::CarrMadanPricer;
pub use quadrature::GilPelaezPricer;
pub use quadrature::LewisPricer;
pub use rough_heston::RoughHeston;
pub use rough_heston::RoughHestonCf;
pub use rough_heston::RoughHestonParams;

/// Trait for models that expose a characteristic function of the log-price,
/// enabling pricing via Fourier inversion.
pub trait FourierModelExt {
  /// $E^{\mathbb{Q}}[e^{i\xi X_T}]$ of $X_T = \ln(S_T/F_T)$, $F_T = S_0 e^{(r-q)T}$: drift-free, so
  /// $E[e^{X_T}] = 1$, and a pricer applies the carry $e^{i\xi(r-q)T}$ from its query.
  fn chf(&self, t: f64, xi: Complex64) -> Complex64;

  /// Cumulants of $X_T = \ln(S_T/F_T)$ at maturity `t`.
  fn cumulants(&self, t: f64) -> Cumulants;
}

/// Every [`FourierModelExt`] model prices through [`LewisPricer`]: the payoff at `tau = 0`, NaN
/// past expiry.
impl<T: FourierModelExt> crate::traits::ModelPricer for T {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    LewisPricer::price_call(self, s, k, r, q, tau)
  }

  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    pricer::parity_put(|| self.price_call(s, k, r, q, tau), s, k, r, q, tau)
  }
}

/// [`LewisPricer::price_call`] is the vanilla European call on the forward $Se^{(r-q)\tau}$.
impl<T: FourierModelExt> crate::traits::VanillaEuropeanCall for T {}

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
mod tests;

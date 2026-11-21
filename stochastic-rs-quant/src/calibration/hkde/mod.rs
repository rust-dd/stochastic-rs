//! Heston with Kou double-exponential jumps (HKDE), fitted by Levenberg–Marquardt, vega-weighted:
//!
//! $$
//! \begin{aligned}
//! dS_t &= (r-q-\lambda\bar k)S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S + S_t(e^{J_t}-1)\,dN_t \\
//! dv_t &= \kappa(\theta-v_t)\,dt + \sigma\sqrt{v_t}\,dW_t^v, \quad
//! d\langle W^S,W^v\rangle_t = \rho\,dt
//! \end{aligned}
//! $$
//!
//! $$
//! f_J(y) = p\,\eta_1 e^{-\eta_1 y}\,\mathbb{1}_{y\ge 0}
//!        + (1-p)\,\eta_2 e^{\eta_2 y}\,\mathbb{1}_{y<0},\qquad \eta_1>1,\ \eta_2>0.
//! $$
//!
//! $$\phi_{\mathrm{Hkde}}(\xi,t) = \phi_{\mathrm{Hes}}(\xi,t)\,\phi_{\mathrm{Kou}}(\xi,t)$$
//!
//! $$
//! \theta^{\*} = \arg\min_{\theta\in\Theta}
//! \sum_{n=1}^{N}\sum_{j=1}^{N_n} w_j^{(n)}\bigl(\mathcal V_j^{(n)}(\theta)-v_j^{(n)}\bigr)^2,
//! $$
//!
//! $$
//! w_j^{(n)} := \bigl(S_0\,\psi(d_1)\,\sqrt{T_n}\bigr)^{-1},\qquad
//! d_1 = \frac{\log(S_0/K_j^{(n)}) + T_n\bigl(r_n-q_n+\tfrac12(\sigma_j^{(n)})^2\bigr)}
//!            {\sigma_j^{(n)}\sqrt{T_n}},
//! $$
//!
//! with `N_t` Poisson at rate `λ`, `ψ` the normal pdf and `σ_j^{(n)}` the quote's vol, held fixed.
//! Reference: Agazzotti et al. (2025), arXiv:2502.13824, eqs. (12)–(13)
//! Reference: Kou (2002), Manag. Sci. 48(8), 1086–1101, DOI 10.1287/mnsc.48.8.1086.166
//! Reference: Heston (1993), Rev. Financ. Stud. 6(2), 327–343, DOI 10.1093/rfs/6.2.327
//! Reference: Levenberg (1944), Q. Appl. Math. 2(2), 164–168, DOI 10.1090/qam/10666
//! Reference: Marquardt (1963), J. Soc. Ind. Appl. Math. 11(2), 431–441, DOI 10.1137/0111030

mod calibrator;
mod loss;
mod params;

pub use calibrator::HkdeCalibrator;

use crate::calibration::Calibrated;
use crate::pricing::fourier::HkdeParams;
use crate::traits::ToModel;

impl ToModel for Calibrated<HkdeParams> {
  type Model = crate::pricing::fourier::HkdeFourier;

  fn to_model(&self) -> Self::Model {
    crate::pricing::fourier::HkdeFourier::new(self.params)
  }
}

#[cfg(test)]
mod tests;

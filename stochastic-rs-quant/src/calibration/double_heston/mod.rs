//! Double Heston calibration: two variance factors, each with its own spot Brownian:
//!
//! $$
//! \begin{aligned}
//! dS_t &= (r-q)S_t\,dt + \sqrt{v_{1,t}}\,S_t\,dW_{1,t}^S + \sqrt{v_{2,t}}\,S_t\,dW_{2,t}^S \\
//! dv_{1,t} &= \kappa_1(\theta_1 - v_{1,t})\,dt + \sigma_1\sqrt{v_{1,t}}\,dW_{1,t}^v \\
//! dv_{2,t} &= \kappa_2(\theta_2 - v_{2,t})\,dt + \sigma_2\sqrt{v_{2,t}}\,dW_{2,t}^v
//! \end{aligned}
//! $$
//! with `dW_i^S dW_i^v = ρ_i dt` and every other pair independent; the chf sums two Heston terms:
//! $$
//! \phi_T(u) = \exp\!\left(iu(r-q)T + \sum_{j=1}^2\bigl[C_j(u,T) + D_j(u,T)\,v_{j,0}\bigr]\right)
//! $$
//!
//! Reference: Christoffersen et al. (2009), Manag. Sci. 55, 1914–1932, DOI 10.1287/mnsc.1090.1065
//! Reference: Mehrdoust et al. (2021), J. Comput. Appl. Math. 392, DOI 10.1016/j.cam.2021.113422
//! Reference: Levenberg (1944), Q. Appl. Math. 2(2), 164–168, DOI 10.1090/qam/10666
//! Reference: Marquardt (1963), J. Soc. Ind. Appl. Math. 11(2), 431–441, DOI 10.1137/0111030

mod calibrator;
mod loss;
mod params;

pub use calibrator::DoubleHestonCalibrator;
pub use params::DoubleHestonParams;

use crate::calibration::Calibrated;
use crate::pricing::fourier::DoubleHestonFourier;
use crate::traits::ToModel;

impl ToModel for Calibrated<DoubleHestonParams> {
  type Model = DoubleHestonFourier;

  fn to_model(&self) -> DoubleHestonFourier {
    self.params.to_model()
  }
}

#[cfg(test)]
mod quadrature_tests;

#[cfg(test)]
mod tests;

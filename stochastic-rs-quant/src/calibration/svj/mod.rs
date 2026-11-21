//! SVJ (Bates) calibration: Heston with i.i.d. log-jumps `J_t ~ N(μ_J, σ_J²)` at Poisson rate `λ`:
//!
//! $$
//! \begin{aligned}
//! dS_t &= (r-q-\lambda\bar k)S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S + J_t\,S_t\,dN_t \\
//! dv_t &= \kappa(\theta-v_t)\,dt + \sigma\sqrt{v_t}\,dW_t^v, \quad
//! d\langle W^S,W^v\rangle_t = \rho\,dt
//! \end{aligned}
//! $$
//!
//! with the characteristic function
//! $$
//! \phi_T(\xi) = \phi_T^{\mathrm{Heston}}(\xi)\cdot
//! \exp\!\bigl[T\,\lambda\bigl(e^{i\mu_J\xi - \tfrac12\sigma_J^2\xi^2} - 1\bigr)\bigr].
//! $$
//!
//! Reference: Bates (1996), Rev. Financ. Stud. 9(1), 69–107, DOI 10.1093/rfs/9.1.69
//! Reference: Heston (1993), Rev. Financ. Stud. 6(2), 327–343, DOI 10.1093/rfs/6.2.327

mod calibrator;
mod loss;
mod params;

pub use calibrator::SvjCalibrator;

use crate::calibration::Calibrated;
use crate::pricing::fourier::BatesFourier;
use crate::pricing::fourier::SvjParams;
use crate::traits::ToModel;

impl ToModel for Calibrated<SvjParams> {
  type Model = BatesFourier;

  fn to_model(&self) -> BatesFourier {
    BatesFourier::new(self.params)
  }
}

#[cfg(test)]
mod quadrature_tests;

#[cfg(test)]
mod tests;

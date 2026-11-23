//! Rough Heston by its characteristic function, three ways: the fractional Riccati equation by
//! the Adams scheme, its rational approximation, and the lifted model's ordinary Riccati system.
//!
//! $$
//! V_t = V_0 + \int_0^t K(t-s)\,\kappa(\theta - V_s)\,ds + \int_0^t K(t-s)\,\nu\sqrt{V_s}\,dW^v_s,
//! \qquad K(t) = \frac{t^{\alpha-1}}{\Gamma(\alpha)},\quad \alpha = H + \tfrac12
//! $$
//!
//! $$
//! \ln\phi(t,\xi) = \kappa\theta\,I^{1}h(t) + V_0\,I^{1-\alpha}h(t),\qquad
//! D^{\alpha}h = F(z,h) = \tfrac12(z^2-z) + (\rho\nu z-\kappa)h + \tfrac12\nu^2h^2,\quad z = i\xi
//! $$
//!
//! Reference: El Euch & Rosenbaum (2019), Math. Finance 29(1), 3–38, DOI 10.1111/mafi.12173

mod adams;
mod lifted;
mod pade;
mod params;
mod scalar;

#[cfg(test)]
mod tests;

pub use params::RoughHestonParams;

use num_complex::Complex64;
use scalar::Jet4;
use scalar::RiccatiScalar;

use super::Cumulants;

/// `ln E[e^{zX_t}]` in a scalar: complex `z = iξ` for the chf, the jet variable for cumulants.
pub(super) trait LogChf {
  fn log_chf<S: RiccatiScalar>(&self, t: f64, z: S) -> S;
}

/// 1 at `t = 0` and where `m₀ = 0` (`ξ = 0`, `−i`); NaN for a `t` or `ξ` that is not finite, or
/// `t < 0`.
pub(super) fn chf(cf: &impl LogChf, t: f64, xi: Complex64) -> Complex64 {
  if !(t >= 0.0 && t.is_finite() && xi.is_finite()) {
    return Complex64::new(f64::NAN, f64::NAN);
  }
  let z = Complex64::i() * xi;
  if t == 0.0 || Riccati::moment(z) == Complex64::ZERO {
    return Complex64::ONE;
  }
  cf.log_chf(t, z).exp()
}

/// The Taylor coefficients of `ln E[e^{sX_t}]` at `s = 0`, from one solve in the jet.
pub(super) fn cumulants(cf: &impl LogChf, t: f64) -> Cumulants {
  if !(t >= 0.0 && t.is_finite()) {
    return Cumulants {
      c1: f64::NAN,
      c2: f64::NAN,
      c4: Some(f64::NAN),
    };
  }
  let Jet4(series) = if t == 0.0 {
    Jet4::ZERO
  } else {
    cf.log_chf(t, Jet4::VARIABLE)
  };
  Cumulants {
    c1: series[1],
    c2: 2.0 * series[2],
    c4: Some(24.0 * series[4]),
  }
}

/// The Riccati right-hand side all three papers share, `F(z, x) = m₀ + (ρνz − κ)x + ½ν²x²`.
#[derive(Clone, Copy, Debug)]
pub(super) struct Riccati {
  pub(super) kappa: f64,
  pub(super) nu: f64,
  pub(super) rho: f64,
}

impl Riccati {
  pub(super) fn of(params: &RoughHestonParams) -> Self {
    Self {
      kappa: params.kappa,
      nu: params.nu,
      rho: params.rho,
    }
  }

  /// `m₀ = ½z(z − 1) = −½ξ(ξ + i)` as a product: at `ξ = u − i` the sum rounds `u²` away against 1.
  pub(super) fn moment<S: RiccatiScalar>(z: S) -> S {
    z * (z - S::real(1.0)) * 0.5
  }

  pub(super) fn rhs<S: RiccatiScalar>(&self, moment: S, z: S, x: S) -> S {
    moment
      + (z * (self.rho * self.nu) - S::real(self.kappa)) * x
      + x * x * (0.5 * self.nu * self.nu)
  }
}

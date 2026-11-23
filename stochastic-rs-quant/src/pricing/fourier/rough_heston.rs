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
//! Reference: Gatheral & Radoičić (2019), IJTAF 22(3), 1950010, DOI 10.1142/S0219024919500109
//! Reference: Abi Jaber (2019), Quant. Finance 19(12), 1995–2013, DOI 10.1080/14697688.2019.1615113

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
use super::FourierModelExt;

/// Which characteristic function a [`RoughHeston`] prices by; the default is the (4,4) approximant.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum RoughHestonCf {
  /// The fractional Riccati equation by the Adams scheme on `steps` steps, its implicit corrector
  /// solved exactly: the reference, `O(steps²)` a frequency.
  Adams { steps: usize },
  /// Gatheral & Radoičić's rational approximation of order 3 or 4: the fast one, within about
  /// 1e-3 of Adams in implied vol for `ν ≤ 0.6`, and NaN off the strip `−1 ≤ Im ξ ≤ 0`.
  Pade { order: usize },
  /// Abi Jaber's lifted model, `factors` factors on the geometric partition of ratio `ratio`, its
  /// Riccati system by implicit Euler on `steps` steps: another model, near the rough one.
  Lifted {
    factors: usize,
    ratio: f64,
    steps: usize,
  },
}

impl RoughHestonCf {
  /// 200 steps, within 1e-5 of the limit in implied vol.
  pub const ADAMS: Self = Self::Adams { steps: 200 };

  /// Abi Jaber's (3.10) and the 300 steps of his §3.1.
  pub const LIFTED: Self = Self::Lifted {
    factors: 20,
    ratio: 2.5,
    steps: 300,
  };

  /// The first of [`RoughHeston::new`]'s checks this selector fails, in its panic's words.
  #[doc(hidden)]
  pub fn violation(&self) -> Option<String> {
    let at_least_one = |steps: usize| {
      (steps < 1).then(|| format!("steps must satisfy `steps >= 1`, got steps = {steps:?}"))
    };
    match *self {
      Self::Adams { steps } => at_least_one(steps),
      Self::Pade { order } => (order != 3 && order != 4)
        .then(|| format!("order must satisfy `order == 3 || order == 4`, got order = {order:?}")),
      Self::Lifted {
        factors,
        ratio,
        steps,
      } => {
        if factors < 1 {
          Some(format!(
            "factors must satisfy `factors >= 1`, got factors = {factors:?}"
          ))
        } else if !(ratio.is_finite() && ratio > 1.0) {
          Some(format!(
            "ratio must satisfy `ratio.is_finite() && ratio > 1.0`, got ratio = {ratio:?}"
          ))
        } else {
          at_least_one(steps)
        }
      }
    }
  }
}

impl Default for RoughHestonCf {
  fn default() -> Self {
    Self::Pade { order: 4 }
  }
}

#[derive(Clone, Debug)]
enum Engine {
  Adams(adams::AdamsCf),
  Pade(pade::PadeCf),
  Lifted(lifted::LiftedCf),
}

/// The rough Heston model for Fourier pricing, by the characteristic function `cf` selects.
#[derive(Clone, Debug)]
pub struct RoughHeston {
  params: RoughHestonParams,
  cf: RoughHestonCf,
  engine: Engine,
}

impl RoughHeston {
  /// Panics unless `params` passes [`RoughHestonParams::new`], and `cf` has `steps ≥ 1`, an order
  /// of 3 or 4, `factors ≥ 1` and a finite `ratio > 1`.
  pub fn new(params: RoughHestonParams, cf: RoughHestonCf) -> Self {
    let params = params.checked();
    if let Some(reason) = cf.violation() {
      panic!("{reason}");
    }
    let engine = match cf {
      RoughHestonCf::Adams { steps } => Engine::Adams(adams::AdamsCf::new(params, steps)),
      RoughHestonCf::Pade { order } => Engine::Pade(pade::PadeCf::new(params, order)),
      RoughHestonCf::Lifted {
        factors,
        ratio,
        steps,
      } => Engine::Lifted(lifted::LiftedCf::new(params, factors, ratio, steps)),
    };
    Self { params, cf, engine }
  }

  pub fn params(&self) -> RoughHestonParams {
    self.params
  }

  pub fn cf(&self) -> RoughHestonCf {
    self.cf
  }
}

impl FourierModelExt for RoughHeston {
  /// 1 at `t = 0`; NaN for a `t` that is negative or not finite, and under `Pade` off its strip.
  fn chf(&self, t: f64, xi: Complex64) -> Complex64 {
    match &self.engine {
      Engine::Adams(cf) => chf(cf, t, xi),
      Engine::Pade(cf) => cf.chf(t, xi),
      Engine::Lifted(cf) => chf(cf, t, xi),
    }
  }

  /// Those of the selected characteristic function itself, from one solve in a Taylor jet.
  fn cumulants(&self, t: f64) -> Cumulants {
    match &self.engine {
      Engine::Adams(cf) => cumulants(cf, t),
      Engine::Pade(cf) => cumulants(cf, t),
      Engine::Lifted(cf) => cumulants(cf, t),
    }
  }
}

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

  /// `x = known + c·F(z, x)` for the root whose slope is `−√disc`: the explicit value as `c → 0`
  /// and the stable equilibrium `r₋/ν` as `c → ∞`, so an implicit step holds at any `c`.
  pub(super) fn implicit<S: RiccatiScalar>(&self, moment: S, z: S, c: f64, known: S) -> S {
    let square = 0.5 * self.nu * self.nu * c;
    let linear = (z * (self.rho * self.nu) - S::real(self.kappa)) * c - S::real(1.0);
    let constant = known + moment * c;
    let root = (linear * linear - constant * (4.0 * square)).sqrt();
    if linear.re() <= 0.0 {
      constant * 2.0 / (root - linear)
    } else {
      (-linear - root) / S::real(2.0 * square)
    }
  }

  pub(super) fn rhs<S: RiccatiScalar>(&self, moment: S, z: S, x: S) -> S {
    moment
      + (z * (self.rho * self.nu) - S::real(self.kappa)) * x
      + x * x * (0.5 * self.nu * self.nu)
  }
}

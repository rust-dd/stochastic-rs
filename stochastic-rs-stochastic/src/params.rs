//! Constructor vocabulary shared by several processes.

use std::fmt;

use crate::traits::FloatExt;
use crate::traits::Fn1D;

/// How a square-root state is kept non-negative after an Euler step: floored at zero or reflected.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum VarianceBoundary {
  /// `max(x, 0)`.
  #[default]
  Floor,
  /// `|x|`.
  Reflect,
}

/// The drift of a log-price model: a rate, a cost of carry, or the domestic–foreign pair (`r − r_f`).
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Drift<T> {
  Mu(T),
  CostOfCarry(T),
  Rates { r: T, r_f: T },
}

impl<T: FloatExt> Drift<T> {
  /// The price's drift rate: `μ`, `b` or `r − r_f`.
  pub fn rate(&self) -> T {
    match *self {
      Drift::Mu(mu) => mu,
      Drift::CostOfCarry(b) => b,
      Drift::Rates { r, r_f } => r - r_f,
    }
  }

  #[track_caller]
  pub(crate) fn assert_finite(&self) {
    assert!(
      self.rate().is_finite(),
      "drift must satisfy `drift.rate().is_finite()`, got drift = {self:?}"
    );
  }
}

/// What ends an event-driven path: a fixed number of events or a time horizon.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Horizon<T> {
  Events(usize),
  Until(T),
}

impl<T: FloatExt> Horizon<T> {
  #[track_caller]
  pub(crate) fn assert_valid(&self) {
    if let Horizon::Until(t_max) = *self {
      assert!(
        t_max > T::zero() && t_max.is_finite(),
        "t_max must satisfy `t_max > T::zero() && t_max.is_finite()`, got t_max = {t_max:?}"
      );
    }
  }
}

/// Ho–Lee's drift: a constant θ, or the forward curve `f(0, T)` the drift `∂_T f(0, t) + σ²t` is
/// read from.
#[derive(Clone)]
#[non_exhaustive]
pub enum HoLeeDrift<T: FloatExt> {
  Constant(T),
  ForwardCurve(Fn1D<T>),
}

// By hand: `Fn1D` has no `Debug`, so the curve prints as a placeholder.
impl<T: FloatExt> fmt::Debug for HoLeeDrift<T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      HoLeeDrift::Constant(theta) => f.debug_tuple("Constant").field(theta).finish(),
      HoLeeDrift::ForwardCurve(_) => f.write_str("ForwardCurve(..)"),
    }
  }
}

/// No process carries a `use_sym` flag: the step's boundary is a [`VarianceBoundary`].
///
/// ```compile_fail,E0560
/// use stochastic_rs_stochastic::diffusion::cir::Cir;
/// let c = Cir::<f64> { use_sym: Some(true), ..Cir::default() };
/// ```
#[cfg(doctest)]
pub struct UseSymIsGone;

/// No log-price model takes four optional rates: its drift is one [`Drift`].
///
/// ```compile_fail,E0061
/// use stochastic_rs_stochastic::diffusion::gbm_log::GbmLog;
/// use stochastic_rs_stochastic::simd_rng::Unseeded;
/// let g = GbmLog::<f64>::new(Some(0.05), None, None, None, 0.2, 64, Some(1.0), None, Unseeded);
/// ```
#[cfg(doctest)]
pub struct DriftIsOneArgument;

/// No event process takes an optional count beside an optional horizon: its end is one [`Horizon`].
///
/// ```compile_fail,E0061
/// use stochastic_rs_stochastic::process::poisson::Poisson;
/// use stochastic_rs_stochastic::simd_rng::Unseeded;
/// let p = Poisson::<f64>::new(2.0, None, Some(1.0), Unseeded);
/// ```
#[cfg(doctest)]
pub struct HorizonIsOneArgument;

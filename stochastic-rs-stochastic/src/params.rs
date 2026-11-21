//! Constructor vocabulary shared by several processes.

use std::fmt;

use crate::traits::FloatExt;
use crate::traits::Fn1D;

/// How an Euler step treats a square-root state `x̃` that leaves the half-line, as the `f`s of
/// $$\tilde x_{t+\Delta} = f_1(\tilde x_t) + \kappa(\theta - f_2(\tilde x_t))\Delta + \sigma\sqrt{f_3(\tilde x_t)}\,\Delta W$$
/// Reference: Lord, Koekkoek & van Dijk (2010), 10(2), 177–194, DOI 10.1080/14697680802392496
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum VarianceBoundary {
  /// `f₁ = x`, `f₂ = f₃ = x⁺`: the carried state keeps its sign and the path reports `x⁺`.
  #[default]
  FullTruncation,
  /// `f₁ = f₂ = f₃ = x⁺`.
  Absorption,
  /// `f₁ = f₂ = f₃ = |x|`.
  Reflect,
}

impl VarianceBoundary {
  /// `f₂ = f₃`: what drift, diffusion and the reported path read of the carried state.
  #[inline]
  pub(crate) fn observed<T: FloatExt>(self, carried: T) -> T {
    match self {
      VarianceBoundary::FullTruncation | VarianceBoundary::Absorption => carried.max(T::zero()),
      VarianceBoundary::Reflect => carried.abs(),
    }
  }

  /// The state carried out of a step or in from a start: `f₁` is applied here, so only full
  /// truncation keeps a sign.
  #[inline]
  pub(crate) fn carried<T: FloatExt>(self, stepped: T) -> T {
    match self {
      VarianceBoundary::FullTruncation => stepped,
      VarianceBoundary::Absorption | VarianceBoundary::Reflect => self.observed(stepped),
    }
  }

  /// One Euler step of `dx = κ(θ − x)dt + σ√x dW` from the carried state to the carried state,
  /// `dw` the Brownian increment; [`observed`](Self::observed) of either is the path's value.
  #[inline]
  pub(crate) fn square_root_step<T: FloatExt>(
    self,
    carried: T,
    kappa: T,
    theta: T,
    sigma: T,
    dt: T,
    dw: T,
  ) -> T {
    let x = self.observed(carried);
    let dx = kappa * (theta - x) * dt + sigma * x.sqrt() * dw;
    self.carried(carried + dx)
  }
}

/// The drift of a price model: a rate, a cost of carry, or the domestic–foreign pair (`r − r_f`).
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

/// What ends an event-driven path: `Events(n)`, the origin and the `n` arrivals after it (so
/// `Events(0)` is the origin alone), or `Until(t)`, a time horizon.
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
/// use stochastic_rs_core::simd_rng::Unseeded;
/// let g = GbmLog::<f64>::new(Some(0.05), None, None, None, 0.2, 64, Some(1.0), None, Unseeded);
/// ```
#[cfg(doctest)]
pub struct DriftIsOneArgument;

/// No event process takes an optional count beside an optional horizon: its end is one [`Horizon`].
///
/// ```compile_fail,E0061
/// use stochastic_rs_stochastic::process::poisson::Poisson;
/// use stochastic_rs_core::simd_rng::Unseeded;
/// let p = Poisson::<f64>::new(2.0, None, Some(1.0), Unseeded);
/// ```
#[cfg(doctest)]
pub struct HorizonIsOneArgument;

/// A process's initial spot and variance are values, not options.
///
/// ```compile_fail,E0308
/// use stochastic_rs_stochastic::params::VarianceBoundary;
/// use stochastic_rs_core::simd_rng::Unseeded;
/// use stochastic_rs_stochastic::volatility::HestonPow;
/// use stochastic_rs_stochastic::volatility::heston::Heston;
/// let h = Heston::<f64>::new(Some(100.0), Some(0.04), 2.0, 0.04, 0.5, -0.7, 0.03, 64, None, HestonPow::Sqrt, VarianceBoundary::Absorption, Unseeded);
/// ```
#[cfg(doctest)]
pub struct InitialValuesArePlain;

#[cfg(test)]
mod tests {
  use ndarray::array;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_core::simd_rng::Unseeded;
  use stochastic_rs_distributions::exp::SimdExp;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::Horizon;
  use crate::euler::EulerCoefficients;
  use crate::euler::EulerSystem;
  use crate::process::ccustom::CompoundCustom;
  use crate::process::cpoisson::CompoundPoisson;
  use crate::process::customjt::CustomJt;
  use crate::process::hawkes::Hawkes;
  use crate::process::multivariate_hawkes::MultivariateHawkes;
  use crate::process::poisson::Poisson;
  use crate::traits::ProcessExt;

  fn rows_and_grids(n: usize) -> (Vec<usize>, Vec<usize>) {
    let horizon = Horizon::Events(n);
    let seed = || Deterministic::new(7);
    let poisson = Poisson::new(2.0, horizon, seed());
    let hawkes = Hawkes::new(0.5, 0.3, 1.0, horizon, seed());
    let customjt = CustomJt::new(horizon, SimdExp::new(2.0), seed());
    let ccustom = CompoundCustom::new(
      SimdNormal::new(0.0, 0.1),
      CustomJt::new(horizon, SimdExp::new(2.0), Unseeded),
      seed(),
    );
    let cpoisson = CompoundPoisson::new(
      SimdNormal::new(0.0, 0.1),
      Poisson::new(2.0, horizon, Unseeded),
      seed(),
    );
    let one_component =
      MultivariateHawkes::new(array![0.5], array![[0.3]], array![[1.0]], horizon, seed());
    let mut rows = vec![
      poisson.sample().len(),
      hawkes.sample().len(),
      customjt.sample().len(),
    ];
    let c = ccustom.sample();
    rows.extend([c.arrivals.len(), c.cumulative.len(), c.jumps.len()]);
    let c = cpoisson.sample();
    rows.extend([c.arrivals.len(), c.cumulative.len(), c.jumps.len()]);
    rows.push(one_component.sample()[0].len());
    let grids = vec![
      EulerCoefficients::grid_points(&poisson),
      EulerCoefficients::grid_points(&hawkes),
      EulerCoefficients::grid_points(&customjt),
      EulerSystem::<f64, 3>::grid_points(&ccustom),
      EulerSystem::<f64, 3>::grid_points(&cpoisson),
    ];
    (rows, grids)
  }

  /// So reading `observed` where a recursion read the state and its modulus moves no path that
  /// starts at or above zero.
  #[test]
  fn a_state_at_or_above_zero_is_observed_and_carried_bit_for_bit() {
    use super::VarianceBoundary::*;
    let states = [
      0.0,
      5e-324,
      f64::MIN_POSITIVE,
      0.04,
      1.0,
      f64::MAX,
      f64::INFINITY,
    ];
    for boundary in [FullTruncation, Absorption, Reflect] {
      for x in states {
        let read = [boundary.observed(x), boundary.carried(x), x.abs()].map(f64::to_bits);
        assert_eq!(read, [x.to_bits(); 3], "{boundary:?} at {x:e}");
      }
    }
  }

  #[test]
  fn events_n_is_the_origin_and_n_arrivals_in_every_event_process() {
    let got = [0, 10].map(rows_and_grids);
    let want = [0, 10].map(|n| (vec![n + 1; 10], vec![n + 1; 5]));
    assert_eq!(got, want);
  }
}

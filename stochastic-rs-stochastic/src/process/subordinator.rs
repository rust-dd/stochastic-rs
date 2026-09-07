//! # Subordinator
//!
//! $$
//! \mathbb E[e^{-\lambda S_t}] = e^{-t\phi(\lambda)},\qquad \lambda \ge 0
//! $$
//!
//! Collection of monotone Levy-process style drivers and a Ctrw helper.

pub mod alpha_stable;
pub mod ctrw;
pub mod gamma_subordinator;
pub mod ig_subordinator;
pub mod inverse_alpha_stable;
pub mod poisson_subordinator;
pub mod tempered_stable;

use std::f64::consts::PI;

pub use alpha_stable::AlphaStableSubordinator;
pub use ctrw::Ctrw;
pub use ctrw::CtrwJumpLaw;
pub use ctrw::CtrwWaitingLaw;
pub use gamma_subordinator::GammaSubordinator;
pub use ig_subordinator::IGSubordinator;
pub use inverse_alpha_stable::InverseAlphaStableSubordinator;
pub use poisson_subordinator::PoissonSubordinator;
use stochastic_rs_distributions::uniform::SimdUniform;
pub use tempered_stable::TemperedStableSubordinator;

#[inline]
pub(crate) fn clamp_open01(u: f64) -> f64 {
  u.clamp(1e-12, 1.0 - 1e-12)
}

/// One positive-stable increment of scale `e^{log_scale}` — Kanter's form of
/// the Chambers-Mallows-Stuck transform, with every factor that can leave the
/// representable range gathered into a single exponential.
///
/// As a product the transform is `(cΔ)^{1/α} · sin(αu) · sin(u)^{−1/α} ·
/// (sin((1−α)u)/w)^{(1−α)/α}`, and each of those factors overflows on its own
/// far earlier than the draw does. At `α = 0.01` the exponent is 100:
/// `sin(u)^{1/α}` underflows to zero whenever `sin u < 8e-4`, the tail factor
/// overflows whenever `w < 7e-4`, and the folded scale `(cΔ)^{1/α}` is
/// already zero at `Δ = 1/4095` — so the quotient came back `+inf`, or `0 · ∞`
/// came back NaN, where the factors would have cancelled to an ordinary
/// number. 81 % of the points of an `α = 0.01` path were NaN this way, 0.16 %
/// at `α = 0.02`. Summed in logs they cancel before anything is raised to a
/// power, and the exponential overflows only where the draw itself is
/// genuinely unrepresentable.
#[inline]
pub(crate) fn sample_positive_stable(
  alpha: f64,
  log_scale: f64,
  uniform: &SimdUniform<f64>,
) -> f64 {
  let u = clamp_open01(uniform.sample_fast()) * PI;
  let w = -clamp_open01(uniform.sample_fast()).ln();
  let log_x = log_scale - u.sin().ln() / alpha
    + ((1.0 - alpha) / alpha) * (((1.0 - alpha) * u).sin().ln() - w.ln());
  (alpha * u).sin() * log_x.exp()
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_core::simd_rng::Unseeded;

  use super::AlphaStableSubordinator;
  use super::Ctrw;
  use super::CtrwJumpLaw;
  use super::CtrwWaitingLaw;
  use super::GammaSubordinator;
  use super::IGSubordinator;
  use super::InverseAlphaStableSubordinator;
  use super::PoissonSubordinator;
  use super::TemperedStableSubordinator;
  use crate::traits::ProcessExt;

  #[test]
  fn alpha_stable_subordinator_is_non_decreasing() {
    let p = AlphaStableSubordinator::new(0.7_f64, 1.0, 256, Some(0.0), Some(1.0), Unseeded);
    let x = p.sample();
    assert!(x.windows(2).into_iter().all(|w| w[1] >= w[0]));
  }

  /// A small `α` on a fine grid still produces a path, not a row of zeros.
  ///
  /// Kanter's transform used to be formed as a product, with the scale folded
  /// on the host as `(c·dt)^{1/α}`. That factor leaves the representable range
  /// on its own — at `α = 0.05` and `c·dt = 3.9e-18` it is `1e-348`, which is
  /// zero even as a double — so every increment came back `0 · s₁ · s₂` and
  /// the whole path was identically zero, while the terminal value the law
  /// asks for, near `(c·T)^{1/α} = 1e-300`, is perfectly representable. The
  /// same fold at `α = 0.01` on a 4096-point grid turned 81 % of a path's
  /// points into NaN instead, once `s₁` had overflowed to `+inf` and met the
  /// vanished scale.
  #[test]
  fn a_small_alpha_keeps_its_scale() {
    let p = AlphaStableSubordinator::new(
      0.05_f64,
      1e-15,
      256,
      Some(0.0),
      Some(1.0),
      Deterministic::new(4242),
    );
    let x = p.sample();
    assert!(
      x.iter().all(|v| v.is_finite()),
      "a path point is not finite"
    );
    assert!(
      x.windows(2).into_iter().all(|w| w[1] >= w[0]),
      "a subordinator path went backwards"
    );
    let terminal = x[x.len() - 1];
    assert!(
      terminal > 0.0,
      "the whole path collapsed to zero: terminal = {terminal:e}"
    );
  }

  #[test]
  fn inverse_stable_is_non_decreasing() {
    let p =
      InverseAlphaStableSubordinator::new(0.7_f64, 1.0, 128, Some(1.0), 2048, Some(4.0), Unseeded);
    let e = p.sample();
    assert_eq!(e[0], 0.0);
    assert!(e.windows(2).into_iter().all(|w| w[1] >= w[0]));
  }

  #[test]
  fn poisson_subordinator_is_non_decreasing() {
    let p = PoissonSubordinator::new(2.0_f64, 256, Some(0.0), Some(1.0), Unseeded);
    let x = p.sample();
    assert!(x.windows(2).into_iter().all(|w| w[1] >= w[0]));
  }

  #[test]
  fn gamma_subordinator_is_non_decreasing() {
    let p = GammaSubordinator::new(3.0_f64, 5.0, 256, Some(0.0), Some(1.0), Unseeded);
    let x = p.sample();
    assert!(x.windows(2).into_iter().all(|w| w[1] >= w[0]));
  }

  #[test]
  fn ig_subordinator_is_non_decreasing() {
    let p = IGSubordinator::new(1.5_f64, 2.0, 256, Some(0.0), Some(1.0), Unseeded);
    let x = p.sample();
    assert!(x.windows(2).into_iter().all(|w| w[1] >= w[0]));
  }

  #[test]
  fn tempered_stable_subordinator_is_non_decreasing() {
    let p =
      TemperedStableSubordinator::new(0.7_f64, 1.0, 2.0, 0.05, 256, Some(0.0), Some(1.0), Unseeded);
    let x = p.sample();
    assert!(x.windows(2).into_iter().all(|w| w[1] >= w[0]));
  }

  #[test]
  fn ctrw_path_is_finite() {
    let p = Ctrw::new(
      CtrwWaitingLaw::Exponential { rate: 2.0_f64 },
      CtrwJumpLaw::Normal {
        mean: 0.0,
        std: 1.0,
      },
      512,
      Some(0.0),
      Some(1.0),
      Unseeded,
    );
    let x = p.sample();
    assert_eq!(x.len(), 512);
    assert!(x.iter().all(|v| v.is_finite()));
  }
}

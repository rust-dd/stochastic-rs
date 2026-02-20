//! # Subordinator
//!
//! $$
//! \mathbb E[e^{-\lambda S_t}] = e^{-t\phi(\lambda)},\qquad \lambda \ge 0
//! $$
//!
//! Collection of monotone Levy-process style drivers and a CTRW helper.

pub mod alpha_stable;
pub mod ctrw;
pub mod gamma_subordinator;
pub mod ig_subordinator;
pub mod inverse_alpha_stable;
pub mod poisson_subordinator;
pub mod tempered_stable;

use std::f64::consts::PI;

pub use alpha_stable::AlphaStableSubordinator;
pub use ctrw::CTRW;
pub use ctrw::CtrwJumpLaw;
pub use ctrw::CtrwWaitingLaw;
pub use gamma_subordinator::GammaSubordinator;
pub use ig_subordinator::IGSubordinator;
pub use inverse_alpha_stable::InverseAlphaStableSubordinator;
pub use poisson_subordinator::PoissonSubordinator;
use rand::Rng;
pub use tempered_stable::TemperedStableSubordinator;

#[inline]
pub(crate) fn clamp_open01(u: f64) -> f64 {
  u.clamp(1e-12, 1.0 - 1e-12)
}

#[inline]
pub(crate) fn sample_positive_stable(alpha: f64, rng: &mut impl Rng) -> f64 {
  let u = clamp_open01(rng.random::<f64>()) * PI;
  let w = -clamp_open01(rng.random::<f64>()).ln();
  let s1 = (alpha * u).sin() / u.sin().powf(1.0 / alpha);
  let s2 = (((1.0 - alpha) * u).sin() / w).powf((1.0 - alpha) / alpha);
  s1 * s2
}

#[cfg(test)]
mod tests {
  use super::AlphaStableSubordinator;
  use super::CTRW;
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
    let p = AlphaStableSubordinator::new(0.7_f64, 1.0, 256, Some(0.0), Some(1.0));
    let x = p.sample();
    assert!(x.windows(2).into_iter().all(|w| w[1] >= w[0]));
  }

  #[test]
  fn inverse_stable_is_non_decreasing() {
    let p = InverseAlphaStableSubordinator::new(0.7_f64, 1.0, 128, Some(1.0), 2048, Some(4.0));
    let e = p.sample();
    assert_eq!(e[0], 0.0);
    assert!(e.windows(2).into_iter().all(|w| w[1] >= w[0]));
  }

  #[test]
  fn poisson_subordinator_is_non_decreasing() {
    let p = PoissonSubordinator::new(2.0_f64, 256, Some(0.0), Some(1.0));
    let x = p.sample();
    assert!(x.windows(2).into_iter().all(|w| w[1] >= w[0]));
  }

  #[test]
  fn gamma_subordinator_is_non_decreasing() {
    let p = GammaSubordinator::new(3.0_f64, 5.0, 256, Some(0.0), Some(1.0));
    let x = p.sample();
    assert!(x.windows(2).into_iter().all(|w| w[1] >= w[0]));
  }

  #[test]
  fn ig_subordinator_is_non_decreasing() {
    let p = IGSubordinator::new(1.5_f64, 2.0, 256, Some(0.0), Some(1.0));
    let x = p.sample();
    assert!(x.windows(2).into_iter().all(|w| w[1] >= w[0]));
  }

  #[test]
  fn tempered_stable_subordinator_is_non_decreasing() {
    let p = TemperedStableSubordinator::new(0.7_f64, 1.0, 2.0, 0.05, 256, Some(0.0), Some(1.0));
    let x = p.sample();
    assert!(x.windows(2).into_iter().all(|w| w[1] >= w[0]));
  }

  #[test]
  fn ctrw_path_is_finite() {
    let p = CTRW::new(
      CtrwWaitingLaw::Exponential { rate: 2.0_f64 },
      CtrwJumpLaw::Normal {
        mean: 0.0,
        std: 1.0,
      },
      512,
      Some(0.0),
      Some(1.0),
    );
    let x = p.sample();
    assert_eq!(x.len(), 512);
    assert!(x.iter().all(|v| v.is_finite()));
  }
}

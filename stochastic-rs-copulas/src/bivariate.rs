//! # Bivariate
//!
//! $$
//! F_{X_1,\dots,X_d}(x)=C\left(F_1(x_1),\dots,F_d(x_d)\right)
//! $$
//!
pub use crate::traits::BivariateExt;

pub mod clayton;
pub mod fgm;
pub mod frank;
pub mod gumbel;
pub mod independence;
pub mod joe;
pub mod plackett;

#[derive(Debug, Clone, Copy)]
pub enum CopulaType {
  Clayton,
  Fgm,
  Frank,
  Gumbel,
  Independence,
  Joe,
  Plackett,
}

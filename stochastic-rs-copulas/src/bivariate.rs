//! # Bivariate
//!
//! $$
//! F_{X_1,\dots,X_d}(x)=C\left(F_1(x_1),\dots,F_d(x_d)\right)
//! $$
//!
pub use crate::traits::BivariateExt;

pub mod amh;
pub mod clayton;
pub mod fgm;
pub mod frank;
pub mod galambos;
pub mod gumbel;
pub mod husler_reiss;
pub mod independence;
pub mod joe;
pub mod marshall_olkin;
pub mod plackett;
pub mod t_copula;

#[derive(Debug, Clone, Copy)]
pub enum CopulaType {
  Amh,
  Clayton,
  Fgm,
  Frank,
  Galambos,
  Gumbel,
  HuslerReiss,
  Independence,
  Joe,
  MarshallOlkin,
  Plackett,
  TCopula,
}

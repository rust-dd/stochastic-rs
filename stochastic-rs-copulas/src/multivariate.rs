//! # Multivariate
//!
//! $$
//! F_{X_1,\dots,X_d}(x)=C\left(F_1(x_1),\dots,F_d(x_d)\right)
//! $$
//!
pub use crate::traits::MultivariateExt;

pub mod cvine;
pub mod dvine;
pub mod gaussian;
pub mod nac;
pub mod rvine;
pub mod t;
pub mod tree;
pub mod vine;

pub enum CopulaType {
  CVine,
  DVine,
  Gaussian,
  NestedArchimedean,
  RVine,
  TMultivariate,
  Tree,
  Vine,
}

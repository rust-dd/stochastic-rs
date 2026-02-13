pub use crate::traits::MultivariateExt;

pub mod gaussian;
pub mod tree;
pub mod vine;

pub enum CopulaType {
  Gaussian,
  Tree,
  Vine,
}

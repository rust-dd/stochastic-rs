pub use crate::traits::BivariateExt;

pub mod clayton;
pub mod frank;
pub mod gumbel;
pub mod independence;

#[derive(Debug, Clone, Copy)]
pub enum CopulaType {
  Clayton,
  Frank,
  Gumbel,
  Independence,
}

//! # Volatility
//!
//! $$
//! dX_t=a(t,X_t)dt+b(t,X_t)dW_t
//! $$
//!
pub mod bates_svj;
pub mod bergomi;
pub mod fheston;
pub mod heston;
pub mod rbergomi;
pub mod sabr;
pub mod svcgmy;

#[derive(Debug, Clone, Copy, Default)]
pub enum HestonPow {
  #[default]
  Sqrt,
  ThreeHalves,
}

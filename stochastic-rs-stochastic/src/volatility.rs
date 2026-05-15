//! # Volatility
//!
//! $$
//! dX_t=a(t,X_t)dt+b(t,X_t)dW_t
//! $$
//!
pub mod bates_svj;
pub mod bergomi;
pub mod double_heston;
pub mod fbates_svj;
pub mod fheston;
pub mod heston;
pub mod heston2d;
pub mod heston_log;
pub mod hkde;
pub mod rbergomi;
pub mod sabr;
pub mod svcgmy;

#[derive(Debug, Clone, Copy, Default)]
pub enum HestonPow {
  #[default]
  Sqrt,
  ThreeHalves,
}

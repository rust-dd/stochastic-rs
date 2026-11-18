//! Stochastic-volatility models: an asset or forward driven by a positive variance or volatility
//! factor. The Heston and SABR families, Bergomi-style variance, jump variants and rough lifts.
//!
pub mod bergomi;
pub mod bns;
pub mod double_heston;
pub mod fbates_svj;
pub mod fheston;
pub mod heston;
pub mod heston2d;
pub mod heston_log;
pub mod heston_slv;
pub mod hkde;
pub mod multifactor_heston;
pub mod multifactor_sabr;
pub mod rough_bergomi;
pub mod sabr;
pub mod svcgmy;

#[derive(Debug, Clone, Copy, Default)]
pub enum HestonPow {
  #[default]
  Sqrt,
  ThreeHalves,
}

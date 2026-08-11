//! # Volatility
//!
//! Two-factor stochastic-volatility models: an asset (or forward) factor
//! driven by a second, strictly-positive variance/volatility factor with
//! its own mean-reverting or log-normal dynamics, correlated with the
//! asset through an instantaneous ρ or a shared driving noise. Covers the
//! Heston family (square-root variance), SABR family (log-normal
//! volatility, CEV-elastic forward), Bergomi-style log-normal variance,
//! jump-augmented variants (Bates, HKDE), and rough/fractional lifts.
//! Each module's own header states its concrete SDE pair.
//!
pub mod bates_svj;
pub mod bergomi;
pub mod bns;
pub mod double_heston;
pub mod fbates_svj;
pub mod fheston;
pub mod heston;
pub mod heston2d;
pub mod heston_log;
pub mod hkde;
pub mod multifactor_heston;
pub mod multifactor_sabr;
pub mod rbergomi;
pub mod sabr;
pub mod svcgmy;

#[derive(Debug, Clone, Copy, Default)]
pub enum HestonPow {
  #[default]
  Sqrt,
  ThreeHalves,
}

#![doc = include_str!("../README.md")]
// Defaults to `warn`, which is how 7 broken doc links accumulated
// unnoticed; deny so a regression fails the build instead of drifting.
#![deny(rustdoc::broken_intra_doc_links)]

mod input;

pub mod error;
pub mod traits;

pub use error::StatsError;

pub mod acf;
pub mod bayesian_diffusion;
pub mod critical_values;
pub mod distfit;
pub mod econometrics;
pub mod evt;
pub mod filtering;
pub mod fou;
pub mod fractal_dim;
pub mod garch;
pub mod gmm_cir;
pub mod goodness_of_fit;
pub mod heston_mle;
pub mod heston_nmle_cekf;
pub mod hurst;
pub mod kde;
pub mod log_realized_variance_qml;
pub mod mle;
pub mod normality;
pub(crate) mod optim;
pub mod particle_mle;
pub mod qmle;
pub mod realized;
pub mod spectral;
pub mod stationarity;

#[cfg(feature = "python")]
#[doc(hidden)]
pub mod python;

#[cfg(test)]
mod tests {
  #[test]
  fn cusum_result_types_do_not_collide() {
    use crate::econometrics::changepoint::*;
    use crate::stationarity::cusum::*;
    // Both names must be nameable in one scope.
    fn _a(_x: ChangepointCusumResult) {}
    fn _b(_x: CusumTest) {}
  }
}

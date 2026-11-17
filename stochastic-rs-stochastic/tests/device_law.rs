//! Every process the Euler engine serves, checked against its CPU sampler's law: host and device
//! draw different streams, so a terminal statistic over thousands of paths is what can agree.

#![cfg(device_backend)]

mod device_law {
  pub(crate) mod bounded;
  pub(crate) mod coefficients;
  pub(crate) mod common;
  pub(crate) mod conditional_variance;
  pub(crate) mod counts;
  pub(crate) mod curves;
  pub(crate) mod fractional;
  pub(crate) mod gaussian;
  pub(crate) mod jumps;
  pub(crate) mod levy;
  pub(crate) mod memory;
  pub(crate) mod rows;
  pub(crate) mod seed;
  pub(crate) mod series;
  pub(crate) mod sheet;
  pub(crate) mod systems;
}

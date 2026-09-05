//! Every process the Euler engine serves, checked against the law its own CPU
//! sampler produces.
//!
//! A device and the host draw different noise streams — the kernel hashes
//! `(path, step, seed)` where the host runs this crate's SIMD generator — so
//! what a device can be held to is the law, not the path. Each case therefore
//! samples the same process twice, once on the CPU and once on the device,
//! and compares a terminal statistic across several thousand paths, plus
//! whatever the family's own boundary guarantees: non-negativity for a
//! square-root diffusion, `[0, 1]` for the Jacobi and Kimura families,
//! `[−0.9999, 0.9999]` for a bounded correlation.
//!
//! The submodules group by what the comparison has to allow for rather than
//! by source directory: [`gaussian`] holds the unbounded diffusions, whose
//! terminal mean is a stable statistic; [`bounded`] holds the ones whose
//! family clamps, truncates or reflects, where the boundary is as much the
//! point as the mean; [`fractional`] holds the processes whose increments are
//! fractional Gaussian noise, which the device produces itself and keeps in
//! its own memory rather than reading back; [`systems`] holds the ones whose
//! recursion carries more than one state component; [`curves`] holds the
//! ones whose coefficients vary with time; [`levy`] holds the ones whose
//! increment is a draw rather than a step.
//!
//! Split across files only to stay under this crate's line-count limit; all
//! four compile into one test binary.

#![cfg(any(feature = "metal", feature = "cuda"))]

mod device_law {
  pub(crate) mod bounded;
  pub(crate) mod common;
  pub(crate) mod curves;
  pub(crate) mod fractional;
  pub(crate) mod gaussian;
  pub(crate) mod levy;
  pub(crate) mod systems;
}

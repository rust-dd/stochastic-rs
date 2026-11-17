//! The fGN-driven processes reach a device through the Euler engine: the
//! device runs the fractional-noise pipeline itself, keeps the increments in
//! its own memory and hands them to the same families that serve the Gaussian
//! processes. The two sides draw different streams, so what is pinned here is
//! the law and the boundary, not the path.

use stochastic_rs_stochastic::traits::ProcessExt;

use super::common::Device;
use super::common::agrees;
use super::common::all_finite;
use super::common::means_agree;
use super::common::spreads_agree;
use super::common::terminal_mean;
use super::common::terminal_std;
use super::common::within;

/// Fewer paths than the Gaussian cases use: each one carries a 512-point fGN
/// draw, and the comparisons below are on statistics that settle early.
const M: usize = 2_000;
const N: usize = 512;

mod correlated;
mod driven;
mod volterra;

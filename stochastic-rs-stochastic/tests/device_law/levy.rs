//! The processes whose increment is a draw rather than a step: a Poisson
//! count, an inverse-Gaussian subordinator, and Brownian motion under that
//! clock. Each is one expression in the kernel because its sampler needs no
//! rejection — a Poisson count by Knuth's product of uniforms, an
//! inverse-Gaussian draw by Michael-Schucany-Haas — so what these cases pin
//! is that the device draws the same law the host's own sampler draws.

use stochastic_rs_stochastic::traits::ProcessExt;

use super::common::Device;
use super::common::M;
use super::common::agrees;
use super::common::all_finite;
use super::common::means_agree;
use super::common::terminal_mean;
use super::common::terminal_std;
use super::common::within;

const N: usize = 253;

mod arrivals;
mod subordinators;

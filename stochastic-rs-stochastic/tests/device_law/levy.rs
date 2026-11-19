//! The processes whose increment is a draw, not a step: a Poisson count (Knuth's product of
//! uniforms) and an inverse-Gaussian clock (Michael-Schucany-Haas), each rejection-free.

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

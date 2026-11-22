//! The multi-component processes: the engine steps every component in one kernel, so a device
//! must reproduce each component's own law and the boundary its family promises.

use ndarray::Array1;
use stochastic_rs_stochastic::traits::ProcessExt;

use super::common::Device;
use super::common::agrees;
use super::common::slots;

/// Fewer paths than the scalar cases use: a system draw carries two paths.
const M: usize = 3_000;

fn terminal_mean_of<const D: usize>(paths: &[[Array1<f32>; D]], component: usize) -> f64 {
  let last = paths[0][component].len() - 1;
  paths.iter().map(|p| p[component][last] as f64).sum::<f64>() / paths.len() as f64
}

fn terminal_mean(paths: &[[Array1<f32>; 2]], component: usize) -> f64 {
  let last = paths[0][component].len() - 1;
  paths.iter().map(|p| p[component][last] as f64).sum::<f64>() / paths.len() as f64
}

mod affine;
mod heston;
mod heston_hull_white_2f;
mod jumps;
mod regimes;
mod volatility;

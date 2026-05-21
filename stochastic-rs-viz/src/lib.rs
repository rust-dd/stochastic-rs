//! # stochastic-rs-viz
//!
//! Plotly-based visualization for stochastic processes and distributions.
//!
//! Module layout:
//! - [`plottable`] — `Plottable<T>` trait + impls for the canonical
//!   `ProcessExt::Output` shapes (1D path, complex path, fixed-arity tuple,
//!   2D matrix).
//! - [`grid_plotter`] — `GridPlotter` builder for multi-subplot HTML grids.
//! - [`convenience`] — one-shot `plot_process` / `plot_distribution` /
//!   `plot_vol_surface` HTML writers.

#![allow(non_snake_case)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]

pub mod convenience;
pub mod grid_plotter;
pub mod plottable;

pub use convenience::plot_distribution;
pub use convenience::plot_process;
pub use convenience::plot_vol_surface;
pub use grid_plotter::GridPlotter;
pub use plottable::Plottable;

#[cfg(test)]
mod tests;

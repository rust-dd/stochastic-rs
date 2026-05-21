use plotly::Layout;
use plotly::Plot;
use plotly::Surface;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::sheet::fbs::Fbs;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::HestonPow;
use stochastic_rs_stochastic::volatility::bergomi::Bergomi;
use stochastic_rs_stochastic::volatility::fheston::RoughHeston;
use stochastic_rs_stochastic::volatility::heston::Heston;
use stochastic_rs_stochastic::volatility::rbergomi::RoughBergomi;
use stochastic_rs_stochastic::volatility::sabr::Sabr;
use stochastic_rs_stochastic::volatility::svcgmy::Svcgmy;

use crate::GridPlotter;

pub(crate) fn register_volatility(
  mut grid: GridPlotter,
  n: usize,
  j: usize,
  traj: usize,
) -> GridPlotter {
  grid = grid.register(
    &Heston::new(
      Some(100.0),
      Some(0.04),
      2.0,
      0.04,
      0.3,
      -0.7,
      0.05,
      n,
      Some(1.0),
      HestonPow::Sqrt,
      Some(false),
      Unseeded,
    ),
    "Volatility: Heston",
    traj,
  );
  grid = grid.register(
    &Bergomi::new(
      0.4,
      Some(0.2),
      Some(100.0),
      0.01,
      -0.6,
      n,
      Some(1.0),
      Unseeded,
    ),
    "Volatility: Bergomi",
    traj,
  );
  grid = grid.register(
    &RoughBergomi::new(
      0.1,
      0.4,
      Some(0.2),
      Some(100.0),
      0.01,
      -0.6,
      n,
      Some(1.0),
      Unseeded,
    ),
    "Volatility: Rough Bergomi",
    traj,
  );
  grid = grid.register(
    &RoughHeston::new(
      0.8,
      Some(0.2),
      0.04,
      1.5,
      0.3,
      None,
      None,
      Some(1.0),
      n,
      Unseeded,
    ),
    "Volatility: Rough Heston",
    traj,
  );
  grid = grid.register(
    &Sabr::new(0.4, 0.7, -0.3, n, Some(1.0), Some(0.3), Some(1.0), Unseeded),
    "Volatility: Sabr",
    traj,
  );
  grid = grid.register(
    &Svcgmy::new(
      3.0,
      4.0,
      0.7,
      1.5,
      0.04,
      0.3,
      -0.4,
      n,
      j,
      Some(0.0),
      Some(0.04),
      Some(1.0),
      Unseeded,
    ),
    "Volatility: Svcgmy",
    traj,
  );
  grid
}

pub(crate) fn show_fbs_sheet(sheet_m: usize, sheet_n: usize) {
  let fbs_field = Fbs::new(0.7, sheet_m, sheet_n, 2.0, Unseeded).sample();
  let z: Vec<Vec<f64>> = fbs_field.outer_iter().map(|row| row.to_vec()).collect();
  let x: Vec<f64> = (0..sheet_n)
    .map(|i| i as f64 / (sheet_n.saturating_sub(1).max(1) as f64))
    .collect();
  let y: Vec<f64> = (0..sheet_m)
    .map(|i| i as f64 / (sheet_m.saturating_sub(1).max(1) as f64))
    .collect();

  let mut sheet_plot = Plot::new();
  let surface = Surface::new(z).x(x).y(y).name("Sheet: Fbs");
  sheet_plot.add_trace(surface);
  sheet_plot.set_layout(
    Layout::new()
      .title("Sheet: Fbs (3D Surface)")
      .height(900)
      .width(1200),
  );
  sheet_plot.show();
}

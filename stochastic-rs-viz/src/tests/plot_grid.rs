use crate::GridPlotter;
use crate::tests::categories::autoregressive;
use crate::tests::categories::diffusion;
use crate::tests::categories::interest;
use crate::tests::categories::jump;
use crate::tests::categories::noise_and_process;
use crate::tests::categories::volatility_and_sheet;

#[test]
fn plot_grid() {
  let n = 96;
  let traj = 1;
  let j = 64;
  let sheet_m = 3;
  let sheet_n = 64;

  let mut grid = GridPlotter::new()
    .title("Stochastic Processes (Grid)")
    .cols(4)
    .show_legend(false)
    .line_width(1.2)
    .x_gap(0.80)
    .y_gap(5.00);

  grid = autoregressive::register(grid, n, traj);
  grid = noise_and_process::register_noise(grid, n, traj);
  grid = noise_and_process::register_process(grid, n, traj);
  grid = diffusion::register(grid, n, traj);
  grid = interest::register(grid, n, traj);
  grid = jump::register(grid, n, j, traj);
  grid = volatility_and_sheet::register_volatility(grid, n, j, traj);

  grid.show();

  volatility_and_sheet::show_fbs_sheet(sheet_m, sheet_n);
}

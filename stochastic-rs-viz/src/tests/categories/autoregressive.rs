use ndarray::Array1;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::autoregressive::agrach::Agarch;
use stochastic_rs_stochastic::autoregressive::ar::ARp;
use stochastic_rs_stochastic::autoregressive::arch::Arch;
use stochastic_rs_stochastic::autoregressive::arima::Arima;
use stochastic_rs_stochastic::autoregressive::egarch::Egarch;
use stochastic_rs_stochastic::autoregressive::garch::Garch;
use stochastic_rs_stochastic::autoregressive::ma::MAq;
use stochastic_rs_stochastic::autoregressive::sarima::Sarima;
use stochastic_rs_stochastic::autoregressive::tgarch::Tgarch;

use crate::GridPlotter;

pub(crate) fn register(mut grid: GridPlotter, n: usize, traj: usize) -> GridPlotter {
  grid = grid.register(
    &ARp::new(Array1::from_vec(vec![0.65, -0.2]), 0.08, n, None, Unseeded),
    "Autoreg: AR(2)",
    traj,
  );
  grid = grid.register(
    &MAq::new(Array1::from_vec(vec![0.5, -0.2]), 0.1, n, Unseeded),
    "Autoreg: MA(2)",
    traj,
  );
  grid = grid.register(
    &Arima::new(
      Array1::from_vec(vec![0.4]),
      Array1::from_vec(vec![0.3]),
      1,
      0.1,
      n,
      Unseeded,
    ),
    "Autoreg: Arima(1,1,1)",
    traj,
  );
  grid = grid.register(
    &Sarima::new(
      Array1::from_vec(vec![0.3]),
      Array1::from_vec(vec![0.2]),
      Array1::from_vec(vec![0.2]),
      Array1::from_vec(vec![0.1]),
      1,
      1,
      12,
      0.08,
      n,
      Unseeded,
    ),
    "Autoreg: Sarima",
    traj,
  );
  grid = grid.register(
    &Arch::new(0.05, Array1::from_vec(vec![0.2, 0.1]), n, Unseeded),
    "Autoreg: Arch",
    traj,
  );
  grid = grid.register(
    &Garch::new(
      0.03,
      Array1::from_vec(vec![0.12]),
      Array1::from_vec(vec![0.8]),
      n,
      Unseeded,
    ),
    "Autoreg: Garch",
    traj,
  );
  grid = grid.register(
    &Tgarch::new(
      0.03,
      Array1::from_vec(vec![0.08]),
      Array1::from_vec(vec![0.05]),
      Array1::from_vec(vec![0.85]),
      n,
      Unseeded,
    ),
    "Autoreg: Tgarch",
    traj,
  );
  grid = grid.register(
    &Egarch::new(
      -0.1,
      Array1::from_vec(vec![0.1]),
      Array1::from_vec(vec![-0.05]),
      Array1::from_vec(vec![0.9]),
      n,
      Unseeded,
    ),
    "Autoreg: Egarch",
    traj,
  );
  grid = grid.register(
    &Agarch::new(
      0.03,
      Array1::from_vec(vec![0.1]),
      Array1::from_vec(vec![0.04]),
      Array1::from_vec(vec![0.84]),
      n,
      Unseeded,
    ),
    "Autoreg: Agarch",
    traj,
  );
  grid
}

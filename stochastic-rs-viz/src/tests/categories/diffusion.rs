use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::diffusion::cev::Cev;
use stochastic_rs_stochastic::diffusion::cfou::Cfou;
use stochastic_rs_stochastic::diffusion::cir::Cir as DiffCIR;
use stochastic_rs_stochastic::diffusion::fcir::Fcir;
use stochastic_rs_stochastic::diffusion::feller::FellerLogistic;
use stochastic_rs_stochastic::diffusion::fgbm::Fgbm;
use stochastic_rs_stochastic::diffusion::fjacobi::FJacobi;
use stochastic_rs_stochastic::diffusion::fou::Fou;
use stochastic_rs_stochastic::diffusion::fouque::FouqueOU2D;
use stochastic_rs_stochastic::diffusion::gbm::Gbm;
use stochastic_rs_stochastic::diffusion::gbm_ih::GbmIh;
use stochastic_rs_stochastic::diffusion::gompertz::Gompertz;
use stochastic_rs_stochastic::diffusion::jacobi::Jacobi;
use stochastic_rs_stochastic::diffusion::kimura::Kimura;
use stochastic_rs_stochastic::diffusion::ou::Ou;
use stochastic_rs_stochastic::diffusion::quadratic::Quadratic;
use stochastic_rs_stochastic::diffusion::verhulst::Verhulst;

use crate::GridPlotter;

pub(crate) fn register(mut grid: GridPlotter, n: usize, traj: usize) -> GridPlotter {
  grid = grid.register(
    &Ou::new(2.0, 0.0, 0.2, n, Some(0.0), Some(1.0), Unseeded),
    "Diffusion: Ou",
    traj,
  );
  grid = grid.register(
    &Gbm::new(0.05, 0.2, n, Some(100.0), Some(1.0), Unseeded),
    "Diffusion: Gbm",
    traj,
  );
  grid = grid.register(
    &DiffCIR::new(
      2.5,
      0.04,
      0.2,
      n,
      Some(0.04),
      Some(1.0),
      Some(false),
      Unseeded,
    ),
    "Diffusion: Cir",
    traj,
  );
  grid = grid.register(
    &Cev::new(0.04, 0.2, 0.8, n, Some(1.0), Some(1.0), Unseeded),
    "Diffusion: Cev",
    traj,
  );
  grid = grid.register(
    &FellerLogistic::new(
      2.0,
      1.0,
      0.3,
      n,
      Some(0.5),
      Some(1.0),
      Some(false),
      Unseeded,
    ),
    "Diffusion: Feller Logistic",
    traj,
  );
  grid = grid.register(
    &Verhulst::new(1.2, 2.0, 0.2, n, Some(0.5), Some(1.0), Some(true), Unseeded),
    "Diffusion: Verhulst",
    traj,
  );
  grid = grid.register(
    &Gompertz::new(1.0, 0.8, 0.2, n, Some(1.0), Some(1.0), Unseeded),
    "Diffusion: Gompertz",
    traj,
  );
  grid = grid.register(
    &Kimura::new(1.0, 0.3, n, Some(0.4), Some(1.0), Unseeded),
    "Diffusion: Kimura",
    traj,
  );
  grid = grid.register(
    &Quadratic::new(0.1, -0.2, 0.05, 0.15, n, Some(1.0), Some(1.0), Unseeded),
    "Diffusion: Quadratic",
    traj,
  );
  grid = grid.register(
    &Jacobi::new(0.8, 1.4, 0.4, n, Some(0.3), Some(1.0), Unseeded),
    "Diffusion: Jacobi",
    traj,
  );
  grid = grid.register(
    &Fcir::new(
      0.7,
      2.5,
      0.04,
      0.2,
      n,
      Some(0.04),
      Some(1.0),
      Some(false),
      Unseeded,
    ),
    "Diffusion: Fcir",
    traj,
  );
  grid = grid.register(
    &FJacobi::new(0.7, 0.8, 1.4, 0.35, n, Some(0.3), Some(1.0), Unseeded),
    "Diffusion: FJacobi",
    traj,
  );
  grid = grid.register(
    &Fou::new(0.7, 2.0, 0.0, 0.2, n, Some(0.0), Some(1.0), Unseeded),
    "Diffusion: Fou",
    traj,
  );
  grid = grid.register(
    &Cfou::new(
      0.7,
      1.8,
      3.0,
      0.4,
      n,
      Some(0.0),
      Some(0.0),
      Some(1.0),
      Unseeded,
    ),
    "Diffusion: Complex fOU",
    traj,
  );
  grid = grid.register(
    &Fgbm::new(0.7, 0.04, 0.2, n, Some(100.0), Some(1.0), Unseeded),
    "Diffusion: Fgbm",
    traj,
  );
  grid = grid.register(
    &GbmIh::new(0.04, 0.2, n, Some(100.0), Some(1.0), None, Unseeded),
    "Diffusion: GbmIh",
    traj,
  );
  grid = grid.register(
    &FouqueOU2D::new(
      1.5,
      0.0,
      0.3,
      0.0,
      n,
      Some(0.0),
      Some(0.0),
      Some(1.0),
      Unseeded,
    ),
    "Diffusion: Fouque Ou 2D",
    traj,
  );
  grid
}

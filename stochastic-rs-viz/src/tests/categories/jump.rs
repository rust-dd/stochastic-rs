use rand_distr::Exp;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::jump::bates::Bates1996;
use stochastic_rs_stochastic::jump::cgmy::Cgmy;
use stochastic_rs_stochastic::jump::cts::Cts;
use stochastic_rs_stochastic::jump::ig::Ig;
use stochastic_rs_stochastic::jump::jump_fou::JumpFou;
use stochastic_rs_stochastic::jump::jump_fou_custom::JumpFOUCustom;
use stochastic_rs_stochastic::jump::kobol::KoBoL;
use stochastic_rs_stochastic::jump::kou::Kou;
use stochastic_rs_stochastic::jump::levy_diffusion::LevyDiffusion;
use stochastic_rs_stochastic::jump::merton::Merton;
use stochastic_rs_stochastic::jump::nig::Nig;
use stochastic_rs_stochastic::jump::rdts::Rdts;
use stochastic_rs_stochastic::jump::vg::Vg;

use crate::GridPlotter;
use crate::tests::normal_cpoisson;

pub(crate) fn register(mut grid: GridPlotter, n: usize, j: usize, traj: usize) -> GridPlotter {
  grid = grid.register(
    &Vg::new(0.0, 0.2, 0.15, n, Some(0.0), Some(1.0), Unseeded),
    "Jump: Vg",
    traj,
  );
  grid = grid.register(
    &Nig::new(0.0, 0.2, 0.3, n, Some(0.0), Some(1.0), Unseeded),
    "Jump: Nig",
    traj,
  );
  grid = grid.register(
    &Ig::new(1.0, n, Some(0.0), Some(1.0), Unseeded),
    "Jump: Ig",
    traj,
  );
  grid = grid.register(
    &Rdts::new(4.0, 5.0, 0.7, n, j, Some(0.0), Some(1.0), Unseeded),
    "Jump: Rdts",
    traj,
  );
  grid = grid.register(
    &Cts::new(4.0, 5.0, 0.7, n, j, Some(0.0), Some(1.0), Unseeded),
    "Jump: Cts",
    traj,
  );

  let g = 4.0;
  let m = 5.0;
  let y = 0.7;
  let c = Cgmy::<f64>::c_for_unit_variance(g, m, y);
  // KoBoL: when p=q=1, d_for_unit_variance == c_for_unit_variance.
  let d = KoBoL::<f64>::d_for_unit_variance(1.0, 1.0, g, m, y);

  grid = grid.register(
    &Cgmy::<f64>::new(c, g, m, y, n, j, Some(0.0), Some(1.0), Unseeded),
    "Jump: Cgmy (unit var, symmetric)",
    traj,
  );
  grid = grid.register(
    &KoBoL::<f64>::new(d, 1.0, 1.0, g, m, y, n, j, Some(0.0), Some(1.0), Unseeded),
    "Jump: KoBoL (unit var, p=q=1)",
    traj,
  );

  grid = grid.register(
    &Merton::new(
      0.03,
      0.2,
      1.0,
      0.0,
      n,
      Some(0.0),
      Some(1.0),
      normal_cpoisson(1.0, n, 0.1),
      Unseeded,
    ),
    "Jump: Merton",
    traj,
  );
  grid = grid.register(
    &Kou::new(
      0.03,
      0.2,
      1.0,
      0.0,
      n,
      Some(0.0),
      Some(1.0),
      normal_cpoisson(1.0, n, 0.12),
      Unseeded,
    ),
    "Jump: Kou",
    traj,
  );
  grid = grid.register(
    &LevyDiffusion::new(
      0.01,
      0.2,
      n,
      Some(0.0),
      Some(1.0),
      normal_cpoisson(1.0, n, 0.08),
      Unseeded,
    ),
    "Jump: Levy Diffusion",
    traj,
  );
  grid = grid.register(
    &JumpFou::new(
      0.7,
      2.0,
      0.03,
      0.2,
      n,
      Some(0.03),
      Some(1.0),
      normal_cpoisson(1.0, n, 0.08),
      Unseeded,
    ),
    "Jump: Jump-Fou",
    traj,
  );
  grid = grid.register(
    &JumpFOUCustom::new(
      0.7,
      2.0,
      0.03,
      0.2,
      n,
      Some(0.03),
      Some(1.0),
      Exp::new(20.0).expect("positive exponential rate"),
      Exp::new(30.0).expect("positive exponential rate"),
      Unseeded,
    ),
    "Jump: Jump-Fou Custom",
    traj,
  );
  grid = grid.register_with_component_labels(
    &Bates1996::new(
      Some(0.03),
      None,
      None,
      None,
      0.8,
      0.0,
      1.5,
      0.8,
      0.3,
      -0.5,
      n,
      Some(100.0),
      Some(0.04),
      Some(1.0),
      Some(false),
      normal_cpoisson(0.8, n, 0.05),
      Unseeded,
    ),
    "Jump: Bates 1996 (S: solid, v: dashed)",
    &["S", "v"],
    traj,
  );
  grid
}

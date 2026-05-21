use ndarray::Array1;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::interest::adg::Adg;
use stochastic_rs_stochastic::interest::bgm::Bgm;
use stochastic_rs_stochastic::interest::cir::Cir as RateCIR;
use stochastic_rs_stochastic::interest::cir_2f::Cir2F;
use stochastic_rs_stochastic::interest::duffie_kan::DuffieKan;
use stochastic_rs_stochastic::interest::duffie_kan_jump_exp::DuffieKanJumpExp;
use stochastic_rs_stochastic::interest::fractional_vasicek::FVasicek;
use stochastic_rs_stochastic::interest::hjm::Hjm;
use stochastic_rs_stochastic::interest::ho_lee::HoLee;
use stochastic_rs_stochastic::interest::hull_white::HullWhite;
use stochastic_rs_stochastic::interest::hull_white_2f::HullWhite2F;
use stochastic_rs_stochastic::interest::vasicek::Vasicek;
use stochastic_rs_stochastic::interest::wu_zhang::WuZhangD;

use crate::GridPlotter;
use crate::tests::{
  f_adg_b, f_adg_c, f_adg_k, f_adg_phi, f_adg_theta, f_const_001, f_const_002, f_hjm_alpha,
  f_hjm_p, f_hjm_q, f_hjm_sigma, f_hjm_v, f_linear_small, f_phi_small,
};

pub(crate) fn register(mut grid: GridPlotter, n: usize, traj: usize) -> GridPlotter {
  grid = grid.register(
    &Vasicek::new(3.0, 0.03, 0.02, n, Some(0.03), Some(1.0), Unseeded),
    "Interest: Vasicek",
    traj,
  );
  grid = grid.register(
    &FVasicek::new(0.7, 2.0, 0.03, 0.02, n, Some(0.03), Some(1.0), Unseeded),
    "Interest: Fractional Vasicek",
    traj,
  );
  grid = grid.register(
    &RateCIR::new(
      2.5,
      0.04,
      0.2,
      n,
      Some(0.04),
      Some(1.0),
      Some(false),
      Unseeded,
    ),
    "Interest: Cir (Alias)",
    traj,
  );
  grid = grid.register(
    &HoLee::new(None, Some(0.01), 0.01, n, Some(1.0), Unseeded),
    "Interest: Ho-Lee",
    traj,
  );
  grid = grid.register(
    &HullWhite::new(
      f_linear_small as fn(f64) -> f64,
      0.4,
      0.02,
      n,
      Some(0.02),
      Some(1.0),
      Unseeded,
    ),
    "Interest: Hull-White",
    traj,
  );
  grid = grid.register(
    &HullWhite2F::new(
      f_const_001 as fn(f64) -> f64,
      0.5,
      0.02,
      0.015,
      -0.3,
      0.4,
      Some(0.02),
      Some(1.0),
      n,
      Unseeded,
    ),
    "Interest: Hull-White 2F",
    traj,
  );
  grid = grid.register(
    &Hjm::new(
      f_const_001 as fn(f64) -> f64,
      f_const_002 as fn(f64) -> f64,
      f_hjm_p as fn(f64, f64) -> f64,
      f_hjm_q as fn(f64, f64) -> f64,
      f_hjm_v as fn(f64, f64) -> f64,
      f_hjm_alpha as fn(f64, f64) -> f64,
      f_hjm_sigma as fn(f64, f64) -> f64,
      n,
      Some(0.01),
      Some(1.0),
      Some(0.01),
      Some(1.0),
      Unseeded,
    ),
    "Interest: Hjm",
    traj,
  );
  grid = grid.register(
    &Bgm::new(
      Array1::from_vec(vec![0.2, 0.15]),
      Array1::from_vec(vec![0.02, 0.025]),
      2,
      Some(1.0),
      n,
      Unseeded,
    ),
    "Interest: Bgm",
    traj,
  );
  grid = grid.register(
    &Adg::new(
      f_adg_k as fn(f64) -> f64,
      f_adg_theta as fn(f64) -> f64,
      Array1::from_vec(vec![0.02, 0.018]),
      f_adg_phi as fn(f64) -> f64,
      f_adg_b as fn(f64) -> f64,
      f_adg_c as fn(f64) -> f64,
      n,
      2,
      Array1::from_vec(vec![0.01, 0.015]),
      Some(1.0),
      Unseeded,
    ),
    "Interest: Adg",
    traj,
  );
  grid = grid.register(
    &DuffieKan::new(
      0.2,
      0.1,
      0.05,
      -0.3,
      -0.1,
      0.2,
      0.01,
      0.1,
      0.15,
      -0.2,
      0.01,
      0.12,
      n,
      Some(0.02),
      Some(0.01),
      Some(1.0),
      Unseeded,
    ),
    "Interest: Duffie-Kan",
    traj,
  );
  grid = grid.register(
    &DuffieKanJumpExp::new(
      0.2,
      0.1,
      0.05,
      -0.3,
      -0.1,
      0.2,
      0.01,
      0.1,
      0.15,
      -0.2,
      0.01,
      0.12,
      2.0,
      0.02,
      n,
      Some(0.02),
      Some(0.01),
      Some(1.0),
      Unseeded,
    ),
    "Interest: Duffie-Kan Jump Exp",
    traj,
  );
  grid = grid.register(
    &WuZhangD::new(
      Array1::from_vec(vec![0.05, 0.04]),
      Array1::from_vec(vec![1.2, 1.0]),
      Array1::from_vec(vec![0.3, 0.25]),
      Array1::from_vec(vec![0.4, 0.3]),
      Array1::from_vec(vec![0.02, 0.025]),
      Array1::from_vec(vec![0.04, 0.03]),
      2,
      Some(1.0),
      n,
      Unseeded,
    ),
    "Interest: Wu-Zhang",
    traj,
  );
  grid = grid.register(
    &Cir2F::new(
      RateCIR::new(
        2.5,
        0.03,
        0.12,
        n,
        Some(0.03),
        Some(1.0),
        Some(false),
        Unseeded,
      ),
      RateCIR::new(
        2.0,
        0.02,
        0.1,
        n,
        Some(0.02),
        Some(1.0),
        Some(false),
        Unseeded,
      ),
      f_phi_small as fn(f64) -> f64,
      Unseeded,
    ),
    "Interest: Cir 2F",
    traj,
  );
  grid
}

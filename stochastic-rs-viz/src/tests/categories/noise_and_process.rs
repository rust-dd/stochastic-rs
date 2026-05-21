use rand_distr::Exp;
use rand_distr::Normal;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::isonormal::IsoNormal;
use stochastic_rs_stochastic::isonormal::fbm_custom_inc_cov;
use stochastic_rs_stochastic::noise::cfgns::Cfgns;
use stochastic_rs_stochastic::noise::cgns::Cgns;
use stochastic_rs_stochastic::noise::fgn::Fgn;
use stochastic_rs_stochastic::noise::gn::Gn;
use stochastic_rs_stochastic::noise::wn::Wn;
use stochastic_rs_stochastic::process::bm::Bm;
use stochastic_rs_stochastic::process::cbms::Cbms;
use stochastic_rs_stochastic::process::ccustom::CompoundCustom;
use stochastic_rs_stochastic::process::cfbms::Cfbms;
use stochastic_rs_stochastic::process::cpoisson::CompoundPoisson;
use stochastic_rs_stochastic::process::customjt::CustomJt;
use stochastic_rs_stochastic::process::fbm::Fbm;
use stochastic_rs_stochastic::process::lfsm::Lfsm;
use stochastic_rs_stochastic::process::poisson::Poisson;
use stochastic_rs_stochastic::process::subordinator::AlphaStableSubordinator;
use stochastic_rs_stochastic::process::subordinator::Ctrw;
use stochastic_rs_stochastic::process::subordinator::CtrwJumpLaw;
use stochastic_rs_stochastic::process::subordinator::CtrwWaitingLaw;
use stochastic_rs_stochastic::process::subordinator::GammaSubordinator;
use stochastic_rs_stochastic::process::subordinator::IGSubordinator;
use stochastic_rs_stochastic::process::subordinator::InverseAlphaStableSubordinator;
use stochastic_rs_stochastic::process::subordinator::PoissonSubordinator;
use stochastic_rs_stochastic::process::subordinator::TemperedStableSubordinator;

use crate::GridPlotter;

pub(crate) fn register_noise(mut grid: GridPlotter, n: usize, traj: usize) -> GridPlotter {
  grid = grid.register(
    &Wn::new(n, Some(0.0), Some(1.0), Unseeded),
    "Noise: White",
    traj,
  );
  grid = grid.register(&Gn::new(n, Some(1.0), Unseeded), "Noise: Gaussian", traj);
  grid = grid.register(&Fgn::new(0.7, n, Some(1.0), Unseeded), "Noise: Fgn", traj);
  grid = grid.register(
    &Cgns::new(-0.4, n, Some(1.0), Unseeded),
    "Noise: Cgns",
    traj,
  );
  grid = grid.register(
    &Cfgns::new(0.7, -0.3, n, Some(1.0), Unseeded),
    "Noise: Cfgns",
    traj,
  );
  grid
}

pub(crate) fn isonormal_fbm_paths(n: usize, traj: usize) -> Vec<Vec<f64>> {
  let mut isonormal_fbm = IsoNormal::new(
    |aux_idx, idx| fbm_custom_inc_cov(aux_idx.abs_diff(idx), 0.7),
    (0..n).collect(),
  );
  let mut isonormal_paths = Vec::with_capacity(traj);
  for _ in 0..traj {
    let increments = isonormal_fbm.get_path();
    let mut path = Vec::with_capacity(n);
    path.push(0.0);
    let mut acc = 0.0;
    for &dx in &increments {
      acc += dx;
      path.push(acc);
    }
    isonormal_paths.push(path);
  }
  isonormal_paths
}

pub(crate) fn register_process(mut grid: GridPlotter, n: usize, traj: usize) -> GridPlotter {
  grid = grid.register(&Bm::new(n, Some(1.0), Unseeded), "Process: Bm", traj);
  grid = grid.register(&Fbm::new(0.7, n, Some(1.0), Unseeded), "Process: Fbm", traj);
  let isonormal_paths = isonormal_fbm_paths(n, traj);
  grid = grid.register_paths(isonormal_paths, "Process: fBM via IsoNormal (H=0.7)");
  grid = grid.register(
    &Poisson::new(2.0, Some(n), Some(1.0), Unseeded),
    "Process: Poisson",
    traj,
  );
  grid = grid.register(
    &CustomJt::new(
      Some(n),
      Some(1.0),
      Exp::new(10.0).expect("positive exponential rate"),
      Unseeded,
    ),
    "Process: CustomJt",
    traj,
  );
  grid = grid.register(
    &CompoundPoisson::new(
      Normal::new(0.0, 0.15).expect("valid normal"),
      Poisson::new(1.2, Some(n), Some(1.0), Unseeded),
      Unseeded,
    ),
    "Process: CompoundPoisson",
    traj,
  );
  grid = grid.register(
    &CompoundCustom::new(
      Some(n),
      Some(1.0),
      Normal::new(0.0, 0.1).expect("valid normal"),
      Exp::new(15.0).expect("positive exponential rate"),
      CustomJt::new(
        Some(n),
        Some(1.0),
        Exp::new(15.0).expect("positive exponential rate"),
        Unseeded,
      ),
      Unseeded,
    ),
    "Process: CompoundCustom",
    traj,
  );
  grid = grid.register(
    &Cbms::new(0.35, n, Some(1.0), Unseeded),
    "Process: Cbms",
    traj,
  );
  grid = grid.register(
    &Cfbms::new(0.7, 0.35, n, Some(1.0), Unseeded),
    "Process: Cfbms",
    traj,
  );
  grid = grid.register(
    &Lfsm::new(1.7, 0.0, 0.8, 1.0, n, Some(0.0), Some(1.0), Unseeded),
    "Process: Lfsm",
    traj,
  );
  grid = grid.register(
    &AlphaStableSubordinator::new(0.7, 1.0, n, Some(0.0), Some(1.0), Unseeded),
    "Process: AlphaStable Subordinator",
    traj,
  );
  grid = grid.register(
    &InverseAlphaStableSubordinator::new(0.7, 1.0, n, Some(1.0), 2048, Some(4.0), Unseeded),
    "Process: Inverse AlphaStable",
    traj,
  );
  grid = grid.register(
    &PoissonSubordinator::new(2.0, n, Some(0.0), Some(1.0), Unseeded),
    "Process: Poisson Subordinator",
    traj,
  );
  grid = grid.register(
    &GammaSubordinator::new(3.0, 5.0, n, Some(0.0), Some(1.0), Unseeded),
    "Process: Gamma Subordinator",
    traj,
  );
  grid = grid.register(
    &IGSubordinator::new(1.5, 2.0, n, Some(0.0), Some(1.0), Unseeded),
    "Process: Ig Subordinator",
    traj,
  );
  grid = grid.register(
    &TemperedStableSubordinator::new(0.7, 1.0, 2.0, 0.05, n, Some(0.0), Some(1.0), Unseeded),
    "Process: Tempered Stable Subordinator",
    traj,
  );
  grid = grid.register(
    &Ctrw::new(
      CtrwWaitingLaw::Exponential { rate: 2.0 },
      CtrwJumpLaw::Normal {
        mean: 0.0,
        std: 0.3,
      },
      n,
      Some(0.0),
      Some(1.0),
      Unseeded,
    ),
    "Process: Ctrw",
    traj,
  );
  grid
}

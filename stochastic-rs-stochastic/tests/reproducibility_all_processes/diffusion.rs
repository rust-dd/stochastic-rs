//! `diffusion/` slice of the exhaustive reproducibility guard — see
//! `../reproducibility_all_processes.rs` for the full rationale, the
//! derivation of the 131-type list, and shared methodology notes.

use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_stochastic::diffusion::ait_sahalia::AitSahalia;
use stochastic_rs_stochastic::diffusion::bessel::Bessel;
use stochastic_rs_stochastic::diffusion::bessel::SquaredBessel;
use stochastic_rs_stochastic::diffusion::cev::Cev;
use stochastic_rs_stochastic::diffusion::cfou::Cfou;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::diffusion::ckls::Ckls;
use stochastic_rs_stochastic::diffusion::displaced_diffusion::DisplacedDiffusion;
use stochastic_rs_stochastic::diffusion::fcir::Fcir;
use stochastic_rs_stochastic::diffusion::feller::FellerLogistic;
use stochastic_rs_stochastic::diffusion::feller_root::FellerRoot;
use stochastic_rs_stochastic::diffusion::fgbm::Fgbm;
use stochastic_rs_stochastic::diffusion::fjacobi::FJacobi;
use stochastic_rs_stochastic::diffusion::fou::Fou;
use stochastic_rs_stochastic::diffusion::fouque::FouqueOU2D;
use stochastic_rs_stochastic::diffusion::gbm::Gbm;
use stochastic_rs_stochastic::diffusion::gbm_ih::GbmIh;
use stochastic_rs_stochastic::diffusion::gbm_log::GbmLog;
use stochastic_rs_stochastic::diffusion::gompertz::Gompertz;
use stochastic_rs_stochastic::diffusion::hyperbolic::Hyperbolic;
use stochastic_rs_stochastic::diffusion::hyperbolic2::Hyperbolic2;
use stochastic_rs_stochastic::diffusion::jacobi::Jacobi;
use stochastic_rs_stochastic::diffusion::kimura::Kimura;
use stochastic_rs_stochastic::diffusion::linear_sde::LinearSDE;
use stochastic_rs_stochastic::diffusion::logistic::Logistic;
use stochastic_rs_stochastic::diffusion::modified_cir::ModifiedCIR;
use stochastic_rs_stochastic::diffusion::multi_gbm::MultiGbm;
use stochastic_rs_stochastic::diffusion::nonlinear_sde::NonLinearSDE;
use stochastic_rs_stochastic::diffusion::ou::Ou;
use stochastic_rs_stochastic::diffusion::pearson::Pearson;
use stochastic_rs_stochastic::diffusion::quadratic::Quadratic;
use stochastic_rs_stochastic::diffusion::radial_ou::RadialOU;
use stochastic_rs_stochastic::diffusion::regime_switching::RegimeSwitchingDiffusion;
use stochastic_rs_stochastic::diffusion::three_half::ThreeHalf;
use stochastic_rs_stochastic::diffusion::verhulst::Verhulst;
use stochastic_rs_stochastic::diffusion::wishart::Wishart;

use crate::common::N;
use crate::common::guard;

guard!(ait_sahalia, "AitSahalia", |s| AitSahalia::new(
  0.01,
  0.02,
  -0.1,
  0.0,
  0.1,
  0.0,
  0.0,
  0.0,
  N,
  Some(0.05),
  Some(1.0),
  s
));

guard!(squared_bessel, "SquaredBessel", |s| SquaredBessel::new(
  3.0,
  N,
  Some(1.0),
  Some(1.0),
  None,
  s
));

guard!(bessel, "Bessel", |s| Bessel::new(
  3.0,
  N,
  Some(1.0),
  Some(1.0),
  None,
  s
));

guard!(cev, "Cev", |s| Cev::new(
  0.05,
  0.2,
  0.5,
  N,
  Some(100.0),
  Some(1.0),
  s
));

guard!(cfou, "Cfou", |s| Cfou::new(
  0.7,
  1.0,
  0.5,
  1.0,
  N,
  Some(0.0),
  Some(0.0),
  Some(1.0),
  s
));

guard!(cir, "Cir", |s| Cir::new(
  1.0,
  0.04,
  0.2,
  N,
  Some(0.04),
  Some(1.0),
  Some(false),
  s
));

guard!(ckls, "Ckls", |s| Ckls::new(
  0.02,
  -0.1,
  0.2,
  0.5,
  N,
  Some(0.05),
  Some(1.0),
  s
));

guard!(displaced_diffusion, "DisplacedDiffusion", |s| {
  DisplacedDiffusion::new(0.05, 0.2, 0.3, N, Some(100.0), Some(1.0), s)
});

guard!(fcir, "Fcir", |s| Fcir::new(
  0.7,
  1.0,
  0.04,
  0.2,
  N,
  Some(0.04),
  Some(1.0),
  Some(false),
  s
));

guard!(feller_logistic, "FellerLogistic", |s| FellerLogistic::new(
  1.0,
  0.04,
  0.2,
  N,
  Some(0.04),
  Some(1.0),
  Some(false),
  s
));

guard!(feller_root, "FellerRoot", |s| FellerRoot::new(
  1.0,
  0.04,
  0.2,
  N,
  Some(0.04),
  Some(1.0),
  s
));

guard!(fgbm, "Fgbm", |s| Fgbm::new(
  0.7,
  0.05,
  0.2,
  N,
  Some(100.0),
  Some(1.0),
  s
));

guard!(fjacobi, "FJacobi", |s| FJacobi::new(
  0.7,
  0.5,
  1.5,
  0.2,
  N,
  Some(0.5),
  Some(1.0),
  s
));

guard!(fou, "Fou", |s| Fou::new(
  0.7,
  1.0,
  0.0,
  0.2,
  N,
  Some(0.0),
  Some(1.0),
  s
));

guard!(fouque_ou2d, "FouqueOU2D", |s| FouqueOU2D::new(
  1.0,
  0.0,
  0.1,
  0.3,
  N,
  Some(0.0),
  Some(0.0),
  Some(1.0),
  s
));

guard!(gbm, "Gbm", |s| Gbm::new(
  0.05,
  0.2,
  N,
  Some(100.0),
  Some(1.0),
  s
));

guard!(gbm_ih, "GbmIh", |s| GbmIh::new(
  0.05,
  0.2,
  N,
  Some(100.0),
  Some(1.0),
  None,
  s
));

guard!(gbm_log, "GbmLog", |s| GbmLog::new(
  Some(0.05),
  None,
  None,
  None,
  0.2,
  N,
  Some(100.0),
  Some(1.0),
  s
));

guard!(gompertz, "Gompertz", |s| Gompertz::new(
  0.1,
  1.0,
  0.2,
  N,
  Some(1.0),
  Some(1.0),
  s
));

guard!(hyperbolic, "Hyperbolic", |s| Hyperbolic::new(
  1.0,
  0.2,
  N,
  Some(0.0),
  Some(1.0),
  s
));

guard!(hyperbolic2, "Hyperbolic2", |s| Hyperbolic2::new(
  0.1,
  1.0,
  1.0,
  0.0,
  0.2,
  N,
  Some(0.0),
  Some(1.0),
  s
));

guard!(jacobi, "Jacobi", |s| Jacobi::new(
  0.5,
  1.5,
  0.2,
  N,
  Some(0.5),
  Some(1.0),
  s
));

// `x0 = 0.0` sits exactly on the Wright–Fisher diffusion's absorbing
// boundary: `sigma * sqrt(x0 * (1 - x0))` vanishes there, so the path stays
// at 0 for every remaining step regardless of any Gaussian draw (correct
// model behavior, not a sampler bug) and no seed could ever change the
// output. `0.5` sits in the interior where the diffusion term is live.
guard!(kimura, "Kimura", |s| Kimura::new(
  0.1,
  0.2,
  N,
  Some(0.5),
  Some(1.0),
  s
));

guard!(linear_sde, "LinearSDE", |s| LinearSDE::new(
  0.1,
  0.0,
  0.2,
  N,
  Some(0.0),
  Some(1.0),
  s
));

guard!(logistic, "Logistic", |s| Logistic::new(
  1.0,
  0.2,
  N,
  Some(0.5),
  Some(1.0),
  s
));

guard!(modified_cir, "ModifiedCIR", |s| ModifiedCIR::new(
  1.0,
  0.2,
  N,
  Some(0.04),
  Some(1.0),
  s
));

guard!(nonlinear_sde, "NonLinearSDE", |s| NonLinearSDE::new(
  0.01,
  0.02,
  -0.1,
  0.0,
  0.1,
  0.0,
  0.0,
  0.0,
  N,
  Some(0.05),
  Some(1.0),
  s
));

guard!(ou, "Ou", |s| Ou::new(
  0.5,
  0.02,
  0.1,
  N,
  Some(0.03),
  Some(1.0),
  s
));

guard!(pearson, "Pearson", |s| Pearson::new(
  1.0,
  0.0,
  1.0,
  0.5,
  0.1,
  N,
  Some(0.0),
  Some(1.0),
  s
));

guard!(quadratic, "Quadratic", |s| Quadratic::new(
  0.1,
  0.0,
  0.0,
  0.2,
  N,
  Some(0.0),
  Some(1.0),
  s
));

guard!(radial_ou, "RadialOU", |s| RadialOU::new(
  1.0,
  0.2,
  N,
  Some(0.5),
  Some(1.0),
  s
));

guard!(regime_switching, "RegimeSwitchingDiffusion", |s| {
  RegimeSwitchingDiffusion::new(
    0.05,
    Array2::from_shape_vec((2, 2), vec![-0.5, 0.5, 0.5, -0.5]).unwrap(),
    Array1::from(vec![0.2, 0.3]),
    0,
    N,
    Some(100.0),
    Some(1.0),
    s,
  )
});

guard!(three_half, "ThreeHalf", |s| ThreeHalf::new(
  1.0,
  0.04,
  0.2,
  N,
  Some(0.04),
  Some(1.0),
  s
));

guard!(verhulst, "Verhulst", |s| Verhulst::new(
  0.5,
  1.0,
  0.1,
  N,
  Some(0.5),
  Some(1.0),
  Some(false),
  s
));
guard!(multi_gbm, "MultiGbm", |s| MultiGbm::new(
  ndarray::array![0.05, 0.02],
  ndarray::array![0.2, 0.3],
  ndarray::array![[1.0, 0.3], [0.3, 1.0]],
  N,
  ndarray::array![100.0, 50.0],
  Some(1.0),
  s
));
guard!(wishart, "Wishart", |s| Wishart::new(
  2.5,
  ndarray::array![[-0.5, 0.1], [0.05, -0.3]],
  ndarray::array![[0.3, 0.1], [0.0, 0.2]],
  ndarray::array![[1.0, 0.2], [0.2, 0.5]],
  N,
  Some(1.0),
  s
));

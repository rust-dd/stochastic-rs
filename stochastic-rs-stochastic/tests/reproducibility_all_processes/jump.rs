//! The `jump/` slice of the reproducibility guard. The distribution-taking types run at
//! `LAMBDA = 50`; `JumpFouCustom` draws its timing and sizes from two independent `SimdExp` laws.

use stochastic_rs_distributions::exp::SimdExp;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stochastic::jump::bates::Bates;
use stochastic_rs_stochastic::jump::bilateral_gamma::BilateralGamma;
use stochastic_rs_stochastic::jump::bilateral_gamma::BilateralGammaMotion;
use stochastic_rs_stochastic::jump::cgmy::Cgmy;
use stochastic_rs_stochastic::jump::cts::Cts;
use stochastic_rs_stochastic::jump::hawkes_jd::HawkesJd;
use stochastic_rs_stochastic::jump::jump_fou::JumpFou;
use stochastic_rs_stochastic::jump::jump_fou_custom::JumpFouCustom;
use stochastic_rs_stochastic::jump::kobol::Kobol;
use stochastic_rs_stochastic::jump::kou::Kou;
use stochastic_rs_stochastic::jump::levy_diffusion::LevyDiffusion;
use stochastic_rs_stochastic::jump::merton::Merton;
use stochastic_rs_stochastic::jump::mjd_log::MjdLog;
use stochastic_rs_stochastic::jump::nig::Nig;
use stochastic_rs_stochastic::jump::rdts::Rdts;
use stochastic_rs_stochastic::jump::vg::Vg;
use stochastic_rs_stochastic::params::Drift;
use stochastic_rs_stochastic::params::VarianceBoundary;

use crate::common::J;
use crate::common::LAMBDA;
use crate::common::N;
use crate::common::guard;

guard!(bates, "Bates", |s| Bates::new(
  Drift::Mu(0.05),
  LAMBDA,
  0.0,
  0.04,
  1.5,
  0.3,
  -0.6,
  SimdNormal::new(0.0, 0.05),
  N,
  100.0,
  0.04,
  Some(1.0),
  VarianceBoundary::FullTruncation,
  s
));

guard!(bilateral_gamma, "BilateralGamma", |s| {
  BilateralGamma::new(1.0, 10.0, 1.0, 10.0, N, Some(0.0), Some(1.0), s)
});

guard!(bilateral_gamma_motion, "BilateralGammaMotion", |s| {
  BilateralGammaMotion::new(0.1, 1.0, 10.0, 1.0, 10.0, N, Some(0.0), Some(1.0), s)
});

guard!(cgmy, "Cgmy", |s| Cgmy::new(
  1.0,
  3.0,
  3.0,
  0.5,
  N,
  J,
  Some(0.0),
  Some(1.0),
  s
));

guard!(cts, "Cts", |s| Cts::new(
  3.0,
  3.0,
  0.5,
  N,
  J,
  Some(0.0),
  Some(1.0),
  s
));

guard!(hawkes_jd, "HawkesJd", |s| HawkesJd::new(
  0.05,
  0.2,
  0.5,
  0.3,
  1.0,
  0.0,
  0.1,
  N,
  Some(0.0),
  Some(1.0),
  s
));

guard!(jump_fou, "JumpFou", |s| JumpFou::new(
  0.65,
  1.5,
  0.0,
  0.2,
  LAMBDA,
  SimdNormal::new(0.0, 0.05),
  N,
  Some(0.0),
  Some(1.0),
  s
));

guard!(jump_fou_custom, "JumpFouCustom", |s| {
  JumpFouCustom::new(
    0.65,
    1.5,
    0.0,
    0.2,
    N,
    Some(0.0),
    Some(1.0),
    SimdExp::new(20.0),
    SimdExp::new(5.0),
    s,
  )
});

guard!(kobol, "Kobol", |s| Kobol::new(
  1.0,
  1.0,
  1.0,
  3.0,
  3.0,
  0.5,
  N,
  J,
  Some(0.0),
  Some(1.0),
  s
));

guard!(kou, "Kou", |s| Kou::new(
  0.03,
  0.2,
  LAMBDA,
  0.0,
  SimdNormal::new(0.0, 0.12),
  N,
  Some(0.0),
  Some(1.0),
  s
));

guard!(levy_diffusion, "LevyDiffusion", |s| LevyDiffusion::new(
  0.01,
  0.2,
  LAMBDA,
  SimdNormal::new(0.0, 0.08),
  N,
  Some(0.0),
  Some(1.0),
  s
));

guard!(merton, "Merton", |s| Merton::new(
  0.03,
  0.2,
  LAMBDA,
  0.0,
  SimdNormal::new(0.0, 0.1),
  N,
  Some(0.0),
  Some(1.0),
  s
));

guard!(mjd_log, "MjdLog", |s| MjdLog::new(
  Drift::Mu(0.05),
  0.2,
  1.0,
  0.0,
  0.1,
  N,
  100.0,
  Some(1.0),
  s
));

guard!(nig, "Nig", |s| Nig::new(
  0.0,
  0.2,
  1.0,
  N,
  Some(0.0),
  Some(1.0),
  s
));

guard!(rdts, "Rdts", |s| Rdts::new(
  3.0,
  3.0,
  0.5,
  N,
  J,
  Some(0.0),
  Some(1.0),
  s
));

guard!(vg, "Vg", |s| Vg::new(
  0.05,
  0.2,
  0.3,
  N,
  Some(0.0),
  Some(1.0),
  s
));

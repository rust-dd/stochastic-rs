//! `jump/` slice of the exhaustive reproducibility guard — see
//! `../reproducibility_all_processes.rs` for the full rationale, the
//! derivation of the 129-type list, and shared methodology notes.
//! `Bates1996`, `JumpFou`, `Kou`, `LevyDiffusion` and `Merton` are five of
//! the eight distribution-taking types named there, driven at `LAMBDA = 50`
//! so a jump-component reproducibility bug cannot hide behind a
//! diffusion-only comparison; `JumpFOUCustom` is the sixth (its jump
//! timing/size draws use two independent `ScalarExp` distributions instead
//! of one `ScalarNormal`) and one of the ten backend-generic types besides.

use stochastic_rs_distributions::scalar::ScalarExp;
use stochastic_rs_distributions::scalar::ScalarNormal;
use stochastic_rs_stochastic::jump::bates::Bates1996;
use stochastic_rs_stochastic::jump::bilateral_gamma::BilateralGamma;
use stochastic_rs_stochastic::jump::bilateral_gamma::BilateralGammaMotion;
use stochastic_rs_stochastic::jump::cgmy::Cgmy;
use stochastic_rs_stochastic::jump::cts::Cts;
use stochastic_rs_stochastic::jump::hawkes_jd::HawkesJD;
use stochastic_rs_stochastic::jump::ig::Ig;
use stochastic_rs_stochastic::jump::jump_fou::JumpFou;
use stochastic_rs_stochastic::jump::jump_fou_custom::JumpFOUCustom;
use stochastic_rs_stochastic::jump::kobol::KoBoL;
use stochastic_rs_stochastic::jump::kou::Kou;
use stochastic_rs_stochastic::jump::levy_diffusion::LevyDiffusion;
use stochastic_rs_stochastic::jump::merton::Merton;
use stochastic_rs_stochastic::jump::mjd_log::MjdLog;
use stochastic_rs_stochastic::jump::nig::Nig;
use stochastic_rs_stochastic::jump::rdts::Rdts;
use stochastic_rs_stochastic::jump::vg::Vg;

use crate::common::J;
use crate::common::LAMBDA;
use crate::common::N;
use crate::common::guard;

guard!(bates1996, "Bates1996", |s| Bates1996::new(
  Some(0.05),
  None,
  None,
  None,
  LAMBDA,
  0.0,
  0.04,
  1.5,
  0.3,
  -0.6,
  ScalarNormal::new(0.0, 0.05),
  N,
  Some(100.0),
  Some(0.04),
  Some(1.0),
  Some(false),
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

guard!(hawkes_jd, "HawkesJD", |s| HawkesJD::new(
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

guard!(ig, "Ig", |s| Ig::new(1.0, N, Some(0.0), Some(1.0), s));

guard!(jump_fou, "JumpFou", |s| JumpFou::new(
  0.65,
  1.5,
  0.0,
  0.2,
  LAMBDA,
  ScalarNormal::new(0.0, 0.05),
  N,
  Some(0.0),
  Some(1.0),
  s
));

guard!(jump_fou_custom, "JumpFOUCustom", |s| {
  JumpFOUCustom::new(
    0.65,
    1.5,
    0.0,
    0.2,
    N,
    Some(0.0),
    Some(1.0),
    ScalarExp::new(20.0),
    ScalarExp::new(5.0),
    s,
  )
});

guard!(kobol, "KoBoL", |s| KoBoL::new(
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
  ScalarNormal::new(0.0, 0.12),
  N,
  Some(0.0),
  Some(1.0),
  s
));

guard!(levy_diffusion, "LevyDiffusion", |s| LevyDiffusion::new(
  0.01,
  0.2,
  LAMBDA,
  ScalarNormal::new(0.0, 0.08),
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
  ScalarNormal::new(0.0, 0.1),
  N,
  Some(0.0),
  Some(1.0),
  s
));

guard!(mjd_log, "MjdLog", |s| MjdLog::new(
  Some(0.05),
  None,
  None,
  None,
  0.2,
  1.0,
  0.0,
  0.1,
  N,
  Some(100.0),
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

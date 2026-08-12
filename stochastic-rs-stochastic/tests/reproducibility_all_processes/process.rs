//! `process/` slice of the exhaustive reproducibility guard — see
//! `../reproducibility_all_processes.rs` for the full rationale, the
//! derivation of the 124-type list, and shared methodology notes.
//! `CompoundPoisson` and `CustomJt` are two of the eight named
//! distribution-taking types; `CompoundCustom` is the ninth,
//! unnamed-by-the-usual-count case mentioned there — it takes two
//! distributions (`D1`, `D2`) plus a nested `CustomJt<T, D2>`, whose own
//! `Unseeded` seed field is inert (its sampler drives it through an
//! *externally* supplied seed instead, exactly like `CompoundPoisson`'s
//! nested `Poisson` below). `Fbm` is one of the ten backend-generic types.

use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::scalar::ScalarExp;
use stochastic_rs_distributions::scalar::ScalarNormal;
use stochastic_rs_stochastic::process::bm::Bm;
use stochastic_rs_stochastic::process::brownian_bridge::BrownianBridge;
use stochastic_rs_stochastic::process::cbms::Cbms;
use stochastic_rs_stochastic::process::ccustom::CompoundCustom;
use stochastic_rs_stochastic::process::cfbms::Cfbms;
use stochastic_rs_stochastic::process::cpoisson::CompoundPoisson;
use stochastic_rs_stochastic::process::customjt::CustomJt;
use stochastic_rs_stochastic::process::fbm::Fbm;
use stochastic_rs_stochastic::process::hawkes::Hawkes;
use stochastic_rs_stochastic::process::lfsm::Lfsm;
use stochastic_rs_stochastic::process::multivariate_hawkes::MultivariateHawkes;
use stochastic_rs_stochastic::process::poisson::Poisson;
use stochastic_rs_stochastic::process::subordinator::alpha_stable::AlphaStableSubordinator;
use stochastic_rs_stochastic::process::subordinator::ctrw::Ctrw;
use stochastic_rs_stochastic::process::subordinator::ctrw::CtrwJumpLaw;
use stochastic_rs_stochastic::process::subordinator::ctrw::CtrwWaitingLaw;
use stochastic_rs_stochastic::process::subordinator::gamma_subordinator::GammaSubordinator;
use stochastic_rs_stochastic::process::subordinator::ig_subordinator::IGSubordinator;
use stochastic_rs_stochastic::process::subordinator::inverse_alpha_stable::InverseAlphaStableSubordinator;
use stochastic_rs_stochastic::process::subordinator::poisson_subordinator::PoissonSubordinator;
use stochastic_rs_stochastic::process::subordinator::tempered_stable::TemperedStableSubordinator;
use stochastic_rs_stochastic::process::volterra::Volterra;
use stochastic_rs_stochastic::process::volterra::VolterraKernel;

use crate::common::LAMBDA;
use crate::common::N;
use crate::common::N_VOLTERRA;
use crate::common::guard;

guard!(bm, "Bm", |s| Bm::new(N, Some(1.0), s));

guard!(brownian_bridge, "BrownianBridge", |s| BrownianBridge::new(
  1.0,
  N,
  None,
  None,
  Some(1.0),
  s
));

guard!(cbms, "Cbms", |s| Cbms::new(0.3, N, Some(1.0), s));

guard!(compound_custom, "CompoundCustom", |s| {
  CompoundCustom::new(
    Some(N),
    None,
    ScalarNormal::new(0.0, 0.1),
    ScalarExp::new(2.0),
    CustomJt::new(Some(N), None, ScalarExp::new(2.0), Unseeded),
    s,
  )
});

guard!(cfbms, "Cfbms", |s| Cfbms::new(0.7, 0.3, N, Some(1.0), s));

guard!(compound_poisson, "CompoundPoisson", |s| {
  CompoundPoisson::new(
    ScalarNormal::new(0.0, 0.1),
    Poisson::new(LAMBDA, Some(N), Some(1.0), Unseeded),
    s,
  )
});

guard!(custom_jt, "CustomJt", |s| CustomJt::new(
  Some(N),
  None,
  ScalarExp::new(2.0),
  s
));

guard!(fbm, "Fbm", |s| Fbm::new(0.7, N, Some(1.0), s));

guard!(hawkes, "Hawkes", |s| Hawkes::new(
  0.5,
  0.3,
  1.0,
  Some(N),
  None,
  s
));

guard!(lfsm, "Lfsm", |s| Lfsm::new(
  1.5,
  0.0,
  0.75,
  1.0,
  N,
  Some(0.0),
  Some(1.0),
  s
));

guard!(multivariate_hawkes, "MultivariateHawkes", |s| {
  MultivariateHawkes::new(
    Array1::from(vec![0.1, 0.1]),
    Array2::from_shape_vec((2, 2), vec![0.05, 0.02, 0.02, 0.05]).unwrap(),
    Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0]).unwrap(),
    5.0,
    s,
  )
});

guard!(poisson, "Poisson", |s| Poisson::new(
  2.0,
  Some(N),
  Some(1.0),
  s
));

guard!(alpha_stable_subordinator, "AlphaStableSubordinator", |s| {
  AlphaStableSubordinator::new(0.5, 1.0, N, Some(0.0), Some(1.0), s)
});

guard!(ctrw, "Ctrw", |s| Ctrw::new(
  CtrwWaitingLaw::Exponential { rate: 2.0 },
  CtrwJumpLaw::Normal {
    mean: 0.0,
    std: 1.0
  },
  N,
  Some(0.0),
  Some(1.0),
  s
));

guard!(gamma_subordinator, "GammaSubordinator", |s| {
  GammaSubordinator::new(1.0, 1.0, N, Some(0.0), Some(1.0), s)
});

guard!(ig_subordinator, "IGSubordinator", |s| IGSubordinator::new(
  1.0,
  1.0,
  N,
  Some(0.0),
  Some(1.0),
  s
));

guard!(
  inverse_alpha_stable_subordinator,
  "InverseAlphaStableSubordinator",
  |s| InverseAlphaStableSubordinator::new(0.5, 1.0, N, Some(1.0), 8, None, s)
);

guard!(poisson_subordinator, "PoissonSubordinator", |s| {
  PoissonSubordinator::new(2.0, N, Some(0.0), Some(1.0), s)
});

// `c = 1.0, mu = 1.0` gives a large-jump arrival rate (`lambda0 * dt`) of
// ~0.12/step with an acceptance probability at the minimum jump size
// (`exp(-mu * epsilon)`) of only ~0.61, so over `N` steps this process's
// accepted-jump count is 0 for a non-negligible fraction of seeds —
// measured: seeds 42 and 43 both realized zero accepted jumps, making
// `sample()` exactly the deterministic small-jump drift term for both,
// bit-identical regardless of seed. `c = 5.0, mu = 0.3` raises the
// candidate rate roughly fivefold and the minimum-jump acceptance to
// ~0.86, making an empty path negligibly unlikely.
guard!(
  tempered_stable_subordinator,
  "TemperedStableSubordinator",
  |s| TemperedStableSubordinator::new(0.5, 5.0, 0.3, 0.5, N, Some(0.0), Some(1.0), s)
);

guard!(volterra, "Volterra", |s| Volterra::new(
  VolterraKernel::FractionalBM { h: 0.7 },
  N_VOLTERRA,
  Some(1.0),
  s
));

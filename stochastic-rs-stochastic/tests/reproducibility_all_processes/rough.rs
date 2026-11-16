//! `rough/` slice of the exhaustive reproducibility guard — see
//! `../reproducibility_all_processes.rs` for the full rationale, the
//! derivation of the 129-type list, and shared methodology notes.

use stochastic_rs_stochastic::rough::rl_bs::RlBlackScholes;
use stochastic_rs_stochastic::rough::rl_fbm::RlFbm;
use stochastic_rs_stochastic::rough::rl_fou::RlFou;
use stochastic_rs_stochastic::rough::rl_heston::RlHeston;

use crate::common::N;
use crate::common::guard;

guard!(rl_black_scholes, "RlBlackScholes", |s| {
  RlBlackScholes::new(0.1, 100.0, 0.02, 0.2, N, Some(1.0), None, s)
});

guard!(rl_fbm, "RlFbm", |s| RlFbm::new(0.1, N, Some(1.0), None, s));

guard!(rl_fou, "RlFou", |s| RlFou::new(
  0.1,
  1.0,
  0.0,
  0.3,
  N,
  Some(0.0),
  Some(1.0),
  None,
  s
));

guard!(rl_heston, "RlHeston", |s| RlHeston::new(
  0.1,
  100.0,
  0.04,
  1.5,
  0.04,
  0.3,
  -0.6,
  0.0,
  N,
  Some(1.0),
  None,
  s
));

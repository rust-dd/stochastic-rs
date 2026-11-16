//! `correlation/` slice of the exhaustive reproducibility guard — see
//! `../reproducibility_all_processes.rs` for the full rationale, the
//! derivation of the 130-type list, and shared methodology notes.

use stochastic_rs_stochastic::correlation::heston_stoch_corr::HestonStochCorr;
use stochastic_rs_stochastic::correlation::teng::TengSCP;
use stochastic_rs_stochastic::correlation::transformed_ou::Transformation;
use stochastic_rs_stochastic::correlation::transformed_ou::TransformedOU;
use stochastic_rs_stochastic::correlation::van_emmerich::VanEmmerich;

use crate::common::N;
use crate::common::guard;

guard!(heston_stoch_corr, "HestonStochCorr", |s| {
  HestonStochCorr::new(
    0.02,
    100.0,
    0.04,
    1.5,
    0.04,
    0.3,
    -0.5,
    1.0,
    0.0,
    0.2,
    0.3,
    N,
    Some(1.0),
    s,
  )
});

guard!(teng_scp, "TengSCP", |s| TengSCP::new(
  1.0,
  0.0,
  0.3,
  0.2,
  N,
  Some(1.0),
  s
));

guard!(transformed_ou, "TransformedOU", |s| TransformedOU::new(
  1.0,
  0.0,
  0.3,
  0.2,
  Transformation::Tanh,
  N,
  Some(1.0),
  s
));

guard!(van_emmerich, "VanEmmerich", |s| VanEmmerich::new(
  1.0,
  0.0,
  0.3,
  0.2,
  N,
  Some(1.0),
  s
));

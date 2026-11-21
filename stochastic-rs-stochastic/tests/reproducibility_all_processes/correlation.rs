//! The `correlation/` slice of the reproducibility guard, `../reproducibility_all_processes.rs`.

use stochastic_rs_stochastic::correlation::heston_stoch_corr::HestonStochCorr;
use stochastic_rs_stochastic::correlation::teng::TengScp;
use stochastic_rs_stochastic::correlation::transformed_ou::Transformation;
use stochastic_rs_stochastic::correlation::transformed_ou::TransformedOu;
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

guard!(teng_scp, "TengScp", |s| TengScp::new(
  1.0,
  0.0,
  0.3,
  0.2,
  N,
  Some(1.0),
  s
));

guard!(transformed_ou, "TransformedOu", |s| TransformedOu::new(
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

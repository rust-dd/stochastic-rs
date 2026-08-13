//! `volterra/` slice of the exhaustive reproducibility guard — see
//! `../reproducibility_all_processes.rs` for the full rationale, the
//! derivation of the 126-type list, and shared methodology notes.

use stochastic_rs_stochastic::rough::kernel::RlKernel;
use stochastic_rs_stochastic::volterra::ExponentialKernel;
use stochastic_rs_stochastic::volterra::VolterraSde;
use stochastic_rs_stochastic::volterra::VolterraSquareRoot;

use crate::common::N;
use crate::common::guard;

fn mean_reverting_drift(_t: f64, x: f64) -> f64 {
  0.3 * (0.5 - x)
}

fn const_diffusion(_t: f64, _x: f64) -> f64 {
  0.2
}

guard!(volterra_sde, "VolterraSde", |s| VolterraSde::new(
  ExponentialKernel::new(0.7, 1.0),
  mean_reverting_drift as fn(f64, f64) -> f64,
  const_diffusion as fn(f64, f64) -> f64,
  N,
  Some(0.1),
  Some(1.0),
  s
));

// `VolterraSquareRoot` is guarded at parameters that VIOLATE the Feller
// condition (2*kappa*theta = 0.02 vs nu^2 = 0.25), so the boundary-truncation
// branch is actually exercised here rather than only in the type's own tests.
guard!(volterra_square_root, "VolterraSquareRoot", |s| {
  VolterraSquareRoot::new(
    RlKernel::<f64>::new(0.1, 24),
    0.5,
    0.02,
    0.5,
    N,
    Some(0.02),
    Some(1.0),
    s,
  )
});

//! What a fit says where its estimate is a limit of its family, on normal samples.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stats::garch::GarchSpec;
use stochastic_rs_stats::garch::fit_garch;

fn normals(seed: u64) -> Array1<f64> {
  let mut z = vec![0.0; 500];
  SimdNormal::<f64>::new(0.0, 1.0)
    .seeded(&Deterministic::new(seed))
    .fill_slice(&mut z);
  Array1::from(z)
}

fn garch(seed: u64) -> (bool, String) {
  let fit = fit_garch(normals(seed).view(), GarchSpec::garch(1, 1)).unwrap();
  (fit.converged, fit.message)
}

/// White noise whose `α̂` is `4.9e-15`: the constant-variance law fits as well.
#[test]
fn a_garch_fit_with_no_arch_effect_stops_at_the_bound_of_alpha() {
  assert_eq!(garch(2), (false, "stopped at the bound of alpha[1]".into()));
}

/// White noise whose `β̂` is `2.5e-13`: the ARCH(1) law fits as well.
#[test]
fn a_garch_fit_with_no_memory_stops_at_the_bound_of_beta() {
  assert_eq!(garch(3), (false, "stopped at the bound of beta[1]".into()));
}

#[test]
fn a_garch_fit_inside_its_region_converges() {
  assert_eq!(garch(1), (true, "converged".into()));
}

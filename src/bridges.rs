//! # Estimator -> process bridges
//!
//! Small, deliberately thin conversions from a `stochastic-rs-stats`
//! estimator result to a `stochastic-rs-stochastic` process constructor, so
//! "estimate a parameter, then simulate with it" is one documented call
//! instead of the user re-deriving which field feeds which constructor
//! argument. Conceptually mirrors
//! [`stochastic_rs_quant::calibration::rbergomi::RBergomiParams::seed_from_fou`],
//! which performs the analogous fOU-estimate -> rBergomi-params
//! conversion — but it cannot live in the same *shape* (an inherent method
//! on the target type): [`Fgn`] and [`Fbm`] are defined in
//! `stochastic-rs-stochastic`, which `stochastic-rs-stats` depends on (not
//! the reverse), so Rust's orphan rule forbids an inherent `impl` on them
//! from a crate that sees [`HurstResult`], and adding a dependency the
//! other way would make the workspace's dependency graph circular. The
//! umbrella crate depends on both, so the bridge lives here instead, as
//! free functions.
//!
//! # Only valid for path-based Hurst estimators
//!
//! [`HurstResult`] is the shared return type of every
//! [`HurstEstimator`](stochastic_rs_stats::hurst::HurstEstimator) impl in
//! `stochastic-rs-stats`, but the impls do not all estimate the same
//! quantity:
//!
//! - [`RescaledRange`](stochastic_rs_stats::hurst::RescaledRange),
//!   [`Dfa`](stochastic_rs_stats::hurst::Dfa),
//!   [`Gph`](stochastic_rs_stats::hurst::Gph),
//!   [`Wavelet`](stochastic_rs_stats::hurst::Wavelet),
//!   [`Variations`](stochastic_rs_stats::hurst::Variations), and the
//!   fractal-dimension adapters
//!   ([`stochastic_rs_stats::fractal_dim::Higuchi`],
//!   [`stochastic_rs_stats::fractal_dim::Variogram`]) estimate the
//!   self-similarity exponent of the series handed to `.estimate()`
//!   directly. Feed *these* results here.
//! - [`Whittle`](stochastic_rs_stats::hurst::Whittle) (Fukasawa) estimates
//!   the roughness of a **latent volatility** process from a
//!   realized-variance series (see its module doc) — a different quantity
//!   from the self-similarity exponent of a path sampled from [`Fgn`] /
//!   [`Fbm`]. Its `.hurst` field compiles here (the return type is the
//!   same `HurstResult<T>`) but does not describe the data that produced
//!   the path these functions build; feeding it in silently produces an
//!   unrelated process. This is not a hypothetical: `tests/doctest_stats_hurst.rs`
//!   already documents the same trap for the estimate-only half of this
//!   workflow.
//!
//! Excluding `Whittle` at the type level would mean splitting
//! `HurstEstimator` by contract, which is out of scope for this bridge —
//! so the boundary is documented, not enforced.

use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_distributions::traits::FloatExt;
use stochastic_rs_stats::hurst::HurstResult;
use stochastic_rs_stochastic::device::Cpu;
use stochastic_rs_stochastic::noise::fgn::Fgn;
use stochastic_rs_stochastic::process::fbm::Fbm;

/// Build an [`Fgn`] whose Hurst exponent is a path-based Hurst-estimator
/// fit. See the module doc for which estimators are valid inputs.
pub fn fgn_from_hurst_result<T: FloatExt, S: SeedExt>(
  result: &HurstResult<T>,
  n: usize,
  t: Option<T>,
  seed: S,
) -> Fgn<T, S, Cpu> {
  Fgn::new(result.hurst, n, t, seed)
}

/// Build an [`Fbm`] whose Hurst exponent is a path-based Hurst-estimator
/// fit. See the module doc for which estimators are valid inputs.
pub fn fbm_from_hurst_result<T: FloatExt, S: SeedExt>(
  result: &HurstResult<T>,
  n: usize,
  t: Option<T>,
  seed: S,
) -> Fbm<T, S, Cpu> {
  Fbm::new(result.hurst, n, t, seed)
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_stats::hurst::HurstEstimator;
  use stochastic_rs_stats::hurst::rs::RescaledRange;
  use stochastic_rs_stochastic::traits::ProcessExt;

  use super::*;

  /// Round-trip: sample an `Fgn` at a known `H`, estimate `H` back out with
  /// a path-based estimator, rebuild an `Fgn` from that estimate via
  /// [`fgn_from_hurst_result`], and check the rebuilt process both carries
  /// the recovered `H` and re-estimates close to the original — proving
  /// this is a real, working composition and not a unit mismatch.
  #[test]
  fn fgn_from_hurst_result_round_trips_through_rescaled_range() {
    let true_h = 0.3;
    let path = Fgn::<f64, _, Cpu>::new(true_h, 4096, Some(1.0), Deterministic::new(42)).sample();

    let estimator = RescaledRange {
      take_differences: false,
      ..RescaledRange::default()
    };
    let result = estimator.estimate(path.view()).unwrap();
    assert!(
      (result.hurst - true_h).abs() < 0.1,
      "estimate H = {:.3} (true = {:.3})",
      result.hurst,
      true_h
    );

    let rebuilt = fgn_from_hurst_result(&result, 4096, Some(1.0), Deterministic::new(7));
    assert_eq!(rebuilt.hurst, result.hurst);

    let resampled = rebuilt.sample();
    let reestimated = estimator.estimate(resampled.view()).unwrap();
    assert!(
      (reestimated.hurst - true_h).abs() < 0.15,
      "re-estimated H = {:.3} (true = {:.3})",
      reestimated.hurst,
      true_h
    );
  }

  /// Lighter check for [`fbm_from_hurst_result`]: `Fbm::new` just wraps
  /// `Fgn::new` internally (see `stochastic-rs-stochastic/src/process/fbm.rs`),
  /// so the statistical round-trip is already covered above — this
  /// confirms the field plumbing (`n`, `t`, `hurst`) is wired correctly and
  /// the constructed process samples without panicking.
  #[test]
  fn fbm_from_hurst_result_wires_fields_and_samples() {
    let estimator = RescaledRange {
      take_differences: false,
      ..RescaledRange::default()
    };
    let path = Fgn::<f64, _, Cpu>::new(0.6, 2048, Some(1.0), Deterministic::new(3)).sample();
    let result = estimator.estimate(path.view()).unwrap();

    let fbm = fbm_from_hurst_result(&result, 512, Some(2.0), Deterministic::new(9));
    assert_eq!(fbm.hurst, result.hurst);
    assert_eq!(fbm.n, 512);
    assert_eq!(fbm.t, Some(2.0));

    let sample = fbm.sample();
    assert_eq!(sample.len(), 512);
    assert!(sample.iter().all(|v| v.is_finite()));
  }
}

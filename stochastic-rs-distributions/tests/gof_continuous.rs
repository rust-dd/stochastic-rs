//! Kolmogorov-Smirnov goodness-of-fit: continuous samplers against their
//! own `cdf`. See `tests/gof_support/mod.rs` for the full design
//! rationale, citations, alpha, and the complete per-type coverage
//! table. `SimdNormal`, `SimdExp`, `SimdExpZig` and `SimdGamma` are
//! covered by their own (refactored) in-crate unit tests instead of
//! here, to avoid duplicate coverage — see that table for exactly where
//! each type's test lives.
//!
//! This file covers the eleven remaining "plain" continuous types (i.e.
//! excluding the four `Truncated*` wrappers, which get their own file:
//! `tests/gof_truncated.rs`): `SimdUniform`, `SimdLogNormal`,
//! `SimdBeta`, `SimdCauchy`, `SimdChiSquared`, `SimdStudentT`,
//! `SimdPareto`, `SimdWeibull`, `SimdInverseGauss`, `SimdGed`, and
//! `SimdGev` (all three of its Gumbel / Frechet / reverse-Weibull
//! shapes).

mod gof_support;

use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::DistributionExt;
use stochastic_rs_distributions::beta::SimdBeta;
use stochastic_rs_distributions::cauchy::SimdCauchy;
use stochastic_rs_distributions::chi_square::SimdChiSquared;
use stochastic_rs_distributions::ged::SimdGed;
use stochastic_rs_distributions::gev::SimdGev;
use stochastic_rs_distributions::gpd::SimdGpd;
use stochastic_rs_distributions::inverse_gauss::SimdInverseGauss;
use stochastic_rs_distributions::johnson_su::SimdJohnsonSu;
use stochastic_rs_distributions::lognormal::SimdLogNormal;
use stochastic_rs_distributions::pareto::SimdPareto;
use stochastic_rs_distributions::skew_t::SimdSkewT;
use stochastic_rs_distributions::studentt::SimdStudentT;
use stochastic_rs_distributions::uniform::SimdUniform;
use stochastic_rs_distributions::weibull::SimdWeibull;

const N: usize = 20_000;

#[test]
fn simd_uniform_matches_own_cdf() {
  gof_support::assert_ks_accepts(N, |seed| {
    let dist = SimdUniform::<f64>::new(-2.0, 3.0, &Deterministic::new(seed));
    let mut xs = vec![0.0; N];
    dist.fill_slice(&mut xs);
    (
      xs,
      Box::new(move |x| dist.cdf(x)) as Box<dyn Fn(f64) -> f64>,
    )
  });
}

#[test]
fn simd_lognormal_matches_own_cdf() {
  gof_support::assert_ks_accepts(N, |seed| {
    let dist = SimdLogNormal::<f64>::new(0.2, 0.6, &Deterministic::new(seed));
    let mut xs = vec![0.0; N];
    dist.fill_slice(&mut xs);
    (
      xs,
      Box::new(move |x| dist.cdf(x)) as Box<dyn Fn(f64) -> f64>,
    )
  });
}

#[test]
fn simd_beta_matches_own_cdf() {
  gof_support::assert_ks_accepts(N, |seed| {
    let dist = SimdBeta::<f64>::new(2.5, 4.0, &Deterministic::new(seed));
    let mut xs = vec![0.0; N];
    dist.fill_slice(&mut xs);
    (
      xs,
      Box::new(move |x| dist.cdf(x)) as Box<dyn Fn(f64) -> f64>,
    )
  });
}

#[test]
fn simd_cauchy_matches_own_cdf() {
  gof_support::assert_ks_accepts(N, |seed| {
    let dist = SimdCauchy::<f64>::new(1.0, 0.5, &Deterministic::new(seed));
    let mut xs = vec![0.0; N];
    dist.fill_slice(&mut xs);
    (
      xs,
      Box::new(move |x| dist.cdf(x)) as Box<dyn Fn(f64) -> f64>,
    )
  });
}

#[test]
fn simd_chi_squared_matches_own_cdf() {
  gof_support::assert_ks_accepts(N, |seed| {
    let dist = SimdChiSquared::<f64>::new(6.0, &Deterministic::new(seed));
    let mut xs = vec![0.0; N];
    dist.fill_slice(&mut xs);
    (
      xs,
      Box::new(move |x| dist.cdf(x)) as Box<dyn Fn(f64) -> f64>,
    )
  });
}

#[test]
fn simd_studentt_matches_own_cdf() {
  gof_support::assert_ks_accepts(N, |seed| {
    let dist = SimdStudentT::<f64>::new(6.0, &Deterministic::new(seed));
    let mut xs = vec![0.0; N];
    dist.fill_slice(&mut xs);
    (
      xs,
      Box::new(move |x| dist.cdf(x)) as Box<dyn Fn(f64) -> f64>,
    )
  });
}

/// `alpha = 1.16` is the classic "80/20" Pareto (finite mean, infinite
/// variance) — see `src/pareto.rs`'s own moment-threshold tests.
#[test]
fn simd_pareto_matches_own_cdf() {
  gof_support::assert_ks_accepts(N, |seed| {
    let dist = SimdPareto::<f64>::new(1.0, 1.16, &Deterministic::new(seed));
    let mut xs = vec![0.0; N];
    dist.fill_slice(&mut xs);
    (
      xs,
      Box::new(move |x| dist.cdf(x)) as Box<dyn Fn(f64) -> f64>,
    )
  });
}

#[test]
fn simd_weibull_matches_own_cdf() {
  gof_support::assert_ks_accepts(N, |seed| {
    let dist = SimdWeibull::<f64>::new(2.0, 1.5, &Deterministic::new(seed));
    let mut xs = vec![0.0; N];
    dist.fill_slice(&mut xs);
    (
      xs,
      Box::new(move |x| dist.cdf(x)) as Box<dyn Fn(f64) -> f64>,
    )
  });
}

#[test]
fn simd_inverse_gauss_matches_own_cdf() {
  gof_support::assert_ks_accepts(N, |seed| {
    let dist = SimdInverseGauss::<f64>::new(1.5, 3.0, &Deterministic::new(seed));
    let mut xs = vec![0.0; N];
    dist.fill_slice(&mut xs);
    (
      xs,
      Box::new(move |x| dist.cdf(x)) as Box<dyn Fn(f64) -> f64>,
    )
  });
}

/// `beta = 1.5` — between Laplace (`beta=1`) and Gaussian (`beta=2`).
#[test]
fn simd_ged_matches_own_cdf() {
  gof_support::assert_ks_accepts(N, |seed| {
    let dist = SimdGed::<f64>::new(0.0, 1.0, 1.5, &Deterministic::new(seed));
    let mut xs = vec![0.0; N];
    dist.fill_slice(&mut xs);
    (
      xs,
      Box::new(move |x| dist.cdf(x)) as Box<dyn Fn(f64) -> f64>,
    )
  });
}

#[test]
fn simd_gpd_heavy_tail_matches_own_cdf() {
  gof_support::assert_ks_accepts(N, |seed| {
    let dist = SimdGpd::<f64>::new(0.0, 1.0, 0.3, &Deterministic::new(seed));
    let mut xs = vec![0.0; N];
    dist.fill_slice(&mut xs);
    (
      xs,
      Box::new(move |x| dist.cdf(x)) as Box<dyn Fn(f64) -> f64>,
    )
  });
}

#[test]
fn simd_gpd_exponential_matches_own_cdf() {
  gof_support::assert_ks_accepts(N, |seed| {
    let dist = SimdGpd::<f64>::new(0.0, 1.0, 0.0, &Deterministic::new(seed));
    let mut xs = vec![0.0; N];
    dist.fill_slice(&mut xs);
    (
      xs,
      Box::new(move |x| dist.cdf(x)) as Box<dyn Fn(f64) -> f64>,
    )
  });
}

#[test]
fn simd_gpd_bounded_tail_matches_own_cdf() {
  gof_support::assert_ks_accepts(N, |seed| {
    let dist = SimdGpd::<f64>::new(0.0, 1.0, -0.3, &Deterministic::new(seed));
    let mut xs = vec![0.0; N];
    dist.fill_slice(&mut xs);
    (
      xs,
      Box::new(move |x| dist.cdf(x)) as Box<dyn Fn(f64) -> f64>,
    )
  });
}

#[test]
fn simd_johnson_su_matches_own_cdf() {
  gof_support::assert_ks_accepts(N, |seed| {
    let dist = SimdJohnsonSu::<f64>::new(-0.5, 1.5, 0.2, 2.0, &Deterministic::new(seed));
    let mut xs = vec![0.0; N];
    dist.fill_slice(&mut xs);
    (
      xs,
      Box::new(move |x| dist.cdf(x)) as Box<dyn Fn(f64) -> f64>,
    )
  });
}

#[test]
fn simd_skew_t_matches_own_cdf() {
  gof_support::assert_ks_accepts(N, |seed| {
    let dist = SimdSkewT::<f64>::new(5.0, -0.3, &Deterministic::new(seed));
    let mut xs = vec![0.0; N];
    dist.fill_slice(&mut xs);
    (
      xs,
      Box::new(move |x| dist.cdf(x)) as Box<dyn Fn(f64) -> f64>,
    )
  });
}

#[test]
fn simd_gev_gumbel_matches_own_cdf() {
  gof_support::assert_ks_accepts(N, |seed| {
    let dist = SimdGev::<f64>::new(0.0, 1.0, 0.0, &Deterministic::new(seed));
    let mut xs = vec![0.0; N];
    dist.fill_slice(&mut xs);
    (
      xs,
      Box::new(move |x| dist.cdf(x)) as Box<dyn Fn(f64) -> f64>,
    )
  });
}

#[test]
fn simd_gev_frechet_matches_own_cdf() {
  gof_support::assert_ks_accepts(N, |seed| {
    let dist = SimdGev::<f64>::new(0.0, 1.0, 0.3, &Deterministic::new(seed));
    let mut xs = vec![0.0; N];
    dist.fill_slice(&mut xs);
    (
      xs,
      Box::new(move |x| dist.cdf(x)) as Box<dyn Fn(f64) -> f64>,
    )
  });
}

#[test]
fn simd_gev_reverse_weibull_matches_own_cdf() {
  gof_support::assert_ks_accepts(N, |seed| {
    let dist = SimdGev::<f64>::new(0.0, 1.0, -0.3, &Deterministic::new(seed));
    let mut xs = vec![0.0; N];
    dist.fill_slice(&mut xs);
    (
      xs,
      Box::new(move |x| dist.cdf(x)) as Box<dyn Fn(f64) -> f64>,
    )
  });
}

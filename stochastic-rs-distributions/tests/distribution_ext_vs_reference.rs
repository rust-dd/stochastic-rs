//! Cross-validation: every closed-form `DistributionExt` impl is tested
//! against `statrs` at representative points. `statrs` is a dev-dep only —
//! production code never delegates to it.

use statrs::distribution::Continuous as _;
use statrs::distribution::ContinuousCDF as _;
use statrs::distribution::Discrete as _;
use statrs::distribution::DiscreteCDF as _;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionExt;
use stochastic_rs_distributions::beta::SimdBeta;
use stochastic_rs_distributions::binomial::SimdBinomial;
use stochastic_rs_distributions::cauchy::SimdCauchy;
use stochastic_rs_distributions::chi_square::SimdChiSquared;
use stochastic_rs_distributions::exp::SimdExpZig;
use stochastic_rs_distributions::gamma::SimdGamma;
use stochastic_rs_distributions::hypergeometric::SimdHypergeometric;
use stochastic_rs_distributions::lognormal::SimdLogNormal;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::pareto::SimdPareto;
use stochastic_rs_distributions::poisson::SimdPoisson;
use stochastic_rs_distributions::studentt::SimdStudentT;
use stochastic_rs_distributions::uniform::SimdUniform;
use stochastic_rs_distributions::weibull::SimdWeibull;

fn close(a: f64, b: f64, abs_tol: f64, rel_tol: f64) -> bool {
  if a.is_nan() && b.is_nan() {
    return true;
  }
  if a.is_infinite() && b.is_infinite() && a.signum() == b.signum() {
    return true;
  }
  let abs = (a - b).abs();
  abs <= abs_tol || abs <= rel_tol * b.abs().max(a.abs())
}

#[test]
fn normal_matches_statrs() {
  let ours = SimdNormal::<f64>::new(1.5, 2.5, &Unseeded);
  let theirs = statrs::distribution::Normal::new(1.5, 2.5).unwrap();
  for &x in &[-3.0, -1.0, 0.0, 1.5, 4.0] {
    assert!(close(ours.pdf(x), theirs.pdf(x), 1e-12, 1e-10));
    assert!(close(ours.cdf(x), theirs.cdf(x), 1e-7, 1e-7));
  }
  for &p in &[0.01, 0.1, 0.5, 0.9, 0.99] {
    assert!(close(ours.inv_cdf(p), theirs.inverse_cdf(p), 1e-6, 1e-6));
  }
}

#[test]
fn lognormal_matches_statrs() {
  let ours = SimdLogNormal::<f64>::new(0.0, 0.5, &Unseeded);
  let theirs = statrs::distribution::LogNormal::new(0.0, 0.5).unwrap();
  for &x in &[0.1, 0.5, 1.0, 2.5, 10.0] {
    assert!(close(ours.pdf(x), theirs.pdf(x), 1e-12, 1e-7));
    assert!(close(ours.cdf(x), theirs.cdf(x), 1e-7, 1e-7));
  }
  for &p in &[0.05, 0.25, 0.5, 0.75, 0.95] {
    assert!(close(ours.inv_cdf(p), theirs.inverse_cdf(p), 1e-6, 1e-6));
  }
}

#[test]
fn gamma_matches_statrs() {
  let ours = SimdGamma::<f64>::new(2.5, 1.5, &Unseeded);
  let theirs = statrs::distribution::Gamma::new(2.5, 1.0 / 1.5).unwrap();
  for &x in &[0.1, 0.5, 1.0, 3.0, 10.0] {
    assert!(close(ours.pdf(x), theirs.pdf(x), 1e-12, 1e-9));
    assert!(close(ours.cdf(x), theirs.cdf(x), 1e-9, 1e-9));
  }
  for &p in &[0.05, 0.25, 0.5, 0.75, 0.95] {
    assert!(close(ours.inv_cdf(p), theirs.inverse_cdf(p), 1e-5, 1e-5));
  }
}

#[test]
fn uniform_matches_statrs() {
  let ours = SimdUniform::<f64>::new(-1.0, 3.0, &Unseeded);
  let theirs = statrs::distribution::Uniform::new(-1.0, 3.0).unwrap();
  for &x in &[-2.0, 0.0, 1.5, 4.0] {
    assert!(close(ours.pdf(x), theirs.pdf(x), 1e-12, 1e-12));
    assert!(close(ours.cdf(x), theirs.cdf(x), 1e-12, 1e-12));
  }
  // Closed-form moments — cross-checked analytically.
  assert!(close(ours.mean(), 1.0, 1e-12, 1e-12));
  assert!(close(ours.variance(), 16.0 / 12.0, 1e-12, 1e-12));
}

#[test]
fn beta_matches_statrs() {
  let ours = SimdBeta::<f64>::new(2.5, 4.0, &Unseeded);
  let theirs = statrs::distribution::Beta::new(2.5, 4.0).unwrap();
  for &x in &[0.05, 0.2, 0.5, 0.8, 0.95] {
    assert!(close(ours.pdf(x), theirs.pdf(x), 1e-9, 1e-9));
    assert!(close(ours.cdf(x), theirs.cdf(x), 1e-7, 1e-7));
  }
  for &p in &[0.1, 0.5, 0.9] {
    assert!(close(ours.inv_cdf(p), theirs.inverse_cdf(p), 1e-5, 1e-5));
  }
}

#[test]
fn cauchy_matches_statrs() {
  let ours = SimdCauchy::<f64>::new(1.0, 0.5, &Unseeded);
  let theirs = statrs::distribution::Cauchy::new(1.0, 0.5).unwrap();
  for &x in &[-2.0, 0.0, 1.0, 2.5] {
    assert!(close(ours.pdf(x), theirs.pdf(x), 1e-12, 1e-12));
    assert!(close(ours.cdf(x), theirs.cdf(x), 1e-12, 1e-12));
  }
}

#[test]
fn chi_squared_matches_statrs() {
  let ours = SimdChiSquared::<f64>::new(5.0, &Unseeded);
  let theirs = statrs::distribution::ChiSquared::new(5.0).unwrap();
  for &x in &[0.5, 2.0, 5.0, 10.0, 20.0] {
    assert!(close(ours.pdf(x), theirs.pdf(x), 1e-12, 1e-9));
    assert!(close(ours.cdf(x), theirs.cdf(x), 1e-7, 1e-7));
  }
  for &p in &[0.1, 0.5, 0.9] {
    assert!(close(ours.inv_cdf(p), theirs.inverse_cdf(p), 1e-5, 1e-5));
  }
}

#[test]
fn studentt_matches_statrs() {
  let ours = SimdStudentT::<f64>::new(5.0, &Unseeded);
  let theirs = statrs::distribution::StudentsT::new(0.0, 1.0, 5.0).unwrap();
  for &x in &[-3.0, -0.5, 0.0, 0.5, 3.0] {
    assert!(close(ours.pdf(x), theirs.pdf(x), 1e-12, 1e-9));
    assert!(close(ours.cdf(x), theirs.cdf(x), 1e-7, 1e-7));
  }
  for &p in &[0.05, 0.5, 0.95] {
    assert!(close(ours.inv_cdf(p), theirs.inverse_cdf(p), 1e-4, 1e-4));
  }
}

#[test]
fn exp_matches_statrs() {
  let ours = SimdExpZig::<f64>::new(2.5, &Unseeded);
  let theirs = statrs::distribution::Exp::new(2.5).unwrap();
  for &x in &[0.05, 0.5, 1.0, 3.0] {
    assert!(close(ours.pdf(x), theirs.pdf(x), 1e-12, 1e-12));
    assert!(close(ours.cdf(x), theirs.cdf(x), 1e-12, 1e-12));
  }
}

#[test]
fn pareto_matches_statrs() {
  let ours = SimdPareto::<f64>::new(2.0, 3.0, &Unseeded);
  let theirs = statrs::distribution::Pareto::new(2.0, 3.0).unwrap();
  for &x in &[2.5, 5.0, 10.0] {
    assert!(close(ours.pdf(x), theirs.pdf(x), 1e-12, 1e-12));
    assert!(close(ours.cdf(x), theirs.cdf(x), 1e-12, 1e-12));
  }
}

#[test]
fn weibull_matches_statrs() {
  let ours = SimdWeibull::<f64>::new(2.0, 1.5, &Unseeded);
  let theirs = statrs::distribution::Weibull::new(1.5, 2.0).unwrap();
  for &x in &[0.5, 1.0, 2.0, 5.0] {
    assert!(close(ours.pdf(x), theirs.pdf(x), 1e-12, 1e-12));
    assert!(close(ours.cdf(x), theirs.cdf(x), 1e-12, 1e-12));
  }
}

#[test]
fn binomial_matches_statrs() {
  let ours = SimdBinomial::<u32>::new(10, 0.4, &Unseeded);
  let theirs = statrs::distribution::Binomial::new(0.4, 10).unwrap();
  for k in 0..=10 {
    assert!(close(ours.pdf(k as f64), theirs.pmf(k), 1e-9, 1e-9));
    assert!(close(ours.cdf(k as f64), theirs.cdf(k), 1e-9, 1e-9));
  }
}

#[test]
fn poisson_matches_statrs() {
  let ours = SimdPoisson::<u32>::new(3.5, &Unseeded);
  let theirs = statrs::distribution::Poisson::new(3.5).unwrap();
  for k in 0..15 {
    assert!(close(ours.pdf(k as f64), theirs.pmf(k), 1e-9, 1e-9));
    assert!(close(ours.cdf(k as f64), theirs.cdf(k), 1e-7, 1e-7));
  }
}

#[test]
fn hypergeometric_matches_statrs() {
  let ours = SimdHypergeometric::<u32>::new(20, 7, 12, &Unseeded);
  let theirs = statrs::distribution::Hypergeometric::new(20, 7, 12).unwrap();
  for k in 0..=7 {
    assert!(close(ours.pdf(k as f64), theirs.pmf(k), 1e-9, 1e-9));
    assert!(close(ours.cdf(k as f64), theirs.cdf(k), 1e-9, 1e-9));
  }
}

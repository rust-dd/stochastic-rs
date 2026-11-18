use ndarray::Array1;

use super::*;
use crate::qmle::fit_diffusion_qmle;
use crate::qmle::test_support::simulate_cir;
use crate::qmle::test_support::simulate_ou;

const OU: DiffusionKind = DiffusionKind::OrnsteinUhlenbeck;
const DT: f64 = 1.0 / 252.0;
const STEP: [f64; 3] = [0.3, 0.3, 0.05];

fn cir_path(law: [f64; 3], n: usize, seed: u64) -> Array1<f64> {
  simulate_cir(law[0], law[1], law[2], law[1], DT, n, seed)
}

/// Four years of daily data do not exclude a speed of zero: under Jeffreys' prior the posterior
/// mean speed is still above 0.4 of the QMLE's, with `σ` beside the QMLE's.
#[test]
fn the_posterior_speed_is_identified_on_a_thousand_points() {
  let path = simulate_ou(2.0, 0.5, 0.3, 0.5, DT, 1000, 5);
  let qmle = fit_diffusion_qmle(path.view(), DT, OU).unwrap();
  for chain in [1, 2, 3] {
    let fit = fit_diffusion_bayesian(path.view(), DT, OU, STEP, 6000, 2000, chain).unwrap();
    assert!(
      fit.theta_mean > 0.4 * qmle.theta,
      "chain {chain}: speed {} for the QMLE's {}",
      fit.theta_mean,
      qmle.theta
    );
    assert!(
      (fit.sigma_mean / qmle.sigma - 1.0).abs() < 0.04,
      "chain {chain}: σ {} for the QMLE's {}",
      fit.sigma_mean,
      qmle.sigma
    );
  }
}

/// CIR's level prior is the path's own spread about its mean, so the posterior mean level stays
/// within a factor of 2 of the path's mean.
#[test]
fn the_posterior_cir_level_stays_by_the_path() {
  let path = cir_path([2.0, 0.04, 0.2], 1000, 5);
  let mean = path.sum() / path.len() as f64;
  for chain in [1, 2, 3] {
    let cir = DiffusionKind::Cir;
    let fit = fit_diffusion_bayesian(path.view(), DT, cir, STEP, 6000, 2000, chain).unwrap();
    assert!(
      0.5 * mean < fit.mu_mean && fit.mu_mean < 2.0 * mean,
      "chain {chain}: level {} for a path mean of {mean}",
      fit.mu_mean
    );
  }
}

/// On 60 points, where the QMLE's speed is 9 to 13 times the truth, the 95 % interval still holds
/// the true speed on at least 25 of 30 paths: the prior is not centred on that estimate.
#[test]
fn the_speeds_interval_covers_the_truth_on_short_paths() {
  for kind in [OU, DiffusionKind::Cir] {
    let covered = (1..=30)
      .filter(|seed| {
        let path = if kind == OU {
          simulate_ou(2.0, 0.5, 0.3, 0.5, DT, 60, 7000 + seed)
        } else {
          cir_path([2.0, 0.04, 0.2], 60, 7000 + seed)
        };
        let fit = fit_diffusion_bayesian(path.view(), DT, kind, STEP, 12_000, 3_000, 1).unwrap();
        fit.theta_ci.0 <= 2.0 && 2.0 <= fit.theta_ci.1
      })
      .count();
    assert!(covered >= 25, "{kind:?}: {covered} of 30");
  }
}

/// The speed's prior reads `θΔ` alone and the chain starts at the data's `θ₀`, so a step counted
/// in days gives the same posterior with the speed per day and `σ` per root day.
#[test]
fn a_step_in_days_gives_the_same_posterior_in_days() {
  let paths = [
    (OU, simulate_ou(2.0, 0.5, 0.3, 0.5, DT, 500, 5)),
    (DiffusionKind::Cir, cir_path([2.0, 0.04, 0.2], 500, 5)),
  ];
  for (kind, path) in paths {
    let years = fit_diffusion_bayesian(path.view(), DT, kind, STEP, 3000, 1000, 1).unwrap();
    let days = fit_diffusion_bayesian(path.view(), 1.0, kind, STEP, 3000, 1000, 1).unwrap();
    let pairs = [
      (days.theta_mean * 252.0, years.theta_mean),
      (days.theta_ci.1 * 252.0, years.theta_ci.1),
      (days.mu_mean, years.mu_mean),
      (days.sigma_mean * f64::sqrt(252.0), years.sigma_mean),
    ];
    for (got, want) in pairs {
      assert!(
        (got / want - 1.0).abs() < 1e-6,
        "{kind:?}: {got} for {want}"
      );
    }
    assert_eq!(days.acceptance_rate, years.acceptance_rate, "{kind:?}");
  }
}

/// The speed's log-prior in `q = ln θ` is that of `ρ/√(1 − ρ²)` in `θ` with the Jacobian `θ`, from
/// `θΔ = 1e-8` to 50, and its mass over `θ` is the arcsine's `π/(2Δ)`: the prior is proper.
#[test]
fn the_speeds_prior_is_jeffreys_for_the_autoregression_and_proper() {
  for dt in [DT, 1.0] {
    for rate in [1e-8, 1e-5, 1e-3, 0.1, 1.0, 7.0, 50.0] {
      let theta = rate / dt;
      let rho = f64::exp(-rate);
      // `1 − ρ²` as `(1 − ρ)(1 + ρ)` with `1 − ρ` from its series below `1e-4`, where it cancels.
      let one_less = if rate < 1e-4 {
        rate * (1.0 - rate / 2.0)
      } else {
        1.0 - rho
      };
      let want = (theta * rho / (one_less * (1.0 + rho)).sqrt()).ln();
      let got = speed_log_prior(theta.ln(), dt);
      assert!((got - want).abs() < 1e-9, "θΔ = {rate}: {got} for {want}");
    }
    let (low, high, steps) = (-80.0, f64::ln(60.0 / dt), 400_000);
    let h = (high - low) / steps as f64;
    let mass = (0..=steps)
      .map(|i| h * speed_log_prior(low + h * i as f64, dt).exp())
      .sum::<f64>();
    let want = std::f64::consts::FRAC_PI_2 / dt;
    assert!(
      (mass / want - 1.0).abs() < 1e-6,
      "Δ = {dt}: mass {mass} for {want}"
    );
  }
  let underflowed = speed_log_prior(-800.0, DT);
  assert!((underflowed + 400.0 + 0.5 * (2.0 * DT).ln()).abs() < 1e-9);
}

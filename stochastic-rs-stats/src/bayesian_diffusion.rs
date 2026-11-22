//! Random-walk Metropolis on the QMLE's quasi-likelihood of an OU or CIR path.
//!
//! `proposal_scale` acts on `ln θ`, `ln σ` and the level: CIR's `ln μ`, OU's `(μ − x̄)/s`.
//!
//! Reference: Eraker (2001), J. Bus. Econ. Stat. 19(2), 177-191, DOI: 10.1198/073500101316970403
//! Reference: Robert & Casella (2004), Monte Carlo Stat. Methods, ch. 7, 10.1007/978-1-4757-4145-2

use ndarray::Array1;
use ndarray::ArrayView1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::diffusion::ou::Ou;
use stochastic_rs_stochastic::params::VarianceBoundary;

use crate::StatsError;
use crate::filtering::mcmc::random_walk_metropolis;
use crate::qmle::Carrier;
use crate::qmle::DiffusionKind;
use crate::qmle::ar1_start;
use crate::qmle::assert_dt;
use crate::qmle::cir_of;
use crate::qmle::diffusion_path;
use crate::qmle::in_data_units;
use crate::qmle::ou_of;
use crate::qmle::path_spread;
use crate::qmle::quasi_log_likelihood;
use crate::stationarity::common::unit_scaled;
use crate::traits::FloatExt;

/// Posterior means and equal-tailed 95 % intervals in the names of [`Ou`] and [`Cir`]: `theta` the
/// speed, `mu` the level.
#[derive(Debug, Clone)]
pub struct BayesianDiffusionFit {
  /// Over the QMLE's speed it grows with the span `θT` of the path: 0.5–0.7 at up to 8, 0.78 at
  /// 20, 0.91 at 38, 0.98–1.00 from 200.
  pub theta_mean: f64,
  pub mu_mean: f64,
  pub sigma_mean: f64,
  pub theta_ci: (f64, f64),
  pub mu_ci: (f64, f64),
  pub sigma_ci: (f64, f64),
  /// Over the chain after burn-in.
  pub acceptance_rate: f64,
  pub kind: DiffusionKind,
}

impl BayesianDiffusionFit {
  /// The [`Ou`] at the posterior means; panics unless `kind` is `OrnsteinUhlenbeck`, and as
  /// [`Ou::new`].
  #[track_caller]
  pub fn to_ou<S: SeedExt>(
    &self,
    n: usize,
    x0: Option<f64>,
    t: Option<f64>,
    seed: S,
  ) -> Ou<f64, S> {
    ou_of(self.kind, self.means(), n, x0, t, seed)
  }

  /// The [`Cir`] at the posterior means; panics unless `kind` is `Cir`, and as [`Cir::new`].
  #[track_caller]
  pub fn to_cir<S: SeedExt>(
    &self,
    n: usize,
    x0: Option<f64>,
    t: Option<f64>,
    boundary: VarianceBoundary,
    seed: S,
  ) -> Cir<f64, S> {
    cir_of(self.kind, self.means(), n, x0, t, boundary, seed)
  }

  fn means(&self) -> [f64; 3] {
    [self.theta_mean, self.mu_mean, self.sigma_mean]
  }
}

/// Metropolis posterior under the priors below, defined where the QMLE refuses; `Err` as the path
/// checks and [`random_walk_metropolis`]; panics unless `0 < dt < ∞` and `n_samples >= 1`.
/// $$
/// p(\theta) \propto \frac{\rho}{\sqrt{1-\rho^2}},\ \rho = e^{-\theta\Delta}; \qquad
/// \mu \sim N(\bar x, (4s)^2),\ \mu > 0 \text{ for CIR}; \qquad
/// \ln\sigma \sim N(\ln\sigma_0, 4^2)
/// $$
/// Reference: Lee (2005), J. Korean Data & Information Science Society 16(1), 71-78, Lemma.
pub fn fit_diffusion_bayesian<T: FloatExt>(
  series: ArrayView1<T>,
  dt: f64,
  kind: DiffusionKind,
  proposal_scale: [f64; 3],
  n_samples: usize,
  burn_in: usize,
  seed: u64,
) -> Result<BayesianDiffusionFit, StatsError> {
  assert_dt(dt);
  // At unit scale, as the QMLE: the chain's steps and its prior then hold in any unit of the path.
  let (x, unit) = unit_scaled(diffusion_path(series, kind, 4)?);
  let carrier = Carrier::new(&x, kind)?;
  let start = ar1_start(&x, dt, kind, carrier)?;
  let ln_sigma = start[2];
  let level_prior = LevelPrior::new(&x, carrier)?;
  let tau = 4.0_f64;
  let inv_2tau2 = 1.0 / (2.0 * tau * tau);

  let log_post = |p: ArrayView1<f64>| -> f64 {
    let q = [p[0], p[1], p[2]];
    let [theta, mu, sigma] = carrier.params(q);
    if !(theta.is_finite() && mu.is_finite() && sigma.is_finite()) {
      return f64::NEG_INFINITY;
    }
    let ll = quasi_log_likelihood(&x, dt, kind, theta, mu, sigma);
    let log_prior = speed_log_prior(q[0], dt) + level_prior.ln_density(q[1], mu)
      - inv_2tau2 * (q[2] - ln_sigma).powi(2);
    ll + log_prior
  };

  let init = Array1::from(start.to_vec());
  let scale = Array1::from(vec![
    proposal_scale[0],
    proposal_scale[1],
    proposal_scale[2],
  ]);
  let mh = random_walk_metropolis(
    init.view(),
    log_post,
    scale.view(),
    n_samples,
    burn_in,
    seed,
  )?;

  let n = mh.samples.nrows();
  let mut thetas = Vec::with_capacity(n);
  let mut mus = Vec::with_capacity(n);
  let mut sigmas = Vec::with_capacity(n);
  for r in 0..n {
    let [theta, mu, sigma] =
      carrier.params([mh.samples[[r, 0]], mh.samples[[r, 1]], mh.samples[[r, 2]]]);
    thetas.push(theta);
    mus.push(mu);
    sigmas.push(sigma);
  }

  let in_units = |theta: f64, mu: f64, sigma: f64| in_data_units(kind, [theta, mu, sigma], unit);
  let [theta_mean, mu_mean, sigma_mean] = in_units(mean(&thetas), mean(&mus), mean(&sigmas))?;
  let theta_ci = credible_interval(&mut thetas);
  let (mu_ci, sigma_ci) = (credible_interval(&mut mus), credible_interval(&mut sigmas));
  let [_, mu_low, sigma_low] = in_units(theta_ci.0, mu_ci.0, sigma_ci.0)?;
  let [_, mu_high, sigma_high] = in_units(theta_ci.1, mu_ci.1, sigma_ci.1)?;

  Ok(BayesianDiffusionFit {
    theta_mean,
    mu_mean,
    sigma_mean,
    theta_ci,
    mu_ci: (mu_low, mu_high),
    sigma_ci: (sigma_low, sigma_high),
    acceptance_rate: mh.acceptance_rate,
    kind,
  })
}

/// Jeffreys' log-prior of the speed in `q = ln θ`, up to a constant: that of the Gaussian
/// autoregression's coefficient `ρ = e^{−θΔ}`, whose information in `n` steps is `n/(1 − ρ²)`.
/// $$
/// \pi(\rho) \propto (1-\rho^2)^{-1/2}, \qquad
/// \ln\pi(q) = q - \theta\Delta - \tfrac12\ln\left(1 - e^{-2\theta\Delta}\right)
/// $$
/// Reference: Lee (2005), J. Korean Data & Information Science Society 16(1), 71-78, Lemma.
fn speed_log_prior(ln_theta: f64, dt: f64) -> f64 {
  let rate = ln_theta.exp() * dt;
  if rate == 0.0 {
    // The limit `(2θΔ)^{−1/2}` in `q`, where the speed itself underflows.
    return 0.5 * (ln_theta - (2.0 * dt).ln());
  }
  ln_theta - rate - 0.5 * (-(-2.0 * rate).exp_m1()).ln()
}

/// `μ ~ N(x̄, (4s)²)` in the chain's level coordinate: OU's `(μ − x̄)/s`, and CIR's `ln μ`, which
/// keeps `μ > 0` and brings its Jacobian `μ`.
#[derive(Clone, Copy)]
enum LevelPrior {
  Standardised,
  Log { mean: f64, spread: f64 },
}

impl LevelPrior {
  /// `Err` for a CIR path whose spread is not finite and positive, as OU's carrier.
  fn new(x: &[f64], carrier: Carrier) -> Result<Self, StatsError> {
    match carrier {
      Carrier::Standardised { .. } => Ok(Self::Standardised),
      Carrier::Log { mean } => {
        let spread = path_spread(x, mean)?;
        Ok(Self::Log { mean, spread })
      }
    }
  }

  fn ln_density(self, q: f64, mu: f64) -> f64 {
    match self {
      Self::Standardised => -q.powi(2) / 32.0,
      Self::Log { mean, spread } => q - ((mu - mean) / spread).powi(2) / 32.0,
    }
  }
}

fn mean(v: &[f64]) -> f64 {
  v.iter().sum::<f64>() / v.len() as f64
}

/// Equal-tailed 95 % credible interval (2.5 % / 97.5 % empirical quantiles).
fn credible_interval(v: &mut [f64]) -> (f64, f64) {
  v.sort_by(f64::total_cmp);
  let n = v.len();
  let lo = v[((0.025 * n as f64) as usize).min(n - 1)];
  let hi = v[((0.975 * n as f64) as usize).min(n - 1)];
  (lo, hi)
}

#[cfg(test)]
mod prior_tests;
#[cfg(test)]
mod scale_tests;

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_core::simd_rng::Unseeded;
  use stochastic_rs_stochastic::traits::ProcessExt;

  use super::*;
  use crate::qmle::test_support::simulate_ou;

  /// Coverage is asked of the best of three pinned paths, as two 95 % intervals miss together once
  /// in ten; the point estimates, whose bands are wide, of every path.
  #[test]
  fn bayesian_ou_posterior_recovers_true_params() {
    let (tht, mt, st) = (1.5, 0.05, 0.15);
    let dt = 1.0 / 252.0;
    let mut covered = false;
    for (path_seed, chain_seed) in [(9u64, 42u64), (11, 43), (13, 44)] {
      let path = simulate_ou(tht, mt, st, 0.05, dt, 6000, path_seed);
      let res = fit_diffusion_bayesian(
        path.view(),
        dt,
        DiffusionKind::OrnsteinUhlenbeck,
        [0.08, 0.08, 0.04],
        20_000,
        4_000,
        chain_seed,
      )
      .unwrap();
      assert!(
        res.acceptance_rate > 0.15 && res.acceptance_rate < 0.7,
        "acceptance rate {} out of healthy band on path {path_seed}",
        res.acceptance_rate
      );
      assert!(
        (res.sigma_mean - st).abs() / st < 0.1,
        "σ posterior mean {} vs true {st} on path {path_seed}",
        res.sigma_mean
      );
      assert!(
        (res.theta_mean - tht).abs() / tht < 0.5,
        "θ posterior mean {} vs true {tht} on path {path_seed}",
        res.theta_mean
      );
      covered |=
        res.mu_ci.0 <= mt && mt <= res.mu_ci.1 && res.sigma_ci.0 <= st && st <= res.sigma_ci.1;
    }
    assert!(
      covered,
      "no pinned path had both 95% credible intervals cover the truth"
    );
  }

  /// A level below zero lies inside the posterior's interval: the chain carries the level in the
  /// path's standard deviations about its mean.
  #[test]
  fn a_negative_level_is_inside_the_credible_interval() {
    let (tht, mt, st) = (1.5, -0.03, 0.15);
    let dt = 1.0 / 252.0;
    let ou = DiffusionKind::OrnsteinUhlenbeck;
    let covered = [(9u64, 42u64), (11, 43), (13, 44)]
      .iter()
      .any(|&(path_seed, chain_seed)| {
        let path = simulate_ou(tht, mt, st, mt, dt, 6000, path_seed);
        let res = fit_diffusion_bayesian(
          path.view(),
          dt,
          ou,
          [0.08, 0.08, 0.04],
          8_000,
          2_000,
          chain_seed,
        )
        .unwrap();
        res.mu_ci.0 <= mt && mt <= res.mu_ci.1 && res.sigma_ci.0 <= st && st <= res.sigma_ci.1
      });
    assert!(
      covered,
      "no pinned path had both 95% credible intervals cover the truth"
    );
  }

  /// The posterior means are the speed and the level, as in `Ou`, so the conversion hands them
  /// over.
  #[test]
  fn a_bayesian_ou_fit_round_trips_into_the_process_it_names() {
    let truth = Ou::<f64, _>::new(
      5.0,
      -0.5,
      0.3,
      10_000,
      Some(-0.5),
      Some(100.0),
      Deterministic::new(5),
    );
    let path = truth.sample();
    let ou = DiffusionKind::OrnsteinUhlenbeck;
    let dt = 100.0 / 9_999.0;
    let fit = fit_diffusion_bayesian(path.view(), dt, ou, [0.08, 0.08, 0.04], 400, 100, 7).unwrap();
    assert_eq!(fit.kind, ou);
    assert!(
      (fit.theta_mean - 5.0).abs() < 1.5,
      "speed {}",
      fit.theta_mean
    );
    assert!((fit.mu_mean + 0.5).abs() < 0.05, "level {}", fit.mu_mean);
    let rebuilt = fit.to_ou(10_000, Some(-0.5), Some(100.0), Deterministic::new(6));
    assert_eq!(
      (rebuilt.theta, rebuilt.mu, rebuilt.sigma),
      (fit.theta_mean, fit.mu_mean, fit.sigma_mean)
    );
  }

  #[test]
  #[should_panic(expected = "kind must satisfy `kind == DiffusionKind::OrnsteinUhlenbeck`")]
  fn to_ou_on_a_bayesian_cir_fit_is_a_precondition() {
    let fit = BayesianDiffusionFit {
      theta_mean: 1.5,
      mu_mean: 0.04,
      sigma_mean: 0.2,
      theta_ci: (1.0, 2.0),
      mu_ci: (0.03, 0.05),
      sigma_ci: (0.1, 0.3),
      acceptance_rate: 0.3,
      kind: DiffusionKind::Cir,
    };
    let _ = fit.to_ou(10, None, None, Unseeded);
  }

  #[test]
  fn bayesian_diffusion_rejects_a_short_series() {
    let s = Array1::from(vec![0.05, 0.06, 0.04]);
    let fit = fit_diffusion_bayesian(
      s.view(),
      1.0 / 252.0,
      DiffusionKind::OrnsteinUhlenbeck,
      [0.1, 0.1, 0.1],
      100,
      10,
      1,
    );
    assert_eq!(
      fit.unwrap_err(),
      StatsError::InsufficientData { needed: 4, got: 3 }
    );
  }

  #[test]
  fn bayesian_cir_rejects_a_negative_observation() {
    let s = Array1::from(vec![0.04, 0.05, 0.045, -0.01, 0.04, 0.05]);
    let fit = fit_diffusion_bayesian(
      s.view(),
      1.0 / 252.0,
      DiffusionKind::Cir,
      [0.1, 0.1, 0.1],
      100,
      10,
      1,
    );
    assert_eq!(
      fit.unwrap_err(),
      StatsError::InvalidObservation {
        index: 3,
        value: -0.01,
        constraint: "finite and >= 0"
      }
    );
  }
}

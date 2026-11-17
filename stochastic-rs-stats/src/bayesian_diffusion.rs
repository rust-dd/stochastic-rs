//! Random-walk Metropolis posterior of an OU or CIR diffusion's `(θ, μ, σ)` on the QMLE's
//! quasi-likelihood, with an independent `N(·, 4²)` prior on each of `ln θ`, `ln μ` and `ln σ`.
//!
//! Reference: Eraker, B. (2001), "MCMC analysis of diffusion models with application to finance", Journal of Business & Economic Statistics 19(2), 177-191.
//! Reference: Robert, C. P. & Casella, G. (2004), Monte Carlo Statistical Methods, Springer, ch. 7 (random-walk Metropolis).

use ndarray::Array1;
use ndarray::ArrayView1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::diffusion::ou::Ou;
use stochastic_rs_stochastic::params::VarianceBoundary;

use crate::StatsError;
use crate::filtering::mcmc::random_walk_metropolis;
use crate::qmle::DiffusionKind;
use crate::qmle::assert_dt;
use crate::qmle::cir_of;
use crate::qmle::diffusion_path;
use crate::qmle::ou_of;
use crate::qmle::quasi_log_likelihood;
use crate::traits::FloatExt;

/// Posterior means and equal-tailed 95 % intervals in the names of [`Ou`] and [`Cir`]: `theta` the
/// speed, `mu` the level.
#[derive(Debug, Clone)]
pub struct BayesianDiffusionFit {
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
  /// The [`Ou`] at the posterior means; panics unless `kind` is `OrnsteinUhlenbeck`.
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

  /// The [`Cir`] at the posterior means; panics unless `kind` is `Cir`.
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

/// Metropolis posterior of `(ln θ, ln μ, ln σ)` from the prior means, `Err` if it is not finite
/// there; path `Err`s and panics as [`crate::qmle::fit_diffusion_qmle`], panics if `n_samples = 0`.
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
  let x = diffusion_path(series, kind)?;

  let sample_mean = x.iter().sum::<f64>() / x.len() as f64;
  // The speed about 1, the level about the sample mean and `σ` about 0.2.
  let prior_mean = [0.0_f64, sample_mean.max(1e-8).ln(), (-1.6_f64)];
  let tau = 4.0_f64;
  let inv_2tau2 = 1.0 / (2.0 * tau * tau);

  let log_post = |p: ArrayView1<f64>| -> f64 {
    let (lth, lmu, ls) = (p[0], p[1], p[2]);
    let (theta, mu, sigma) = (lth.exp(), lmu.exp(), ls.exp());
    if !(theta.is_finite() && mu.is_finite() && sigma.is_finite()) {
      return f64::NEG_INFINITY;
    }
    let ll = quasi_log_likelihood(&x, dt, kind, theta, mu, sigma);
    let log_prior = -inv_2tau2
      * ((lth - prior_mean[0]).powi(2)
        + (lmu - prior_mean[1]).powi(2)
        + (ls - prior_mean[2]).powi(2));
    ll + log_prior
  };

  let init = Array1::from(vec![prior_mean[0], prior_mean[1], prior_mean[2]]);
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
    thetas.push(mh.samples[[r, 0]].exp());
    mus.push(mh.samples[[r, 1]].exp());
    sigmas.push(mh.samples[[r, 2]].exp());
  }

  Ok(BayesianDiffusionFit {
    theta_mean: mean(&thetas),
    mu_mean: mean(&mus),
    sigma_mean: mean(&sigmas),
    theta_ci: credible_interval(&mut thetas),
    mu_ci: credible_interval(&mut mus),
    sigma_ci: credible_interval(&mut sigmas),
    acceptance_rate: mh.acceptance_rate,
    kind,
  })
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
mod tests {
  use ndarray::Array1;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_core::simd_rng::Unseeded;
  use stochastic_rs_distributions::SimdDistribution;
  use stochastic_rs_distributions::normal::SimdNormal;
  use stochastic_rs_stochastic::traits::ProcessExt;

  use super::*;

  fn simulate_ou(
    theta: f64,
    mu: f64,
    sigma: f64,
    x0: f64,
    dt: f64,
    n: usize,
    seed: u64,
  ) -> Array1<f64> {
    let a = (-theta * dt).exp();
    let sd = (sigma * sigma * (1.0 - a * a) / (2.0 * theta)).sqrt();
    let mut normal = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(seed));
    let mut path = Array1::<f64>::zeros(n);
    path[0] = x0;
    for t in 1..n {
      let z = normal.sample();
      path[t] = mu + (path[t - 1] - mu) * a + sd * z;
    }
    path
  }

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

  /// The posterior means are the speed and the level, as in `Ou`, so the conversion hands them
  /// over.
  #[test]
  fn a_bayesian_ou_fit_round_trips_into_the_process_it_names() {
    let truth = Ou::<f64, _>::new(
      5.0,
      0.5,
      0.3,
      10_000,
      Some(0.5),
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
    assert!((fit.mu_mean - 0.5).abs() < 0.05, "level {}", fit.mu_mean);
    let rebuilt = fit.to_ou(10_000, Some(0.5), Some(100.0), Deterministic::new(6));
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
    let s = Array1::from(vec![0.05, 0.06]);
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
      StatsError::InsufficientData { needed: 3, got: 2 }
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

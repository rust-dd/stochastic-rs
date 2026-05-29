//! # Bayesian calibration of mean-reverting diffusions
//!
//! Posterior inference for the $(\kappa, \theta, \sigma)$ parameters of an
//! Ornstein-Uhlenbeck or CIR diffusion by random-walk Metropolis. The
//! log-posterior composes the Gaussian quasi-likelihood of
//! [`crate::qmle`] with a weakly-informative prior, and the sampler is the
//! generic [`crate::filtering::mcmc::random_walk_metropolis`].
//!
//! The chain is run in log-parameter space $(\ln\kappa, \ln\theta,
//! \ln\sigma)$ so the positivity constraints hold automatically and the
//! Gaussian random-walk proposal is well-scaled across the orders of
//! magnitude the parameters span. The reported posterior summaries (mean,
//! 95 % credible interval) are computed on the back-transformed draws.
//!
//! The prior is independent $\mathcal{N}(\cdot, \tau^2)$ on each
//! log-parameter (a log-normal / scale prior on the original parameter),
//! with $\tau = 4$ — broad enough to let the likelihood dominate for any
//! reasonably-informative sample while keeping the posterior proper.
//!
//! References:
//! - Eraker, B. (2001), "MCMC analysis of diffusion models with
//!   application to finance", *Journal of Business & Economic Statistics*
//!   19(2), 177-191.
//! - Robert, C.P., Casella, G. (2004), *Monte Carlo Statistical Methods*,
//!   Springer, ch. 7 (random-walk Metropolis).

use ndarray::Array1;
use ndarray::ArrayView1;

use crate::filtering::mcmc::random_walk_metropolis;
use crate::qmle::DiffusionKind;
use crate::qmle::quasi_log_likelihood;
use crate::traits::FloatExt;

/// Posterior summary of a Bayesian diffusion calibration.
#[derive(Debug, Clone)]
pub struct BayesianDiffusionResult {
  /// Posterior mean of $\kappa$.
  pub kappa_mean: f64,
  /// Posterior mean of $\theta$.
  pub theta_mean: f64,
  /// Posterior mean of $\sigma$.
  pub sigma_mean: f64,
  /// 95 % credible interval for $\kappa$.
  pub kappa_ci: (f64, f64),
  /// 95 % credible interval for $\theta$.
  pub theta_ci: (f64, f64),
  /// 95 % credible interval for $\sigma$.
  pub sigma_ci: (f64, f64),
  /// Metropolis acceptance rate over the post-burn-in chain.
  pub acceptance_rate: f64,
}

/// Run random-walk Metropolis on the diffusion posterior.
///
/// `proposal_scale` is the per-dimension Gaussian step in **log-parameter**
/// space, ordered $(\ln\kappa, \ln\theta, \ln\sigma)$. Tune it so the
/// acceptance rate lands in the 0.2-0.5 band.
pub fn bayesian_diffusion<T: FloatExt>(
  series: ArrayView1<T>,
  dt: f64,
  kind: DiffusionKind,
  proposal_scale: [f64; 3],
  n_samples: usize,
  burn_in: usize,
  seed: u64,
) -> BayesianDiffusionResult {
  let n_obs = series.len();
  assert!(n_obs >= 3, "bayesian_diffusion requires at least 3 observations");
  assert!(dt.is_finite() && dt > 0.0, "dt must be finite and positive");
  let x: Vec<f64> = series.iter().map(|v| v.to_f64().unwrap()).collect();

  let sample_mean = x.iter().sum::<f64>() / x.len() as f64;
  // Prior means on the log-parameters: κ around 1, θ around the sample
  // mean, σ around 0.2; all with broad sd τ so the likelihood dominates.
  let prior_mu = [0.0_f64, sample_mean.max(1e-8).ln(), (-1.6_f64)];
  let tau = 4.0_f64;
  let inv_2tau2 = 1.0 / (2.0 * tau * tau);

  let log_post = |p: ArrayView1<f64>| -> f64 {
    let (lk, lt, ls) = (p[0], p[1], p[2]);
    let (kappa, theta, sigma) = (lk.exp(), lt.exp(), ls.exp());
    if !(kappa.is_finite() && theta.is_finite() && sigma.is_finite()) {
      return f64::NEG_INFINITY;
    }
    let ll = quasi_log_likelihood(&x, dt, kind, kappa, theta, sigma);
    let log_prior = -inv_2tau2
      * ((lk - prior_mu[0]).powi(2) + (lt - prior_mu[1]).powi(2) + (ls - prior_mu[2]).powi(2));
    ll + log_prior
  };

  // Initialise the chain at the prior means (broadly central).
  let init = Array1::from(vec![prior_mu[0], prior_mu[1], prior_mu[2]]);
  let scale = Array1::from(vec![proposal_scale[0], proposal_scale[1], proposal_scale[2]]);
  let mh = random_walk_metropolis(init.view(), log_post, scale.view(), n_samples, burn_in, seed);

  // Back-transform the log-parameter draws and summarise.
  let n = mh.samples.nrows();
  let mut kappas = Vec::with_capacity(n);
  let mut thetas = Vec::with_capacity(n);
  let mut sigmas = Vec::with_capacity(n);
  for r in 0..n {
    kappas.push(mh.samples[[r, 0]].exp());
    thetas.push(mh.samples[[r, 1]].exp());
    sigmas.push(mh.samples[[r, 2]].exp());
  }

  BayesianDiffusionResult {
    kappa_mean: mean(&kappas),
    theta_mean: mean(&thetas),
    sigma_mean: mean(&sigmas),
    kappa_ci: credible_interval(&mut kappas),
    theta_ci: credible_interval(&mut thetas),
    sigma_ci: credible_interval(&mut sigmas),
    acceptance_rate: mh.acceptance_rate,
  }
}

fn mean(v: &[f64]) -> f64 {
  v.iter().sum::<f64>() / v.len() as f64
}

/// Equal-tailed 95 % credible interval (2.5 % / 97.5 % empirical quantiles).
fn credible_interval(v: &mut [f64]) -> (f64, f64) {
  v.sort_by(|a, b| a.partial_cmp(b).unwrap());
  let n = v.len();
  let lo = v[((0.025 * n as f64) as usize).min(n - 1)];
  let hi = v[((0.975 * n as f64) as usize).min(n - 1)];
  (lo, hi)
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use rand::SeedableRng;
  use rand::rngs::StdRng;
  use rand_distr::Distribution;
  use rand_distr::Normal;

  use super::*;

  fn simulate_ou(kappa: f64, theta: f64, sigma: f64, x0: f64, dt: f64, n: usize, seed: u64) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let a = (-kappa * dt).exp();
    let sd = (sigma * sigma * (1.0 - a * a) / (2.0 * kappa)).sqrt();
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut path = Array1::<f64>::zeros(n);
    path[0] = x0;
    for t in 1..n {
      let z = normal.sample(&mut rng);
      path[t] = theta + (path[t - 1] - theta) * a + sd * z;
    }
    path
  }

  /// The posterior mean must recover the true OU parameters on a simulated
  /// path, and the 95 % credible interval must cover the truth. σ is the
  /// best-identified; θ carries the widest interval (its information scales
  /// with the time span, not the number of observations).
  #[test]
  fn bayesian_ou_posterior_recovers_true_params() {
    let (kt, tt, st) = (1.5, 0.05, 0.15);
    let dt = 1.0 / 252.0;
    let path = simulate_ou(kt, tt, st, 0.05, dt, 6000, 9);
    let res = bayesian_diffusion(
      path.view(),
      dt,
      DiffusionKind::OrnsteinUhlenbeck,
      [0.08, 0.08, 0.04],
      20_000,
      4_000,
      42,
    );
    assert!(
      res.acceptance_rate > 0.15 && res.acceptance_rate < 0.7,
      "acceptance rate {} out of healthy band",
      res.acceptance_rate
    );
    assert!(
      (res.sigma_mean - st).abs() / st < 0.1,
      "σ posterior mean {} vs true {st}",
      res.sigma_mean
    );
    assert!(
      (res.kappa_mean - kt).abs() / kt < 0.5,
      "κ posterior mean {} vs true {kt}",
      res.kappa_mean
    );
    // Credible intervals must cover the truth.
    assert!(
      res.theta_ci.0 <= tt && tt <= res.theta_ci.1,
      "θ 95% CI {:?} must cover true {tt}",
      res.theta_ci
    );
    assert!(
      res.sigma_ci.0 <= st && st <= res.sigma_ci.1,
      "σ 95% CI {:?} must cover true {st}",
      res.sigma_ci
    );
  }

  #[test]
  #[should_panic(expected = "at least 3 observations")]
  fn bayesian_diffusion_panics_on_short_series() {
    let s = Array1::from(vec![0.05, 0.06]);
    let _ = bayesian_diffusion(
      s.view(),
      1.0 / 252.0,
      DiffusionKind::OrnsteinUhlenbeck,
      [0.1, 0.1, 0.1],
      100,
      10,
      1,
    );
  }
}

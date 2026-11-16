//! # Particle-filter maximum likelihood for stochastic volatility
//!
//! Maximum-likelihood estimation of a discrete-time stochastic-volatility
//! (SV) state-space model by maximising the **particle-filter
//! approximation of the marginal likelihood** (Kitagawa 1996). The latent
//! log-variance follows an AR(1),
//!
//! $$
//! h_t = \mu + \phi(h_{t-1} - \mu) + \sigma_\eta\,\eta_t, \quad \eta_t \sim \mathcal{N}(0,1),
//! $$
//!
//! and the observed return is conditionally Gaussian with that variance,
//!
//! $$
//! y_t = e^{h_t/2}\,\varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0,1)
//! \;\Longrightarrow\; \log p(y_t \mid h_t)
//! = -\tfrac{1}{2}\bigl(h_t + y_t^2 e^{-h_t} + \ln 2\pi\bigr).
//! $$
//!
//! The marginal log-likelihood $\sum_t \log p(y_t \mid y_{1:t-1})$ has no
//! closed form because $h_t$ is latent and the model is non-linear /
//! non-Gaussian. A bootstrap particle filter
//! ([`crate::filtering::particle::ParticleFilter`]) approximates each
//! incremental term; summing them gives the objective the optimiser
//! maximises over $(\mu, \phi, \sigma_\eta)$.
//!
//! **Common random numbers.** The PF likelihood is a Monte-Carlo estimate
//! and would be too noisy for a deterministic optimiser if re-seeded each
//! evaluation. We fix the filter seed across all parameter evaluations so
//! the particle draws are shared and the likelihood surface is a smooth
//! (if slightly biased) function of the parameters — the standard
//! variance-reduction device for simulated-likelihood optimisation
//! (Hürzeler-Künsch 1998, Pitt 2002).
//!
//! References:
//! - Kitagawa, G. (1996), "Monte Carlo filter and smoother for
//!   non-Gaussian nonlinear state space models", *Journal of
//!   Computational and Graphical Statistics* 5(1), 1-25.
//! - Pitt, M.K. (2002), "Smooth particle filters for likelihood
//!   evaluation and maximisation", Warwick Economic Research Paper 651.
//! - Hürzeler, M., Künsch, H.R. (1998), "Monte Carlo approximations for
//!   general state-space models", *JCGS* 7(2), 175-193.

use ndarray::ArrayView1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::traits::SimdFloatExt;

use crate::StatsError;
use crate::input::reject_constant;
use crate::input::validate;
use crate::optim::nelder_mead;
use crate::traits::FloatExt;
use crate::traits::RealExt;

/// Result of a particle-filter MLE fit of the SV model.
#[derive(Debug, Clone)]
pub struct ParticleMleFit {
  /// Long-run log-variance level $\mu$.
  pub mu: f64,
  /// Log-variance persistence $\phi \in (-1, 1)$.
  pub phi: f64,
  /// Log-variance innovation standard deviation $\sigma_\eta > 0$.
  pub sigma_eta: f64,
  /// Maximised particle-filter log-likelihood.
  pub log_likelihood: f64,
  /// Nelder-Mead iterations.
  pub iterations: usize,
  /// Whether the optimiser converged.
  pub converged: bool,
}

/// Particle-filter log-likelihood of the SV model at parameters
/// $(\mu, \phi, \sigma_\eta)$ for the return series `y`, using
/// `n_particles` and a fixed `pf_seed` (common random numbers).
///
/// This is a specialised, allocation-free 1-D bootstrap filter: the latent
/// log-variance lives in flat `Vec<f64>` buffers reused across steps, so
/// the optimiser's inner loop performs no per-particle heap allocation
/// (the generic [`crate::filtering::particle::ParticleFilter`] returns an
/// `Array1` per particle, which is the right shape for general state-space
/// work but allocation-bound for the MLE hot loop). The buffered
/// `SimdNormal` amortises its 64-sample buffer across the per-particle
/// propagation draws, and both the propagation normals and the
/// systematic-resampling uniforms are seeded deterministically from
/// `pf_seed`, giving common random numbers across parameter evaluations so
/// the likelihood surface is smooth in $(\mu, \phi, \sigma_\eta)$.
fn sv_pf_loglik(
  y: &[f64],
  mu: f64,
  phi: f64,
  sigma_eta: f64,
  n_particles: usize,
  pf_seed: u64,
) -> f64 {
  let n = n_particles;
  let ln_n = (n as f64).ln();
  let ln_2pi = (2.0 * std::f64::consts::PI).ln();
  let stat_sd = (sigma_eta * sigma_eta / (1.0 - phi * phi)).sqrt();

  // Propagation normals (buffer-amortised) + resampling uniforms, both
  // deterministic for common random numbers.
  let mut normal =
    SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(pf_seed ^ 0x1234_5678));
  let mut resample_rng = Deterministic::new(pf_seed ^ 0xA5A5_5A5A).rng();

  // Initialise particles from the stationary distribution.
  let mut h: Vec<f64> = (0..n).map(|_| mu + stat_sd * normal.sample()).collect();
  let mut resampled = vec![0.0_f64; n];
  let mut log_obs = vec![0.0_f64; n];

  let mut total = 0.0;
  for (t, &yt) in y.iter().enumerate() {
    let max_lo = observation_log_densities(&h, yt, ln_2pi, &mut log_obs);
    // Incremental marginal log-likelihood log[(1/N) Σ_i e^{log_obs_i}].
    let mut sum_exp = 0.0;
    for &lo in &log_obs {
      sum_exp += (lo - max_lo).exp();
    }
    total += max_lo + sum_exp.ln() - ln_n;

    // Systematic resampling on the normalised observation weights.
    let total_w = sum_exp; // = Σ e^{lo - max_lo}
    let u0 = f64::sample_uniform_simd(&mut resample_rng) / n as f64;
    let mut cumulative = 0.0;
    let mut idx = 0usize;
    for (k, slot) in resampled.iter_mut().enumerate() {
      let point = u0 + k as f64 / n as f64;
      while idx + 1 < n && cumulative + (log_obs[idx] - max_lo).exp() / total_w < point {
        cumulative += (log_obs[idx] - max_lo).exp() / total_w;
        idx += 1;
      }
      *slot = h[idx];
    }
    std::mem::swap(&mut h, &mut resampled);

    // Propagate to the next step (skip after the last observation).
    if t + 1 < y.len() {
      for hi in h.iter_mut() {
        *hi = mu + phi * (*hi - mu) + sigma_eta * normal.sample();
      }
    }
  }
  total
}

/// Writes each particle's `log p(y_t | h)` into `log_obs` and returns their maximum, NaN if any is.
fn observation_log_densities(h: &[f64], yt: f64, ln_2pi: f64, log_obs: &mut [f64]) -> f64 {
  for (lo, &hi) in log_obs.iter_mut().zip(h) {
    *lo = -0.5 * (hi + yt * yt * (-hi).exp() + ln_2pi);
  }
  f64::max_or_nan_fold(log_obs.iter().copied())
}

/// SV particle-filter MLE, 500-2000 `n_particles` (0 panics), a common-random-numbers `seed`;
/// `Err` below 10 returns, on a non-finite or constant one or a non-finite start likelihood.
pub fn fit_sv_particle_mle<T: FloatExt>(
  returns: ArrayView1<T>,
  n_particles: usize,
  seed: u64,
) -> Result<ParticleMleFit, StatsError> {
  assert!(
    n_particles >= 1,
    "n_particles must satisfy `n_particles >= 1`, got n_particles = {n_particles:?}"
  );
  let y = returns
    .iter()
    .map(|v| v.to_f64().unwrap())
    .collect::<Vec<f64>>();
  validate(&y, 10)?;
  reject_constant(&y, "constant series")?;

  // Unconstrained reparameterisation: μ free, φ = tanh(p1) ∈ (-1,1),
  // σ_η = exp(p2) > 0.
  let unpack = |p: &[f64; 3]| -> (f64, f64, f64) { (p[0], p[1].tanh(), p[2].exp()) };

  // Initial guess: μ from log of sample variance, φ persistent, σ_η modest.
  let sample_var = y.iter().map(|v| v * v).sum::<f64>() / y.len() as f64;
  let mu0 = sample_var.ln();
  let phi0 = 0.9_f64;
  let sigma0 = 0.3_f64;
  let start = [mu0, phi0.atanh(), sigma0.ln()];

  let neg_ll = |p: &[f64; 3]| -> f64 {
    let (mu, phi, sigma_eta) = unpack(p);
    if !(mu.is_finite() && phi.abs() < 0.9999 && sigma_eta > 0.0 && sigma_eta.is_finite()) {
      return f64::INFINITY;
    }
    -sv_pf_loglik(&y, mu, phi, sigma_eta, n_particles, seed)
  };
  // A subnormal mean square has a finite log, yet `e^{−h}` then overflows at every particle.
  if !neg_ll(&start).is_finite() {
    return Err(StatsError::Numerical {
      stage: "starting point",
    });
  }

  // The PF likelihood is a Monte-Carlo estimate with small
  // resampling-induced kinks, so the simplex never reaches the analytic
  // tolerance; cap the iterations to bound runtime while still locating
  // the optimum region.
  let (p, iters, converged) = nelder_mead(start, 250, neg_ll);
  let (mu, phi, sigma_eta) = unpack(&p);

  Ok(ParticleMleFit {
    mu,
    phi,
    sigma_eta,
    log_likelihood: -neg_ll(&p),
    iterations: iters,
    converged,
  })
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::*;

  /// Simulate the SV model: AR(1) latent log-variance + conditionally
  /// Gaussian returns.
  fn simulate_sv(mu: f64, phi: f64, sigma_eta: f64, n: usize, seed: u64) -> Array1<f64> {
    let mut normal = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(seed));
    let stat_sd = (sigma_eta * sigma_eta / (1.0 - phi * phi)).sqrt();
    let mut h = mu + stat_sd * normal.sample();
    let mut y = Array1::<f64>::zeros(n);
    for t in 0..n {
      let eps = normal.sample();
      y[t] = (h / 2.0).exp() * eps;
      let eta = normal.sample();
      h = mu + phi * (h - mu) + sigma_eta * eta;
    }
    y
  }

  /// Round-trip: simulate SV data with known parameters, then PF-MLE must
  /// recover them. SV likelihood estimation is genuinely hard — μ (the
  /// log-variance level) is the best-identified; φ (persistence) and
  /// σ_η (vol-of-log-vol) carry wider sampling bands. Tolerances reflect
  /// realistic PF-MLE accuracy on a 2000-point series.
  #[test]
  fn particle_mle_recovers_sv_params() {
    let (mu_t, phi_t, sig_t) = (-9.0, 0.9, 0.30);
    let returns = simulate_sv(mu_t, phi_t, sig_t, 1500, 17);
    let res = fit_sv_particle_mle(returns.view(), 600, 4242).unwrap();
    assert!(res.converged, "PF-MLE optimiser must converge");
    assert!(
      (res.mu - mu_t).abs() < 0.7,
      "μ = {} vs true {mu_t} (>0.7 abs off)",
      res.mu
    );
    assert!(
      (res.phi - phi_t).abs() < 0.15,
      "φ = {} vs true {phi_t} (>0.15 abs off)",
      res.phi
    );
    assert!(
      res.sigma_eta > 0.1 && res.sigma_eta < 0.7,
      "σ_η = {} out of plausible [0.1, 0.7] band (true {sig_t})",
      res.sigma_eta
    );
  }

  #[test]
  fn a_nan_log_density_makes_the_particles_maximum_nan() {
    let ln_2pi = (2.0 * std::f64::consts::PI).ln();
    let mut log_obs = [0.0; 3];
    let max = observation_log_densities(&[0.0, -1.0, 2.0], 0.1, ln_2pi, &mut log_obs);
    assert_eq!(max.to_bits(), log_obs[1].to_bits());
    let max = observation_log_densities(&[0.0, f64::NAN, -1.0], 0.1, ln_2pi, &mut log_obs);
    assert!(max.is_nan(), "a NaN particle was dropped: {max}");
  }

  /// `μ` is a log-variance: returns scaled by `c` shift it by `2 ln c` and leave `φ` and `σ_η`.
  #[test]
  fn rescaled_returns_shift_mu_by_twice_the_log_scale() {
    let returns = simulate_sv(-9.0, 0.9, 0.3, 120, 3);
    let unit = fit_sv_particle_mle(returns.view(), 64, 11).unwrap();
    let c = 1e-7_f64;
    let scaled = fit_sv_particle_mle(returns.mapv(|r| r * c).view(), 64, 11).unwrap();
    assert!(
      (scaled.mu - 2.0 * c.ln() - unit.mu).abs() < 1e-6,
      "{scaled:?} against {unit:?}"
    );
    assert!(
      (scaled.phi - unit.phi).abs() < 1e-6,
      "{scaled:?} against {unit:?}"
    );
    assert!(
      (scaled.sigma_eta - unit.sigma_eta).abs() < 1e-6,
      "{scaled:?} against {unit:?}"
    );
  }

  #[test]
  fn a_mean_square_that_underflows_or_overflows_is_a_numerical_error() {
    let returns = simulate_sv(-9.0, 0.9, 0.3, 50, 3);
    for c in [1e-170, 1e160] {
      assert_eq!(
        fit_sv_particle_mle(returns.mapv(|r| r * c).view(), 32, 11).unwrap_err(),
        StatsError::Numerical {
          stage: "starting point"
        },
        "c = {c:e}"
      );
    }
  }

  /// A subnormal mean square has a finite log, but `e^{−h}` then overflows at every particle.
  #[test]
  fn a_start_without_a_finite_likelihood_is_a_numerical_error() {
    let returns = simulate_sv(-9.0, 0.9, 0.3, 50, 3);
    for c in [1e-154, 1e-158] {
      assert_eq!(
        fit_sv_particle_mle(returns.mapv(|r| r * c).view(), 32, 11).unwrap_err(),
        StatsError::Numerical {
          stage: "starting point"
        },
        "c = {c:e}"
      );
    }
  }

  /// The stationary start `N(μ, σ_η²/(1 − φ²))` does not exist at `|φ| >= 1`.
  #[test]
  fn the_likelihood_is_nan_without_a_stationary_law() {
    let returns = simulate_sv(-9.0, 0.9, 0.3, 20, 3);
    for phi in [1.0, 1.5] {
      assert!(sv_pf_loglik(returns.as_slice().unwrap(), -9.0, phi, 0.3, 16, 11).is_nan());
    }
  }

  #[test]
  fn particle_mle_rejects_a_short_series() {
    let s = Array1::from(vec![0.01, -0.02, 0.005]);
    assert_eq!(
      fit_sv_particle_mle(s.view(), 100, 1).unwrap_err(),
      StatsError::InsufficientData { needed: 10, got: 3 }
    );
  }
}

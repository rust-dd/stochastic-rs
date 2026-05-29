//! # Quasi-Maximum Likelihood Estimation (QMLE) for diffusions
//!
//! QMLE fits the drift / diffusion parameters of a one-dimensional
//! diffusion by maximising a **Gaussian quasi-likelihood**: each
//! transition $X_{t+\Delta}\mid X_t$ is treated as
//! $\mathcal{N}\bigl(m(X_t;\psi),\,v(X_t;\psi)\bigr)$, using the model's
//! **exact** conditional mean and variance, even when the true transition
//! density is not Gaussian (Kessler 1997). The quasi-log-likelihood is
//!
//! $$
//! \ell(\psi) = -\tfrac{1}{2}\sum_{t}\Bigl[\ln\bigl(2\pi\,v(X_t;\psi)\bigr) +
//! \frac{\bigl(X_{t+\Delta} - m(X_t;\psi)\bigr)^2}{v(X_t;\psi)}\Bigr].
//! $$
//!
//! For the **Ornstein-Uhlenbeck / Vasicek** process the exact transition
//! *is* Gaussian, so QMLE coincides with the exact maximum-likelihood
//! estimator and matches the closed-form Vasicek AR(1) MLE. For the
//! **CIR** square-root process the transition is non-central $\chi^2$;
//! QMLE plugs in the exact CIR conditional mean / variance and the
//! Gaussian quasi-likelihood remains consistent (Bibby-Sørensen 1995,
//! Kessler 1997).
//!
//! Supported diffusions (both $dX = \kappa(\theta - X)\,dt + \dots$):
//! - [`DiffusionKind::OrnsteinUhlenbeck`]: $\sigma\,dW$, exact-Gaussian.
//! - [`DiffusionKind::Cir`]: $\sigma\sqrt{X}\,dW$, exact moments.
//!
//! References:
//! - Kessler, M. (1997), "Estimation of an ergodic diffusion from
//!   discrete observations", *Scandinavian Journal of Statistics* 24(2),
//!   211-229.
//! - Bibby, B.M., Sørensen, M. (1995), "Martingale estimation functions
//!   for discretely observed diffusion processes", *Bernoulli* 1(1/2),
//!   17-39.
//! - White, H. (1982), "Maximum likelihood estimation of misspecified
//!   models", *Econometrica* 50(1), 1-25 (quasi-likelihood theory).

use ndarray::ArrayView1;

use crate::optim::nelder_mead;
use crate::traits::FloatExt;

/// Diffusion family whose drift is mean-reverting $\kappa(\theta - X)$ and
/// whose diffusion coefficient distinguishes the member.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffusionKind {
  /// Ornstein-Uhlenbeck / Vasicek: $\sigma\,dW$ (constant diffusion,
  /// exact-Gaussian transition).
  OrnsteinUhlenbeck,
  /// Cox-Ingersoll-Ross: $\sigma\sqrt{X}\,dW$ (square-root diffusion).
  Cir,
}

/// QMLE fit of a mean-reverting diffusion.
#[derive(Debug, Clone)]
pub struct QmleResult {
  /// Mean-reversion speed $\kappa$.
  pub kappa: f64,
  /// Long-run level $\theta$.
  pub theta: f64,
  /// Diffusion scale $\sigma$.
  pub sigma: f64,
  /// Maximised Gaussian quasi-log-likelihood.
  pub log_likelihood: f64,
  /// Nelder-Mead iterations.
  pub iterations: usize,
  /// Whether the optimiser converged.
  pub converged: bool,
}

/// Exact conditional mean and variance of the transition $X_{t+\Delta}
/// \mid X_t$ for the given diffusion family.
pub(crate) fn conditional_moments(
  kind: DiffusionKind,
  x: f64,
  kappa: f64,
  theta: f64,
  sigma: f64,
  dt: f64,
) -> (f64, f64) {
  let a = (-kappa * dt).exp();
  let mean = theta + (x - theta) * a;
  let s2 = sigma * sigma;
  let var = match kind {
    // OU: Var = σ²(1 − e^{−2κΔ}) / (2κ).
    DiffusionKind::OrnsteinUhlenbeck => s2 * (1.0 - a * a) / (2.0 * kappa),
    // CIR: Var = X σ²/κ (a − a²) + θ σ²/(2κ) (1 − a)².
    DiffusionKind::Cir => {
      x * (s2 / kappa) * (a - a * a) + theta * (s2 / (2.0 * kappa)) * (1.0 - a) * (1.0 - a)
    }
  };
  (mean, var.max(1e-300))
}

/// Gaussian quasi-log-likelihood of a discretely-observed diffusion path
/// `x` (interval `dt`) at parameters $(\kappa, \theta, \sigma)$. Shared by
/// [`qmle`] and the Bayesian sampler in
/// [`crate::bayesian_diffusion`](crate::bayesian_diffusion).
pub(crate) fn quasi_log_likelihood(
  x: &[f64],
  dt: f64,
  kind: DiffusionKind,
  kappa: f64,
  theta: f64,
  sigma: f64,
) -> f64 {
  let mut ll = 0.0;
  for t in 0..x.len() - 1 {
    let (m, var) = conditional_moments(kind, x[t], kappa, theta, sigma, dt);
    let resid = x[t + 1] - m;
    ll += -0.5 * ((2.0 * std::f64::consts::PI * var).ln() + resid * resid / var);
  }
  ll
}

/// Fit a mean-reverting diffusion by Gaussian QMLE.
///
/// `series` is the discretely-observed path, `dt` the sampling interval.
/// Requires at least 3 observations.
pub fn qmle<T: FloatExt>(series: ArrayView1<T>, dt: f64, kind: DiffusionKind) -> QmleResult {
  let n_obs = series.len();
  assert!(n_obs >= 3, "qmle requires at least 3 observations");
  assert!(dt.is_finite() && dt > 0.0, "dt must be finite and positive");
  let x: Vec<f64> = series.iter().map(|v| v.to_f64().unwrap()).collect();

  // Negative quasi-log-likelihood (minimised). Parameters carried in
  // log-space so κ, θ, σ stay strictly positive.
  let neg_ll =
    |p: &[f64; 3]| -> f64 { -quasi_log_likelihood(&x, dt, kind, p[0].exp(), p[1].exp(), p[2].exp()) };

  // Initial guess from the AR(1) regression of X_{t+1} on X_t.
  let sample_mean = x.iter().sum::<f64>() / x.len() as f64;
  let theta0 = sample_mean.max(1e-8);
  let (mut num, mut den) = (0.0, 0.0);
  for t in 0..n_obs - 1 {
    num += (x[t] - sample_mean) * (x[t + 1] - sample_mean);
    den += (x[t] - sample_mean).powi(2);
  }
  let ac1 = if den > 0.0 {
    (num / den).clamp(1e-4, 0.999)
  } else {
    0.5
  };
  let kappa0 = (-ac1.ln() / dt).clamp(1e-3, 50.0);
  // Residual variance → σ² via the family's conditional-variance scale.
  let mut resid_var = 0.0;
  for t in 0..n_obs - 1 {
    let a = (-kappa0 * dt).exp();
    let m = theta0 + (x[t] - theta0) * a;
    resid_var += (x[t + 1] - m).powi(2);
  }
  resid_var /= (n_obs - 1) as f64;
  let a0 = (-kappa0 * dt).exp();
  let scale = match kind {
    DiffusionKind::OrnsteinUhlenbeck => (1.0 - a0 * a0) / (2.0 * kappa0),
    DiffusionKind::Cir => {
      sample_mean * (a0 - a0 * a0) / kappa0 + theta0 * (1.0 - a0).powi(2) / (2.0 * kappa0)
    }
  };
  let sigma0 = (resid_var / scale.max(1e-12)).sqrt().clamp(1e-3, 10.0);

  let (p, iters, converged) = nelder_mead([kappa0.ln(), theta0.ln(), sigma0.ln()], 2000, neg_ll);
  let (kappa, theta, sigma) = (p[0].exp(), p[1].exp(), p[2].exp());

  QmleResult {
    kappa,
    theta,
    sigma,
    log_likelihood: -neg_ll(&p),
    iterations: iters,
    converged,
  }
}

/// Closed-form exact MLE for the Ornstein-Uhlenbeck / Vasicek process, used
/// both as a fast path and as the reference the QMLE must match (the OU
/// transition is Gaussian, so QMLE = exact MLE).
///
/// AR(1) representation: $X_{t+\Delta} = \theta(1-\beta) + \beta X_t +
/// \varepsilon_t$ with $\beta = e^{-\kappa\Delta}$ and
/// $\operatorname{Var}(\varepsilon) = \sigma^2(1-\beta^2)/(2\kappa)$.
pub fn mle_ou_closed_form<T: FloatExt>(series: ArrayView1<T>, dt: f64) -> QmleResult {
  let n_obs = series.len();
  assert!(n_obs >= 3, "mle_ou_closed_form requires at least 3 observations");
  assert!(dt.is_finite() && dt > 0.0, "dt must be finite and positive");
  let x: Vec<f64> = series.iter().map(|v| v.to_f64().unwrap()).collect();
  let m = n_obs - 1;
  let mf = m as f64;

  let sx: f64 = (0..m).map(|t| x[t]).sum();
  let sy: f64 = (0..m).map(|t| x[t + 1]).sum();
  let sxx: f64 = (0..m).map(|t| x[t] * x[t]).sum();
  let sxy: f64 = (0..m).map(|t| x[t] * x[t + 1]).sum();

  let denom = mf * sxx - sx * sx;
  let beta = ((mf * sxy - sx * sy) / denom).clamp(1e-6, 0.999_999);
  let alpha = (sy - beta * sx) / mf;
  let theta = alpha / (1.0 - beta);
  let kappa = -beta.ln() / dt;

  let mut resid_var = 0.0;
  for t in 0..m {
    let pred = alpha + beta * x[t];
    resid_var += (x[t + 1] - pred).powi(2);
  }
  resid_var /= mf;
  let sigma2 = resid_var * 2.0 * kappa / (1.0 - beta * beta);
  let sigma = sigma2.max(1e-12).sqrt();

  // Gaussian log-likelihood at the closed-form estimate.
  let var_eps = sigma * sigma * (1.0 - beta * beta) / (2.0 * kappa);
  let mut ll = 0.0;
  for t in 0..m {
    let pred = theta + (x[t] - theta) * beta;
    ll += -0.5 * ((2.0 * std::f64::consts::PI * var_eps).ln() + (x[t + 1] - pred).powi(2) / var_eps);
  }

  QmleResult {
    kappa,
    theta,
    sigma,
    log_likelihood: ll,
    iterations: 0,
    converged: true,
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use rand::SeedableRng;
  use rand::rngs::StdRng;
  use rand_distr::Distribution;
  use rand_distr::Gamma;
  use rand_distr::Normal;
  use rand_distr::Poisson;

  use super::*;

  /// Exact OU transition: Gaussian with mean θ+(X−θ)e^{−κΔ} and variance
  /// σ²(1−e^{−2κΔ})/(2κ).
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

  /// Exact CIR transition via non-central χ² (Poisson mixture of Gamma).
  fn simulate_cir(kappa: f64, theta: f64, sigma: f64, x0: f64, dt: f64, n: usize, seed: u64) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let a = (-kappa * dt).exp();
    let s2 = sigma * sigma;
    let c = s2 * (1.0 - a) / (4.0 * kappa);
    let d = 4.0 * kappa * theta / s2;
    let mut path = Array1::<f64>::zeros(n);
    path[0] = x0;
    for t in 1..n {
      let lambda = path[t - 1] * 4.0 * kappa * a / (s2 * (1.0 - a));
      let n_pois: f64 = Poisson::new(lambda / 2.0).unwrap().sample(&mut rng);
      let shape = d / 2.0 + n_pois;
      let chi2 = if shape > 0.0 {
        Gamma::new(shape, 2.0).unwrap().sample(&mut rng)
      } else {
        0.0
      };
      path[t] = c * chi2;
    }
    path
  }

  /// For OU the Gaussian QMLE must match the closed-form Vasicek MLE to
  /// optimiser tolerance — the OU transition is exactly Gaussian, so the
  /// quasi-likelihood IS the true likelihood.
  #[test]
  fn qmle_ou_matches_closed_form_mle() {
    let path = simulate_ou(1.8, 0.06, 0.12, 0.06, 1.0 / 252.0, 6000, 3);
    let q = qmle(path.view(), 1.0 / 252.0, DiffusionKind::OrnsteinUhlenbeck);
    let cf = mle_ou_closed_form(path.view(), 1.0 / 252.0);
    assert!(
      (q.kappa - cf.kappa).abs() / cf.kappa < 1e-3,
      "QMLE κ={} vs closed-form κ={}",
      q.kappa,
      cf.kappa
    );
    assert!(
      (q.theta - cf.theta).abs() / cf.theta.abs() < 1e-3,
      "QMLE θ={} vs closed-form θ={}",
      q.theta,
      cf.theta
    );
    assert!(
      (q.sigma - cf.sigma).abs() / cf.sigma < 1e-3,
      "QMLE σ={} vs closed-form σ={}",
      q.sigma,
      cf.sigma
    );
  }

  /// OU round-trip: recover the true parameters within finite-sample
  /// noise. σ is the best-identified (it tracks the quadratic variation,
  /// scaling with the number of observations); κ is moderate; θ (the
  /// long-run mean) is the noisiest because its information scales with
  /// the continuous-time span $T_{\text{years}} \approx 32$ here, giving a
  /// one-sigma band $\sqrt{\sigma^2/(2\kappa T)} \approx 19\%$ of θ.
  #[test]
  fn qmle_ou_recovers_true_params() {
    let (kt, tt, st) = (1.8, 0.06, 0.12);
    let path = simulate_ou(kt, tt, st, 0.06, 1.0 / 252.0, 8000, 5);
    let q = qmle(path.view(), 1.0 / 252.0, DiffusionKind::OrnsteinUhlenbeck);
    assert!(q.converged);
    assert!((q.theta - tt).abs() / tt < 0.2, "θ={} vs {tt}", q.theta);
    assert!((q.kappa - kt).abs() / kt < 0.35, "κ={} vs {kt}", q.kappa);
    assert!((q.sigma - st).abs() / st < 0.06, "σ={} vs {st}", q.sigma);
  }

  /// CIR round-trip: QMLE with exact CIR conditional moments recovers the
  /// true parameters from an exactly-distributed CIR path.
  #[test]
  fn qmle_cir_recovers_true_params() {
    let (kt, tt, st) = (2.0, 0.04, 0.25);
    let path = simulate_cir(kt, tt, st, 0.04, 1.0 / 252.0, 8000, 7);
    let q = qmle(path.view(), 1.0 / 252.0, DiffusionKind::Cir);
    assert!(q.converged);
    assert!((q.theta - tt).abs() / tt < 0.12, "θ={} vs {tt}", q.theta);
    assert!((q.kappa - kt).abs() / kt < 0.5, "κ={} vs {kt}", q.kappa);
    assert!((q.sigma - st).abs() / st < 0.3, "σ={} vs {st}", q.sigma);
  }

  #[test]
  #[should_panic(expected = "at least 3 observations")]
  fn qmle_panics_on_short_series() {
    let s = Array1::from(vec![0.05, 0.06]);
    let _ = qmle(s.view(), 1.0 / 252.0, DiffusionKind::OrnsteinUhlenbeck);
  }
}

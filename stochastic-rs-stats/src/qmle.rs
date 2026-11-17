//! Gaussian quasi-likelihood of `dX = θ(μ − X) dt + σ X^γ dW`, OU's `γ = 0` and CIR's `γ = ½`: each
//! transition a normal with the model's exact conditional mean and variance, for OU its exact law.
//!
//! $$
//! \ell(\psi) = -\tfrac{1}{2}\sum_{t}\Bigl[\ln\bigl(2\pi\,v(X_t;\psi)\bigr) +
//! \frac{\bigl(X_{t+\Delta} - m(X_t;\psi)\bigr)^2}{v(X_t;\psi)}\Bigr]
//! $$
//!
//! Reference: Kessler, M. (1997), "Estimation of an ergodic diffusion from discrete observations", Scandinavian Journal of Statistics 24(2), 211-229.
//! Reference: Bibby, B. M. & Sørensen, M. (1995), "Martingale estimation functions for discretely observed diffusion processes", Bernoulli 1(1/2), 17-39.
//! Reference: White, H. (1982), "Maximum likelihood estimation of misspecified models", Econometrica 50(1), 1-25.

use ndarray::ArrayView1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::diffusion::ou::Ou;
use stochastic_rs_stochastic::params::VarianceBoundary;

use crate::StatsError;
use crate::input::reject_constant;
use crate::input::validate;
use crate::input::validate_domain;
use crate::optim::nelder_mead;
use crate::traits::FloatExt;

#[cfg(test)]
mod tests;

/// A mean-reverting diffusion `dX = θ(μ − X) dt + σ X^γ dW`, named by its `γ`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DiffusionKind {
  /// Ornstein–Uhlenbeck / Vasicek, `γ = 0`: the transition is Gaussian.
  OrnsteinUhlenbeck,
  /// Cox–Ingersoll–Ross, `γ = ½`.
  Cir,
}

/// A fit of `dX = θ(μ − X) dt + …` in the names of [`Ou`] and [`Cir`]: `theta` the speed, `mu` the
/// level.
#[derive(Debug, Clone)]
pub struct QmleFit {
  pub theta: f64,
  pub mu: f64,
  pub sigma: f64,
  pub kind: DiffusionKind,
  /// The Gaussian quasi-log-likelihood at the fit, for OU the exact one.
  pub log_likelihood: f64,
  /// Simplex iterations, 0 for [`fit_ou_mle`]'s closed form.
  pub iterations: usize,
  pub converged: bool,
}

impl QmleFit {
  /// The fitted [`Ou`]; panics unless `kind` is `OrnsteinUhlenbeck`.
  #[track_caller]
  pub fn to_ou<S: SeedExt>(
    &self,
    n: usize,
    x0: Option<f64>,
    t: Option<f64>,
    seed: S,
  ) -> Ou<f64, S> {
    ou_of(self.kind, [self.theta, self.mu, self.sigma], n, x0, t, seed)
  }

  /// The fitted [`Cir`]; panics unless `kind` is `Cir`.
  #[track_caller]
  pub fn to_cir<S: SeedExt>(
    &self,
    n: usize,
    x0: Option<f64>,
    t: Option<f64>,
    boundary: VarianceBoundary,
    seed: S,
  ) -> Cir<f64, S> {
    cir_of(
      self.kind,
      [self.theta, self.mu, self.sigma],
      n,
      x0,
      t,
      boundary,
      seed,
    )
  }
}

/// [`Ou`] at `[theta, mu, sigma]`; panics unless `kind` is `OrnsteinUhlenbeck`.
#[track_caller]
pub(crate) fn ou_of<S: SeedExt>(
  kind: DiffusionKind,
  [theta, mu, sigma]: [f64; 3],
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
  seed: S,
) -> Ou<f64, S> {
  assert!(
    kind == DiffusionKind::OrnsteinUhlenbeck,
    "kind must satisfy `kind == DiffusionKind::OrnsteinUhlenbeck`, got kind = {kind:?}"
  );
  Ou::new(theta, mu, sigma, n, x0, t, seed)
}

/// [`Cir`] at `[theta, mu, sigma]`; panics unless `kind` is `Cir`.
#[track_caller]
pub(crate) fn cir_of<S: SeedExt>(
  kind: DiffusionKind,
  [theta, mu, sigma]: [f64; 3],
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
  boundary: VarianceBoundary,
  seed: S,
) -> Cir<f64, S> {
  assert!(
    kind == DiffusionKind::Cir,
    "kind must satisfy `kind == DiffusionKind::Cir`, got kind = {kind:?}"
  );
  Cir::new(theta, mu, sigma, n, x0, t, boundary, seed)
}

/// The exact mean `μ + (x − μ) a` of `X_{t+Δ} | X_t = x`, `a = e^{−θΔ}`, and its variance
/// `σ²(1 − a²)/(2θ)` (OU) or `x σ²/θ (a − a²) + μ σ²/(2θ) (1 − a)²` (CIR).
pub(crate) fn conditional_moments(
  kind: DiffusionKind,
  x: f64,
  theta: f64,
  mu: f64,
  sigma: f64,
  dt: f64,
) -> (f64, f64) {
  let a = (-theta * dt).exp();
  let mean = mu + (x - mu) * a;
  let s2 = sigma * sigma;
  let var = match kind {
    DiffusionKind::OrnsteinUhlenbeck => s2 * (1.0 - a * a) / (2.0 * theta),
    DiffusionKind::Cir => {
      x * (s2 / theta) * (a - a * a) + mu * (s2 / (2.0 * theta)) * (1.0 - a) * (1.0 - a)
    }
  };
  (mean, var)
}

/// Gaussian quasi-log-likelihood of `x` every `dt` at $(\theta, \mu, \sigma)$; NaN where a
/// conditional variance is not positive, a point the simplex and the Metropolis sampler reject.
pub(crate) fn quasi_log_likelihood(
  x: &[f64],
  dt: f64,
  kind: DiffusionKind,
  theta: f64,
  mu: f64,
  sigma: f64,
) -> f64 {
  let mut ll = 0.0;
  for t in 0..x.len() - 1 {
    let (m, var) = conditional_moments(kind, x[t], theta, mu, sigma, dt);
    let resid = x[t + 1] - m;
    ll += -0.5 * ((2.0 * std::f64::consts::PI * var).ln() + resid * resid / var);
  }
  ll
}

/// Gaussian QMLE of a mean-reverting diffusion every `dt`; `Err` below 4 points, on one non-finite
/// or (CIR) negative, a constant path or a non-finite start or optimum; panics unless `0 < dt < ∞`.
pub fn fit_diffusion_qmle<T: FloatExt>(
  series: ArrayView1<T>,
  dt: f64,
  kind: DiffusionKind,
) -> Result<QmleFit, StatsError> {
  assert_dt(dt);
  // Three points are two transitions, which the drift's two coefficients fit exactly.
  let x = diffusion_path(series, kind, 4)?;
  // In logs, so that the speed, the level and `σ` stay positive.
  let neg_ll = |p: &[f64; 3]| -> f64 {
    -quasi_log_likelihood(&x, dt, kind, p[0].exp(), p[1].exp(), p[2].exp())
  };
  let start = ar1_start(&x, dt, kind)?;

  let (p, iterations, converged) = nelder_mead(start.map(f64::ln), 2000, neg_ll);
  let (theta, mu, sigma) = (p[0].exp(), p[1].exp(), p[2].exp());
  let log_likelihood = -neg_ll(&p);
  if !log_likelihood.is_finite() {
    return Err(StatsError::Numerical {
      stage: "quasi-log-likelihood",
    });
  }

  Ok(QmleFit {
    theta,
    mu,
    sigma,
    kind,
    log_likelihood,
    iterations,
    converged,
  })
}

/// The path as `f64`: `min_n` or more finite points, not all equal and, for CIR, whose conditional
/// variance turns negative below zero, none negative.
pub(crate) fn diffusion_path<T: FloatExt>(
  series: ArrayView1<T>,
  kind: DiffusionKind,
  min_n: usize,
) -> Result<Vec<f64>, StatsError> {
  let x = series
    .iter()
    .map(|v| v.to_f64().unwrap())
    .collect::<Vec<f64>>();
  match kind {
    DiffusionKind::OrnsteinUhlenbeck => validate(&x, min_n)?,
    DiffusionKind::Cir => validate_domain(&x, min_n, "finite and >= 0", |v| v >= 0.0)?,
  }
  reject_constant(&x, "constant series")?;
  Ok(x)
}

/// The path's mean, which centres the start and the prior; `Err` when it overflows.
pub(crate) fn path_mean(x: &[f64]) -> Result<f64, StatsError> {
  let mean = x.iter().sum::<f64>() / x.len() as f64;
  if mean.is_finite() {
    Ok(mean)
  } else {
    Err(StatsError::Numerical {
      stage: "starting point",
    })
  }
}

/// `[θ₀, μ₀, σ₀]`: the level at the mean, the speed from the lag-1 autocorrelation and `σ` from
/// the residuals about both; `Err` unless each is finite before the clamp that would hide it.
pub(crate) fn ar1_start(x: &[f64], dt: f64, kind: DiffusionKind) -> Result<[f64; 3], StatsError> {
  let nobs = x.len();
  let mean = path_mean(x)?;
  let mu0 = mean.max(1e-8);
  let (mut num, mut den) = (0.0, 0.0);
  for t in 0..nobs - 1 {
    num += (x[t] - mean) * (x[t + 1] - mean);
    den += (x[t] - mean).powi(2);
  }
  // `den` is zero only when the deviations' squares underflow: a constant path is refused before.
  let ac1 = num / den;
  let theta_raw = -ac1.clamp(1e-4, 0.999).ln() / dt;
  let theta0 = theta_raw.clamp(1e-3, 50.0);
  let a0 = (-theta0 * dt).exp();
  let mut resid_var = 0.0;
  for t in 0..nobs - 1 {
    let m = mu0 + (x[t] - mu0) * a0;
    resid_var += (x[t + 1] - m).powi(2);
  }
  resid_var /= (nobs - 1) as f64;
  let scale = match kind {
    DiffusionKind::OrnsteinUhlenbeck => (1.0 - a0 * a0) / (2.0 * theta0),
    DiffusionKind::Cir => {
      mean * (a0 - a0 * a0) / theta0 + mu0 * (1.0 - a0).powi(2) / (2.0 * theta0)
    }
  };
  let sigma_raw = (resid_var / scale).sqrt();
  if !(ac1.is_finite() && theta_raw.is_finite() && sigma_raw.is_finite()) {
    return Err(StatsError::Numerical {
      stage: "starting point",
    });
  }
  Ok([theta0, mu0, sigma_raw.clamp(1e-3, 10.0)])
}

/// Closed-form OU / Vasicek MLE through the AR(1) below; `Err` below 4 points, on a non-finite one,
/// a constant `x[..n - 1]`, a slope outside `(0, 1)` or zero residuals; panics unless `0 < dt < ∞`.
/// $$
/// X_{t+\Delta} = \mu(1-\beta) + \beta X_t + \varepsilon_t,\qquad \beta = e^{-\theta\Delta}
/// $$
pub fn fit_ou_mle<T: FloatExt>(series: ArrayView1<T>, dt: f64) -> Result<QmleFit, StatsError> {
  assert_dt(dt);
  let x = series
    .iter()
    .map(|v| v.to_f64().unwrap())
    .collect::<Vec<f64>>();
  // Two coefficients fit two transitions exactly, leaving the residual variance nothing to read.
  validate(&x, 4)?;
  let nobs = x.len();
  // The regressor is the path without its last point; its spread is the slope's denominator.
  reject_constant(&x[..nobs - 1], "constant series")?;
  let m = nobs - 1;
  let mf = m as f64;

  let sx: f64 = (0..m).map(|t| x[t]).sum();
  let sy: f64 = (0..m).map(|t| x[t + 1]).sum();
  let sxx: f64 = (0..m).map(|t| x[t] * x[t]).sum();
  let sxy: f64 = (0..m).map(|t| x[t] * x[t + 1]).sum();

  let denom = mf * sxx - sx * sx;
  let beta = (mf * sxy - sx * sy) / denom;
  if !beta.is_finite() {
    return Err(StatsError::Numerical {
      stage: "ar(1) slope",
    });
  }
  if beta >= 1.0 {
    return Err(StatsError::DegenerateData("series without mean reversion"));
  }
  if beta <= 0.0 {
    return Err(StatsError::DegenerateData(
      "series without positive autocorrelation",
    ));
  }
  let alpha = (sy - beta * sx) / mf;
  let mu = alpha / (1.0 - beta);
  let theta = -beta.ln() / dt;

  let mut resid_var = 0.0;
  for t in 0..m {
    let pred = alpha + beta * x[t];
    resid_var += (x[t + 1] - pred).powi(2);
  }
  resid_var /= mf;
  let sigma2 = resid_var * 2.0 * theta / (1.0 - beta * beta);
  if !sigma2.is_finite() {
    return Err(StatsError::Numerical { stage: "sigma" });
  }
  if sigma2 == 0.0 {
    return Err(StatsError::DegenerateData("zero residual variance"));
  }
  let sigma = sigma2.sqrt();

  // Gaussian log-likelihood at the closed-form estimate.
  let var_eps = sigma * sigma * (1.0 - beta * beta) / (2.0 * theta);
  let mut ll = 0.0;
  for t in 0..m {
    let pred = mu + (x[t] - mu) * beta;
    ll +=
      -0.5 * ((2.0 * std::f64::consts::PI * var_eps).ln() + (x[t + 1] - pred).powi(2) / var_eps);
  }

  Ok(QmleFit {
    theta,
    mu,
    sigma,
    kind: DiffusionKind::OrnsteinUhlenbeck,
    log_likelihood: ll,
    iterations: 0,
    converged: true,
  })
}

/// The sampling-interval precondition the diffusion fitters share.
#[track_caller]
pub(crate) fn assert_dt(dt: f64) {
  assert!(
    dt.is_finite() && dt > 0.0,
    "dt must satisfy `dt.is_finite() && dt > 0.0`, got dt = {dt:?}"
  );
}

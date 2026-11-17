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
use crate::stationarity::common::reject_exact_fit;
use crate::traits::FloatExt;

#[cfg(test)]
pub(crate) mod test_support;
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
  /// The fitted [`Ou`]; panics unless `kind` is `OrnsteinUhlenbeck`, and as [`Ou::new`].
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

  /// The fitted [`Cir`]; panics unless `kind` is `Cir`, and as [`Cir::new`].
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

/// [`Ou`] at `[theta, mu, sigma]`; panics unless `kind` is `OrnsteinUhlenbeck`, and as [`Ou::new`].
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

/// [`Cir`] at `[theta, mu, sigma]`; panics unless `kind` is `Cir`, and as [`Cir::new`].
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

/// `X_{t+Δ} | X_t = x`: the exact mean `μ + (x − μ) a`, `a = e^{−θΔ}`, and variance
/// `σ²(1 − a²)/(2θ)` (OU) or `x σ²/θ (a − a²) + μ σ²/(2θ) (1 − a)²` (CIR), linear in `x`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Transition {
  mu: f64,
  a: f64,
  var0: f64,
  var_x: f64,
}

impl Transition {
  /// At `(θ, μ, σ)` every `dt`; `1 − a` and `1 − a²` come from `expm1`, as a subtraction leaves
  /// them few digits where `θΔ` is small.
  pub(crate) fn new(kind: DiffusionKind, theta: f64, mu: f64, sigma: f64, dt: f64) -> Self {
    let a = (-theta * dt).exp();
    let s2 = sigma * sigma;
    let (var0, var_x) = match kind {
      DiffusionKind::OrnsteinUhlenbeck => (s2 * -(-2.0 * theta * dt).exp_m1() / (2.0 * theta), 0.0),
      DiffusionKind::Cir => {
        let b = -(-theta * dt).exp_m1();
        (mu * (s2 / (2.0 * theta)) * b * b, (s2 / theta) * (a * b))
      }
    };
    Self { mu, a, var0, var_x }
  }

  pub(crate) fn moments(self, x: f64) -> (f64, f64) {
    (self.mu + (x - self.mu) * self.a, self.var0 + self.var_x * x)
  }
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
  let transition = Transition::new(kind, theta, mu, sigma, dt);
  let mut ll = 0.0;
  for t in 0..x.len() - 1 {
    let (m, var) = transition.moments(x[t]);
    let resid = x[t + 1] - m;
    ll += -0.5 * ((2.0 * std::f64::consts::PI * var).ln() + resid * resid / var);
  }
  ll
}

/// QMLE every `dt`; `Err` below 4 points, on one non-finite or (CIR) negative, a constant path, a
/// non-finite start or optimum, an OU slope ∉ `(0, 1)` or exact fit; panics unless `0 < dt < ∞`.
pub fn fit_diffusion_qmle<T: FloatExt>(
  series: ArrayView1<T>,
  dt: f64,
  kind: DiffusionKind,
) -> Result<QmleFit, StatsError> {
  assert_dt(dt);
  // Three points are two transitions, which the drift's two coefficients fit exactly.
  let x = diffusion_path(series, kind, 4)?;
  let carrier = Carrier::new(&x, kind)?;
  let neg_ll = |p: &[f64; 3]| -> f64 {
    let [theta, mu, sigma] = carrier.params(*p);
    -quasi_log_likelihood(&x, dt, kind, theta, mu, sigma)
  };
  let start = ar1_start(&x, dt, kind, carrier)?;
  if kind == DiffusionKind::OrnsteinUhlenbeck {
    // The OU quasi-likelihood is the exact one: with no slope in `(0, 1)` its supremum lies on a
    // ridge or at the iid limit, and on an exact fit it grows without bound as `σ` falls.
    Ar1::of(&x, kind)?;
  }

  let (p, iterations, converged) = nelder_mead(start, 2000, neg_ll);
  let [theta, mu, sigma] = carrier.params(p);
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
fn path_mean(x: &[f64]) -> Result<f64, StatsError> {
  let mean = x.iter().sum::<f64>() / x.len() as f64;
  if mean.is_finite() {
    Ok(mean)
  } else {
    Err(StatsError::Numerical {
      stage: "starting point",
    })
  }
}

/// How the simplex and the chain carry `(θ, μ, σ)`: in logs, but OU's level as `x̄ + s p`, `s` the
/// path's standard deviation, so that it takes either sign and moves with the data's scale.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Carrier {
  Standardised { mean: f64, spread: f64 },
  Log { mean: f64 },
}

impl Carrier {
  /// `kind`'s carrier on `x`; `Err` unless the mean and, for OU, the standard deviation are finite,
  /// the deviation positive.
  pub(crate) fn new(x: &[f64], kind: DiffusionKind) -> Result<Self, StatsError> {
    let mean = path_mean(x)?;
    match kind {
      DiffusionKind::OrnsteinUhlenbeck => {
        let spread = (x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64).sqrt();
        // Zero only when the deviations' squares underflow: a constant path is refused before.
        if !spread.is_finite() || spread == 0.0 {
          return Err(StatsError::Numerical {
            stage: "starting point",
          });
        }
        Ok(Self::Standardised { mean, spread })
      }
      DiffusionKind::Cir => Ok(Self::Log { mean }),
    }
  }

  /// `[θ, μ, σ]` at the coordinates `p`.
  pub(crate) fn params(self, p: [f64; 3]) -> [f64; 3] {
    let mu = match self {
      Self::Standardised { mean, spread } => mean + spread * p[1],
      Self::Log { .. } => p[1].exp(),
    };
    [p[0].exp(), mu, p[2].exp()]
  }

  /// The level at the path's mean, kept above zero for the logarithm, and its coordinate.
  pub(crate) fn centre(self) -> (f64, f64) {
    match self {
      Self::Standardised { mean, .. } => (mean, 0.0),
      Self::Log { mean } => {
        let mu = mean.max(1e-8);
        (mu, mu.ln())
      }
    }
  }

  fn mean(self) -> f64 {
    match self {
      Self::Standardised { mean, .. } | Self::Log { mean } => mean,
    }
  }
}

/// The simplex's start `[ln θ₀, p₀, ln σ₀]`: the carrier's centre, the speed from the lag-1
/// autocorrelation and `σ` from the residuals, in the path's units; `Err` unless each is finite.
pub(crate) fn ar1_start(
  x: &[f64],
  dt: f64,
  kind: DiffusionKind,
  carrier: Carrier,
) -> Result<[f64; 3], StatsError> {
  let nobs = x.len();
  let mean = carrier.mean();
  let (mu0, p0) = carrier.centre();
  let (mut num, mut den) = (0.0, 0.0);
  for t in 0..nobs - 1 {
    num += (x[t] - mean) * (x[t + 1] - mean);
    den += (x[t] - mean).powi(2);
  }
  // `den` is zero only when the deviations' squares underflow: a constant path is refused before.
  let ac1 = num / den;
  let theta_raw = -ac1.clamp(1e-4, 0.999).ln() / dt;
  let theta0 = theta_raw.clamp(1e-3, 50.0);
  let unit = Transition::new(kind, theta0, mu0, 1.0, dt);
  let mut resid_var = 0.0;
  for t in 0..nobs - 1 {
    resid_var += (x[t + 1] - unit.moments(x[t]).0).powi(2);
  }
  resid_var /= (nobs - 1) as f64;
  // The conditional variance at `σ = 1` from the path's mean.
  let sigma_raw = (resid_var / unit.moments(mean).1).sqrt();
  if !(ac1.is_finite() && theta_raw.is_finite() && sigma_raw.is_finite()) {
    return Err(StatsError::Numerical {
      stage: "starting point",
    });
  }
  // Zero only where every residual's square underflows; a bound in absolute units would move the
  // start off the data's scale and leave the simplex on a plateau of the likelihood.
  let sigma0 = if sigma_raw > 0.0 { sigma_raw } else { 1e-3 };
  Ok([theta0.ln(), p0, sigma0.ln()])
}

/// Closed-form OU MLE; `Err` under 4 points, a non-finite one, a constant `x[..n-1]` or `x[1..]`, a
/// slope not in `(0, 1)`, an exact fit or a `σ²` not in `(0, ∞)`; panics unless `0 < dt < ∞`.
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
  let Ar1 {
    lagged_mean,
    next_mean,
    beta,
    residuals,
  } = Ar1::of(&x, DiffusionKind::OrnsteinUhlenbeck)?;
  let mu = (next_mean - beta * lagged_mean) / (1.0 - beta);
  let theta = -beta.ln() / dt;

  let mf = residuals.len() as f64;
  let resid_var = residuals.iter().map(|r| r * r).sum::<f64>() / mf;
  let sigma2 = resid_var * 2.0 * theta / (1.0 - beta * beta);
  if !sigma2.is_finite() || sigma2 <= 0.0 {
    return Err(StatsError::Numerical { stage: "sigma" });
  }
  let sigma = sigma2.sqrt();

  let var_eps = sigma * sigma * (1.0 - beta * beta) / (2.0 * theta);
  let mut ll = 0.0;
  for r in &residuals {
    ll += -0.5 * ((2.0 * std::f64::consts::PI * var_eps).ln() + r * r / var_eps);
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

/// `x[1..]` regressed on `x[..n-1]` about their means, the closed form's slope and residuals.
struct Ar1 {
  lagged_mean: f64,
  next_mean: f64,
  beta: f64,
  residuals: Vec<f64>,
}

impl Ar1 {
  /// The regression of `x`; `Err` on a constant `x[..n-1]` or `x[1..]`, a slope not finite or, for
  /// OU, not in `(0, 1)`, as no mean reversion fits it, or an exact fit.
  fn of(x: &[f64], kind: DiffusionKind) -> Result<Self, StatsError> {
    let m = x.len() - 1;
    let (lagged, next) = (&x[..m], &x[1..]);
    // The regressor's spread is the slope's denominator, and a constant regressand leaves the slope
    // nothing but rounding.
    reject_constant(lagged, "constant series")?;
    reject_constant(next, "constant series")?;
    let mf = m as f64;

    // About the means, as raw sums cancel where the level dwarfs the spread.
    let lagged_mean = lagged.iter().sum::<f64>() / mf;
    let next_mean = next.iter().sum::<f64>() / mf;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (u, v) in lagged.iter().zip(next) {
      let d = u - lagged_mean;
      sxy += d * (v - next_mean);
      sxx += d * d;
    }
    let beta = sxy / sxx;
    if !beta.is_finite() {
      return Err(StatsError::Numerical {
        stage: "ar(1) slope",
      });
    }
    if kind == DiffusionKind::OrnsteinUhlenbeck && beta >= 1.0 {
      return Err(StatsError::DegenerateData("series without mean reversion"));
    }
    if kind == DiffusionKind::OrnsteinUhlenbeck && beta <= 0.0 {
      return Err(StatsError::DegenerateData(
        "series without positive autocorrelation",
      ));
    }
    let residuals = lagged
      .iter()
      .zip(next)
      .map(|(u, v)| (v - next_mean) - beta * (u - lagged_mean))
      .collect::<Vec<f64>>();
    reject_exact_fit(next, &residuals, 1.0)?;
    Ok(Self {
      lagged_mean,
      next_mean,
      beta,
      residuals,
    })
  }
}

/// The sampling-interval precondition the diffusion fitters share.
#[track_caller]
pub(crate) fn assert_dt(dt: f64) {
  assert!(
    dt.is_finite() && dt > 0.0,
    "dt must satisfy `dt.is_finite() && dt > 0.0`, got dt = {dt:?}"
  );
}

//! Gaussian quasi-maximum likelihood for GARCH, Glosten–Jagannathan–Runkle GJR-GARCH and EGARCH.
//!
//! $$
//! \ell(\theta) = -\frac12\sum_{t=1}^{n}\Bigl[\log 2\pi + \log\sigma_t^2 +
//! \frac{\varepsilon_t^2}{\sigma_t^2}\Bigr],\qquad \varepsilon_t = r_t - \mu,\qquad
//! z_t = \varepsilon_t/\sigma_t
//! $$
//!
//! $$
//! \begin{aligned}
//! \text{GARCH}(p,q):&\quad \sigma_t^2 = \omega + \sum_{i=1}^{p}\alpha_i\varepsilon_{t-i}^2 +
//! \sum_{j=1}^{q}\beta_j\sigma_{t-j}^2,\\
//! \text{GJR}(p,q):&\quad \sigma_t^2 = \omega + \sum_{i=1}^{p}\bigl(\alpha_i +
//! \gamma_i\mathbf 1_{\{\varepsilon_{t-i}<0\}}\bigr)\varepsilon_{t-i}^2 +
//! \sum_{j=1}^{q}\beta_j\sigma_{t-j}^2,\\
//! \text{EGARCH}(p,q):&\quad \log\sigma_t^2 = \omega + \sum_{i=1}^{p}\bigl[\alpha_i(|z_{t-i}| -
//! \sqrt{2/\pi}) + \gamma_i z_{t-i}\bigr] + \sum_{j=1}^{q}\beta_j\log\sigma_{t-j}^2,
//! \end{aligned}
//! $$
//!
//! $$
//! \bar\sigma^2 = \frac1n\sum_t (r_t - \bar r)^2,\qquad
//! \widehat{\mathrm{Cov}}_{H} = (-H)^{-1},\qquad
//! \widehat{\mathrm{Cov}}_{BW} = H^{-1} J H^{-1},\quad J = \sum_t s_t s_t'
//! $$
//!
//! Pre-sample terms take the backcast `σ̄²`, as `arch` does; the sandwich is the robust covariance.
//!
//! Reference: Bollerslev (1986), J. Econometrics 31(3), 307-327, DOI: 10.1016/0304-4076(86)90063-1
//! Reference: Nelson (1991), Econometrica 59(2), 347-370, DOI: 10.2307/2938260
//! Reference: Straumann & Mikosch (2006), Ann. Statist. 34(5), 2449-2495, Example 3.8,
//! DOI: 10.1214/009053606000000803 (EGARCH(1,1)'s invertible region)
//! Reference: Glosten et al. (1993), J. Fin. 48(5), 1779-1801, 10.1111/j.1540-6261.1993.tb05128.x
//! Reference: Bollerslev–Wooldridge (1992), Econom. Rev. 11(2), 143-172, 10.1080/07474939208800229

mod inference;
mod recursion;
#[cfg(test)]
mod tests;
mod transform;
mod units;

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;

use crate::StatsError;
use crate::input::reject_constant;
use crate::input::validate;
use crate::optim::search::NO_VALUE;
use crate::optim::search::by_differences;
use crate::optim::search::minimise;
use crate::stationarity::common::unit_scaled;
use crate::traits::FloatExt;

/// Conditional-variance recursion to fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GarchKind {
  /// Bollerslev (1986) symmetric GARCH.
  Garch,
  /// Glosten–Jagannathan–Runkle (1993) threshold GARCH on the variance level.
  GjrGarch,
  /// Nelson (1991) exponential GARCH on the log-variance.
  Egarch,
}

/// Conditional-mean specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MeanSpec {
  /// $\varepsilon_t = r_t$.
  Zero,
  /// $\varepsilon_t = r_t - \mu$ with $\mu$ estimated jointly.
  Constant,
}

/// Model to fit: recursion, ARCH order `p`, GARCH order `q` and mean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GarchSpec {
  /// Variance recursion.
  pub kind: GarchKind,
  /// Number of lagged shocks ($\alpha_i$, and $\gamma_i$ where the kind has them).
  pub p: usize,
  /// Number of lagged variances ($\beta_j$).
  pub q: usize,
  /// Conditional mean.
  pub mean: MeanSpec,
}

impl GarchSpec {
  /// GARCH($p$, $q$) with a constant mean.
  pub fn garch(p: usize, q: usize) -> Self {
    Self {
      kind: GarchKind::Garch,
      p,
      q,
      mean: MeanSpec::Constant,
    }
  }

  /// GJR-GARCH($p$, $q$) with a constant mean.
  pub fn gjr(p: usize, q: usize) -> Self {
    Self {
      kind: GarchKind::GjrGarch,
      p,
      q,
      mean: MeanSpec::Constant,
    }
  }

  /// EGARCH($p$, $q$) with a constant mean, fitted on $\alpha_i \ge |\gamma_i|$,
  /// $\beta_1 \ge \dots \ge \beta_q \ge 0$, $\sum\beta_j < 1$: for (1,1), the invertible region.
  pub fn egarch(p: usize, q: usize) -> Self {
    Self {
      kind: GarchKind::Egarch,
      p,
      q,
      mean: MeanSpec::Constant,
    }
  }

  /// The same model with another mean specification.
  pub fn with_mean(mut self, mean: MeanSpec) -> Self {
    self.mean = mean;
    self
  }

  /// Whether the recursion carries the asymmetry coefficients $\gamma_i$.
  pub fn has_gamma(&self) -> bool {
    matches!(self.kind, GarchKind::GjrGarch | GarchKind::Egarch)
  }

  /// Number of estimated parameters.
  pub fn n_params(&self) -> usize {
    let mean = usize::from(self.mean == MeanSpec::Constant);
    let gamma = if self.has_gamma() { self.p } else { 0 };
    mean + 1 + self.p + gamma + self.q
  }

  /// Parameter names in the flat layout `[mu, omega, alpha[1..p], gamma[1..p], beta[1..q]]`, with
  /// `mu` and `gamma` only where the model has them.
  pub fn param_names(&self) -> Vec<String> {
    let mut names = Vec::with_capacity(self.n_params());
    if self.mean == MeanSpec::Constant {
      names.push("mu".to_string());
    }
    names.push("omega".to_string());
    names.extend((1..=self.p).map(|i| format!("alpha[{i}]")));
    if self.has_gamma() {
      names.extend((1..=self.p).map(|i| format!("gamma[{i}]")));
    }
    names.extend((1..=self.q).map(|j| format!("beta[{j}]")));
    names
  }

  fn validate(&self) {
    assert!(self.p >= 1, "p must satisfy `p >= 1`, got p = {:?}", self.p);
  }
}

/// A fitted GARCH-family model.
#[derive(Debug, Clone)]
pub struct GarchFit {
  /// The specification that was fitted.
  pub spec: GarchSpec,
  /// Conditional mean $\hat\mu$ (zero under [`MeanSpec::Zero`]).
  pub mu: f64,
  /// $\hat\omega$.
  pub omega: f64,
  /// $\hat\alpha_1, \ldots, \hat\alpha_p$.
  pub alpha: Array1<f64>,
  /// $\hat\gamma_1, \ldots, \hat\gamma_p$; empty for plain GARCH.
  pub gamma: Array1<f64>,
  /// $\hat\beta_1, \ldots, \hat\beta_q$.
  pub beta: Array1<f64>,
  /// All parameters in the [`GarchSpec::param_names`] order.
  pub params: Array1<f64>,
  /// Inverse-Hessian standard errors, aligned with `params`.
  pub std_errors: Array1<f64>,
  /// Bollerslev–Wooldridge robust standard errors, aligned with `params`.
  pub robust_std_errors: Array1<f64>,
  /// Inverse-Hessian covariance $(-H)^{-1}$, `±∞` past the double range.
  pub covariance: Array2<f64>,
  /// Sandwich covariance $H^{-1} J H^{-1}$, `±∞` past the double range.
  pub robust_covariance: Array2<f64>,
  /// Fitted conditional variances $\hat\sigma_t^2$.
  pub conditional_variance: Array1<f64>,
  /// Demeaned returns $\hat\varepsilon_t$.
  pub residuals: Array1<f64>,
  /// $\hat\varepsilon_t / \hat\sigma_t$.
  pub standardized_residuals: Array1<f64>,
  /// Maximised Gaussian quasi-log-likelihood.
  pub log_likelihood: f64,
  /// $2k - 2\ell$.
  pub aic: f64,
  /// $k\log n - 2\ell$.
  pub bic: f64,
  /// $\sum\alpha_i + \sum\beta_j$ (GARCH), $\sum\alpha_i + \tfrac12\sum\gamma_i +
  /// \sum\beta_j$ (GJR) or $\sum\beta_j$ (EGARCH).
  pub persistence: f64,
  /// Backcast variance seeding the recursion.
  pub backcast: f64,
  /// Iterations of the simplex and of every L-BFGS polish and refit at a limit.
  pub iterations: usize,
  /// No step along one parameter gains `1e-7` in log-likelihood, and no limit of the family that
  /// the quadratic model puts within 36 standard errors fits as well: stationary, not global.
  pub converged: bool,
  /// `converged`, or why not: `not stationary`, `stopped at the bound of <name>`,
  /// `<name> is not identified`, `iteration cap reached`, …
  pub message: String,
  /// Number of observations.
  pub nobs: usize,
}

/// Gaussian QMLE of `spec` at unit scale (NaN errors at a singular Hessian); `Err` below 20 points,
/// a non-finite one, constant returns or a fit past the double range; panics unless `spec.p >= 1`.
pub fn fit_garch<T: FloatExt>(
  returns: ArrayView1<T>,
  spec: GarchSpec,
) -> Result<GarchFit, StatsError> {
  spec.validate();
  let r = returns
    .iter()
    .map(|x| x.to_f64().unwrap())
    .collect::<Vec<f64>>();
  validate(&r, 20)?;
  reject_constant(&r, "zero-variance returns")?;
  let (r, unit) = unit_scaled(r);
  let n = r.len();
  let mean = r.iter().sum::<f64>() / n as f64;
  let backcast = r.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;

  let problem = Problem {
    spec,
    returns: r.clone(),
    backcast,
  };
  let starts = transform::starts(&spec, mean, backcast);
  let groups = starts.iter().map(Vec::as_slice).collect::<Vec<_>>();
  let (coordinates, words) = transform::coordinates(&spec, mean, backcast);
  let objective = |theta: &[f64]| problem.objective(theta);
  let found = minimise(objective, by_differences, &groups, &coordinates, n);
  let theta = &found.at;

  let fitted = transform::to_natural(&spec, theta);
  let (log_likelihood, _) = recursion::log_likelihood_terms(&spec, &fitted, &r, backcast);
  if !(fitted.iter().all(|v| v.is_finite()) && log_likelihood.is_finite()) {
    return Err(StatsError::Numerical { stage: "estimate" });
  }
  let mut residuals = vec![0.0; n];
  let mut sigma2 = vec![0.0; n];
  recursion::variance_path(&spec, &fitted, &r, backcast, &mut residuals, &mut sigma2);
  let standardized: Vec<f64> = residuals
    .iter()
    .zip(&sigma2)
    .map(|(e, v)| e / v.sqrt())
    .collect();
  let scales = transform::natural_scales(&spec, backcast);
  let inference = inference::sandwich(&spec, &fitted, &r, backcast, &scales);

  let natural = units::params(&spec, &fitted, unit)?;
  let variance = |v: f64| v / unit / unit;
  let backcast = variance(backcast);
  let sigma2 = sigma2.into_iter().map(variance).collect::<Vec<f64>>();
  let residuals = residuals
    .into_iter()
    .map(|e| e / unit)
    .collect::<Vec<f64>>();
  let lost = !(backcast > 0.0 && backcast.is_finite())
    || !sigma2.iter().all(|&v| v > 0.0 && v.is_finite())
    || !residuals.iter().all(|v| v.is_finite());
  if lost {
    return Err(StatsError::Numerical {
      stage: "data scale",
    });
  }
  let log_likelihood = log_likelihood + n as f64 * unit.ln();
  let (covariance, std_errors) = units::covariance(&spec, &inference.covariance, unit);
  let (robust_covariance, robust_std_errors) =
    units::covariance(&spec, &inference.robust_covariance, unit);
  let k = natural.len();

  let split = spec.split(&natural);
  let alpha = Array1::from(split.alpha.to_vec());
  let gamma = Array1::from(split.gamma.to_vec());
  let beta = Array1::from(split.beta.to_vec());
  let persistence = match spec.kind {
    GarchKind::Garch => alpha.sum() + beta.sum(),
    GarchKind::GjrGarch => alpha.sum() + 0.5 * gamma.sum() + beta.sum(),
    GarchKind::Egarch => beta.sum(),
  };
  let kf = k as f64;
  Ok(GarchFit {
    spec,
    mu: split.mu,
    omega: split.omega,
    alpha,
    gamma,
    beta,
    params: Array1::from(natural),
    std_errors,
    robust_std_errors,
    covariance,
    robust_covariance,
    conditional_variance: Array1::from(sigma2),
    residuals: Array1::from(residuals),
    standardized_residuals: Array1::from(standardized),
    log_likelihood,
    aic: 2.0 * kf - 2.0 * log_likelihood,
    bic: kf * (n as f64).ln() - 2.0 * log_likelihood,
    persistence,
    backcast,
    iterations: found.iterations,
    converged: found.stop.converged(),
    message: found
      .stop
      .message(&words.iter().map(String::as_str).collect::<Vec<_>>()),
    nobs: n,
  })
}

/// The negative quasi-log-likelihood in unconstrained coordinates.
struct Problem {
  spec: GarchSpec,
  returns: Vec<f64>,
  backcast: f64,
}

impl Problem {
  fn objective(&self, theta: &[f64]) -> f64 {
    let natural = transform::to_natural(&self.spec, theta);
    let ll = recursion::total_log_likelihood(&self.spec, &natural, &self.returns, self.backcast);
    if ll.is_finite() { -ll } else { NO_VALUE }
  }
}

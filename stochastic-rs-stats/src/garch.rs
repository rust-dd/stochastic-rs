//! # GARCH-family quasi-maximum likelihood
//!
//! Fits Bollerslev's GARCH, the Glosten–Jagannathan–Runkle GJR-GARCH and
//! Nelson's EGARCH to a return series by maximising the Gaussian
//! quasi-log-likelihood of the demeaned returns $\varepsilon_t = r_t - \mu$,
//!
//! $$
//! \ell(\theta) = -\frac12\sum_{t=1}^{n}\Bigl[\log 2\pi + \log\sigma_t^2 + \frac{\varepsilon_t^2}{\sigma_t^2}\Bigr],
//! $$
//!
//! under the conditional-variance recursions
//!
//! $$
//! \begin{aligned}
//! \text{GARCH}(p,q):&\quad \sigma_t^2 = \omega + \sum_{i=1}^{p}\alpha_i\varepsilon_{t-i}^2 + \sum_{j=1}^{q}\beta_j\sigma_{t-j}^2,\\
//! \text{GJR}(p,q):&\quad \sigma_t^2 = \omega + \sum_{i=1}^{p}\bigl(\alpha_i + \gamma_i\mathbf 1_{\{\varepsilon_{t-i}<0\}}\bigr)\varepsilon_{t-i}^2 + \sum_{j=1}^{q}\beta_j\sigma_{t-j}^2,\\
//! \text{EGARCH}(p,q):&\quad \log\sigma_t^2 = \omega + \sum_{i=1}^{p}\bigl[\alpha_i(|z_{t-i}| - \sqrt{2/\pi}) + \gamma_i z_{t-i}\bigr] + \sum_{j=1}^{q}\beta_j\log\sigma_{t-j}^2,
//! \end{aligned}
//! $$
//!
//! with $z_t = \varepsilon_t/\sigma_t$. Pre-sample terms are the backcast
//! $\bar\sigma^2 = n^{-1}\sum_t(r_t - \bar r)^2$: every $\varepsilon_{t-i}^2$
//! and $\sigma_{t-j}^2$ with a non-positive index is $\bar\sigma^2$, the GJR
//! asymmetry term gets $\tfrac12\bar\sigma^2$, and EGARCH drops its shock
//! terms and uses $\log\bar\sigma^2$ — the conventions of the `arch`
//! reference implementation, whose fits the tests reproduce.
//!
//! The optimiser works in unconstrained coordinates that map onto the
//! stationarity region (see the `transform` module), so a fit never leaves
//! $\omega > 0$, $\alpha_i \ge 0$, $\alpha_i + \gamma_i \ge 0$, $\beta_j \ge 0$ and
//! a persistence below one. Standard errors come in two flavours: the
//! inverse Hessian $(-H)^{-1}$, exact only under Gaussian innovations, and the
//! Bollerslev–Wooldridge sandwich $H^{-1} J H^{-1}$ with $J = \sum_t s_t s_t'$
//! the outer product of the per-observation scores, which stays valid under
//! non-Gaussian innovations — the "quasi" in QMLE. Both derivatives are
//! central finite differences at the optimum in the natural parameters.
//!
//! Reference: Bollerslev, "Generalized Autoregressive Conditional
//! Heteroskedasticity", Journal of Econometrics, 31(3), 307-327 (1986).
//! DOI: 10.1016/0304-4076(86)90063-1
//!
//! Reference: Nelson, "Conditional Heteroskedasticity in Asset Returns: A
//! New Approach", Econometrica, 59(2), 347-370 (1991). DOI: 10.2307/2938260
//!
//! Reference: Glosten, Jagannathan, Runkle, "On the Relation between the
//! Expected Value and the Volatility of the Nominal Excess Return on Stocks",
//! Journal of Finance, 48(5), 1779-1801 (1993).
//! DOI: 10.1111/j.1540-6261.1993.tb05128.x
//!
//! Reference: Bollerslev, Wooldridge, "Quasi-Maximum Likelihood Estimation
//! and Inference in Dynamic Models with Time-Varying Covariances",
//! Econometric Reviews, 11(2), 143-172 (1992). DOI: 10.1080/07474939208800229

mod inference;
mod recursion;
#[cfg(test)]
mod tests;
mod transform;

use argmin::core::CostFunction;
use argmin::core::Executor;
use argmin::core::Gradient;
use argmin::core::State;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;

use crate::optim::nelder_mead_vec;
use crate::traits::FloatExt;

/// Conditional-variance recursion to fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

  /// EGARCH($p$, $q$) with a constant mean.
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

  /// Parameter names in the flat layout
  /// `[mu, omega, alpha[1..p], gamma[1..p], beta[1..q]]` (`mu` and `gamma`
  /// only where the model has them).
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
    assert!(self.p >= 1, "p must be at least 1, got {}", self.p);
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
  /// Inverse-Hessian covariance $(-H)^{-1}$.
  pub covariance: Array2<f64>,
  /// Sandwich covariance $H^{-1} J H^{-1}$.
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
  /// Simplex iterations plus L-BFGS polishing iterations.
  pub iterations: usize,
  /// Whether the simplex met its tolerance.
  pub converged: bool,
  /// Number of observations.
  pub nobs: usize,
}

/// Fits `spec` to `returns` by Gaussian QMLE.
///
/// # Panics
///
/// If `spec.p` is zero, if there are fewer than 20 observations, or if the
/// returns have no positive finite variance.
pub fn garch_fit<T: FloatExt>(returns: ArrayView1<T>, spec: GarchSpec) -> GarchFit {
  spec.validate();
  let r: Vec<f64> = returns
    .iter()
    .map(|x| x.to_f64().unwrap_or(f64::NAN))
    .collect();
  let n = r.len();
  assert!(n >= 20, "need at least 20 observations, got {n}");
  let mean = r.iter().sum::<f64>() / n as f64;
  let backcast = r.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
  assert!(
    backcast > 0.0 && backcast.is_finite(),
    "returns must have a positive, finite variance"
  );

  let problem = Problem {
    spec,
    returns: r.clone(),
    backcast,
  };
  let start = transform::best_start(&spec, mean, backcast, |theta| problem.objective(theta));
  let (theta_nm, iters_nm, converged) =
    nelder_mead_vec(&start, 5_000, |theta| problem.objective(theta));
  let (theta_nm, iters_restart, converged_restart) =
    nelder_mead_vec(&theta_nm, 5_000, |theta| problem.objective(theta));
  let (theta, iters_lbfgs) = problem.polish(theta_nm);

  let natural = transform::to_natural(&spec, &theta);
  let (log_likelihood, _) = recursion::log_likelihood_terms(&spec, &natural, &r, backcast);
  let mut residuals = vec![0.0; n];
  let mut sigma2 = vec![0.0; n];
  recursion::variance_path(&spec, &natural, &r, backcast, &mut residuals, &mut sigma2);
  let standardized: Vec<f64> = residuals
    .iter()
    .zip(&sigma2)
    .map(|(e, v)| e / v.sqrt())
    .collect();

  let scales = transform::natural_scales(&spec, backcast);
  let inference = inference::sandwich(&spec, &natural, &r, backcast, &scales);
  let k = natural.len();
  let std_errors = Array1::from_iter((0..k).map(|i| inference.covariance[[i, i]].sqrt()));
  let robust_std_errors =
    Array1::from_iter((0..k).map(|i| inference.robust_covariance[[i, i]].sqrt()));

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
  GarchFit {
    spec,
    mu: split.mu,
    omega: split.omega,
    alpha,
    gamma,
    beta,
    params: Array1::from(natural),
    std_errors,
    robust_std_errors,
    covariance: inference.covariance,
    robust_covariance: inference.robust_covariance,
    conditional_variance: Array1::from(sigma2),
    residuals: Array1::from(residuals),
    standardized_residuals: Array1::from(standardized),
    log_likelihood,
    aic: 2.0 * kf - 2.0 * log_likelihood,
    bic: kf * (n as f64).ln() - 2.0 * log_likelihood,
    persistence,
    backcast,
    iterations: iters_nm + iters_restart + iters_lbfgs,
    converged: converged || converged_restart,
    nobs: n,
  }
}

/// The negative quasi-log-likelihood in unconstrained coordinates, shared by
/// the simplex and the L-BFGS polish.
#[derive(Clone)]
struct Problem {
  spec: GarchSpec,
  returns: Vec<f64>,
  backcast: f64,
}

impl Problem {
  fn objective(&self, theta: &[f64]) -> f64 {
    let natural = transform::to_natural(&self.spec, theta);
    let ll = recursion::total_log_likelihood(&self.spec, &natural, &self.returns, self.backcast);
    if ll.is_finite() { -ll } else { 1e300 }
  }

  /// L-BFGS from the simplex optimum; keeps the simplex point when the
  /// polish fails or does not improve the objective.
  fn polish(&self, theta: Vec<f64>) -> (Vec<f64>, usize) {
    let start_cost = self.objective(&theta);
    let linesearch = MoreThuenteLineSearch::new();
    let solver = LBFGS::new(linesearch, 10);
    let result = Executor::new(self.clone(), solver)
      .configure(|state| state.param(theta.clone()).max_iters(200))
      .run();
    match result {
      Ok(res) => {
        let iters = res.state.get_iter() as usize;
        match res.state.get_best_param() {
          Some(best) if self.objective(best) < start_cost => (best.clone(), iters),
          _ => (theta, iters),
        }
      }
      Err(_) => (theta, 0),
    }
  }
}

impl CostFunction for Problem {
  type Param = Vec<f64>;
  type Output = f64;

  fn cost(&self, theta: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
    Ok(self.objective(theta))
  }
}

impl Gradient for Problem {
  type Param = Vec<f64>;
  type Gradient = Vec<f64>;

  fn gradient(&self, theta: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
    let mut grad = vec![0.0; theta.len()];
    for i in 0..theta.len() {
      let step = 1e-6 * (1.0 + theta[i].abs());
      let mut plus = theta.clone();
      let mut minus = theta.clone();
      plus[i] += step;
      minus[i] -= step;
      grad[i] = (self.objective(&plus) - self.objective(&minus)) / (2.0 * step);
    }
    Ok(grad)
  }
}

//! # Surrogate calibration
//!
//! $$
//! \hat\theta = \arg\min_{\theta}\ \sum_{k}
//!   w_k^2\bigl(\tilde F(\theta)_k - \sigma^{\text{mkt}}_k\bigr)^2
//! $$
//!
//! Second step of the two-step deep-learning-volatility approach of Horvath,
//! Muguruza & Tomas (2021): a trained network $\tilde F$ replaces the pricer
//! on its fixed strike–maturity grid, and the calibration is a deterministic
//! least-squares problem on the surrogate, solved with Levenberg–Marquardt on
//! the network's exact Jacobian — reverse-mode differentiation through the
//! layers (their §3.2; the reference notebooks pass the analytic network
//! gradient to `least_squares(method="lm")`). The optimisation runs in the
//! network's scaled coordinates $x_j = (\theta_j - c_j)/h_j$ (centre and
//! half-range of the training box), and the result records whether
//! $\hat\theta$ stayed inside the box: the surrogate is untrained outside it,
//! so a boundary solution is a warning rather than an answer.
//!
//! Reference: Horvath, B., Muguruza, A. & Tomas, M. (2021), *Deep learning
//! volatility: a deep neural network perspective on pricing and calibration in
//! (rough) volatility models*, Quantitative Finance 21(1), 11–27.

use std::cell::RefCell;

use anyhow::Result;
use anyhow::bail;
use levenberg_marquardt::LeastSquaresProblem;
use levenberg_marquardt::LevenbergMarquardt;
use nalgebra::DMatrix;
use nalgebra::DVector;
use nalgebra::Dyn;
use nalgebra::Owned;
use ndarray::Array2;
use stochastic_rs_quant::calibration::heston::HestonParams;
use stochastic_rs_quant::calibration::rbergomi::RBergomiParams;
use stochastic_rs_quant::calibration::rbergomi::RBergomiXi0;
use stochastic_rs_quant::pricing::fourier::HestonFourier;
use stochastic_rs_quant::pricing::rbergomi::RBergomiPricer;
use stochastic_rs_quant::traits::CalibrationResult;
use stochastic_rs_quant::traits::Calibrator;
use stochastic_rs_quant::traits::ToModel;

use crate::volatility::common::StochVolNn;
use crate::volatility::heston::HestonNn;
use crate::volatility::rbergomi::RBergomiNn;

/// A trained surrogate the calibrator can differentiate: anything that wraps
/// a [`StochVolNn`].
pub trait SurrogateModel {
  /// The network behind the model.
  fn nn(&self) -> &StochVolNn;
}

impl SurrogateModel for StochVolNn {
  fn nn(&self) -> &StochVolNn {
    self
  }
}

/// Result of a surrogate calibration in the network's own parameter order.
#[derive(Clone, Debug, PartialEq)]
pub struct SurrogateCalibrationResult {
  /// Calibrated parameters, in the surrogate's input order and units.
  pub params: Vec<f64>,
  /// Unweighted root-mean-square implied-volatility error on the grid.
  pub rmse: f64,
  /// Largest absolute implied-volatility error on the grid.
  pub max_error: f64,
  /// Whether Levenberg–Marquardt reported a successful termination.
  pub converged: bool,
  /// Whether every parameter lies inside the training box.
  pub in_bounds: bool,
  /// Residual evaluations the optimiser consumed.
  pub evaluations: usize,
  /// The optimiser's termination reason.
  pub message: String,
}

impl CalibrationResult for SurrogateCalibrationResult {
  type Params = Vec<f64>;
  fn rmse(&self) -> f64 {
    self.rmse
  }
  fn converged(&self) -> bool {
    self.converged
  }
  fn params(&self) -> Vec<f64> {
    self.params.clone()
  }
  fn iterations(&self) -> Option<usize> {
    Some(self.evaluations)
  }
  fn message(&self) -> Option<&str> {
    Some(&self.message)
  }
  fn max_error(&self) -> f64 {
    self.max_error
  }
}

/// Levenberg–Marquardt calibration of a surrogate to a market surface given
/// on the surrogate's grid (flat, row-major `maturities × strikes`, in
/// implied volatility).
pub struct SurrogateCalibrator<'m, M: SurrogateModel> {
  model: &'m M,
  market: Vec<f64>,
  weights: Vec<f64>,
  initial: Option<Vec<f64>>,
  tolerance: f64,
  patience: usize,
}

impl<'m, M: SurrogateModel> SurrogateCalibrator<'m, M> {
  /// `market` must have the surrogate's `output_dim` entries and be finite.
  pub fn new(model: &'m M, market: Vec<f64>) -> Result<Self> {
    let dim = model.nn().spec().output_dim;
    if market.len() != dim {
      bail!(
        "market surface has {} entries, the surrogate grid has {dim}",
        market.len()
      );
    }
    if market.iter().any(|v| !v.is_finite()) {
      bail!("market surface contains a non-finite value");
    }
    Ok(Self {
      model,
      market,
      weights: vec![1.0; dim],
      initial: None,
      tolerance: 1e-10,
      patience: 200,
    })
  }

  /// Per-grid-point residual weights `w_k` (one per market entry).
  pub fn with_weights(mut self, weights: Vec<f64>) -> Result<Self> {
    if weights.len() != self.market.len() {
      bail!(
        "{} weights for {} grid points",
        weights.len(),
        self.market.len()
      );
    }
    if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
      bail!("weights must be finite and non-negative");
    }
    self.weights = weights;
    Ok(self)
  }

  /// Starting point in parameter units; the box centre otherwise.
  pub fn with_initial(mut self, initial: Vec<f64>) -> Result<Self> {
    if initial.len() != self.model.nn().spec().input_dim {
      bail!(
        "{} initial values for {} parameters",
        initial.len(),
        self.model.nn().spec().input_dim
      );
    }
    self.initial = Some(initial);
    Ok(self)
  }

  /// Levenberg–Marquardt `ftol`/`xtol`/`gtol`, all set to `tolerance`.
  pub fn with_tolerance(mut self, tolerance: f64) -> Self {
    self.tolerance = tolerance;
    self
  }

  /// Maximum number of residual evaluations.
  pub fn with_patience(mut self, patience: usize) -> Self {
    self.patience = patience;
    self
  }

  fn bounds(&self) -> (Vec<f64>, Vec<f64>) {
    let spec = self.model.nn().spec();
    (
      spec.param_lb.iter().map(|&v| v as f64).collect(),
      spec.param_ub.iter().map(|&v| v as f64).collect(),
    )
  }

  /// Runs the calibration from `initial` (parameter units), or from the
  /// builder's starting point, or from the box centre.
  pub fn run(&self, initial: Option<Vec<f64>>) -> Result<SurrogateCalibrationResult> {
    let (lb, ub) = self.bounds();
    let n = lb.len();
    let start = initial.or_else(|| self.initial.clone());
    if let Some(s) = &start
      && s.len() != n
    {
      bail!("{} initial values for {n} parameters", s.len());
    }
    let x0 = DVector::from_iterator(
      n,
      (0..n).map(|j| match &start {
        Some(s) => ((s[j] - 0.5 * (lb[j] + ub[j])) / (0.5 * (ub[j] - lb[j]))).clamp(-1.0, 1.0),
        None => 0.0,
      }),
    );
    let problem = Problem {
      model: self.model,
      market: &self.market,
      weights: &self.weights,
      lb,
      ub,
      x: x0,
      cache: RefCell::new(None),
    };
    let (problem, report) = LevenbergMarquardt::new()
      .with_tol(self.tolerance)
      .with_patience(self.patience)
      .minimize(problem);
    let (surface, _) = problem
      .evaluate()
      .ok_or_else(|| anyhow::anyhow!("the surrogate could not be evaluated at the solution"))?;
    let errors: Vec<f64> = surface
      .iter()
      .zip(&self.market)
      .map(|(&f, &m)| f as f64 - m)
      .collect();
    let rmse = (errors.iter().map(|e| e * e).sum::<f64>() / errors.len() as f64).sqrt();
    let max_error = errors.iter().fold(0.0_f64, |acc, e| acc.max(e.abs()));
    let in_bounds = problem.x.iter().all(|x| x.abs() <= 1.0 + 1e-9);
    Ok(SurrogateCalibrationResult {
      params: problem.theta().iter().map(|&t| t as f64).collect(),
      rmse,
      max_error,
      converged: report.termination.was_successful(),
      in_bounds,
      evaluations: report.number_of_evaluations,
      message: format!("{:?}", report.termination),
    })
  }
}

impl<M: SurrogateModel> Calibrator for SurrogateCalibrator<'_, M> {
  type InitialGuess = Vec<f64>;
  type Params = Vec<f64>;
  type Output = SurrogateCalibrationResult;
  type Error = anyhow::Error;

  fn calibrate(&self, initial: Option<Vec<f64>>) -> Result<SurrogateCalibrationResult> {
    self.run(initial)
  }
}

/// The least-squares problem in scaled coordinates; the network evaluation is
/// cached per parameter vector because the optimiser asks for residuals and
/// Jacobian separately.
struct Problem<'m, M: SurrogateModel> {
  model: &'m M,
  market: &'m [f64],
  weights: &'m [f64],
  lb: Vec<f64>,
  ub: Vec<f64>,
  x: DVector<f64>,
  cache: RefCell<Option<(Vec<f32>, Array2<f32>)>>,
}

impl<M: SurrogateModel> Problem<'_, M> {
  fn theta(&self) -> Vec<f32> {
    (0..self.lb.len())
      .map(|j| {
        (0.5 * (self.lb[j] + self.ub[j]) + 0.5 * (self.ub[j] - self.lb[j]) * self.x[j]) as f32
      })
      .collect()
  }

  fn evaluate(&self) -> Option<(Vec<f32>, Array2<f32>)> {
    if let Some(cached) = self.cache.borrow().as_ref() {
      return Some(cached.clone());
    }
    let value = self
      .model
      .nn()
      .predict_surface_with_jacobian(&self.theta())
      .ok()?;
    if value.0.iter().any(|v| !v.is_finite()) || value.1.iter().any(|v| !v.is_finite()) {
      return None;
    }
    *self.cache.borrow_mut() = Some(value.clone());
    Some(value)
  }
}

impl<M: SurrogateModel> LeastSquaresProblem<f64, Dyn, Dyn> for Problem<'_, M> {
  type ResidualStorage = Owned<f64, Dyn>;
  type JacobianStorage = Owned<f64, Dyn, Dyn>;
  type ParameterStorage = Owned<f64, Dyn>;

  fn set_params(&mut self, x: &DVector<f64>) {
    self.x = x.clone();
    *self.cache.borrow_mut() = None;
  }

  fn params(&self) -> DVector<f64> {
    self.x.clone()
  }

  fn residuals(&self) -> Option<DVector<f64>> {
    let (surface, _) = self.evaluate()?;
    Some(DVector::from_iterator(
      surface.len(),
      surface
        .iter()
        .zip(self.market)
        .zip(self.weights)
        .map(|((&f, &m), &w)| w * (f as f64 - m)),
    ))
  }

  fn jacobian(&self) -> Option<DMatrix<f64>> {
    let (_, jacobian) = self.evaluate()?;
    let (rows, cols) = jacobian.dim();
    Some(DMatrix::from_fn(rows, cols, |k, j| {
      self.weights[k] * jacobian[(k, j)] as f64 * 0.5 * (self.ub[j] - self.lb[j])
    }))
  }
}

/// The Heston surrogate's input order `[v0, ρ, σ, θ, κ]` as quant's
/// [`HestonParams`].
pub fn heston_params_from_surrogate(v: &[f64]) -> HestonParams {
  HestonParams {
    v0: v[0],
    kappa: v[4],
    theta: v[3],
    sigma: v[2],
    rho: v[1],
  }
}

/// Quant's [`HestonParams`] in the Heston surrogate's input order.
pub fn heston_params_to_surrogate(p: &HestonParams) -> Vec<f64> {
  vec![p.v0, p.rho, p.sigma, p.theta, p.kappa]
}

/// The rough Bergomi surrogate's input order `[ξ₀, η, ρ, H]` (read off the
/// training box the same way as the Heston order: `[0.01, 0.16]` is a
/// forward variance, `[0.3, 4]` a vol-of-vol, `[−0.95, −0.1]` a correlation
/// and `[0.025, 0.5]` a Hurst exponent) as quant's [`RBergomiParams`] with a
/// flat forward-variance curve.
pub fn rbergomi_params_from_surrogate(v: &[f64]) -> RBergomiParams {
  RBergomiParams {
    hurst: v[3],
    rho: v[2],
    eta: v[1],
    xi0: RBergomiXi0::Constant(v[0]),
  }
}

/// Quant's [`RBergomiParams`] in the rough Bergomi surrogate's input order;
/// a non-constant forward-variance curve is represented by its value at
/// `t = 0`, the only shape the flat-ξ₀ surrogate knows.
pub fn rbergomi_params_to_surrogate(p: &RBergomiParams) -> Vec<f64> {
  let xi0 = match &p.xi0 {
    RBergomiXi0::Constant(v) => *v,
    other => other.value(0.0),
  };
  vec![xi0, p.eta, p.rho, p.hurst]
}

/// Heston calibration on the [`HestonNn`] surrogate, producing
/// [`HestonParams`] and a [`HestonFourier`] pricer.
pub struct HestonSurrogateCalibrator<'m> {
  inner: SurrogateCalibrator<'m, HestonNn>,
}

/// [`HestonSurrogateCalibrator`]'s output.
#[derive(Clone, Debug, PartialEq)]
pub struct HestonSurrogateResult {
  pub params: HestonParams,
  pub fit: SurrogateCalibrationResult,
}

impl<'m> HestonSurrogateCalibrator<'m> {
  pub fn new(model: &'m HestonNn, market: Vec<f64>) -> Result<Self> {
    Ok(Self {
      inner: SurrogateCalibrator::new(model, market)?,
    })
  }

  pub fn with_weights(mut self, weights: Vec<f64>) -> Result<Self> {
    self.inner = self.inner.with_weights(weights)?;
    Ok(self)
  }

  pub fn with_tolerance(mut self, tolerance: f64) -> Self {
    self.inner = self.inner.with_tolerance(tolerance);
    self
  }

  pub fn with_patience(mut self, patience: usize) -> Self {
    self.inner = self.inner.with_patience(patience);
    self
  }
}

impl CalibrationResult for HestonSurrogateResult {
  type Params = HestonParams;
  fn rmse(&self) -> f64 {
    self.fit.rmse
  }
  fn converged(&self) -> bool {
    self.fit.converged
  }
  fn params(&self) -> HestonParams {
    self.params.clone()
  }
  fn iterations(&self) -> Option<usize> {
    Some(self.fit.evaluations)
  }
  fn message(&self) -> Option<&str> {
    Some(&self.fit.message)
  }
  fn max_error(&self) -> f64 {
    self.fit.max_error
  }
}

impl ToModel for HestonSurrogateResult {
  type Model = HestonFourier;
  fn to_model(&self, r: f64, q: f64) -> HestonFourier {
    self.params.to_model(r, q)
  }
}

impl Calibrator for HestonSurrogateCalibrator<'_> {
  type InitialGuess = HestonParams;
  type Params = HestonParams;
  type Output = HestonSurrogateResult;
  type Error = anyhow::Error;

  fn calibrate(&self, initial: Option<HestonParams>) -> Result<HestonSurrogateResult> {
    let fit = self
      .inner
      .run(initial.as_ref().map(heston_params_to_surrogate))?;
    Ok(HestonSurrogateResult {
      params: heston_params_from_surrogate(&fit.params),
      fit,
    })
  }
}

/// Rough Bergomi calibration on the [`RBergomiNn`] surrogate, producing
/// [`RBergomiParams`] and an [`RBergomiPricer`].
pub struct RBergomiSurrogateCalibrator<'m> {
  inner: SurrogateCalibrator<'m, RBergomiNn>,
}

/// [`RBergomiSurrogateCalibrator`]'s output.
#[derive(Clone, Debug)]
pub struct RBergomiSurrogateResult {
  pub params: RBergomiParams,
  pub fit: SurrogateCalibrationResult,
}

impl<'m> RBergomiSurrogateCalibrator<'m> {
  pub fn new(model: &'m RBergomiNn, market: Vec<f64>) -> Result<Self> {
    Ok(Self {
      inner: SurrogateCalibrator::new(model, market)?,
    })
  }

  pub fn with_weights(mut self, weights: Vec<f64>) -> Result<Self> {
    self.inner = self.inner.with_weights(weights)?;
    Ok(self)
  }

  pub fn with_tolerance(mut self, tolerance: f64) -> Self {
    self.inner = self.inner.with_tolerance(tolerance);
    self
  }

  pub fn with_patience(mut self, patience: usize) -> Self {
    self.inner = self.inner.with_patience(patience);
    self
  }
}

impl CalibrationResult for RBergomiSurrogateResult {
  type Params = RBergomiParams;
  fn rmse(&self) -> f64 {
    self.fit.rmse
  }
  fn converged(&self) -> bool {
    self.fit.converged
  }
  fn params(&self) -> RBergomiParams {
    self.params.clone()
  }
  fn iterations(&self) -> Option<usize> {
    Some(self.fit.evaluations)
  }
  fn message(&self) -> Option<&str> {
    Some(&self.fit.message)
  }
  fn max_error(&self) -> f64 {
    self.fit.max_error
  }
}

impl ToModel for RBergomiSurrogateResult {
  type Model = RBergomiPricer;
  fn to_model(&self, _r: f64, _q: f64) -> RBergomiPricer {
    RBergomiPricer::new(self.params.clone())
  }
}

impl Calibrator for RBergomiSurrogateCalibrator<'_> {
  type InitialGuess = RBergomiParams;
  type Params = RBergomiParams;
  type Output = RBergomiSurrogateResult;
  type Error = anyhow::Error;

  fn calibrate(&self, initial: Option<RBergomiParams>) -> Result<RBergomiSurrogateResult> {
    let fit = self
      .inner
      .run(initial.as_ref().map(rbergomi_params_to_surrogate))?;
    Ok(RBergomiSurrogateResult {
      params: rbergomi_params_from_surrogate(&fit.params),
      fit,
    })
  }
}

#[cfg(test)]
mod tests;

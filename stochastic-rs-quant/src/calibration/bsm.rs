//! # Bsm
//!
//! $$
//! C=S_0e^{(b-r)T}N(d_1)-Ke^{-rT}N(d_2),\quad d_{1,2}=\frac{\ln(S_0/K)+(b\pm\tfrac12\sigma^2)T}{\sigma\sqrt T}
//! $$
//!
use std::cell::RefCell;

use levenberg_marquardt::LeastSquaresProblem;
use levenberg_marquardt::LevenbergMarquardt;
use nalgebra::DMatrix;
use nalgebra::DVector;
use nalgebra::Dyn;
use nalgebra::Owned;

use crate::CalibrationLossScore;
use crate::LossMetric;
use crate::OptionType;
use crate::calibration::CalibrationHistory;
use crate::pricing::bsm::BSMCoc;
use crate::pricing::bsm::BSMPricer;
use crate::traits::ModelPricer;

/// Lower edge of the projection box for `v`, matching
/// [`SabrCalibrator`](crate::calibration::sabr::SabrCalibrator)'s
/// `ALPHA_MIN`: the same kind of parameter under the same optimiser. No
/// upper edge, for the same reason SABR gives `alpha` none — a legitimate
/// fit to a distressed surface can sit anywhere above it.
const V_MIN: f64 = 1e-6;

/// Calibration result for the BSM model.
#[derive(Clone, Debug)]
pub struct BSMCalibrationResult {
  /// Calibrated implied volatility.
  pub v: f64,
  /// Calibration loss metrics.
  pub loss: CalibrationLossScore,
  /// Whether the optimiser converged.
  pub converged: bool,
}

impl BSMCalibrationResult {
  /// Convert to a [`BSMFourier`](crate::pricing::fourier::BSMFourier) model for pricing
  /// / vol surface generation.
  pub fn to_model(&self, r: f64, q: f64) -> crate::pricing::fourier::BSMFourier {
    crate::pricing::fourier::BSMFourier {
      sigma: self.v,
      r,
      q,
    }
  }
}

impl crate::traits::ToModel for BSMCalibrationResult {
  type Model = crate::pricing::fourier::BSMFourier;
  fn to_model(&self, r: f64, q: f64) -> Self::Model {
    BSMCalibrationResult::to_model(self, r, q)
  }
}

impl crate::traits::CalibrationResult for BSMCalibrationResult {
  type Params = BSMParams;
  fn rmse(&self) -> f64 {
    self.loss.get(LossMetric::Rmse)
  }

  fn converged(&self) -> bool {
    self.converged
  }

  fn params(&self) -> Self::Params {
    BSMParams { v: self.v }
  }

  fn loss_score(&self) -> Option<&CalibrationLossScore> {
    Some(&self.loss)
  }
}

impl crate::traits::Calibrator for BSMCalibrator {
  type InitialGuess = BSMParams;
  type Params = BSMParams;
  type Output = BSMCalibrationResult;
  type Error = anyhow::Error;

  fn calibrate(&self, initial: Option<Self::InitialGuess>) -> Result<Self::Output, Self::Error> {
    let mut this = self.clone();
    if let Some(p) = initial {
      this.set_initial_guess(p);
    }
    Ok(this.solve())
  }
}

#[derive(Clone, Debug)]
pub struct BSMParams {
  /// Implied volatility
  pub v: f64,
}

impl BSMParams {
  /// Project onto the admissible set $[\text{V\_MIN}, \infty)$, in the
  /// shape of
  /// [`SabrParams::project_in_place`](crate::calibration::sabr::SabrParams::project_in_place).
  ///
  /// A negative `v` does not announce itself: $d_1$ and $d_2$ both flip
  /// sign with $\sigma$ (the $\sigma^2/2$ in the numerator does not), so
  /// [`BSMPricer`] hands back a finite, put-shaped number for a call and
  /// the residual is built from it as if nothing happened.
  ///
  /// `abs()` rather than `max(V_MIN)` alone so the step keeps its
  /// magnitude — the optimiser that overshot to `-0.3` is told `0.3`, not
  /// `V_MIN` — which is what stops the box from stalling a live
  /// calibration on one bad step. `sigma_is_reflected_not_clamped` pins
  /// both halves.
  ///
  /// A `NaN` iterate lands on `V_MIN`, since `f64::max` returns the
  /// non-`NaN` operand. Intended, and not the crate's laundering shape:
  /// what is replaced is an optimiser *iterate*, not a computed price, and
  /// a projection onto a set has to be total.
  pub fn project_in_place(&mut self) {
    self.v = self.v.abs().max(V_MIN);
  }

  /// [`project_in_place`](Self::project_in_place), by value.
  pub fn projected(mut self) -> Self {
    self.project_in_place();
    self
  }
}

impl From<BSMParams> for DVector<f64> {
  fn from(params: BSMParams) -> Self {
    DVector::from_vec(vec![params.v])
  }
}

impl From<DVector<f64>> for BSMParams {
  fn from(params: DVector<f64>) -> Self {
    BSMParams { v: params[0] }
  }
}

#[derive(Clone)]
pub struct BSMCalibrator {
  /// Params to calibrate.
  pub params: BSMParams,
  /// Option prices from the market (flattened across all maturities).
  pub c_market: DVector<f64>,
  /// Underlying spot per quote.
  pub s: DVector<f64>,
  /// Strike per quote (flattened).
  pub k: DVector<f64>,
  /// Risk-free rate.
  pub r: f64,
  /// Domestic risk-free rate
  pub r_d: Option<f64>,
  /// Foreign risk-free rate
  pub r_f: Option<f64>,
  /// Dividend yield.
  pub q: Option<f64>,
  /// Time to maturity in years (kept for the legacy single-tau constructor).
  pub tau: f64,
  /// Time to maturity per quote (flattened). Supports multi-maturity
  /// joint calibration. Always populated — for the single-tau
  /// `BSMCalibrator::new` constructor every entry equals `tau`.
  pub flat_t: Vec<f64>,
  /// Option type
  pub option_type: OptionType,
  /// Which loss metrics to compute when recording history.
  pub loss_metrics: &'static [LossMetric],
  /// Levenberg-Marquardt algorithm residauls.
  calibration_history: RefCell<Vec<CalibrationHistory<BSMParams>>>,
}

impl BSMCalibrator {
  /// Create a calibrator for a single maturity slice (backwards compatible).
  pub fn new(
    params: BSMParams,
    c_market: DVector<f64>,
    s: DVector<f64>,
    k: DVector<f64>,
    r: f64,
    r_d: Option<f64>,
    r_f: Option<f64>,
    q: Option<f64>,
    tau: f64,
    option_type: OptionType,
  ) -> Self {
    let n = c_market.len();
    Self {
      params,
      c_market,
      s,
      k,
      r,
      r_d,
      r_f,
      q,
      tau,
      flat_t: vec![tau; n],
      option_type,
      loss_metrics: &LossMetric::ALL,
      calibration_history: RefCell::new(Vec::new()),
    }
  }

  /// Create a calibrator from multiple maturity slices for joint
  /// surface calibration. Mirrors the API of the Heston / SVJ
  /// calibrators so a single chain of `MarketSlice`s can be used to
  /// fit BSM, Heston and Bates side by side.
  pub fn from_slices(
    params: BSMParams,
    slices: &[super::levy::MarketSlice],
    s: f64,
    r: f64,
    r_d: Option<f64>,
    r_f: Option<f64>,
    q: Option<f64>,
    option_type: OptionType,
  ) -> Self {
    let mut flat_prices = Vec::new();
    let mut flat_strikes = Vec::new();
    let mut flat_t = Vec::new();
    let mut flat_s = Vec::new();

    for slice in slices {
      for i in 0..slice.strikes.len() {
        flat_prices.push(slice.prices[i]);
        flat_strikes.push(slice.strikes[i]);
        flat_t.push(slice.tau);
        flat_s.push(s);
      }
    }

    Self {
      params,
      c_market: DVector::from_vec(flat_prices),
      s: DVector::from_vec(flat_s),
      k: DVector::from_vec(flat_strikes),
      r,
      r_d,
      r_f,
      q,
      tau: 0.0,
      flat_t,
      option_type,
      loss_metrics: &LossMetric::ALL,
      calibration_history: RefCell::new(Vec::new()),
    }
  }
}

impl BSMCalibrator {
  fn solve(&self) -> BSMCalibrationResult {
    let (result, report) = LevenbergMarquardt::new().minimize(self.clone());
    let converged = report.termination.was_successful();
    let fitted = result.effective_params();
    let c_model: Vec<f64> = result
      .c_market
      .iter()
      .enumerate()
      .map(|(idx, _)| {
        BSMPricer::new(fitted.v, BSMCoc::Bsm1973).price_option(
          result.s[idx],
          result.k[idx],
          result.r,
          result.q.unwrap_or(0.0),
          result.flat_t[idx],
          result.option_type,
        )
      })
      .collect();
    let loss = CalibrationLossScore::compute_selected(
      result.c_market.as_slice(),
      &c_model,
      result.loss_metrics,
    );

    BSMCalibrationResult {
      v: fitted.v,
      loss,
      converged,
    }
  }

  /// Set the starting point, projected onto the admissible set — the
  /// optimiser is never started outside the box it is confined to.
  pub fn set_initial_guess(&mut self, params: BSMParams) {
    self.params = params.projected();
  }

  /// The parameters the model is actually priced at: the stored ones,
  /// projected — [`SabrCalibrator`](crate::calibration::sabr::SabrCalibrator)'s
  /// `effective_params` under the same name.
  ///
  /// [`set_params`](LeastSquaresProblem::set_params) alone does not keep
  /// [`BSMPricer::new`] inside the box: `LevenbergMarquardt::minimize`
  /// evaluates residuals and the Jacobian at the *starting* point before it
  /// has a step to hand back, so the first call into the pricer reads
  /// whatever the caller left in the `pub params` field. Projecting on read
  /// closes that path and every other one at once;
  /// `the_optimisers_first_evaluation_is_already_inside_the_box` fails
  /// without it.
  fn effective_params(&self) -> BSMParams {
    self.params.clone().projected()
  }

  /// `(s, k, q, tau)` for quote `idx` — the per-quote half of the
  /// `(s, k, r, q, tau)` query the [`BSMPricer`] model is priced at.
  /// `q` defaults to `0.0`; the hard-coded [`BSMCoc::Bsm1973`] carries at
  /// `b = r` and never reads it.
  fn query(&self, idx: usize) -> (f64, f64, f64, f64) {
    (
      self.s[idx],
      self.k[idx],
      self.q.unwrap_or(0.0),
      self.flat_t[idx],
    )
  }
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for BSMCalibrator {
  type JacobianStorage = Owned<f64, Dyn, Dyn>;
  type ParameterStorage = Owned<f64, Dyn>;
  type ResidualStorage = Owned<f64, Dyn>;

  /// Levenberg-Marquardt is unconstrained and steps wherever the
  /// linearised model points, so the raw iterate is stored only after
  /// projection — the same hook `SabrCalibrator` uses (`HestonStochCorr`
  /// carries `BOUNDS`, and Heston moves in bounded logistic coordinates
  /// instead).
  fn set_params(&mut self, params: &DVector<f64>) {
    self.params = BSMParams::from(params.clone()).projected();
  }

  fn params(&self) -> DVector<f64> {
    self.effective_params().into()
  }

  fn residuals(&self) -> Option<DVector<f64>> {
    let n = self.c_market.len();
    let mut c_model = DVector::zeros(n);
    let mut vegas: Vec<f64> = Vec::with_capacity(n);

    for (idx, _) in self.c_market.iter().enumerate() {
      let model = BSMPricer::new(self.effective_params().v, BSMCoc::Bsm1973);
      let (s, k, q, tau) = self.query(idx);
      let (call, put) = model.call_put(s, k, self.r, q, tau);

      match self.option_type {
        OptionType::Call => c_model[idx] = call,
        OptionType::Put => c_model[idx] = put,
      }

      // Collect vega for vega-weighted residuals (calibration in vol space)
      let vega = model.vega(s, k, self.r, q, tau).abs().max(1e-8);
      vegas.push(vega);

      self
        .calibration_history
        .borrow_mut()
        .push(CalibrationHistory {
          residuals: c_model.clone() - self.c_market.clone(),
          call_put: vec![(call, put)].into(),
          params: self.effective_params(),
          loss_scores: CalibrationLossScore::compute_selected(
            self.c_market.as_slice(),
            c_model.as_slice(),
            self.loss_metrics,
          ),
        });
    }

    // Vega-weighted residuals approximate minimizing implied vol differences
    let mut residuals = DVector::zeros(n);
    for i in 0..n {
      residuals[i] = (c_model[i] - self.c_market[i]) / vegas[i];
    }

    Some(residuals)
  }

  fn jacobian(&self) -> Option<DMatrix<f64>> {
    // For vega-weighted residuals r = (C_model - C_mkt)/Vega,
    // dr/dsigma = 1 - r * (Vomma / Vega)
    let n = self.c_market.len();
    let mut J = DMatrix::zeros(n, 1);

    for idx in 0..n {
      let model = BSMPricer::new(self.effective_params().v, BSMCoc::Bsm1973);
      let (s, k, q, tau) = self.query(idx);

      let c_model_i = model.price_option(s, k, self.r, q, tau, self.option_type);

      let vega = model.vega(s, k, self.r, q, tau).abs().max(1e-8);
      let vomma = model.vomma(s, k, self.r, q, tau);
      let r_i = (c_model_i - self.c_market[idx]) / vega;

      J[(idx, 0)] = 1.0 - r_i * (vomma / vega);
    }

    Some(J)
  }
}

#[cfg(test)]
mod tests;

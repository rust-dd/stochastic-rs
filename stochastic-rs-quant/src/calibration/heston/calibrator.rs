use ndarray::Array1;
use stochastic_rs_stats::heston_nmle_cekf::HestonNmleCekfConfig;

use super::params::HestonJacobianMethod;
use super::params::HestonMleSeedMethod;
use crate::OptionQuote;
use crate::QuantError;
use crate::calibration::Calibrated;
use crate::calibration::CalibrationHistory;
use crate::calibration::CalibrationOptions;
use crate::calibration::least_squares::LmOptions;
use crate::calibration::least_squares::minimize;
use crate::calibration::quote::assert_spot_and_rates;
use crate::calibration::quote::validate_quotes;
use crate::pricing::heston::HestonParams;

/// Spot and variance series whose NMLE, PMLE or NMLE-CEKF fit starts the calibration.
/// Wang et al. (2018), Sci. China Inf. Sci. 61, 042202, DOI 10.1007/s11432-017-9215-8.
#[derive(Clone, Debug, Default)]
pub struct HestonMleSeed {
  pub s: Array1<f64>,
  /// NMLE-CEKF filters its own variance and reads only a positive first value as `v0`.
  pub v: Array1<f64>,
  /// Drift of `s`; `None` reads the calibrator's `r`.
  pub r: Option<f64>,
  pub method: HestonMleSeedMethod,
  /// Sampling step; `None` is `1 / (n − 1)`, one unit of time over the series.
  pub delta: Option<f64>,
  /// NMLE-CEKF settings, whose `r` and `delta` are the seed's; `None` is the default.
  pub nmle_cekf: Option<HestonNmleCekfConfig>,
}

/// Heston fit to vanilla quotes by Levenberg–Marquardt in logistic coordinates, with Cui et al.'s
/// analytic Jacobian by default.
/// Levenberg (1944), DOI 10.1090/qam/10666; Marquardt (1963), DOI 10.1137/0111030.
#[derive(Clone, Debug)]
pub struct HestonCalibrator {
  pub s: f64,
  pub r: f64,
  pub q: f64,
  pub quotes: Vec<OptionQuote>,
  /// `None`: the MLE seed when `mle_seed` is set, else the fallback (0.04, 1.5, 0.04, 0.5, −0.5).
  pub start: Option<HestonParams>,
  /// Levenberg–Marquardt knobs; `regularization` pulls `(v0, κ, θ, σ, ρ)`.
  pub options: CalibrationOptions,
  /// Positive per-quote weights, normalised to unit root mean square; `None` is unit weights.
  pub residual_weights: Option<Array1<f64>>,
  pub jacobian_method: HestonJacobianMethod,
  /// Lower σ of the start and of every iterate to Feller's bound `σ² ≤ 2κθ` (off by default).
  pub feller_projection: bool,
  pub mle_seed: Option<HestonMleSeed>,
  /// The trail of the clone that runs the fit, moved into its result.
  pub(super) history: Vec<CalibrationHistory<HestonParams>>,
  /// Weight of the row on σ's overshoot past the Feller cap; set only inside a projected fit.
  pub(super) feller_row_weight: Option<f64>,
}

impl HestonCalibrator {
  /// Panics unless `s` is finite and positive and `r`, `q` finite; `calibrate` checks the quotes.
  pub fn new(s: f64, r: f64, q: f64, quotes: Vec<OptionQuote>) -> Self {
    assert_spot_and_rates(s, r, q);
    Self {
      s,
      r,
      q,
      quotes,
      start: None,
      options: CalibrationOptions::default(),
      residual_weights: None,
      jacobian_method: HestonJacobianMethod::default(),
      feller_projection: false,
      mle_seed: None,
      history: Vec::new(),
      feller_row_weight: None,
    }
  }

  /// Panics unless `start` passes [`HestonParams::new`]; the fit starts from its projection onto
  /// the calibration box.
  pub fn with_start(mut self, start: HestonParams) -> Self {
    self.start = Some(start.checked());
    self
  }

  pub fn with_options(mut self, options: CalibrationOptions) -> Self {
    self.options = options;
    self
  }

  /// Panics unless `weights.len() == quotes.len()` and every weight is finite and positive, with a
  /// root mean square that neither overflows nor underflows.
  pub fn with_residual_weights(mut self, weights: impl Into<Array1<f64>>) -> Self {
    let weights = weights.into();
    check_residual_weights(&weights, self.quotes.len()).unwrap_or_else(|e| panic!("{e}"));
    let rms = (weights.iter().map(|w| w * w).sum::<f64>() / weights.len() as f64).sqrt();
    assert!(
      rms.is_finite() && rms > 0.0,
      "the weights' root mean square must satisfy `rms.is_finite() && rms > 0.0`, got rms = {rms:?}"
    );
    self.residual_weights = Some(weights / rms);
    self
  }

  pub fn with_jacobian_method(mut self, method: HestonJacobianMethod) -> Self {
    self.jacobian_method = method;
    self
  }

  /// With `true` every iterate passes [`HestonParams::satisfies_feller_condition`]: a residual on
  /// σ's overshoot past √(2κθ) brings it back, and a violating surface is fitted on the boundary.
  pub fn with_feller_projection(mut self, enabled: bool) -> Self {
    self.feller_projection = enabled;
    self
  }

  /// Panics unless `seed.r` and `seed.delta`, where given, are finite and `delta` positive.
  pub fn with_mle_seed(mut self, seed: HestonMleSeed) -> Self {
    if let Some(r) = seed.r {
      assert!(
        r.is_finite(),
        "seed.r must satisfy `seed.r.is_finite()`, got seed.r = {r:?}"
      );
    }
    if let Some(delta) = seed.delta {
      assert!(
        delta.is_finite() && delta > 0.0,
        "seed.delta must satisfy `seed.delta.is_finite() && seed.delta > 0.0`, got seed.delta = {delta:?}"
      );
    }
    self.mle_seed = Some(seed);
    self
  }

  /// The seed's fit, projected, or the fallback without a seed; `InvalidInput` when the seed fails.
  fn seeded_start(&self) -> Result<HestonParams, QuantError> {
    match &self.mle_seed {
      Some(seed) => seed
        .fit(self.r)
        .map(HestonParams::projected)
        .map_err(|e| QuantError::invalid("heston mle seed", e.to_string())),
      None => Ok(Self::fallback_params()),
    }
  }
}

/// `DimensionMismatch` unless there is one weight per quote, `InvalidInput` for the first weight
/// that is not finite and positive.
fn check_residual_weights(weights: &Array1<f64>, quotes: usize) -> Result<(), QuantError> {
  if weights.len() != quotes {
    return Err(QuantError::DimensionMismatch {
      what: "residual weights, one per quote",
      expected: quotes,
      got: weights.len(),
    });
  }
  match weights
    .iter()
    .enumerate()
    .find(|(_, w)| !(w.is_finite() && **w > 0.0))
  {
    Some((i, weight)) => Err(QuantError::invalid(
      "residual weights",
      format!(
        "weight {i} must satisfy `weight.is_finite() && weight > 0.0`, got weight = {weight:?}"
      ),
    )),
    None => Ok(()),
  }
}

impl crate::traits::Calibrator for HestonCalibrator {
  type Params = HestonParams;
  type Output = Calibrated<HestonParams>;
  type Error = QuantError;

  /// A start replaces `self.start`; `InvalidInput` unless the one used passes `HestonParams::new`.
  fn calibrate(
    &self,
    initial: Option<HestonParams>,
  ) -> Result<Calibrated<HestonParams>, QuantError> {
    validate_quotes(&self.quotes)?;
    self.options.regularization_for(5)?;
    if let Some(weights) = &self.residual_weights {
      check_residual_weights(weights, self.quotes.len())?;
    }
    let start = match initial.or(self.start) {
      Some(start) => start,
      None => self.seeded_start()?,
    };
    if let Some(reason) = start.violation() {
      return Err(QuantError::invalid("start", reason));
    }
    let mut problem = self.clone();
    problem.start = Some(start);
    if problem.feller_projection {
      problem.feller_row_weight = Some(problem.feller_row_weight_at_start());
    }
    let (mut fitted, report) = minimize(
      problem,
      LmOptions {
        tolerance: self.options.tolerance,
        patience: self.options.patience,
        pivoted_qr: true,
        max_iterations: self.options.max_iterations,
      },
    );
    let params = fitted.effective_params();
    let model = fitted.compute_model_prices_for(&params).to_vec();
    let market = self
      .quotes
      .iter()
      .map(|quote| quote.price)
      .collect::<Vec<_>>();
    let history = std::mem::take(&mut fitted.history);
    Ok(report.calibrated(params, &market, &model, self.options.loss_metrics, history))
  }
}

use ndarray::Array1;
use stochastic_rs_distributions::RealExt;
use stochastic_rs_stats::heston_nmle_cekf::HestonNmleCekfConfig;

use super::lsq::HestonProblem;
use super::params::HestonJacobianMethod;
use super::params::HestonMleSeedMethod;
use crate::OptionQuote;
use crate::QuantError;
use crate::calibration::Calibrated;
use crate::calibration::CalibrationOptions;
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

/// Heston fit to vanilla quotes by Levenberg–Marquardt, Cui et al.'s analytic Jacobian by default.
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
  /// Positive per-quote weights, `None` for unit ones; the builder scales them to unit root mean
  /// square, while a write to this field is used as it stands.
  pub residual_weights: Option<Array1<f64>>,
  pub jacobian_method: HestonJacobianMethod,
  /// Lower σ of the start and of every iterate to Feller's bound `σ² ≤ 2κθ` (off by default).
  pub feller_projection: bool,
  pub mle_seed: Option<HestonMleSeed>,
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

  /// Panics unless `weights.len() == quotes.len()` and every weight is finite and positive; stores
  /// them scaled to unit root mean square.
  pub fn with_residual_weights(mut self, weights: impl Into<Array1<f64>>) -> Self {
    let weights = weights.into();
    let quotes = self.quotes.len();
    assert!(
      weights.len() == quotes,
      "weights must satisfy `weights.len() == quotes.len()`, got weights.len() = {} for {quotes} quotes",
      weights.len()
    );
    check_residual_weights(&weights).unwrap_or_else(|e| panic!("{e}"));
    self.residual_weights = Some(unit_root_mean_square(weights));
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

/// `weights` over their root mean square, summed plainly; where the squares leave `f64`, the same
/// quantity is factored through the largest weight, which rounds differently.
fn unit_root_mean_square(weights: Array1<f64>) -> Array1<f64> {
  let n = weights.len() as f64;
  let rms = (weights.iter().map(|w| w * w).sum::<f64>() / n).sqrt();
  if rms.is_finite() && rms > 0.0 {
    return weights / rms;
  }
  let largest = f64::max_or_nan_fold(weights.iter().copied());
  let scaled = (weights.iter().map(|w| (w / largest).powi(2)).sum::<f64>() / n).sqrt();
  weights / (largest * scaled)
}

/// `InvalidInput` for the first weight that is not finite and positive.
fn check_residual_weights(weights: &Array1<f64>) -> Result<(), QuantError> {
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
    let lm = self.options.lm(true, 5)?;
    if let Some(weights) = &self.residual_weights {
      if weights.len() != self.quotes.len() {
        return Err(QuantError::DimensionMismatch {
          what: "residual weights, one per quote",
          expected: self.quotes.len(),
          got: weights.len(),
        });
      }
      check_residual_weights(weights)?;
    }
    let start = match initial.or(self.start) {
      Some(start) => start,
      None => self.seeded_start()?,
    };
    if let Some(reason) = start.violation() {
      return Err(QuantError::invalid("start", reason));
    }
    let mut problem = HestonProblem::new(self, start);
    if self.feller_projection {
      problem.feller_row_weight = Some(self.feller_row_weight_at(problem.effective_params()));
    }
    let (fitted, report) = minimize(problem, lm);
    let params = fitted.effective_params();
    let model = self.compute_model_prices_for(&params).to_vec();
    report.calibrated(
      params,
      &fitted.market,
      &model,
      self.options.loss_metrics,
      fitted.trail.into_entries(),
    )
  }
}

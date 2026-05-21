use super::calibrator::LevyCalibrator;
use crate::CalibrationLossScore;

/// Supported Lévy model types for calibration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LevyModelType {
  /// Variance Gamma: $\psi(\xi)=-\frac{1}{\nu}\ln\!\bigl(1-i\theta\nu\xi+\tfrac12\sigma^2\nu\xi^2\bigr)$
  VarianceGamma,
  /// Normal Inverse Gaussian: $\psi(\xi)=\delta\bigl(\sqrt{\alpha^2-\beta^2}-\sqrt{\alpha^2-(\beta+i\xi)^2}\bigr)$
  Nig,
  /// Cgmy: $\psi(\xi)=C\,\Gamma(-Y)\bigl[(M-i\xi)^Y-M^Y+(G+i\xi)^Y-G^Y\bigr]$
  Cgmy,
  /// Merton Jump-Diffusion: $\psi(\xi)=-\tfrac12\sigma^2\xi^2+\lambda\bigl(e^{i\mu_J\xi-\frac12\sigma_J^2\xi^2}-1\bigr)$
  MertonJD,
  /// Kou Double-Exponential: $\psi(\xi)=-\tfrac12\sigma^2\xi^2+\lambda\bigl(\frac{p\eta_1}{\eta_1-i\xi}+\frac{(1-p)\eta_2}{\eta_2+i\xi}-1\bigr)$
  Kou,
}

/// Market data for a single maturity slice.
#[derive(Clone, Debug)]
pub struct MarketSlice {
  /// Strike prices.
  pub strikes: Vec<f64>,
  /// Market option prices.
  pub prices: Vec<f64>,
  /// `true` for call, `false` for put.
  pub is_call: Vec<bool>,
  /// Time to maturity in years.
  pub t: f64,
}

impl MarketSlice {
  /// Build a `MarketSlice` from evaluation and expiration dates using a
  /// day-count convention.
  ///
  /// Bridges the `tau`-only calibrator API with date-native market data —
  /// callers can pass `chrono::NaiveDate` instead of pre-computed
  /// year-fractions. Mirrors the `eval` / `expiration` pattern available on
  /// pricers via [`crate::traits::TimeExt::tau_with_dcc`].
  pub fn from_dates(
    strikes: Vec<f64>,
    prices: Vec<f64>,
    is_call: Vec<bool>,
    eval: chrono::NaiveDate,
    expiration: chrono::NaiveDate,
    dcc: crate::calendar::DayCountConvention,
  ) -> Self {
    let t = dcc.year_fraction(eval, expiration);
    Self {
      strikes,
      prices,
      is_call,
      t,
    }
  }
}

/// Calibrated parameter set for a Lévy model — the parameter vector together
/// with its [`LevyModelType`] tag (the vector layout depends on the model).
#[derive(Clone, Debug)]
pub struct LevyParams {
  /// Calibrated parameter vector. Layout is model-specific.
  pub values: Vec<f64>,
  /// Lévy model variant the vector belongs to.
  pub model_type: LevyModelType,
}

/// Calibration result for a Lévy model.
#[derive(Clone, Debug)]
pub struct LevyCalibrationResult {
  /// Calibrated parameter vector.
  pub params: Vec<f64>,
  /// Model type that was calibrated.
  pub model_type: LevyModelType,
  /// Calibration loss metrics.
  pub loss: CalibrationLossScore,
  /// Whether the optimiser converged.
  pub converged: bool,
  /// Number of LM iterations performed.
  pub iterations: usize,
}

/// Variant-dispatching wrapper around the five Lévy Fourier pricers.
///
/// Used by [`LevyCalibrationResult::to_model`] so the result remains a
/// concrete `ModelPricer` (no `Box<dyn>`).
#[derive(Clone, Debug)]
pub enum LevyModel {
  VarianceGamma(crate::pricing::fourier::VarianceGammaFourier),
  Nig(crate::pricing::fourier::NigFourier),
  Cgmy(crate::pricing::fourier::CGMYFourier),
  MertonJd(crate::pricing::fourier::MertonJDFourier),
  Kou(crate::pricing::fourier::KouFourier),
}

impl crate::traits::ModelPricer for LevyModel {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    match self {
      LevyModel::VarianceGamma(m) => m.price_call(s, k, r, q, tau),
      LevyModel::Nig(m) => m.price_call(s, k, r, q, tau),
      LevyModel::Cgmy(m) => m.price_call(s, k, r, q, tau),
      LevyModel::MertonJd(m) => m.price_call(s, k, r, q, tau),
      LevyModel::Kou(m) => m.price_call(s, k, r, q, tau),
    }
  }
}

impl crate::traits::ToModel for LevyCalibrationResult {
  type Model = LevyModel;
  fn to_model(&self, r: f64, q: f64) -> Self::Model {
    LevyCalibrationResult::to_model(self, r, q)
  }
}

impl crate::traits::CalibrationResult for LevyCalibrationResult {
  type Params = LevyParams;
  fn rmse(&self) -> f64 {
    self.loss.get(crate::LossMetric::Rmse)
  }
  fn params(&self) -> Self::Params {
    LevyParams {
      values: self.params.clone(),
      model_type: self.model_type,
    }
  }
  fn converged(&self) -> bool {
    self.converged
  }
  fn loss_score(&self) -> Option<&crate::CalibrationLossScore> {
    Some(&self.loss)
  }
}

impl crate::traits::Calibrator for LevyCalibrator {
  type InitialGuess = Vec<f64>;
  type Params = LevyParams;
  type Output = LevyCalibrationResult;
  type Error = anyhow::Error;

  fn calibrate(&self, initial: Option<Self::InitialGuess>) -> Result<Self::Output, Self::Error> {
    Ok(self.solve(initial))
  }
}

impl LevyCalibrationResult {
  /// Convert to a Fourier model for pricing / vol surface generation.
  pub fn to_model(&self, r: f64, q: f64) -> LevyModel {
    use crate::pricing::fourier::*;
    let p = &self.params;
    match self.model_type {
      LevyModelType::VarianceGamma => LevyModel::VarianceGamma(VarianceGammaFourier {
        sigma: p[0],
        theta: p[1],
        nu: p[2],
        r,
        q,
      }),
      LevyModelType::Nig => LevyModel::Nig(NigFourier {
        alpha: p[0],
        beta: p[1],
        delta: p[2],
        r,
        q,
      }),
      LevyModelType::Cgmy => LevyModel::Cgmy(CGMYFourier {
        c: p[0],
        g: p[1],
        m: p[2],
        y: p[3],
        r,
        q,
      }),
      LevyModelType::MertonJD => LevyModel::MertonJd(MertonJDFourier {
        sigma: p[0],
        lambda: p[1],
        mu_j: p[2],
        sigma_j: p[3],
        r,
        q,
      }),
      LevyModelType::Kou => LevyModel::Kou(KouFourier {
        sigma: p[0],
        lambda: p[1],
        p_up: p[2],
        eta1: p[3],
        eta2: p[4],
        r,
        q,
      }),
    }
  }
}

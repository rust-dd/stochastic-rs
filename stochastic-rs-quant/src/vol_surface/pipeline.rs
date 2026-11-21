//! # Volatility Surface Pipeline
//!
//! End-to-end pipeline: market prices → implied vol surface → SVI per slice →
//! SSVI global fit → arbitrage validation → local volatility extraction.
//!
//! Reference: Gatheral & Jacquier (2012), arXiv:1204.0646

use ndarray::Array2;
use ndarray::ArrayView2;

use super::analytics::SmileAnalytics;
use super::implied::ImpliedVolSurface;
use super::model_surface::ModelSurface;
use super::ssvi::SsviSurface;
use super::svi::SviRawParams;
use crate::OptionType;
use crate::total_order::ascending;
use crate::traits::ToModel;

/// Result of the full vol-surface pipeline.
#[derive(Clone, Debug)]
pub struct VolSurfaceResult {
  /// Implied volatility surface from market data
  pub iv_surface: ImpliedVolSurface,
  /// SVI fit per maturity slice
  pub svi_params: Vec<SviRawParams<f64>>,
  /// SSVI global surface fit
  pub ssvi_surface: SsviSurface<f64>,
  /// Smile analytics per maturity
  pub analytics: Vec<SmileAnalytics<f64>>,
  /// Butterfly arbitrage check per slice: `(is_free, min_g)`
  pub butterfly_checks: Vec<(bool, f64)>,
  /// Calendar-spread arbitrage: `true` if free
  pub calendar_spread_free: bool,
}

impl VolSurfaceResult {
  /// Whether the entire surface is arbitrage-free.
  ///
  /// - `Some(true)` — every slice was checked and none violates the butterfly
  ///   or the calendar-spread condition.
  /// - `Some(false)` — a checked slice violates one of them.
  /// - `None` — **nothing was checkable**, so there is no answer to give.
  ///
  /// The third case is why this is not a `bool`. Both arbitrage checks are
  /// universal quantifiers over a grid — `check_butterfly_ssvi` seeds
  /// `min_g` at $+\infty$ and lowers it only where the density evaluates,
  /// [`SsviSurface::is_calendar_spread_free`](super::ssvi::SsviSurface::is_calendar_spread_free)
  /// returns early only on a violation — so both are vacuously `true` over an
  /// empty grid. And the grid *does* go empty:
  /// [`ImpliedVolSurface::smile_slice`](super::implied::ImpliedVolSurface::smile_slice)
  /// deliberately drops nodes whose implied vol did not invert, which is
  /// right in itself but leaves a surface built from out-of-band prices with
  /// no nodes at all. "No violation found" and "nothing to look at" then
  /// produce the same `true`, and a caller cannot separate them.
  ///
  /// A slice is treated as checked when its `min_g` came back finite: `+inf`
  /// is the untouched seed, so it is exactly the marker for a slice on which
  /// the loop body never ran. That also settles the calendar leg, which shares
  /// the same per-slice log-moneyness grids.
  ///
  /// This is the `bool` analogue of case 2 of the crate's [failure
  /// convention](crate::traits::ModelPricer#how-pricing-fails): the inputs
  /// were legitimate, the answer is genuinely undefined here, and `None`
  /// propagates where a `true` would have been mistaken for a clean bill of
  /// health.
  pub fn is_arbitrage_free(&self) -> Option<bool> {
    let anything_checked = !self.butterfly_checks.is_empty()
      && self
        .butterfly_checks
        .iter()
        .all(|(_, min_g)| min_g.is_finite());
    if !anything_checked {
      return None;
    }
    Some(self.calendar_spread_free && self.butterfly_checks.iter().all(|(free, _)| *free))
  }

  /// Compute local volatility surface on a grid from the SSVI fit.
  pub fn local_vol_surface(&self, ks: &[f64], ts: &[f64]) -> Array2<f64> {
    self.ssvi_surface.local_vol_surface(ks, ts)
  }
}

/// The IV surface, SVI/SSVI fits, analytics and arbitrage diagnostics of an **undiscounted**
/// `(N_T, N_K)` price grid on ascending axes, one forward per maturity.
pub fn build_surface(
  strikes: Vec<f64>,
  maturities: Vec<f64>,
  forwards: Vec<f64>,
  prices: ArrayView2<f64>,
  option_type: OptionType,
) -> VolSurfaceResult {
  let iv_surface =
    ImpliedVolSurface::from_prices(strikes, maturities, forwards, prices, option_type);

  build_surface_from_iv(&iv_surface)
}

/// Build a complete volatility surface from any calibrated model.
///
/// Works with all [`ModelSurface`] implementations: Heston, Bates/SVJ, Lévy
/// (Vg, Nig, Cgmy, Merton, Kou), HSCM, Sabr, or any custom model.
///
/// The bound is [`ModelSurface`] and not
/// [`ModelPricer`](crate::traits::ModelPricer) because this function inverts
/// the model's calls through the Black formula — see
/// [`VanillaEuropeanCall`](crate::traits::VanillaEuropeanCall) for why a
/// digital or American `ModelPricer` must not reach it.
///
/// # Arguments
/// * `model` - Calibrated model implementing [`ModelSurface`]
/// * `s` - Spot price
/// * `r` - Risk-free rate
/// * `q` - Dividend yield
/// * `strikes` - Strike prices (ascending)
/// * `maturities` - Maturities in years (ascending)
pub fn build_surface_from_model<M: ModelSurface + ?Sized>(
  model: &M,
  s: f64,
  r: f64,
  q: f64,
  strikes: &[f64],
  maturities: &[f64],
) -> VolSurfaceResult {
  let iv_surface = model.vol_surface(s, r, q, strikes, maturities);
  build_surface_from_iv(&iv_surface)
}

/// The volatility surface of a calibration result whose model is a European vanilla call pricer;
/// [`ToModel`] alone also admits a model with no surface, hence the `where` clause.
///
/// ```
/// use stochastic_rs_quant::{OptionQuote, OptionType};
/// use stochastic_rs_quant::calibration::{BsmCalibrator, BsmParams};
/// use stochastic_rs_quant::pricing::bsm::{BsmCoc, BsmPricer};
/// use stochastic_rs_quant::traits::{Calibrator, ModelPricer};
/// use stochastic_rs_quant::vol_surface::build_surface_from_calibration;
///
/// let (s, k, r, tau) = (100.0, 100.0, 0.05, 1.0);
/// let call = BsmPricer::new(0.2, BsmCoc::Bsm1973).price_call(s, k, r, 0.0, tau);
/// let quotes = vec![OptionQuote::new(k, tau, call, OptionType::Call)];
/// let calibrator = BsmCalibrator::new(s, r, 0.0, quotes).with_start(BsmParams { sigma: 0.3 });
/// let result = calibrator.calibrate(None).unwrap();
/// let surface = build_surface_from_calibration(&result, s, r, 0.0, &[k], &[tau]);
/// assert_eq!(surface.iv_surface.ivs.dim(), (1, 1));
/// ```
pub fn build_surface_from_calibration<C: ToModel>(
  calibration: &C,
  s: f64,
  r: f64,
  q: f64,
  strikes: &[f64],
  maturities: &[f64],
) -> VolSurfaceResult
where
  C::Model: ModelSurface,
{
  let model = calibration.to_model();
  build_surface_from_model(&model, s, r, q, strikes, maturities)
}

/// Build SVI/SSVI fits and diagnostics from an existing implied vol surface.
pub fn build_surface_from_iv(iv_surface: &ImpliedVolSurface) -> VolSurfaceResult {
  let nt = iv_surface.maturities.len();

  let svi_params: Vec<_> = iv_surface
    .fit_svi_slices()
    .into_iter()
    .map(|fit| fit.params)
    .collect();

  let ssvi_surface = iv_surface.fit_ssvi(None).params;

  let analytics: Vec<SmileAnalytics<f64>> = svi_params
    .iter()
    .zip(iv_surface.maturities.iter())
    .map(|(svi, &tau)| super::analytics::svi_analytics(svi, tau))
    .collect();

  // Union of log-moneyness grids across all maturity slices — used both for
  // butterfly checks (per slice) and for the full smile-wide calendar-spread
  // condition (Gatheral & Jacquier 2014 Theorem 4.2).
  let mut calendar_ks: Vec<f64> = Vec::new();
  let butterfly_checks: Vec<(bool, f64)> = (0..nt)
    .map(|j| {
      let slice = iv_surface.smile_slice(j);
      let theta = slice.to_ssvi_slice().theta;
      let ks: Vec<f64> = slice.log_moneyness.clone();
      let result = super::arbitrage::check_butterfly_ssvi(&ssvi_surface.params, theta, &ks);
      calendar_ks.extend(ks);
      result
    })
    .collect();

  // De-dupe & sort calendar k-grid for cheaper / deterministic checking.
  calendar_ks.sort_by(ascending);
  calendar_ks.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

  let calendar_spread_free = ssvi_surface.is_calendar_spread_free(&calendar_ks);

  VolSurfaceResult {
    iv_surface: iv_surface.clone(),
    svi_params,
    ssvi_surface,
    analytics,
    butterfly_checks,
    calendar_spread_free,
  }
}

#[cfg(test)]
mod tests;

//! # Volatility Surface Pipeline
//!
//! End-to-end pipeline: market prices → implied vol surface → SVI per slice →
//! SSVI global fit → arbitrage validation → local volatility extraction.
//!
//! Reference: Gatheral & Jacquier (2012), arXiv:1204.0646

use ndarray::Array2;

use super::analytics::SmileAnalytics;
use super::implied::ImpliedVolSurface;
use super::model_surface::ModelSurface;
use super::ssvi::SsviSurface;
use super::svi::SviRawParams;
use crate::traits::ModelPricer;
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
  pub fn is_arbitrage_free(&self) -> bool {
    self.calendar_spread_free && self.butterfly_checks.iter().all(|(free, _)| *free)
  }

  /// Compute local volatility surface on a grid from the SSVI fit.
  pub fn local_vol_surface(&self, ks: &[f64], ts: &[f64]) -> Array2<f64> {
    self.ssvi_surface.local_vol_surface(ks, ts)
  }
}

/// Build a complete volatility surface from market option prices.
///
/// # Arguments
/// * `strikes` - Strike prices (ascending)
/// * `maturities` - Maturities in years (ascending)
/// * `forwards` - Forward prices per maturity
/// * `prices` - **Undiscounted** option price grid (N_T, N_K)
/// * `is_call` - Whether prices are calls
///
/// # Returns
/// A [`VolSurfaceResult`] containing IV surface, SVI/SSVI fits,
/// analytics, and arbitrage diagnostics.
pub fn build_surface(
  strikes: Vec<f64>,
  maturities: Vec<f64>,
  forwards: Vec<f64>,
  prices: &Array2<f64>,
  is_call: bool,
) -> VolSurfaceResult {
  let iv_surface = ImpliedVolSurface::from_prices(strikes, maturities, forwards, prices, is_call);

  build_surface_from_iv(&iv_surface)
}

/// Build a complete volatility surface from any calibrated model.
///
/// Works with all `ModelPricer` implementations: Heston, Bates/SVJ, Lévy
/// (VG, NIG, CGMY, Merton, Kou), HSCM, SABR, or any custom model.
///
/// # Arguments
/// * `model` - Calibrated model implementing [`ModelPricer`]
/// * `s` - Spot price
/// * `r` - Risk-free rate
/// * `q` - Dividend yield
/// * `strikes` - Strike prices (ascending)
/// * `maturities` - Maturities in years (ascending)
pub fn build_surface_from_model(
  model: &dyn ModelPricer,
  s: f64,
  r: f64,
  q: f64,
  strikes: &[f64],
  maturities: &[f64],
) -> VolSurfaceResult {
  let iv_surface = model.vol_surface(s, r, q, strikes, maturities);
  build_surface_from_iv(&iv_surface)
}

/// Build a complete volatility surface directly from a calibration result.
///
/// Accepts any type implementing [`ToModel`] (all calibration results).
///
/// ```rust,ignore
/// let result = heston_calibrator.calibrate();
/// let surface = build_surface_from_calibration(&result, s, r, q, &strikes, &mats);
/// ```
pub fn build_surface_from_calibration(
  calibration: &dyn ToModel,
  s: f64,
  r: f64,
  q: f64,
  strikes: &[f64],
  maturities: &[f64],
) -> VolSurfaceResult {
  let model = calibration.to_model(r, q);
  build_surface_from_model(model.as_ref(), s, r, q, strikes, maturities)
}

/// Build SVI/SSVI fits and diagnostics from an existing implied vol surface.
pub fn build_surface_from_iv(iv_surface: &ImpliedVolSurface) -> VolSurfaceResult {
  let nt = iv_surface.maturities.len();

  let svi_params = iv_surface.fit_svi_slices();

  let ssvi_surface = iv_surface.fit_ssvi(None);

  let analytics: Vec<SmileAnalytics<f64>> = svi_params
    .iter()
    .zip(iv_surface.maturities.iter())
    .map(|(svi, &tau)| super::analytics::svi_analytics(svi, tau))
    .collect();

  let butterfly_checks: Vec<(bool, f64)> = (0..nt)
    .map(|j| {
      let slice = iv_surface.smile_slice(j);
      let theta = slice.to_ssvi_slice().theta;
      let ks: Vec<f64> = slice.log_moneyness;
      super::arbitrage::check_butterfly_ssvi(&ssvi_surface.params, theta, &ks)
    })
    .collect();

  let calendar_spread_free = ssvi_surface.is_calendar_spread_free();

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
mod tests {
  use ndarray::Array2;

  use super::*;

  fn make_test_prices() -> (Vec<f64>, Vec<f64>, Vec<f64>, Array2<f64>) {
    use statrs::distribution::ContinuousCDF;
    use statrs::distribution::Normal;

    let normal = Normal::new(0.0, 1.0).unwrap();
    let s = 100.0;
    let r = 0.05;

    let strikes = vec![85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0];
    let maturities = vec![0.25, 0.50, 1.0];
    let forwards: Vec<f64> = maturities.iter().map(|&t| s * f64::exp(r * t)).collect();

    let base_vol = 0.20;
    let skew = -0.1;

    let mut prices = Array2::<f64>::zeros((maturities.len(), strikes.len()));
    for (j, &t) in maturities.iter().enumerate() {
      let f = forwards[j];
      for (i, &k) in strikes.iter().enumerate() {
        let moneyness = (k / f).ln();
        let sigma = base_vol + skew * moneyness;
        let d1 = ((f / k).ln() + 0.5 * sigma * sigma * t) / (sigma * t.sqrt());
        let d2 = d1 - sigma * t.sqrt();
        prices[[j, i]] = f * normal.cdf(d1) - k * normal.cdf(d2);
      }
    }

    (strikes, maturities, forwards, prices)
  }

  #[test]
  fn full_pipeline() {
    let (strikes, maturities, forwards, prices) = make_test_prices();

    let result = build_surface(strikes, maturities, forwards, &prices, true);

    assert_eq!(result.svi_params.len(), 3);
    assert_eq!(result.analytics.len(), 3);
    assert_eq!(result.butterfly_checks.len(), 3);

    for a in &result.analytics {
      assert!(
        a.atm_vol > 0.0 && a.atm_vol.is_finite(),
        "ATM vol should be positive"
      );
      assert!(a.atm_skew.is_finite(), "ATM skew should be finite");
    }

    for svi in &result.svi_params {
      assert!(svi.is_admissible(), "SVI should be admissible");
    }

    assert!(
      result.calendar_spread_free,
      "should be calendar-spread free"
    );
  }

  #[test]
  fn pipeline_local_vol() {
    let (strikes, maturities, forwards, prices) = make_test_prices();
    let result = build_surface(strikes, maturities, forwards, &prices, true);

    let ks: Vec<f64> = (-10..=10).map(|i| i as f64 * 0.05).collect();
    let ts = vec![0.3, 0.5, 0.75];
    let lv = result.local_vol_surface(&ks, &ts);

    assert_eq!(lv.dim(), (3, 21));

    let center_lv = lv[[1, 10]];
    assert!(
      center_lv.is_finite() && center_lv > 0.0,
      "center local vol should be positive: {center_lv}"
    );
  }

  #[test]
  fn build_surface_from_model_heston() {
    use crate::quant::pricing::fourier::HestonFourier;

    let model = HestonFourier {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma: 0.3,
      rho: -0.7,
      r: 0.05,
      q: 0.0,
    };
    let strikes: Vec<f64> = (85..=115).step_by(5).map(|k| k as f64).collect();
    let maturities = vec![0.25, 0.5, 1.0];

    let result = build_surface_from_model(&model, 100.0, 0.05, 0.0, &strikes, &maturities);

    assert_eq!(result.svi_params.len(), 3);
    assert_eq!(result.analytics.len(), 3);
    for a in &result.analytics {
      assert!(a.atm_vol > 0.0 && a.atm_vol.is_finite());
    }
  }

  #[test]
  fn build_surface_from_calibration_heston_multi_maturity() {
    use crate::quant::calibration::heston::{HestonCalibrator, HestonParams};
    use crate::quant::calibration::levy::MarketSlice;
    use crate::quant::pricing::fourier::HestonFourier;
    use crate::quant::OptionType;
    use crate::traits::ModelPricer;

    // Generate synthetic prices from known Heston params
    let true_model = HestonFourier {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma: 0.3,
      rho: -0.7,
      r: 0.05,
      q: 0.0,
    };
    let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
    let taus = [0.25, 0.5, 1.0];

    let slices: Vec<MarketSlice> = taus
      .iter()
      .map(|&t| {
        let prices: Vec<f64> = strikes
          .iter()
          .map(|&k| true_model.price_call(100.0, k, 0.05, 0.0, t))
          .collect();
        MarketSlice {
          strikes: strikes.clone(),
          prices,
          is_call: vec![true; strikes.len()],
          t,
        }
      })
      .collect();

    let cal = HestonCalibrator::from_slices(
      Some(HestonParams {
        v0: 0.06,
        kappa: 1.5,
        theta: 0.06,
        sigma: 0.4,
        rho: -0.5,
      }),
      &slices,
      100.0,
      0.05,
      Some(0.0),
      OptionType::Call,
      false,
    );
    let params = cal.calibrate();

    let result = build_surface_from_calibration(
      &params,
      100.0,
      0.05,
      0.0,
      &[90., 95., 100., 105., 110.],
      &[0.25, 0.5, 1.0],
    );

    assert_eq!(result.svi_params.len(), 3);
    for a in &result.analytics {
      assert!(a.atm_vol > 0.0 && a.atm_vol.is_finite());
      assert!(a.atm_skew < 0.0, "negative rho should produce negative skew");
    }
  }

  #[test]
  fn build_surface_from_calibration_svj_multi_maturity() {
    use crate::quant::calibration::levy::MarketSlice;
    use crate::quant::calibration::svj::SVJCalibrator;
    use crate::quant::pricing::fourier::BatesFourier;
    use crate::quant::OptionType;
    use crate::traits::ModelPricer;

    let true_model = BatesFourier {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma_v: 0.3,
      rho: -0.7,
      lambda: 0.5,
      mu_j: -0.1,
      sigma_j: 0.15,
      r: 0.05,
      q: 0.0,
    };
    let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
    let taus = [0.5, 1.0];

    let slices: Vec<MarketSlice> = taus
      .iter()
      .map(|&t| {
        let prices: Vec<f64> = strikes
          .iter()
          .map(|&k| true_model.price_call(100.0, k, 0.05, 0.0, t))
          .collect();
        MarketSlice {
          strikes: strikes.clone(),
          prices,
          is_call: vec![true; strikes.len()],
          t,
        }
      })
      .collect();

    let cal = SVJCalibrator::from_slices(None, &slices, 100.0, 0.05, Some(0.0), OptionType::Call, false);
    let result = cal.calibrate(None);
    assert!(result.converged, "SVJ should converge");

    let surface = build_surface_from_calibration(
      &result,
      100.0,
      0.05,
      0.0,
      &[90., 95., 100., 105., 110.],
      &[0.5, 1.0],
    );
    assert_eq!(surface.analytics.len(), 2);
    for a in &surface.analytics {
      assert!(a.atm_vol > 0.0 && a.atm_vol.is_finite());
    }
  }

  #[test]
  fn build_surface_from_calibration_levy_vg() {
    use crate::quant::calibration::levy::{LevyCalibrator, LevyModelType, MarketSlice};
    use crate::quant::pricing::fourier::VarianceGammaFourier;
    use crate::traits::ModelPricer;

    let true_model = VarianceGammaFourier {
      sigma: 0.12,
      theta: -0.14,
      nu: 0.2,
      r: 0.05,
      q: 0.0,
    };
    let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];

    let slices: Vec<MarketSlice> = [0.5, 1.0]
      .iter()
      .map(|&t| {
        let prices: Vec<f64> = strikes
          .iter()
          .map(|&k| true_model.price_call(100.0, k, 0.05, 0.0, t))
          .collect();
        MarketSlice {
          strikes: strikes.clone(),
          prices,
          is_call: vec![true; strikes.len()],
          t,
        }
      })
      .collect();

    let cal = LevyCalibrator::new(LevyModelType::VarianceGamma, 100.0, 0.05, 0.0, slices);
    let result = cal.calibrate(None);

    let surface = build_surface_from_calibration(
      &result,
      100.0,
      0.05,
      0.0,
      &[90., 95., 100., 105., 110.],
      &[0.5, 1.0],
    );
    assert_eq!(surface.analytics.len(), 2);
    for a in &surface.analytics {
      assert!(a.atm_vol > 0.0 && a.atm_vol.is_finite());
    }
  }

  #[test]
  fn build_surface_from_model_sabr() {
    use crate::quant::pricing::sabr::SabrModel;

    let model = SabrModel {
      alpha: 0.2,
      beta: 1.0,
      nu: 0.4,
      rho: -0.3,
    };
    let result = build_surface_from_model(
      &model,
      100.0,
      0.05,
      0.0,
      &[90., 95., 100., 105., 110.],
      &[0.25, 0.5, 1.0],
    );
    assert_eq!(result.analytics.len(), 3);
    for a in &result.analytics {
      assert!(a.atm_vol > 0.0 && a.atm_vol.is_finite());
    }
  }

  #[test]
  fn to_model_trait_works_with_pipeline() {
    use crate::quant::pricing::fourier::HestonFourier;
    use crate::traits::ToModel;

    let model = HestonFourier {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma: 0.3,
      rho: -0.7,
      r: 0.05,
      q: 0.0,
    };
    // Use HestonParams (implements ToModel) via the trait
    let params = crate::quant::calibration::heston::HestonParams {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma: 0.3,
      rho: -0.7,
    };
    let boxed: Box<dyn crate::traits::ModelPricer> = ToModel::to_model(&params, 0.05, 0.0);

    let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
    let maturities = vec![0.25, 0.5, 1.0];

    let result = build_surface_from_model(boxed.as_ref(), 100.0, 0.05, 0.0, &strikes, &maturities);
    assert_eq!(result.svi_params.len(), 3);

    // Verify prices match direct model
    use crate::traits::ModelPricer;
    let direct = model.price_call(100.0, 100.0, 0.05, 0.0, 1.0);
    let via_trait = boxed.price_call(100.0, 100.0, 0.05, 0.0, 1.0);
    assert!(
      (direct - via_trait).abs() < 1e-10,
      "ToModel should produce identical prices: direct={direct}, trait={via_trait}"
    );
  }

  #[test]
  fn smile_slice_svi_fit() {
    let (strikes, maturities, forwards, prices) = make_test_prices();
    let iv_surface = ImpliedVolSurface::from_prices(strikes, maturities, forwards, &prices, true);

    let slice = iv_surface.smile_slice(1);
    let svi = slice.fit_svi(None);

    assert!(svi.is_admissible());

    for (&k, &w_mkt) in slice.log_moneyness.iter().zip(slice.total_variance.iter()) {
      let w_model = svi.total_variance(k);
      let rel_err = (w_model - w_mkt).abs() / w_mkt.max(1e-10);
      assert!(
        rel_err < 0.10,
        "SVI fit error too large at k={k}: model={w_model} market={w_mkt} rel_err={rel_err}"
      );
    }
  }
}

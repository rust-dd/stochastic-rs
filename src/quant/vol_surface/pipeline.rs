//! # Volatility Surface Pipeline
//!
//! End-to-end pipeline: market prices → implied vol surface → SVI per slice →
//! SSVI global fit → arbitrage validation → local volatility extraction.
//!
//! Reference: Gatheral & Jacquier (2012), arXiv:1204.0646

use ndarray::Array2;

use super::analytics::SmileAnalytics;
use super::implied::ImpliedVolSurface;
use super::ssvi::SsviSurface;
use super::svi::SviRawParams;

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
  use super::*;
  use ndarray::Array2;

  fn make_test_prices() -> (Vec<f64>, Vec<f64>, Vec<f64>, Array2<f64>) {
    use statrs::distribution::{ContinuousCDF, Normal};

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
      assert!(a.atm_vol > 0.0 && a.atm_vol.is_finite(), "ATM vol should be positive");
      assert!(a.atm_skew.is_finite(), "ATM skew should be finite");
    }

    for svi in &result.svi_params {
      assert!(svi.is_admissible(), "SVI should be admissible");
    }

    assert!(result.calendar_spread_free, "should be calendar-spread free");
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
  fn smile_slice_svi_fit() {
    let (strikes, maturities, forwards, prices) = make_test_prices();
    let iv_surface =
      ImpliedVolSurface::from_prices(strikes, maturities, forwards, &prices, true);

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

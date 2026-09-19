use ndarray::Array2;

use super::*;
use crate::traits::Calibrator;

fn make_test_prices() -> (Vec<f64>, Vec<f64>, Vec<f64>, Array2<f64>) {
  use stochastic_rs_distributions::special::norm_cdf;

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
      prices[[j, i]] = f * norm_cdf(d1) - k * norm_cdf(d2);
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
  use crate::pricing::fourier::HestonFourier;

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
  use crate::OptionType;
  use crate::calibration::heston::HestonCalibrator;
  use crate::calibration::heston::HestonParams;
  use crate::calibration::levy::MarketSlice;
  use crate::pricing::fourier::HestonFourier;
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
    .map(|&tau| {
      let prices: Vec<f64> = strikes
        .iter()
        .map(|&k| true_model.price_call(100.0, k, 0.05, 0.0, tau))
        .collect();
      MarketSlice {
        strikes: strikes.clone(),
        prices,
        is_call: vec![true; strikes.len()],
        tau,
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
  let calibration = cal.calibrate(None).unwrap();

  let result = build_surface_from_calibration(
    &calibration,
    100.0,
    0.05,
    0.0,
    &[90., 95., 100., 105., 110.],
    &[0.25, 0.5, 1.0],
  );

  assert_eq!(result.svi_params.len(), 3);
  for a in &result.analytics {
    assert!(a.atm_vol > 0.0 && a.atm_vol.is_finite());
    assert!(
      a.atm_skew < 0.0,
      "negative rho should produce negative skew"
    );
  }
}

#[test]
fn build_surface_from_calibration_svj_multi_maturity() {
  use crate::OptionType;
  use crate::calibration::levy::MarketSlice;
  use crate::calibration::svj::SVJCalibrator;
  use crate::pricing::fourier::BatesFourier;
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
    .map(|&tau| {
      let prices: Vec<f64> = strikes
        .iter()
        .map(|&k| true_model.price_call(100.0, k, 0.05, 0.0, tau))
        .collect();
      MarketSlice {
        strikes: strikes.clone(),
        prices,
        is_call: vec![true; strikes.len()],
        tau,
      }
    })
    .collect();

  let cal = SVJCalibrator::from_slices(
    None,
    &slices,
    100.0,
    0.05,
    Some(0.0),
    OptionType::Call,
    false,
  );
  let result = cal.calibrate(None).unwrap();
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
  use crate::calibration::levy::LevyCalibrator;
  use crate::calibration::levy::LevyModelType;
  use crate::calibration::levy::MarketSlice;
  use crate::pricing::fourier::VarianceGammaFourier;
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
    .map(|&tau| {
      let prices: Vec<f64> = strikes
        .iter()
        .map(|&k| true_model.price_call(100.0, k, 0.05, 0.0, tau))
        .collect();
      MarketSlice {
        strikes: strikes.clone(),
        prices,
        is_call: vec![true; strikes.len()],
        tau,
      }
    })
    .collect();

  let cal = LevyCalibrator::new(LevyModelType::VarianceGamma, 100.0, 0.05, 0.0, slices);
  let result = cal.calibrate(None).unwrap();

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
  use crate::pricing::sabr::SabrPricer;

  let model = SabrPricer {
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
  use crate::pricing::fourier::HestonFourier;
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
  let params = crate::calibration::heston::HestonParams {
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
  };
  let pricer = ToModel::to_model(&params, 0.05, 0.0);

  let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
  let maturities = vec![0.25, 0.5, 1.0];

  let result = build_surface_from_model(&pricer, 100.0, 0.05, 0.0, &strikes, &maturities);
  assert_eq!(result.svi_params.len(), 3);

  // Verify prices match direct model
  use crate::traits::ModelPricer;
  let direct = model.price_call(100.0, 100.0, 0.05, 0.0, 1.0);
  let via_trait = pricer.price_call(100.0, 100.0, 0.05, 0.0, 1.0);
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

/// A surface whose every implied vol failed to invert has no grid left to
/// check, and both arbitrage checks pass vacuously over the empty grid. The
/// `bool` this method used to return could not say so; `None` can.
#[test]
fn all_nan_surface_reports_nothing_checkable() {
  // An undiscounted call worth 1e6 against a forward of 100 is outside the
  // no-arbitrage band, so Black inversion fails at every node.
  let prices = Array2::<f64>::from_elem((2, 3), 1e6);
  let result = build_surface(
    vec![90.0, 100.0, 110.0],
    vec![0.5, 1.0],
    vec![100.0, 100.0],
    &prices,
    true,
  );

  let finite = result
    .iv_surface
    .ivs
    .iter()
    .filter(|v| v.is_finite())
    .count();
  assert_eq!(finite, 0, "every node must have failed to invert");

  // The mechanism, pinned: `min_g` is the untouched `+inf` seed on both
  // slices, and the calendar leg still reports clean.
  for (free, min_g) in &result.butterfly_checks {
    assert!(
      *free && min_g.is_infinite(),
      "vacuous butterfly pass expected"
    );
  }
  assert!(
    result.calendar_spread_free,
    "vacuous calendar pass expected"
  );

  assert_eq!(
    result.is_arbitrage_free(),
    None,
    "nothing checkable must not report a clean surface"
  );
}

/// The guard must not fire on a surface that really was checked.
#[test]
fn checkable_surface_still_answers() {
  let (strikes, maturities, forwards, prices) = make_test_prices();
  let result = build_surface(strikes, maturities, forwards, &prices, true);

  for (_, min_g) in &result.butterfly_checks {
    assert!(min_g.is_finite(), "every slice must have been evaluated");
  }
  assert_eq!(result.is_arbitrage_free(), Some(true));
}

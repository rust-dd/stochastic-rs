//! # Model-Generated Implied Volatility Surfaces
//!
//! Generates [`ImpliedVolSurface`] from any calibrated stochastic volatility
//! model by pricing a grid of European calls and inverting via Black-Scholes.
//!
//! Supports all models implementing [`FourierModelExt`] (Heston, VG, NIG,
//! CGMY, MertonJD, Kou, HKDE, regime-switching) via Gil-Pelaez quadrature,
//! plus SABR via the Hagan (2002) analytical formula.

use ndarray::Array2;

use super::implied::ImpliedVolSurface;
use crate::quant::calibration::heston::HestonParams;
use crate::quant::calibration::svj::SVJCalibrationResult;
use crate::quant::pricing::fourier::BatesFourier;
use crate::quant::pricing::fourier::CarrMadanPricer;
use crate::quant::pricing::fourier::FourierModelExt;
use crate::quant::pricing::fourier::GilPelaezPricer;
use crate::quant::pricing::fourier::HestonFourier;
use crate::quant::pricing::heston_stoch_corr::HestonStochCorrPricer;
use crate::quant::pricing::sabr::hagan_implied_vol;

/// Generate an implied vol surface from any [`FourierModelExt`] model
/// (Heston, Variance Gamma, NIG, CGMY, Merton JD, Kou, HKDE, etc.).
///
/// Prices European calls at every (strike, maturity) point via Gil-Pelaez
/// quadrature, then inverts to implied volatility.
///
/// # Arguments
/// * `model_fn` - Closure that builds the Fourier model for a given maturity.
///   This allows term-structure-dependent parameters (e.g., different θ per T).
///   For a static model, just ignore the `tau` argument.
/// * `s` - Spot price
/// * `r` - Risk-free rate
/// * `q` - Dividend yield
/// * `strikes` - Strike grid (ascending)
/// * `maturities` - Maturity grid in years (ascending)
pub fn fourier_model_surface(
  model: &dyn FourierModelExt,
  s: f64,
  r: f64,
  q: f64,
  strikes: &[f64],
  maturities: &[f64],
) -> ImpliedVolSurface {
  let nt = maturities.len();
  let nk = strikes.len();
  let forwards: Vec<f64> = maturities
    .iter()
    .map(|&t| s * ((r - q) * t).exp())
    .collect();

  let mut prices = Array2::<f64>::zeros((nt, nk));

  for (j, &t) in maturities.iter().enumerate() {
    let df = (-r * t).exp();
    for (i, &k) in strikes.iter().enumerate() {
      let call = GilPelaezPricer::price_call(model, s, k, r, q, t);
      let undiscounted = if df > 0.0 { call / df } else { call };
      prices[[j, i]] = undiscounted;
    }
  }

  ImpliedVolSurface::from_prices(
    strikes.to_vec(),
    maturities.to_vec(),
    forwards,
    &prices,
    true,
  )
}

/// Generate an implied vol surface from calibrated Heston parameters.
///
/// Convenience wrapper around [`fourier_model_surface`].
pub fn heston_vol_surface(
  v0: f64,
  kappa: f64,
  theta: f64,
  sigma: f64,
  rho: f64,
  s: f64,
  r: f64,
  q: f64,
  strikes: &[f64],
  maturities: &[f64],
) -> ImpliedVolSurface {
  let model = HestonFourier {
    v0,
    kappa,
    theta,
    sigma,
    rho,
    r,
    q,
  };
  fourier_model_surface(&model, s, r, q, strikes, maturities)
}

/// Generate an implied vol surface directly from [`HestonParams`]
/// (output of [`HestonCalibrator::calibrate()`]).
pub fn heston_params_surface(
  params: &HestonParams,
  s: f64,
  r: f64,
  q: f64,
  strikes: &[f64],
  maturities: &[f64],
) -> ImpliedVolSurface {
  heston_vol_surface(
    params.v0,
    params.kappa,
    params.theta,
    params.sigma,
    params.rho,
    s,
    r,
    q,
    strikes,
    maturities,
  )
}

/// Generate an implied vol surface directly from [`SVJCalibrationResult`]
/// (output of [`SVJCalibrator::calibrate()`]).
pub fn svj_result_surface(
  result: &SVJCalibrationResult,
  s: f64,
  r: f64,
  q: f64,
  strikes: &[f64],
  maturities: &[f64],
) -> ImpliedVolSurface {
  bates_vol_surface(
    result.v0,
    result.kappa,
    result.theta,
    result.sigma_v,
    result.rho,
    result.lambda,
    result.mu_j,
    result.sigma_j,
    s,
    r,
    q,
    strikes,
    maturities,
  )
}

/// Generate an implied vol surface from calibrated SABR parameters.
///
/// Uses Hagan et al. (2002) analytical implied-vol approximation directly
/// (no Fourier pricing needed).
pub fn sabr_vol_surface(
  alpha: f64,
  beta: f64,
  nu: f64,
  rho: f64,
  s: f64,
  r: f64,
  q: f64,
  strikes: &[f64],
  maturities: &[f64],
) -> ImpliedVolSurface {
  let nt = maturities.len();
  let nk = strikes.len();
  let forwards: Vec<f64> = maturities
    .iter()
    .map(|&t| s * ((r - q) * t).exp())
    .collect();

  let mut ivs = Array2::<f64>::from_elem((nt, nk), f64::NAN);
  let mut total_variance = Array2::<f64>::from_elem((nt, nk), f64::NAN);
  let mut log_moneyness = Array2::<f64>::zeros((nt, nk));

  for (j, &t) in maturities.iter().enumerate() {
    let f = forwards[j];
    for (i, &k) in strikes.iter().enumerate() {
      log_moneyness[[j, i]] = (k / f).ln();
      let iv = hagan_implied_vol(k, f, t, alpha, beta, nu, rho);
      if iv.is_finite() && iv > 0.0 {
        ivs[[j, i]] = iv;
        total_variance[[j, i]] = iv * iv * t;
      }
    }
  }

  ImpliedVolSurface {
    strikes: strikes.to_vec(),
    maturities: maturities.to_vec(),
    forwards,
    ivs,
    total_variance,
    log_moneyness,
  }
}

/// Generate an implied vol surface from calibrated Bates/SVJ parameters.
///
/// Heston + Merton log-normal jumps.
#[allow(clippy::too_many_arguments)]
pub fn bates_vol_surface(
  v0: f64,
  kappa: f64,
  theta: f64,
  sigma_v: f64,
  rho: f64,
  lambda: f64,
  mu_j: f64,
  sigma_j: f64,
  s: f64,
  r: f64,
  q: f64,
  strikes: &[f64],
  maturities: &[f64],
) -> ImpliedVolSurface {
  let model = BatesFourier {
    v0,
    kappa,
    theta,
    sigma_v,
    rho,
    lambda,
    mu_j,
    sigma_j,
    r,
    q,
  };
  fourier_model_surface(&model, s, r, q, strikes, maturities)
}

/// Generate an implied vol surface from calibrated Heston with stochastic
/// correlation parameters.
///
/// Uses the Carr-Madan FFT pricer from [`HestonStochCorrPricer`].
#[allow(clippy::too_many_arguments)]
pub fn heston_stoch_corr_vol_surface(
  v0: f64,
  kappa_v: f64,
  theta_v: f64,
  sigma_v: f64,
  rho0: f64,
  kappa_r: f64,
  mu_r: f64,
  sigma_r: f64,
  rho2: f64,
  s: f64,
  r: f64,
  strikes: &[f64],
  maturities: &[f64],
) -> ImpliedVolSurface {
  let nt = maturities.len();
  let nk = strikes.len();
  let forwards: Vec<f64> = maturities.iter().map(|&t| s * (r * t).exp()).collect();

  let mut prices = Array2::<f64>::zeros((nt, nk));

  for (j, &t) in maturities.iter().enumerate() {
    let df = (-r * t).exp();
    for (i, &k) in strikes.iter().enumerate() {
      let pricer = HestonStochCorrPricer::new(
        s, r, k, v0, kappa_v, theta_v, sigma_v, rho0, kappa_r, mu_r, sigma_r, rho2, t,
      );
      let call = pricer.price_call_carr_madan();
      let undiscounted = if df > 0.0 { call / df } else { call };
      prices[[j, i]] = undiscounted;
    }
  }

  ImpliedVolSurface::from_prices(
    strikes.to_vec(),
    maturities.to_vec(),
    forwards,
    &prices,
    true,
  )
}

/// Generate an implied vol surface using Carr-Madan FFT for faster pricing.
///
/// For large grids, FFT is significantly faster than per-strike Gil-Pelaez
/// quadrature since it prices all log-strikes simultaneously.
pub fn fourier_model_surface_fft(
  model: &dyn FourierModelExt,
  s: f64,
  r: f64,
  q: f64,
  strikes: &[f64],
  maturities: &[f64],
) -> ImpliedVolSurface {
  let nt = maturities.len();
  let nk = strikes.len();
  let forwards: Vec<f64> = maturities
    .iter()
    .map(|&t| s * ((r - q) * t).exp())
    .collect();

  let cm = CarrMadanPricer::new(12, 0.75);

  let mut prices = Array2::<f64>::zeros((nt, nk));

  for (j, &t) in maturities.iter().enumerate() {
    let df = (-r * t).exp();
    for (i, &k) in strikes.iter().enumerate() {
      let call = cm.price_call(model, s, k, r, t);
      let undiscounted = if df > 0.0 { call / df } else { call };
      prices[[j, i]] = undiscounted;
    }
  }

  ImpliedVolSurface::from_prices(
    strikes.to_vec(),
    maturities.to_vec(),
    forwards,
    &prices,
    true,
  )
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::quant::pricing::fourier::VarianceGammaFourier;

  #[test]
  fn heston_surface_generation() {
    let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
    let maturities = vec![0.25, 0.5, 1.0];

    let surface = heston_vol_surface(
      0.04,  // v0
      2.0,   // kappa
      0.04,  // theta
      0.3,   // sigma
      -0.7,  // rho
      100.0, // s
      0.05,  // r
      0.0,   // q
      &strikes,
      &maturities,
    );

    for j in 0..maturities.len() {
      for i in 0..strikes.len() {
        let iv = surface.ivs[[j, i]];
        assert!(
          iv.is_finite() && iv > 0.0 && iv < 2.0,
          "Heston IV should be reasonable: iv={iv} at T={}, K={}",
          maturities[j],
          strikes[i]
        );
      }
    }

    // Heston with negative rho should show skew
    let slice = surface.smile_slice(2);
    let atm_idx = slice
      .log_moneyness
      .iter()
      .enumerate()
      .min_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
      .map(|(i, _)| i)
      .unwrap();
    let otm_put_iv = slice.implied_vols[0];
    let atm_iv = slice.implied_vols[atm_idx];
    assert!(
      otm_put_iv > atm_iv,
      "OTM put IV ({otm_put_iv}) should be > ATM IV ({atm_iv}) with rho=-0.7"
    );
  }

  #[test]
  fn sabr_surface_generation() {
    let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
    let maturities = vec![0.25, 0.5, 1.0];

    let surface = sabr_vol_surface(
      0.2,   // alpha
      1.0,   // beta (lognormal)
      0.4,   // nu
      -0.3,  // rho
      100.0, // s
      0.05,  // r
      0.0,   // q
      &strikes,
      &maturities,
    );

    for j in 0..maturities.len() {
      for i in 0..strikes.len() {
        let iv = surface.ivs[[j, i]];
        assert!(
          iv.is_finite() && iv > 0.0,
          "SABR IV should be positive: iv={iv}"
        );
      }
    }
  }

  #[test]
  fn vg_surface_generation() {
    let model = VarianceGammaFourier {
      sigma: 0.12,
      theta: -0.14,
      nu: 0.2,
      r: 0.05,
      q: 0.0,
    };

    let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
    let maturities = vec![0.25, 0.5, 1.0];

    let surface = fourier_model_surface(&model, 100.0, 0.05, 0.0, &strikes, &maturities);

    for j in 0..maturities.len() {
      for i in 0..strikes.len() {
        let iv = surface.ivs[[j, i]];
        assert!(
          iv.is_finite() && iv > 0.0,
          "VG IV should be positive: iv={iv} at T={}, K={}",
          maturities[j],
          strikes[i]
        );
      }
    }
  }

  #[test]
  fn heston_surface_then_ssvi_fit() {
    let strikes: Vec<f64> = (80..=120).step_by(2).map(|k| k as f64).collect();
    let maturities = vec![0.25, 0.5, 1.0, 2.0];

    let surface = heston_vol_surface(
      0.04,
      2.0,
      0.04,
      0.3,
      -0.7,
      100.0,
      0.05,
      0.0,
      &strikes,
      &maturities,
    );

    let ssvi = surface.fit_ssvi(None);
    assert!(ssvi.is_calendar_spread_free());

    let iv_model = surface.ivs[[1, 10]];
    let iv_ssvi = ssvi.implied_vol(surface.log_moneyness[[1, 10]], maturities[1]);
    let err = (iv_model - iv_ssvi).abs();
    assert!(
      err < 0.01,
      "SSVI should fit Heston surface closely: model={iv_model} ssvi={iv_ssvi} err={err}"
    );
  }

  #[test]
  fn bates_surface_generation() {
    let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
    let maturities = vec![0.25, 0.5, 1.0];

    let surface = bates_vol_surface(
      0.04,  // v0
      2.0,   // kappa
      0.04,  // theta
      0.3,   // sigma_v
      -0.7,  // rho
      0.5,   // lambda (jump intensity)
      -0.1,  // mu_j (negative mean jump)
      0.15,  // sigma_j
      100.0, // s
      0.05,  // r
      0.0,   // q
      &strikes,
      &maturities,
    );

    for j in 0..maturities.len() {
      for i in 0..strikes.len() {
        let iv = surface.ivs[[j, i]];
        assert!(
          iv.is_finite() && iv > 0.0 && iv < 2.0,
          "Bates IV should be reasonable: iv={iv} at T={}, K={}",
          maturities[j],
          strikes[i]
        );
      }
    }
  }

  #[test]
  fn heston_stoch_corr_surface_generation() {
    let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
    let maturities = vec![0.5, 1.0];

    let surface = heston_stoch_corr_vol_surface(
      0.04,  // v0
      2.0,   // kappa_v
      0.04,  // theta_v
      0.3,   // sigma_v
      -0.5,  // rho0
      5.0,   // kappa_r
      -0.5,  // mu_r
      0.3,   // sigma_r
      0.1,   // rho2
      100.0, // s
      0.05,  // r
      &strikes,
      &maturities,
    );

    for j in 0..maturities.len() {
      for i in 0..strikes.len() {
        let iv = surface.ivs[[j, i]];
        assert!(
          iv.is_finite() && iv > 0.0 && iv < 2.0,
          "HSCM IV should be reasonable: iv={iv} at T={}, K={}",
          maturities[j],
          strikes[i]
        );
      }
    }
  }
}

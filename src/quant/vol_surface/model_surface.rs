//! # Model-Generated Implied Volatility Surfaces
//!
//! Any calibrated model implementing [`ModelPricer`] automatically gets
//! [`ModelSurface::vol_surface()`] via a blanket implementation that prices
//! a grid of European calls and inverts to implied volatility.
//!
//! All [`FourierModelExt`] models (Heston, Bates, VG, NIG, CGMY, MertonJD,
//! Kou, HKDE) get [`ModelPricer`] via a blanket impl in `fourier.rs`.
//! Non-Fourier models ([`SabrModel`], [`HscmModel`]) have explicit impls.

use ndarray::Array2;

use super::implied::ImpliedVolSurface;
use crate::quant::pricing::fourier::CarrMadanPricer;
use crate::quant::pricing::fourier::FourierModelExt;
use crate::traits::ModelPricer;

/// Trait for generating an implied vol surface from a calibrated model.
///
/// Any [`ModelPricer`] automatically gets this via a blanket implementation.
/// The default prices calls across the (strike, maturity) grid and inverts
/// to implied vol. Models like SABR can override for efficiency.
pub trait ModelSurface: ModelPricer {
  /// Generate an implied vol surface on the given grid.
  fn vol_surface(
    &self,
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
        let call = self.price_call(s, k, r, q, t);
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
}

/// Blanket: every [`ModelPricer`] gets [`ModelSurface`] for free.
impl<T: ModelPricer + ?Sized> ModelSurface for T {}

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
  use crate::quant::pricing::fourier::BatesFourier;
  use crate::quant::pricing::fourier::HestonFourier;
  use crate::quant::pricing::fourier::VarianceGammaFourier;
  use crate::quant::pricing::heston_stoch_corr::HscmModel;
  use crate::quant::pricing::sabr::SabrModel;

  #[test]
  fn heston_via_model_surface() {
    let model = HestonFourier {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma: 0.3,
      rho: -0.7,
      r: 0.05,
      q: 0.0,
    };

    let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
    let maturities = vec![0.25, 0.5, 1.0];
    let surface = model.vol_surface(100.0, 0.05, 0.0, &strikes, &maturities);

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

    let slice = surface.smile_slice(2);
    let atm_idx = slice
      .log_moneyness
      .iter()
      .enumerate()
      .min_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
      .map(|(i, _)| i)
      .unwrap();
    assert!(
      slice.implied_vols[0] > slice.implied_vols[atm_idx],
      "OTM put IV should be > ATM IV with rho=-0.7"
    );
  }

  #[test]
  fn sabr_via_model_surface() {
    let model = SabrModel {
      alpha: 0.2,
      beta: 1.0,
      nu: 0.4,
      rho: -0.3,
    };

    let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
    let maturities = vec![0.25, 0.5, 1.0];
    let surface = model.vol_surface(100.0, 0.05, 0.0, &strikes, &maturities);

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
  fn vg_via_model_surface() {
    let model = VarianceGammaFourier {
      sigma: 0.12,
      theta: -0.14,
      nu: 0.2,
      r: 0.05,
      q: 0.0,
    };

    let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
    let maturities = vec![0.25, 0.5, 1.0];
    let surface = model.vol_surface(100.0, 0.05, 0.0, &strikes, &maturities);

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
  fn bates_via_model_surface() {
    let model = BatesFourier {
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
    let maturities = vec![0.25, 0.5, 1.0];
    let surface = model.vol_surface(100.0, 0.05, 0.0, &strikes, &maturities);

    for j in 0..maturities.len() {
      for i in 0..strikes.len() {
        let iv = surface.ivs[[j, i]];
        assert!(
          iv.is_finite() && iv > 0.0 && iv < 2.0,
          "Bates IV should be reasonable: iv={iv}"
        );
      }
    }
  }

  #[test]
  fn hscm_via_model_surface() {
    let model = HscmModel {
      v0: 0.04,
      kappa_v: 2.0,
      theta_v: 0.04,
      sigma_v: 0.3,
      rho0: -0.5,
      kappa_r: 5.0,
      mu_r: -0.5,
      sigma_r: 0.3,
      rho2: 0.1,
    };

    let strikes = vec![90.0, 95.0, 100.0, 105.0, 110.0];
    let maturities = vec![0.5, 1.0];
    let surface = model.vol_surface(100.0, 0.05, 0.0, &strikes, &maturities);

    for j in 0..maturities.len() {
      for i in 0..strikes.len() {
        let iv = surface.ivs[[j, i]];
        assert!(
          iv.is_finite() && iv > 0.0 && iv < 2.0,
          "HSCM IV should be reasonable: iv={iv}"
        );
      }
    }
  }

  #[test]
  fn model_surface_then_ssvi_fit() {
    let model = HestonFourier {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma: 0.3,
      rho: -0.7,
      r: 0.05,
      q: 0.0,
    };

    let strikes: Vec<f64> = (80..=120).step_by(2).map(|k| k as f64).collect();
    let maturities = vec![0.25, 0.5, 1.0, 2.0];
    let surface = model.vol_surface(100.0, 0.05, 0.0, &strikes, &maturities);

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
}

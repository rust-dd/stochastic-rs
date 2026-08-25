//! # Model-Generated Implied Volatility Surfaces
//!
//! Any calibrated model implementing [`VanillaEuropeanCall`] automatically
//! gets [`ModelSurface::vol_surface()`] via a blanket implementation that
//! prices a grid of European calls and inverts to implied volatility.
//!
//! [`ModelPricer`](crate::traits::ModelPricer) alone is deliberately not
//! enough. The inversion is a European Black inversion, so it is only
//! meaningful for a European vanilla call, and `ModelPricer` covers digital,
//! Asian and American payoffs too — [`VanillaEuropeanCall`] is the narrower
//! bound that says the inversion applies. See that trait for what widening
//! this bound silently produces.
//!
//! All [`FourierModelExt`] models (Heston, Bates, Vg, Nig, Cgmy, MertonJD,
//! Kou, Hkde) get both traits via blanket impls in `fourier/mod.rs`.
//! Non-Fourier models ([`SabrPricer`](crate::pricing::sabr::SabrPricer),
//! [`HestonStochCorrPricer`](crate::pricing::heston_stoch_corr::HestonStochCorrPricer))
//! have explicit impls.

use ndarray::Array2;

use super::implied::ImpliedVolSurface;
use crate::pricing::fourier::CarrMadanPricer;
use crate::pricing::fourier::FourierModelExt;
use crate::traits::VanillaEuropeanCall;

/// Trait for generating an implied vol surface from a calibrated model.
///
/// Every [`VanillaEuropeanCall`] gets this via a blanket implementation. The
/// default prices calls across the (strike, maturity) grid and inverts each
/// through the Black formula. Models like Sabr can override for efficiency.
///
/// The supertrait is the whole point of the bound. A
/// [`ModelPricer`](crate::traits::ModelPricer) that is *not* a European
/// vanilla call has no method here at all:
///
/// ```compile_fail,E0599
/// use stochastic_rs_quant::pricing::CashOrNothingPricer;
/// use stochastic_rs_quant::vol_surface::ModelSurface;
///
/// let model = CashOrNothingPricer::new(10.0, 0.35);
/// let _ = model.vol_surface(100.0, 0.05, 0.0, &[100.0], &[1.0]);
/// ```
///
/// The same call on a model that *is* one compiles, so the failure above is
/// the bound and not a typo:
///
/// ```
/// use stochastic_rs_quant::pricing::fourier::BSMFourier;
/// use stochastic_rs_quant::vol_surface::ModelSurface;
///
/// let model = BSMFourier { sigma: 0.35, r: 0.05, q: 0.0 };
/// let surface = model.vol_surface(100.0, 0.05, 0.0, &[100.0], &[1.0]);
/// assert!((surface.ivs[[0, 0]] - 0.35).abs() < 1e-6);
/// ```
pub trait ModelSurface: VanillaEuropeanCall {
  /// Generate an implied vol surface on the given grid.
  ///
  /// Each maturity's slice is inverted against
  /// [`vanilla_call_forward`](VanillaEuropeanCall::vanilla_call_forward)
  /// rather than an assumed $Se^{(r-q)\tau}$, so a model that carries at
  /// something else inverts against its own forward. An implementor that
  /// reports [`f64::NAN`] there — it is not a European vanilla call at that
  /// query — leaves the whole slice `NaN`, including
  /// [`log_moneyness`](ImpliedVolSurface::log_moneyness).
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
      .map(|&t| self.vanilla_call_forward(s, r, q, t))
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

/// Blanket: every [`VanillaEuropeanCall`] gets [`ModelSurface`] for free.
///
/// This was `impl<T: ModelPricer + ?Sized>` until the pricer population grew
/// past European vanillas; [`VanillaEuropeanCall`] records what the wider
/// bound had been asserting without saying so.
impl<T: VanillaEuropeanCall + ?Sized> ModelSurface for T {}

/// Generate an implied vol surface using Carr-Madan FFT for faster pricing
/// with the default `(N_pow2 = 12, alpha = 0.75)` damping settings.
///
/// For large grids, FFT is significantly faster than per-strike Gil-Pelaez
/// quadrature since it prices all log-strikes simultaneously.
///
/// To override the FFT grid size or damping factor, use
/// [`fourier_model_surface_fft_with`] which exposes both as parameters.
pub fn fourier_model_surface_fft(
  model: &impl FourierModelExt,
  s: f64,
  r: f64,
  q: f64,
  strikes: &[f64],
  maturities: &[f64],
) -> ImpliedVolSurface {
  fourier_model_surface_fft_with(model, s, r, q, strikes, maturities, 12, 0.75)
}

/// Generate an implied vol surface using Carr-Madan FFT with caller-supplied
/// FFT settings.
///
/// - `n_pow2`: FFT grid is `2^n_pow2` log-strike points. Higher gives a denser
///   strike grid and a wider strike range (so deep wings hit-the-grid less
///   often) at the cost of an O(N log N) FFT. Typical: 12 (4096) or 13 (8192).
/// - `alpha`: Carr-Madan damping factor. Must be > 0. Lower values reduce
///   the integrand's contribution near `u=0`; higher values pull the
///   integration tail in. Typical: 0.75 (the Lord-Kahl 2010 sweet spot for
///   moderate-vol equity smiles); use 0.25-0.5 for low-vol smiles, 1.0-1.5
///   for high-skew rough-vol surfaces.
pub fn fourier_model_surface_fft_with(
  model: &impl FourierModelExt,
  s: f64,
  r: f64,
  q: f64,
  strikes: &[f64],
  maturities: &[f64],
  n_pow2: usize,
  alpha: f64,
) -> ImpliedVolSurface {
  let nt = maturities.len();
  let nk = strikes.len();
  let forwards: Vec<f64> = maturities
    .iter()
    .map(|&t| s * ((r - q) * t).exp())
    .collect();

  let cm = CarrMadanPricer::new(n_pow2, alpha);

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
  use crate::pricing::fourier::BatesFourier;
  use crate::pricing::fourier::DoubleHestonFourier;
  use crate::pricing::fourier::HestonFourier;
  use crate::pricing::fourier::VarianceGammaFourier;
  use crate::OptionStyle;
  use crate::pricing::bsm::BSMCoc;
  use crate::pricing::bsm::BSMPricer;
  use crate::pricing::finite_difference::FiniteDifferenceMethod;
  use crate::pricing::finite_difference::FiniteDifferencePricer;
  use crate::pricing::heston_stoch_corr::HestonStochCorrPricer;
  use crate::pricing::sabr::SabrPricer;

  const GRID_K: [f64; 5] = [90.0, 95.0, 100.0, 105.0, 110.0];
  const GRID_T: [f64; 2] = [0.25, 1.0];

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
    let model = SabrPricer {
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
          "Sabr IV should be positive: iv={iv}"
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
          "Vg IV should be positive: iv={iv} at T={}, K={}",
          maturities[j],
          strikes[i]
        );
      }
    }
  }

  #[test]
  fn double_heston_via_model_surface() {
    let model = DoubleHestonFourier {
      v1_0: 0.02,
      kappa1: 3.0,
      theta1: 0.02,
      sigma1: 0.4,
      rho1: -0.6,
      v2_0: 0.02,
      kappa2: 0.5,
      theta2: 0.03,
      sigma2: 0.2,
      rho2: -0.3,
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
          "Double Heston IV should be reasonable: iv={iv} at T={}, K={}",
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
      "Double Heston OTM put IV should be > ATM IV with negative rho1/rho2"
    );
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
    let model = HestonStochCorrPricer {
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
    let ks: Vec<f64> = (-3..=3).map(|i| i as f64 * 0.5).collect();
    assert!(ssvi.is_calendar_spread_free(&ks));

    let iv_model = surface.ivs[[1, 10]];
    let iv_ssvi = ssvi.implied_vol(surface.log_moneyness[[1, 10]], maturities[1]);
    let err = (iv_model - iv_ssvi).abs();
    assert!(
      err < 0.01,
      "SSVI should fit Heston surface closely: model={iv_model} ssvi={iv_ssvi} err={err}"
    );
  }

  /// The default forward is the literal expression `vol_surface` used to
  /// inline, so every model that does not override the hook keeps the
  /// surface it had. `assert_eq!` on `f64` is the point: "equal to within a
  /// tolerance" would not distinguish a re-association that moves an ulp
  /// from the expression itself.
  #[test]
  fn default_forward_is_bit_identical_to_the_inlined_expression() {
    let heston = HestonFourier {
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      sigma: 0.3,
      rho: -0.7,
      r: 0.05,
      q: 0.0,
    };
    let sabr = SabrPricer {
      alpha: 0.2,
      beta: 1.0,
      nu: 0.4,
      rho: -0.3,
    };

    for &(s, r, q) in &[
      (100.0, 0.05, 0.0),
      (87.5, 0.031, 0.017),
      (250.0, -0.004, 0.02),
      (1.234_5, 0.0, 0.0),
    ] {
      for &t in &[0.25_f64, 1.0, 3.0, 7.5] {
        let expected = s * ((r - q) * t).exp();
        assert_eq!(heston.vanilla_call_forward(s, r, q, t), expected);
        assert_eq!(sabr.vanilla_call_forward(s, r, q, t), expected);
        // The two cost-of-carry conventions with `b = r - q` must land on
        // the same expression rather than merely near it.
        for coc in [BSMCoc::Merton1973, BSMCoc::GarmanKohlhagen1983] {
          assert_eq!(
            BSMPricer::new(0.2, coc).vanilla_call_forward(s, r, q, t),
            expected
          );
        }
      }
    }
  }

  /// A flat-volatility model must invert to a flat surface. Under
  /// [`BSMCoc::Black1976`] the carry is `b = 0`, so the forward is `s` — and
  /// against the default `s * exp((r - q) * tau)` these same *correct* prices
  /// invert to a smile running 0.080 to 0.150 across this grid, every point
  /// finite and none of them 0.20. That is what
  /// [`BSMPricer::vanilla_call_forward`] exists to prevent, and deleting the
  /// override fails this test rather than degrading it.
  #[test]
  fn futures_carry_surface_recovers_the_models_flat_vol() {
    for coc in [BSMCoc::Black1976, BSMCoc::Asay1982] {
      let model = BSMPricer::new(0.20, coc);
      let surface = model.vol_surface(100.0, 0.05, 0.0, &GRID_K, &GRID_T);

      for (j, &t) in GRID_T.iter().enumerate() {
        assert_eq!(surface.forwards[j], 100.0, "{coc:?} forward at T={t}");
        for (i, &k) in GRID_K.iter().enumerate() {
          let iv = surface.ivs[[j, i]];
          assert!(
            (iv - 0.20).abs() < 1e-6,
            "{coc:?} should invert flat: iv={iv} at T={t}, K={k}"
          );
        }
      }
    }
  }

  /// The same failure with the dividend yield rather than the carry
  /// convention: [`BSMCoc::Bsm1973`] carries at `b = r` and ignores `q`, so a
  /// surface asked for at `q = 0.03` must still invert flat. Against the
  /// default forward it produces a 0.274-to-0.233 skew out of a model that
  /// has none.
  #[test]
  fn dividend_ignoring_carry_surface_recovers_the_models_flat_vol() {
    let model = BSMPricer::new(0.20, BSMCoc::Bsm1973);
    let surface = model.vol_surface(100.0, 0.05, 0.03, &GRID_K, &GRID_T);

    for (j, &t) in GRID_T.iter().enumerate() {
      for (i, &k) in GRID_K.iter().enumerate() {
        let iv = surface.ivs[[j, i]];
        assert!(
          (iv - 0.20).abs() < 1e-6,
          "Bsm1973 at q=0.03 should invert flat: iv={iv} at T={t}, K={k}"
        );
      }
    }
  }

  /// [`FiniteDifferencePricer`] is the one implementor whose exercise style
  /// is a field, so it is the one that has to answer per instance. At
  /// [`OptionStyle::American`] the whole surface must be `NaN`.
  ///
  /// Without the `NaN` this grid comes back 10/10 finite, every point within
  /// 0.008 of the model's own `v` — an American price pushed through a
  /// European inversion looks like a slightly noisy European one, which is
  /// exactly why a `NaN` and not a warning.
  #[test]
  fn american_finite_difference_surface_is_all_nan() {
    let model = FiniteDifferencePricer::new(
      0.25,
      200,
      100,
      OptionStyle::American,
      FiniteDifferenceMethod::CrankNicolson,
    );
    let surface = model.vol_surface(100.0, 0.05, 0.06, &GRID_K, &GRID_T);

    for (j, &t) in GRID_T.iter().enumerate() {
      assert!(surface.forwards[j].is_nan(), "forward at T={t}");
      for (i, &k) in GRID_K.iter().enumerate() {
        assert!(
          surface.ivs[[j, i]].is_nan(),
          "iv at T={t}, K={k} is {}",
          surface.ivs[[j, i]]
        );
        assert!(surface.total_variance[[j, i]].is_nan(), "w at T={t}, K={k}");
        assert!(surface.log_moneyness[[j, i]].is_nan(), "k at T={t}, K={k}");
      }
    }
  }

  /// Control for `american_finite_difference_surface_is_all_nan`: the same
  /// solver at [`OptionStyle::European`] still produces a surface, so the
  /// `NaN` above is the exercise style and not the type.
  #[test]
  fn european_finite_difference_surface_is_finite() {
    let model = FiniteDifferencePricer::new(
      0.25,
      200,
      100,
      OptionStyle::European,
      FiniteDifferenceMethod::CrankNicolson,
    );
    let surface = model.vol_surface(100.0, 0.05, 0.06, &GRID_K, &GRID_T);

    for (j, &t) in GRID_T.iter().enumerate() {
      assert_eq!(surface.forwards[j], 100.0 * ((0.05 - 0.06) * t).exp());
      for (i, &k) in GRID_K.iter().enumerate() {
        let iv = surface.ivs[[j, i]];
        assert!(
          (iv - 0.25).abs() < 5e-3,
          "European FD should recover its own v: iv={iv} at T={t}, K={k}"
        );
      }
    }
  }
}

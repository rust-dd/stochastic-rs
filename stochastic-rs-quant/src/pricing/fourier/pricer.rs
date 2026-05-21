//! Carr-Madan FFT, Gil-Pelaez and Lewis quadrature pricers.

use std::f64::consts::FRAC_1_PI;
use std::f64::consts::PI;

use ndarray::Array1;
use ndrustfft::FftHandler;
use ndrustfft::ndfft;
use num_complex::Complex64;
use quadrature::double_exponential;

use super::FourierModelExt;

/// Carr–Madan FFT pricer.
///
/// Evaluates call prices across a grid of strikes with a single FFT pass.
#[derive(Debug, Clone)]
pub struct CarrMadanPricer {
  /// Number of FFT points (must be a power of 2).
  pub n: usize,
  /// Dampening parameter (typically 0.75 for calls).
  pub alpha: f64,
  /// Integration step size (default 0.25).
  pub eta: f64,
}

impl Default for CarrMadanPricer {
  fn default() -> Self {
    Self {
      n: 4096,
      alpha: 0.75,
      eta: 0.25,
    }
  }
}

impl CarrMadanPricer {
  pub fn new(n: usize, alpha: f64) -> Self {
    assert!(n.is_power_of_two());
    Self {
      n,
      alpha,
      eta: 0.25,
    }
  }

  /// Build a CarrMadan pricer with grid width sized via the model's cumulants
  /// (Lord-Kahl 2007 / Andersen-Andreasen 2002 cumulant-based grid sizing).
  ///
  /// The current Carr-Madan FFT implementation centres the log-strike grid at
  /// $\ln K = 0$ with half-width $L_{grid} = n \lambda / 2 = \pi / \eta$.
  /// This routine adjusts $\eta$ so that $L_{grid} \ge \ln S_0 + L_{factor}
  /// \cdot \sqrt{|c_2| + \sqrt{|c_4|}}$ — i.e., wide enough to bracket the
  /// at-the-money log-strike $\ln S_0$ plus an `L_factor` (default 12) tail
  /// buffer scaled by the model's second/fourth cumulants at maturity $t$.
  ///
  /// Skewed models (Heston/Bates with large |ρ|, KOU/HKDE with strong jump
  /// bias) and very long maturities produce wider distributions; the default
  /// symmetric grid is fine for typical equity options but can crop the wings
  /// for long-dated FX risk-reversals or deep-OTM catastrophe options. This
  /// builder produces a grid sized to keep the relevant strikes inside.
  ///
  /// Falls back to the default `(n=4096, η=0.25)` when cumulants are not
  /// finite. `alpha` is left at the default damping coefficient (0.75).
  ///
  /// # Example
  /// ```ignore
  /// let pricer = CarrMadanPricer::cumulant_sized(&heston, /* t */ 5.0, /* s */ 100.0, 12.0);
  /// let price = pricer.price_call(&heston, 100.0, 100.0, 0.05, 5.0);
  /// ```
  pub fn cumulant_sized(model: &impl FourierModelExt, t: f64, s: f64, l_factor: f64) -> Self {
    let cumulants = model.cumulants(t);
    if !cumulants.c2.is_finite() || cumulants.c2 <= 0.0 || !s.is_finite() || s <= 0.0 {
      return Self::default();
    }
    let c4_term = if cumulants.c4.is_finite() && cumulants.c4 >= 0.0 {
      cumulants.c4.sqrt()
    } else {
      0.0
    };
    let cumulant_buffer = l_factor * (cumulants.c2.abs() + c4_term).sqrt();
    let required_half_width = s.ln().abs() + cumulant_buffer;
    if !required_half_width.is_finite() || required_half_width <= 0.0 {
      return Self::default();
    }
    let n = 4096_usize;
    let eta = PI / required_half_width;
    Self {
      n,
      alpha: 0.75,
      eta: eta.max(0.001),
    }
  }

  /// Compute call prices on the FFT strike grid.
  ///
  /// Returns `(log_strikes, call_prices)` where `log_strikes` are
  /// $k_u = b + \lambda u$.
  pub fn price_call_surface(
    &self,
    model: &impl FourierModelExt,
    s: f64,
    r: f64,
    t: f64,
  ) -> (Array1<f64>, Array1<f64>) {
    let n = self.n;
    let alpha = self.alpha;
    let eta = self.eta;
    let lambda = 2.0 * PI / (n as f64 * eta);
    let b = -0.5 * n as f64 * lambda;

    let i_unit = Complex64::i();
    let ln_s = s.ln();
    let disc = (-r * t).exp();

    let mut input = Array1::<Complex64>::zeros(n);

    for j in 0..n {
      let v_j = eta * j as f64;

      let simpson = if j == 0 {
        eta / 3.0
      } else if j % 2 == 1 {
        eta * 4.0 / 3.0
      } else {
        eta * 2.0 / 3.0
      };

      let xi = Complex64::new(v_j, -(alpha + 1.0));
      let phi = model.chf(t, xi) * (i_unit * xi * ln_s).exp();

      let denom = Complex64::new(alpha * alpha + alpha - v_j * v_j, (2.0 * alpha + 1.0) * v_j);

      let psi = disc * phi / denom;

      input[j] = (i_unit * v_j * b).exp() * psi * simpson;
    }

    let handler = FftHandler::<f64>::new(n);
    let mut output = Array1::<Complex64>::zeros(n);
    ndfft(&input, &mut output, &handler, 0);

    let mut log_strikes = Array1::<f64>::zeros(n);
    let mut prices = Array1::<f64>::zeros(n);
    for u in 0..n {
      let k_u = b + lambda * u as f64;
      log_strikes[u] = k_u;
      prices[u] = ((-alpha * k_u).exp() * output[u].re / PI).max(0.0);
    }

    (log_strikes, prices)
  }

  /// Price a single call option by interpolating the FFT surface.
  ///
  /// **Out-of-grid strikes:** when `k` is so deep ITM/OTM that `ln(k)` falls
  /// outside `[log_strikes.first(), log_strikes.last()]`, returns
  /// `f64::NAN`. The previous v2.0.0-rc.0 behavior returned `0.0`, which
  /// produced silent-zero residuals at the wings of any calibration that
  /// stretched past the FFT grid (catastrophe options, very deep ITM/OTM FX
  /// risk-reversals). NaN propagation forces calibration loops to detect
  /// the issue rather than carry on with a falsely-perfect zero residual.
  /// Use [`Self::strike_in_grid`] to test before pricing, or construct the
  /// pricer with a larger `n` / smaller `eta` for wider coverage.
  pub fn price_call(&self, model: &impl FourierModelExt, s: f64, k: f64, r: f64, t: f64) -> f64 {
    let (log_strikes, prices) = self.price_call_surface(model, s, r, t);
    let target = k.ln();
    let n = log_strikes.len();

    if target < log_strikes[0] || target > log_strikes[n - 1] {
      return f64::NAN;
    }

    for i in 0..n - 1 {
      if log_strikes[i] <= target && target <= log_strikes[i + 1] {
        let w = (target - log_strikes[i]) / (log_strikes[i + 1] - log_strikes[i]);
        return prices[i] * (1.0 - w) + prices[i + 1] * w;
      }
    }
    unreachable!("CarrMadan interpolation fall-through despite grid bracketing");
  }

  /// Returns `true` if a strike `k` falls inside the FFT log-strike grid for
  /// the given model / market state. Use this before [`Self::price_call`] to
  /// detect / extend the grid rather than accept a NaN.
  pub fn strike_in_grid(
    &self,
    model: &impl FourierModelExt,
    s: f64,
    k: f64,
    r: f64,
    t: f64,
  ) -> bool {
    let (log_strikes, _) = self.price_call_surface(model, s, r, t);
    let target = k.ln();
    let n = log_strikes.len();
    target >= log_strikes[0] && target <= log_strikes[n - 1]
  }

  /// Price a single put option via put-call parity.
  pub fn price_put(
    &self,
    model: &impl FourierModelExt,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    t: f64,
  ) -> f64 {
    let call = self.price_call(model, s, k, r, t);
    call - s * (-q * t).exp() + k * (-r * t).exp()
  }
}

/// Gil-Pelaez (1951) quadrature pricer.
///
/// Uses numerical quadrature on the two half-integrals $P_1,P_2$.
pub struct GilPelaezPricer;

impl GilPelaezPricer {
  /// Price a European call using double-exponential quadrature.
  pub fn price_call(model: &impl FourierModelExt, s: f64, k: f64, r: f64, q: f64, t: f64) -> f64 {
    let ln_ks = (k / s).ln();

    let i_unit = Complex64::i();

    let integrand_p1 = |u: f64| -> f64 {
      if u.abs() < 1e-12 {
        return 0.0;
      }
      let xi = Complex64::new(u, -1.0);
      let phi = model.chf(t, xi);
      let phi_neg_i = model.chf(t, Complex64::new(0.0, -1.0));
      if phi_neg_i.norm() < 1e-30 {
        return 0.0;
      }
      let kernel = (-i_unit * u * ln_ks).exp() / (i_unit * u);
      (kernel * phi / phi_neg_i).re
    };

    let integrand_p2 = |u: f64| -> f64 {
      if u.abs() < 1e-12 {
        return 0.0;
      }
      let xi = Complex64::new(u, 0.0);
      let phi = model.chf(t, xi);
      let kernel = (-i_unit * u * ln_ks).exp() / (i_unit * u);
      (kernel * phi).re
    };

    let p1 =
      0.5 + FRAC_1_PI * double_exponential::integrate(integrand_p1, 1e-8, 100.0, 1e-8).integral;
    let p2 =
      0.5 + FRAC_1_PI * double_exponential::integrate(integrand_p2, 1e-8, 100.0, 1e-8).integral;

    let call = s * (-q * t).exp() * p1 - k * (-r * t).exp() * p2;
    call.max(0.0)
  }
}

/// Lewis (2001) single-strike quadrature pricer.
///
/// $$
/// C=S F e^{-rT}-\frac{\sqrt{SK}\,e^{-rT}}{\pi}
/// \int_0^\infty\!\operatorname{Re}\!\left[\frac{\phi_T(u-\tfrac i2)\,e^{-iu\ln(K/S)}}{u^2+\tfrac14}\right]du
/// $$
pub struct LewisPricer;

impl LewisPricer {
  pub fn price_call(model: &impl FourierModelExt, s: f64, k: f64, r: f64, q: f64, t: f64) -> f64 {
    let ln_ks = (k / s).ln();
    let disc = (-r * t).exp();
    let fwd_factor = ((r - q) * t).exp();
    let sqrt_sk = (s * k).sqrt();

    let i_unit = Complex64::i();

    let integrand = |u: f64| -> f64 {
      let xi = Complex64::new(u, -0.5);
      let phi = model.chf(t, xi);
      let kernel = (-i_unit * u * ln_ks).exp() / (u * u + 0.25);
      (kernel * phi).re
    };

    let integral = double_exponential::integrate(integrand, 1e-8, 100.0, 1e-8).integral;

    let call = s * disc * fwd_factor - sqrt_sk * disc * FRAC_1_PI * integral;
    call.max(0.0)
  }
}

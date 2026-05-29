//! Fractional FFT (Bailey-Swarztrauber 1991) and the FRFT-based Carr-Madan
//! pricer with true grid centring.
//!
//! The classic [`super::pricer::CarrMadanPricer`] is bound by the FFT
//! reciprocity $\lambda\eta = 2\pi/N$ — the log-strike step $\lambda$ and
//! the integration step $\eta$ cannot be chosen independently, and the
//! grid is forced to centre on $\ln K = 0$. The **fractional FFT**
//! evaluates the chirp sum
//!
//! $$
//! G_k = \sum_{j=0}^{N-1} x_j\,e^{-2\pi i \gamma j k}, \qquad k = 0,\dots,N-1
//! $$
//!
//! for an **arbitrary** $\gamma$ (the standard DFT is the special case
//! $\gamma = 1/N$). Decoupling $\gamma = \eta\lambda/(2\pi)$ lets us pick a
//! fine integration step $\eta$ and an independent strike step $\lambda$,
//! and shift the grid to centre on the model-implied log-asset mean
//! $\mu_T = \ln S_0 + c_1$ (Lord-Kahl 2010 §4). Centring on the forward
//! puts the FFT resolution where the option is price-sensitive, so a
//! centred grid reaches the same accuracy as a much larger 0-centred grid.
//!
//! ## Algorithm (Bailey-Swarztrauber 1991)
//!
//! Using $jk = \tfrac{1}{2}(j^2 + k^2 - (k-j)^2)$ the chirp sum becomes a
//! convolution:
//!
//! $$
//! G_k = e^{-i\pi\gamma k^2}\sum_{j} \bigl(x_j\,e^{-i\pi\gamma j^2}\bigr)\,
//!       e^{i\pi\gamma (k-j)^2},
//! $$
//!
//! i.e. the linear convolution of $y_j = x_j e^{-i\pi\gamma j^2}$ with the
//! kernel $h_m = e^{i\pi\gamma m^2}$, evaluated through three length-$2N$
//! FFTs (zero-pad $y$, wrap $h$ symmetrically since $h_{-m} = h_m$).
//!
//! References:
//! - Bailey, D.H., Swarztrauber, P.N. (1991), "The fractional Fourier
//!   transform and applications", *SIAM Review* 33(3), 389-404.
//! - Chourdakis, K. (2005), "Option pricing using the fractional FFT",
//!   *Journal of Computational Finance* 8(2), 1-18.
//! - Lord, R., Kahl, C. (2010), "Complex logarithms in Heston-like
//!   models", *Mathematical Finance* 20(4), §4 (grid centring).

use std::f64::consts::PI;

use ndarray::Array1;
use ndrustfft::FftHandler;
use ndrustfft::ndfft;
use ndrustfft::ndifft;
use num_complex::Complex64;

use super::FourierModelExt;

/// Fractional Fourier transform $G_k = \sum_j x_j e^{-2\pi i \gamma jk}$.
///
/// `x.len()` must be a power of two. Recovers the standard forward DFT
/// when `gamma == 1.0 / x.len() as f64`. Computed via the
/// Bailey-Swarztrauber three-FFT convolution of length `2 * x.len()`.
pub fn frft(x: &Array1<Complex64>, gamma: f64) -> Array1<Complex64> {
  let n = x.len();
  assert!(n.is_power_of_two(), "FRFT length must be a power of two");
  // n power of two ⟹ m = 2n power of two, so the inner FFTs stay fast.
  let m = 2 * n;

  let mut y = Array1::<Complex64>::zeros(m);
  let mut z = Array1::<Complex64>::zeros(m);
  for j in 0..n {
    let jj = j as f64;
    let phase = PI * jj * jj * gamma;
    y[j] = x[j] * Complex64::new(0.0, -phase).exp();
    z[j] = Complex64::new(0.0, phase).exp();
  }
  // Kernel wrap-around: h_{-m} = h_m (m² is even), so the negative lags
  // live in the upper half of the length-2N circular buffer.
  for j in 1..n {
    let jj = j as f64;
    let phase = PI * jj * jj * gamma;
    z[m - j] = Complex64::new(0.0, phase).exp();
  }

  let handler = FftHandler::<f64>::new(m);
  let mut fy = Array1::<Complex64>::zeros(m);
  let mut fz = Array1::<Complex64>::zeros(m);
  ndfft(&y, &mut fy, &handler, 0);
  ndfft(&z, &mut fz, &handler, 0);
  let prod = &fy * &fz;
  let mut conv = Array1::<Complex64>::zeros(m);
  ndifft(&prod, &mut conv, &handler, 0);

  let mut out = Array1::<Complex64>::zeros(n);
  for k in 0..n {
    let kk = k as f64;
    let phase = PI * kk * kk * gamma;
    out[k] = Complex64::new(0.0, -phase).exp() * conv[k];
  }
  out
}

/// FRFT-based Carr-Madan pricer with grid centring on the model-implied
/// log-asset mean. Unlike [`super::pricer::CarrMadanPricer`] the
/// integration step `eta` and the log-strike step `lambda` are
/// independent.
#[derive(Debug, Clone)]
pub struct FrftCarrMadanPricer {
  /// Number of FFT points (must be a power of two).
  pub n: usize,
  /// Carr-Madan damping parameter (typically 0.75 for calls).
  pub alpha: f64,
  /// Integration step $\eta$ — controls the frequency-domain resolution
  /// and the truncation range $N\eta$.
  pub eta: f64,
  /// Log-strike step $\lambda$ — controls the strike-grid resolution,
  /// chosen independently of `eta`.
  pub lambda: f64,
}

impl Default for FrftCarrMadanPricer {
  fn default() -> Self {
    // A ±3 log-strike span around the forward (e^{±3} ≈ [0.05·S, 20·S])
    // at 4096 points with a fine η = 0.25 integration step.
    Self {
      n: 4096,
      alpha: 0.75,
      eta: 0.25,
      lambda: 6.0 / 4096.0,
    }
  }
}

impl FrftCarrMadanPricer {
  /// Construct with explicit integration / strike steps.
  pub fn new(n: usize, alpha: f64, eta: f64, lambda: f64) -> Self {
    assert!(n.is_power_of_two(), "n must be a power of two");
    assert!(eta > 0.0 && lambda > 0.0, "eta, lambda must be positive");
    Self {
      n,
      alpha,
      eta,
      lambda,
    }
  }

  /// Cumulant-sized constructor (Lord-Kahl 2010): the log-strike half-width
  /// is $L_{\text{factor}}\sqrt{|c_2| + \sqrt{|c_4|}}$, so the grid spans
  /// $[\mu_T - \text{hw}, \mu_T + \text{hw}]$ centred on the forward. The
  /// integration step `eta` stays fine and fixed (the FRFT decouples it
  /// from the strike step), giving an $N\eta$ truncation range that fully
  /// captures the Heston / Bates / Lévy integrand tail.
  ///
  /// Falls back to [`Default`] when the cumulants are not finite.
  pub fn cumulant_sized(model: &impl FourierModelExt, t: f64, l_factor: f64) -> Self {
    let cumulants = model.cumulants(t);
    if !cumulants.c2.is_finite() || cumulants.c2 <= 0.0 {
      return Self::default();
    }
    let c4_term = if cumulants.c4.is_finite() && cumulants.c4 >= 0.0 {
      cumulants.c4.sqrt()
    } else {
      0.0
    };
    let half_width = l_factor * (cumulants.c2.abs() + c4_term).sqrt();
    if !half_width.is_finite() || half_width <= 0.0 {
      return Self::default();
    }
    let n = 4096_usize;
    // Integration step: the FRFT decouples η from the strike step, so we
    // pick η to give a frequency truncation range Nη that comfortably
    // covers the integrand tail (negligible past v* where the Gaussian
    // core e^{-½ c₂ v²} ≈ e^{-40}, i.e. v* = √(80/c₂)), then take a 2×
    // safety margin. Finer η than the classic reciprocity rule yields a
    // markedly tighter Simpson quadrature near the money.
    let v_max = 2.0 * (80.0 / cumulants.c2.abs()).sqrt();
    let eta = (v_max / n as f64).clamp(0.005, 0.25);
    Self {
      n,
      alpha: 0.75,
      eta,
      lambda: 2.0 * half_width / n as f64,
    }
  }

  /// Compute call prices on the centred FRFT strike grid. Returns
  /// `(log_strikes, call_prices)` with `log_strikes` centred on
  /// $\mu_T = \ln S_0 + c_1$.
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
    let lambda = self.lambda;
    let gamma = eta * lambda / (2.0 * PI);

    let cumulants = model.cumulants(t);
    let ln_s = s.ln();
    // Centre the log-strike grid on the model-implied log-asset mean.
    let mu_t = ln_s + cumulants.c1;
    let b = mu_t - 0.5 * n as f64 * lambda;

    let i_unit = Complex64::i();
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
      // e^{-i v_j b}: the Carr-Madan kernel is e^{-i v_j k_u} =
      // e^{-i v_j b} · e^{-2πi γ j u}, so the b-phase carries a negative
      // sign. (The 0-centred standard FFT pricer can use +i v_j b because
      // a symmetric grid makes the real part sign-insensitive; a forward-
      // shifted grid does not, so the sign must be correct here.)
      input[j] = (-i_unit * v_j * b).exp() * psi * simpson;
    }

    let output = frft(&input, gamma);

    let mut log_strikes = Array1::<f64>::zeros(n);
    let mut prices = Array1::<f64>::zeros(n);
    for u in 0..n {
      let k_u = b + lambda * u as f64;
      log_strikes[u] = k_u;
      prices[u] = ((-alpha * k_u).exp() * output[u].re / PI).max(0.0);
    }
    (log_strikes, prices)
  }

  /// Price a single call by interpolating the centred FRFT surface.
  /// Returns `f64::NAN` for strikes outside the grid (same truncation
  /// contract as [`super::pricer::CarrMadanPricer::price_call`]).
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
    unreachable!("FRFT Carr-Madan interpolation fall-through despite grid bracketing");
  }

  /// Price a single put via put-call parity.
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

#[cfg(test)]
mod tests {
  use num_complex::Complex64;

  use super::super::bsm::BSMFourier;
  use super::super::heston::HestonFourier;
  use super::super::pricer::GilPelaezPricer;
  use super::*;

  /// FRFT with γ = 1/N must reproduce the standard forward DFT.
  #[test]
  fn frft_reduces_to_dft_at_unit_gamma() {
    let n = 16usize;
    let x = Array1::from_shape_fn(n, |j| Complex64::new(j as f64, 0.5 * j as f64));
    let frft_out = frft(&x, 1.0 / n as f64);
    // Direct DFT.
    let mut dft = Array1::<Complex64>::zeros(n);
    for (k, slot) in dft.iter_mut().enumerate() {
      let mut acc = Complex64::new(0.0, 0.0);
      for j in 0..n {
        let ang = -2.0 * std::f64::consts::PI * (j * k) as f64 / n as f64;
        acc += x[j] * Complex64::new(0.0, ang).exp();
      }
      *slot = acc;
    }
    for k in 0..n {
      assert!(
        (frft_out[k] - dft[k]).norm() < 1e-9,
        "FRFT(γ=1/N)[{k}] = {} vs DFT = {}",
        frft_out[k],
        dft[k]
      );
    }
  }

  /// FRFT Carr-Madan must match the analytic Black-Scholes call. The
  /// reference value 7.94871378854164 is the same one the standard
  /// Carr-Madan / Lewis / Gil-Pelaez tests pin. The centred grid should
  /// hit it far tighter than the 0-centred standard pricer.
  #[test]
  fn frft_carr_madan_bsm_reference() {
    let model = BSMFourier {
      sigma: 0.15,
      r: 0.05,
      q: 0.01,
    };
    let pricer = FrftCarrMadanPricer::cumulant_sized(&model, 1.0, 12.0);
    let price = pricer.price_call(&model, 100.0, 100.0, 0.05, 1.0);
    let expected = 7.94871378854164;
    assert!(
      (price - expected).abs() < 1e-4,
      "FRFT Carr-Madan BSM: got={price}, expected={expected}"
    );
  }

  /// Centred FRFT must agree with Gil-Pelaez (the quadrature reference)
  /// for Heston across a strike ladder, tighter than the loose Fourier
  /// tolerance the 0-centred FFT needs.
  #[test]
  fn frft_carr_madan_heston_matches_gil_pelaez() {
    let model = HestonFourier {
      v0: 0.04,
      kappa: 1.5,
      theta: 0.04,
      sigma: 0.3,
      rho: -0.7,
      r: 0.03,
      q: 0.0,
    };
    let s = 100.0;
    let r = 0.03;
    let t = 1.0;
    let pricer = FrftCarrMadanPricer::cumulant_sized(&model, t, 12.0);
    for k in [80.0, 90.0, 100.0, 110.0, 120.0] {
      let frft_price = pricer.price_call(&model, s, k, r, t);
      let gp_price = GilPelaezPricer::price_call(&model, s, k, r, 0.0, t);
      assert!(
        (frft_price - gp_price).abs() < 5e-2,
        "Heston K={k}: FRFT={frft_price} vs Gil-Pelaez={gp_price}"
      );
    }
  }

  /// Out-of-grid strikes return NaN, not a silent zero.
  #[test]
  fn frft_out_of_grid_returns_nan() {
    let model = BSMFourier {
      sigma: 0.15,
      r: 0.05,
      q: 0.0,
    };
    let pricer = FrftCarrMadanPricer::cumulant_sized(&model, 1.0, 12.0);
    let deep = pricer.price_call(&model, 100.0, 1e12, 0.05, 1.0);
    assert!(deep.is_nan(), "out-of-grid strike must be NaN, got {deep}");
  }
}

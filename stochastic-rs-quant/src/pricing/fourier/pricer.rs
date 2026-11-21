//! The Carr–Madan FFT pricer.

use std::f64::consts::PI;

use ndarray::Array1;
use ndrustfft::FftHandler;
use ndrustfft::ndfft;
use num_complex::Complex64;

use super::FourierModelExt;

/// Floors quadrature or FFT round-off below zero at zero and lets a NaN through, which
/// `f64::max` would turn into a plausible `0.0`.
#[inline]
pub(super) fn floor_price(x: f64) -> f64 {
  if x.is_nan() { x } else { x.max(0.0) }
}

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
    assert!(
      n.is_power_of_two(),
      "n must satisfy `n.is_power_of_two()`, got n = {n:?}"
    );
    Self {
      n,
      alpha,
      eta: 0.25,
    }
  }

  /// The default grid, with `eta` refined until it spans `|ln s| + truncation_half_width(l_factor)`
  /// either side of `ln K = 0`; panics unless `s`, `l_factor` and `c₂` are finite and positive.
  ///
  /// Lord & Kahl (2010), "Complex logarithms in Heston-like models", Math. Finance 20(4), §3.2.
  ///
  /// ```
  /// use stochastic_rs_quant::pricing::fourier::{CarrMadanPricer, HestonFourier};
  /// use stochastic_rs_quant::pricing::heston::HestonParams;
  ///
  /// let p = HestonParams { v0: 0.04, kappa: 1.5, theta: 0.04, sigma: 0.3, rho: -0.6 };
  /// let heston = HestonFourier::new(p);
  /// let pricer = CarrMadanPricer::cumulant_sized(&heston, /* t */ 5.0, /* s */ 100.0, 12.0);
  /// let price = pricer.price_call(&heston, 100.0, 100.0, 0.05, 0.0, 5.0);
  /// assert!(price.is_finite() && price > 0.0);
  /// ```
  pub fn cumulant_sized(model: &impl FourierModelExt, t: f64, s: f64, l_factor: f64) -> Self {
    assert!(
      s.is_finite() && s > 0.0,
      "s must satisfy `s.is_finite() && s > 0.0`, got s = {s:?}"
    );
    assert!(
      l_factor.is_finite() && l_factor > 0.0,
      "l_factor must satisfy `l_factor.is_finite() && l_factor > 0.0`, got l_factor = {l_factor:?}"
    );
    let cumulants = model.cumulants(t);
    let c2 = cumulants.c2;
    assert!(
      c2.is_finite() && c2 > 0.0,
      "cumulants.c2 must satisfy `c2.is_finite() && c2 > 0.0`, got c2 = {c2:?}"
    );
    let required_half_width = s.ln().abs() + cumulants.truncation_half_width(l_factor);
    let default = Self::default();
    // The trapezoid's image of the damped price a period away, `Se^{−2πα/η}`, grows with `η`:
    // a narrow range must not coarsen it past the default.
    let eta = (PI / required_half_width).min(default.eta);
    Self {
      eta: eta.max(0.001),
      ..default
    }
  }

  /// Call prices on the FFT strike grid `(log_strikes, call_prices)`, `k_u = b + λu` with
  /// `b = −Nλ/2` centring it on `ln K = 0`; floored at zero, a NaN passed through.
  pub fn price_call_surface(
    &self,
    model: &impl FourierModelExt,
    s: f64,
    r: f64,
    q: f64,
    t: f64,
  ) -> (Array1<f64>, Array1<f64>) {
    self.surface(model, s, r, q, t, 0.0)
  }

  /// The grid centred on `centre`, by the trapezoid rule of Carr & Madan's (17): `η/2` at `v = 0`.
  /// Reference: Carr & Madan (1999), J. Comput. Finance 2(4), (17), DOI 10.21314/JCF.1999.043
  fn surface(
    &self,
    model: &impl FourierModelExt,
    s: f64,
    r: f64,
    q: f64,
    t: f64,
    centre: f64,
  ) -> (Array1<f64>, Array1<f64>) {
    let n = self.n;
    let alpha = self.alpha;
    let eta = self.eta;
    let lambda = 2.0 * PI / (n as f64 * eta);
    let b = centre - 0.5 * n as f64 * lambda;

    let i_unit = Complex64::i();
    let ln_f = s.ln() + (r - q) * t;
    let disc = (-r * t).exp();

    let mut input = Array1::<Complex64>::zeros(n);

    for j in 0..n {
      let v_j = eta * j as f64;

      let weight = if j == 0 { 0.5 * eta } else { eta };

      let xi = Complex64::new(v_j, -(alpha + 1.0));
      let phi = model.chf(t, xi) * (i_unit * xi * ln_f).exp();

      let denom = Complex64::new(alpha * alpha + alpha - v_j * v_j, (2.0 * alpha + 1.0) * v_j);

      let psi = disc * phi / denom;

      // `e^{−iv_j k_u} = e^{−iv_j b}e^{−2πiju/N}`; only at `b = −Nλ/2` do both signs give `(−1)^j`.
      input[j] = (-i_unit * v_j * b).exp() * psi * weight;
    }

    let handler = FftHandler::<f64>::new(n);
    let mut output = Array1::<Complex64>::zeros(n);
    ndfft(&input, &mut output, &handler, 0);

    let mut log_strikes = Array1::<f64>::zeros(n);
    let mut prices = Array1::<f64>::zeros(n);
    for u in 0..n {
      let k_u = b + lambda * u as f64;
      log_strikes[u] = k_u;
      prices[u] = floor_price((-alpha * k_u).exp() * output[u].re / PI);
    }

    (log_strikes, prices)
  }

  /// The call at `k`, node `N/2` of a grid centred on `ln k`, nothing interpolated; NaN outside
  /// [`Self::price_call_surface`]'s grid, past which `e^{−α ln k}` amplifies the round-off.
  pub fn price_call(
    &self,
    model: &impl FourierModelExt,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    t: f64,
  ) -> f64 {
    let half_span = 0.5 * self.n as f64 * 2.0 * PI / (self.n as f64 * self.eta);
    let log_strike = k.ln();
    if log_strike.is_nan() || log_strike.abs() > half_span {
      return f64::NAN;
    }
    let (_, prices) = self.surface(model, s, r, q, t, log_strike);
    prices[self.n / 2]
  }

  /// Whether `k` falls inside [`Self::price_call_surface`]'s log-strike grid.
  pub fn strike_in_grid(
    &self,
    model: &impl FourierModelExt,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    t: f64,
  ) -> bool {
    let (log_strikes, _) = self.price_call_surface(model, s, r, q, t);
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
    let call = self.price_call(model, s, k, r, q, t);
    call - s * (-q * t).exp() + k * (-r * t).exp()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::pricing::fourier::BsmFourier;
  use crate::pricing::fourier::Cumulants;
  use crate::pricing::fourier::GilPelaezPricer;
  use crate::pricing::fourier::LewisPricer;

  /// A characteristic function that overflows to `NaN`, standing in for a
  /// user model blowing up at a parameter corner. Mirrors `cos.rs`'s
  /// `NanModel`, which pins the same convention for `CosEngine::price`.
  struct NanChfModel;

  impl FourierModelExt for NanChfModel {
    fn chf(&self, _t: f64, _xi: Complex64) -> Complex64 {
      Complex64::new(f64::NAN, f64::NAN)
    }

    fn cumulants(&self, _t: f64) -> Cumulants {
      Cumulants {
        c1: 0.0,
        c2: 0.04,
        c4: Some(0.0),
      }
    }
  }

  fn bsm() -> BsmFourier {
    BsmFourier { sigma: 0.15 }
  }

  /// A NaN strike prices NaN rather than falling through the grid bracket.
  #[test]
  fn carr_madan_nan_strike_prices_nan() {
    let price = CarrMadanPricer::default().price_call(&bsm(), 100.0, f64::NAN, 0.05, 0.0, 1.0);
    assert!(price.is_nan(), "NaN strike must price NaN, got {price}");
  }

  /// The FFT carries a `NaN` characteristic function straight through to the
  /// strike grid; only the floor could lose it. Before the poison-check the
  /// whole grid came back as exactly `0.0` and the interpolation on top of it
  /// returned a plausible zero call price.
  #[test]
  fn carr_madan_surface_preserves_nan_from_chf_blowup() {
    let pricer = CarrMadanPricer::default();
    let (_, prices) = pricer.price_call_surface(&NanChfModel, 100.0, 0.05, 0.0, 1.0);
    assert!(
      prices.iter().all(|p| p.is_nan()),
      "a NaN chf must leave every grid price NaN"
    );

    let interpolated = pricer.price_call(&NanChfModel, 100.0, 100.0, 0.05, 0.0, 1.0);
    assert!(
      interpolated.is_nan(),
      "interpolating a poisoned grid must stay NaN, got {interpolated}"
    );
  }

  /// A NaN characteristic function prices to NaN through Gil-Pelaez and Lewis, not to a
  /// well-scaled fake such as the spot.
  #[test]
  fn quadrature_pricers_preserve_nan_from_chf_blowup() {
    let gp = GilPelaezPricer::price_call(&NanChfModel, 100.0, 100.0, 0.05, 0.0, 1.0);
    let lw = LewisPricer::price_call(&NanChfModel, 100.0, 100.0, 0.05, 0.0, 1.0);
    for (name, p) in [("gil-pelaez", gp), ("lewis", lw)] {
      assert!(p.is_nan(), "{name} must exit as NaN on a NaN chf, got {p}");
    }
  }

  /// A NaN `s`, `k`, `r` or `tau` reaches the final floor as NaN, not as a price of `0.0`.
  #[test]
  fn gil_pelaez_preserves_nan_market_inputs() {
    let m = bsm();
    let nan = f64::NAN;
    let gp = |s, k, r, t| GilPelaezPricer::price_call(&m, s, k, r, 0.0, t);
    for (name, price) in [
      ("tau", gp(100.0, 100.0, 0.05, nan)),
      ("s", gp(nan, 100.0, 0.05, 1.0)),
      ("k", gp(100.0, nan, 0.05, 1.0)),
      ("r", gp(100.0, 100.0, nan, 1.0)),
    ] {
      assert!(price.is_nan(), "NaN {name} must exit as NaN, got {price}");
    }
  }

  #[test]
  fn lewis_preserves_nan_market_inputs() {
    let m = bsm();
    let nan = f64::NAN;
    let lewis = |s, k, r, t| LewisPricer::price_call(&m, s, k, r, 0.0, t);
    for (name, price) in [
      ("tau", lewis(100.0, 100.0, 0.05, nan)),
      ("s", lewis(nan, 100.0, 0.05, 1.0)),
      ("k", lewis(100.0, nan, 0.05, 1.0)),
    ] {
      assert!(price.is_nan(), "NaN {name} must exit as NaN, got {price}");
    }
  }

  /// The floor and the poison-check must stay separate operations: a negative
  /// price is round-off and still floors to zero, a `NaN` is undefined and
  /// travels on.
  #[test]
  fn floor_price_separates_the_floor_from_the_poison_check() {
    assert_eq!(floor_price(-1e-12), 0.0);
    assert_eq!(floor_price(-5.0), 0.0);
    assert_eq!(floor_price(0.0), 0.0);
    assert_eq!(floor_price(3.5), 3.5);
    assert!(floor_price(f64::NAN).is_nan());
  }

  /// The poison-check must not disturb a price that was never poisoned.
  #[test]
  fn finite_prices_are_unchanged_by_the_poison_check() {
    let m = bsm();
    let gp = GilPelaezPricer::price_call(&m, 100.0, 100.0, 0.05, 0.0, 1.0);
    let lw = LewisPricer::price_call(&m, 100.0, 100.0, 0.05, 0.0, 1.0);
    let cm = CarrMadanPricer::default().price_call(&m, 100.0, 100.0, 0.05, 0.0, 1.0);
    for (name, p) in [("gil-pelaez", gp), ("lewis", lw), ("carr-madan", cm)] {
      assert!(
        p.is_finite() && p > 0.0,
        "{name} must still price finitely, got {p}"
      );
    }
  }

  #[test]
  #[should_panic(
    expected = "l_factor must satisfy `l_factor.is_finite() && l_factor > 0.0`, got l_factor = 0.0"
  )]
  fn cumulant_sizing_rejects_a_non_positive_l_factor() {
    let model = BsmFourier { sigma: 0.2 };
    let _ = CarrMadanPricer::cumulant_sized(&model, 1.0, 100.0, 0.0);
  }

  #[test]
  #[should_panic(expected = "cumulants.c2 must satisfy `c2.is_finite() && c2 > 0.0`, got c2 = 0.0")]
  fn cumulant_sizing_rejects_a_vanishing_second_cumulant() {
    let model = BsmFourier { sigma: 0.2 };
    let _ = CarrMadanPricer::cumulant_sized(&model, 0.0, 100.0, 12.0);
  }
}

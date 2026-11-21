//! Fang–Oosterlee COS pricing of any [`FourierModelExt`] model in `y = ln(S_T/K)` from
//! `x = ln(F_T/K)`: `chf(t, u)·e^{iux}` is their `φ(u; x)`, and Eqs. (22)–(23), (29)–(30) follow.
//!
//! Reference: Fang, F. & Oosterlee, C.W. (2008), "A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions", SIAM J. Sci. Comput. 31(2), 826–848, DOI 10.1137/080718061

use std::f64::consts::PI;

use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayViewMut1;
use ndarray::aview_mut1;
use ndarray::aview1;
use num_complex::Complex64;

use super::FourierModelExt;
use crate::OptionType;
use crate::pricing::digital::expired;
use crate::pricing::digital::vanilla_payoff;

/// Fang–Oosterlee (2008) COS pricer for any [`FourierModelExt`] over `l` (default 10, §5.1)
/// cumulant standard deviations, so understated `cumulants()` misprice silently.
#[derive(Debug, Clone)]
pub struct CosEngine {
  /// Cosine terms; `None`, the default, adds terms from 256 until `|chf| < 1e-9`, up to 2,048.
  pub n: Option<usize>,
  /// Truncation half-width parameter (Fang-Oosterlee's `L`).
  pub l: f64,
}

/// The terms the default sums before the decay rule adds any; a slowly decaying chf needs more.
const MIN_TERMS: usize = 256;
const MAX_TERMS: usize = 2048;
const CHF_FLOOR: f64 = 1e-9;

impl Default for CosEngine {
  fn default() -> Self {
    Self::adaptive(10.0)
  }
}

impl CosEngine {
  /// Exactly `n` terms over `l` cumulant standard deviations; panics unless `n >= 2` and `l` is
  /// finite and positive.
  pub fn new(n: usize, l: f64) -> Self {
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n}");
    Self {
      n: Some(n),
      ..Self::adaptive(l)
    }
  }

  /// The default's term rule over `l` cumulant standard deviations; panics unless `l` is finite
  /// and positive.
  pub fn adaptive(l: f64) -> Self {
    assert!(
      l.is_finite() && l > 0.0,
      "l must satisfy `l.is_finite() && l > 0.0`, got l = {l:?}"
    );
    Self { n: None, l }
  }

  /// A call is the put plus parity: the payoff at `t = 0`; NaN past expiry and, before it, for a
  /// rate not finite, an `s0` or `k` not finite and positive, or a NaN `chf` or truncation range.
  pub fn price<M: FourierModelExt>(
    &self,
    model: &M,
    s0: f64,
    k: f64,
    r: f64,
    q: f64,
    t: f64,
    option_type: OptionType,
  ) -> f64 {
    if let Some(value) = expired(t, &[r, q], || vanilla_payoff(option_type, s0, k)) {
      return value;
    }
    if !(priceable(s0) && priceable(k) && r.is_finite() && q.is_finite()) {
      return f64::NAN;
    }
    let cumulants = model.cumulants(t);
    let half_width = cumulants.truncation_half_width(self.l);
    if !half_width.is_finite() {
      return f64::NAN;
    }

    let x = (s0 / k).ln() + (r - q) * t;
    let mut put = [0.0];
    self.put_sums(
      model,
      t,
      cumulants.c1 - half_width,
      2.0 * half_width,
      aview1(&[x]),
      aview1(&[k]),
      aview_mut1(&mut put),
    );
    discounted(put[0], s0, k, r, q, t, option_type)
  }

  /// [`CosEngine::price`] at each strike, bit for bit, on one pass over the `chf`: the range is as
  /// wide at every strike, so its nodes `kπ/(b − a)` are shared and only the payoff terms differ.
  /// Fang & Oosterlee (2008), §3.3, price a strike vector on one common range.
  pub fn price_strikes<M: FourierModelExt>(
    &self,
    model: &M,
    s0: f64,
    strikes: ArrayView1<f64>,
    r: f64,
    q: f64,
    t: f64,
    option_type: OptionType,
  ) -> Array1<f64> {
    let summed = expired(t, &[r, q], || f64::NAN).is_none();
    if !(summed && priceable(s0) && r.is_finite() && q.is_finite()) {
      return strikes.mapv(|k| self.price(model, s0, k, r, q, t, option_type));
    }
    let cumulants = model.cumulants(t);
    let half_width = cumulants.truncation_half_width(self.l);
    if !half_width.is_finite() {
      return Array1::from_elem(strikes.len(), f64::NAN);
    }

    let xs = strikes.mapv(|k| (s0 / k).ln() + (r - q) * t);
    let mut puts = Array1::<f64>::zeros(strikes.len());
    self.put_sums(
      model,
      t,
      cumulants.c1 - half_width,
      2.0 * half_width,
      xs.view(),
      strikes,
      puts.view_mut(),
    );
    puts
      .iter()
      .zip(strikes)
      .map(|(&put, &k)| {
        if priceable(k) {
          discounted(put, s0, k, r, q, t, option_type)
        } else {
          f64::NAN
        }
      })
      .collect()
  }

  /// Adds each strike's undiscounted put series to `puts`, on `[a, a + bma]` with `a = x + lower`;
  /// `e^{iux}·e^{−iua}` is `e^{−iu·lower}` at every strike, so each term calls `chf` once.
  fn put_sums<M: FourierModelExt>(
    &self,
    model: &M,
    t: f64,
    lower: f64,
    bma: f64,
    xs: ArrayView1<f64>,
    strikes: ArrayView1<f64>,
    mut puts: ArrayViewMut1<f64>,
  ) {
    for k_idx in 0..self.n.unwrap_or(MAX_TERMS) {
      let kf = k_idx as f64;
      let u = kf * PI / bma;

      let phi = model.chf(t, Complex64::new(u, 0.0));
      if self.n.is_none() && k_idx >= MIN_TERMS && phi.norm_sqr() < CHF_FLOOR * CHF_FLOOR {
        break;
      }
      let f_k = (phi * Complex64::new(0.0, -u * lower).exp()).re;

      let weight = if k_idx == 0 { 0.5 } else { 1.0 };
      for ((put, &x), &strike) in puts.iter_mut().zip(xs).zip(strikes) {
        let a = x + lower;
        *put += weight * f_k * put_coefficient(kf, a, a + bma, bma, strike);
      }
    }
  }
}

/// A spot or strike whose `ln` is finite, the only kind COS prices before expiry.
fn priceable(x: f64) -> bool {
  x.is_finite() && x > 0.0
}

/// The call's coefficients grow like `e^b`, the put's stay below `K`, so a call is the put and
/// parity: Le Floc'h (2020), §3. A NaN stays NaN; any other price is floored at zero.
fn discounted(
  put_sum: f64,
  s0: f64,
  k: f64,
  r: f64,
  q: f64,
  t: f64,
  option_type: OptionType,
) -> f64 {
  let put = (-r * t).exp() * put_sum;
  let price = match option_type {
    OptionType::Call => put + s0 * (-q * t).exp() - k * (-r * t).exp(),
    OptionType::Put => put,
  };
  if price.is_nan() {
    return f64::NAN;
  }
  price.max(0.0)
}

/// Put payoff coefficient $V_k$ on $[a,b]\cap(-\infty,0]$, Fang-Oosterlee (2008) Eq. (30)
/// for $a<0<b$.
fn put_coefficient(kf: f64, a: f64, b: f64, bma: f64, strike: f64) -> f64 {
  if a >= 0.0 {
    return 0.0;
  }
  let (chi, psi) = chi_psi(kf, a, bma, a, b.min(0.0));
  (2.0 / bma) * strike * (psi - chi)
}

/// Fang-Oosterlee (2008) Eq. (22)-(23): cosine-series coefficients of $e^y$
/// (`chi`) and of the constant function `1` (`psi`) on `[c, d] ⊆ [a, b]`.
fn chi_psi(kf: f64, a: f64, bma: f64, c: f64, d: f64) -> (f64, f64) {
  if kf.abs() < 1e-14 {
    return (d.exp() - c.exp(), d - c);
  }

  let w = kf * PI / bma;
  let (sin_d, cos_d) = (w * (d - a)).sin_cos();
  let (sin_c, cos_c) = (w * (c - a)).sin_cos();

  let chi =
    (cos_d * d.exp() - cos_c * c.exp() + w * (sin_d * d.exp() - sin_c * c.exp())) / (1.0 + w * w);
  let psi = (sin_d - sin_c) / w;

  (chi, psi)
}

#[cfg(test)]
mod tests;

//! Fang–Oosterlee COS pricing of any [`FourierModelExt`] model in `y = ln(S_T/K)` from
//! `x = ln(F_T/K)`: `chf(t, u)·e^{iux}` is their `φ(u; x)`; Eqs. (19), (22)–(23), (25), (50).
//!
//! Reference: Fang & Oosterlee (2008), SIAM J. Sci. Comput. 31(2), 826–848, DOI 10.1137/080718061
//! Reference: Le Floc'h (2020), arXiv:2005.13248, §3

use std::f64::consts::PI;

use ndarray::Array1;
use ndarray::ArrayView1;
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
  pub fn price<M: FourierModelExt + ?Sized>(
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
    let Some((lower, bma)) = self.range(model, t) else {
      return f64::NAN;
    };
    let mut put = [PutSeries::new((s0 / k).ln() + (r - q) * t + lower, bma, k)];
    self.put_sums(model, t, lower, bma, &mut put);
    discounted(put[0].sum, s0, k, r, q, t, option_type)
  }

  /// [`CosEngine::price`] at each strike, bit for bit for runtime model parameters: equal ranges
  /// share nodes and `chf`. Fang & Oosterlee's Eq. (30) prices on one common range instead.
  pub fn price_strikes<M: FourierModelExt + ?Sized>(
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
    let live = strikes.iter().any(|&k| priceable(k));
    let Some((lower, bma)) = self.range(model, t).filter(|_| live) else {
      return Array1::from_elem(strikes.len(), f64::NAN);
    };
    let carry = (r - q) * t;
    let mut puts = strikes
      .iter()
      .map(|&k| PutSeries::new((s0 / k).ln() + carry + lower, bma, k))
      .collect::<Vec<_>>();
    self.put_sums(model, t, lower, bma, &mut puts);
    strikes
      .iter()
      .zip(&puts)
      .map(|(&k, put)| {
        if put.live {
          discounted(put.sum, s0, k, r, q, t, option_type)
        } else {
          f64::NAN
        }
      })
      .collect()
  }

  /// Eq. (49)'s range less each strike's `x`, as `(lower, width)`; `None` where it has no width.
  fn range<M: FourierModelExt + ?Sized>(&self, model: &M, t: f64) -> Option<(f64, f64)> {
    let cumulants = model.cumulants(t);
    let half_width = cumulants.truncation_half_width(self.l);
    half_width
      .is_finite()
      .then_some((cumulants.c1 - half_width, 2.0 * half_width))
  }

  /// Adds Eq. (19)'s terms to each live strike's put: `e^{iux}·e^{−iua}` is `e^{−iu·lower}` at
  /// every strike, so each term calls `chf` once.
  fn put_sums<M: FourierModelExt + ?Sized>(
    &self,
    model: &M,
    t: f64,
    lower: f64,
    bma: f64,
    puts: &mut [PutSeries],
  ) {
    for k_idx in 0..self.n.unwrap_or(MAX_TERMS) {
      let u = k_idx as f64 * PI / bma;
      let phi = model.chf(t, Complex64::new(u, 0.0));
      if self.n.is_none() && k_idx >= MIN_TERMS && phi.norm_sqr() < CHF_FLOOR * CHF_FLOOR {
        break;
      }
      let f_k = (phi * Complex64::new(0.0, -u * lower).exp()).re;
      let weight = if k_idx == 0 { 0.5 } else { 1.0 };
      let (weighted, denom) = (weight * f_k, 1.0 + u * u);
      for put in puts.iter_mut().filter(|put| put.live) {
        put.sum += weighted * put.coefficient(k_idx == 0, u, denom);
      }
    }
  }
}

/// One strike's put series: Eq. (25) on `[a, min(b, 0)]`, its exponentials taken once, so a term
/// costs one `sin_cos`; `χ_k` and `ψ_k` of Eqs. (22)–(23) at `c = a`.
struct PutSeries {
  live: bool,
  inside: bool,
  exp_a: f64,
  exp_d: f64,
  d_minus_a: f64,
  scale: f64,
  sum: f64,
}

impl PutSeries {
  fn new(a: f64, bma: f64, strike: f64) -> Self {
    let d = (a + bma).min(0.0);
    Self {
      live: priceable(strike),
      // A NaN `a` stays in the sum, so its price is NaN.
      inside: a < 0.0 || a.is_nan(),
      exp_a: a.exp(),
      exp_d: d.exp(),
      d_minus_a: d - a,
      scale: (2.0 / bma) * strike,
      sum: 0.0,
    }
  }

  /// `V_k` at `w = kπ/(b − a)` and `denom = 1 + w²`; zero where `a ≥ 0`, the put paying nothing.
  fn coefficient(&self, first: bool, w: f64, denom: f64) -> f64 {
    if !self.inside {
      return 0.0;
    }
    let (chi, psi) = if first {
      (self.exp_d - self.exp_a, self.d_minus_a)
    } else {
      let (sin_d, cos_d) = (w * self.d_minus_a).sin_cos();
      let chi = (cos_d * self.exp_d - self.exp_a + w * (sin_d * self.exp_d)) / denom;
      (chi, sin_d / w)
    };
    self.scale * (psi - chi)
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

#[cfg(test)]
mod tests;

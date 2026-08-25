//! Poisson-weighted-series Greeks for the Merton (1976) jump-diffusion
//! model, at an explicit `(s, k, r, q, tau)` query point.
//!
//! `delta`/`gamma`/`rho` are exact closed-form series over the
//! corresponding [`BSMPricer`](crate::pricing::bsm::BSMPricer) Greek
//! (`Σ w_n · greek(σ_n)`): neither the per-term volatility `σ_n` nor the
//! Poisson weights `w_n` depend on spot or rate, so the naive series *is*
//! the true derivative. `vega`/`theta`/`vanna`/`charm`/`volga`/`veta` bump
//! `v`/`tau` on a copied pricer instead — `σ_n` is itself a function of
//! both (via `Merton1976Pricer::term_vol`), so a naive `Σ w_n · greek(σ_n)`
//! would silently drop the chain-rule term and stop being the true
//! derivative of the price. `theta`/`charm`/`veta` use the calendar
//! `-∂/∂τ` convention (matching `BSMPricer::theta` / `charm` /
//! `dvega_dtime`, and the `λ ≤ 0` Black-Scholes limit below).
//!
//! `theta`/`charm`/`veta`'s `λ > 0` path additionally guards near expiry —
//! case 2 of the crate's [failure
//! convention](crate::traits::ModelPricer#how-pricing-fails):
//! at `τ ≤ h_τ` the down-`τ` bump would evaluate the price series at a
//! negative time-to-maturity, producing per-term `NaN`s that
//! `greek_series`'s `NaN`-floor silently zeroes out of the down-leg —
//! turning an undefined derivative into large finite garbage instead of
//! `NaN`. Mirrors
//! [`HestonPricer`](crate::pricing::heston::HestonPricer)'s identical
//! guard.
//!
//! All 9 methods price through `series_price`, not
//! [`call_put`](Merton1976Pricer::call_put)'s own loop — so, unlike that
//! loop, every Greek here stays finite for `m` past the `usize`-factorial
//! overflow threshold the pre-`poisson_weight` implementation had.

use super::Merton1976Pricer;
use crate::OptionType;
use crate::pricing::bsm::BSMPricer;
use crate::traits::Greeks;
use crate::traits::ModelPricer;

impl Merton1976Pricer {
  const H_TAU: f64 = 1e-5;

  fn h_s(s: f64) -> f64 {
    s.abs() * 1e-4
  }

  fn h_v(&self) -> f64 {
    self.v.abs().max(0.01) * 1e-4
  }

  /// Copy with `v` bumped, floored at `1e-8`.
  fn with_v_bump(&self, dv: f64) -> Self {
    let mut bumped = *self;
    bumped.v = (bumped.v + dv).max(1e-8);
    bumped
  }

  /// Poisson-weighted series over a closed-form BSM Greek. Exact whenever
  /// the Greek's bump variable enters neither `term_vol` nor
  /// `poisson_weight` — true for spot and rate, which is why
  /// `delta`/`gamma`/`rho` use this path. `λ ≤ 0` returns the single
  /// surviving (`n = 0`, weight 1) term directly, sidestepping the `0/0`
  /// singularity `jump_size_std` would otherwise hit.
  ///
  /// `n = 0` is always priced at `term_vol(0, τ) = 0` exactly (a property
  /// of the existing price series, not of this method), which sends
  /// `1/v`-shaped closed forms like `BSMPricer::gamma` to `0/0`. That
  /// term's true contribution is its `v → 0⁺` limit, which is `0` for any
  /// off-the-money strike (`norm_pdf(d1) → 0` exponentially, beating the
  /// linear `1/v`) — so a `NaN` contribution here is floored to `0` rather
  /// than poisoning the whole sum.
  fn greek_series(&self, tau: f64, greek: impl Fn(&BSMPricer) -> f64) -> f64 {
    if self.lambda <= 0.0 {
      return greek(&self.base_bsm());
    }
    (0..self.m)
      .map(|n| {
        let contribution = self.poisson_weight(n, tau) * greek(&self.term_bsm(n, tau));
        if contribution.is_nan() {
          0.0
        } else {
          contribution
        }
      })
      .sum()
  }

  /// Overflow-safe re-implementation of [`call_put`](Self::call_put)'s
  /// Poisson sum, built on [`greek_series`](Self::greek_series) instead of
  /// that method's own loop. Numerically identical for `m ≤ 20` — both
  /// compute `Σ w_n · BS_n(σ_n)`, just accumulating the same weight `w_n`
  /// via a different (equally valid) route.
  fn series_price(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    self.greek_series(tau, |bsm| bsm.price_option(s, k, r, q, tau, option_type))
  }

  /// Delta — $\partial V/\partial S$.
  pub fn delta(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    self.greek_series(tau, |bsm| bsm.delta(s, k, r, q, tau, option_type))
  }

  /// Gamma — $\partial^2 V/\partial S^2$.
  pub fn gamma(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.greek_series(tau, |bsm| bsm.gamma(s, k, r, q, tau))
  }

  /// Rho — $\partial V/\partial r$.
  pub fn rho(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    self.greek_series(tau, |bsm| bsm.rho(s, k, r, q, tau, option_type))
  }

  /// Vega — $\partial V/\partial\sigma$.
  pub fn vega(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    if self.lambda <= 0.0 {
      return self.base_bsm().vega(s, k, r, q, tau);
    }
    let h = self.h_v();
    (self
      .with_v_bump(h)
      .series_price(s, k, r, q, tau, option_type)
      - self
        .with_v_bump(-h)
        .series_price(s, k, r, q, tau, option_type))
      / (2.0 * h)
  }

  /// Theta — $\partial V/\partial t$ (calendar convention).
  ///
  /// On the `λ > 0` path, returns `NaN` for a `tau` that is non-finite or not
  /// larger than `H_TAU`; the `λ ≤ 0` path delegates to [`BSMPricer`] and
  /// inherits its behaviour instead.
  pub fn theta(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    if self.lambda <= 0.0 {
      return self.base_bsm().theta(s, k, r, q, tau, option_type);
    }
    let h = Self::H_TAU;
    if !(tau.is_finite() && tau > h) {
      return f64::NAN;
    }
    -(self.series_price(s, k, r, q, tau + h, option_type)
      - self.series_price(s, k, r, q, tau - h, option_type))
      / (2.0 * h)
  }

  /// Vanna — $\partial^2 V/\partial S\partial\sigma$.
  pub fn vanna(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    if self.lambda <= 0.0 {
      return self.base_bsm().vanna(s, k, r, q, tau);
    }
    let hs = Self::h_s(s);
    let hv = self.h_v();
    let up = self.with_v_bump(hv);
    let dn = self.with_v_bump(-hv);
    (up.series_price(s + hs, k, r, q, tau, option_type)
      - dn.series_price(s + hs, k, r, q, tau, option_type)
      - up.series_price(s - hs, k, r, q, tau, option_type)
      + dn.series_price(s - hs, k, r, q, tau, option_type))
      / (4.0 * hs * hv)
  }

  /// Charm — $\partial^2 V/\partial S\partial t$ (delta decay).
  ///
  /// On the `λ > 0` path, returns `NaN` for a `tau` that is non-finite or not
  /// larger than `H_TAU`; the `λ ≤ 0` path delegates to [`BSMPricer`] and
  /// inherits its behaviour instead.
  pub fn charm(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    if self.lambda <= 0.0 {
      return self.base_bsm().charm(s, k, r, q, tau, option_type);
    }
    let ht = Self::H_TAU;
    if !(tau.is_finite() && tau > ht) {
      return f64::NAN;
    }
    let hs = Self::h_s(s);
    -(self.series_price(s + hs, k, r, q, tau + ht, option_type)
      - self.series_price(s + hs, k, r, q, tau - ht, option_type)
      - self.series_price(s - hs, k, r, q, tau + ht, option_type)
      + self.series_price(s - hs, k, r, q, tau - ht, option_type))
      / (4.0 * hs * ht)
  }

  /// Volga / vomma — $\partial^2 V/\partial\sigma^2$.
  pub fn volga(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    if self.lambda <= 0.0 {
      return self.base_bsm().vomma(s, k, r, q, tau);
    }
    let h = self.h_v();
    let p0 = self.series_price(s, k, r, q, tau, option_type);
    (self
      .with_v_bump(h)
      .series_price(s, k, r, q, tau, option_type)
      - 2.0 * p0
      + self
        .with_v_bump(-h)
        .series_price(s, k, r, q, tau, option_type))
      / (h * h)
  }

  /// Veta — $\partial^2 V/\partial\sigma\partial t$ (vega decay).
  ///
  /// On the `λ > 0` path, returns `NaN` for a `tau` that is non-finite or not
  /// larger than `H_TAU`; the `λ ≤ 0` path delegates to [`BSMPricer`] and
  /// inherits its behaviour instead.
  pub fn veta(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    if self.lambda <= 0.0 {
      return self.base_bsm().dvega_dtime(s, k, r, q, tau);
    }
    let ht = Self::H_TAU;
    if !(tau.is_finite() && tau > ht) {
      return f64::NAN;
    }
    let hv = self.h_v();
    let up = self.with_v_bump(hv);
    let dn = self.with_v_bump(-hv);
    -(up.series_price(s, k, r, q, tau + ht, option_type)
      - up.series_price(s, k, r, q, tau - ht, option_type)
      - dn.series_price(s, k, r, q, tau + ht, option_type)
      + dn.series_price(s, k, r, q, tau - ht, option_type))
      / (4.0 * hv * ht)
  }

  /// Every Greek at one query point, in a single [`Greeks`] struct.
  ///
  /// The `volga → Greeks::volga` and `veta → Greeks::veta` mapping lives
  /// here and nowhere else, so a caller cannot get it wrong by
  /// hand-writing the nine-field literal.
  pub fn greeks(
    &self,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
    option_type: OptionType,
  ) -> Greeks {
    Greeks {
      delta: self.delta(s, k, r, q, tau, option_type),
      gamma: self.gamma(s, k, r, q, tau),
      vega: self.vega(s, k, r, q, tau, option_type),
      theta: self.theta(s, k, r, q, tau, option_type),
      rho: self.rho(s, k, r, q, tau, option_type),
      vanna: self.vanna(s, k, r, q, tau, option_type),
      charm: self.charm(s, k, r, q, tau, option_type),
      volga: self.volga(s, k, r, q, tau, option_type),
      veta: self.veta(s, k, r, q, tau, option_type),
    }
  }
}

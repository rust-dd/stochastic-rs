//! Poisson-weighted-series Greeks for the Merton (1976) jump-diffusion
//! model, at an explicit `(s, k, r, q, tau)` query point.
//!
//! Spot and rate Greeks sum the corresponding Black–Scholes Greeks.
//! Volatility Greeks apply the chain rule to each conditional volatility
//! `sigma_n = v * sqrt(1 - gamma + gamma * n / (lambda * tau))`.
//! Calendar-time derivatives use centred maturity differences, including
//! the dependence of the Poisson weights and conditional volatilities on time.
//!
//! A **degenerate term** — one whose `σ_n` is exactly `0`, reachable at
//! `v == 0` and, for `n = 0` alone, at the pure-jump corner `gamma == 1` —
//! has no closed form to evaluate, only a `σ → 0⁺` limit, and the limit
//! differs per Greek and per side of the forward. Each accessor states its
//! own against [`TermRegime`]; `greek_series` no longer floors anything.
//! `delta → ½e^{(b−r)τ}` and `rho → ½Kτe^{−rτ}` at the forward and keep
//! their saturated closed forms away from it, `gamma → +∞` at the forward
//! and `0` away from it. Volatility Greeks use their right limits at zero
//! volatility; maturity differences inherit the degenerate price limit.
//!
//! `theta`/`charm`/`veta`'s `λ > 0` path additionally guards near expiry —
//! case 2 of the crate's [failure
//! convention](crate::traits::ModelPricer#how-pricing-fails):
//! at `τ ≤ h_τ` the down-`τ` bump would evaluate the price series at a
//! negative time-to-maturity, whose derivative is undefined. The guard is
//! an explicit statement of that, not the only thing standing behind it:
//! nothing zeroes those terms, so a negative bumped maturity reaches the
//! caller as `NaN` on its own. It stays
//! because saying so at the accessor is clearer than relying on which
//! terms happen to go non-finite, and because it also covers a
//! non-finite `τ`. Mirrors
//! [`HestonPricer`](crate::pricing::heston::HestonPricer)'s identical
//! guard.
//!
//! # Which Greeks take an `option_type`
//!
//! Five do — `delta`, `theta`, `rho`, `charm` and (vacuously) the
//! aggregator — and four do not: `vega`, `vanna`, `volga` and `veta`.
//! The line between them is generalised put-call parity, whose spread
//! $C-P=Se^{(b-r)\tau}-Ke^{-r\tau}$ carries **no $\sigma$**. A derivative
//! that touches $\sigma$ even once annihilates the spread, so it is one
//! number rather than two:
//!
//! | Greek | derivative | in $\sigma$ | takes `option_type` |
//! |---|---|---|---|
//! | `vega` | $\partial_\sigma$ | yes | no |
//! | `vanna` | $\partial_S\partial_\sigma$ | yes | no |
//! | `volga` | $\partial_\sigma\partial_\sigma$ | yes | no |
//! | `veta` | $\partial_\sigma\partial_\tau$ | yes | no |
//! | `gamma` | $\partial_S\partial_S$ | no | no |
//! | `delta` | $\partial_S$ | no | yes |
//! | `theta` | $-\partial_\tau$ | no | yes |
//! | `rho` | $\partial_r$ | no | yes |
//! | `charm` | $-\partial_S\partial_\tau$ | no | yes |
//!
//! `gamma` is in the "no `option_type`" column for the *other* reason the
//! spread admits — it is linear in $S$, so a second $S$-derivative kills
//! it too — and has never taken one. The four that do take one each leave
//! a surviving spread term, whose form depends on which
//! [`BSMCoc`](crate::pricing::bsm::BSMCoc) supplies $b$.
//!

use stochastic_rs_distributions::special::norm_pdf;

use super::Merton1976Pricer;
use super::TermRegime;
use crate::OptionType;
use crate::pricing::bsm::BSMPricer;
use crate::traits::Greeks;

impl Merton1976Pricer {
  const H_TAU: f64 = 1e-5;

  fn h_s(s: f64) -> f64 {
    s.abs() * 1e-4
  }

  fn volatility_series(&self, tau: f64, f: impl Fn(&BSMPricer, f64) -> f64) -> f64 {
    let unit_volatility = Self { v: 1.0, ..*self };
    (0..self.m)
      .map(|n| {
        let weight = self.poisson_weight(n, tau);
        if weight == 0.0 {
          return 0.0;
        }
        weight * f(&self.term_bsm(n, tau), unit_volatility.term_vol(n, tau))
      })
      .sum()
  }

  /// Poisson-weighted spot and rate Greeks. Each caller supplies its own
  /// zero-volatility limit; the no-jump case uses the base Black-Scholes model.
  fn greek_series(&self, tau: f64, greek: impl Fn(&BSMPricer) -> f64) -> f64 {
    if self.lambda <= 0.0 {
      return greek(&self.base_bsm());
    }
    (0..self.m)
      .map(|n| self.poisson_weight(n, tau) * greek(&self.term_bsm(n, tau)))
      .sum()
  }

  /// Conditional prices used by the maturity differences in theta and charm.
  /// Uses the same zero-volatility limits as `call_put`.
  pub(super) fn series_price(
    &self,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
    option_type: OptionType,
  ) -> f64 {
    self.greek_series(tau, |bsm| {
      let (call, put) = Merton1976Pricer::term_call_put(bsm, s, k, r, q, tau);
      match option_type {
        OptionType::Call => call,
        OptionType::Put => put,
      }
    })
  }

  /// Delta — $\partial V/\partial S$.
  ///
  /// A degenerate term at the forward (`TermRegime::AtTheForward`)
  /// contributes the $\sigma \to 0^+$ limit of $e^{(b-r)\tau}N(d_1)$, which
  /// is $\tfrac12 e^{(b-r)\tau}$ for the call and
  /// $-\tfrac12 e^{(b-r)\tau}$ for the put — the closed form at
  /// $N(d_1) = \tfrac12$, and no more than that. Written as an expression
  /// in `r` and `τ` rather than as the number it evaluates to, so a
  /// non-finite rate still propagates.
  pub fn delta(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    self.greek_series(tau, |bsm| {
      match Merton1976Pricer::term_regime(bsm, s, k, r, q, tau) {
        TermRegime::AtTheForward => {
          let half_carry = 0.5 * ((bsm.b(r, q) - r) * tau).exp();
          match option_type {
            OptionType::Call => half_carry,
            OptionType::Put => -half_carry,
          }
        }
        TermRegime::Saturated | TermRegime::Ordinary => bsm.delta(s, k, r, q, tau, option_type),
      }
    })
  }

  /// Gamma — $\partial^2 V/\partial S^2$.
  ///
  /// **Returns $+\infty$ at a degenerate term's
  /// forward (`TermRegime::AtTheForward`)**, which is the value of the limit
  /// and not a failure to compute one. $\Gamma =
  /// e^{(b-r)\tau}\varphi(d_1)/(S\sigma\sqrt\tau)$ has a finite, strictly
  /// positive numerator there ($\varphi(0) = 1/\sqrt{2\pi}$) over a
  /// vanishing $\sigma$, so it diverges like $1/\sigma$ — measured at
  /// $(S, K, r, \tau) = (100, 100, 0.05, 0.5)$ under
  /// [`BSMCoc::Black1976`](crate::pricing::bsm::BSMCoc::Black1976),
  /// $\sigma\Gamma$ is `0.0063285` to sixteen figures across
  /// $\sigma = 10^{-3} \ldots 10^{-8}$. The frozen underlying's payoff is a
  /// step at the forward and its second derivative is a Dirac delta; an
  /// unbounded gamma is what that *is*.
  ///
  /// So the arm below is the closed form with $d_1$ at its limit of `0`,
  /// left to divide by the term's own `+0.0`. IEEE gives $+\infty$ for the
  /// positive numerator and keeps propagating a non-finite `r`, which a
  /// literal [`f64::INFINITY`] would not. Case 2 of the crate's [failure
  /// convention](crate::traits::ModelPricer#how-pricing-fails) is for a
  /// quantity that is *undefined* here; this one is defined and unbounded,
  /// so `NaN` would understate it and `0.0` — what the old floor returned —
  /// inverts it. `MertonCreditPricer::credit_spread` and
  /// `g_digital_put_2d`'s logarithmic corner are the crate's precedent for
  /// letting a real divergence through as one.
  ///
  /// Away from the forward the same term contributes `0`, and both are
  /// pinned: `the_degenerate_term_floor_is_exact_away_from_the_forward` and
  /// `the_forward_point_greeks_of_a_degenerate_term_are_their_limits`.
  pub fn gamma(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.greek_series(tau, |bsm| {
      match Merton1976Pricer::term_regime(bsm, s, k, r, q, tau) {
        TermRegime::AtTheForward => {
          ((bsm.b(r, q) - r) * tau).exp() * norm_pdf(0.0) / (s * bsm.v * tau.sqrt())
        }
        TermRegime::Saturated => ((bsm.b(r, q) - r) * tau).exp() * 0.0,
        TermRegime::Ordinary => bsm.gamma(s, k, r, q, tau),
      }
    })
  }

  /// Rho — $\partial V/\partial r$.
  ///
  /// A degenerate term at the forward (`TermRegime::AtTheForward`)
  /// contributes the $\sigma \to 0^+$ limit of $K\tau e^{-r\tau}N(d_2)$,
  /// which is $\tfrac12 K\tau e^{-r\tau}$ for the call and its negation for
  /// the put — the closed form at $N(d_2) = \tfrac12$. `rho` is the one of
  /// the three whose limit carries the *discount* factor rather than the
  /// carry factor, so it is the accessor that would catch a mix-up of the
  /// two.
  pub fn rho(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, option_type: OptionType) -> f64 {
    self.greek_series(tau, |bsm| {
      match Merton1976Pricer::term_regime(bsm, s, k, r, q, tau) {
        TermRegime::AtTheForward => {
          let half = 0.5 * k * tau * (-r * tau).exp();
          match option_type {
            OptionType::Call => half,
            OptionType::Put => -half,
          }
        }
        TermRegime::Saturated | TermRegime::Ordinary => bsm.rho(s, k, r, q, tau, option_type),
      }
    })
  }

  /// Vega — $\partial V/\partial\sigma$.
  ///
  /// Takes no `option_type`. Put-call parity's spread
  /// $Se^{(b-r)\tau}-Ke^{-r\tau}$ carries no $\sigma$, so a derivative
  /// that touches $\sigma$ annihilates it and the call and the put share
  /// one answer. `gamma` loses its parameter for the sibling reason — the
  /// spread is linear in $S$ — while `delta`, `theta`, `rho` and `charm`
  /// each leave a surviving spread term and keep theirs.
  pub fn vega(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    if self.lambda <= 0.0 {
      return self.base_bsm().vega(s, k, r, q, tau);
    }
    self.volatility_series(tau, |bsm, scale| {
      scale
        * match Self::term_regime(bsm, s, k, r, q, tau) {
          TermRegime::AtTheForward => {
            s * ((bsm.b(r, q) - r) * tau).exp() * norm_pdf(0.0) * tau.sqrt()
          }
          TermRegime::Saturated => ((bsm.b(r, q) - r) * tau).exp() * 0.0,
          TermRegime::Ordinary => bsm.vega(s, k, r, q, tau),
        }
    })
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
  ///
  /// Takes no `option_type`. Put-call parity's spread
  /// $Se^{(b-r)\tau}-Ke^{-r\tau}$ carries no $\sigma$, so a derivative
  /// that touches $\sigma$ annihilates it and the call and the put share
  /// one answer. `gamma` loses its parameter for the sibling reason — the
  /// spread is linear in $S$ — while `delta`, `theta`, `rho` and `charm`
  /// each leave a surviving spread term and keep theirs.
  pub fn vanna(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    if self.lambda <= 0.0 {
      return self.base_bsm().vanna(s, k, r, q, tau);
    }
    self.volatility_series(tau, |bsm, scale| {
      scale
        * match Self::term_regime(bsm, s, k, r, q, tau) {
          TermRegime::AtTheForward => {
            0.5 * ((bsm.b(r, q) - r) * tau).exp() * norm_pdf(0.0) * tau.sqrt()
          }
          TermRegime::Saturated => ((bsm.b(r, q) - r) * tau).exp() * 0.0,
          TermRegime::Ordinary => bsm.vanna(s, k, r, q, tau),
        }
    })
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
  ///
  /// Takes no `option_type`. Put-call parity's spread
  /// $Se^{(b-r)\tau}-Ke^{-r\tau}$ carries no $\sigma$, so a derivative
  /// that touches $\sigma$ annihilates it and the call and the put share
  /// one answer. `gamma` loses its parameter for the sibling reason — the
  /// spread is linear in $S$ — while `delta`, `theta`, `rho` and `charm`
  /// each leave a surviving spread term and keep theirs.
  pub fn volga(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    if self.lambda <= 0.0 {
      return self.base_bsm().vomma(s, k, r, q, tau);
    }
    self.volatility_series(tau, |bsm, scale| {
      scale
        * scale
        * match Self::term_regime(bsm, s, k, r, q, tau) {
          TermRegime::AtTheForward | TermRegime::Saturated => ((bsm.b(r, q) - r) * tau).exp() * 0.0,
          TermRegime::Ordinary => bsm.vomma(s, k, r, q, tau),
        }
    })
  }

  /// Veta — $\partial^2 V/\partial\sigma\partial t$ (vega decay).
  ///
  /// Takes no `option_type`. Put-call parity's spread
  /// $Se^{(b-r)\tau}-Ke^{-r\tau}$ carries no $\sigma$, so a derivative
  /// that touches $\sigma$ annihilates it and the call and the put share
  /// one answer. `gamma` loses its parameter for the sibling reason — the
  /// spread is linear in $S$ — while `delta`, `theta`, `rho` and `charm`
  /// each leave a surviving spread term and keep theirs.
  ///
  /// On the `λ > 0` path, returns `NaN` for a `tau` that is non-finite or not
  /// larger than `H_TAU`; the `λ ≤ 0` path delegates to [`BSMPricer`] and
  /// inherits its behaviour instead.
  pub fn veta(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    if self.lambda <= 0.0 {
      return self.base_bsm().dvega_dtime(s, k, r, q, tau);
    }
    let ht = Self::H_TAU;
    if !(tau.is_finite() && tau > ht) {
      return f64::NAN;
    }
    -(self.vega(s, k, r, q, tau + ht) - self.vega(s, k, r, q, tau - ht)) / (2.0 * ht)
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
      vega: self.vega(s, k, r, q, tau),
      theta: self.theta(s, k, r, q, tau, option_type),
      rho: self.rho(s, k, r, q, tau, option_type),
      vanna: self.vanna(s, k, r, q, tau),
      charm: self.charm(s, k, r, q, tau, option_type),
      volga: self.volga(s, k, r, q, tau),
      veta: self.veta(s, k, r, q, tau),
    }
  }
}

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
//! A **degenerate term** — one whose `σ_n` is exactly `0`, reachable at
//! `v == 0` and, for `n = 0` alone, at the pure-jump corner `gamma == 1` —
//! has no closed form to evaluate, only a `σ → 0⁺` limit, and the limit
//! differs per Greek and per side of the forward. Each accessor states its
//! own against [`TermRegime`]; `greek_series` no longer floors anything.
//! `delta → ½e^{(b−r)τ}` and `rho → ½Kτe^{−rτ}` at the forward and keep
//! their saturated closed forms away from it, `gamma → +∞` at the forward
//! and `0` away from it, and the six that difference a price inherit
//! `term_call_put`'s `½(Se^{(b−r)τ} − Ke^{−rτ}) = 0`.
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
//! All 9 methods price through `series_price`, not
//! [`call_put`](Merton1976Pricer::call_put)'s own loop — so, unlike that
//! loop, every Greek here stays finite for `m` past the `usize`-factorial
//! overflow threshold the pre-`poisson_weight` implementation had.
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
//! The four volatility Greeks are central differences of `series_price`,
//! which still has to price *something*; they price the call. The put's
//! own difference is the same number to well under an ulp of the
//! differenced prices, which `the_volatility_greeks_are_the_same_for_a_put`
//! pins directly rather than by re-deriving the parity argument. The lone
//! exception is not a parity failure but an `erf` one, documented on
//! [`volga`](Merton1976Pricer::volga).

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

  fn h_v(&self) -> f64 {
    self.v.abs().max(0.01) * 1e-4
  }

  /// Copy with `v` bumped, floored at `1e-8`.
  fn with_v_bump(&self, dv: f64) -> Self {
    let mut bumped = *self;
    bumped.v = (bumped.v + dv).max(1e-8);
    bumped
  }

  /// Poisson-weighted series over a per-term quantity. Exact whenever the
  /// Greek's bump variable enters neither `term_vol` nor `poisson_weight` —
  /// true for spot and rate, which is why `delta`/`gamma`/`rho` use this
  /// path.
  ///
  /// `λ ≤ 0` returns the single surviving (`n = 0`, weight 1) term
  /// directly. At `λ == 0` that is what the series would produce anyway —
  /// `Merton1976Pricer::jump_size_std` reports the no-jump state's `z = 0`,
  /// so `σ_n = v` and the weights are `1, 0, 0, …` — and the branch is kept
  /// for the bump-based Greeks below, whose `λ ≤ 0` legs need the *closed
  /// form* rather than a central difference of it to match `BSMPricer` to
  /// `1e-10`.
  ///
  /// One residual survives there on purpose. At `λ == 0` **and** `v == 0`
  /// the six bump-based Greeks come back `NaN` while `delta`, `gamma`,
  /// `rho` and the price return their limits, because those six take their
  /// own `λ ≤ 0` legs straight to [`BSMPricer`], which has no degenerate
  /// handling of its own. The asymmetry is `BSMPricer`'s rather than this
  /// pricer's — the same query prices `NaN` on a bare `BSMPricer` — and
  /// `a_frozen_no_jump_model_inherits_black_scholess_own_gaps` pins it so
  /// it cannot be mistaken for a regression here.
  ///
  /// **This sum has no `NaN` floor**, and that is the point. It used to
  /// end `if contribution.is_nan() && term.v == 0.0 { 0.0 }`, which was
  /// exact away from the forward and wrong at it — `d₁` is `0/0` there, so
  /// every closed form goes `NaN` and `0` is the limit of only some of
  /// them. Each caller now states its own `σ → 0⁺` limit against
  /// [`TermRegime`], which is the shape
  /// [`term_call_put`](Merton1976Pricer::term_call_put) already used for
  /// the price, so a term that has no limit worth claiming propagates its
  /// `NaN` instead of being laundered into a confident `0.0`.
  fn greek_series(&self, tau: f64, greek: impl Fn(&BSMPricer) -> f64) -> f64 {
    if self.lambda <= 0.0 {
      return greek(&self.base_bsm());
    }
    (0..self.m)
      .map(|n| self.poisson_weight(n, tau) * greek(&self.term_bsm(n, tau)))
      .sum()
  }

  /// Overflow-safe re-implementation of [`call_put`](Self::call_put)'s
  /// Poisson sum, built on [`greek_series`](Self::greek_series) instead of
  /// that method's own loop. Numerically identical for `m ≤ 20` — both
  /// compute `Σ w_n · BS_n(σ_n)`, just accumulating the same weight `w_n`
  /// via a different (equally valid) route.
  ///
  /// `theta` and `charm` pass the caller's own `option_type` — the spread
  /// their `τ`-derivative sees is not `σ`-free. The four volatility Greeks
  /// pass [`series_call`](Self::series_call) instead.
  ///
  /// Prices each term through
  /// [`term_call_put`](Merton1976Pricer::term_call_put) rather than through
  /// [`ModelPricer::price_option`](crate::traits::ModelPricer::price_option),
  /// so the degenerate term's forward limit is
  /// the *same expression* the price uses rather than a second copy of it.
  /// The value it fills in there is `½(Se^{(b-r)τ} − Ke^{-rτ})`, which is
  /// `0` — being at the forward is the statement `Se^{bτ} = K` — and that
  /// is why `theta`, `charm` and the rest of the bump-based Greeks were
  /// already right at the forward while `delta`, `gamma` and `rho` were
  /// not: the quantity they floor is a price, whose limit really is zero.
  ///
  /// `pub(super)` only so
  /// `the_greeks_price_a_degenerate_term_exactly_as_the_price_does` can
  /// assert the two against each other directly; nothing outside this
  /// module calls it.
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

  /// [`series_price`](Self::series_price) at the call, for the four Greeks
  /// that differentiate in `σ` at least once.
  ///
  /// The choice of leg is free there and *only* there: the put's series
  /// differs from the call's by `(Σ w_n)·(Ke^{-rτ} − Se^{(b-r)τ})`, which
  /// carries no `σ`, so it contributes nothing to a `σ`-derivative and
  /// every one of the four is a single number rather than one per leg.
  fn series_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.series_price(s, k, r, q, tau, OptionType::Call)
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
    let h = self.h_v();
    (self.with_v_bump(h).series_call(s, k, r, q, tau)
      - self.with_v_bump(-h).series_call(s, k, r, q, tau))
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
    let hs = Self::h_s(s);
    let hv = self.h_v();
    let up = self.with_v_bump(hv);
    let dn = self.with_v_bump(-hv);
    (up.series_call(s + hs, k, r, q, tau)
      - dn.series_call(s + hs, k, r, q, tau)
      - up.series_call(s - hs, k, r, q, tau)
      + dn.series_call(s - hs, k, r, q, tau))
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
  ///
  /// Takes no `option_type`. Put-call parity's spread
  /// $Se^{(b-r)\tau}-Ke^{-r\tau}$ carries no $\sigma$, so a derivative
  /// that touches $\sigma$ annihilates it and the call and the put share
  /// one answer. `gamma` loses its parameter for the sibling reason — the
  /// spread is linear in $S$ — while `delta`, `theta`, `rho` and `charm`
  /// each leave a surviving spread term and keep theirs.
  ///
  /// **Unreliable in one narrow band of query points**, and the cause is
  /// upstream of this crate's model code.
  /// [`erf`](stochastic_rs_distributions::special::erf) is Abramowitz &
  /// Stegun 7.1.26, whose five coefficients sum to `0.999999999`, and it
  /// is made odd by a sign branch — so it carries a `2e-9` **jump** across
  /// the origin, `-1e-9` to `+1e-9`, where the true `erf` passes through
  /// `0`. `volga` is the only one of the nine exposed to it, because it is
  /// the only one that evaluates the series at the **unbumped** `v` and
  /// then divides by `h_v² ≈ 4e-10`; the others step off the point.
  ///
  /// The band is where a Poisson term's `d₁` or `d₂` changes sign inside
  /// the `v ± h_v` stencil. Measured at `(v, λ, γ) = (0.2, 0.5, 0.3)`,
  /// `(S, K, r, τ) = (110, 110, 0.05, 1)` under `Bsm1973` — where the
  /// `n = 3` term has `σ₃² = 2b` exactly, which is the `d₂ = 0` condition
  /// at `S = K` — this returns `14.6357` against `11.3318` from a 100×
  /// coarser bump, and it still returns `14.6356` / `8.0247` one part in
  /// `10⁷` either side in strike.
  ///
  /// The call and the put are wrong **together** there, by the same
  /// `±3.3`, since `norm_cdf(-x)` is `1 - norm_cdf(x)` exactly. The single
  /// argument where that identity fails is `±0.0` itself — `-0.0 < 0.0` is
  /// false, so the sign branch does not flip — and it is the *only* reason
  /// the four Greeks above can differ between a call and a put at all.
  /// `a_poisson_term_on_erfs_origin_jump_wobbles_volga` pins both faults.
  ///
  pub fn volga(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    if self.lambda <= 0.0 {
      return self.base_bsm().vomma(s, k, r, q, tau);
    }
    let h = self.h_v();
    let p0 = self.series_call(s, k, r, q, tau);
    (self.with_v_bump(h).series_call(s, k, r, q, tau) - 2.0 * p0
      + self.with_v_bump(-h).series_call(s, k, r, q, tau))
      / (h * h)
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
    let hv = self.h_v();
    let up = self.with_v_bump(hv);
    let dn = self.with_v_bump(-hv);
    -(up.series_call(s, k, r, q, tau + ht)
      - up.series_call(s, k, r, q, tau - ht)
      - dn.series_call(s, k, r, q, tau + ht)
      + dn.series_call(s, k, r, q, tau - ht))
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

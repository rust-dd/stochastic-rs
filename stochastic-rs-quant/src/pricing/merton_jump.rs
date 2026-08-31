//! # Merton Jump
//!
//! $$
//! V=\sum_{n=0}^{\infty}e^{-\lambda\tau}\frac{(\lambda\tau)^n}{n!}V_{BS}(\sigma_n),
//! \qquad\sigma_n=\sqrt{d^2+z^2\tfrac{n}{\tau}}
//! $$
//!
//! `d` is the diffusive volatility and `z` the per-jump log-size standard
//! deviation, both implied by the total volatility `v` and the jump
//! variance share `gamma`.
//!
//! Discount rate and cost of carry are the same in every term because the
//! jumps are taken with $E(Y)=1$: Merton's per-term
//! $r_n = r-\lambda k+n\ln(1+k)/\tau$ collapses to `r` at
//! $k = E(Y)-1 = 0$, and the intensity needs no $\lambda(1+k)$ re-weighting
//! either. That is the same specialisation Haug prints, and the reason this
//! model carries no mean-jump-size parameter.
//!
use super::bsm::BSMCoc;
use super::bsm::BSMPricer;
use crate::traits::ModelPricer;
use crate::traits::VanillaEuropeanCall;

mod greeks;

/// Merton (1976) jump-diffusion pricer.
///
/// The struct holds **model state only** — the total volatility, the jump
/// intensity and jump-variance share, the Poisson-series truncation limit
/// and the cost-of-carry convention. Spot, strike, rate, dividend yield and
/// maturity are the pricing *query* and travel as arguments to
/// [`ModelPricer::price_call`] and to every Greek, so one instance prices a
/// whole strike/maturity grid.
///
/// The query's `(r, q)` pair is **`(discount rate, carry offset)`**, not
/// necessarily `(risk-free rate, dividend yield)`: this pricer discounts at
/// `r` and carries at [`BSMPricer::b(r, q)`](BSMPricer::b), whose value
/// depends on `self.b`. To reproduce a discount `r₀` with a carry `b₀` —
/// Garman-Kohlhagen's `b₀ = r_d − r_f`, say — pass
/// `(r, q) = (r₀, r₀ − b₀)`, which is an identity rather than an
/// approximation because GK's `b(r, q) = r − q`. See
/// `merton_gk_carries_at_rd_minus_rf_and_discounts_at_r`.
///
/// ```
/// use stochastic_rs_quant::pricing::bsm::BSMCoc;
/// use stochastic_rs_quant::pricing::merton_jump::Merton1976Pricer;
/// use stochastic_rs_quant::traits::ModelPricer;
///
/// let model = Merton1976Pricer::new(0.2, 0.5, 0.4, 10, BSMCoc::Bsm1973);
/// let atm = model.price_call(100.0, 100.0, 0.05, 0.0, 0.5);
/// let otm = model.price_call(100.0, 120.0, 0.05, 0.0, 0.5);
/// assert!(atm > otm);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Merton1976Pricer {
  /// Volatility
  pub v: f64,
  /// Expected number of jumps
  pub lambda: f64,
  /// Percentage of the volatility due to jumps
  pub gamma: f64,
  /// Iteration limit
  pub m: usize,
  /// Cost of carry
  pub b: BSMCoc,
}

/// Where one Poisson term's query sits, which is what decides whether its
/// closed forms are the value or stand in need of a $\sigma \to 0^+$ limit.
///
/// A term whose [`term_vol`](Merton1976Pricer::term_vol) is exactly `0`
/// prices a zero-volatility Black-Scholes option, and its
/// $d_1 = (\ln(S/K) + b\tau)/(\sigma\sqrt\tau) + \sigma\sqrt\tau/2$ is then
/// $\infty \cdot x$ — never finite. *Which* infinity decides everything, and
/// the two cases have different limits, which is why one blanket `0` could
/// not serve both.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TermRegime {
  /// $\sigma_n > 0$, or a query the closed forms already answer with `NaN`
  /// and must go on answering with one. The closed form *is* the value.
  Ordinary,
  /// $\sigma_n = 0$ away from the forward. $d_1 \to \pm\infty$, both normal
  /// CDFs saturate, and everything but the $1/\sigma$-shaped quantities is
  /// already its own limit. Those — `gamma` alone among the nine — are
  /// $0/0$, and their limit is `0`: $\varphi(d_1)$ decays like
  /// $e^{-c^2/2\sigma^2}$ and beats the linear $1/\sigma$.
  Saturated,
  /// $\sigma_n = 0$ **at** the forward $Se^{b\tau} = K$. $d_1$ is $0/0$, so
  /// every closed form is `NaN` and each needs its own limit. Both CDFs
  /// converge to $\tfrac12$ — $d_1 = \sigma\sqrt\tau/2 \to 0^+$ and
  /// $d_2 = -\sigma\sqrt\tau/2 \to 0^-$ — which leaves the price and the
  /// first-order Greeks finite; `gamma` still divides by $\sigma$ and
  /// diverges.
  AtTheForward,
}

impl Merton1976Pricer {
  /// Validating constructor.
  ///
  /// # Panics
  /// - if `v` is negative or `NaN`. Every use of `v` in the price squares
  ///   it, so a negative volatility silently prices as its own absolute
  ///   value; the Greeks are worse, since `with_v_bump` floors the bumped
  ///   volatility at `1e-8` and both legs of the central difference then
  ///   land on the floor, returning a `vega` of `0`.
  /// - if `m` is `0`. It is the Poisson-series length, so an empty series
  ///   runs the sum zero times and [`call_put`](Self::call_put) returns
  ///   `(0.0, 0.0)` — the plausible-looking sentinel the crate's [failure
  ///   convention](crate::traits::ModelPricer#how-pricing-fails) rules out,
  ///   indistinguishable from a genuinely worthless option.
  ///
  /// `lambda` and `gamma` are deliberately **not** checked. `lambda == 0`
  /// is a *supported* state rather than an invalid one — price and Greeks
  /// both collapse to plain Black-Scholes at `v` there, which
  /// `merton_price_lambda_zero_equals_bs` and
  /// `merton_greeks_lambda_zero_equals_bs` pin — and a `gamma` outside
  /// `[0, 1]` drives $\sigma^2 - \lambda z^2$ negative, which announces
  /// itself as `NaN` rather than as a number.
  ///
  /// A **negative** `lambda` is neither, and the two halves still disagree
  /// about it: the price is `NaN`, which is the convention, while the
  /// Greeks' `λ ≤ 0` branch answers with the Black-Scholes value. Left
  /// alone here because narrowing that branch to `λ == 0` would route a
  /// negative intensity into `greek_series`'s
  /// `NaN`-floor and turn a visible `NaN` into a confident `0.0`, which is
  /// worse than the disagreement.
  pub fn new(v: f64, lambda: f64, gamma: f64, m: usize, b: BSMCoc) -> Self {
    assert!(
      v >= 0.0,
      "Merton1976Pricer::new: v must be a non-negative volatility (got {v})"
    );
    assert!(
      m >= 1,
      "Merton1976Pricer::new: m must be at least 1 (got {m})"
    );
    Self {
      v,
      lambda,
      gamma,
      m,
      b,
    }
  }

  /// Poisson-weighted series $\sum_{n=0}^{m-1} w_n \cdot V_{BS}(\sigma_n)$,
  /// routed through `poisson_weight`'s
  /// running-product weight rather than an integer `n!` — the latter
  /// overflows `usize` past `n \approx 21` (the crate's Python binding
  /// documents a default of `m = 50`), silently producing garbage instead
  /// of a price. Numerically identical to the pre-refactor factorial-based
  /// loop for `m \le 20` (see
  /// `merton_price_m10_matches_the_reference_value` for the regression pin).
  pub fn call_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> (f64, f64) {
    let mut call = 0.0;
    let mut put = 0.0;

    for i in 0..self.m {
      let weight = self.poisson_weight(i, tau);
      let (c, p) = Self::term_call_put(&self.term_bsm(i, tau), s, k, r, q, tau);
      call += c * weight;
      put += p * weight;
    }

    (call, put)
  }

  /// Jump-size standard deviation implied by decomposing total volatility
  /// `v` into a diffusive part and a jump part that together explain a
  /// `gamma` fraction of the variance.
  ///
  /// `lambda == 0` is the no-jump state, and a jump that never happens has
  /// no size: `z = 0`, so the whole variance is diffusive and the series
  /// collapses to Black-Scholes at `v`. The closed form cannot say that on
  /// its own — `v²γ/λ` is `∞` at `γ > 0` and `NaN` at `γ = 0`, and either
  /// way [`diffusive_std`](Self::diffusive_std)'s `λ·z²` becomes `0·∞`, so
  /// **every** price at `lambda == 0` used to be `NaN` while the Greeks
  /// returned the Black-Scholes value. Only `λz²`, the jump *variance
  /// rate*, ever enters the model, and that is `0` here whatever `z` would
  /// have been.
  ///
  /// This is a value at a point, not a limit. Holding `gamma` fixed while
  /// `λ → 0⁺` keeps a `γ` share of the variance in ever-rarer, ever-larger
  /// jumps (`z² = γv²/λ → ∞`), so the price tends to Black-Scholes at
  /// `v√(1-γ)` — the diffusive part alone — and not to the value here. The
  /// two agree exactly when `gamma == 0`, where there is no jump variance
  /// to lose. `the_lambda_zero_limit_is_discontinuous_in_gamma` pins both
  /// halves.
  fn jump_size_std(&self) -> f64 {
    if self.lambda == 0.0 {
      return 0.0;
    }
    (self.v.powi(2) * self.gamma / self.lambda).sqrt()
  }

  /// Diffusive volatility component: total variance minus the jump
  /// variance rate $\lambda z^2$.
  ///
  /// $\lambda z^2$ is the only form in which the jump size enters the
  /// model, and it is $v^2\gamma$ by construction — substitute
  /// [`jump_size_std`](Self::jump_size_std)'s $z^2 = v^2\gamma/\lambda$
  /// and the $\lambda$ cancels — so it is taken directly rather than by
  /// squaring that method's `sqrt` back out. The round-trip was not
  /// value-preserving: at the pure-jump corner `gamma == 1` it landed on
  /// `d = 0` for `(v, lambda)` of `(0.5, 1)`, `(0.2, 1)`, `(0.3, 0.5)` and
  /// `(0.2, 0.25)`, and one ulp *below* zero — a `NaN` — for `(0.2, 0.5)`
  /// and `(0.25, 2)`. The same model priced or did not by rounding;
  /// `the_pure_jump_corner_is_zero_for_every_intensity` is the pin.
  ///
  /// The two branches are the states where $\lambda z^2 = v^2\gamma$ is
  /// *not* an identity, and in both the round-trip announced something the
  /// bare $v^2(1-\gamma)$ would silence:
  ///
  /// - `lambda == 0` has no jump size to speak of, so
  ///   [`jump_size_std`](Self::jump_size_std) answers `0` and the whole
  ///   variance is diffusive: `d = v`, not `v√(1-γ)`. That second value is
  ///   the `λ → 0⁺` *limit*, a different number, and
  ///   `the_lambda_zero_limit_is_discontinuous_in_gamma` pins the gap.
  /// - a `z` that is not a real number — $v^2\gamma/\lambda < 0$, which is
  ///   every `gamma` and `lambda` of opposite sign — must stay `NaN`
  ///   rather than become a finite $v^2(1-\gamma)$, since that is how a
  ///   `gamma` outside `[0, 1]` announces itself as [`new`](Self::new)
  ///   documents. Testing [`jump_size_std`](Self::jump_size_std)'s own
  ///   output rather than re-deriving the condition keeps the `NaN` set
  ///   identical to the round-trip's by construction.
  fn diffusive_std(&self) -> f64 {
    let jump_variance_rate = if self.lambda == 0.0 {
      0.0
    } else if self.jump_size_std().is_nan() {
      f64::NAN
    } else {
      self.v.powi(2) * self.gamma
    };
    (self.v.powi(2) - jump_variance_rate).sqrt()
  }

  /// Per-term volatility of the `n`-jump-conditional Black-Scholes term,
  /// $\sigma_n = \sqrt{d^2 + z^2 n/\tau}$ — Merton (1976) eq. (18), whose
  /// per-term option is "a Black-Scholes option where the formal variance
  /// per unit time on the stock is $\sigma^2 + n\delta^2/\tau$", and
  /// §6.9.1 of Haug's *Complete Guide to Option Pricing Formulas*, which
  /// prints it as $\sigma_i = \sqrt{z^2 + \delta^2(i/T)}$ at exactly this
  /// pricer's parameterisation.
  ///
  /// Conditioning on `n` jumps over the option's life leaves the log-return
  /// as the diffusion plus `n` i.i.d. jump sizes, so its variance is
  /// `d²·τ + n·z²`: the diffusion runs for the *whole* of `τ` however many
  /// jumps land in it, and only the jump part scales with the count. A
  /// Black-Scholes term consumes that as `σ_n²·τ`, which gives the
  /// expression above — and in particular $\sigma_0 = d$, the diffusive
  /// volatility, rather than `0`.
  ///
  /// This is what makes `v` the total volatility the field claims it is.
  /// Averaging the conditional variance over `N ~ Poisson(λτ)` gives
  /// `d²τ + λτ·z² = v²τ` exactly, which is the identity
  /// [`diffusive_std`](Self::diffusive_std)'s `v² − λz²` subtraction exists
  /// to arrange; the subtraction is dead weight under any other `σ_n`.
  /// `merton_gamma_zero_is_black_scholes` and
  /// `merton_conditional_variance_averages_to_the_total` pin the two ends
  /// of that statement.
  fn term_vol(&self, n: usize, tau: f64) -> f64 {
    (self.diffusive_std().powi(2) + self.jump_size_std().powi(2) * n as f64 / tau).sqrt()
  }

  /// Poisson weight `e^{-λτ}(λτ)^n / n!` for the `n`-th term. Accumulates
  /// `(λτ)^n / n!` as a running `f64` product rather than an integer `n!`
  /// (which overflows `usize` past `n ≈ 20`, unlike the Poisson weight
  /// itself — bounded in `[0, 1]` for every `n`).
  fn poisson_weight(&self, n: usize, tau: f64) -> f64 {
    let lt = self.lambda * tau;
    let ratio = (0..n).fold(1.0, |acc, i| acc * lt / (i as f64 + 1.0));
    (-lt).exp() * ratio
  }

  /// `BSMPricer` at this pricer's cost-of-carry convention and total
  /// volatility `self.v` (the no-jump / Black-Scholes limit);
  /// [`term_bsm`](Self::term_bsm) swaps in the per-term volatility.
  fn base_bsm(&self) -> BSMPricer {
    BSMPricer::new(self.v, self.b)
  }

  fn term_bsm(&self, n: usize, tau: f64) -> BSMPricer {
    BSMPricer::new(self.term_vol(n, tau), self.b)
  }

  /// Classify one Poisson term's query — see [`TermRegime`].
  ///
  /// The three guards are what keep a *poisoned* query out of the two
  /// degenerate arms, where a limit expression would answer it with a
  /// plausible number instead of propagating. At `v == 0` the term
  /// volatility is exactly `0` for an infinite or negative `tau` just as
  /// much as for a good one — `(-0.0f64).sqrt()` is `-0.0`, which compares
  /// equal to `0.0` — and `d₁` is `NaN` for a non-positive `s` or `k` for a
  /// reason that has nothing to do with the forward. All three keep the
  /// closed form's own `NaN`, which is case 2 of the crate's [failure
  /// convention](crate::traits::ModelPricer#how-pricing-fails).
  ///
  /// The `NaN`/`Saturated` split needs no third arm: at `σ_n = 0` the
  /// leading `1/(σ√τ)` is `+∞`, so `d₁` is `±∞` for any non-zero numerator
  /// and `NaN` for a zero one, and cannot come out finite.
  fn term_regime(term: &BSMPricer, s: f64, k: f64, r: f64, q: f64, tau: f64) -> TermRegime {
    if term.v != 0.0 || !(tau.is_finite() && tau > 0.0 && s > 0.0 && k > 0.0) {
      return TermRegime::Ordinary;
    }
    if term.d1_d2(s, k, r, q, tau).0.is_nan() {
      TermRegime::AtTheForward
    } else {
      TermRegime::Saturated
    }
  }

  /// Call and put of one Poisson term, with the one point where the
  /// zero-volatility term is a *removable singularity* filled in.
  ///
  /// A term whose [`term_vol`](Self::term_vol) is exactly `0` prices a
  /// zero-volatility Black-Scholes call. Its
  /// $d_1 = (\ln(S/K) + b\tau)/(\sigma\sqrt\tau) + \sigma\sqrt\tau/2$ is
  /// $\pm\infty$ wherever $Se^{b\tau} \ne K$, which saturates both normal
  /// CDFs and collapses the term to its discounted intrinsic forward value.
  /// **At** the forward the leading numerator vanishes too, leaving $0/0$,
  /// and a single `NaN` term poisons the whole Poisson sum.
  ///
  /// $\sigma_n = \sqrt{d^2 + z^2n/\tau}$ is zero only where the *diffusive*
  /// component `d` is, so this is reachable at `v == 0` — the zero total
  /// volatility `new` accepts on purpose, where every term of the series is
  /// degenerate — and, for the `n = 0` term alone, at the pure-jump corner
  /// `gamma == 1`, where `v² − λz²` is now exactly `0` for every intensity
  /// rather than for the half of them that rounded that way.
  /// `zero_total_volatility_at_the_forward_is_the_limit` is
  /// the live pin; `an_ordinary_configuration_never_reaches_the_branch` is
  /// the counterpart that keeps the narrowed reachability honest.
  ///
  /// That point is a removable singularity, not an undefined quantity. Let
  /// $\sigma \to 0^+$ along $Se^{b\tau} = K$: then $d_1 = \sigma\sqrt\tau/2
  /// \to 0^+$ and $d_2 = -\sigma\sqrt\tau/2 \to 0^-$, so both CDFs converge
  /// to $\tfrac12$ and the term tends to
  /// $\tfrac12(Se^{(b-r)\tau} - Ke^{-r\tau})$ — which is zero, because
  /// being at the forward *is* the statement $Se^{b\tau} = K$. The branch
  /// below writes that limit out rather than returning the constant, so a
  /// non-finite discount rate still propagates instead of being replaced by
  /// a confident zero. This is a degenerate limit with a value, not case 2
  /// of the crate's [failure
  /// convention](crate::traits::ModelPricer#how-pricing-fails).
  ///
  /// Only the $0/0$ point is intercepted; every other term keeps the value
  /// [`BSMPricer::call_put`] already produced, degenerate or not.
  ///
  /// Which strike is singular is set by the cost of carry, not by the
  /// volatility: [`BSMCoc::Black1976`] and [`BSMCoc::Asay1982`] have
  /// $b = 0$, putting the forward at $S$, so a degenerate configuration
  /// loses its at-the-money point — the most-quoted one on a futures-option
  /// surface. The three carrying conventions put it at $Se^{b\tau}$, a
  /// strike nobody asks for exactly.
  ///
  /// Takes the term rather than its index so the Greeks'
  /// [`series_price`](Self::series_price) can price through **this**
  /// function rather than through a second copy of the same limit; the
  /// price and the Greeks' idea of the price then cannot come apart, which
  /// `the_greeks_price_a_degenerate_term_exactly_as_the_price_does` pins.
  fn term_call_put(term: &BSMPricer, s: f64, k: f64, r: f64, q: f64, tau: f64) -> (f64, f64) {
    if Self::term_regime(term, s, k, r, q, tau) == TermRegime::AtTheForward {
      let half_carry = 0.5 * s * ((term.b(r, q) - r) * tau).exp();
      let half_disc = 0.5 * k * (-r * tau).exp();
      return (half_carry - half_disc, half_disc - half_carry);
    }
    term.call_put(s, k, r, q, tau)
  }
}

impl ModelPricer for Merton1976Pricer {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.call_put(s, k, r, q, tau).0
  }

  /// Overrides the trait's vanilla-parity default for the same reason
  /// [`BSMPricer::price_put`] does, inherited term by term: each element of
  /// the Poisson series carries at $e^{(b-r)\tau}$, which equals the
  /// default's $e^{-q\tau}$ only when $b = r - q$ — false for
  /// [`BSMCoc::Bsm1973`] at `q != 0` and for [`BSMCoc::Black1976`] /
  /// [`BSMCoc::Asay1982`]. See `merton_price_put_overrides_vanilla_parity`.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.call_put(s, k, r, q, tau).1
  }
}

/// European vanilla call — every term of the Poisson series is one, priced
/// through [`BSMPricer`], so this inherits that type's carry question along
/// with its answer.
impl VanillaEuropeanCall for Merton1976Pricer {
  /// $Se^{b\tau}$ at the [`BSMCoc`] convention
  /// held in `self.b`, delegated to
  /// [`BSMPricer::vanilla_call_forward`] for the same reason
  /// [`price_put`](ModelPricer::price_put) delegates: the series carries term
  /// by term at whatever the underlying `BSMPricer` carries at, so the two
  /// must not be able to disagree.
  fn vanilla_call_forward(&self, s: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.base_bsm().vanilla_call_forward(s, r, q, tau)
  }
}

#[cfg(test)]
mod tests;

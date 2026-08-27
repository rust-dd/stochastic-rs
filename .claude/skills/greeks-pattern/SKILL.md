---
name: greeks-pattern
description: How to expose first- and second-order Greeks in stochastic-rs — an inherent greeks(s, k, r, q, tau, option_type) aggregator on a pricer, or the no-argument GreeksExt trait for the two Monte Carlo Malliavin estimators. Invoke when adding a pricer that needs delta/gamma/vega/theta/rho/vanna/charm/volga/veta, or when an MC estimator is missing the single-pass `greeks()` override.
---

# Greeks pattern — stochastic-rs

First- and second-order Greeks reach callers one of two ways, and which
one applies depends on whether the type carries its own market data. The
`GreeksExt` trait in `stochastic-rs-quant::traits` serves the two types
that do. **Five** pricers instead expose query-taking inherent
accessors plus a `greeks(s, k, r, q, tau, option_type)` aggregate —
`BSMPricer`, `HestonPricer`, `Merton1976Pricer`, `CashOrNothingPricer`,
`AssetOrNothingPricer`. Most `ModelPricer`s expose **no** Greeks at all
(`GapPricer` and `SuperSharePricer`, both in `pricing/digital.rs`,
are the in-tree examples); implementing `ModelPricer` does not oblige
you to add them. Section 1 shows both surfaces.

Note that `GreeksExt`'s accessors default to `f64::NAN`, **not** to a
finite difference — a pricer that does not override `vega` reports NaN so
a consumer can distinguish "not exposed" from a real zero.

A third, unrelated `greeks()` exists and is easy to confuse with these
two: `PricingResult::greeks(&self) -> Option<Greeks>`
(`traits/instrument.rs`), part of the QuantLib-style engine surface.
It reports what an engine happened to compute, and is not a way to
*implement* Greeks on a pricer.

The single-pass `greeks()` aggregator is the load-bearing part for MC
pricers: calling each Greek separately means N independent
re-pricings (with different random seeds, in general). Overriding
`greeks()` to share a single set of MC paths across all Greeks is
**mandatory** for MC pricers; otherwise users get visibly inconsistent
delta/gamma/vega from re-runs.

## 1. The trait surface

```rust
// stochastic-rs-quant/src/traits/pricing.rs

// `Default` is implemented by hand as `Greeks::nan()`, NOT derived —
// a derived `Default` would be all-zeros, which is exactly the
// plausible-looking sentinel section 8 forbids.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
    pub vanna: f64,
    pub charm: f64,
    pub volga: f64,
    pub veta: f64,
}

// `GreeksExt` has no supertrait, and only `delta` is required — every
// other accessor defaults to `f64::NAN`, NOT to a finite difference. A
// pricer that does not override `vega` reports NaN, so a consumer can
// tell "not exposed" from a real zero.
pub trait GreeksExt {
    fn delta(&self) -> f64;
    fn gamma(&self) -> f64 { f64::NAN }
    fn vega(&self)  -> f64 { f64::NAN }
    fn theta(&self) -> f64 { f64::NAN }
    fn rho(&self)   -> f64 { f64::NAN }
    fn vanna(&self) -> f64 { f64::NAN }
    fn charm(&self) -> f64 { f64::NAN }
    fn volga(&self) -> f64 { f64::NAN }
    fn veta(&self)  -> f64 { f64::NAN }

    // Aggregate. The default calls every accessor; the two MC estimators
    // override it so one simulation backs the whole set (section 4).
    fn greeks(&self) -> Greeks { /* default: calls every accessor */ }
}
```

**Which shape to use is decided by whether the type carries its own
market data**, and since 3.0 most pricers do not:

- A type that **bundles** its query — the two Monte Carlo Malliavin
  estimators `GbmMalliavinGreeks` / `HestonMalliavinGreeks` — can answer
  `delta(&self)` with no arguments, so it implements `GreeksExt`. These
  two are the only implementors. The analytic pricers, digitals included,
  deliberately do not and cannot: they hold model state only.
- A **`ModelPricer`** holds model parameters only, so `delta(&self)` has
  no spot to differentiate against. It exposes **inherent, query-taking**
  accessors plus an aggregate instead, and does *not* implement
  `GreeksExt`:

```rust
impl BSMPricer {
    pub fn delta(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, ot: OptionType) -> f64 { ... }
    pub fn gamma(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 { ... }

    /// Aggregate. Provide this whenever you provide the accessors —
    /// without it every caller hand-writes the nine-field `Greeks { .. }`
    /// literal, and a mis-mapped field (volga vs veta) has nowhere to be
    /// caught. Monte Carlo pricers should override it to share one set
    /// of paths rather than calling each accessor separately.
    pub fn greeks(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, ot: OptionType) -> Greeks { ... }
}
```

## 2. Which Greek when

| Greek  | Definition                                       | Bump variable |
|--------|--------------------------------------------------|---------------|
| delta  | $\partial V / \partial S$                        | spot S        |
| gamma  | $\partial^2 V / \partial S^2$                    | spot S        |
| vega   | $\partial V / \partial \sigma$                   | vol σ         |
| theta  | $-\partial V / \partial T$                       | maturity T    |
| rho    | $\partial V / \partial r$                        | rate r        |
| vanna  | $\partial^2 V / (\partial S \partial \sigma)$    | spot + vol    |
| charm  | $\partial \Delta / \partial T = -\partial^2 V/(\partial S \partial T)$ | spot + T |
| volga  | $\partial^2 V / \partial \sigma^2$               | vol σ         |
| veta   | $\partial \mathcal{V} / \partial T = -\partial^2 V/(\partial \sigma \partial T)$ | vol + T |

The Greeks struct is a flat `f64` bundle by design. Per-pricer overrides
typically fill only the Greeks that have analytic closed forms (e.g.
BSM does delta/gamma/vega/theta/rho analytically) and leave the
second-order cross-Greeks at the trait default (numerical).

## 3. NaN defaults — when you don't compute a Greek

If a pricer cannot compute a particular Greek (e.g. a fixed-strike
basket pricer that has no σ parameter to bump), set the field to
`f64::NAN` rather than 0.0. Convention:

```rust
fn vega(&self) -> f64 {
    f64::NAN  // basket has no σ; consumers detect via .is_nan()
}
```

**Do not return 0.0 for "not applicable" cases.** Zero is a valid
sensitivity value and silently masks bugs (a basket reporting 0 vega
looks like a vol-immunised portfolio, which is misleading).

## 4. Single-pass MC override (mandatory for MC pricers)

The default `greeks()` aggregator calls each Greek method
sequentially. For an analytic pricer that re-uses the same closed-form
inputs, this is fine. For a Monte Carlo pricer, each call re-samples
paths from a *different* RNG state (or the same seed twice, depending
on construction), and the resulting deltas / gammas / vegas don't share
control variates. The user observes "delta-from-greeks() ≠
delta-direct()" within numerical noise.

The mandated pattern: override `greeks()` so **one** simulation feeds
every returned Greek. Both in-tree implementors do this with **Malliavin
weights**, not with bump-and-reprice — there is no bumping machinery in
this crate. Do not reach for `with_spot_bump(...)`, `with_vol_bump(...)`,
`with_bump_sizes(...)`, `price_from_paths(...)` or `simulate_with_seed(...)`:
none of them exist anywhere in the repo.

`GbmMalliavinGreeks` (`pricing/malliavin_greeks/gbm.rs`) is the shape to
copy. Its `greeks()` is a one-line delegation to an inherent
`all_greeks()` that simulates once and accumulates four Malliavin
weights over the same paths:

```rust
pub fn all_greeks(&self) -> Greeks {
    let (s_t, w_t) = self.simulate();          // ONE simulation
    let discount = (-self.r * self.tau).exp();
    let (m, t) = (self.n_paths as f64, self.tau);
    let (mut sum_delta, mut sum_gamma, mut sum_vega, mut sum_rho) = (0.0, 0.0, 0.0, 0.0);

    for i in 0..self.n_paths {
        let disc_payoff = discount * (s_t[i] - self.k).max(0.0);
        let w = w_t[i];                        // the Brownian increment
        sum_delta += disc_payoff * (w / (self.s * self.sigma * t));
        sum_gamma += disc_payoff
            * ((w * w - self.sigma * t * w - t)
               / (self.s * self.s * self.sigma * self.sigma * t * t));
        sum_vega  += disc_payoff * ((w * w - t) / (self.sigma * t) - w);
        sum_rho   += disc_payoff * (w / self.sigma - t);
    }

    Greeks {
        delta: sum_delta / m,
        gamma: sum_gamma / m,
        vega:  sum_vega / m,
        rho:   sum_rho / m,
        // No Malliavin weight exists for these, so they stay NaN —
        // spelled out field by field. There is no `Greeks::nan()`
        // struct-update shorthand in use here.
        theta: f64::NAN,
        vanna: f64::NAN, charm: f64::NAN, volga: f64::NAN, veta: f64::NAN,
    }
}

impl GreeksExt for GbmMalliavinGreeks {
    fn greeks(&self) -> Greeks { self.all_greeks() }
}
```

`HestonMalliavinGreeks` (`pricing/malliavin_greeks/heston.rs`) does the
same with two helpers instead of one accumulator loop:

```rust
fn greeks(&self) -> Greeks {
    let (delta, gamma) = self.delta_gamma_single_pass();
    let vega = self.vega_v0();
    Greeks { delta, gamma, vega,
             theta: f64::NAN, rho: f64::NAN,
             vanna: f64::NAN, charm: f64::NAN, volga: f64::NAN, veta: f64::NAN }
}
```

The point of the override is stated in `all_greeks`'s own doc comment:
calling `delta()`, `gamma()`, `vega()`, `rho()` individually each runs a
**fresh** simulation, so the four would come from four different sample
paths and be mutually inconsistent. Sharing the paths is the whole
reason the override exists.

If you do write a bump-and-reprice estimator, common random numbers
(Glasserman 2003 §7.1) is the right technique — but you would be adding
the bumping API, not using an existing one. See
`add-mc-variance-reduction` §6.

## 5. Analytic-pricer minimal impl

For an analytic (closed-form) pricer, you usually only override the
Greeks you know analytically:

An analytic pricer holds model parameters only, so it does **not**
implement `GreeksExt` — there is no spot on the struct to differentiate
against. It exposes query-taking inherent methods, and every Greek it does
not implement is simply absent rather than defaulting to anything:

```rust
impl BSMPricer {
    pub fn delta(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, ot: OptionType) -> f64 {
        let (d1, _) = self.d1_d2(s, k, r, q, tau);
        match ot {
            OptionType::Call => norm_cdf(d1),
            OptionType::Put  => norm_cdf(d1) - 1.0,
        }
    }

    pub fn gamma(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
        let (d1, _) = self.d1_d2(s, k, r, q, tau);
        norm_pdf(d1) / (s * self.v * tau.sqrt())
    }

    pub fn vega(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
        let (d1, _) = self.d1_d2(s, k, r, q, tau);
        s * norm_pdf(d1) * tau.sqrt()
    }

    /// Provide this whenever you provide the accessors. Without it every
    /// caller hand-writes the nine-field literal, and a mis-mapped field
    /// (volga vs veta) has nowhere to be caught.
    pub fn greeks(&self, s: f64, k: f64, r: f64, q: f64, tau: f64, ot: OptionType) -> Greeks {
        Greeks {
            delta: self.delta(s, k, r, q, tau, ot),
            gamma: self.gamma(s, k, r, q, tau),
            vega:  self.vega(s, k, r, q, tau),
            ..Greeks::nan()
        }
    }
}
```

Note `..Greeks::nan()`, not `..Default::default()` — they are the same
thing (`Default for Greeks` returns `nan()`), but spelling it `nan()` says
the intent. **Never fill an unimplemented Greek with `0.0`**: a zero is a
legitimate value for several Greeks, so a consumer cannot tell it from
"not exposed", which is the whole reason the members default to `NaN`.

Reference: `BSMPricer` in `pricing/bsm/greeks.rs`.

## 6. Bump-size conventions

**There is no default finite-difference machinery in this crate**, and
no `with_bump_sizes(...)` builder. `GreeksExt`'s accessors default to
`f64::NAN`, and the five inherent aggregators are closed-form. A Greek
is either derived analytically, estimated by a Malliavin weight, or
absent.

If you are *adding* a bump-and-reprice estimator, these are reasonable
starting bump sizes — as guidance for new code, not as a description of
an existing default:

| Parameter | Suggested bump | Rationale |
|-----------|----------------|-----------|
| spot S    | `S * 1e-4`     | relative bump; absolute bump fails for large S |
| vol σ     | `1e-4`         | absolute (vols are O(1)); relative would be too small for low vol |
| rate r    | `1e-5`         | absolute (rates are O(0.01–0.10)) |
| maturity T| `1.0 / 365.0`  | one calendar day |

Expose them however the pricer's own builder style suggests. If you find
yourself needing ad-hoc bumps in calibration,
that's a hint to switch to analytic Greeks.

## 7. Testing

A new pricer's inherent `greeks(...)` aggregator must ship at least:

1. **Sign tests:** call delta is positive, put delta is negative,
   gamma is positive for both, vega is positive.
2. **Put-call parity for delta:** `Δ_call - Δ_put = e^{-qT}`.
3. **Bumped-NPV consistency:** `(price(S+h) - price(S-h)) / (2h)` agrees
   with `delta()` to the bump precision.
4. **Single-pass consistency (MC only):** `greeks().delta == delta()`
   within MC tolerance, sourced from the same seed.

Reference: `pricing/bsm/tests.rs`. The MC tests live next to each
Malliavin-Greeks estimator (`pricing/malliavin_greeks/tests.rs`).

## 8. Anti-patterns

- **Do not** return `0.0` for "not applicable" Greeks. Use `f64::NAN`.
- **Do not** override an individual Greek but leave `greeks()` at the
  default in an MC pricer. The user-facing Greeks then have asymmetric
  precision across components (the override Greek is single-pass, the
  rest re-sample).
- **Do not** finite-difference the price by re-running an MC pricer
  with a different RNG state per Greek. Always seed-share via
  `with_*_bump(...)` constructors.
- **Do not** invent new Greek field names. `Greeks` is a fixed flat
  struct; if you need an extra Greek (e.g. zomma = `∂Γ/∂σ`), extend the
  struct in one PR rather than smuggling it through a `HashMap`-style
  side channel.

## 9. Reference impls

- `pricing/bsm/greeks.rs` — analytic, inherent aggregator.
- `pricing/heston/greeks.rs` — analytic, fills all nine fields.
- `pricing/merton_jump/greeks.rs` — analytic, inherent aggregator.
- `pricing/digital.rs` — the cash-or-nothing and asset-or-nothing
  aggregators.
- `pricing/malliavin_greeks/gbm.rs` — Malliavin-weighted single-pass
  `GreeksExt`.
- `pricing/malliavin_greeks/heston.rs` — the second `GreeksExt` impl.

## Related SKILLs

- `add-mc-variance-reduction` — explains the Common Random Numbers
  requirement that the single-pass `greeks()` override depends on.
- `calibration-pattern` — calibrators consume Greeks via the result's
  `to_model(r, q).greeks(s, k, r, q, tau, option_type)` chain.
- `release-checklist` — `MIGRATION.md` should note any new Greek
  added to the public surface.

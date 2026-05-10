---
name: greeks-pattern
description: How to expose first- and second-order Greeks via the GreeksExt trait in stochastic-rs. Invoke when adding a new pricer that needs delta/gamma/vega/theta/rho/vanna/charm/volga/veta, or when a Monte Carlo pricer is missing the single-pass `greeks()` override.
---

# Greeks pattern — stochastic-rs

The `GreeksExt` trait in `stochastic-rs-quant::traits` lets pricers
report first- and second-order Greeks via a uniform interface. The
trait has default implementations that compute each Greek by
finite-differencing the pricer's `calculate_price()`, **plus** a single
`greeks()` aggregator that returns the full bundle.

The single-pass `greeks()` aggregator is the load-bearing part for MC
pricers: defaults call each Greek separately, which means N independent
re-pricings (with different random seeds, in general). Overriding
`greeks()` to share a single set of MC paths across all Greeks is
**mandatory** for MC pricers; otherwise users get visibly inconsistent
delta/gamma/vega from re-runs.

## 1. The trait surface

```rust
// stochastic-rs-quant/src/traits/pricing.rs

#[derive(Debug, Clone, Default)]
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

pub trait GreeksExt: PricerExt {
    fn delta(&self) -> f64 { /* default: central diff in spot */ }
    fn gamma(&self) -> f64 { /* default: central second-diff in spot */ }
    fn vega(&self)  -> f64 { /* default: central diff in vol */ }
    fn theta(&self) -> f64 { /* default: forward diff in tau, sign-flipped */ }
    fn rho(&self)   -> f64 { /* default: central diff in r */ }
    fn vanna(&self) -> f64 { /* default: cross-diff (S, σ) */ }
    fn charm(&self) -> f64 { /* default: cross-diff (S, T) */ }
    fn volga(&self) -> f64 { /* default: second-diff in σ */ }
    fn veta(&self)  -> f64 { /* default: cross-diff (σ, T) */ }

    /// Single-pass aggregator. Default implementation calls each
    /// per-Greek method individually, which is fine for analytic
    /// pricers (delta() + gamma() + vega() are independent
    /// finite-difference calls) but is INCORRECT for Monte Carlo
    /// pricers — they must override this to share a single set of paths.
    fn greeks(&self) -> Greeks {
        Greeks {
            delta: self.delta(),
            gamma: self.gamma(),
            vega:  self.vega(),
            theta: self.theta(),
            rho:   self.rho(),
            vanna: self.vanna(),
            charm: self.charm(),
            volga: self.volga(),
            veta:  self.veta(),
        }
    }
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

The mandated pattern: override `greeks()` to do one big simulation and
populate all Greeks from the same path set:

```rust
impl GreeksExt for MyMcPricer {
    fn greeks(&self) -> Greeks {
        // Generate *one* set of paths.
        let paths = self.simulate_with_seed(self.master_seed);
        // Re-price under each Greek bump using the same path noise
        // (Common Random Numbers — Glasserman 2003 §7.1).
        let v_base = price_from_paths(&paths, self);
        let v_sup  = price_from_paths(&paths, &self.with_spot_bump( h));
        let v_sdn  = price_from_paths(&paths, &self.with_spot_bump(-h));
        let v_vup  = price_from_paths(&paths, &self.with_vol_bump( hs));
        let v_vdn  = price_from_paths(&paths, &self.with_vol_bump(-hs));
        // ... etc ...
        Greeks {
            delta: (v_sup - v_sdn) / (2.0 * h),
            gamma: (v_sup - 2.0 * v_base + v_sdn) / (h * h),
            vega:  (v_vup - v_vdn) / (2.0 * hs),
            // ...
            ..Default::default()  // unfilled Greeks → 0.0; or use NAN sentinel
        }
    }
}
```

The `with_spot_bump(...)` / `with_vol_bump(...)` constructors must clone
the pricer with the bumped parameter and re-use the same seed so
"common random numbers" gives variance reduction. Two reference
implementations:

- `GbmMalliavinGreeks` (`pricing/malliavin_thalmaier/gbm.rs`): the
  cleanest single-pass example; uses Malliavin weights to compute Greeks
  *without* finite differencing, but the surrounding `greeks()` shape is
  the canonical pattern.
- `HestonMalliavinGreeks` (`pricing/malliavin_thalmaier/heston.rs`):
  multi-asset extension; same single-pass shape with cross-Greeks.

## 5. Analytic-pricer minimal impl

For an analytic (closed-form) pricer, you usually only override the
Greeks you know analytically:

```rust
impl GreeksExt for BsmPricer {
    fn delta(&self) -> f64 {
        let d1 = self.d1();
        match self.option_type {
            OptionType::Call => norm_cdf(d1),
            OptionType::Put  => norm_cdf(d1) - 1.0,
        }
    }

    fn gamma(&self) -> f64 {
        let d1 = self.d1();
        norm_pdf(d1) / (self.s * self.sigma * self.tau.sqrt())
    }

    fn vega(&self) -> f64 {
        let d1 = self.d1();
        self.s * norm_pdf(d1) * self.tau.sqrt()
    }

    // theta, rho, vanna, charm, volga, veta — fall through to default
    // finite-difference, which is fine for an analytic pricer.

    // greeks() — fall through to default; the cost of N calls is N
    // closed-form evaluations, no random-noise problem.
}
```

Reference: `BsmPricer` in `pricing/bsm.rs`.

## 6. Bump-size conventions

The default finite-difference Greeks pick bump sizes that scale with
the parameter magnitude:

| Parameter | Default bump   | Rationale |
|-----------|----------------|-----------|
| spot S    | `S * 1e-4`     | relative bump; absolute bump fails for large S |
| vol σ     | `1e-4`         | absolute (vols are O(1)); relative would be too small for low vol |
| rate r    | `1e-5`         | absolute (rates are O(0.01–0.10)) |
| maturity T| `1.0 / 365.0`  | one calendar day |

Custom bump sizes for a specific pricer can be exposed as
`with_bump_sizes(...)` builder methods, but the defaults work for 95%
of cases. If you find yourself needing ad-hoc bumps in calibration,
that's a hint to switch to analytic Greeks.

## 7. Testing

A new pricer with `GreeksExt` must ship at least:

1. **Sign tests:** call delta is positive, put delta is negative,
   gamma is positive for both, vega is positive.
2. **Put-call parity for delta:** `Δ_call - Δ_put = e^{-qT}`.
3. **Bumped-NPV consistency:** `(price(S+h) - price(S-h)) / (2h)` agrees
   with `delta()` to the bump precision.
4. **Single-pass consistency (MC only):** `greeks().delta == delta()`
   within MC tolerance, sourced from the same seed.

Reference: `pricing/bsm.rs` tests. The MC tests live next to each
Malliavin-Greeks pricer (`pricing/malliavin_thalmaier/*.rs`).

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

- `pricing/bsm.rs` — analytic + default fall-through.
- `pricing/heston.rs` — analytic delta/vega/theta + default fall-through.
- `pricing/malliavin_thalmaier/gbm.rs` — Malliavin-weighted single-pass.
- `pricing/malliavin_thalmaier/heston.rs` — multi-asset Malliavin.
- `pricing/malliavin_thalmaier/sabr.rs` — SABR Malliavin.

## Related SKILLs

- `add-mc-variance-reduction` — explains the Common Random Numbers
  requirement that the single-pass `greeks()` override depends on.
- `calibration-pattern` — calibrators consume Greeks via the result's
  `to_model().greeks()` chain.
- `release-checklist` — the rc.X CHANGELOG should note any new Greek
  added to the public surface.

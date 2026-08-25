# A2 follow-up queue

Work remaining after the A2 quant-consistency wave. The wave itself is closed and
recorded in [`A2_WAVE_LEDGER.md`](A2_WAVE_LEDGER.md); this is the queue of defects
and consistency work it surfaced but did not own.

Ordering is dependency-driven, not priority-driven: items 8 and 12 needed the
digitals split first, and 16 is the gate everything else feeds.

## Done

- **1 — `HestonStochCorrPricer` double discount** (`d8a2f30`). A real 3.68 %
  mispricing: `exp(-r·tau)` applied inside `char_func_complex` and again in
  `price_call_carr_madan`. Verified against a from-scratch DOP853 + QUADPACK
  reference sharing no code with the crate; relative error fell 3.69 % → 6e-5.
  Found in passing: the two errors were **partially cancelling**, which is why a
  15 % cross-check tolerance looked healthy.
- **2 — Six non-`NaN` sentinel returns** (`f263dd6`, `a3ca7e4`, `98c2830`,
  `f201190`, `2ba9c34`). Two deliberately did *not* become panics, with reasons in
  the ledger. Turned up a crate-wide trap: `f64::max` discards a `NaN` operand, so
  any `.max(0.0)` floor converts poison into a plausible zero.
- **3 — The four digitals carried both conventions at once** (`06b8593`,
  `944207a`, `251f8cb`, `0de4035`). Each now holds `sigma` plus its own contract
  parameter. Dissolved two other items with it: `analytic_bs` no longer builds a
  pricer per query point, and the `d1` copies fell five → two.

## In progress

- **4 — Bound `ModelSurface` to the payoffs its inversion is valid for.** The
  blanket `impl<T: ModelPricer + ?Sized>` applies a European Black inversion, but
  the wave took the population from six vanilla types to eighteen, adding four
  digitals and three American pricers. A cash-or-nothing lands *inside* the
  no-arbitrage band, so the caller gets a finite, confident, meaningless surface.

## Queued — real defects

- **5 — `replication_weights` returns an all-zero vector** for `n < 2 ||
  maturity <= 0` (`variance_swap.rs:~210`). Same class as the six sentinels, and
  now inconsistent with the sibling `fair_strike_replication`, which panics on
  exactly those conditions. Found while closing item 2.
- **6 — Deep-ITM Carr-Madan inversion is unreliable.** At spot 100, `τ=2, K=0.01`
  returns **881 915.7**; `K=20` returns 10.46 against a lower bound of 77.98. The
  `K^{−α}` damping prefactor amplifies quadrature error. Root cause localised to
  `integrate_to_convergence`'s `tol = 1e-8` and width-50 initial panels — **not**
  the RK4 step, which moves the result by 2e-12. Prices at `K >= 0.2·S` are
  unaffected, which is why nothing catches it. Found while closing item 1.
- **7 — `HestonSlvPricer` spot anchoring.** `LeverageSurface` is indexed by
  absolute spot and `calibrate_leverage` takes a specific `s0`, so pricing far from
  it walks into the surface's clamped boundary, unguarded. Same "precomputed
  against something the query can contradict" shape as the rate bug fixed in
  `a3cb8a7` — honoured rather than discarded, but equally unchecked. While in the
  file: `slv.rs` is 572 lines against a 600 cap; the clean split is
  `slv/{mod,calibration,pricer}.rs`.

## Queued — soundness and discoverability

- **8 — Move the Heston `v0` validation to construction.** The assert added during
  the wave sits on the Greeks accessor, but `HestonPricer::new` is a `const fn`
  with no validation. So `new(-0.01, …).price_call(…)` returns a finite number off
  a negative variance and `.vol_surface(…)` builds a whole surface from it, while
  `.vega(…)` panics — one invalid parameter, two opposite responses chosen by which
  method the caller reaches for. Validating means giving up `const fn`; weigh it.
- **9 — Re-export `ShortRatePricer` from the umbrella `traits` hub.** `src/traits.rs`
  re-exports every other quant trait and omits this one. The prelude exclusion is
  deliberate and documented; the hub omission is not, and CLAUDE.md's own stated
  pattern for prelude-excluded traits is "reachable via `traits::*`". One line.
- **10 — Guard `LevyModel` and `CrrModel` in the pricer registry**, and close the
  registry's proven blind spot. Both implement `ModelPricer` and appear nowhere in
  the file, because its inventory derives from `pub struct *(Pricer|Engine)` and
  neither is named that way — while the file's opening line claims "every
  pricer/engine struct in this crate". The deeper fix is a runtime test that
  re-derives the inventory from source and diffs it against the hand-curated list;
  a probe struct added during the wave's own review left the test passing.
- **11 — Disambiguate the two meanings of `price` and `price_call`.**
  `KirkSpreadPricer::price_call(&self, f1, f2, x, r, tau)` shares the name and the
  5-`f64` arity of `ModelPricer::price_call(s, k, r, q, tau)` with different
  meanings in positions 1-4, and with the prelude imported the inherent method
  silently wins. Separately, `price` means two opposite things inside `pricing/`.
  Re-measure after item 3, which removed part of it.
- **12 — Give `DigitalOption` the same date story as `EuropeanOption`.** It carries
  the same `tau`/`eval`/`expiry` triple, does not implement `TimeExt`, has no
  `new_dates`, and its date fields are read by nothing — so a date-constructed
  digital silently prices at `NaN`. Decide at the same time whether `TimeExt`'s
  unfinished move toward the calendar module happens or is dropped: it now has one
  production implementor and 10 call sites, down from 35.

## Queued — consistency

- **13 — Convert the seven remaining multi-asset pricers to model/query.**
  `Margrabe`, `StulzRainbow`, `McRainbow`, `GeometricBasket`,
  `ArithmeticBasketLevy`, `McBasket`, `McSpread` still hold `s`/`k`/`r`/`tau`
  behind a no-argument `price()`; `KirkSpreadPricer` is the only one of the eight
  converted. Not mechanical — `Margrabe` splits cleanly, `GeometricBasket` has nine
  fields where the split is genuinely ambiguous. **Acceptable to ship in beta, not
  in 3.0.0 stable**: seven `pub` structs, and the beta window is where breaking is
  free.
- **14 — Decide what `GreeksExt` is for.** It remains no-argument — the retired
  convention — with four implementors, **zero** generic consumers anywhere, and a
  prelude slot, while every real Greek in the crate is now a query-taking inherent
  method. A newcomer importing the prelude gets a pricing trait on the new
  convention and a Greeks trait on the old one, side by side.
- **15 — Unify the doc phrasing for `tau`, `s` and `r`.** One *identifier* per
  concept was achieved; one *phrasing* was not. Measured at `9465768`:

  | field | fields | distinct wordings | most common |
  |---|---|---|---|
  | `pub tau:` | 38 | 12 | `Time to maturity in years.` (21) |
  | `pub s:` | 39 | **18** | `Spot.` (5) |
  | `pub r:` | 49 | 7 | `Risk-free rate.` (27) |

  ```
  cd stochastic-rs-quant/src && grep -rB1 "pub s: " --include="*.rs" . \
    | grep "///" | sed 's/.*\/\/\/ *//' | sort -u | wc -l
  ```

  **`s` is the worst of the three, not `tau`** — 18 wordings across 39 fields, with
  the most common covering only five. The final review's write-up put the emphasis
  on `tau`, where the canonical phrasing already covers 21 of 38; its per-field
  counts for all three were also off (42/26/42 against the measured 38/39/49).
- **16 — Align the `vega` signature across `ModelPricer` implementors.**
  `Merton1976Pricer`'s takes `option_type`; `HestonPricer`'s and `BSMPricer`'s do
  not. This item **owns a golden move**: `option_type` is consumed by
  `series_price` inside a central difference, so removing it shifts put-side values
  by finite-difference round-off. That is why the wave could not do it.
- **17 — Update the nine READMEs pinning `stochastic-rs = "2.6"`** against a
  workspace at `3.0.0-beta.1`. A user copying that line gets a crate where
  `PricerExt` still exists and `HestonPricer::new` takes 13 arguments.

## Gate

- **18 — Full release battery, then push.** 38 commits sit unpushed; nothing from
  this wave is on the remote. The battery **must** include
  `cargo test --workspace --exclude stochastic-rs-py --doc`: this repo
  `include_str!`s its README into the umbrella crate, so README blocks are real
  doctests that `cargo check` cannot see, and omitting that line is how a broken
  README doctest reached `main` mid-wave.

  Also unresolved: **this branch has never been built on x86_64.** Dev is
  aarch64-darwin, CI is ubuntu x86_64, and every golden in the wave was verified on
  one target only. One count-valued assertion was already loosened for this reason
  (`037c232`); nothing has swept for others.

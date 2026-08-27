# A2 follow-up queue

Work remaining after the A2 quant-consistency wave. The wave itself is closed and
recorded in [`A2_WAVE_LEDGER.md`](A2_WAVE_LEDGER.md); this is the queue of defects
and consistency work it surfaced but did not own.

Ordering is dependency-driven, not priority-driven: items 8 and 12 needed the
digitals split first, and 16 is the gate everything else feeds.

## Execution order

Fifteen open items grouped into seven steps. Several are too small to justify a
round of their own, so they travel with the item they share a file or a failure
mode with. Ordering is driven by two things only: a live wrong number outranks a
consistency wart, and **anything that breaks a `pub` API must land inside the
`3.0.0-beta` window**, because after stable it costs a second major version.

| Step | Items | Why here |
|---|---|---|
| **1** | 19, 20, 5 | Live sentinels returning a plausible wrong answer. All three are the shape already closed twice; 19 sits in the hottest path in the crate. |
| **2** | 21, 22 | Invalid state accepted at construction, surfacing later somewhere else. Same layer, same fix shape. |
| **3** | 6, 7 | Numerics. Both are quadrature or surface-extent problems needing an independent reference to verify, not a reread. |
| **4** | 10 | The registry's deep blind spot — a runtime inventory. Best done before the breaking work, so it guards it. |
| **5** | 13, 14, 11, 12 | **Breaking. Beta window or never.** 13 is the largest single item left. |
| **6** | 15, 16, 17 | Cosmetic and documentation. 16 owns a deliberate golden move. |
| **7** | 18 | The gate: full battery, then push. |

Step 5 is the one with a deadline. Everything else can slip past 3.0.0 without
costing anything but tidiness.


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
- **4 — `ModelSurface` bounded to the payoffs its inversion is valid for**
  (`3f0888e`, `c9b3ab8`, `f7feb7b`, `01e89ac`, `bd5f6ae`). **My own population
  figure was wrong**: I said eighteen, counting only named `impl ModelPricer`
  blocks. `ModelPricer` is itself blanket-implemented over `FourierModelExt`, so
  the real count was **30** concrete types (18 + 12, disjoint sets).

  Seven could not be described by the Black inversion, and each was *measured*
  rather than argued: `AssetOrNothingPricer` returned **10/10 finite** implied vols
  between 1.32 and 5.53 at σ=0.25. The three American pricers are worse than the
  digitals, not better — they land **within 0.008 of the model's own volatility**,
  so nothing in the output marks them as wrong.

  A **second instance with nothing to do with payoff shape** turned up in the
  process: `BSMPricer`/`Merton1976Pricer` under `Black1976`/`Asay1982` carry at
  `b = 0`, so their forward is `S`, not `S·e^{rτ}`. Inverting *correct* prices at
  the wrong forward fabricated a **0.080 → 0.150 smile out of a flat σ=0.20
  model**, every point finite. That is why the fix is not a bare marker: a bare
  marker would have been *false* for `BSMPricer` and the fabricated smile would
  have survived the change meant to kill exactly that class of bug. The marker
  carries `vanilla_call_forward(s, r, q, tau)`, so it is checkable rather than a
  comment with a compiler — and it also resolves `FiniteDifferencePricer`, whose
  exercise style is a runtime field a bare marker could only handle by breaking
  European FD surfaces or leaving American ones silently wrong.

  21/21 vanilla surfaces bit-identical by raw `to_bits()` diff; the only three that
  moved are the carry-mismatch cases, from wrong to right.

- **8 — Heston `v0` validation moved to construction** (`f6cdcb0`). `const fn` was
  dropped after measuring the cost: **33 call sites, zero const/static items** of
  that type, and the calibrator can't trip it — its optimizer runs in bounded
  logistic coordinates over a strictly admissible box. Validated `v0`, `theta`,
  `sigma` and `rho`, not just `v0`: fixing one would have swapped the old asymmetry
  for a new one, and `sigma == 0` is not a degenerate-but-priceable state here
  (`C`/`D` divide by `sigma²`, giving `NaN` with nothing naming the cause).
  Deliberately still accepted: any `kappa`, and any set violating Feller — a
  warning condition in this crate, not an error.

  The accessor guard **stayed**, because `new` is a front door and not a wall: the
  fields are `pub` and written directly in three in-tree sites. The two panic
  messages were made deliberately **non-nesting**, so a `should_panic` anchored on
  one cannot be satisfied by the other firing. Six of nine new tests were run at
  BASE and failed; the other three pin the boundary against over-tightening.
- **9 — Umbrella traits hub completed** (`f30776a`). **Three** traits were missing,
  not one: `ShortRatePricer`, `VanillaEuropeanCall` (added two commits earlier, same
  omission) and `ToShortRateModel` — the last paired with a `ToModel` that *was*
  already there. `tests/prelude_completeness.rs` now fails to compile if the hub
  drops one; it errored with three `E0432` at BASE. The prelude itself is unchanged
  at 28.

## In progress

(nothing)

## Queued — real defects

- **5 — CLOSED** (`2cdc2c0`). `replication_weights` now panics with its sibling's
  exact wording. `fair_strike_heston`'s old `tau <= 0.0` branch was **split**:
  `tau == 0` is the genuine `T → 0` limit of `(1-e^{-κT})/(κT) → 1` and stays, a
  negative or `NaN` `tau` panics. Six of seven new tests were confirmed failing at
  BASE; the seventh pins the deliberate keep.

  ~~Original:~~ `replication_weights` returned an all-zero vector for `n < 2 ||
  maturity <= 0` (`variance_swap.rs:~210`). Same class as the six sentinels, and
  now inconsistent with the sibling `fair_strike_replication`, which panics on
  exactly those conditions. Found while closing item 2.
- **6 — CLOSED** (`2488d33`). The error in `integrate_to_convergence` is now
  controlled. The proof is the part that matters: **all three pinned
  `heston_stoch_corr` goldens moved, and each moved toward the independently
  computed DOP853 + QUADPACK reference by a factor of ~2.6e8** — from ~3e-4 off to
  ~1e-12 off, i.e. to the reference's own precision.

  | quantity | \|old − ref\| | \|new − ref\| |
  |---|---|---|
  | `q=0` call | 3.006e-04 | **1.091e-12** |
  | `q=0.02` call | 2.945e-04 | **1.128e-12** |
  | `K=110` | 2.924e-04 | **1.502e-12** |

  **The band test's floor came all the way down**, `K = 20` → `K = 0.01` — the exact
  strike that used to return 881 915.7 against a spot of 100. Its doc comment had
  said it stopped at 20 to avoid asserting the quadrature rather than the model;
  that reason no longer holds. The inversion was also found wrong at **every**
  strike on the grid, not only deep ones — the deep ones were merely where the
  `K^{−α}` prefactor made it visible.

  ~~Original:~~ deep-ITM Carr-Madan inversion was unreliable At spot 100, `τ=2, K=0.01`
  returns **881 915.7**; `K=20` returns 10.46 against a lower bound of 77.98. The
  `K^{−α}` damping prefactor amplifies quadrature error. Root cause localised to
  `integrate_to_convergence`'s `tol = 1e-8` and width-50 initial panels — **not**
  the RK4 step, which moves the result by 2e-12. Prices at `K >= 0.2·S` are
  unaffected, which is why nothing catches it. Found while closing item 1.
- **7 — CLOSED** (`19e2982` split, `8c2aa86` guard). The bound is the **surface's own
  grid extent** — `spots[0]..=spots[last]`, `tau in 0..=times[last]` — not a picked
  tolerance, and out-of-extent queries return documented `NaN`.

  **What the surface actually does at its edge:** a *flat hold* of the nearest edge
  value, forever, on both axes and in both directions. Measured — and the
  measurement is what settles it: two surfaces with **identical edge columns** but
  interiors differing 3x returned **bit-identical** prices at `s = 1000`, while
  disagreeing by **95 %** at the calibration spot. Far enough out, the calibrated
  interior contributes literally nothing.

  **How invisible it was:** the clamped price at `s = 1000` was **60 % low** against
  a surface that carried the calibrated shape out there — yet finite, positive,
  strictly inside the no-arbitrage band, and **within 0.2 % of a plain Black price
  at sigma = 0.2**. Nothing marked it.

  **Why the grid edge and not a tighter bound:** the "interior discarded" regime
  does *not* start at the edge — at `s = 131` and `s = 60` paths still diffuse back
  in. Any tighter line would be a diffusion distance depending on `tau`, `v0` and
  `L` itself. The grid edge is the only line the **data** draws.

  **`NaN`, not panic**, and correctly reasoned: the crate's convention names *"a
  strike outside a Fourier pricer's truncation grid"* as case 2, which is literally
  this shape. The rate anchor panics because it contradicts state the pricer
  *recorded*; a moved spot is a market state. The rate check runs first, so a query
  wrong on both counts still panics. Unlike the rate anchor this applies to **both**
  constructors — extent belongs to the surface, rate provenance only to a
  calibration.

  **Two bonus defects, both predicted by the standing trap list.** A `NaN`
  coordinate was **laundered into an edge value** (`L(NaN, 0.5) = 1.9`), because
  `fractional_index` falls through its loop and returns `n-1`. And
  `price_call(-100, …)` returned a confident **`0.0`**: `ln(s)` is `NaN` for `s < 0`
  and `(NaN - k).max(0.0)` is `0.0` — the `f64::max` trap again, third sighting.

  **Scope note:** the *maturity* half of the same clamp was closed with it (6.8 %
  error at `tau=1`, 22.7 % at `tau=2` past a 0.5 horizon). One object, one flat
  hold, one silence — closing half would have left the defect live.

  `slv.rs` (577 lines, the queue's 572 was stale) split into
  `slv/{mod,calibration,pricer}.rs` + `pricer/tests.rs` at 306/207/218/181, all
  under the 400 soft target. Verified a pure move.

  ~~Original:~~ `HestonSlvPricer` spot anchoring `LeverageSurface` is indexed by
  absolute spot and `calibrate_leverage` takes a specific `s0`, so pricing far from
  it walks into the surface's clamped boundary, unguarded. Same "precomputed
  against something the query can contradict" shape as the rate bug fixed in
  `a3cb8a7` — honoured rather than discarded, but equally unchecked. While in the
  file: `slv.rs` is 572 lines against a 600 cap; the clean split is
  `slv/{mod,calibration,pricer}.rs`.

## Queued — soundness and discoverability

- **10 — CLOSED** (`1505ff9`). `registry_matches_crate_source` re-derives the
  inventory from `src/**/*.rs` with `syn` and diffs it against the hand-curated
  lists, so the lists are now the assertion rather than the source. The parser is
  the point: a regex over lines is the technique behind thirteen counting errors
  in this project, one of them inside `pricer_registry.rs` itself.

  **Two signals, unioned, because neither alone closes it.** Trait membership
  (`ModelPricer`, `PricingEngine`, `ShortRatePricer`, `VanillaEuropeanCall`)
  catches what name shape misses — `LevyModel`, `CrrModel`, `Cir`, `HullWhite`,
  `Vasicek`. Name shape (`pub struct *Pricer` / `*Engine`, matched on the parsed
  identifier, so `*PricerBuilder` and `*EngineConfig` fall out structurally)
  catches the orphan, which by definition implements nothing. Everything in the
  union must sit in a trait list or in `NO_TRAIT_BY_DESIGN` with a reason.

  **Teeth proven the way the blind spot was.** Both probes were added to
  `pricing/asian.rs` and both failed the test: `ReviewProbeOrphanPricer` — the
  same struct that passed unnoticed during the wave's own review — and a
  `ReviewProbeQuietModel` carrying `ModelPricer` under a model's name.

  Turned up in passing: `AnalyticBSEngine`'s second instrument (`DigitalOption`)
  was a prose caveat, not an assertion. The engine list is now
  `(engine, instrument)` pairs and went 2 → 3. The `FourierModelExt` blanket
  impls are pinned too, so a *second* blanket widening what `ModelPricer` covers
  is a failure rather than a silent change. **Residual blind spot**: a struct
  that neither carries a pricing trait nor is named `*Pricer`/`*Engine` — a
  `FooModel` with an inherent `price()` — is in neither signal. Rust cannot
  enumerate a trait's implementors at run time, so closing that needs a signal
  the compiler exposes.
- **11 — CLOSED** (`e3739b1`), and **renamed rather than only documented**:
  `spread_call_put` / `spread_call` / `spread_put`.

  **My shadowing claim was wrong.** `KirkSpreadPricer` does not implement
  `ModelPricer`, so no trait method was being shadowed. The real defect is arity
  plus type identity: five `f64`s, same name, meanings differing in four of five
  positions, nothing for the compiler to catch.

  **And the sharper instance was one I had not listed.** `call_put` has **eight**
  siblings taking `(s, k, r, q, tau)` against Kirk's `(f1, f2, x, r, tau)`.
  Renaming only `price_call`/`price_put` would have fixed the smaller half and left
  the worse one live — so all three moved together.

  **Measured what the confusion actually produced:** reading Kirk's query as a
  vanilla one returned **10.943403655286877** — finite, positive, and squarely
  inside the band a Black call at those inputs would occupy. Pinned as a test so
  the rename cannot later be reverted as cosmetic, with a `compile_fail` doctest
  proving the old name no longer resolves.

  Kirk is the **only** member that needed it: the other seven are separated by
  their signatures already (`ArrayView1` legs, or no strike at all). Re-measured
  in passing: 11 no-argument `price(&self)` against 6 query-taking `price(` in
  `pricing/`, all path-dependent and out of scope.
  `KirkSpreadPricer::price_call(&self, f1, f2, x, r, tau)` shares the name and the
  5-`f64` arity of `ModelPricer::price_call(s, k, r, q, tau)` with different
  meanings in positions 1-4, and with the prelude imported the inherent method
  silently wins. Separately, `price` means two opposite things inside `pricing/`.
  Re-measure after item 3, which removed part of it.
- **12 — CLOSED** (`1b9906b`). `DigitalOption` implements `TimeExt` and gains dated
  constructors; `AnalyticBSEngine::digital_query` resolves through
  `tau_or_from_dates()` instead of reading `opt.tau` directly. The test asserts
  both halves — that a dated digital prices at all, **and** that it prices
  bit-identically to its explicit-`tau` twin; `is_finite()` alone would have passed
  on any wrong number.

  **`TimeExt`'s unfinished move is dropped, not deferred**, and recorded in three
  places. The reasoning: the pricer half *already happened* (`PricerExt: TimeExt`
  is gone, no pricer implements it), and the "one implementor" premise was itself
  out of date — this item took it from 1 to **2**, both instruments. The calendar
  module already owns the arithmetic (`DayCountConvention::year_fraction`, which
  both derivations call); what `TimeExt` adds is *which* maturity slot is
  populated, an instrument concern. Relocating it would put that inside a
  date-arithmetic module.

  No validation added, deliberately: `EuropeanOption::new_dates` accepts
  `expiry < eval`, and guarding only the digital would swap the asymmetry this item
  exists to remove for a new one. It carries
  the same `tau`/`eval`/`expiry` triple, does not implement `TimeExt`, has no
  `new_dates`, and its date fields are read by nothing — so a date-constructed
  digital silently prices at `NaN`. Decide at the same time whether `TimeExt`'s
  unfinished move toward the calendar module happens or is dropped: it now has one
  production implementor and 10 call sites, down from 35.

## Queued — consistency

- **13 — CLOSED** (`f12ee4c` … `3232496`, nine commits, one per pricer). The whole
  eight-member multi-asset family is now on one convention; all eight structs sit
  at 4-6 fields. **43 analytic goldens are bit-identical** to BASE, captured on
  deliberately asymmetric configurations — distinct spots, weights, both yields,
  negative off-diagonal correlation, non-round strike, non-unit maturity — so none
  could survive by coinciding with a default. 20 are now pinned in-tree at 1e-12.

  **The rule that made it uniform:** `OptionType` is a *method selector*
  everywhere; every other contract enum or number stays on the struct, as
  `cash`/`k2`/`x_high` do on the digitals.

  **Margrabe was verified rather than assumed**, and it is the clean case for a
  reason: `σ² = σ1²+σ2²−2ρσ1σ2` really is model-only — exactly the property Kirk
  lacks. It is still recomputed per call rather than cached, so no field is ever a
  number left over from a query. It has **no `r` at all** (an exchange option's two
  discount factors cancel), documented as an *absence* so nobody "fixes" it later.

  **`GeometricBasketPricer`'s `weights` were the ambiguous case, and both readings
  agree:** contract, because `∏ Sᵢ^{wᵢ}` is what the term sheet writes; and
  inseparable from the model, because `σ_G² = Σ wᵢwⱼρᵢⱼσᵢσⱼ`. `σ_G` *could* be
  cached honestly — no query enters it — and deliberately is not, because `μ_G` and
  the geometric forward both carry the query, and caching one of three would put a
  struct field next to two that can never be one.

  **`n_paths` sits on the struct as method state**, decided by in-crate evidence
  rather than taste: `GbmMalliavinPricer` already documents itself as holding
  "model and method state only — the volatility, the Monte Carlo path/step counts".

  **One honest wart, flagged not hidden:** `RainbowPayoff` is a contract term but
  *contains* a call/put axis, so the family departs from `price_call`/`price_put`
  in exactly one place. Splitting it into `{Max,Min}` × `OptionType` would break a
  second public enum, past this item's remit. Kept whole, written into the doc.

  **Validation deliberately omitted**, following this project's own Kirk sequencing
  — reshape (`dad4a78`) then validate (`bc3afc5`) as separate commits. The
  dimension and SPD checks stayed in `try_price` rather than moving to `new`,
  because `try_price` is the only advertised way to surface them as `Err` and a
  panicking constructor would leave it nothing to report.

  The registry **passed unchanged and correctly** — no type changed category — but
  its *prose* had become false ("the other seven still bundle their query"). That
  was rewritten as a factual correction to a comment, **not a list edited to match
  the code**.

## Queued — found while closing step 5a

- **27 — CLOSED** (`5728af4`). All four now propagate. Two findings beyond what I
  listed: the `4.877057549928611` is **exactly** the price of the same basket at
  `sigma = [0, 0]`, so the test asserts that *identity* rather than the constant;
  and `RainbowPayoff` had **two** copies of the trap, not one — `CallOnMin` and
  `PutOnMax` returned `0.0` through a surviving `(min_p - k).max(0.0)` floor, so
  fixing only the fold would have left half the payoffs laundering. Every floor is
  preserved where it is genuinely a floor.

  ~~Original:~~ four live `NaN`-laundering defects, measured
  rather than argued, preserved byte-for-byte because fixing them is item 19's
  shape and would have wrecked the one-commit-per-pricer property:
  - **`ArithmeticBasketLevyPricer` with a `NaN` *model* `sigma` returns
    `4.877057549928611`** — a plausible ATM basket call. `basket.rs:276`'s
    `(m2/(m1*m1)).ln().max(1e-14)` turns `NaN` into `1e-14`, so `σ_eff ≈ 1e-7` and
    the price collapses to the zero-vol intrinsic. Nothing in the query is wrong;
    a poisoned *model* parameter yields a healthy-looking number. The sharpest of
    the four.
  - **`RainbowPayoff::evaluate` silently drops a `NaN` leg** (`rainbow.rs:54-55`).
    `CallOnMax` on `[120, NaN, 90]` at `K=100` returns **20.0** — a three-asset
    best-of prices as a two-asset best-of.
  - `MargrabePricer::price` returns **`0.0`** for `tau = NaN` on the degenerate-vol
    branch — and `tau` arrives as `NaN` legitimately from `TimeExt`.
  - `McSpreadPricer` returns **`0.0`** for a `NaN` spot *or* a `NaN` model `rho`;
    the per-path floor zeroes every poisoned payoff and averages them.

  Clean under the same probe: Stulz, Geometric basket, and Levy with a `NaN` tau.
- **28 — CLOSED** (`f764f14`). Guarded by *measuring the wrong number each admits*
  — `StulzRainbowPricer` at `sigma2 < 0` returned a **negative call** (−11.38);
  `MargrabePricer` at `rho = 5` returned exactly the intrinsic, because a negative
  combined variance trips the degenerate branch.

  **Two deliberate omissions, both better-reasoned than a blanket rule.** The MC
  array pricers' `rho` stays unguarded because
  `mc_rainbow_try_price_reports_a_non_spd_correlation` constructs `[[1,2],[2,1]]`
  — an element-range check at `new` would have **pre-empted the exact test that
  pins `try_price`'s role**, and a correlation check that inspects the diagonal but
  not the entries is the asymmetry item 22 warns against. And the **weight sum** is
  not a domain constraint: `w = [-1, 2]` is a long/short basket, a real product. A
  field doc falsely asserting "must sum to one" was corrected.

  ~~Original:~~ the seven had unguarded public constructors that item 22's standard
  would guard. Owed by the reshape-then-validate sequencing above.


  `Margrabe`, `StulzRainbow`, `McRainbow`, `GeometricBasket`,
  `ArithmeticBasketLevy`, `McBasket`, `McSpread` still hold `s`/`k`/`r`/`tau`
  behind a no-argument `price()`; `KirkSpreadPricer` is the only one of the eight
  converted. Not mechanical — `Margrabe` splits cleanly, `GeometricBasket` has nine
  fields where the split is genuinely ambiguous. **Acceptable to ship in beta, not
  in 3.0.0 stable**: seven `pub` structs, and the beta window is where breaking is
  free.
- **14 — CLOSED** (`178fdd0`), by **demoting rather than deleting or rebuilding**.
  Out of the prelude (28 -> 27), kept in both traits hubs.

  **The premise's own number was stale, and correcting it cut both ways.**
  `GreeksExt` has **two** implementors, not four — item 3 stripped the two digitals'
  fields and their impls went with them. The survivors, `GbmMalliavinGreeks` and
  `HestonMalliavinGreeks`, are **not pricers**: they are Monte Carlo estimator
  objects that legitimately own their query.

  **The finding that decided it:** five analytic aggregators already share **one
  identical signature** — `greeks(&self, s, k, r, q, tau, option_type) -> Greeks`
  across `BSMPricer`, `HestonPricer`, `Merton1976Pricer`, `CashOrNothingPricer`,
  `AssetOrNothingPricer`. Five consecutive Greek-bearing additions **declined** the
  trait because they structurally could not satisfy it. It is not a trait waiting
  for implementors; it is a trait its natural candidates cannot join.

  **Why not delete:** `HestonMalliavinGreeks::greeks()` exists *only* through the
  trait — there is no inherent `all_greeks`. And for the two MC types the bundled
  query is **correct**, not a wart: one simulation, nine consistent estimators. A
  query-taking signature would re-simulate per accessor.

  **Why not invent `ModelGreeks`:** five implementors, still zero consumers.
  Trading a trait nothing calls for another trait nothing calls is churn. It remains no-argument — the retired
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

## Queued — found while closing item 4

- **19 — HALF CLOSED** (`f6ca4d7`). The three `.max(0.0)` sites in
  `pricing/fourier/pricer.rs` now route through
  `floor_price(x) = if x.is_nan() { x } else { x.max(0.0) }`, separating the floor
  (a genuinely negative deep-wing price is quadrature round-off and should be
  floored) from the poison check (a `NaN` has no price to floor).

  **The stated mechanism held at one site of three, and the truth is worse.** A
  fully-`NaN` characteristic function never reaches the Gil-Pelaez or Lewis floors
  at all: `integrate_to_convergence(|_| NAN, …)` returns **`0.0`**, so `p1 = p2 =
  0.5` and those pricers returned **2.438528774964297** and **100.00000000000001**
  (the spot) — well-scaled fake prices, not zeros. What actually reached those two
  floors as `NaN` is a non-finite *market input*, `tau` included — and `tau`
  arrives as `NaN` legitimately from `TimeExt::tau_or_from_dates`, so an option
  whose expiry never resolved priced at zero through the crate's busiest path.
  That half is now fixed.
- **19b — CLOSED** (`b0e01b6`). A `Cell`-based watcher poisons the result when the
  integrand is `NaN` at any node the rule evaluates. **The precise mechanism is
  sharper than "the quadrature eats a NaN":** `double_exponential::integrate`
  rewrites every non-finite sample to `0.0` *before the rule sees it*, which is
  why no downstream floor could ever have caught it.

  **`±∞` deliberately keeps the third-party behaviour.** An overflowing integrand
  is a different case from an undefined one, and the crate's own Lévy loss
  integrand reaches `∞` transiently on unprojected calibration iterates — poisoning
  there would abort a run that currently recovers. Same shape as the decision to
  leave `BSMPricer::new` unvalidated in item 22: the rule is right in the abstract
  and wrong against a live caller. **Do not "fix" this.**

  ~~Original:~~ the quadrature swallowed a `NaN` integrand
  `pricing/cf_quadrature.rs`'s `integrate_to_convergence` returns `0.0` for an
  integrand that is `NaN` everywhere, so item 19's headline — a `NaN` chf yielding
  a plausible price across all 12 Fourier models — is **still live** on the two
  quadrature paths. Only the FFT path is closed. The swallow originates in the
  third-party `quadrature` crate's `double_exponential`; our wrapper is the only
  place we control, so the guard belongs there.
- **19c — CLOSED** (`65d86d9`). A `NaN` vol-of-vol now propagates through the
  volatility-swap strike instead of being floored to a plausible `0.2`.

  ~~Original:~~ the same trap, one line from a file already fixed
  `variance_swap.rs:372` — `Self::fair_strike_from_var(k_var, dispersion.max(0.0))`.
  A `NaN` `sigma` passes the `k_var > 0.0` guard, because `k_var` does not depend
  on `sigma`, and the floor then turns the `NaN` dispersion into `0.0`. Measured:
  `VolatilitySwapPricer::fair_strike_heston(0.04, 1.5, 0.04, NAN, 1.0)` returns
  **0.2**, exactly `sqrt(k_var)` — indistinguishable from a real zero-dispersion
  strike.
- **20 — CLOSED** (`2bda294`). `is_arbitrage_free` returns `Option<bool>`; `None`
  is the `bool` analogue of the convention's case-2 `NaN`.

  **The cause was not `NaN` comparison.** `ImpliedVolSurface::smile_slice`
  **deliberately drops** non-finite IVs — pinned by its own
  `smile_slice_filters_nans` test — so an all-`NaN` surface leaves *empty* grids
  and both arbitrage checks are universal quantifiers over an empty set.
  `check_butterfly_ssvi` returned `(true, inf)` off its untouched `+inf` seed, and
  `atm_total_variance` hit its `n == 0` branch returning `0.0`, making `[0.0, 0.0]`
  trivially non-decreasing. "Nothing violated" and "nothing checked" collapsed to
  one answer.
- **21 — CLOSED** (`8ad5e81`). The `0/0` point is a **removable singularity**, so it
  is computed rather than documented away. Along `σ → 0⁺` at the forward,
  `d₁ = σ√τ/2 → 0⁺` and `d₂ = −σ√τ/2 → 0⁻`, so **both CDFs converge to ½** and the
  term tends to `½(Se^{(b−r)τ} − Ke^{−rτ})` — which is 0, because being at the
  forward *is* `Se^{bτ} = K`. The fix writes that as an expression rather than the
  constant `0`, so a non-finite `r` still propagates instead of becoming a
  confident zero.

  **Why it went unnoticed, and it is the carry fix that explains it:** under
  `Black1976`/`Asay1982` `b = 0` puts the forward at `S`, so the singular strike
  *is* the ATM strike — the most-quoted point on a futures-option surface. Under
  the carrying conventions it sits at `Se^{bτ}`, a strike nobody quotes exactly.
  Proved not to be a `BSMCoc` special case: `Bsm1973` at `r = 0` moves its forward
  onto the strike and reproduces the `NaN`.

## Queued — found while closing items 8 and 9

- **22 — CLOSED** (`bc3afc5`). Thirteen constructors now validate and are no longer
  `const fn`. The split was **measured, not reasoned**: a probe constructed all 14
  with invalid parameters and recorded what came back. 13 of 14 returned a wrong
  *number* rather than an obvious failure — `AsianPricer` a **negative call**
  (−4.455) at `v = −0.25`, `SabrPricer` **the spot** (100.0) at `beta = 5`,
  `FiniteDifferencePricer` 0.667 at `t_n = 0`.

  **`BSMPricer::new` is deliberately left unguarded**, and this is the finding
  worth keeping: three measured callers depend on it accepting what a guard would
  reject. `BSMCalibrator::residuals`/`jacobian` construct it from an
  **unprojected** Levenberg-Marquardt iterate, so a guard would abort a live
  calibration on a transient negative step; `AnalyticBSEngine` builds it from
  `read_quote(volatility)`, which is `NaN` for an unlinked handle **by design**;
  and `Merton1976Pricer::term_bsm(0, ·)` constructs it at `v == 0` on every single
  price. Guarding it needs `BSMCalibrator` to gain a projection box first — its
  own item, below.

  **Two places where copying the Heston template would have been wrong:**
  `HestonStochCorrPricer` gets `sigma_v >= 0`, not `> 0`, because its Riccati
  system only ever *multiplies* by `sigma_v` (Heston's `> 0` exists because its
  closed form divides by `σ²`) — zero vol-of-vol is the deterministic-variance
  limit. And the digital `sigma` guards **permit `NaN`**: the inverse trap fired
  for real here, `sigma >= 0.0` turning an `analytic_bs` test red, because
  `read_quote` documents `NaN` as missing data and it flows straight into
  `CashOrNothingPricer::new`.

  ~~Original:~~ `HestonStaticParams::new` was an unvalidated `const fn`
  (`pricing/engines/analytic_heston.rs:38`), feeding the now-validated
  `HestonPricer::new` at line 107 — so an invalid parameter surfaces at pricing
  time from the inner constructor rather than where the caller supplied it. Item 8
  one layer up. Thirteen other `pub const fn new` constructors in `pricing/` shared
  the shape (the command below counts 14 **including** this one):
  ```
  grep -rn "pub const fn new" --include='*.rs' stochastic-rs-quant/src/pricing/ | wc -l
  ```

## Queued — found while closing step 2

- **23 — CLOSED** (`05a9b5a`, `e393783`). **The crate's formula was wrong, and this
  is the most consequential finding of the whole effort.** `sigma_n` was
  `sqrt((d² + z²)·n/tau)`, scaling the *diffusive* variance by the jump count.
  Conditional on `n` jumps the diffusion runs for the whole of `tau` however many
  jumps land in it, so `Var = d²·tau + n·z²` and `sigma_n = sqrt(d² + z²·n/tau)`,
  giving `sigma_0 = d`.

  **Confirmed against the primary source**, with a citation correction: Merton
  (1976) MIT Sloan WP 787-75 §III eq. (18) — *"a Black-Scholes option where the
  formal variance per unit time on the stock is σ² + nδ²/τ"* — and Haug §6.9.1 at
  **pp. 253-255**, not the 205-207 the crate cited. The old formula matches no
  source and is provably not a reparameterisation of one.

  **Six independent adjudications, run before any golden was touched:** the
  `gamma = 0` limit (must be Black-Scholes — new formula exact to 2e-15, old off by
  up to 16.20 against 4.58), the `lambda -> inf` CLT, `E_N[sigma_N² tau]` against
  the declared `v² tau` (old: −30 % to +40 %), Gil-Pelaez CF inversion sharing no
  code (old: −54 %), an 8M-path Monte Carlo (**old formula: −758 standard
  errors**), and **Haug's own published examples** — 0.241746 against his 0.2417,
  21.7354755 against his 21.735476.

  **All 14 goldens moved; zero survived.** The reference call went
  `1.9630 -> 4.2761`, a **54 % underprice**, and `volga` **changed sign**
  (−0.4009 -> +3.6928). Haug's examples are now the only pin in the crate whose
  expected value comes from published literature.

  **The prediction discipline is worth copying:** a bit-exact Python replica was
  validated by reproducing **all 14 old goldens bit-for-bit** before it was
  trusted to predict the new ones — so the replica was proven against known
  values, not against the answer it was about to give.

  **Item 21 is not redundant, but its reachable set collapsed.** With
  `sigma_0 = d`, the `n = 0` term is degenerate only where the *diffusive* vol is
  exactly zero. Its test module was retargeted from `v = 0.2, gamma = 0.4` to
  `v = 0` — from firing on every ATM futures price to firing only on a frozen
  underlying. The crate uses
  `σₙ = √((d² + z²)·n/τ)`; Haug's is `σₙ = √(d² + z²·n/τ)`, giving `σ₀ = d` — the
  diffusive vol — rather than 0. **That difference is why the `n = 0` term is
  degenerate at all.** If the formula is wrong, item 21 is filling a hole that
  should not exist. Not touched because correcting it moves every Merton golden,
  so it needs an item that owns that move.
- **24 — CLOSED** (`171d84a`). **The price was wrong; the Greeks were right.**
  `lambda = 0` means no jumps, so the model *is* Black-Scholes at `v`. The `NaN`
  came from evaluating the per-jump *size* `z = sqrt(v² gamma / lambda)` — a
  quantity the model never needs, since only `lambda z²` enters.

  **The limit is genuinely discontinuous, and that is pinned too.** At fixed
  `gamma`, `lambda -> 0+` sends `z² -> inf`: jumps get rarer *and* larger, holding
  their variance share all the way down, so the price tends to
  `BS(v·sqrt(1−gamma))` = **3.3205**, not to the value at `lambda = 0` = **4.5817**.
  The two coincide only at `gamma = 0`. Asserting both halves stops anyone
  "fixing" the discontinuity by assuming continuity.

  **`lambda < 0` deliberately left disagreeing**, with the reason recorded:
  narrowing the Greeks' `lambda <= 0` branch to `== 0` would route a negative
  intensity into `greek_series`'s `NaN` floor and return a confident **`0.0`** —
  strictly worse than the disagreement.

  ~~Original:~~ price and Greeks disagreed at `lambda = 0`
  return the Black-Scholes value.** `greek_series` has an explicit
  `if self.lambda <= 0.0` branch; `call_put` has none, so `jump_size_std` hits
  `v²γ/0`. Price and Greeks disagree about whether `λ = 0` is a supported state —
  and `merton_greeks_lambda_zero_equals_bs` pins the Greeks side, so the
  disagreement is asserted rather than accidental.
- **25 — NOW FULLY CLOSED** (`6352360` mitigation, `c2a6adc` seed source).
  `GbmMalliavinPricer<S: SeedExt = Unseeded>` takes a seed **last**, matching
  `Gbm::new(mu, sigma, n, x0, t, seed)` — the very process it drives — rather than
  `MCBarrierPricer`'s `price_seeded(..., seed)`, which puts the seed on the query
  where this ratio-over-a-path-block estimator cannot use it. It `clone()`s rather
  than `derive()`s, because `derive` advances the pricer's own state and two
  identical queries would still differ.

  Seeds `[2718, 999, 42]` are the triple **printed verbatim** in the crate's own
  testing skill, taken as printed rather than searched for — no seed-fishing.
  Best-of-three is kept on top and that is deliberate: the SIMD stream differs
  between aarch64-darwin and the x86_64 CI runner, so any single seed verified here
  is unverified there. Only the *source of independence* changed, from entropy to
  three pinned streams.

  ~~Earlier:~~ mitigated only (`6352360`), and the agent was explicit that
  it could not follow the crate's own §1.1: **`GbmMalliavinPricer` has no seed to
  pin.** `sample_paths` builds `Gbm::new(…, Unseeded)` internally and the struct's
  fields carry no seed source, so pinning one means adding a field to a `pub`
  struct — externally breaking and out of scope.

  It measured the hazard rather than estimating it: **17 of 2000 runs (0.85 %)
  breach `c <= S`**, about one CI run in 118, worst point **12229.43** against a
  spot of 100. And **more paths make it worse** — 400 -> 2000 moved the worst
  observed call from 57.0 to 89.3 — because the estimator is a ratio whose
  denominator is a Heaviside-weighted count that can be nearly empty.

  Applied §1.2 instead (best-of-three replication): **0 breaches in 666 groups**,
  a ~6e-7 false-failure rate. `is_finite` is asserted *before* the running `max`,
  since `f64::max` would discard a `NaN` and hand back a plausible number.

  **Follow-up owed (beta window):** give `GbmMalliavinPricer` a seed source and
  convert to three pinned seeds. Two sibling tests in the same file are unseeded
  with the same estimator and the same tail. It failed once with
  `call 103.976 out of bounds` against its own `c <= S = 100` bound, then passed
  5/5 on re-run. **Unseeded** Monte Carlo (`Gbm::new(…, Unseeded)`). The upper
  bound is violated, not just monotonicity, so this is a genuine CI hazard rather
  than a tolerance question — and the crate's own testing conventions mandate
  pinned seeds for exactly this reason.
- **26 — CLOSED** (`f66ebc7`), matching `SabrCalibrator` — same optimiser, same
  parameter shape — rather than Heston's bounded-logistic coordinates (which would
  have required rewriting the analytic Jacobian chain rule) or HSC's `BOUNDS`
  (a different optimiser). **Reflection, not clamping**: an overshoot to `-0.3` is
  told `0.3`, so the calibration does not stall.

  **`set_params` alone was not enough, and testing caught it:**
  `LevenbergMarquardt::minimize` prices the *starting* point before it has any step
  to hand back, so a directly-written `pub params` still reached `BSMPricer::new`
  unprojected. Three projection sites, three tests, **each verified to fail when
  its own site is removed**.

  **Blocker status: `BSMCalibrator` is cleared, `BSMPricer::new` is still not
  validatable.** The two other cited callers are live and confirmed —
  `AnalyticBSEngine` feeds it an unlinked handle's `NaN` *by design*, and
  `Merton1976Pricer::term_bsm(0, ·)` constructs at `v == 0`. Item 29 made the
  latter **more** frequent: at `gamma = 1` the `n = 0` term is now exactly `0` on
  every price, where before it was `0` about half the time. `set_params` writes the raw
  Levenberg-Marquardt vector, so the optimizer can and does pass through negative
  volatilities. Every sibling calibrator (`SabrCalibrator`, the HSC bounds) projects
  into a strictly admissible box. This is the blocker on item 22's one deliberate
  omission.

## Queued — found while closing step 6a

- **29 — CLOSED** (`296a4b6`), and **the formula I put in this queue was wrong at
  two corners.** I wrote that `d² = v²(1−gamma)` is "exact and `lambda`-free". The
  agent checked before applying and found the naive substitution moves **299 of 704
  grid points**:
  - At **`lambda == 0`**, `jump_size_std` deliberately early-returns `z = 0`, so
    `d = v`, *not* `v·sqrt(1−gamma)`. `merton_price_lambda_zero_equals_bs` asserts
    this — and `the_lambda_zero_limit_is_discontinuous_in_gamma` exists precisely
    to pin that `v·sqrt(1−gamma)` is the **limit**, a different number. The bare
    formula would have broken both.
  - Where **`gamma` and `lambda` have opposite signs**, `z` is imaginary and the
    round-trip announced `NaN`; the bare formula silences an announcement `new`
    documents.

  What shipped keeps both branches and removes only the round-trip, and tests
  `jump_size_std()`'s own output rather than re-deriving the condition — so the
  `NaN` set is identical to the round-trip's **by construction**. All six reported
  `(v, lambda)` pairs now give `d = 0`; a 400×4 sweep at `gamma = 1` gives `d == 0`
  everywhere, against 28 exceptions at BASE. Nothing moved: worst relative change
  on ordinary configurations is **9.9e-15**. It computes
  `v² − lambda·(sqrt(v² gamma / lambda))²` where `lambda z² = v² gamma`
  analytically. At the pure-jump corner `gamma = 1` this lands on `d = 0` for
  some `(v, lambda)` pairs and on **`NaN`** for others — a valid model that prices
  or doesn't by floating-point luck. `d² = v²(1−gamma)` is exact and
  `lambda`-free.
- **30 — CLOSED** (`629e0aa`), and it was **five live errors in two files, not
  one**. The four in `stochastic-rs-stochastic/src/autoregressive/arima.rs:149`
  were already errors under the same flag and are the same trap without the LaTeX:
  `X[0] = Y[0], X[t] = X[t-1] + Y[t]` in a private fn's doc.

  **It matched the project's own prior fix** rather than inventing one: commit
  `d016a61` had converted `$E[C]$` to `$E\left[C\right]$` crate-wide, and
  `pricing/fourier/levy.rs:171` already carries the **identical sentence**
  correctly — `loss.rs` was simply the copy that got missed, because it sits on a
  `pub(super)` fn only `--document-private-items` reaches.

  A shape sweep found **27 bracket candidates in 14 files**; the other 24 are inert
  and that was *proved*, not assumed — `cargo doc --workspace
  --document-private-items` now exits 0, and they sit inside doctest fences or on
  `#[cfg(test)]` items rustdoc never documents. **They become live the moment such
  an item stops being test-gated.**
  `calibration/levy/loss.rs:12` writes `$E[S_T] = …$` and `[S_T]` parses as a
  link. Today it errors only under `cargo doc --document-private-items`; plain
  `cargo doc` passes. **One flag away from a hard CI failure.**
- **31 — CLOSED** (`9cc336b`), by **narrowing rather than removing**. The floor's
  stated justification covered **one** case and it was catching **seven** — and all
  seven already had a `NaN` *price*, so price and Greeks disagreed about every one
  of them: unresolved `tau` from `TimeExt`, `NaN` `r`/`s`/`k`, negative spot or
  strike, `tau <= 0` or infinite, a `gamma` outside `[0,1]` that `new` documents as
  *announcing itself*, and an overflowing Poisson weight. Now
  `contribution.is_nan() && term.v == 0.0`; degenerate configurations are
  bit-unchanged, everything else propagates.

  **The residual is pinned with both the wrong values and the right ones**, so it
  cannot drift: *at* the forward a degenerate term's `d1` is `0/0` and the floor
  returns `0.0` for delta, gamma and rho, where the `sigma -> 0+` limits are
  0.48765, `+inf` and 24.3827 — measured along `v = 1e-3 … 1e-6`. Fixing that needs
  a per-Greek limit nine times over, so it is its own work.

  ~~Original:~~ the `NaN` floor whose justification item 23 removed (`if contribution.is_nan() { 0.0 }`) is the
  crate's named laundering shape, and item 23 removed its stated justification.
  Re-documented honestly rather than removed, because removing it changes
  degenerate-config Greeks.

## Queued — found while closing step 6c

- **32 — A degenerate Merton term's Greeks are wrong *at* the forward.** Item 31
  narrowed the floor but kept it there, because `d1` is genuinely `0/0`: delta,
  gamma and rho return `0.0` where the `sigma -> 0+` limits are **0.48765**,
  **`+inf`** and **24.3827**. `theta` is right for a reason that does not
  generalise — the bumped Greeks floor a *price*, whose forward limit really is 0.
  Fixing it is item 21's shape applied nine times, and moves every
  degenerate-configuration Greek. Both the wrong and the right values are already
  asserted, so it cannot drift while it waits.

## Gate

- **18 — Full release battery, then push.** 38 commits sit unpushed; nothing from
  this wave is on the remote. The battery **must** include
  `cargo test --workspace --exclude stochastic-rs-py --doc`: this repo
  `include_str!`s its README into the umbrella crate, so README blocks are real
  doctests that `cargo check` cannot see, and omitting that line is how a broken
  README doctest reached `main` mid-wave.

  **Check disk headroom before running it.** The battery is seven full workspace
  builds across `--all-targets`, `--features openblas`, `--features python`, doc
  and clippy. Running it on a volume already near capacity took this machine to
  **zero bytes free**, which disabled every tool including the one needed to clean
  up. `df -h` first; `target/debug/incremental` and duplicate hash-variants in
  `target/debug/deps` are the two reclaimable pools, worth ~65 GB together here.

  Also unresolved: **this branch has never been built on x86_64.** Dev is
  aarch64-darwin, CI is ubuntu x86_64, and every golden in the wave was verified on
  one target only. One count-valued assertion was already loosened for this reason
  (`037c232`); nothing has swept for others.

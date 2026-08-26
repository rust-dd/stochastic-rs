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

- **10 — Close the registry's proven blind spot.** The shallow half is **done**:
  `LevyModel` and `CrrModel<f64>` are now registered and `assert_model_pricer!`
  went 16 → 18 with the header arithmetic re-derived. What remains is the deeper
  half — Both implement `ModelPricer` and appear nowhere in
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

- **23 — `term_vol` departs from Haug, which may make item 21 moot.** The crate uses
  `σₙ = √((d² + z²)·n/τ)`; Haug's is `σₙ = √(d² + z²·n/τ)`, giving `σ₀ = d` — the
  diffusive vol — rather than 0. **That difference is why the `n = 0` term is
  degenerate at all.** If the formula is wrong, item 21 is filling a hole that
  should not exist. Not touched because correcting it moves every Merton golden,
  so it needs an item that owns that move.
- **24 — `Merton1976Pricer::call_put` returns `NaN` at `λ = 0`, but its Greeks
  return the Black-Scholes value.** `greek_series` has an explicit
  `if self.lambda <= 0.0` branch; `call_put` has none, so `jump_size_std` hits
  `v²γ/0`. Price and Greeks disagree about whether `λ = 0` is a supported state —
  and `merton_greeks_lambda_zero_equals_bs` pins the Greeks side, so the
  disagreement is asserted rather than accidental.
- **25 — `malliavin_one_model_prices_a_grid` is flaky.** It failed once with
  `call 103.976 out of bounds` against its own `c <= S = 100` bound, then passed
  5/5 on re-run. **Unseeded** Monte Carlo (`Gbm::new(…, Unseeded)`). The upper
  bound is violated, not just monotonicity, so this is a genuine CI hazard rather
  than a tolerance question — and the crate's own testing conventions mandate
  pinned seeds for exactly this reason.
- **26 — `BSMCalibrator` has no projection box.** `set_params` writes the raw
  Levenberg-Marquardt vector, so the optimizer can and does pass through negative
  volatilities. Every sibling calibrator (`SabrCalibrator`, the HSC bounds) projects
  into a strictly admissible box. This is the blocker on item 22's one deliberate
  omission.

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

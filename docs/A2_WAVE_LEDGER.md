# SDD ledger — plan: docs/superpowers/plans/2026-08-23-a2-quant-consistency.md

Spec: `docs/superpowers/specs/2026-08-23-a2-quant-consistency-design.md`. Range
`c6c7347..b9447bd` — 30 commits, 149 files, +5655/−5166. Acceptance criterion set
by the maintainer and binding on every task: **every pre-existing test passes with
unchanged expected values.** Test count may rise; a moved golden means the task is
wrong. It held throughout — no golden moved anywhere in the wave.

**Outcome.** Three parallel pricing APIs collapsed to two traits plus a convention.
`PricerExt` (14 implementors) retired; 16 pricers on `ModelPricer`, 3 bond types on
the new `ShortRatePricer`, `KirkSpreadPricer` on the model/query convention without
a trait. Field vocabulary unified to `s`/`tau`/`r`. Prelude 29 → 28.

## Task 1 — compile-time pricer registry (`c6c7347..8b3e69e`)

`tests/pricer_registry.rs`: 71 pricer/engine structs, 21 PyO3 wrappers excluded, 50
in scope, 15 carrying a pricing trait. Shipped three assertion macros rather than
one, so each type is checked against the trait it actually implements.

**The spec's own count was wrong, found here.** It claimed `ModelPricer` had 8
implementors; the truth was 6 concrete plus the `FourierModelExt` blanket. The
measuring grep `impl.*ModelPricer.*for` also matched
`impl<T: ModelPricer + ?Sized> ModelSurface for T` — counting a `ModelSurface` impl
as a `ModelPricer` one. Same loose-pattern class as the earlier "130 processes"
error, and the reason this registry exists.

**Limitation proven by experiment, not assumed.** The reviewer added a real
`ReviewProbeOrphanPricer` to `pricing/asian.rs`, never touched the registry, and the
test passed — unaware the struct existed. A hand-curated list catches a *stale
claim* about a listed struct but structurally cannot catch a *new* struct never
added. Closing that needs a runtime test re-deriving the inventory from source.
**Follow-up owed.**

## Task 2 — `ShortRatePricer` + a live mispricing (`8b3e69e..f64218e`)

`Cir`/`HullWhite`/`Vasicek` now hold only `theta`/`mu`/`sigma`.

**`HullWhite` could not honour the trait's contract and failed silently.**
`from_curve` baked `p0_at_maturity` for one `tau`; `zero_coupon_price(r0, tau)` took
another, feeding `B(t,T)` while the prefactor stayed fixed. Same model at `tau=1`
gave 0.9290243369 built for `tau=1` but 0.7606207950 built for `tau=5` — both inside
(0,1), nothing looking wrong. The trait's own doc advertised "one model, many
maturities", so the *encouraged* usage was the unsound one.

Fixed better than instructed: `from_curve` stores the curve and reprojects
`P^M(0,t+tau)` per query; the direct constructor, which has no curve to store, gets
`pinned_tau` and panics naming both values. The re-reviewer reimplemented
Brigo-Mercurio §3.3.2 from scratch in Python and matched the live code to 13-14
significant figures.

**Cascade, forced not chosen:** dropping the date pair made `TimeExt`
unimplementable, and since `PricerExt: TimeExt`, `PricerExt` came off too — 14 → 11.
This recurred in every later task and is why Task 6 shrank to almost nothing.

## Task 3 — digitals gain `ModelPricer` (`f64218e..c67439f`)

**A trap in the trait itself.** `ModelPricer`'s default `price_put` is vanilla
put-call parity, which is mathematically wrong for three of the four digitals and
undefined for the fourth. Left alone it returns plausible wrong numbers for any
non-vanilla implementor. All four now override; `SuperSharePricer::price_put`
returns a documented `NaN` — a genuine literature gap (a supershare's band
complement is two disjoint half-lines), verified against Reiner-Rubinstein (1991)
and Haug, not a shortcut.

**Live bug found, out of scope:** `pricing/slv.rs:377-380` —
`price_call(&self, s, k, _r, _q, tau)` discards the rate arguments and prices off
`self.r`/`self.q`, so callers passing different rates get identical numbers. **Still
open.**

## Task 4 — naming sweep (`c67439f..d931d41`)

`spot`/`s0` → `s`, `t`/`maturity` → `tau`, `risk_free` → `r`.

**The plan's counts were exactly right and following them literally would still have
been wrong.** Per-site reading found four fields where renaming would silently merge
two different quantities: `HullWhite.t` is evaluation time from curve inception (and
`zero_coupon_price` already takes maturity as its own `tau`, so merging would have
recreated the conflation Task 2 had just fixed); `MarketBar.t` is a time-series
timestamp; `CdsQuote.maturity` and `Deposit.maturity` are `NaiveDate` where `tau` is
`f64` everywhere else; `DeltaHedge` carries **both** `s` and `s0` as genuinely
different points in time.

Purity was proven mechanically rather than by reading: numeral multisets over the
full contents of all 70 changed files, zero differences anywhere.

**Fix round:** the rename created three local-shadows-field collisions where the
local held a *different* quantity (`chooser.rs:171`, `cliquet.rs:59,128` — the last
using both `tau` and `self.tau` in one expression). Not a bug, but it reintroduced
in three functions exactly the ambiguity the sweep exists to remove.

## Task 5a — `BSMPricer` sets the shape (`d931d41..15628a9`)

Split into 5a/5b deliberately: ten pricers are one shape applied ten times, so the
shape gets reviewed once before replication. `BSMPricer` went from 13 fields to
**two** — `v` and the cost-of-carry convention `b`.

**The reading that generalised:** the query is `(discount rate, carry offset)`, not
`(rate, dividend)`. Since `b(r,q) = r − q`, the query reproducing discount `r₀` and
carry `b₀` is `(r₀, r₀ − b₀)`. This let Garman-Kohlhagen be expressed exactly through
`ModelPricer`'s fixed signature, and later repaired an accidental change to
`Merton1976Pricer` bit-identically for every input.

**Fix round, 5 items.** A deleted test was mislabelled "unrepresentable": it pinned
the precedence branch of a live `TimeExt` default, and after its removal *nothing*
anywhere asserted that an explicit `tau` beats dates. Restored. Also caught: a
cross-arch hazard where 26 `norm_cdf`-derived goldens used `assert_eq!` while the
crate's own precedent is `< 1e-12` — verification ran on aarch64-darwin, CI runs
x86_64.

**Maintainer ruling — Greeks.** `GreeksExt`'s `delta(&self)` takes no query, so a
model-only struct cannot implement it. Ruled: query-taking inherent accessors plus an
inherent `greeks(...) -> Greeks` aggregate; **no new trait**, and `GreeksExt` is
*not* retired — it stays on the four types that can still satisfy it. The deciding
evidence was that `GreeksExt` has **zero** generic consumers anywhere — no
`fn f<T: GreeksExt>` exists — which is precisely the spec's own test for when a trait
abstracts over nothing.

## Task 5b — the remaining nine (`15628a9..a7ad4fa`)

One commit per pricer. Interrupted twice by API limits and recovered both times
without loss, because the report was written incrementally per pricer rather than
saved to the end. **Worth repeating on any long multi-item task.**

**`FiniteDifferencePricer` gained a `q` term in its PDE** — the one deliberate
numerics change in the wave. It had no dividend-yield input at all, so implementing
`ModelPricer` meant either discarding the trait's `q` (the `slv.rs` defect) or
extending the PDE. Extended: drift becomes `(r − q)·S·∂ₛV`, call upper boundary
`S·e^{−q(T−t)} − K·e^{−r(T−t)}`, discount term untouched. The reviewer extracted both
solver revisions into a standalone crate and compared `to_bits()` across 72
configurations — **zero mismatches**, and confirmed the pre-refactor American and
European calls were always exactly equal, i.e. that path was genuinely untested
before.

**Live bug found here, fixed afterwards in `d8a2f30`:** `HestonStochCorrPricer`
**discounted twice** — `exp(-r·tau)` inside `char_func_complex` (`cf.rs:73`) and
again in `price_call_carr_madan` (`pricer.rs:27`). Invisible at the source paper's
`r = 0`; **3.68 % under-price** at `r=0.05, τ=0.75`. It was pinned rather than
fixed during the wave so that the reshape could be verified against the behaviour
it actually replaced.

Why the suite never caught it, and this is the durable lesson: one cross-check ran
against a **15 %** tolerance, both put-call-parity tests were structurally vacuous
(the put is *derived* from the call by parity, so parity held by construction), and
— found only during the fix — **the two errors were partially cancelling.**
`compare_with_standard_heston` read 0.95 % before the fix and 2.47 % after, which
decomposes as −1.49 % (the discount) + 2.47 % (the affine approximation). The old
number looked healthy because it was wrong twice.

The fix replaced both vacuous parity tests with guards that have no free
parameters: `char_func_reproduces_the_forward` asserts φ(−i) = S·e^{(r−q)τ}, which
a folded-in discount breaks by exactly `1 − e^{−rτ}`, and
`call_respects_no_arbitrage_bounds` asserts the model-free band, which the double
discount breached by 12–49×. Tolerances tightened 15 % → 3 %, 1 % → 0.3 %,
1e-2 → 1e-14.

There is **no sound pinned deep-ITM floor case**, established by measurement rather
than assumed: a European put is strictly positive at every finite strike, so the
exact unfloored parity is never negative, and the numerical residual's sign
oscillates with both strike and maturity. The replacement asserts the floor's
*contract* over a 3×8 grid with a `rescued >= 3` counter to stop it going vacuous.

**Two type deletions:** `SabrModel` → `SabrPricer`, `HscmModel` →
`HestonStochCorrPricer`. Capability fully preserved in both; same fields, same order,
same names; `ToModel` consumers unaffected; Python class names unchanged.

**Critical caught only at review:** commit `86ff761` broke the umbrella README
doctest. `src/lib.rs:7` is `#![doc = include_str!("../README.md")]`, so the README's
Rust blocks *are* doctests — and neither `cargo check --all-targets` nor `cargo doc`
builds them. The task's battery omitted `--doc`, so it was structurally blind to it;
CI would have gone red on `main`. **Any battery omitting `--doc` cannot see breakage
in a README or doc comment, and this repo wires its README into the crate.**

## Task 6 — retire `PricerExt` (`a7ad4fa..b3d17fc`)

**The plan's Step 1 was false**: it said to confirm zero remaining implementors, but
`KirkSpreadPricer` still carried the trait with seven call sites. The spec places its
family under "convention only, no trait" — excluded from `ModelPricer` is not
excluded from D3 — so it got the model/query split as inherent methods.

The split lands where the arithmetic forces it: the combined volatility is weighted
by `f_temp = f2/(f2+x)`, which is **pure query**, so it cannot be precomputed onto
the struct. That is the `HullWhite` trap from Task 2, on the one pricer whose
formula actively invites it.

Prelude 29 → 28, agreeing in all four places that carry the count.

**Five `.claude/skills/*.md` files presented `PricerExt`/`calculate_price()` as the
current pattern** — uncompilable after this task, and read by future coding agents,
including ones executing later tasks in this same wave. Fixed. `greeks-pattern` was
independently wrong too: it claimed `GreeksExt`'s accessors default to
finite-differencing when they default to `NaN`, which is the whole point — it lets a
consumer distinguish "not exposed" from a real zero.

**A `cargo doc` break created by the deletion:** `stochastic-rs-quant/src/lib.rs:73`
carried `PricerExt` as an *intra-doc link* under
`#![deny(rustdoc::broken_intra_doc_links)]`.

## Task 7 — the error convention (`b3d17fc..b9447bd`)

D4's three-way rule stated once on `ModelPricer` rather than restated 87 times.
Classification of all 87 `f64::NAN` sites: 13 doc-comment, 6 test, **68 production**
— sub-classified into eight groups so "all 68 are not-computable" is checkable rather
than asserted. Result: 64 documented-NaN, **4 converted to panics**, 1 deliberate
exception.

The four conversions split a guard that conflated two things: `v0 == 0` is admissible
(Heston legitimately starts at zero variance; the σ = √v₀ chain rule is what's
undefined) while `v0 < 0` is invalid input. Now `v0 < 0` panics naming the value,
`v0 == 0` still returns `NaN`. Six `#[should_panic(expected = …)]` tests, including a
guard-ordering test and a negative-direction test proving `delta`/`rho` stay finite —
so the guard cannot later widen into a blanket parameter check unnoticed.

**The inverse of the wave's usual finding:** the `tau` guards *look* like conversion
candidates and must not be. A non-finite `tau` reaches them legitimately from
`TimeExt::tau_or_from_dates`, which **documents** `NaN` as its missing-data return —
so asserting `tau.is_finite()` would break a documented convention one layer up. A
sentinel that looks wrong but is load-bearing.

**Six non-`NaN` sentinels found, none fixed.** Sharpest two:
`sabr/pricer.rs:109-113` floors **both** legs to zero on a degenerate Hagan vol — a
zero call *and* zero put is not a price any instrument has, and since `sigma()`
panics first on bad inputs, the reachable trigger is a *calibration-output* parameter
shape. And `analytic_bs.rs:110-115` has **two opposite missing-data conventions in
one function**: missing `tau` → `NaN` (poisons the result), unset spot/vol/rate
`Handle` → `0.0` (silently prices at S = 0), so an unpopulated handle yields a
confident fictitious NPV.

## Open follow-ups

1. ~~**`HestonStochCorrPricer` double discount**~~ — **closed** in `d8a2f30`,
   verified against a from-scratch DOP853 + QUADPACK reference sharing no code
   with the crate: relative error fell from 3.69 % to 6e-5, a 592× improvement.
2. ~~**`pricing/slv.rs:377-380`**~~ — **closed** in `a3cb8a7`. The leverage surface
   turned out to be genuinely rate-dependent, so the fix follows the `HullWhite`
   two-constructor precedent rather than substituting the query rates.
3. ~~**Six non-`NaN` sentinels**~~ — **closed** across `f263dd6`, `a3ca7e4`,
   `98c2830`, `f201190`, `2ba9c34`. Two of the six did **not** become panics, and
   the reasoning is the durable part:

   - **SABR's degenerate Hagan vol → documented `NaN`, not a panic.** The bracket
     `1 + (a+b+c)τ` in Eq. A.69a goes negative because `c = (2−3ρ²)ν²/24` is itself
     negative for `|ρ| > √(2/3)`. Every individual argument is legal — `sigma()`
     has already asserted `k`, forward, `alpha`, `rho` — so this is
     *not computable here*, not *invalid input*. Decisive: `(α,β,ν,ρ) =
     (0.2, 1, 3, −0.9)` at `τ=10` yields `σ = −0.3925`, and all four values lie
     **inside `SabrCalibrator`'s own projection box**. A panic would abort an
     entire calibration over one Levenberg-Marquardt probe point.
   - **`laplace_pdf`'s `l == 0` → kept.** `l < 0` is unreachable and now panics;
     `l == 0` is reachable exactly when every simulated payoff is zero, and there
     the returned `0.0` is the correct Dirac limit. The test written for it found
     a subtlety worth pinning: at the atom `x == 0` the branch returns `1.0` where
     the symmetric Laplace cdf is `0.5` for every `l > 0`, so that value is a
     **convention, not a limit** — and it is load-bearing, cancelling the diagonal
     `j == i` term.

   **A trap found in the process, and worth remembering crate-wide:**
   `fair_strike_replication` ended in `fair.max(0.0)`, and `f64::max` **discards a
   `NaN` operand** — `f64::NAN.max(0.0)` is `0.0`, verified. So a single `NaN` in
   `otm_prices` returned exactly zero, re-entering the sentinel behind the new
   guards. Any `.max(0.0)` floor on a possibly-`NaN` quantity is this bug.

   Teeth verified by reverting rather than asserted: an unlinked spot handle used
   to price a European put at **95.12** and a Heston call at **−47.56**.
4. **`replication_weights`** (`variance_swap.rs:~210`) returns an all-zero weight
   vector for `n < 2 || maturity <= 0` — the same defect class, now inconsistent
   with the sibling `fair_strike_replication` that panics on exactly those
   conditions. Found while closing (3); left because it was outside that scope.
5. **`price_call_carr_madan` is unreliable deep in the money**, found while fixing
   (1) and separate from it: the `K^{−α}` damping prefactor amplifies quadrature
   error, worsening with maturity. At `τ = 2, K = 0.01` against spot 100 it returns
   **881 915.7**; at `K = 20` it returns 10.46 against a lower bound of 77.98. A
   multiplicative discount cannot create a blow-up, so this predates the double
   discount. Prices at `K >= 0.2·S` are unaffected. Root cause localised to
   `integrate_to_convergence`'s `tol = 1e-8` and width-50 initial panels — **not**
   the RK4 step, which moves the result by only 2e-12.
6. **`HestonSlvPricer` spot anchoring** — `LeverageSurface` is indexed by absolute
   spot and `calibrate_leverage` takes a specific `s0`, so pricing far from it walks
   into the surface's clamped boundary, unguarded. Same "precomputed against
   something the query can contradict" shape as (2), which is now fixed for rates
   but not for spot.
4. **Multi-asset asymmetry** — `KirkSpreadPricer` is now the only one of eight
   multi-asset pricers following the model/query split. `Margrabe`, `StulzRainbow`,
   `McRainbow`, `GeometricBasket`, `ArithmeticBasketLevy`, `McBasket` and `McSpread`
   still bundle `s`/`k`/`r`/`tau` behind a no-arg `price()`. None was ever a
   `PricerExt` implementor, so no task in this plan touched them.
5. **Registry cannot see a never-added struct** — needs a runtime inventory test.
6. **One phrasing per concept** — the crate has one *identifier* per concept but ~8
   `pub tau:` fields carry divergent doc wording.
7. **`vega`'s signature differs three ways** across `ModelPricer` implementors
   (`Merton1976Pricer`'s takes `option_type`; `HestonPricer`'s and `BSMPricer`'s do
   not).

## Method notes worth carrying forward

**Ten counting errors were found in this wave** — in the spec, the plan, three task
reports, twice inside `pricer_registry.rs` (the file built to prevent them), and four
of the controller's own. Recurring mechanisms:

- A loose `impl.*Trait.*for` grep matches *other* traits' impls that merely bound on
  it.
- A filename filter (`grep -v tests.rs`) is blind to inline `#[cfg(test)] mod tests`,
  so it under-counts test code and over-counts production code simultaneously.
- `git grep`'s POSIX-ERE engine **silently drops `\b`** under `-E`, yielding false
  zeros. Use plain `grep -rnE` on an extracted tree when re-deriving counts.
- A derivation command published beside its number is worthless if it does not
  reproduce that number. Several did not.

**Hollow gates** — mechanisms that exist but quietly do not cover what people assume
— appeared six times. The sharpest: a test battery omitting `--doc` cannot see a
broken README doctest in a crate that `include_str!`s its README.

**Proving a rename is pure** is better done mechanically than by reading: extract
numeral multisets from both revisions of every changed file and diff the sets.
**Proving no test was deleted** is better done by set-diffing `#[test]` fn names at
both endpoints than by scanning a diff.

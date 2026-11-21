---
name: new-process
description: Complete checklist for adding a new stochastic process to stochastic-rs — host sampler, Euler device engine, the mandatory dispatch overrides, the four test suites, Python bindings, docs and counts. Invoke whenever a new `ProcessExt` implementor is added or an existing one is ported to the device engine.
---

# New process — stochastic-rs-stochastic

Complements, never repeats, `.claude/skills/new-module/SKILL.md` (placement, derives,
`ndarray`, references, benches) and `.claude/skills/dev-rules/SKILL.md` (comments, blank
lines, turbofish, no version tags). Read both first. This file is only what a *process*
adds: the device engine, the dispatch overrides, the four test suites, the bindings, and
the counts that go stale.

Worked examples, all verified against the code: `src/jump/merton.rs` (one component,
generic jump law recognised at runtime, host fallback), `src/volatility/fheston.rs`
(two-component system, `history` block, `HISTORY_SLOTS` guard),
`src/diffusion/multi_gbm.rs` (a launch **view** for an `Output` the engine does not
produce, with a runtime cap), `src/volterra/gaussian_polynomial.rs` (a `lift` family
with a coefficient cap). Paths below are relative to `stochastic-rs-stochastic/` unless
said otherwise.

Every number in this file is a snapshot of the day it was written, and they move
whenever a process lands. Trust the **invariants** and the **recount commands**, never
the digits: re-run the command beside a figure before quoting it anywhere.

## 1. Host side

File `src/<dir>/<name>.rs`, registered `pub mod <name>;` alphabetically in the
directory's module root. Check the form first: `jump`, `diffusion`, `volatility`,
`interest`, `noise`, `process`, `sheet`, `autoregressive` use a sibling file
(`src/jump.rs`); `correlation`, `rough`, `sde`, `mc`, `volterra` use `mod.rs` (`find
stochastic-rs-stochastic/src -name mod.rs`). A *new* directory is also declared in that
crate's `src/lib.rs`.

`//!` header with the LaTeX SDE and the paper reference, then:

```rust
  #[derive(Clone)]
  pub struct Foo<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
    pub kappa: T,        // every parameter `pub`, each with its own `///`
    pub n: usize,        // grid points including t = 0
    pub x0: Option<T>,   // optional starts / horizons are `Option<T>`
    pub t: Option<T>,    // `None` means `T::one()` — the crate-wide convention
    pub seed: S,
    pub backend: B,      // `Cpu` by default, a device handle after `.on::<B>()`
  }
```

`new(..)` takes the fields positionally, ending with `seed: S`, and asserts each precondition
in dev-rules §14's form: every float finite (an `Option<T>` `None` or finite), `t` `None` or
positive, and `n >= 2` on a grid (`n >= 1` on a noise or a discrete-time series). Every field
gets a `with_*(mut self, ..) -> Self`; one changing a value a cache derives from rebuilds the
cache (`Fbm::with_hurst`). `Default` is `n = 252`, `t = 1`.

The sampler is a separate `#[doc(hidden)] pub struct FooSampler<..>` holding all
per-call mutable state and implementing `PathSampler<T>` (`src/traits/sampler.rs`):
`type Output`, `sample_into(&mut self, out: &mut Self::Output)`, `sample(&mut self)
-> Self::Output` (allocate via `crate::buffer::array1_from_fill(n, |out| ..)`).
`ProcessExt<T>` then needs only `type Output`, `type Sampler<'s>` and `fn sampler(&self)
-> Self::Sampler<'_>`. Both traits are sealed: write `impl<…> crate::traits::Sealed for Foo<…> {}`
and `… for FooSampler<…> {}` with the impl's own generics directly above each impl.

**Seed derivation — read the "Reproducibility requirement on implementors" block
on `ProcessExt` in `src/traits/process.rs` before writing `sampler()`.** Its rules:
`sampler()` captures its basis with `self.seed.derive()`, never `self.seed.clone()` and
never `&self.seed` read lazily per path (`chunked_samplers` calls `sampler()` once per
chunk, sequentially, before any chunk reaches rayon, and only `derive()` hash-mixes
those bases apart); a sampler may `.derive()` again on its *own* basis for sub-streams;
**never construct `Unseeded` inside a sampler** — a sub-process built there (`Cgns`, a
jump driver) is driven from an owned already-derived seed
(`Cgns::sample_impl(&self.seed)`, a captured `self.jumps.seed.derive()`); override
`advance_chunk_seed` only when the chunk's clone feeds a *persistent* engine
(`CirPlusPlus`).

Then the backend switch from `src/macros.rs`. `backend_switch!` generates `on::<B2>()` /
`with_backend(handle)` (public — `process.with_backend(Cuda::new(1))`) and **must name every field** except the one holding the backend:

```rust
  backend_switch!([T: FloatExt, S: SeedExt] Foo<T, S> { kappa, n, x0, t, seed } via euler);
```

| Arm | Bound | Storage | Use when | Uses (snapshot) |
|---|---|---|---|---|
| `via euler` | `EulerBackend<T>` | `backend: B` | the process declares a family | 119 |
| `via sheet` | `SheetBackend<T>` | `backend: B` | a two-dimensional field through the sheet pipeline (`Fbs`) | 1 |
| `via fgn euler` | `FgnBackend<T> + EulerBackend<T>` | `fgn: Fgn<_, _, B>` | fractional **and** on the engine | 10 |
| `via phantom` | `FgnBackend<T>` | `backend: B` | backend carried, not an engine process | 2 |

Verify by diffing the brace list against the struct's `pub` fields. Never count arms
with a one-line grep — a wrapped `backend_switch!` call is invisible to it.

## 2. Device side — the Euler engine

Never hand-write a kernel: the CUDA and Metal bodies are generated (`grep -c "family =="
src/euler/{cuda,metal}.rs` → `0` for both).

**2a. Declare the family** in the fitting chunk of `src/euler/families/definitions/`,
with the next free integer code. The DSL is specified under "The family DSL" at the end of
this section. The host and CUDA generator `euler_families!` lives in
`src/euler/families/codegen.rs`; the Metal kernel crate expands the same declarations
in `stochastic-rs-metal-kernels/src/euler/families/codegen.rs`. The vocabulary pair is
in `src/euler/families/vocabulary.rs`, mirrored by the kernel crate's `vocabulary.rs`.

```rust
  /// `dX = θ(μ − X) dt + σ dW`.
  110 => Foo { theta, mu, sigma }
    state (x)
    noise (dz)
    step { x + theta * (mu - x) * dt + sigma * dz }
    report { x },
```

Optional clauses after `report`: `lift { drift (..) diffusion (..) shock (..) }`,
`history { push (..) weights (ctK) }`, `series { size (..) }` — sizes one shot-noise
term from the preamble's draws `gj` (a unit-rate arrival), `ej` (`Exp(1)`), `uj`, `uv`
(uniforms), and the step reads the cell's sum as `sj`; written `series { live size (..) }`
the terms are sized in the step of their cell and the size may read the state — and
`table { increment (..) }`
— one increment of a monotone table from `uj`, `uv` and the spacing `tv`, and the step
reads the table's interpolated inverse at its time as `iv`. A step or report may open
with `bind name = expr;` lines. A step reads `x` (its own state names), `dt`, its noise
names, its parameters, plus `ct`/`ct1`..`ct7`, `nj`, `js`, `gm`, `gm2`, `u`..`u4`, `lv`,
`cv`, `sj`, `iv`.
Vocabulary: `sqrt exp ln pow abs negate tanh atan sin recip positive max min lit less
leq geq pick` — a **new** intrinsic goes into host `ops` and `C_PRELUDE` in
`vocabulary.rs` and into `stochastic-rs-metal-kernels/src/euler/vocabulary.rs`.

Traps, each learned from a real failure:

- **A literal never sits left of an operator** — it cannot infer its type there;
  write `c − f(x)` as `negate(f(x) - lit(c))`.
- **State/noise names must not be `slot_a`..`slot_d` / `shock_a`..`shock_d`**
  and every identifier used in more than one macro arm must come from
  `step_inputs(...)`, or macro hygiene splits it in two.
- **Never name a threaded value `ln`** — it is the logarithm in the vocabulary;
  that is why the lifted value is `lv`.
- **Noise components are increments `√dt · z`**, so a term the host adds as a
  standard normal is written `residual_sd * (de / sqrt(dt))`.
- **A family may keep more state slots than it reports** — `Family::components()`
  counts *report* expressions (planes written), `Family::slots()` counts *state*;
  `RoughHestonMemory` has four states, two reports.

**2b. The `EulerSpec` variant.** `src/euler/spec.rs` carries one
`pub enum EulerSpec<T: FloatExt>` variant per family (121 today, one-to-one) and
`src/euler/spec/encode.rs` its `encode()` arm — `(Family::Foo.code(), pad([theta, mu, sigma]))`, `pad` widening to
`PARAM_SLOTS = 20`. Capacities: 4 state slots, 4 noise components, 20 parameters,
`CURVE_SLOTS = 8` (`ct`, `ct1`..`ct7`), 4 uniforms `u`..`u4`, Poisson count `nj`, jump
sum `js`, Gamma draws `gm`/`gm2`, lifted value `lv` over `LIFT_SLOTS = 176` nodes,
history convolution `cv` over `HISTORY_SLOTS = 512` grid points, series cell sum `sj`
over `SERIES_SLOTS = 512` grid points, table inverse `iv` over `TABLE_SLOTS = 512`
table points, `CORRELATED_STREAMS = 4`.

**2c. The trait impl.** One component → `crate::euler::EulerCoefficients<T>`
(needs `ProcessExt<T, Output = Array1<T>>`); 2–4 components →
`crate::euler::EulerSystem<T, D>` (needs an `Output` built `From<[Array1<T>; D]>`: the
`D` arrays themselves, or a named struct of them such as `HjmPaths`).

| Hook | Required | Notes |
|---|---|---|
| `euler_spec()` | yes | the variant, constants folded once |
| `initial_value()` | `EulerCoefficients` only | the reported start |
| `initial_state() -> [T; 4]` | `EulerSystem` only | one component defaults to `[initial_value(), 0, 0, 0]` |
| `grid_points()` / `horizon()` | yes | `n`, `t.unwrap_or(T::one())` |
| `device_seed()` | yes | always `crate::euler::draw_seed(&self.seed)` |
| `host_sample()` | yes | sampler **and** `advance_chunk_seed` (below) |
| `time_step()` | default `horizon / (n − 1)` | override when the host divides differently |
| `curve()` / `curves()` | default `None` | implement **one**, never both; ≤ `CURVE_SLOTS` |
| `jump_intensity()` / `jump_sizes()` | default `None` | `JumpSizes` carries 12 laws |
| `gamma_draws()` | default `None` | `GammaDraws { first, second }` |
| `step_first()` | default `false` | `true` when the first grid point is itself a draw |
| `fgn_spec()` | default `None` | `Some(self.fgn.fgn_spec(streams))` — the embedded `Fgn` builds the `FgnSpec` |
| `lift_spec()` | default `None` | `LiftSpec { decay, weight, drift_scale, drift_boundary, diffusion_boundary, x0 }` |
| `series_terms()` | default `None` | `Some(j)` terms per path for a family with a `series` clause — exactly when, the launch asserts both ways |
| `table_spec()` | default `None` | `Some(TableSpec { points, u_max })` for a family with a `table` clause — same both-ways assert; `points ≤ TABLE_SLOTS` |

A `jump_intensity()` makes the kernel count the step's arrivals, up to 64: `device_fallback()` (§3)
answers `crate::euler::count_fallback(jump_intensity, time_step)` through the trait it implements
(`<Self as crate::euler::EulerCoefficients<T>>::jump_intensity(self)`), after any reason of its own,
and every sampling method branches on `device_ready()`. The dense-count table in
`tests/device_introspection.rs` gains a row; its census of `fn jump_intensity` fails without one.

`host_sample` is always exactly this — the `advance_chunk_seed` call is what keeps a
host fallback chunk-correct:

```rust
  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
```

**2d. Regenerate the Metal kernels.** CUDA renders its C body from the declarations.
Metal embeds precompiled Rust kernels: run `python3 scripts/generate-euler-kernels.py`,
then `bash scripts/build-metal-kernels.sh` on Apple Silicon with the pinned `cargo-metal`
toolchain, and commit the regenerated artifact. CI's `--check` rejects stale entries.

**2e. `src/euler/family_parity.rs`.** `family_name`'s `match` has no wildcard, so
a new `EulerSpec` variant fails to compile until it is named. Then add a probe to
`every_family()` (one component) or `every_two_component_family()` /
`every_three_component_family()` / `every_four_component_family()` (one file each under
`family_parity/`, the last two sharing `three_four_component.rs`), built with the local
`p(spec, x0)` closure — `p_lift(spec, x0)` for a lifted family.
`every_family_has_a_probe` asserts declared-family count equals probe count and checks
each probe's arity. Give `probe_sizes(spec)` an arm when the family exists to exercise a
jump law no other family reaches.

**2f. A new *per-step value* is a frame change**, not a family addition: the
matchers and expansions in `codegen.rs`, its four signature blocks, the
`step_inputs(...)` invocation in `families/definitions/mod.rs`, the C frame in
`src/euler/kernel/`, and the Metal frame and `Inputs` in
`stochastic-rs-metal-kernels/src/euler/`. Script it, and re-read the traps first.

Verify §2: `cargo test -p stochastic-rs-stochastic --lib -- euler::families` renders
every family into the C body, families counted by `cat src/euler/families/definitions/*.rs | grep -cE "^  [0-9]+ => "`.
A probe needs no series or table work of its own: `Probe::series_terms` and
`Probe::table_spec` answer from `Family::has_series()` / `has_table()`.

### The family DSL

A family is written in a subset that is simultaneously a valid Rust expression and a valid C expression: infix arithmetic, parentheses and a small vocabulary of functions. The shared `definitions` chunks feed the host and C generator (`families/codegen.rs`) and the Rust Metal kernel generator in `stochastic-rs-metal-kernels`, preserving each formula's operation order.

`euler::family_parity` launches every declared family on the device present and checks dimensions and finiteness. Its migration fixture also pins exact Metal paths across every family and launch shape.

A step or report may open with `bind name = expr;` lines before its final expression. Each becomes a `let` on the host and a `const REAL` in the emitted C, which is how a family names a clamped or guarded state once and then reads like the host sampler it came from.

A step or report may read `u`, `u2`, `u3` and `u4`, four uniforms in `(0, 1)` for the step, `nj`, the number of jumps it saw, `js`, the sum of their sizes, `gm` and `gm2`, one or two Gamma draws, and `ct`, the step's value of a time-varying coefficient the host supplies as one value per grid point. A family that never names it costs nothing for it.

The names a step may use are fixed: `x` for the state, `dt` for the step size, `dz` for the step's noise **increment** — `sqrt_dt · z` for Gaussian noise, the fractional increment itself for fGN, which is what lets one declaration serve both — and the family's own parameters, which the generated code binds as locals from the parameter buffer in declaration order. The `report` expression maps the state to what the path records and is evaluated at `t = 0` as well, where no noise exists.

A family with a `history { push (..) weights (ctK) }` clause reads `cv`, the convolution of what it has pushed so far with the weights the launch tabulated in curve slot `K`, indexed by lag: at the step that produces grid point `i` the frame appends the pushed value `p_i` (evaluated on the pre-step state, this step's noise and uniforms) and hands the step `cv = Σ_{k≤i} w_k p_{i-k}`. That is the exact O(n²) scheme of a moving average over the whole past — a Volterra integral on the grid, an ARMA filter, a fractional stable motion — for grids up to `HISTORY_SLOTS` points.

A family with a `series { size (..) }` clause reads `sj`, the sum of the terms that fell into the step's grid cell: before the steps the frame draws, per path, as many terms as the launch's `series_terms()` names — for each, `gj` the arrival of a unit-rate Poisson process, `ej` an `Exp(1)`, `uj` and `uv` two uniforms, and a uniform arrival time on the horizon — sizes each with the clause's expression and adds it to the cell its time falls in. That is the shot-noise series representation of a Lévy process, the sort with a rejection-free size bound, run without the host's sort: cells are addressed, not ordered. Grids up to `SERIES_SLOTS` points. A clause written `series { live size (..) }` sizes each term in the step of its cell instead, where the size may read the state — a scale that follows a simulated variance — and the frame keeps the arrivals per term rather than the sums per cell, so the term count is what the slots then bound.

A family with a `table { increment (..) }` clause reads `iv`, the inverse of a monotone table at the step's time: before the steps the frame builds, per path, a table of as many points as the launch's `table_spec()` names over `[0, u_max]` — each increment from the clause's expression of the two uniforms `uj`, `uv` and the spacing `tv` — trying up to ten extents, each double the last, until the table reaches the horizon, exactly as the host does; each step then finds the first table value at or past its time and interpolates the abscissa linearly. That is the inverse subordinator, the first-passage clock of a positive process. Tables up to `TABLE_SLOTS` points; the grid is free.

The function vocabulary is `sqrt`, `exp`, `ln`, `pow`, `abs`, `negate`, `tanh`, `atan`, `sin`, `recip`, `positive`, `max`, `min`, the literal `lit`, the comparisons `less`, `leq` and `geq`, and the branch-free `pick`. Each has a host implementation in `vocabulary::ops` (compiled for tests only, where `host_step` and `host_report` evaluate the declarations) and a CUDA definition in `vocabulary::C_PRELUDE`, the two kept together in `families/vocabulary.rs`; the Metal kernel crate mirrors them in its own `vocabulary` module. Anything outside it fails to compile on the host, which is the intended way to find out that a kernel could not have run it either.

`families/definitions/` stores the table in code order, one chunk per kind of family. `families/tests` pins the generated host step and report against closed forms.

## 3. The mandatory dispatch overrides

**A process on the engine MUST override `sample`, `sample_par`, `sample_map`,
`try_sample` and `try_sample_par`.** The `ProcessExt` defaults route through the host
sampler, so a device-law test on a process that skipped this compares the host against
the host. Two integrations shipped that way before this was written down. Route through
`self.backend` — or `&self.fgn.backend` for a `via fgn euler` process
(`src/jump/jump_fou.rs`):

| `ProcessExt` | one component | system |
|---|---|---|
| `sample` | `euler_sample` | `system_sample` |
| `sample_par` | `euler_paths` | `system_paths` |
| `sample_map` | `euler_paths_map` | `system_paths_map` |
| `try_sample` | `try_sample` | `try_system_sample` |
| `try_sample_par` | `try_euler_paths` | `try_system_paths` |

When the engine cannot serve every configuration, override
`device_fallback()` — a provided method of `ProcessExt`, `None` by default —
**inside the `ProcessExt` impl** and give every dispatch method the host
fallback. Override the reason, never `device_ready()`: that one is defined as
the absence of a reason, and a process that answers `false` without saying
why leaves the caller an order of magnitude slower with nothing to read. This
is the one fallback rule: a fallback samples on the host bit-identically to
the `Cpu` build and never panics; the hook's `assert!` only catches a caller
that bypasses `ProcessExt`. Never leave a device panic as the only answer.

```rust
  /// What a device runs here: <what the kernels carry>.
  fn device_fallback(&self) -> Option<&'static str> {
    (!(self.n <= crate::euler::HISTORY_SLOTS))
      .then_some("a grid longer than the kernels' per-path history")
  }

  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    if self.device_ready() {
      crate::euler::EulerBackend::euler_paths(&self.backend, self, m)
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }
```

`sample` falls back to `self.sampler().sample()` then `self.advance_chunk_seed()`;
`sample_map` to `sample_map_chunked`; `try_*` to `Ok(<Self as
ProcessExt<T>>::sample(self))`. The hook the guard protects carries an
`assert!`/`expect` naming the guard, so a caller bypassing `ProcessExt` panics instead
of silently drawing zeros — `Merton::jump_sizes`,
`GaussianPolynomialVolatility::euler_spec`. Guards in tree: `Merton` (a jump law the
kernels draw, via `src/process/cpoisson.rs::device_jump_sizes`),
`GaussianPolynomialVolatility` (`DEVICE_COEFFICIENTS = 8`), `MultiGbm` (`assets() <=
CORRELATED_STREAMS`), `Fheston` / `RoughBergomi` / `FbatesSvj` / `Arima`
(`HISTORY_SLOTS`), `Cgmy` / `Cts` / `Kobol` / `Rdts` (`SERIES_SLOTS`),
`InverseAlphaStableSubordinator` (`TABLE_SLOTS`), `Lmm` (`CORRELATED_STREAMS`
forwards), `Ctrw` (exponential waits), `Hawkes` (count mode).

When the `Output` is not what the engine produces (`Array2<T>`, a complex path, one row
of a matrix), add a borrowed **launch view**: a `#[doc(hidden)] pub struct FooLaunch<'a,
..>(&'a Foo<..>)` implementing `ProcessExt` and `EulerSystem`/`EulerCoefficients`, then
launch `self.backend.system_paths(&FooLaunch(self), m)` and reshape. It borrows, so the
seed it advances is the process's own. Examples: `MultiGbmLaunch`, `McgnsLaunch`,
`MultifactorHestonLaunch`, `AdgRow(self, i)` (one launch per independent row),
`CfouParts`.

## 4. Tests

**4a. Device law** — a leaf file of `tests/device_law/`: `<group>.rs`, or
`<group>/<part>.rs` for the three groups that are thin parents (`fractional/{correlated,driven,volterra}.rs`,
`levy/{arrivals,subordinators}.rs`, `systems/{affine,heston,jumps,regimes,volatility}.rs`); never the
parent. A new group is declared `pub(crate) mod <group>;` inside the `mod device_law { .. }` block of
`tests/device_law.rs`. The binary is gated `#![cfg(device_backend)]` (an alias the crate's
`build.rs` emits) and is `f32` throughout. Groups — pick by what the comparison must allow for,
not by source directory: `bounded`, `coefficients`, `conditional_variance`, `counts`, `curves`,
`fractional`, `gaussian`, `jumps`, `levy`, `memory`, `rows`, `seed`, `series`, `sheet`, `stable`,
`systems`. Helpers in
`device_law/common.rs`, reached with `use super::common::..`: `Device` (Cuda, else
Metal), `M = 4_000`, `terminal_mean`, `terminal_std`, `agrees(host, device, tol, what)`
(relative error), `all_finite`, `within`.

```rust
  #[test]
  fn foo_agrees_with_the_cpu_law() {
    let build = || Foo::<f32, _>::new(2.0, 0.05, 0.2, N, Some(0.03), Some(1.0), Deterministic::new(11));
    const PATHS: usize = 3 * M;
    let device = build().on::<Device>().sample_par(PATHS);
    let host = build().sample_par(PATHS);
    assert_eq!(device[0][0], 0.03, "every path starts at x0");
    all_finite(&device, "Foo");
    agrees(terminal_mean(&host), terminal_mean(&device), 0.02, "Foo terminal mean");
    agrees(terminal_std(&host), terminal_std(&device), 0.05, "Foo terminal spread");
  }
```

Rules, each already stated in a doc comment in that tree. **Pick a statistic that MOVES
with every parameter** — a martingale's mean does not see `nu`, a spread does not see a
rotation, fBm at `t = 1` looks like Brownian motion (use `t = 4`, where the spreads are
2.64 vs 2.00); give sibling rows and curves distinct slopes so a slot read from its
neighbour moves the statistic past the tolerance.
**Never pad a tolerance** — if the standard error is too thin, raise the path
count (`3 * M`, `4 * M`) and keep the tolerance. **A heavy-tailed law gets a robust
spread**: at a kurtosis above twenty (tempered-stable jumps, stable increments) the
sample standard deviation at 12 000 paths wanders by 5–8 % between seeds, so pin the
interquartile range, as `series.rs`, `jumps.rs` and the `Svcgmy` case do. **Pin exact things exactly** — curve
tabulation, the initial point and the runtime-guard fallback get equality assertions,
not statistics; the fallback must be bit-identical to the plain host build
(`assert_eq!(build().on::<Device>().sample(), build().sample())`).

**4b. Sabotage verification is mandatory.** A passing device-law test proves
nothing until it has been made to fail:

```bash
  F=stochastic-rs-stochastic/src/euler/families/definitions/diffusions.rs  # the chunk that declares it
  cp $F /tmp/families.bak
  # in the new family scale one term by `lit(3.0)`, literal on the RIGHT:
  #   `sigma * dz`  ->  `sigma * dz * lit(3.0)`
  cargo test -p stochastic-rs-stochastic --features metal --test device_law foo_agrees
  # it MUST fail. If it passes, the statistic is blind — pick another one.
  cp /tmp/families.bak $F
  cmp /tmp/families.bak $F && echo reverted
```

**4c. Chunk invariance** for an fGN-fed family. Chunking has no statistical
witness — a batch that repeats its own noise still has the right law — so extend
`metal_fractional_chunks_are_bit_identical_to_one_launch` /
`cuda_fractional_chunks_are_bit_identical_to_one_launch` in `src/euler/tests.rs`:
compare `sample_par(M)` against
`Metal::default().with_batch_budget(small).euler_paths(&p, M)` and assert no two paths
of one batch are equal. The two backends advance the pipeline counter differently, so
both are asserted separately.

**4d. The parity suite** (needs no GPU for the compile-time and rendered-kernel
checks): `cargo test -p stochastic-rs-stochastic --features metal --lib euler::`,
covering `every_family_has_a_probe`, `every_family_runs_on_the_device` and
`kernel::tests::every_family_reaches_the_rendered_kernel`.

**4e. The reproducibility guard.** Add exactly one `guard!` line to
`tests/reproducibility_all_processes/<dir>.rs`, in the submodule matching the source
directory:

```rust
  guard!(foo, "Foo", |s| Foo::new(2.0, 0.05, 0.2, N, Some(0.03), Some(1.0), s));
```

Read the header of `tests/reproducibility_all_processes.rs` for the rules — it carries
the derivation grep, the per-directory counts (diffusion 36, process 20, jump 17,
interest 16, volatility 15, autoregressive 9, noise 6, rough 4, correlation 4, volterra
3, sheet 1; bump the one you touched **and** the total), and the statement that a type
with no line there is a type nothing is proving anything about. That grep only sees a
one-line `impl` header, so long bounds go in a trailing `where`. A new `Output` shape
needs a `ReproBits` impl in that tree's `common.rs`.

**4f. Unit tests.** Default to an inline `#[cfg(test)] mod tests { .. }` at the
bottom of the file (82 files). Only when the file nears the 600-line cap of
`max-file-length`, split to a sibling and declare `#[cfg(test)] #[path =
"<name>_tests.rs"] mod tests;` with a one-line comment saying why (7 files). Pinned
seeds only — `Deterministic::new(42)`, never `Unseeded`, never `rand::rng()`; see
`.claude/skills/integration-test-writing/SKILL.md`.

## 5. Python bindings

At column 0 in the process file, invoke the wrapper macro from `src/macros.rs`:
`py_process_1d!` for an `Array1` output (65 uses), `py_process_2x1d!` for `[Array1; 2]`
(9), `py_process_2d!` for `Array2` (2). Each has a plain and a `device` arm; the
trailing bare `device` token adds a `device=` keyword routed through
`py_device_dispatch!` (all 76 invocations carry it) — add it whenever the process
reaches the engine. `sig:` is the `#[pyo3(signature = ..)]` list, `seed=None,
dtype=None` always last and `lambda_`/`gamma_` for Python keywords; `params:` repeats
the same names with Rust types, in `new()`'s order.

```rust
  py_process_1d!(PyOu, Ou,
    sig: (theta, mu, sigma, n, x0=None, t=None, seed=None, dtype=None),
    params: (theta: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>),
    device
  );
```

A process generic over a distribution has no monomorphic signature, so it gets a
hand-written `#[pyclass]` fixing `D = CallableDist<T>` instead — see `PyMerton`. Put
`#[doc(hidden)]` on the line above that `#[pyclass]` (the macros already carry it) and
above any `pub use python::…` re-export of it; `tests/python_surface_hidden.rs` fails without it.

A hand-written class on the macros' slots (`inner_f32`, `inner_f64`, `seeded_f32`,
`seeded_f64`, and `device` for `py_device_dispatch`) writes only its `#[new]`, inside
`py_process_methods!(PyX, <shape>, <dispatch>, { #[new] … })` (`array1`, `array2` or
`pair`; `py_dispatch` or `py_device_dispatch`), which adds `sample`, `sample_par`,
`device_fallback` and `device_ready` — see `process/poisson/python.rs`.

Register in the single flat `#[pymodule]` of `stochastic-rs-py/src/lib.rs` (its only
source file): a `use` in the `// Stochastic — <dir>` banner group, then
`m.add_class::<PyFoo>()?;` in the bare-ident form. The Python-visible name is the Rust
ident unless `#[pyclass(name = "..")]` renames it, and the macro-generated classes carry
no rename, so it really is `sr.PyFoo`.

No live pytest enumerates classes or asserts a count, so nothing there breaks. Add a
case to `stochastic-rs-py/tests/test_stochastic.py` in the seed-determinism / path-shape
style if the process is user-facing. Build from the repo root with `maturin develop
--release` (`pyproject.toml` `[tool.maturin] manifest-path =
"stochastic-rs-py/Cargo.toml"`, `module-name = "stochastic_rs"`), then `pytest
stochastic-rs-py/tests/ -v`.

Recount for `CLAUDE.md` — do not trust the written number, it drifts:

```bash
  grep -c "m.add_class::<" stochastic-rs-py/src/lib.rs     # total, 3 behind `ai`
  grep -c "m.add_function(" stochastic-rs-py/src/lib.rs    # total, 1 behind `ai`
```

## 6. Re-exports and counts

The umbrella re-exports whole crates (`pub use stochastic_rs_stochastic as stochastic;`
in the repo-root `src/lib.rs`), so `stochastic_rs::stochastic::<dir>::<name>::Foo`
resolves with **no umbrella edit at all**. Touch the umbrella only for a new trait or
prelude item.

What does go stale:

1. `CLAUDE.md` — the `129 processes` in the layout tree and the `ProcessExt`
   bullet's `**138** concrete implementors over **129** processes`. Recount with
   the command that bullet quotes (the `-A1` is load-bearing):
   `grep -rhE -A1 "impl.*ProcessExt<T>" stochastic-rs-stochastic/src --include='*.rs' | grep -oE "for [A-Z][A-Za-z0-9]*" | sed 's/for //' | sort -u | wc -l`
   Implementors exceed processes by the launch/row views, which carry no
   reproducibility guard — today 138 minus `AdgRow`, `CfouParts`,
   `LmmLaunch`, `McgnsLaunch`, `MultiGbmLaunch`, `MultifactorHestonLaunch`,
   `MultivariateHawkesLaunch`, `WishartLaunch`, `WuZhangPair` = 129, matching `grep -rhoE "guard!\(" stochastic-rs-stochastic/tests/reproducibility_all_processes/*.rs | wc -l`.
   The prelude count is unaffected by a process.
2. `CLAUDE.md`'s Python entry counts, from §5's two greps.
3. `website/public/python-parity.json`, regenerated by `bun run python:parity`
   in `website/`. Its class regex is `m\.add_class::<(\w+)>\(\)`, which silently
   drops fully-qualified `add_class` lines — another reason to register in the
   bare-ident form.
4. `website/content/docs/concepts/gpu-support.mdx` — the support-matrix rows
   `89 processes` (one component) and `38 processes` (systems), the explicit
   fractional name list with its `(11, counted above)`, and the prose sentence
   `the other 127 through the Euler engine`, whose list of what the engine covers
   follows it; for a configuration that stays on the host, the matching bullet
   under "What stays on the host". The page carries no family count.
5. `notebooks/colab_cuda_check.ipynb`, cell index 5 — a prose enumeration with no
   hard counts; insert the name into the paragraph matching its family. Edit
   through Python, keeping the file's own formatting:

```python
  import json
  p = "notebooks/colab_cuda_check.ipynb"
  nb = json.load(open(p))
  src = nb["cells"][5]["source"]
  i = next(k for k, line in enumerate(src) if "RoughBergomi" in line)
  src[i] = src[i].replace("`RoughBergomi`", "`RoughBergomi`, `Foo`")
  with open(p, "w") as f:
      json.dump(nb, f, indent=1, ensure_ascii=False)
      f.write("\n")
```

`indent=1` and `ensure_ascii=False` are both required — the file uses one-space
indentation and contains literal `—`, `∏`, `−` characters.

## 7. Gates before commit

```bash
  cargo clippy -p stochastic-rs-stochastic --all-targets -- -D warnings
  cargo clippy -p stochastic-rs-stochastic --all-targets --features metal -- -D warnings
  cargo clippy -p stochastic-rs-stochastic --all-targets --features cuda -- -D warnings
  RUSTDOCFLAGS="-D warnings" cargo doc -p stochastic-rs-stochastic --no-deps
  cargo check --workspace --no-default-features
  cargo test -p stochastic-rs-stochastic --features metal --lib euler::
  cargo test -p stochastic-rs-stochastic --features metal --test device_law
```

All three clippy runs matter, and the `cuda` one compiles on a machine with no GPU
(cudarc links at run time): the CUDA back-end has two launch wrappers and a pipelined
batch path, and a launch scalar added to one and not the others is an arity error only
that build shows — the history/series/table scalars once shipped that way for three
commits. A helper only device launches call is dead code in the no-feature build, so give it the crate's guard, as `flatten_curves` and `history_slot`
in `euler/slots.rs` have — `#[cfg_attr(not(device_backend), allow(dead_code))]`. The crate's `build.rs`
emits `device_backend` (CUDA or Metal), `metal_backend` and `accelerate_backend`, the only
spellings of the device gates inside `stochastic-rs-stochastic`. `src/lib.rs` carries
`#![deny(rustdoc::broken_intra_doc_links)]`, so a mistyped `[`Foo`]` fails `cargo doc`
rather than warning.

The full battery (~10 minutes) goes to a **subagent** with `CARGO_TARGET_DIR` inside the
session scratchpad — never a sibling `target-battery/` in the repo root, which is how
~43k build artifacts once landed in three commits and the push failed on the 2 GiB pack
limit (`.gitignore` now carries `/target-*` as a second line of defence). Tell the
subagent to run every command in the foreground with a 600000 ms timeout, and that
`cargo test --workspace` skips `device_law` entirely — `Device` only exists behind
`cuda`/`metal`, so a GPU run is a separate command.

## 8. Commit

One commit per process, or one per batch of related processes. Bare-prefix one-liner, no
scope, no body, no attribution trailers: `feat: add the foo process`.

## When NOT to put it on the engine

The engine steps a bounded forward recursion over four scalar state slots. What rules a
process out is unbounded work per step, unbounded state, or an output that is not a
fixed grid. Each group below is a *decision*, not a permanent verdict — a new device
pipeline can move one onto the engine, and the named examples go stale as that happens:
a **series whose term sizes read another simulated path** (`Svcgmy`: the CGMY
scale at each arrival is the variance path there, drawn by an exact non-central
χ² step the kernels do not carry); **Rust closures or Python callables in the coefficients** (`Cheyette`'s
`sigma`, `VolterraSde`'s drift and diffusion — the same coefficient written as an
`Expr` compiles to a program the kernel interprets and the process runs; a closure
keeps it on the host through `device_ready()`); an
**output length that is not the grid** (`Hawkes` and `MultivariateHawkes` in horizon
mode, `MultivariateHawkes` above two components or with per-pair decays); **a rank-adaptive step or a Poisson mixture below one degree of freedom**
(`Wishart` below a degree of three, `Svcgmy` at `4κη/ζ² < 1`) — at one degree or more a
non-central χ² is the square of a shifted normal plus the frame's gamma draw, which is
how `Svcgmy`'s exact CIR step and `Wishart`'s squared Bessel draws run; **more than 4
state slots or 4 noise components** (`Lmm` / `MultiGbm` / `Mcgns` above four, `Wishart`
above `d = 2`).

Before declaring a process host-only — none is today, and `backend_switch!` keeps
no host-only form — check the nine documented ways round the cap:

1. **A launch view with a runtime cap** — pad a runtime `k` into the fixed
   four-slot family and fall back to the host above it (`MultiGbmLaunch`), or
   launch once per independent row (`AdgRow`).
2. **Runtime law recognition** — a process generic over `D: Distribution<T>`
   stays generic on the host and reaches the device by `Any` downcast in
   `src/process/cpoisson.rs::device_jump_sizes` / `device_arrival_rate`; an
   unrecognised law falls back, bit-identically to the plain host build.
3. **A `lift` clause** — a rough Volterra kernel as a Markov lift of up to
   `LIFT_SLOTS` nodes.
4. **A `history` clause** — the host's own exact O(n²) convolution for grids up
   to `HISTORY_SLOTS`, weights travelling as a curve. This is what moved
   `Fheston`, `FbatesSvj`, `RoughBergomi`, `Lfsm` and `Arima`/`Sarima` onto
   the engine.
5. **A `series` clause** — a shot-noise series drawn per path before the steps
   and summed into the grid cells its terms fall in, so the host's sort is not
   needed; grids up to `SERIES_SLOTS`. This is what moved `Cgmy`, `Cts`, `Kobol`
   and `Rdts` onto the engine, as one family with host-folded constants.
6. **A `table` clause** — a monotone table built per path before the steps and
   inverted by interpolation at every step, the extent doubled until the table
   reaches the horizon; tables up to `TABLE_SLOTS`. This is what moved
   `InverseAlphaStableSubordinator` onto the engine.
7. **An exact recursion in place of a rejection loop** — a family reads four
   uniforms a step, so a thinning loop with a random number of proposals has no
   home in it, but an exact inverse-transform of the same law does. This is
   what moved `Hawkes` (Dassios–Zhao) onto the engine, and `MultivariateHawkes`
   in count mode by **superposition**: each target's excess clock and the
   baselines' joint Poisson clock are independent given the state, so the next
   event is the earliest of their exact arrivals and the mark its index. The
   first target's clock reads `$u`, the second's `$u3`, the baselines' clock
   `$u2` and its mark `$u4`, each a uniform of its own.
8. **A pipeline of its own** — a process whose sample is a transform rather
   than a recursion gets a capability trait beside `FgnBackend`
   (`SheetBackend<T>` in `device/sheet_backend.rs`), a `via sheet` switch arm, and one
   pipeline file per backend under its module (`sheet/fbs/{metal,cuda}.rs`)
   reusing the fGN pipeline's FFT stages, Philox draws and bit reversal. The host
   devices route through the process's own sampler, so a fallback is
   bit-identical to the `Cpu` build, and `device_ready()` names what the
   kernels take (radix-2: embedding sides that are powers of two). This is
   what moved the sheet `Fbs` onto the devices. A chunk-invariance test
   (`with_batch_budget`) is mandatory: every Philox draw must count a batch-global
   cell, and the correction's a batch-global sheet under a site of its own.
9. **A program** — a coefficient of time and state the user writes as an
   `Expr` (`stochastic_rs_distributions::traits::Expr`, stored as
   `Fn2D::Expr(Program)`) reaches the kernel as postfix code: the process
   returns it from `program_spec()` (`ProgramSpec { first, second }`), the frame
   interprets it before every step at `ct` and the first state slot, and the
   family reads `pv` / `pv2`. `device_ready()` is `fn2d.program().is_some()`;
   a closure stays on the host. Sixteen opcodes, ≤ 62 operations, stack ≤ 8 —
   the interpreter lives in the C frame (`kernel.rs`),
   and the host's `Program::eval` is the reference the parity probes run
   (`ProbePrograms::pair()` uses every opcode). This is what moved `Cheyette`
   and `VolterraSde` onto the engine — the last two host-only processes.

## Definition of done

- [ ] `//!` header with the LaTeX SDE and the paper reference (title, authors, DOI/arXiv)
- [ ] `pub` fields, `seed: S`, `backend: B = Cpu`, `Option<T>` optionals, §14 asserts (finite floats, positive `t`, `n >= 2`)
- [ ] `new(..)`, a `with_*` per field, `Default` at `n = 252`, `t = 1`; `pub mod` registered alphabetically
- [ ] `#[doc(hidden)]` sampler implementing `PathSampler<T>`; `ProcessExt<T>` implementing only `sampler()`; each impl preceded by its `crate::traits::Sealed` marker
- [ ] `sampler()` derives (`self.seed.derive()`), constructs no `Unseeded`, races no shared atomic
- [ ] `backend_switch!` with the right arm and **every** field named
- [ ] Family declared in its `families/definitions/` chunk; the six DSL traps checked
- [ ] `EulerSpec` variant + `encode()` arm; ≤ 20 params, ≤ 8 curves, ≤ 4 states, ≤ 4 noises
- [ ] `EulerCoefficients`/`EulerSystem` impl; `draw_seed`; `host_sample` calls the sampler **and** `advance_chunk_seed`
- [ ] `family_name` arm and a probe in the arity-matching `every_*_family()` list
- [ ] All five `ProcessExt` methods overridden through the backend; a configuration the kernels cannot carry answers `device_ready()` (a `ProcessExt` override) and falls back to the host in every one of them, never a panic; the hook's `assert!` names the bypass
- [ ] Device-law case in the right leaf of `device_law/` (`<group>.rs`, or `<group>/<part>.rs` under `fractional`, `levy`, `systems`); statistic moves with every parameter; tolerance not padded
- [ ] **Sabotage-verified**: term scaled by `lit(3.0)` → test fails → reverted, `cmp` clean
- [ ] Chunk-invariance test extended if the family is fGN-fed
- [ ] `guard!` line added; that tree's per-directory count and total bumped
- [ ] Python macro invoked (`device` flag if on the engine) and `add_class` registered
- [ ] `CLAUDE.md` counts, `python-parity.json`, `gpu-support.mdx` matrix + prose, notebook cell 5 updated
- [ ] Both clippy runs, `cargo doc -D warnings`, `cargo check --workspace --no-default-features`, parity + device-law suites green
- [ ] Full battery run in a subagent with `CARGO_TARGET_DIR` in the scratchpad

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

`new(..)` takes the fields positionally, ending with `seed: S`, and asserts `n >= 2`
where the recursion needs two points (32 sites do). Every field gets a `with_*(mut self,
..) -> Self`; one changing a value mirrored elsewhere re-syncs it
(`Merton::resync_cpoisson_poisson`). `Default` is `n = 252`, `t = 1`.

The sampler is a separate `#[doc(hidden)] pub struct FooSampler<..>` holding all
per-call mutable state and implementing `PathSampler<T>` (`src/traits/sampler.rs`):
`type Output`, `sample_into(&mut self, out: &mut Self::Output)`, `sample(&mut self)
-> Self::Output` (allocate via `crate::buffer::array1_from_fill(n, |out| ..)`).
`ProcessExt<T>` then needs only `type Output`, `type Sampler<'s>` and `fn sampler(&self)
-> Self::Sampler<'_>`.

**Seed derivation — read the "Reproducibility requirement on implementors" block
on `ProcessExt` in `src/traits/process.rs` before writing `sampler()`.** Its rules:
`sampler()` captures its basis with `self.seed.derive()`, never `self.seed.clone()` and
never `&self.seed` read lazily per path (`chunked_samplers` calls `sampler()` once per
chunk, sequentially, before any chunk reaches rayon, and only `derive()` hash-mixes
those bases apart); a sampler may `.derive()` again on its *own* basis for sub-streams;
**never construct `Unseeded` inside a sampler** — a sub-process built there (`Cgns`,
`CompoundPoisson`) is driven from an owned already-derived seed
(`Cgns::sample_impl(&self.seed)`, a captured `self.cpoisson.seed.derive()`); override
`advance_chunk_seed` only when the chunk's clone feeds a *persistent* engine
(`CirPlusPlus`).

Then the backend switch from `src/macros.rs`. `backend_switch!` generates `on::<B2>()` /
`with_backend(..)` and **must name every field** except the one holding the backend:

```rust
  backend_switch!([T: FloatExt, S: SeedExt] Foo<T, S> { kappa, n, x0, t, seed } via euler);
```

| Arm | Bound | Storage | Use when | Uses (snapshot) |
|---|---|---|---|---|
| `via euler` | `EulerBackend<T>` | `backend: B` | the process declares a family | 114 |
| `via host` | `HostBackend` | `backend: B` | host only (see last section) | 6 |
| `via fgn euler` | `FgnBackend<T> + EulerBackend<T>` | `fgn: Fgn<_, _, B>` | fractional **and** on the engine | 10 |
| `via phantom` | `FgnBackend<T>` | `backend: B` | backend carried, not an engine process | 2 |
| `via fgn` | `FgnBackend<T>` | `fgn: Fgn<_, _, B>` | fractional, not on the engine | 0 |

Verify by diffing the brace list against the struct's `pub` fields. Never count arms
with a one-line grep — a wrapped `backend_switch!` call is invisible to it.

## 2. Device side — the Euler engine

Never hand-write a kernel: the CUDA and Metal bodies are generated (`grep -c "family =="
src/euler/{cuda,metal}.rs` → `0` for both).

**2a. Declare the family** in the table `src/euler/families.rs`, appended in code
order with the next free integer code. Read that file's module doc — it is the DSL
specification. The generator `euler_families!` lives in `src/euler/families/codegen.rs`
(pulled in with `#[macro_use]`, not a `use`, because the generated `cube*` child modules
recurse into it); the vocabulary triple is in `src/euler/families/vocabulary.rs`.

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
(uniforms), and the step reads the cell's sum as `sj` — and `table { increment (..) }`
— one increment of a monotone table from `uj`, `uv` and the spacing `tv`, and the step
reads the table's interpolated inverse at its time as `iv`. A step or report may open
with `bind name = expr;` lines. A step reads `x` (its own state names), `dt`, its noise
names, its parameters, plus `ct`/`ct1`..`ct7`, `nj`, `js`, `gm`, `gm2`, `u`, `u2`, `lv`,
`cv`, `sj`, `iv`.
Vocabulary: `sqrt exp ln pow abs negate tanh atan sin recip positive max min lit less
leq geq pick` — a **new** intrinsic goes into all three implementations in
`vocabulary.rs` (host `ops`, `C_PRELUDE`, `cube_ops`).

Traps, each learned from a real failure:

- **A literal never sits left of an operator** — it cannot infer its type there;
  write `c − f(x)` as `negate(f(x) - lit(c))`.
- **State/noise names must not be `slot_a`..`slot_d` / `shock_a`..`shock_d`** —
  the CubeCL functions take the slots as parameters and bind the family's names
  from them, so a collision shadows the slot it is read from.
- **`#[cube]` cannot see through a macro call**, so every identifier used in more
  than one macro arm must come from `step_inputs(...)` or hygiene splits it in two.
- **Never name a threaded value `ln`** — CubeCL treats it as the logarithm
  intrinsic and panics; that is why the lifted value is `lv`.
- **Noise components are increments `√dt · z`**, so a term the host adds as a
  standard normal is written `residual_sd * (de / sqrt(dt))`.
- **A family may keep more state slots than it reports** — `Family::components()`
  counts *report* expressions (planes written), `Family::slots()` counts *state*;
  `RoughHestonMemory` has four states, two reports.

**2b. The `EulerSpec` variant.** `src/euler.rs` carries one
`pub enum EulerSpec<T: FloatExt>` variant per family (110 today, one-to-one) plus an
`encode()` arm — `(Family::Foo.code(), pad([theta, mu, sigma]))`, `pad` widening to
`PARAM_SLOTS = 20`. Capacities: 4 state slots, 4 noise components, 20 parameters,
`CURVE_SLOTS = 8` (`ct`, `ct1`..`ct7`), 2 uniforms `u`/`u2`, Poisson count `nj`, jump
sum `js`, Gamma draws `gm`/`gm2`, lifted value `lv` over `LIFT_SLOTS = 176` nodes,
history convolution `cv` over `HISTORY_SLOTS = 512` grid points, series cell sum `sj`
over `SERIES_SLOTS = 512` grid points, table inverse `iv` over `TABLE_SLOTS = 512`
table points, `CORRELATED_STREAMS = 4`.

**2c. The trait impl.** One component → `crate::euler::EulerCoefficients<T>`
(needs `ProcessExt<T, Output = Array1<T>>`); 2–4 components →
`crate::euler::EulerSystem<T, D>` (needs `Output = [Array1<T>; D]`).

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
| `jump_intensity()` / `jump_sizes()` | default `None` | `JumpSizes` carries 7 laws |
| `gamma_draws()` | default `None` | `GammaDraws { first, second }` |
| `step_first()` | default `false` | `true` when the first grid point is itself a draw |
| `fgn_spec()` | default `None` | `FgnSpec { sqrt_eigenvalues, n, offset, hurst, t, streams }` |
| `lift_spec()` | default `None` | `LiftSpec { decay, weight, drift_scale, drift_boundary, diffusion_boundary, x0 }` |
| `series_terms()` | default `None` | `Some(j)` terms per path for a family with a `series` clause — exactly when, the launch asserts both ways |
| `table_spec()` | default `None` | `Some(TableSpec { points, u_max })` for a family with a `table` clause — same both-ways assert; `points ≤ TABLE_SLOTS` |

`host_sample` is always exactly this — the `advance_chunk_seed` call is what keeps a
host fallback chunk-correct:

```rust
  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
```

**2d. The hand-written CubeCL dispatch** — `src/euler/cubecl.rs`. A new family
needs **two** arms (`fn step`, `fn report`), plus one in `fn lift_coefficient` if it has
a `lift` clause, one in `fn history_push` if it has a `history` clause, one in
`fn series_size` if it has a `series` clause and one in `fn table_increment` if it has
a `table` clause. The invariant is `arms = 2 × families + lift clauses + history
clauses + series clauses + table clauses` (when this was written,
2 × 115 + 6 + 6 + 1 + 1 = 244).

```rust
  if family == 110u32 {
    stepped = cube::Foo(
      component, x0, x1, x2, x3, params, dt, ct, ct1, ct2, ct3, ct4, ct5, ct6, ct7, nj, js, gm,
      gm2, u, u2, lv, cv, dz0, dz1, dz2, dz3,
    );
  }
```

`report` drops `dt` and the four `dz` and calls `cube_report::`; `lift_coefficient` /
`history_push` lead with `which` / `0u32` instead of `component` and call `cube_lift::`
/ `cube_history::`; `series_size` and `table_increment` pass zeros for everything but
`params`, `dt` and their draws and call `cube_series::` / `cube_table::`. A missing arm compiles, runs, and quietly returns a flat path —
which is what §4d catches.

**2e. `src/euler/family_parity.rs`.** `family_name`'s `match` has no wildcard, so
a new `EulerSpec` variant fails to compile until it is named. Then add a probe to
`every_family()` (one component) or `every_two_component_family()` /
`every_three_component_family()` / `every_four_component_family()`, built with the local
`p(spec, x0)` closure — `p_lift(spec, x0)` for a lifted family.
`every_family_has_a_probe` asserts declared-family count equals probe count and checks
each probe's arity. Give `probe_sizes(spec)` an arm when the family exists to exercise a
jump law no other family reaches.

**2f. A new *per-step value* is a frame change**, not a family addition: the
matchers and expansions in `codegen.rs`, its four signature blocks, the
`step_inputs(...)` invocation in `families.rs`, every `cubecl.rs` dispatch arm, and the
frame in `src/euler/kernel.rs`. Script it, and re-read the traps first.

Verify §2: `grep -c "family ==" src/euler/cubecl.rs` must equal `2 × families + lifts +
histories + series + tables`, families from `grep -cE "^  [0-9]+ => " src/euler/families.rs`.
A probe needs no series or table work of its own: `Probe::series_terms` and
`Probe::table_spec` answer from `Family::has_series()` / `has_table()`.

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

When the engine cannot serve every configuration, add a runtime guard and a host
fallback rather than dropping the process off the engine:

```rust
  pub fn device_ready(&self) -> bool { self.n <= crate::euler::HISTORY_SLOTS }

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
CORRELATED_STREAMS`), `RoughHeston` / `RoughBergomi` / `FBatesSvj` / `Arima`
(`HISTORY_SLOTS`), `Cgmy` / `Cts` / `KoBoL` / `Rdts` (`SERIES_SLOTS`),
`InverseAlphaStableSubordinator` (`TABLE_SLOTS`), `Lmm` (`CORRELATED_STREAMS`
forwards), `Ctrw` (exponential waits), `Hawkes` (count mode).

When the `Output` is not what the engine produces (`Array2<T>`, a complex path, one row
of a matrix), add a borrowed **launch view**: a `#[doc(hidden)] pub struct FooLaunch<'a,
..>(&'a Foo<..>)` implementing `ProcessExt` and `EulerSystem`/`EulerCoefficients`, then
launch `self.backend.system_paths(&FooLaunch(self), m)` and reshape. It borrows, so the
seed it advances is the process's own. Examples: `MultiGbmLaunch`, `McgnsLaunch`,
`MultifactorHestonLaunch`, `BgmRow(self, i)` (one launch per independent row),
`CfouParts`.

## 4. Tests

**4a. Device law** — `tests/device_law/<group>.rs`, declared
`pub(crate) mod <group>;` inside the `mod device_law { .. }` block of
`tests/device_law.rs`. The binary is gated `#![cfg(any(feature = "metal", feature =
"cuda"))]` and is `f32` throughout. Groups — pick by what the comparison must allow for,
not by source directory: `bounded`, `conditional_variance`, `curves`, `fractional`,
`gaussian`, `jumps`, `levy`, `memory`, `rows`, `systems`. Helpers in
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
count (`3 * M`, `4 * M`) and keep the tolerance. **Pin exact things exactly** — curve
tabulation, the initial point and the runtime-guard fallback get equality assertions,
not statistics; the fallback must be bit-identical to the plain host build
(`assert_eq!(build().on::<Device>().sample(), build().sample())`).

**4b. Sabotage verification is mandatory.** A passing device-law test proves
nothing until it has been made to fail:

```bash
  cp stochastic-rs-stochastic/src/euler/families.rs /tmp/families.bak
  # in the new family scale one term by `lit(3.0)`, literal on the RIGHT:
  #   `sigma * dz`  ->  `sigma * dz * lit(3.0)`
  cargo test -p stochastic-rs-stochastic --features metal --test device_law foo_agrees
  # it MUST fail. If it passes, the statistic is blind — pick another one.
  cp /tmp/families.bak stochastic-rs-stochastic/src/euler/families.rs
  cmp /tmp/families.bak stochastic-rs-stochastic/src/euler/families.rs && echo reverted
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
checks): `cargo test -p stochastic-rs-stochastic --features metal,cubecl-wgpu
--lib euler::`, covering `every_family_has_a_probe`,
`every_family_runs_on_the_device`, `the_cubecl_kernel_matches_the_generated_one` and
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
`py_process_1d!` for an `Array1` output (69 uses), `py_process_2x1d!` for `[Array1; 2]`
(10), `py_process_2d!` for `Array2` (2). Each has a plain and a `device` arm; the
trailing bare `device` token adds a `device=` keyword routed through
`py_device_dispatch!` (46 of the 81 invocations carry it) — add it whenever the process
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
hand-written `#[pyclass]` fixing `D = CallableDist<T>` instead — see `PyMerton`.

Register in the single flat `#[pymodule]` of `stochastic-rs-py/src/lib.rs` (its only
source file): a `use` in the `// Stochastic — <dir>` banner group, then
`m.add_class::<PyFoo>()?;` in the bare-ident form. The Python-visible name is the Rust
ident unless `#[pyclass(name = "..")]` renames it, and the macro-generated classes carry
no rename, so it really is `sr.PyFoo`.

No live pytest enumerates classes or asserts a count, so nothing there breaks —
`tests/python_bindings_smoke.py` at the repo root has a hardcoded name list but is
stale, outside `testpaths`, and run by neither pytest nor CI; do not add to it. Add a
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

1. `CLAUDE.md` — the `131 processes` in the layout tree and the `ProcessExt`
   bullet's `**132** concrete implementors over **131** processes`. Recount with
   the command that bullet quotes (the `-A1` is load-bearing):
   `grep -rhE -A1 "impl.*ProcessExt<T>" stochastic-rs-stochastic/src --include='*.rs' | grep -oE "for [A-Z][A-Za-z0-9]*" | sed 's/for //' | sort -u | wc -l`
   Implementors exceed processes by the launch/row views, which carry no
   reproducibility guard — today 138 minus `AdgRow`, `BgmRow`, `CfouParts`,
   `McgnsLaunch`, `MultiGbmLaunch`, `MultifactorHestonLaunch`, `WuZhangPair`
   = 131, matching `grep -rhoE "guard!\(" stochastic-rs-stochastic/tests/reproducibility_all_processes/*.rs | wc -l`.
   The prelude count is unaffected by a process.
2. `CLAUDE.md`'s Python entry counts, from §5's two greps.
3. `website/public/python-parity.json`, regenerated by `bun run python:parity`
   in `website/`. Its class regex is `m\.add_class::<(\w+)>\(\)`, which silently
   drops fully-qualified `add_class` lines — another reason to register in the
   bare-ident form.
4. `website/content/docs/concepts/gpu-support.mdx` — the support-matrix rows
   `83 processes` (one component), `33 processes` (systems), the explicit
   fractional name list with its `(11, counted above)`, and `the remaining 14
   processes`; the prose sentence opening `The 116 processes on the engine cover
   ...`; for a host-only process, the matching bullet under "What stays on the
   host, and why"; and, for a new *family*, `the engine's **110 families**`.
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
  cargo clippy -p stochastic-rs-stochastic --all-targets --features metal,cubecl-wgpu -- -D warnings
  RUSTDOCFLAGS="-D warnings" cargo doc -p stochastic-rs-stochastic --no-deps
  cargo check --workspace --no-default-features
  cargo test -p stochastic-rs-stochastic --features metal,cubecl-wgpu --lib euler::
  cargo test -p stochastic-rs-stochastic --features metal --test device_law
```

Both clippy runs matter: a helper only device launches call is dead code in the
no-feature build, so give it the crate's guard, as `flatten_curves` and `history_slot`
in `euler.rs` have — `#[cfg_attr(not(any(feature = "cuda", feature = "metal", feature =
"cubecl-cuda", feature = "cubecl-wgpu")), allow(dead_code))]`. `src/lib.rs` carries
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
χ² step the kernels do not carry); **Rust closures in the coefficients** (`Cheyette`'s
`Fn1D`/`Fn2D`, `VolterraSde`'s two `Fn2D` — no GPU path without a DSL for them); an
**output length that is not the grid** (`MultivariateHawkes`, `Hawkes` in horizon
mode); **state-dependent draws the frame does not carry** (`Wishart`'s non-central χ²
with a state-dependent non-centrality); **more than 4 state slots or 4 noise
components** (`Lmm` / `MultiGbm` / `Mcgns` above four, `Wishart` above `d = 2`); **a
two-dimensional field** (the sheet `Fbs`).

Before declaring a process host-only — the current list is whatever
`grep -rln "via host" stochastic-rs-stochastic/src --include='*.rs'` returns, and it
shrinks as the engine grows — check the seven documented ways round the cap:

1. **A launch view with a runtime cap** — pad a runtime `k` into the fixed
   four-slot family and fall back to the host above it (`MultiGbmLaunch`), or
   launch once per independent row (`BgmRow`).
2. **Runtime law recognition** — a process generic over `D: Distribution<T>`
   stays generic on the host and reaches the device by `Any` downcast in
   `src/process/cpoisson.rs::device_jump_sizes` / `device_arrival_rate`; an
   unrecognised law falls back, bit-identically to the plain host build.
3. **A `lift` clause** — a rough Volterra kernel as a Markov lift of up to
   `LIFT_SLOTS` nodes.
4. **A `history` clause** — the host's own exact O(n²) convolution for grids up
   to `HISTORY_SLOTS`, weights travelling as a curve. This is what moved
   `RoughHeston`, `FBatesSvj`, `RoughBergomi`, `Lfsm` and `Arima`/`Sarima` onto
   the engine.
5. **A `series` clause** — a shot-noise series drawn per path before the steps
   and summed into the grid cells its terms fall in, so the host's sort is not
   needed; grids up to `SERIES_SLOTS`. This is what moved `Cgmy`, `Cts`, `KoBoL`
   and `Rdts` onto the engine, as one family with host-folded constants.
6. **A `table` clause** — a monotone table built per path before the steps and
   inverted by interpolation at every step, the extent doubled until the table
   reaches the horizon; tables up to `TABLE_SLOTS`. This is what moved
   `InverseAlphaStableSubordinator` onto the engine.
7. **An exact recursion in place of a rejection loop** — a family reads two
   uniforms a step, so a thinning loop with a random number of proposals has no
   home in it, but an exact inverse-transform of the same law does. This is
   what moved `Hawkes` (Dassios–Zhao) onto the engine.

## Definition of done

- [ ] `//!` header with the LaTeX SDE and the paper reference (title, authors, DOI/arXiv)
- [ ] `pub` fields, `seed: S`, `backend: B = Cpu`, `Option<T>` optionals, `n >= 2` assert
- [ ] `new(..)`, a `with_*` per field, `Default` at `n = 252`, `t = 1`; `pub mod` registered alphabetically
- [ ] `#[doc(hidden)]` sampler implementing `PathSampler<T>`; `ProcessExt<T>` implementing only `sampler()`
- [ ] `sampler()` derives (`self.seed.derive()`), constructs no `Unseeded`, races no shared atomic
- [ ] `backend_switch!` with the right arm and **every** field named
- [ ] Family declared in `families.rs`; the six DSL traps checked
- [ ] `EulerSpec` variant + `encode()` arm; ≤ 20 params, ≤ 8 curves, ≤ 4 states, ≤ 4 noises
- [ ] `EulerCoefficients`/`EulerSystem` impl; `draw_seed`; `host_sample` calls the sampler **and** `advance_chunk_seed`
- [ ] `cubecl.rs` step + report arms (+ lift, + history, + series, + table), count checked against the family count
- [ ] `family_name` arm and a probe in the arity-matching `every_*_family()` list
- [ ] All five `ProcessExt` methods overridden through the backend, or a `device_ready()` guard with a host fallback and an `expect` in the hook
- [ ] Device-law case in the right `device_law/<group>.rs`; statistic moves with every parameter; tolerance not padded
- [ ] **Sabotage-verified**: term scaled by `lit(3.0)` → test fails → reverted, `cmp` clean
- [ ] Chunk-invariance test extended if the family is fGN-fed
- [ ] `guard!` line added; that tree's per-directory count and total bumped
- [ ] Python macro invoked (`device` flag if on the engine) and `add_class` registered
- [ ] `CLAUDE.md` counts, `python-parity.json`, `gpu-support.mdx` matrix + prose, notebook cell 5 updated
- [ ] Both clippy runs, `cargo doc -D warnings`, `cargo check --workspace --no-default-features`, parity + device-law suites green
- [ ] Full battery run in a subagent with `CARGO_TARGET_DIR` in the scratchpad

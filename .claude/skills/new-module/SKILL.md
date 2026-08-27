---
name: new-module
description: Enforces scalability, integration, and compatibility requirements when creating any new module in stochastic-rs — covers stochastic, quant, stats, distributions, copulas, and ai
---

# New Module Integration Rules

Every new module must be **scalable** (trait-based extensibility), **integrated** (works with existing traits and pipelines), and **compatible** (derives, bounds, API conventions match the rest of the codebase).

## 1. Where to place new code

Determine the correct top-level module first. Do NOT create a new top-level module without explicit approval.

Code lives in **sub-crates**, not in the umbrella. The umbrella's own
`src/` holds exactly three files — `lib.rs`, `traits.rs`, `bridges.rs`
— and `lib.rs` merely re-exports each sub-crate under a short alias
(`pub use stochastic_rs_stochastic as stochastic;`). There is no
`src/stochastic.rs`, `src/quant.rs`, `src/stats.rs`,
`src/distributions.rs` or `src/copulas.rs`; that was the pre-split
layout.

| Domain | Public path | Crate you edit | Registration file | Examples |
|---|---|---|---|---|
| Stochastic processes (diffusion, jump, volatility, noise, interest, autoregressive) | `stochastic_rs::stochastic::…` | `stochastic-rs-stochastic/src/` | that crate's `src/lib.rs` | Gbm, Heston, Cir, Fbm |
| Quantitative finance (pricing, bonds, calibration, calendar, FX, portfolio, strategies, vol_surface) | `stochastic_rs::quant::…` | `stochastic-rs-quant/src/` | that crate's `src/lib.rs` | BSMPricer, schedule builder, FX forward |
| Statistical estimators & tests (MLE, KDE, stationarity, normality, spectral) | `stochastic_rs::stats::…` | `stochastic-rs-stats/src/` | that crate's `src/lib.rs` | Gaussian KDE, ADF test, Hurst estimators |
| Probability distributions | `stochastic_rs::distributions::…` | `stochastic-rs-distributions/src/` | that crate's `src/lib.rs` | SimdNormal, SimdAlphaStable |
| Copula models | `stochastic_rs::copulas::…` | `stochastic-rs-copulas/src/` | that crate's `src/lib.rs` | Clayton, GaussianCopula, TCopula |
| RNG foundation | `stochastic_rs::simd_rng::…` | `stochastic-rs-core/src/` | that crate's `src/lib.rs` | SimdRng, SeedExt, Deterministic |
| Neural network / AI models | `stochastic_rs::ai::…` (feature-gated) | `stochastic-rs-ai/src/` | that crate's `src/lib.rs` | Vol surrogates |

The sub-crate split is transparent to users: the umbrella keeps the
existing public API, so a new module in `stochastic-rs-quant/src/foo.rs`
is reachable as `stochastic_rs::quant::foo::…` with no umbrella edit at
all. You only touch the umbrella when adding a **trait** (mirror it in
`src/traits.rs` — `tests/prelude_completeness.rs` turns a dropped
re-export into a compile error) or a prelude item.

### Adding a submodule within an existing top-level module

Three file patterns exist in the project — choose the simplest that fits:

**Pattern A — Leaf file** (single file, no subdirectory):
```
stochastic-rs-stats/src/my_estimator.rs
```
Use when the implementation is self-contained in one file. Most `-stats`
and `-distributions` modules follow this.

**Pattern B — Sibling root file + directory** (multiple subfiles):
```
stochastic-rs-quant/src/my_module.rs     ← module root: doc header, pub mod, re-exports
stochastic-rs-quant/src/my_module/
  engine.rs
  types.rs
```
Use when the module has 2+ logical components. This is the dominant
shape in `-quant`, where nearly every top-level module appears twice in
a directory listing — `pricing.rs` beside `pricing/`, `calibration.rs`
beside `calibration/`, and so on. Note that the root is a **sibling**
`.rs` file, not a `mod.rs` inside the directory.

**Pattern C — Directory with mod.rs** (when root defines shared types):
```
stochastic-rs-stochastic/src/mc/
  mod.rs                         ← defines McEstimate<T> + pub mod declarations
  antithetic.rs
  mlmc.rs
```
Use when the module root itself defines shared types alongside submodule
declarations. In this workspace **only `mc/` actually follows Pattern
C** — `noise/fgn/` looks like it should but is Pattern B, with its root
in the sibling `noise/fgn.rs` and no `mod.rs` inside the directory.

Both B and C are in active use; pick whichever the surrounding crate
already uses rather than introducing the other one next to it.

### Module root must contain

1. `//!` doc comment with LaTeX formula summarising the core concept
2. `pub mod` declarations for all submodules
3. `pub use` re-exports of user-facing types
4. Shared traits or types that submodules need (define at root, not in a subfile)

### Registration

- **Submodule within an existing module:** add `pub mod my_module;` in
  the parent's root `.rs` (Pattern B) or `mod.rs` (Pattern C),
  alphabetically.
- **New top-level module in a sub-crate:** add `pub mod my_module;` to
  **that sub-crate's** `src/lib.rs` — e.g.
  `stochastic-rs-quant/src/lib.rs`. Not the umbrella's. Requires
  approval per `dev-rules`.
- **Feature-gated module:** `#[cfg(feature = "my_feature")] pub mod my_module;`,
  and propagate the feature from the sub-crate to the umbrella — see
  `feature-flag-management`.
- **A new trait:** additionally mirror it in the sub-crate's
  `src/traits.rs` **and** the umbrella's `src/traits.rs`. (Caveat:
  `-stochastic`, `-quant`, `-stats`, `-distributions` and `-copulas`
  each have a `src/traits.rs`; `-core` and `-ai` do **not** — `-core`'s
  `SeedExt` / `SimdRng` live under `src/simd_rng/` and are re-exported
  from its `lib.rs`.) Decide
  separately whether it belongs in `src/lib.rs`'s `prelude` — hub
  membership and prelude membership are independent (see `CLAUDE.md`).

## 2. Trait integration map

Before writing code, determine which existing traits the new types should implement.

### `stochastic/` modules

| Type | Required trait | Effect |
|---|---|---|
| Any stochastic process | `ProcessExt<T: FloatExt>: Send + Sync` | Gets `sample()`, `sample_map(m, f)`, `sample_par(m)` (rayon parallel). GPU backends are selected with `.on::<B>()`, not by a `sample_cuda` method |
| Process with Malliavin support | `MalliavinExt<T>` or `Malliavin2DExt<T>` | Malliavin derivative computation |
| Probability distribution | `DistributionExt` | CF, PDF, CDF, moments |
| SIMD-accelerated distribution | `DistributionSampler<T>` (`stochastic-rs-distributions/src/traits/distribution.rs`) | Requires `fill_slice()` + `fork()`; `sample_matrix()` / `sample_n()` are provided |

### `quant/` modules

| Type | Required trait | Effect |
|---|---|---|
| Single-underlying option pricer | `ModelPricer` | `price_call(s, k, r, q, tau)` / `price_put`. Vol-surface construction additionally requires the `VanillaEuropeanCall` marker (which carries `vanilla_call_forward`); `ModelSurface` blanket-impls over **that**, not over `ModelPricer` |
| Short-rate / bond model | `ShortRatePricer` | `zero_coupon_price(r0, tau)` / `zero_yield` |
| Multi-asset or path-dependent pricer | none — convention only | Model params on the struct, query passed to inherent `call_put(...)` / `price_call(...)`. A shared trait would abstract over nothing |
| Fourier / characteristic-function model | `FourierModelExt` | Auto-gets `ModelPricer` **and** `VanillaEuropeanCall`, hence `ModelSurface`, via blanket impls |
| Calibration result | `ToModel` | Connects to `build_surface_from_calibration()` pipeline |
| Holiday / business-day calendar | `CalendarExt` | Plugs into `BusinessDayConvention::adjust()` and `ScheduleBuilder` |
| Type needing tau from dates | Use `TimeExt::tau_with_dcc(DayCountConvention)` | Proper day-count instead of hardcoded `/365.0` |

### `copulas/` modules

| Type | Required trait | Effect |
|---|---|---|
| Bivariate copula | `BivariateExt` | 11 required methods; `sample()` / `fit()` / inversion are defaulted. Tau is `tau()` / `set_tau()` accessors, not a `kendall_tau()` method. See `copula-bivariate` |
| Multivariate copula | `MultivariateExt` | `sample()`, `fit()`, `pdf()`, `cdf()` |

### Blanket-impl chains (do NOT duplicate by hand)

```
FourierModelExt  ──blanket──▸  ModelPricer
FourierModelExt  ──blanket──▸  VanillaEuropeanCall (: ModelPricer)
VanillaEuropeanCall  ──blanket──▸  ModelSurface
```

Implement the lowest-level trait; upstream is automatic.

## 3. Extensibility — require a trait, not a concrete type

When a new module accepts a pluggable component, define or reuse a **trait**.

Pattern (from `BusinessDayConvention::adjust`):
```rust
pub fn my_function(calendar: &(impl CalendarExt + ?Sized)) -> NaiveDate {
    // works with &Calendar AND &dyn CalendarExt (trait objects)
}
```

The `+ ?Sized` bound is required to also accept `&dyn Trait`.

If the module creates a **new** extensibility point:
1. Define the trait in the **module root** (e.g., `my_module.rs`)
2. Implement it for the built-in concrete type
3. Re-export it
4. Accept `&(impl MyTrait + ?Sized)` in functions, not the concrete type

## 4. Type requirements

Every new `pub struct` and `pub enum` must have:

| Requirement | How | Why |
|---|---|---|
| `Debug` | `#[derive(Debug)]` | Debugging, error messages |
| `Clone` | `#[derive(Clone)]` | Composability — users clone pricers, processes, calendars |
| `Send + Sync` | Automatic for simple types; verify with `Box<dyn …>` or `Rc` | Required for `ProcessExt`, `sample_par()`, rayon |
| `Display` (enums) | `impl Display` | Logging, error messages |
| `Default` (where meaningful) | `#[derive(Default)]` + `#[default]` on variant | Ergonomic construction |
| `Copy` (small value types) | `#[derive(Copy)]` | Enums and small structs without heap data |
| `Eq + Hash` (identifier types) | `#[derive(PartialEq, Eq, Hash)]` | Map keys, dedup, comparisons |

**Do NOT add** `Serialize` / `Deserialize` — `serde` is not a dependency.

## 5. Numeric conventions

| Rule | Detail |
|---|---|
| Generic float | All numerical structs/functions use `T: FloatExt`. Never hardcode `f64`. Use `T::from_f64_fast()` for constants. |
| Arrays | Use `ndarray::Array1<T>`, `Array2<T>`. Never `Vec<T>` for numerical data. |
| Day fractions | Use `DayCountConvention::year_fraction()` or `TimeExt::tau_with_dcc()`. Never hardcode `/365.0` or `/360.0`. |
| Annualisation | Accept the factor as a parameter. Never hardcode `252.0` or `365.0`. |
| Random sampling | Use the project's `SimdRng` / `SimdFloatExt` infrastructure for SIMD-accelerated generation. |
| Complex numbers | Use `num_complex::Complex<T>`. |

## 6. Re-export conventions

In the module root, re-export **user-facing** types only:

```rust
pub use engine::MyEngine;
pub use types::{MyConfig, MyResult};
```

Keep internal helpers `pub(crate)` or private. Match the pattern of sibling modules in the same top-level module.

## 7. Integration with existing pipelines

### Pricing pipeline (quant)
If the module produces a pricer, verify it works with:
- `build_surface_from_model<M: ModelSurface + ?Sized>(model: &M, s, r, q, strikes, maturities)`
  — vol-surface construction. Generic, **not** `&dyn`, and the bound is
  `ModelSurface`, not `ModelPricer`: the Black inversion is only
  meaningful for a European vanilla call, which is what `ModelSurface`'s
  `VanillaEuropeanCall` supertrait asserts.
- `build_surface_from_calibration<C: ToModel>(calibration: &C, s, r, q, strikes, maturities)
  where C::Model: ModelSurface` — calibration → vol-surface. Note the
  extra `where` clause: implementing `ToModel` is not enough on its own.

### Calendar pipeline (quant)
If the module uses dates, verify it works with:
- `BusinessDayConvention::adjust(date, &calendar)` — business day adjustment
- `ScheduleBuilder::new(…).calendar(cal).build()` — schedule generation
- `TimeExt::tau_with_dcc(dcc)` — year fraction from dates

### Process pipeline (stochastic)
If the module defines a stochastic process, verify:
- `sample()` returns the correct `Output` type
- `sample_par(m)` works (all fields must be `Send + Sync`)
- Noise inputs follow the `SeedExt` pattern if seeded

### Distribution pipeline (distributions)
If the module defines a distribution, verify:
- `DistributionSampler<T>` is implemented for bulk sampling
- `fill_slice()` uses SIMD where possible
- `sample_matrix()` works for multi-core benchmarks

## 8. Feature gating

If the module requires an optional external dependency:

1. Add dependency with `optional = true` in `Cargo.toml`
2. Add feature: `my_feature = ["dep:my_crate"]`
3. Gate module: `#[cfg(feature = "my_feature")] pub mod my_module;`
4. Gate imports in shared code: `#[cfg(feature = "my_feature")]`

Default features remain `default = []`.

## 9. Testing and benchmarks

Every new module must include:

1. **Comparison test** (`tests/my_module_test.rs`):
   - Validate output against reference (Python, R, MATLAB, or paper's tables/figures)
   - Test trait integrations (e.g., custom `CalendarExt` impl, `sample_par` correctness)
   - Test edge cases (zero maturity, degenerate parameters, boundary conditions)

2. **Criterion benchmark** (`benches/my_module.rs`):
   - Benchmark the hot path
   - Register in `Cargo.toml`: `[[bench]] name = "my_module" harness = false`

3. **Integration test** — verify end-to-end with existing pipelines where applicable

## 10. Documentation

Every new file must have:
- `//!` doc header citing the paper/reference (title, authors, DOI or arXiv ID)
- LaTeX formula in the doc header
- `///` docs on all public items

## Quick checklist

Before marking a new module as done:

- [ ] Placed in the correct **sub-crate** (`stochastic-rs-stochastic`, `-quant`, `-stats`, `-distributions`, `-copulas`, `-core`, `-ai`) — see §1; the umbrella `src/` is not where code goes
- [ ] Module root has LaTeX doc header and re-exports
- [ ] Registered in the parent module's `.rs` file (alphabetical order)
- [ ] All numerical code generic over `FloatExt`, arrays use `ndarray`
- [ ] Correct domain traits implemented (see §2 trait integration map)
- [ ] Extensibility points use traits, not concrete types (see §3)
- [ ] `Debug`, `Clone`, `Display`, `Default` derives on public types
- [ ] `Send + Sync` verified (no `Rc`, `Cell`, or unshared interior mutability)
- [ ] No hardcoded `/365.0`, `/360.0`, or `/252.0`
- [ ] Comparison test against reference implementation
- [ ] Criterion benchmark registered in `Cargo.toml`
- [ ] Scientific reference cited in file header
- [ ] `cargo clippy` clean

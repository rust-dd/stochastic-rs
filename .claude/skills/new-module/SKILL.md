---
name: new-module
description: Enforces scalability, integration, and compatibility requirements when creating any new module in stochastic-rs — covers stochastic, quant, stats, distributions, copulas, and ai
---

# New Module Integration Rules

Every new module must be **scalable** (trait-based extensibility), **integrated** (works with existing traits and pipelines), and **compatible** (derives, bounds, API conventions match the rest of the codebase).

## 1. Where to place new code

Determine the correct top-level module first. Do NOT create a new top-level module without explicit approval.

| Domain | Top-level module | Registration file | Examples |
|---|---|---|---|
| Stochastic processes (diffusion, jump, volatility, noise, interest rate, autoregressive) | `stochastic/` | `src/stochastic.rs` | GBM, Heston, CIR, fBM, GARCH |
| Quantitative finance (pricing, bonds, calibration, calendar, FX, portfolio, strategies, vol-surface) | `quant/` | `src/quant.rs` | BSM pricer, schedule builder, FX forward |
| Statistical estimators & tests (MLE, KDE, stationarity, normality, spectral) | `stats/` | `src/stats.rs` | Gaussian KDE, ADF test, Hurst estimator |
| Probability distributions | `distributions/` | `src/distributions.rs` | Normal, Alpha-stable, NIG |
| Copula models | `copulas/` | `src/copulas.rs` | Clayton, Gaussian, Student-t |
| Neural network / AI models | `ai/` (feature-gated) | `src/lib.rs` | Vol calibration NN |

### Adding a submodule within an existing top-level module

Three file patterns exist in the project — choose the simplest that fits:

**Pattern A — Leaf file** (single file, no subdirectory):
```
src/stats/my_estimator.rs
```
Use when the implementation is self-contained in one file. Most `stats/` and `distributions/` modules follow this.

**Pattern B — Root file + directory** (multiple subfiles):
```
src/quant/my_module.rs          ← module root: doc header, pub mod, re-exports, shared traits
src/quant/my_module/
  engine.rs
  types.rs
```
Use when the module has 2+ logical components. Most `quant/` modules (pricing, calibration, calendar, fx, bonds, vol_surface) follow this.

**Pattern C — Directory with mod.rs** (when root defines shared types):
```
src/stochastic/mc/
  mod.rs                         ← defines McEstimate<T> + pub mod declarations
  lsm.rs
  mlmc.rs
```
Use when the module root itself defines shared types alongside submodule declarations. The `mc/` module and `noise/fgn/` follow this.

### Module root must contain

1. `//!` doc comment with LaTeX formula summarising the core concept
2. `pub mod` declarations for all submodules
3. `pub use` re-exports of user-facing types
4. Shared traits or types that submodules need (define at root, not in a subfile)

### Registration

- **Submodule within existing top-level:** add `pub mod my_module;` in the parent's `.rs` file (alphabetical order)
- **New top-level module:** add `pub mod my_module;` in `src/lib.rs` (requires approval per dev-rules)
- **Feature-gated module:** `#[cfg(feature = "my_feature")] pub mod my_module;`

## 2. Trait integration map

Before writing code, determine which existing traits the new types should implement.

### `stochastic/` modules

| Type | Required trait | Effect |
|---|---|---|
| Any stochastic process | `ProcessExt<T: FloatExt>: Send + Sync` | Gets `sample()`, `sample_map(m, f)`, `sample_par(m)` (rayon parallel). GPU backends are selected with `.on::<B>()`, not by a `sample_cuda` method |
| Process with Malliavin support | `MalliavinExt<T>` or `Malliavin2DExt<T>` | Malliavin derivative computation |
| Probability distribution | `DistributionExt` | CF, PDF, CDF, moments |
| SIMD-accelerated distribution | `DistributionSampler<T>` (from `distributions.rs`) | Bulk `fill_slice()` + `sample_matrix()` |

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
| Bivariate copula | `BivariateExt` | `sample()`, `fit()`, `pdf()`, `cdf()`, Kendall's tau |
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
- `build_surface_from_model(&dyn ModelPricer, …)` — vol-surface construction
- `build_surface_from_calibration(&dyn ToModel, …)` — calibration → vol-surface

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

- [ ] Placed in the correct top-level module (`stochastic/`, `quant/`, `stats/`, `distributions/`, `copulas/`)
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

---
name: feature-flag-management
description: Conventions for adding / propagating Cargo features across the stochastic-rs workspace. Invoke when adding a new optional dependency, gating a sub-module, or debugging "feature X enabled in crate A but not B" build errors.
---

# Feature flag management — stochastic-rs

The workspace has 9 sub-crates, several of which carry optional
dependencies (`openblas`, `cuda`, `metal`, `python`, `yahoo`, `ai`,
`hotpath`, `viz`). Without discipline, `cargo check --all-features`
explodes with "feature X needed but not propagated" or, worse, an
unflagged path silently compiles a `Box<dyn Fn(f64) -> f64>` that
panics at runtime.

This SKILL documents the conventions that prevent the **§4.1
feature-flag trap** (rc.0 release-blocker: a `dep:` proxy feature that
compiled in single-crate isolation but blew up under
`--all-features`).

## 1. Three patterns, three uses

### 1.1 `dep:` proxy syntax (preferred for optional dependencies)

```toml
# In stochastic-rs-quant/Cargo.toml:
[dependencies]
yahoo_finance_api = { workspace = true, optional = true }

[features]
yahoo = ["dep:yahoo_finance_api"]
```

Why: the `dep:` prefix tells Cargo that "yahoo" is a feature **only**,
and never a transitive enable. Without `dep:`, `cargo` auto-creates a
feature named `yahoo_finance_api` whenever the crate has an optional
dep of that name, and any sub-crate that lists `yahoo_finance_api` as a
plain dep silently activates it. That's the exact failure mode we hit
in rc.0.

### 1.2 Workspace propagation for cross-crate features

When the umbrella crate (`stochastic-rs/Cargo.toml`) needs to expose a
feature that lives in a sub-crate, propagate via the
`<sub-crate>/<feature>` syntax:

```toml
# In stochastic-rs/Cargo.toml (umbrella):
[features]
ai = ["stochastic-rs-ai/python", "stochastic-rs-ai/openblas"]
gpu = ["stochastic-rs-stochastic/gpu", "stochastic-rs-distributions/gpu"]
openblas = [
  "stochastic-rs-stats/openblas",
  "stochastic-rs-quant/openblas",
  "stochastic-rs-stochastic/openblas",
  "stochastic-rs-distributions/openblas",
]
```

Why: `cargo` does not auto-enable sub-crate features when the umbrella
exposes a same-named one. You must list every sub-crate that needs it
explicitly. Forgetting a sub-crate is the most common cause of
`cargo --all-features` failures (a path that's gated on `openblas` in
two crates but only on `default` in a third compiles inconsistently).

### 1.3 Internal feature gates (sub-crate scoped)

For a sub-crate's *own* internal feature (not exposed via the umbrella):

```rust
// stochastic-rs-stats/src/openblas_estimators.rs
#![cfg(feature = "openblas")]   // module-level guard

// or, at item level:
#[cfg(feature = "openblas")]
pub fn linear_regression_lapack(...) -> ... { ... }

#[cfg(not(feature = "openblas"))]
pub fn linear_regression_lapack(...) -> ... {
  panic!("linear_regression_lapack requires the 'openblas' feature");
}
```

The `#[cfg(not(feature))]` no-op variant is sometimes useful for
keeping the public API stable, but the dev-rules feedback memo
discourages "make absent paths panic"; prefer plain `#[cfg(feature)]`
gating without a fallback so callers see a compile-time error.

## 2. Mandatory pre-publish checks

Before any version bump (see `release-checklist` SKILL):

```bash
# 2.1 Default-features baseline
cargo check --workspace --no-default-features

# 2.2 Most-common combinations
cargo check --workspace --features openblas
cargo check --workspace --features ai
cargo check --workspace --features python
cargo check --workspace --features cuda

# 2.3 The thermonuclear all-features path — catches the §4.1 trap
cargo check --workspace --all-features

# 2.4 Tests on the openblas path (most user-facing)
cargo test --workspace --exclude stochastic-rs-py --features openblas
```

If `cargo check --workspace --all-features` fails but the per-feature
checks pass, the failure is almost always a feature-flag interaction:
two features that work alone but conflict together. Common causes:

- Two optional deps both expose a `Send`-bound impl on a third type;
  `--all-features` makes both visible and the orphan rule rejects.
- A `#[cfg(any(feature = "x", feature = "y"))]` gate where the body
  references a type re-exported by only one of the two.
- A test that uses `#[cfg(feature = "x")]` but indirectly depends on a
  helper gated `#[cfg(feature = "y")]`.

## 3. Adding a new feature — checklist

To add feature `foo` (e.g. a new optional GPU backend, a new dataset
loader, a new distribution backend):

1. **Decide scope.** Is `foo` a sub-crate-only feature, or does it need
   to bubble up to the umbrella?

2. **Sub-crate Cargo.toml:**
   ```toml
   [dependencies]
   foo_dep = { workspace = true, optional = true }

   [features]
   foo = ["dep:foo_dep"]
   ```

3. **If the umbrella needs to expose it:**
   ```toml
   # stochastic-rs/Cargo.toml
   [features]
   foo = ["stochastic-rs-X/foo", "stochastic-rs-Y/foo"]
   ```

4. **Source-side gating:**
   - Module-level: `#![cfg(feature = "foo")]` at the top of `src/foo.rs`.
   - Re-exports: gate the `pub use` line.
   - Tests: `#[cfg(feature = "foo")]` on the test or test module.
   - Benches: in `Cargo.toml`, `[[bench]] required-features = ["foo"]`.

5. **CLAUDE.md update:** if `foo` exposes a new public surface (new
   trait, new type, new module path), document it in the relevant
   sub-crate's `CLAUDE.md` — the prelude listing or the "Key traits"
   block.

6. **Run the full feature matrix** (section 2) to confirm
   `--all-features` is still clean.

## 4. Anti-patterns

- **Do not** use bare `optional = true` without a `dep:` proxy
  feature. The auto-feature-named-after-the-dep behaviour is the
  rc.0 trap.
- **Do not** propagate features in *both* directions
  (umbrella → sub-crate AND sub-crate → umbrella via reexport).
  Pick one. The umbrella's `[features]` block is the canonical surface.
- **Do not** put runtime panics behind `#[cfg(not(feature))]`. Make
  the path a compile error so users know at build time.
- **Do not** add a new optional dependency without also adding a
  feature flag for it. The crate's default build size matters.

## 5. Diagnosing failures

| Symptom | Likely cause |
|---|---|
| `cargo check --all-features` fails, per-feature checks pass | Feature interaction — see section 2 |
| `cannot find type X in module foo` only with `--all-features` | Missing re-export gate, or `pub use` outside the right `cfg` |
| `the trait Send is not implemented for ...` only on certain features | A feature pulled in a non-Send dep through a transitive chain |
| Sub-crate publishes fine, umbrella fails on crates.io | Missing `version = "X.Y.Z"` on a path-only sub-crate dep |
| `unresolved import stochastic_rs_X::foo` only in `[dev-dependencies]` | Dev-deps don't inherit feature flags from main `[dependencies]`; explicit re-listing needed |

## 6. Reference: feature flags currently in the workspace

| Feature | Crates that publish it | Notes |
|---|---|---|
| `openblas` | `-stats`, `-quant`, `-stochastic`, `-distributions`, umbrella | LAPACK-backed linear algebra. |
| `cuda` | `-stochastic`, `-distributions`, `-ai`, umbrella | Native CUDA via cust + NVRTC. |
| `metal` | `-stochastic`, `-distributions`, umbrella | Apple Silicon GPU. |
| `python` | `-distributions`, `-stochastic`, `-quant`, `-stats`, `-copulas`, `-py` | PyO3 bindings (cdylib gated). |
| `yahoo` | `-quant` | Live-data tests; experimental. |
| `ai` | `-ai`, umbrella | Neural surrogates (rc.1 experimental). |
| `hotpath` | umbrella | Profiling-mode build. |
| `viz` | `-viz`, umbrella | Plotly grid plotter. |

## Related SKILLs

- `release-checklist` — uses this SKILL's matrix as a release gate.
- `add-gpu-sampler`, `add-jump-process` — invoke when the new module is
  feature-gated.

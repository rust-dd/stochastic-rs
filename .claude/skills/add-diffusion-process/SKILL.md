---
name: add-diffusion-process
description: How to add a new diffusion / SDE process to stochastic-rs-stochastic. Invoke when implementing GBM-like, OU-like, Vasicek-like, CIR-like, Heston-like models that satisfy `dX_t = drift dt + diffusion dW_t`.
---

# Add diffusion process — stochastic-rs-stochastic

This SKILL covers the recipe for adding a new diffusion process
(`stochastic-rs-stochastic/src/diffusion/<name>.rs`). The recipe
applies equally to volatility / interest-rate / credit-style
diffusions; for *fractional* processes (driven by fBm or a Volterra
kernel) see `add-fractional-process`. For *jump* processes see
`add-jump-process`.

## 1. The trait surface

A new process implements `ProcessExt<T>`
(`stochastic-rs-stochastic/src/traits/process.rs`). **You implement two
associated items and one method — not `sample()`:**

```rust
pub trait ProcessExt<T: FloatExt>: Send + Sync {
    type Output: Send;

    /// Reusable sampling state. #[doc(hidden)] — implementation detail.
    type Sampler<'a>: PathSampler<T, Output = Self::Output> where Self: 'a;

    /// The one required method: builds that state from `&self`.
    fn sampler(&self) -> Self::Sampler<'_>;

    // Everything below is defaulted; do not override without a reason.
    fn advance_chunk_seed(&self) {}                    // no-op for almost every process
    fn chunked_samplers(&self, m: usize) -> …          // sequential, pre-rayon seeding
    fn sample(&self) -> Self::Output                   // one path
    fn sample_map<R>(&self, m: usize, f: …) -> Vec<R>  // the parallel primitive
    fn sample_par(&self, m: usize) -> Vec<Self::Output>
}
```

There is no `sample_pair` / `sample_pair_par`. `sample()`,
`sample_map()` and `sample_par()` are all defaulted on top of
`sampler()`.

The sampler itself implements `PathSampler<T>`
(`traits/sampler.rs`), which is two methods:

```rust
pub trait PathSampler<T: FloatExt>: Send {
    type Output: Send;
    fn sample_into(&mut self, out: &mut Self::Output);  // overwrite, no alloc
    fn sample(&mut self) -> Self::Output;               // allocate + fill
}
```

The split exists so `sample_map` can reuse one sampler and one output
buffer per rayon chunk. `Output` is `Array1<T>` for a 1-D path;
`[Array1<T>; 2]` for a two-component process (Heston: price + vol);
`Array2<T>` for a curve. Marker traits (`OneDimensional`,
`TwoDimensional`, `CurveOutput`, …) blanket-implement off `Output`, so
you never write them by hand.

### Reproducibility requirement on implementors

`sampler()` must derive its per-call basis from `self.seed` with
`derive()`, from inside a single `&self` call — **never** `clone()`.
That is what makes `sample_par(m)` bit-identical across rayon
thread-pool sizes, because `chunked_samplers` calls `sampler()`
sequentially on the calling thread before any chunk reaches rayon.

The narrow exception: if your `sampler()` must `clone()` the seed
(because the clone feeds a persistent engine reused across a whole
chunk), you must also override `advance_chunk_seed`. `CirPlusPlus`
(`interest/cir_pp.rs`) is the in-tree example:
`fn advance_chunk_seed(&self) { self.seed.seed_value(); }`.

## 2. The struct + constructor

There is **one** constructor. It takes the seed source as its **last
parameter**, and the `S: SeedExt` type parameter selects `Unseeded` or
`Deterministic` at compile time. The pre-3.0 `new` / `seeded(...)`
constructor pair is gone — do not add a `seeded(...)`.

```rust
// stochastic-rs-stochastic/src/diffusion/foo.rs

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct Foo<T: FloatExt, S: SeedExt = Unseeded> {
    /// Mean-reversion speed.
    pub theta: T,
    /// Long-run mean.
    pub mu: T,
    /// Diffusion / noise scale.
    pub sigma: T,
    /// Number of discretisation points (≥ 2).
    pub n: usize,
    /// Initial value (defaults to zero / 1.0 — match the SDE convention).
    pub x0: Option<T>,
    /// Total horizon; defaults to 1.0.
    pub t: Option<T>,
    /// Seed strategy (compile-time: Unseeded or Deterministic).
    pub seed: S,
}

/// Every field gets a matching `with_*` builder setter, e.g.
/// `Foo::default().with_theta(0.8)`.
impl<T: FloatExt, S: SeedExt> Foo<T, S> {
    pub fn new(
        theta: T, mu: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>,
        seed: S,                                    // <- LAST parameter
    ) -> Self {
        assert!(n >= 2, "n must be at least 2");
        Self { theta, mu, sigma, n, x0, t, seed }
    }

    pub fn with_theta(mut self, theta: T) -> Self { self.theta = theta; self }
    // … one per field, including `with_seed`.
}

/// Give the process a `Default` at textbook parameters, pinned to
/// `Unseeded`, and say in the doc comment what the numbers mean.
impl<T: FloatExt> Default for Foo<T, Unseeded> {
    fn default() -> Self {
        Self::new(/* … */, Unseeded)
    }
}
```

Call sites then read:

```rust
Foo::<f64, _>::new(0.5, 0.0, 0.1, 100, None, None, Unseeded)              // fresh entropy
Foo::<f64, _>::new(0.5, 0.0, 0.1, 100, None, None, Deterministic::new(42)) // reproducible
```

The `seed: S` field is the compile-time switch. `Unseeded` is a ZST
carrying no state; `Deterministic` holds an atomically-advanced state
and supports `reseed(s)` for sweeping seeds without rebuilding the
process.

If the struct caches anything derived from its parameters (as `Gbm`
caches its terminal-log-normal `(ln_mu, ln_sigma)`), factor the
recomputation into a free function and call it from `new` **and** from
every `with_*` setter that touches an input — that is the only thing
keeping the cache from drifting.

### `Clone` snapshots the seed — deliberately

Cloning a process snapshots `seed` as it stands, so `let b = a.clone();
assert_eq!(a.sample(), b.sample())` holds. That is the
common-random-numbers property behind bump-and-reprice Greeks, and it
intentionally diverges from `stochastic-rs-distributions`, where
cloning a distribution re-seeds independently. For a genuinely
independent stream, construct a fresh seed rather than cloning.

## 3. The naming convention — `theta` vs `mu`

**Mandatory**: the workspace uses

- `theta`: mean-reversion **speed** (κ in many texts, e.g. Brigo).
- `mu`: long-run **mean level** (θ in many texts).

The rc.0 CIR bug shipped because Vasicek and CIR had `theta`/`mu`
swapped between source and tests. Every new diffusion that has a
mean-reversion-speed × long-run-mean structure (`dX = θ(μ-X)dt + ...`)
**must** keep this convention. If the canonical paper uses different
symbols, document the translation in the struct's doc comment but use
our names in the field.

## 4. The sample implementation

For Euler-Maruyama discretisation:

Split it: `sampler()` hoists everything that does not vary per path,
and the `PathSampler` impl runs the recursion.

```rust
impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Foo<T, S> {
    type Output = Array1<T>;
    type Sampler<'s> = FooSampler<T> where Self: 's;

    fn sampler(&self) -> FooSampler<T> {
        // `saturating_sub(1).max(1)` keeps dt finite for degenerate n <= 1.
        let n_increments = self.n.saturating_sub(1).max(1);
        let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
        FooSampler {
            n: self.n,
            x0: self.x0.unwrap_or(T::zero()),
            theta: self.theta,
            mu: self.mu,
            sigma: self.sigma,
            dt,
            // The Gaussian source carries dt.sqrt() as its std, so the
            // recursion multiplies by sigma alone. Seeded from `&self.seed`
            // — this is the `derive()` the reproducibility rule requires.
            normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
        }
    }
}

pub struct FooSampler<T: FloatExt> { /* the hoisted fields above */ }

impl<T: FloatExt> FooSampler<T> {
    fn fill_path(&mut self, out: &mut [T]) {
        if out.is_empty() { return; }
        let mut gn = Array1::<T>::zeros(out.len() - 1);
        if let Some(gn_slice) = gn.as_slice_mut() {
            self.normal.fill_slice(gn_slice);      // bulk fill, one call
        }
        out[0] = self.x0;
        for i in 1..out.len() {
            let drift = self.theta * (self.mu - out[i - 1]);
            out[i] = out[i - 1] + drift * self.dt + self.sigma * gn[i - 1];
        }
    }
}

impl<T: FloatExt> PathSampler<T> for FooSampler<T> {
    type Output = Array1<T>;
    fn sample_into(&mut self, out: &mut Array1<T>) {
        self.fill_path(out.as_slice_mut().expect("Foo output must be contiguous"));
    }
    fn sample(&mut self) -> Array1<T> {
        array1_from_fill(self.n, |out| self.fill_path(out))
    }
}
```

Four conventions in there:

- **Draw Gaussians in bulk** with `SimdNormal::fill_slice`, never one
  per step. Per `dev-rules` §7a, `rand_distr::StandardNormal` is
  reserved for `benches/` — library code uses the workspace's own
  `Simd*` distributions, and the seed goes to the **constructor**, not
  to a `fill_slice(rng, …)` argument (which is ignored).
- **Fold constants into the noise.** Putting `dt.sqrt()` in the
  distribution's std keeps it out of the inner loop.
- Use `T::from_f64_fast` / `T::from_usize_` (not `T::from`) at the
  `FloatExt` boundary.
- Use `array1_from_fill` (`crate::buffer`) for the allocating path so
  the buffer is filled in place rather than zeroed then overwritten.

For higher-order schemes (Milstein, SRK2, SRK4) there is a generic
driver rather than free step functions: `crate::sde::Sde<T, F, G, B>`
takes drift / diffusion closures at `Sde::new(drift, diffusion, noise,
hursts)` and dispatches on an `SdeMethod` enum inside `.solve(…)`.
Reach for it when the scheme, not the model, is the thing you are
adding.

## 5. Python wrapper macro

After the inherent + ProcessExt impls, append the Python wrapper macro
**at the bottom of the source file**:

```rust
py_process_1d!(PyFoo, Foo,
  sig: (theta, mu, sigma, n, x0=None, t=None, seed=None, dtype=None),
  params: (theta: f64, mu: f64, sigma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);
```

`sig` is the full PyO3 signature and **must** end with `seed=None,
dtype=None`; `params` lists only the model parameters — the macro
appends `seed: Option<u64>` and `dtype: Option<&str>` itself. Compare
`Gbm`'s real invocation at the bottom of `diffusion/gbm.rs`.

The macro generates a `#[pyclass]` holding four `Option` slots
(`inner_f32`, `inner_f64`, `seeded_f32`, `seeded_f64`) and dispatching
on `(seed, dtype)` in `__new__`, which is how one Python class serves
both float widths and both seed strategies. It exposes exactly two
methods:

- `sample()` → 1-D numpy array
- `sample_par(m)` → `(m, n)` numpy array

There is **no** `sample_seeded(seed)` method. The seed is a constructor
argument: `PyFoo(..., seed=42)`.

For two-component output (e.g. `[Array1<T>; 2]`) use `py_process_2x1d!`
(`volatility/sabr.rs`, `volatility/heston_log.rs`, `interest/duffie_kan.rs`).
For `Array2<T>` output use `py_process_2d!` (`diffusion/cfou.rs`,
`sheet/fbs.rs`). A process whose constructor takes a Python callable —
a generic jump distribution, say — cannot use the macros at all and is
hand-written (`PyMerton`).

After the macro, **register the class** in `stochastic-rs-py/src/lib.rs`:

```rust
use stochastic_rs_stochastic::diffusion::foo::PyFoo;
// ...
m.add_class::<PyFoo>()?;
```

## 6. Backward-compat aliases

If you rename an existing process, add an alias in
`stochastic-rs-stochastic/src/aliases.rs` — the crate root, not a
per-directory file:

```rust
#[deprecated(since = "2.0.0", note = "renamed to `Foo` for naming consistency")]
pub use crate::diffusion::foo::Foo as OLDFOO;
```

The file is `#![allow(deprecated)]` at the top and currently carries the
v1.x acronym-style names (`GBM`, `OU`, `CIR`, …) renamed to PascalCase
in v2.0.0, scheduled for removal in v3.0.0. Keep entries alphabetised by
module path, as the existing ones are.

## 7. Testing requirements

A new diffusion ships with at least four tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// 1. Seeded determinism.
    #[test]
    fn seeded_is_deterministic() {
        let a = Foo::<f64, _>::new(0.5, 0.0, 0.1, 100, None, None,
                                   Deterministic::new(42));
        let b = a.clone();      // Clone snapshots the seed — see §2
        assert_eq!(a.sample(), b.sample());
    }

    /// 1b. `sample_par(m)` is bit-identical across rayon thread-pool
    ///     sizes. New processes must also be added to
    ///     `tests/reproducibility_all_processes.rs`, the crate-wide guard.
    #[test]
    fn sample_par_is_reproducible() { }

    /// 2. Pure drift (sigma = 0) collapses to deterministic ODE.
    #[test]
    fn zero_diffusion_matches_deterministic() {
        // dX/dt = theta * (mu - X) → X(t) = mu + (X0 - mu) * exp(-theta * t)
        // ...
    }

    /// 3. Theoretical moment recovery on long path.
    #[test]
    fn long_path_mean_matches_theory() {
        // ...
    }

    /// 4. Constructor validates n >= 2.
    #[test]
    #[should_panic(expected = "n must be at least 2")]
    fn rejects_n_below_two() {
        let _ = Foo::<f64, _>::new(0.5, 0.0, 0.1, 1, None, None, Unseeded);
    }
}
```

The first test (seeded determinism) is non-negotiable; without it
calibrators that consume the process get nondeterministic regression
tests downstream.

## 8. CLAUDE.md / prelude updates

Per `CLAUDE.md`, the prelude does NOT include individual process
types — users go through `stochastic_rs::stochastic::diffusion::foo::Foo`.
But the umbrella crate's "Workspace layout" section may mention notable
new processes (e.g. "127 processes, incl. interest::lmm::Lmm"). Update
that line if your new process is material.

## 9. Anti-patterns

- **Do not** call `thread_rng()`, `rand::rng()` or
  `rand_distr::StandardNormal` in library code — `dev-rules` §7a
  reserves them for `benches/`. Seed the workspace's own `Simd*`
  distribution at its constructor, from `&self.seed`.
- **Do not** add a `seeded(...)` constructor. That shape is pre-3.0 and
  gone; the seed is `new`'s last parameter.
- **Do not** implement `sample()` directly. Implement `sampler()` and
  let the defaults build `sample` / `sample_map` / `sample_par` on it —
  overriding `sample()` silently opts the process out of the
  chunk-reuse and thread-count-stability the defaults provide.
- **Do not** `clone()` the seed inside `sampler()`. Use `derive()`; if
  you genuinely must clone, override `advance_chunk_seed` too (§1).
- **Do not** draw Gaussians one per step. Bulk-fill with `fill_slice`.
- **Do not** name fields `kappa` / `theta` (Brigo convention). The
  workspace uses `theta` / `mu`. Sticking to local conventions when the
  surrounding code uses ours produces silent numeric bugs (rc.0 CIR).
- **Do not** put validation behind `debug_assert!`. `assert!(n >= 2)`
  is a permanent invariant; debug_assert hides it from release builds
  and lets users hit cryptic out-of-bounds panics in `path[0]`.

## 10. Reference impls (in increasing complexity)

- `Bm` (`process/bm.rs`) — Brownian motion, no parameters besides `n`.
- `Gbm` (`diffusion/gbm.rs`) — geometric BM; the template for this
  SKILL, and the reference for a parameter-derived cache kept in sync
  across `with_*` setters.
- `Ou` (`diffusion/ou.rs`) — mean-reverting OU; the `theta` / `mu`
  reference.
- `Vasicek` (`interest/vasicek.rs`) — note the directory: mean-reverting
  short-rate models live under `interest/`, not `diffusion/`.
- `Cir` (`diffusion/cir.rs`) — CIR; carries an extra `use_sym` option.
- `Fou` (`diffusion/fou.rs`) — fractional OU; adds a third type
  parameter `B` (backend, defaulting to `Cpu`) and composes an `Fgn`
  driver. See `add-fractional-process`.
- `Heston` (`volatility/heston.rs`) — `Output = [Array1<T>; 2]`, and a
  third type parameter `Sch: HestonScheme = Euler` selecting the
  discretisation at compile time.
- `CirPlusPlus` (`interest/cir_pp.rs`) — the one process that overrides
  `advance_chunk_seed`; read it before writing a clone-based `sampler()`.

## Related SKILLs

- `add-fractional-process` — for Hurst-parameterised processes wrapping
  `Fgn` or extending `MarkovLift`.
- `add-jump-process` — for compound-Poisson / Lévy-driven additions.
- `python-bindings` — invoked by `py_process_*!` and the registration
  step.
- `feature-flag-management` — if your process needs an optional GPU
  backend or LAPACK helper.

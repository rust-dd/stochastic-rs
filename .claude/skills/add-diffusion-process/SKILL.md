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

A new process must implement `ProcessExt<T>` for the standard sample
shape:

```rust
// stochastic-rs-stochastic/src/traits.rs

pub trait ProcessExt<T: FloatExt> {
    type Output;
    fn sample(&self) -> Self::Output;

    // Optional overrides (defaults are: call `sample` and unwrap)
    fn sample_par(&self, _m: usize) -> Vec<Self::Output> { ... }
    fn sample_pair(&self) -> [Self::Output; 2] { ... }
    fn sample_pair_par(&self, _m: usize) -> Vec<[Self::Output; 2]> { ... }
}
```

For the typical 1-D path output `Array1<T>`, defaults are fine. For
2-D / multi-asset output (e.g. Heston returning `[Array1<T>; 2]` for
price + vol), you set `type Output = [Array1<T>; 2]` and provide the
joint sampler in `sample()`.

## 2. The struct + constructor parity

Every diffusion struct ships **both** an unseeded and a seeded
constructor with the same parameter list, plus an explicit `t` total
horizon:

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

impl<T: FloatExt> Foo<T> {
    #[must_use]
    pub fn new(theta: T, mu: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>) -> Self {
        assert!(n >= 2, "n must be at least 2");
        Self {
            theta, mu, sigma, n, x0, t, seed: Unseeded,
        }
    }
}

impl<T: FloatExt> Foo<T, Deterministic> {
    #[must_use]
    pub fn seeded(
        theta: T, mu: T, sigma: T, n: usize, x0: Option<T>, t: Option<T>, seed: u64,
    ) -> Self {
        assert!(n >= 2, "n must be at least 2");
        Self {
            theta, mu, sigma, n, x0, t, seed: Deterministic::new(seed),
        }
    }
}
```

The phantom `seed: S` field is the compile-time switch between the two.
Both constructors validate `n >= 2` upfront — the rc.2 Fukasawa-Hurst
fix taught us to thread seeded constructors through every test that
samples a process.

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

```rust
impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Foo<T, S> {
    type Output = Array1<T>;

    fn sample(&self) -> Self::Output {
        let t = self.t.unwrap_or(T::one());
        let dt = t / T::from_usize_(self.n - 1);
        let mut rng = self.seed.derive();   // <-- rng from the seed strategy

        let mut path = Array1::<T>::zeros(self.n);
        path[0] = self.x0.unwrap_or(T::zero());

        for i in 1..self.n {
            let z = rng.sample::<f64, StandardNormal>();
            let z = T::from_f64_fast(z);
            let drift = self.theta * (self.mu - path[i - 1]);
            let diffusion = self.sigma;
            path[i] = path[i - 1] + drift * dt + diffusion * dt.sqrt() * z;
        }
        path
    }
}
```

Use `T::from_f64_fast` (not `T::from`) for compile-time-known constant
conversions — it's the workspace convention for the `FloatExt`
boundary.

For higher-order schemes (Milstein, SRK2, SRK4), see `crate::sde::*`
helpers — there's a generic `milstein_step`, `srk2_step` etc. that take
drift/diffusion closures.

## 5. Python wrapper macro

After the inherent + ProcessExt impls, append the Python wrapper macro
**at the bottom of the source file**:

```rust
py_process_1d!(PyFoo, Foo,
    sig: (theta, mu, sigma, n, x0 = 0.0, t = 1.0, m = None, seed = None),
    params: (theta: f64, mu: f64, sigma: f64, n: usize, x0: f64, t: f64, m: Option<usize>),
);
```

The macro generates:
- `#[pyclass(unsendable)]` `PyFoo` with `__new__` accepting the listed
  signature
- `sample(n)` returning a numpy array (via the `IntoF64` shim from
  `stochastic-rs-core::python`)
- `sample_par(m, n)` returning a 2-D numpy array (parallel paths)

For 2-D output (e.g. Heston), use `py_process_2x1d!` instead. For
multi-asset correlated (returning `Array2<T>`), use `py_process_2d!`.

After the macro, remember to **register the class** in
`stochastic-rs-py/src/lib.rs`:

```rust
use stochastic_rs_stochastic::diffusion::foo::PyFoo;
// ...
m.add_class::<PyFoo>()?;
```

## 6. Backward-compat aliases

If you rename an existing process (e.g. for the mean-reversion-speed
convention fix), add an alias in
`stochastic-rs-stochastic/src/diffusion/aliases.rs`:

```rust
#[deprecated(since = "X.Y.Z", note = "renamed to Foo; use Foo::new instead")]
pub use super::foo::Foo as OldFoo;
```

The alias keeps the old name compiling for one release cycle. Drop
it in the next major.

## 7. Testing requirements

A new diffusion ships with at least four tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// 1. Seeded determinism.
    #[test]
    fn seeded_is_deterministic() {
        let p1 = Foo::seeded(0.5, 0.0, 0.1, 100, None, None, 42).sample();
        let p2 = Foo::seeded(0.5, 0.0, 0.1, 100, None, None, 42).sample();
        for i in 0..100 {
            assert_eq!(p1[i], p2[i]);
        }
    }

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
        let _ = Foo::<f64>::new(0.5, 0.0, 0.1, 1, None, None);
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
new processes (e.g. "120+ processes, incl. interest::lmm::Lmm"). Update
that line if your new process is material.

## 9. Anti-patterns

- **Do not** call `thread_rng()` directly in `sample()`. Always go
  through `self.seed.derive()` so `Foo::seeded(...)` produces
  reproducible paths.
- **Do not** name fields `kappa` / `theta` (Brigo convention). The
  workspace uses `theta` / `mu`. Sticking to local conventions when the
  surrounding code uses ours produces silent numeric bugs (rc.0 CIR).
- **Do not** add a process without a `seeded` constructor. The user
  will need it for testing eventually; adding it later breaks the API.
- **Do not** put validation behind `debug_assert!`. `assert!(n >= 2)`
  is a permanent invariant; debug_assert hides it from release builds
  and lets users hit cryptic out-of-bounds panics in `path[0]`.

## 10. Reference impls (in increasing complexity)

- `Bm` (`process/bm.rs`) — single-line Brownian motion, no parameters
  besides `n`.
- `Gbm` (`diffusion/gbm.rs`) — geometric BM, two parameters.
- `Vasicek` (`diffusion/vasicek.rs`) — mean-reverting OU; the `theta`
  / `mu` reference.
- `Cir` (`diffusion/cir.rs`) — CIR with reflection at 0; rc.0 fixed
  the field-naming convention.
- `Fou` (`diffusion/fou.rs`) — fractional OU; goes through
  `add-fractional-process` once you wrap `Fgn`.
- `Heston` (`volatility/heston.rs`) — 2-D output (`[price, vol]`); uses
  `py_process_2x1d!`.

## Related SKILLs

- `add-fractional-process` — for Hurst-parameterised processes wrapping
  `Fgn` or extending `MarkovLift`.
- `add-jump-process` — for compound-Poisson / Lévy-driven additions.
- `python-bindings` — invoked by `py_process_*!` and the registration
  step.
- `feature-flag-management` — if your process needs an optional GPU
  backend or LAPACK helper.

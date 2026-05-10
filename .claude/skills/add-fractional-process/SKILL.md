---
name: add-fractional-process
description: How to add a Hurst-parameterised (rough / fractional) process to stochastic-rs-stochastic. Invoke when wrapping fBm / fGn / Volterra-kernel processes — fOU, rBergomi, rough Heston, fBates, fractional CIR, etc.
---

# Add fractional process — stochastic-rs-stochastic

Fractional / rough processes carry a Hurst parameter `H ∈ (0, 1)` and
inherit non-Markovian memory from a fractional Gaussian noise (fGn)
driver. Three implementation paths exist in this codebase, depending
on the structure of the SDE you are simulating:

1. **Wrap `Fgn`** — the simplest path for processes that read like
   `dX = drift dt + diffusion dB^H_t`. Used by `Fou`, `Fbm`, `Fbates`.
2. **Extend `MarkovLift`** — for affine rough processes where the
   Volterra kernel admits a finite-dimensional Markovian lift
   (Abi Jaber 2018). Used internally by `rough_heston`,
   `rough_bergomi`'s second-order moment expansion.
3. **Add a new Volterra kernel** — when neither (1) nor (2) suits. The
   `volterra::kernel::*` module has Riemann-Liouville, Mittag-Leffler,
   Laguerre Padé approximants. Adding a kernel is heavy lifting.

Choose path (1) by default; (2) only when the Markovian lift
substantially improves performance (e.g. avoiding O(n²) Volterra
quadrature on long horizons); (3) only when adding a new kernel family
is genuinely required.

## 1. Path (1) — wrap `Fgn`

The `Fgn` (fractional Gaussian noise) sampler is the workspace's
canonical fGn provider. It supports CPU SIMD, optional GPU (CUDA /
Metal) for long horizons, and seeded variants. Use it via
composition:

```rust
// stochastic-rs-stochastic/src/diffusion/fou.rs (existing reference)

use crate::noise::fgn::Fgn;

pub struct Fou<T: FloatExt, S: SeedExt = Unseeded> {
    pub hurst: T,
    pub theta: T,
    pub mu: T,
    pub sigma: T,
    pub n: usize,
    pub x0: Option<T>,
    pub t: Option<T>,
    pub seed: S,
    fgn: Fgn<T>,                 // <-- composed driver
}

impl<T: FloatExt> Fou<T> {
    pub fn new(hurst: T, theta: T, mu: T, sigma: T, n: usize,
               x0: Option<T>, t: Option<T>) -> Self {
        Self {
            hurst, theta, mu, sigma, n, x0, t,
            seed: Unseeded,
            fgn: Fgn::new(hurst, n - 1, t),    // n - 1 increments for n points
        }
    }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Fou<T, S> {
    type Output = Array1<T>;
    fn sample(&self) -> Self::Output {
        let dt = self.fgn.dt();
        let fgn = self.fgn.sample_cpu_impl(&self.seed.derive());  // delegate seed
        let mut path = Array1::<T>::zeros(self.n);
        path[0] = self.x0.unwrap_or(T::zero());
        for i in 1..self.n {
            path[i] = path[i-1]
                + self.theta * (self.mu - path[i-1]) * dt
                + self.sigma * fgn[i-1];
        }
        path
    }
}
```

Three things matter:

- **`Fgn::new(hurst, n - 1, t)`** — the `n - 1` is the number of
  *increments*. The path has `n` points and `n - 1` Euler steps.
- **`self.fgn.sample_cpu_impl(&self.seed.derive())`** — propagate the
  seed through the Fgn driver. The CPU SIMD path handles the seed
  parity. For GPU dispatch, use `sample_gpu_impl` / `sample_metal_impl`
  behind feature gates.
- **Do NOT use `t.powf(H - 0.5)` shortcuts** for non-Riemann-Liouville
  paths. The §5.10 trap (rc.0 rough-Heston) involved a `sum_z *
  t.powf(H - 0.5)` shortcut that is exact only for the
  Riemann-Liouville kernel and silently mis-scales the moments under
  any other kernel. Always derive the convolution explicitly.

## 2. Path (2) — extend `MarkovLift`

For affine processes (e.g. fractional CIR, rough Heston) where the
fractional kernel `K(t)` admits an exponential-sum approximation
`K(t) ≈ Σ c_i exp(-λ_i t)`, the Markovian lift collapses the O(n²)
Volterra integral to O(n × m) where `m` is the number of exponential
terms. The lift framework lives in
`stochastic-rs-stochastic/src/rough/markov_lift.rs`:

```rust
use crate::rough::markov_lift::{MarkovLift, LiftSpec};

pub struct FooLift<T: FloatExt, S: SeedExt = Unseeded> {
    inner: MarkovLift<T>,
    seed: S,
}

impl<T: FloatExt> FooLift<T> {
    pub fn new(hurst: T, ...) -> Self {
        let lift_spec = LiftSpec::laguerre_pade(hurst, /* num_terms */ 8);
        // ...
    }
}
```

Reference: `rough/rl_fou.rs` — Riemann-Liouville fOU via lift,
matches Markov lift theory of Abi Jaber (2018).

The lift framework handles its own seeded RNG; if your process needs
a *driver* seed plus a *kernel* seed, expose both.

## 3. Path (3) — add a new Volterra kernel

The kernel module `rough/kernel.rs` defines:

```rust
pub trait VolterraKernel<T> {
    fn evaluate(&self, t: T) -> T;
    fn lift_spec(&self) -> LiftSpec<T>;
    fn closed_form_variance(&self, t: T) -> T;
}
```

If you add a kernel (e.g. truncated Mittag-Leffler, Hermite-class), you
must:

- Implement `evaluate`, `lift_spec`, `closed_form_variance`.
- Add a paper reference (Bilokon-Wong 2026, Ma-Wu 2021, Abi Jaber 2018
  are the load-bearing references for the in-tree kernels).
- Add a `kernel_xxx_matches_paper` test that compares to a
  Python/Mathematica reference at fixed Hurst values.
- Reject `H ≥ 0.5` *at construction* (rough is `H < 0.5`); the
  existing `rejects_h_above_half` / `rejects_h_at_half` tests show
  the panic-pattern.

## 4. Required references (cite in the source-file doc comment)

For every fractional process, the source file's `//!` header **must**
cite the paper that defines the kernel and the simulation scheme:

- Abi Jaber, E. (2018), "Lifting the Heston model", *Quantitative Finance*.
- Ma, J. & Wu, J. (2021), "Multifactor approximation of rough volatility models", *Journal of Computational Finance*.
- Bilokon & Wong (2026), "Hermite class approximations for rough volatility kernels".
- Bayer, Friz, Gatheral (2016), "Pricing under rough volatility", *Quantitative Finance* — for rBergomi specifically.
- Mandelbrot & Van Ness (1968), "Fractional Brownian motions, fractional noises and applications" — origin of fBm.

The dev-rules feedback memo on "Follow papers EXACTLY, don't simplify
formulas" is acutely relevant here: rough-vol literature is full of
near-identical-looking formulas that differ in third-order constants,
and shortcuts get caught by the regression tests downstream.

## 5. Testing requirements

Specific to fractional processes:

```rust
#[cfg(test)]
mod tests {
    /// 1. H = 0.5 reduces to standard Brownian motion.
    #[test]
    fn h_half_matches_bm() { ... }

    /// 2. Variance scaling: Var(X_t - X_0) ∝ t^{2H} (theory test).
    #[test]
    fn variance_scaling_matches_2h() { ... }

    /// 3. Long-memory: lag-k autocovariance has the right sign
    ///    (positive for H > 0.5, negative for H < 0.5).
    #[test]
    fn fgn_lag1_correlation_sign_matches_hurst_regime() { ... }

    /// 4. Seeded determinism — non-negotiable.
    #[test]
    fn seeded_is_deterministic() { ... }
}
```

## 6. Anti-patterns

- **Do not** use `sum_z * t.powf(H - 0.5)` to scale a fGn-driven
  process. That shortcut is exact only for Riemann-Liouville fBm; for
  any other kernel the moments are wrong by a Hurst-dependent factor.
- **Do not** add a new kernel without a closed-form-variance reference
  test. Numerical kernels without an analytic check are bug magnets.
- **Do not** reuse `Fgn` instances across threads. `Fgn::sample_*_impl`
  takes a `&Seed`, not a `&self`-bound RNG; share the *seed*, not the
  *Fgn struct*.
- **Do not** skip the H-validation panics. `rejects_h_above_half` /
  `rejects_h_at_half` tests are the documentation that you can't
  silently accept H = 0.5 (degenerate case) or H > 0.5 (smooth, not
  rough).

## 7. Reference impls (in increasing complexity)

- `Fgn` (`noise/fgn/core.rs`) — the canonical fGn driver; not a
  process, but the building block for all fractional processes.
- `Fbm` (`process/fbm.rs`) — fractional Brownian motion; thinnest
  wrapper around `Fgn` (cumulative sum).
- `Fou` (`diffusion/fou.rs`) — fractional OU; reference for path (1).
- `RlFbm` (`rough/rl_fbm.rs`) — Riemann-Liouville fBm; reference for
  path (2) via `MarkovLift`.
- `Fbates` (`volatility/fbates_svj.rs`) — fractional Bates with jumps;
  composition pattern with `CompoundPoisson`.
- `RoughBergomi` — accessed via `rl_*` family + a price-path layer.

## Related SKILLs

- `add-diffusion-process` — for the non-fractional `theta`/`mu` /
  seeded constructor / py macro contract.
- `add-jump-process` — when you want to add jumps on top of a
  fractional driver (Bates → fBates).
- `python-bindings` — `py_process_*!` invocation works the same for
  fractional processes.

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
   `dX = drift dt + diffusion dB^H_t`. Used by `Fou`, `Fbm`, `Fgbm`,
   `FJacobi`, `Cfou`, `FBatesSvj`, `JumpFou`.
2. **Drive `MarkovLift`** — for rough processes whose Riemann-Liouville
   kernel is approximated by a finite sum of exponentials (Abi Jaber
   2018 / Bilokon-Wong 2026). Used by `RlFOU`, `RlFBm`, `RlBs`,
   `RlHeston`.
3. **Add a new Volterra kernel** — when neither (1) nor (2) suits. The
   `volterra` module has `ExponentialKernel`, `GammaKernel`,
   `SumOfExponentials`; `rough/kernel.rs` has `RlKernel`
   (Riemann-Liouville). Adding a kernel is heavy lifting.

Choose path (1) by default; (2) only when the Markovian lift
substantially improves performance (e.g. avoiding O(n²) Volterra
quadrature on long horizons); (3) only when adding a new kernel family
is genuinely required.

## 1. Path (1) — wrap `Fgn`

`Fgn` (`noise/fgn/core.rs`) is the workspace's canonical fGn provider.
Compose it as a private field, and carry a **third type parameter `B`**
for the sampling backend — that is what gives your process CUDA / Metal
/ Accelerate for free (see `add-gpu-sampler`).

```rust
// stochastic-rs-stochastic/src/diffusion/fou.rs (existing reference)

use crate::device::Backend;
use crate::device::Cpu;
use crate::noise::fgn::Fgn;

pub struct Fou<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
    pub hurst: T,
    pub theta: T,
    pub mu: T,
    pub sigma: T,
    pub n: usize,
    pub x0: Option<T>,
    pub t: Option<T>,
    pub seed: S,
    fgn: Fgn<T, Unseeded, B>,       // <- composed driver, always Unseeded
}

impl<T: FloatExt, S: SeedExt, B> Fou<T, S, B> {
    pub fn new(hurst: T, theta: T, mu: T, sigma: T, n: usize,
               x0: Option<T>, t: Option<T>, seed: S) -> Self {   // seed LAST
        assert!(n >= 2, "n must be at least 2");
        Self {
            hurst, theta, mu, sigma, n, x0, t, seed,
            fgn: Fgn::new(hurst, n - 1, t, Unseeded),
        }
    }
}

impl<T: FloatExt, S: SeedExt, B: Backend> ProcessExt<T> for Fou<T, S, B> {
    type Output = Array1<T>;
    type Sampler<'s> = FouSampler<'s, T, S, B> where Self: 's;

    fn sampler(&self) -> FouSampler<'_, T, S, B> {
        FouSampler { fou: self, seed: self.seed.derive() }
    }
}

impl<T: FloatExt, S: SeedExt, B: Backend> FouSampler<'_, T, S, B> {
    fn fill_path(&mut self, out: &mut [T]) {
        if out.is_empty() { return; }
        let p = self.fou;
        let dt = p.fgn.dt();
        let fgn = p.fgn.noise(&self.seed);        // <- backend-dispatched
        out[0] = p.x0.unwrap_or(T::zero());
        let mut prev = out[0];
        for (dst, inc) in out[1..].iter_mut().zip(fgn.iter()) {
            let next = prev + p.theta * (p.mu - prev) * dt + p.sigma * *inc;
            *dst = next;
            prev = next;
        }
    }
}
```

Five things matter:

- **`Fgn::new(hurst, n - 1, t, Unseeded)`** — four arguments. The
  `n - 1` is the number of *increments*: the path has `n` points and
  `n - 1` Euler steps.
- **The embedded `Fgn` is always `Unseeded`.** The wrapper owns the real
  seed and passes it *into* the noise call. That is what lets one seed
  drive both the process and its driver coherently.
- **`p.fgn.noise(&self.seed)`** is the seeded fGn draw —
  `pub(crate)`, and it dispatches to `B::generate`. Do **not** call
  `sample_cpu_impl` (it is `pub(crate)` and now has no production
  caller — `#[cfg(test)]`-only convenience), and there is no
  `sample_gpu_impl` / `sample_metal_impl` to call by hand: those are
  reached through the `Backend` marker, not through `#[cfg]` branches.
  For `m` paths at once there is `noise_batch(m, &seed)`.
- **Seed via `derive()` in `sampler()`**, never `clone()` — the
  `ProcessExt` reproducibility requirement. See `add-diffusion-process` §1.
- **Do NOT use `t.powf(H - 0.5)` shortcuts** for non-Riemann-Liouville
  paths. The §5.10 trap (rc.0 rough-Heston) involved a `sum_z *
  t.powf(H - 0.5)` shortcut that is exact only for the
  Riemann-Liouville kernel and silently mis-scales the moments under
  any other kernel. Always derive the convolution explicitly.

Add `.on::<B2>()` with the `backend_switch!` macro (`macros.rs`) rather
than hand-writing it — there is a form for a real `fgn` field and one
for `PhantomData`.

## 2. Path (2) — drive `MarkovLift`

Where the fractional kernel `K(t)` admits an exponential-sum
approximation `K(t) ≈ Σ w_l exp(-x_l t)`, the Markovian lift collapses
the O(n²) Volterra integral to O(n × N') where `N'` is the number of
exponential terms. The stepper lives in
`stochastic-rs-stochastic/src/rough/markov_lift/` (a **directory** —
`mod.rs`, `stepper.rs`, `simd.rs`, `tests.rs`), and its surface is
small:

```rust
use crate::rough::kernel::RlKernel;
use crate::rough::markov_lift::MarkovLift;

impl<T: FloatExt> MarkovLift<T> {
    pub fn new(kernel: RlKernel<T>, dt: T) -> Self;

    /// One path. `f` is the drift closure, `g` the diffusion closure,
    /// `dw` the pre-drawn Brownian increments.
    pub fn simulate<F, G>(&self, x0: T, f: F, g: G, dw: &[T]) -> Array1<T>;
    pub fn simulate_batch<F, G>(&self, x0: T, f: F, g: G, dw: ArrayView2<T>) -> Array2<T>;
    pub fn simulate_batch_par<F, G>(&self, x0: T, f: F, g: G, dw: ArrayView2<T>) -> Array2<T>;
}
```

There is **no `LiftSpec` type** and no `LiftSpec::laguerre_pade`
constructor — the quadrature is built by `RlKernel::new(hurst, degree)`,
which computes generalised Gauss-Laguerre nodes and weights directly.
`MarkovLift` takes that kernel plus `dt`.

Note the division of labour: `MarkovLift` does **not** own an RNG. It
consumes Brownian increments you supply as `dw`, so your process draws
them (from its own seed, in `sampler()`) and hands them over. Wrap it
the same way path (1) wraps `Fgn`.

References: `rough/rl_fou.rs` (`RlFOU`), `rough/rl_fbm.rs` (`RlFBm`),
`rough/rl_bs.rs`, `rough/rl_heston.rs`.

## 3. Path (3) — add a new Volterra kernel

The trait is `VolterraKernel<T>`, in
`stochastic-rs-stochastic/src/volterra/kernel.rs` (not `rough/kernel.rs`,
which is where the Riemann-Liouville *implementation* `RlKernel` lives):

```rust
pub trait VolterraKernel<T: FloatExt>: Clone {
    fn nodes(&self) -> &Array1<T>;            // quadrature nodes x_l
    fn weights(&self) -> &Array1<T>;          // scaled weights w_l
    fn degree(&self) -> usize { self.nodes().len() }   // defaulted
    fn evaluate(&self, t: T) -> T;            // K(t)
    fn integral_from_zero(&self, dt: T) -> T; // ∫₀^dt K(u) du
}
```

Four required methods; `degree` is defaulted. There is no `lift_spec`
and no `closed_form_variance`.

**The load-bearing invariant**, stated on `weights()`: `Σ w_l e^{-x_l t}
≈ K(t)` must hold using *these* `w_l` and `x_l`, where `K` is exactly
what `evaluate` returns, and `integral_from_zero` must be that same
`K`'s integral. Any normalising constant (e.g. Riemann-Liouville's
`1/Γ(H+1/2)`) is **already folded into** `weights`, `evaluate` and
`integral_from_zero` — a kernel-generic caller must not apply another
one on top. `MarkovLift` deliberately does the opposite: it reads
`RlKernel`'s *inherent*, un-normalised `weights`/`evaluate` and applies
the factor once, outside the sum. That split is specific to
`MarkovLift`'s hand-written loop; do not copy it into anything built on
the trait.

Existing implementors: `ExponentialKernel`, `GammaKernel`,
`SumOfExponentials` (`volterra/kernel.rs`), `RlKernel`
(`rough/kernel.rs`).

If you add a kernel you must:

- Implement all four required methods, honouring the invariant above.
- Add a paper reference (Bilokon-Wong 2026, Ma-Wu 2021, Abi Jaber 2018
  are the load-bearing references for the in-tree kernels).
- Add a test comparing to an external reference at fixed Hurst values;
  `volterra/kernel_tests.rs` and `volterra/reference.rs` are where that
  goes.
- Validate the Hurst domain *at construction*. Note that `Fgn::new`
  itself asserts only `hurst ∈ [0, 1]` — a kernel that is meaningful
  only for rough `H < 0.5` must say so in its own constructor.

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

    /// 4. Seeded determinism — non-negotiable, and `Cpu`-backend only.
    #[test]
    fn seeded_is_deterministic() {
        let a = Foo::<f64, _>::new(0.3, /* … */, Deterministic::new(42));
        let b = a.clone();          // Clone snapshots the seed
        assert_eq!(a.sample(), b.sample());
    }

    /// 5. `sample_par(m)` bit-identical across rayon thread-pool sizes,
    ///    and registration in
    ///    `tests/reproducibility_all_processes.rs`.
    #[test]
    fn sample_par_is_thread_count_stable() { }
}
```

**Backend caveat on tests 4 and 5.** Those guarantees hold on `Cpu`
only. `Accelerate` is bit-stable in seed *consumption* but not in
vDSP's arithmetic; `Cuda` / `Metal` / `CubeclCuda` / `CubeclWgpu` are
reproducible-effort-only, and `Fbm` on those three is not even a
function of the pinned seed. Pin `B = Cpu` in reproducibility tests and
test the other backends distributionally. See `add-gpu-sampler` §2 and
`Backend`'s own reproducibility table.

## 6. Anti-patterns

- **Do not** use `sum_z * t.powf(H - 0.5)` to scale a fGn-driven
  process. That shortcut is exact only for Riemann-Liouville fBm; for
  any other kernel the moments are wrong by a Hurst-dependent factor.
- **Do not** add a new kernel without an external reference test.
  Numerical kernels without an analytic check are bug magnets.
- **Do not** share an `Fgn` *instance* to get independent streams.
  `Fgn::noise` / `noise_batch` take the seed as a `&S2` argument, so the
  embedded `Fgn` stays `Unseeded` and the wrapper's seed is what varies
  — share the *seed*, not the *Fgn struct*.
- **Do not** seed the embedded `Fgn` at construction. `Fgn::new(hurst,
  n - 1, t, Unseeded)` is the shape; a seeded inner `Fgn` would be
  consulted by nothing on the sampling path and silently mislead.
- **Do not** skip the H-validation panics on a rough-only kernel.
  `RlKernel`'s `rejects_h_at_half` / `rejects_h_above_half`
  (`rough/kernel.rs`) are the documentation that you cannot silently
  accept H = 0.5 (degenerate) or H > 0.5 (smooth, not rough). Note
  `Fgn` itself accepts all of `[0, 1]` — the narrower check is the
  kernel's job, not the noise driver's.
- **Do not** `#[cfg]`-branch to reach a GPU fGn path. Carry the `B`
  parameter and let `Backend` dispatch.

## 7. Reference impls (in increasing complexity)

- `Fgn` (`noise/fgn/core.rs`) — the canonical fGn driver; not a
  process, but the building block for all fractional processes.
- `Fbm` (`process/fbm.rs`) — fractional Brownian motion; thinnest
  wrapper around `Fgn` (cumulative sum).
- `Fou` (`diffusion/fou.rs`) — fractional OU; reference for path (1).
- `Fgbm`, `FJacobi`, `Cfou` (`diffusion/`) — the other path-(1)
  wrappers; all three carry `B` and follow `Fou` line for line.
- `RlFBm` (`rough/rl_fbm.rs`) and `RlFOU` (`rough/rl_fou.rs`) —
  Riemann-Liouville via `MarkovLift`; the path-(2) references. Note the
  capitalisation (`RlFBm`, `RlFOU`), which differs from the `Fbm` /
  `Fou` spelling of the path-(1) types.
- `RlBs` (`rough/rl_bs.rs`), `RlHeston` (`rough/rl_heston.rs`) — the
  remaining lift-driven models.
- `FBatesSvj` (`volatility/fbates_svj.rs`) — fractional Bates with
  jumps; composition with `CompoundPoisson`. See `add-jump-process`.
- `RoughBergomi` (`volatility/rbergomi.rs`) — a standalone type, **not**
  assembled from the `rl_*` family. `S`-parameterised, no `B`.

## Related SKILLs

- `add-diffusion-process` — the `ProcessExt` / `sampler()` contract,
  the `theta`/`mu` convention, and the py macro. Read it first.
- `add-gpu-sampler` — the `B` backend parameter these processes carry,
  and what each backend does and does not guarantee.
- `add-jump-process` — when you want to add jumps on top of a
  fractional driver (Bates → fBates).
- `python-bindings` — `py_process_*!` invocation works the same for
  fractional processes.

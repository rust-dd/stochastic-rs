---
name: add-gpu-sampler
description: How to add or extend a GPU / accelerated sampling backend (CUDA, Metal, wgpu, Accelerate) in stochastic-rs-stochastic. Invoke when porting fGN or the Euler engine to a new device, or when a backend-generic process needs to reach one.
---

# Add GPU sampler — stochastic-rs-stochastic

Acceleration in this workspace is **not** a `#[cfg]` fork inside
`sample()`. It is a compile-time type parameter carrying a runtime
handle. A process carries a backend parameter `B` (defaulting to `Cpu`)
in a plain `pub backend: B` field, the capability traits monomorphise
sampling to that handle with **no runtime branch**, and the caller
switches with a turbofish or an explicit handle:

```rust
let fbm = Fbm::<f64, _>::new(0.7, 1024, None, Deterministic::new(42));
let a = fbm.clone().on::<Cuda>().sample_par(256);              // B::default()
let b = fbm.on_device(Cuda::new(1).with_batch_budget(1 << 28)); // explicit
```

A handle only exists when its feature is compiled, and it implements a
capability only for the precisions its kernels compute in, so selecting
an unavailable backend — or `f64` on a `f32`-only device — is a
**compile error**, not a silent fallback. Read
`stochastic-rs-stochastic/src/device.rs` in full before touching any of
this: it is 742 lines, current, and it is the contract. `euler.rs` (510
lines) is the second half.

## 1. The five backends and their real feature names

| Handle | Feature | What it is | Precisions |
|---|---|---|---|
| `Cpu` | *(none — always available)* | Default `B` for every process | `f32` / `f64` |
| `Accelerate` | `accelerate` | Apple vDSP / AMX — a **CPU** path, not a GPU | `f32` / `f64` |
| `Cuda` | `cuda` | `cudarc` + cuFFT + NVRTC, fused Philox kernel | `f32` / `f64` |
| `Metal` | `metal` | Hand-written MSL via the `metal` crate | `f32` |
| `CubeclCuda` | `cubecl-cuda` | cubecl Rust kernels on CubeCL's CUDA runtime | `f32` |
| `CubeclWgpu` | `cubecl-wgpu` | cubecl Rust kernels through wgpu: Metal, Vulkan, WebGPU | `f32` |

The native backends take the bare names (`cuda`, `metal`) and CubeCL's
runtimes are namespaced under it; the `gpu` / `gpu-cuda` / `gpu-wgpu`
aliases and the `cuda-native` spelling were **removed before 3.0**. There
is no `metal-rs` dependency. The CUDA backend is `cudarc`, not
`cust`. Kernels are Rust string constants compiled at runtime by NVRTC
(`noise/fgn/cuda/kernels.rs` for fGN, `euler/kernel.rs` for the
Euler engine) — there are **no `.cu` files** anywhere in the repo.

Renaming a feature silently turns every `cfg(feature = "old")` false and
compiles the code out without a warning. Sweep the cfgs in the same
commit as the rename.

## 2. A handle is a value, not a marker

`Cpu` and `Accelerate` are unit structs. The four device handles carry
their settings:

```rust
pub struct Cuda { pub ordinal: usize, pub batch_budget: usize }
```

`Default` reads `STOCHASTIC_RS_DEVICE` (else `0`) and
`STOCHASTIC_RS_DEVICE_BATCH_BYTES` (else `DEFAULT_BATCH_BUDGET_BYTES`,
1 GiB); `new(ordinal)` and `with_batch_budget(bytes)` build one
explicitly. There is **no process-wide device state** — two processes on
two GPUs are two handles — so never reintroduce a global selector or a
global budget.

`Backend: Copy + Default + Send + Sync` requires exactly one method,
`fn probe(&self) -> Result<DeviceInfo, DeviceError>`, which opens the
device and reports `{ backend, name, precisions, ordinal }` or says why
it cannot be used. `DeviceError` has three kinds: `Unavailable`,
`Compile`, `Launch`.

## 3. Reproducibility: read this before you write a test

It is **not** true that "a GPU sampler must match the CPU sampler
bit-for-bit". The shipped contract (`FgnBackend`'s own trait doc, which
is normative) is:

| Backend | `sample` / `sample_par` reproducible? |
|---|---|
| `Cpu` | **Yes.** Same seed + same `m` ⇒ bit-identical, on any machine, under any rayon thread-pool size. |
| `Accelerate` | **No — measured, not assumed.** Seed *consumption* is thread-count independent, but `vDSP_fft_zip`'s arithmetic is not bit-stable across calls: 400 repeated calls on an idle M4 Max diverged 0 times; the same sweep under core saturation diverged 21/400, worst relative difference `2.08e-3`. `Cpu` stayed bit-exact under identical load. |
| `Cuda` / `Metal` / `CubeclCuda` / `CubeclWgpu` | **Function of the pinned seed, not bit-identical to `Cpu`.** Each batch draws **one** launch seed from the `seed: &S2` the *caller* passed, so the same `Deterministic` seed value gives the same paths and consecutive calls advance the stream, as on the host. Bit-identity across driver versions, vendors or repeated runs is untested and unpromised. |

Two rules follow, and both have already been violated once:

- **The launch seed comes from the caller's seed source, never from
  `fgn.seed`.** A wrapper such as `Fbm` or `Fou` keeps a permanently
  `Unseeded` embedded `fgn` and passes its *own* seed down; reading
  `fgn.seed` inside the sampler makes every such wrapper seed-blind on
  devices. That was a real bug (fixed in `b3cbb26`).
- **Draw that seed once per batch, then offset per chunk.** A
  `Deterministic` source advances on every draw, so drawing inside the
  chunk loop shifts the stream and breaks the chunk-identity guarantee.
  The kernels hash the *global* path index (`first_path + path`), which
  is what makes a chunked batch bit-identical to one launch — assert it.

So: **do not write a `cpu_and_gpu_match_bit_for_bit` test.** What you
test instead:

- **Chunk identity**: a batch that spans several launches equals one
  launch, path for path (`with_batch_budget` forces the split).
- **Seed identity**: same seed twice ⇒ identical paths; different seed ⇒
  different paths.
- **Distributional agreement**: `Var(X_t) ∝ t^{2H}`, lag-1
  autocovariance sign, mean ≈ 0 — the properties
  `add-fractional-process` §5 lists, on the new backend.
- **Shape and finiteness** of the returned `m × n` matrix.
- **`probe()` reports the device**, and a missing device is an `Err`
  with a usable message rather than a panic.
- For a **CPU-side** backend like `Accelerate`, thread-count-independent
  seed consumption is testable and is tested —
  `tests/deterministic_parallelism_accelerate.rs`.

## 4. The three capability traits

`Backend` says nothing about sampling. A handle opts into what it can do:

- **`HostBackend`** — no device kernel; the process's own sampler runs.
  `Cpu` and `Accelerate` implement it. A process bounded on this trait
  gains a device later by *widening* the bound, which breaks no caller.
- **`FgnBackend<T>`** — circulant-embedding fractional Gaussian noise.
  Required: `try_generate`, `try_generate_batch`. Provided on top:
  `generate`, `generate_batch`, `try_generate_pair`, `generate_pair`
  (the panicking ones call `device_panic`). All take `&self`.
- **`EulerBackend<T>`** — Euler–Maruyama paths for `Gbm` / `Ou` / `Cir`.
  Required: `try_sample`, `try_euler_paths`, `try_euler_paths_map`,
  `try_euler_matrix`; provided: the three panicking twins.

A device does **not** implement `EulerBackend` by hand. It implements
the primitive:

```rust
pub trait EulerKernel<T: FloatExt>: Backend {
    /// Paths `first .. first + m` of the launch stream seeded by `seed`.
    fn euler_kernel<P: EulerCoefficients<T>>(
        &self, process: &P, first: usize, m: usize, seed: u64,
    ) -> Result<Array2<T>, DeviceError>;

    fn batch_budget(&self) -> usize;

    /// The whole batch, chunked to the budget. Override to pipeline;
    /// the result must stay bit-identical.
    fn euler_kernel_batch<P: EulerCoefficients<T>>(…) -> Result<Array2<T>, DeviceError> { … }
}
```

and `kernel_euler_backend!` derives `EulerBackend<T>` from it (seed
drawn once, chunking, per-chunk parallel map). The host handles get
theirs from `host_euler_backend!`, which runs the process's own sampler.

**Why a macro and not a blanket impl:** `impl<K: EulerKernel<T>>
EulerBackend<T> for K` beside `impl EulerBackend<T> for Cpu` is E0119 —
coherence cannot know `Cpu` will never implement `EulerKernel`. One
macro line per handle instead.

## 5. Adding an fGN backend

Device-side fGN backends are registered by `gpu_backend!` in
`device.rs`; the trailing scalars are the precisions the device serves:

```rust
gpu_backend!("cuda", Cuda  => sample_cuda_impl, f32, f64);
gpu_backend!("cubecl-cuda", CubeclCuda  => sample_cubecl_cuda_impl, f32);
gpu_backend!("cubecl-wgpu", CubeclWgpu  => sample_cubecl_wgpu_impl, f32);
gpu_backend!("metal",       Metal => sample_metal_impl,       f32);
```

The macro generates one feature-gated `impl FgnBackend<$scalar>` per
scalar, both methods delegating to one inherent method on `Fgn`:

```rust
pub(crate) fn sample_<name>_impl<S2: SeedExt>(
    &self, m: usize, seed_src: &S2, device: &<Handle>,
) -> Result<Array2<T>>
```

The work:

1. Declare the handle in `device.rs`, feature-gated, `#[derive(Clone,
   Copy, Debug, PartialEq, Eq)]` with a hand-written `Default` reading
   the two env vars, plus `new` / `with_batch_budget`.
2. Implement `Backend::probe` for it.
3. Write the sampler in a new `noise/fgn/<name>.rs`, gated on the
   feature: draw the launch seed **once** from `seed_src`, chunk with
   `chunk_rows(device.batch_budget, elems_per_row, elem_size)`, offset
   each chunk's seed by its element offset, and open the device at
   `device.ordinal`.
4. Add one `gpu_backend!` line.
5. Add the feature to `stochastic-rs-stochastic/Cargo.toml` **and**
   propagate it from the umbrella root — see `feature-flag-management`.
6. Extend the reproducibility table in `FgnBackend`'s trait doc, the
   support matrix in `website/content/docs/concepts/gpu-support.mdx`,
   and the backend table in `concepts/backends.mdx`. All three are
   normative; a backend absent from them is undocumented.

A host-side backend (the `Accelerate` shape) does **not** use the macro:
write the `impl FgnBackend<T>` by hand so you can thread `seed` through
`chunk_count` / `chunk_lens` and derive one basis per chunk sequentially
before `into_par_iter()`. Copy `Accelerate`'s impl.

### Why `Cpu` and `Accelerate` chunk differently

`Cpu::try_generate_batch` derives one basis **per path**, not per chunk,
because each path's `ndrustfft::ndfft_inplace_par` is itself a nested
rayon region — measurement showed chunking roughly **doubled** wall time
at `m = 1000`. `Accelerate` does chunk, because `vDSP_fft_zip` is a
plain FFI call with no nested rayon. Do not "unify" these; the
divergence is measured and both impls document it.

## 6. Adding an Euler kernel

A process joins the Euler engine by implementing `EulerCoefficients`
(its `EulerSpec` family, initial value, grid, horizon, device seed,
`host_sample`) and switching its `backend_switch!` line from `via host`
to `via euler`. A new drift/diffusion family is one `EulerSpec` variant
plus one `family` branch in the kernels.

There are **two** kernel texts, not three: `euler/kernel.rs` holds one C
body that the native CUDA and the Metal back-ends both render (the
`Language` struct fills in `REAL`, `SQRT`, `LOG`, `COS` and the buffer
index type), and `euler/cubecl.rs` holds the CubeCL kernel, which repeats
the same integer hash in Rust. Changing the recursion means editing the
shared body and mirroring it in the CubeCL kernel — then re-checking the
CUDA text is byte-identical if you meant it to be, and re-running the
device tests.

## 7. Kernel conventions

The fGN CUDA kernel is a `pub(super) const &str` of CUDA C compiled by
NVRTC at runtime (`cuda/kernels.rs`), fusing RNG + scaling into
one launch to avoid a memory round-trip. Four things carry over to a new
kernel:

- **Counter-based RNG** (Philox-2×32-10 for fGN, a Murmur3-style
  finalizer for the Euler body), not cuRAND and no per-thread state:
  the output must be a pure function of `(global index, seed)` or the
  chunk-identity guarantee dies.
- **Fuse.** One launch for RNG + Box–Muller + `sqrt_eigenvalue` scaling,
  writing interleaved complex, rather than three round-trips.
- **Pad to a power of two.** cuFFT and MPS are tuned for it; `Fgn::new`
  already does `n.next_power_of_two()` and carries the `offset`.
- **Cache device state per size.** `cuda/state.rs` and the Metal
  sampler keep the last `CACHE_SLOTS` (4) `(n, m)` shapes, built before
  the eviction so a failing build cannot empty the cache. The rc.0 GPU
  FBM bench regressed ~30 % from per-call allocation.

## 8. Anti-patterns

- **Do not** write a bit-for-bit CPU↔GPU equality test. See §3.
- **Do not** read `fgn.seed` inside a device sampler. Use the passed
  `seed_src`. See §3.
- **Do not** draw the launch seed inside the chunk loop.
- **Do not** `#[cfg]`-branch inside `sample()`. Add a handle; the
  dispatch is a type parameter with no runtime branch.
- **Do not** silently fall back to CPU when a feature is off. An
  unavailable backend must be a compile error — that is why the handles
  are themselves feature-gated — and a failing device must surface a
  `DeviceError`, never a host result.
- **Do not** implement a capability for a precision the kernels do not
  compute in. `f64` on Metal or CubeCL must not compile.
- **Do not** allocate device memory per call. Cache it per size.
- **Do not** resurrect the `gpu*` aliases, the `cuda-native` spelling, or a
  device name that means "whichever backend this build carries" (`"gpu"` and
  `"cubecl"` were removed for exactly that reason: they hide what ran).
- **Do not** make the `backend` field private: downstream
  `Process { n: 64, ..Default::default() }` struct-update syntax needs
  it public (E0451).
- **Do not** add a backend without extending the three normative tables
  in §5 step 6.

## 9. Reference impls

- `device.rs` — the handles, `Backend` / `HostBackend` / `FgnBackend`,
  the `gpu_backend!` macro, `chunk_rows`, the LRU cache helper, `Cpu`
  and `Accelerate` written out by hand.
- `euler.rs` — `EulerCoefficients`, `EulerKernel`, `EulerBackend`, the
  `host_euler_backend!` and `kernel_euler_backend!` macros.
- `noise/fgn/cuda/` — `mod.rs`, `kernels.rs` (NVRTC sources),
  `sampler.rs`, `state.rs`, `convert.rs`, `tests.rs`. The most complete
  fGN backend, and the only one with the two-stream pipeline.
- `noise/fgn/metal.rs`, `noise/fgn/cubecl.rs`, `noise/fgn/accelerate.rs` —
  the other three; the CubeCL one serves both runtimes through the
  `CubeclRuntime` trait, one implementor per runtime with its own client
  cache.
- `euler/kernel.rs`, `euler/cuda.rs`, `euler/metal.rs`,
  `euler/cubecl.rs` — the Euler engine's shared body and its devices; the
  CubeCL module also owns `CubeclRuntime`, the per-runtime client caches and
  the panic-catching `open`.
- `macros.rs`'s `backend_switch!` — generates `.on::<B2>()` and
  `.on_device(handle)` for a backend-generic process, in four forms
  (`via fgn` / `via phantom` / `via host` / `via euler`). Invoke it
  rather than hand-writing them.
- `tests/deterministic_parallelism_accelerate.rs` — what a
  reproducibility test for a host-side backend looks like.
- `tests/try_sample_matches_sample.rs` — the fallible surface
  (`ProcessExt::try_sample` / `try_sample_par`) agreeing with the
  panicking one on the host.

## Related SKILLs

- `feature-flag-management` — propagating `cuda` / `cubecl` /
  `metal` / `accelerate` from the sub-crate to the umbrella.
- `add-fractional-process` — the backend-generic consumers (`Fou`,
  `Fgbm`, `FJacobi`, `Cfou`, `JumpFou`, … all carry `B`).
- `bench-writing` — the exact `required-features` sets for the gated
  GPU benches.

---
name: add-gpu-sampler
description: How to add or extend a GPU / accelerated sampling backend (CUDA, Metal, wgpu, Accelerate) in stochastic-rs-stochastic. Invoke when porting fGN sampling to a new device, or when a backend-generic process needs to reach one.
---

# Add GPU sampler — stochastic-rs-stochastic

Acceleration in this workspace is **not** a `#[cfg]` fork inside
`sample()`. It is a compile-time type parameter. A process carries a
backend marker `B` (defaulting to `Cpu`), the `Backend` trait
monomorphises sampling to that marker with **no runtime branch**, and
the caller switches with a turbofish:

```rust
let fbm = Fbm::<f64, _>::new(0.7, 1024, None, Deterministic::new(42));
let paths = fbm.on::<CudaNative>().sample_par(256);
```

The markers only exist when their feature is compiled, so selecting an
unavailable backend is a **compile error**, not a silent runtime
fallback to CPU. Read `stochastic-rs-stochastic/src/device.rs` in full
before touching any of this — it is 249 lines, current, and it is the
contract.

## 1. The five backends and their real feature names

| Marker | Feature | What it is |
|---|---|---|
| `Cpu` | *(none — always available)* | Default `B` for every process |
| `CudaNative` | `cuda-native` | `cudarc` + cuFFT + NVRTC, fused Philox kernel |
| `CubeCl` | `gpu` (+ `gpu-cuda` / `gpu-wgpu`) | cubecl Rust kernels |
| `MetalNative` | `metal` | Hand-written MSL via the `metal` crate; **f32 only** |
| `Accelerate` | `accelerate` | Apple vDSP / AMX — a **CPU** path, not a GPU |

There is no bare `cuda` feature and no `metal-rs` dependency. The CUDA
backend is `cudarc`, not `cust`. Kernels are Rust string constants
compiled at runtime by NVRTC (`noise/fgn/cuda_native/kernels.rs`) —
there are **no `.cu` files** anywhere in the repo.

## 2. Reproducibility: read the table before you write a test

This is where the old version of this SKILL was actively harmful. It is
**not** true that "a GPU sampler must match the CPU sampler bit-for-bit".
The shipped, documented contract (`Backend`'s own trait doc) is:

| Backend | `sample` / `sample_par` reproducible? |
|---|---|
| `Cpu` | **Yes.** Same seed + same `m` ⇒ bit-identical, on any machine, under any rayon thread-pool size. |
| `Accelerate` | **No — measured, not assumed.** Seed *consumption* is thread-count independent, but `vDSP_fft_zip`'s arithmetic is not bit-stable across calls: 400 repeated calls on an idle M4 Max diverged 0 times; the same sweep under core saturation diverged 21/400, worst relative difference `2.08e-3`. Consistent with P-core/E-core dispatch. `Cpu` stayed bit-exact under identical load. |
| `CudaNative` / `CubeCl` / `MetalNative` | **Not guaranteed.** Each batch call draws one value from `fgn.seed` and hands it to the device kernel's own Philox/PCG RNG. Output is a function of the pinned seed and *not* of host thread-pool size (no host-side rayon fan-out in `generate_batch`), but bit-identity across driver versions, vendors, or repeated runs is untested and unpromised. |

So: **do not write a `cpu_and_gpu_match_bit_for_bit` test.** It
contradicts the design, and "fixing" a GPU backend to satisfy it would
mean reimplementing the device RNG on the host for no benefit. Treat the
three GPU backends and `Accelerate` as reproducible-effort-only.

One further trap, documented on `Fbm::sample_par`: on the three GPU
backends `Fbm` is **not even a function of the pinned seed**. If you
need seed-pinned fBm, stay on `Cpu`.

What you test instead:

- **Distributional agreement**, not bit equality: variance scaling
  `Var(X_t) ∝ t^{2H}`, lag-1 autocovariance sign, mean ≈ 0 — the same
  properties `add-fractional-process` §5 lists, run on the new backend.
- **Shape and finiteness**: `generate_batch(fgn, m, seed)` returns `m`
  paths of `fgn.out_len`, all finite.
- **Marker-is-a-backend**, the compile-time check `device.rs`'s own
  test module uses:
  ```rust
  #[test]
  fn marker_is_a_backend() {
      fn assert_backend<B: Backend>() {}
      assert_backend::<MyBackend>();
  }
  ```
- For a **CPU-side** backend like `Accelerate`, thread-count-independent
  *seed consumption* is testable and is tested —
  `tests/deterministic_parallelism_accelerate.rs`.

## 3. The `Backend` trait — what you implement

```rust
pub trait Backend: Sized + Send + Sync {
    /// One fGN increment vector. `seed` drives CPU/Accelerate only;
    /// GPU backends ignore it and seed from `fgn.seed`.
    fn generate<T: FloatExt, S: SeedExt, S2: SeedExt>(
        fgn: &Fgn<T, S, Self>, seed: &S2,
    ) -> Array1<T>;

    /// `m` paths in one batched call.
    fn generate_batch<T: FloatExt, S: SeedExt, S2: SeedExt>(
        fgn: &Fgn<T, S, Self>, m: usize, seed: &S2,
    ) -> Vec<Array1<T>>;

    /// Two independent paths. Defaults to `generate_batch(fgn, 2, seed)`;
    /// `Cpu` overrides it with the real/imag parts of a single circulant
    /// FFT (Dietrich & Newsam) — one FFT, two independent fields.
    fn generate_pair<T, S, S2>(fgn: &Fgn<T, S, Self>, seed: &S2)
        -> (Array1<T>, Array1<T>) { … }
}
```

Two methods required, one defaulted. The markers are zero-sized unit
structs, which is what makes the `Send + Sync` supertraits free.

**The `seed: &S2` parameter is the whole reproducibility mechanism.** If
your backend runs on the host, thread it through and derive per-unit
bases *sequentially on the calling thread* before handing work to rayon.
If it runs on-device, ignore it and document that you did.

## 4. Adding a GPU backend

Device-side backends are registered by the `gpu_backend!` macro in
`device.rs` — three lines do all three GPU backends:

```rust
gpu_backend!("cuda-native", CudaNative => sample_cuda_native_impl);
gpu_backend!("gpu",         CubeCl     => sample_gpu_impl);
gpu_backend!("metal",       MetalNative => sample_metal_impl);
```

The macro generates a feature-gated `impl Backend` whose `generate` /
`generate_batch` both delegate to one inherent method on `Fgn` with the
signature `fn <name>(&self, m: usize) -> Result<Array2<T>, _>`, ignoring
the host seed. So the work is:

1. Declare the marker in `device.rs`, feature-gated and `#[derive(Clone, Copy)]`.
2. Write `Fgn::sample_<name>_impl(&self, m) -> Result<Array2<T>, _>` in
   a new `noise/fgn/<name>.rs`, gated on the feature; add the `#[cfg]`
   `use` to `noise/fgn.rs`.
3. Add one `gpu_backend!` line.
4. Add the feature to `stochastic-rs-stochastic/Cargo.toml` **and**
   propagate it from the umbrella root `Cargo.toml` — see
   `feature-flag-management`.
5. Extend the reproducibility table in `Backend`'s trait doc. It is
   normative; a backend absent from it is undocumented.

A host-side backend (the `Accelerate` shape) does **not** use the macro:
write the `impl Backend` by hand so you can thread `seed` through
`chunk_count` / `chunk_lens` and derive one basis per chunk sequentially
before `into_par_iter()`. Copy `Accelerate`'s impl.

### Why `Cpu` and `Accelerate` chunk differently

`Cpu::generate_batch` derives one basis **per path**, not per chunk,
because each path's `ndrustfft::ndfft_inplace_par` is itself a nested
rayon region — measurement showed chunking roughly **doubled** wall time
at `m = 1000`. `Accelerate` does chunk, because `vDSP_fft_zip` is a
plain FFI call with no nested rayon. Do not "unify" these; the
divergence is measured and both impls document it.

## 5. Kernel conventions

The CUDA kernel is a `pub(super) const &str` of CUDA C compiled by
NVRTC at runtime (`cuda_native/kernels.rs`), fusing RNG + scaling into
one launch to avoid a memory round-trip:

```cuda
extern "C" __global__ void gen_scale_f32(
    float* __restrict__ data, const float* __restrict__ sqrt_eigs,
    int traj_size, int total,
    unsigned long long seed, unsigned long long seq)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    /* Philox-2x32-10, counter = tid + seq, key = seed */
    …
    float eig = sqrt_eigs[tid % traj_size];
    data[2*tid]   = r * cs * eig;      /* interleaved complex output */
    data[2*tid+1] = r * sn * eig;
}
```

Four things that carry over to a new kernel:

- **Philox-2×32-10**, not cuRAND. Counter-based, no per-thread state to
  thread through, and it drops the cuRAND dependency plus one round-trip.
  (Note: 2×32, not the 4×32 an earlier version of this SKILL claimed.)
- **Fuse.** One launch for RNG + Box-Muller + `sqrt_eigenvalue` scaling,
  writing interleaved complex, rather than three round-trips.
- **f32 on device.** Apple GPUs have no f64; the CUDA path also runs
  f32. Convert at the host boundary (`cuda_native/convert.rs`).
- **Pad to a power of two.** cuFFT and MPS are tuned for it; `Fgn::new`
  already does `n.next_power_of_two()` and carries the `offset`.

Cache device state on the struct rather than per call —
`cuda_native/state.rs` exists for that. The rc.0 GPU FBM bench regressed
~30 % from per-call allocation.

## 6. Anti-patterns

- **Do not** write a bit-for-bit CPU↔GPU equality test. See §2.
- **Do not** `#[cfg]`-branch inside `sample()`. Add a `Backend` marker;
  the dispatch is a type parameter with no runtime branch.
- **Do not** silently fall back to CPU when a feature is off. An
  unavailable backend must be a compile error — that is why the markers
  are themselves feature-gated.
- **Do not** use cuRAND, or a stateful per-thread RNG.
- **Do not** allocate device memory per call. Cache it in the
  backend's state module.
- **Do not** name the feature `cuda`. It is `cuda-native` (cudarc) or
  `gpu` / `gpu-cuda` / `gpu-wgpu` (cubecl); they are different backends.
- **Do not** add a backend without extending `Backend`'s
  reproducibility table.

## 7. Reference impls

- `device.rs` — the `Backend` trait, all five markers, the
  `gpu_backend!` macro, `Cpu` and `Accelerate` written out by hand.
- `noise/fgn/cuda_native/` — `mod.rs`, `kernels.rs` (NVRTC sources),
  `sampler.rs`, `state.rs` (cached device state), `convert.rs`,
  `tests.rs`. The most complete backend.
- `noise/fgn/metal.rs`, `noise/fgn/gpu.rs`, `noise/fgn/accelerate.rs` —
  the other three.
- `macros.rs`'s `backend_switch!` — generates the `.on::<B2>()` method
  for a backend-generic process, in two forms (real `fgn` field, or
  `PhantomData`). Invoke it rather than hand-writing `on`.
- `tests/deterministic_parallelism_accelerate.rs` — what a
  reproducibility test for a host-side backend actually looks like.

## Related SKILLs

- `feature-flag-management` — propagating `cuda-native` / `gpu` /
  `metal` / `accelerate` from the sub-crate to the umbrella.
- `add-fractional-process` — the backend-generic consumers (`Fou`,
  `Fgbm`, `FJacobi`, `Cfou`, `JumpFou`, … all carry `B`).
- `bench-writing` — the exact `required-features` sets for the gated
  GPU benches.

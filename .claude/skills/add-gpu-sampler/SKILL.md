---
name: add-gpu-sampler
description: How to add a GPU-accelerated sampler (CUDA / Metal) for a process or distribution. Invoke when porting a CPU sampler to GPU for the long-horizon / many-paths regime where CPU SIMD saturates.
---

# Add GPU sampler — stochastic-rs

GPU samplers in this workspace target two backends: **CUDA** (via
`cust` + NVRTC for kernels) on Linux/Windows with NVIDIA hardware, and
**Metal** (via `metal-rs`) on Apple Silicon. Both are feature-gated:

- `--features cuda` → enables `cuda_native` paths.
- `--features metal` → enables `metal` paths.

CPU-fallback parity is **non-negotiable**: a GPU sampler must match
the CPU sampler bit-for-bit (within float-precision noise) on a fixed
seed. Without this, a user toggling features sees inconsistent
results.

## 1. Pre-flight checklist

Before writing the kernel:

1. **FFT length is a power of 2.** GPU FFTs (cuFFT, Metal Performance
   Shaders) are highly tuned for power-of-2 lengths. If your sampler
   needs a non-power-of-2, pad with zeros to the next power-of-2.

2. **RNG choice: Philox vs cuRAND.** Philox-4×32-10 is the workspace
   default (see `noise/fgn/cuda_native.rs`). It is:
   - Counter-based (no state to thread through kernels).
   - Deterministic across CPU and GPU at the bit level (the CPU SIMD
     path uses the same Philox).
   - Identical seed → identical output.

   Avoid cuRAND/curandStateXORWOW: it stateful, GPU-only, and not
   reproducible across CPU.

3. **Fused-kernel layout.** Combine RNG generation + scaling +
   transform into a single kernel rather than three round-trips. The
   FGN GPU pipeline fuses Philox + Gaussian-transform + sqrt-eigenvalue
   scaling in one launch — see `noise/fgn/cuda_native.rs`.

4. **Memory layout.** Output is row-major `Array2<T>` matching the
   CPU shape. Allocate device memory once per Process struct (cache it
   between `sample()` calls) — the rc.0 GPU FBM bench had a 30 %
   regression because each call re-allocated.

## 2. The skeleton

```rust
// stochastic-rs-stochastic/src/noise/fgn/cuda_native.rs (reference)

#[cfg(feature = "cuda")]
pub mod cuda {
    use cust::prelude::*;
    use crate::traits::FloatExt;

    /// Compile-time-baked NVRTC kernel; loaded once per process.
    static FGN_KERNEL_PTX: &str = include_str!("fgn_kernel.cu");

    pub fn fgn_sample_cuda<T: FloatExt>(
        hurst: T,
        n: usize,
        t: T,
        seed: u64,
    ) -> Vec<T> {
        let _ctx = quick_init().expect("CUDA init failed");
        let module = Module::from_ptx(FGN_KERNEL_PTX, &[]).unwrap();
        let stream = Stream::new(StreamFlags::DEFAULT, None).unwrap();

        // Allocate device-side output buffer.
        let mut d_out = unsafe {
            DeviceBuffer::<T>::uninitialized(n).unwrap()
        };

        // Launch fused-kernel: Philox + Gaussian + sqrt-eig + cumsum.
        let kernel = module.get_function("fgn_sample_kernel").unwrap();
        let block_size = 256u32;
        let grid_size = ((n as u32) + block_size - 1) / block_size;

        unsafe {
            launch!(
                kernel<<<grid_size, block_size, 0, stream>>>(
                    d_out.as_device_ptr(),
                    hurst.to_f64().unwrap() as f32,
                    n as u32,
                    t.to_f64().unwrap() as f32,
                    seed,
                )
            ).unwrap();
        }

        stream.synchronize().unwrap();

        let mut h_out = vec![T::zero(); n];
        d_out.copy_to(&mut h_out).unwrap();
        h_out
    }
}
```

The `.cu` source lives next to the `.rs` and is compiled at runtime
via NVRTC. It must define `fgn_sample_kernel(__global__)` with
matching argument types.

## 3. The `.cu` kernel — Philox baseline

```cuda
// fgn_kernel.cu (excerpt)

__device__ inline float philox_normal(uint64_t seed, uint32_t idx) {
    // Philox-4x32-10 → uniform → Box-Muller → Gaussian
    uint4 ctr = {idx, 0u, 0u, 0u};
    uint2 key = {(uint32_t)(seed & 0xffffffff), (uint32_t)(seed >> 32)};
    uint4 u = philox4x32_10(ctr, key);
    float u0 = (float)u.x / 4294967296.0f;
    float u1 = (float)u.y / 4294967296.0f;
    return sqrtf(-2.0f * logf(u0)) * cosf(6.2831853f * u1);
}

extern "C" __global__ void fgn_sample_kernel(
    float* out, float hurst, uint32_t n, float t, uint64_t seed
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float z = philox_normal(seed, idx);
    // ... apply sqrt-eigenvalue scaling + cumsum-style aggregation ...
    out[idx] = z;
}
```

The CPU SIMD path uses the same Philox-4×32-10 with the same `(seed,
idx)` counter, so bit-for-bit determinism is achievable.

## 4. ProcessExt overrides

A process exposing GPU sampling overrides one or more of the four
`ProcessExt::sample_*` methods:

```rust
impl<T: FloatExt, S: SeedExt> ProcessExt<T> for FbmGpu<T, S> {
    type Output = Array1<T>;

    fn sample(&self) -> Self::Output {
        #[cfg(feature = "cuda")]
        {
            return Array1::from(cuda::fgn_sample_cuda(self.hurst, self.n, self.t.unwrap_or(T::one()), self.seed.derive().to_u64()));
        }
        #[cfg(feature = "metal")]
        {
            return Array1::from(metal::fgn_sample_metal(...));
        }
        // CPU fallback
        self.sample_cpu()
    }

    fn sample_par(&self, m: usize) -> Vec<Self::Output> { /* batched GPU launch */ }
}
```

The conditional gates ensure a single binary built without GPU
features still compiles; the cfg-disabled path silently falls through
to CPU.

## 5. Bit-for-bit reproducibility test

```rust
#[cfg(test)]
#[cfg(feature = "cuda")]
mod cuda_tests {
    /// CPU and GPU samplers produce identical paths under the same seed.
    #[test]
    fn cpu_and_gpu_match_bit_for_bit() {
        let cpu = FbmCpu::seeded(0.7, 1024, None, None, 42).sample();
        let gpu = FbmGpu::seeded(0.7, 1024, None, None, 42).sample();
        for i in 0..1024 {
            assert_eq!(cpu[i].to_bits(), gpu[i].to_bits(), "mismatch at i={i}");
        }
    }
}
```

If the test fails, the CPU and GPU code paths diverged — likely a
different RNG implementation or a different Box-Muller order. Don't
relax the tolerance; debug to bit equality.

## 6. Anti-patterns

- **Do not** use cuRAND. Bit-deterministic CPU↔GPU equality is the
  workspace requirement; cuRAND breaks that.
- **Do not** allocate device memory inside the hot loop. Cache the
  buffer on the struct.
- **Do not** mix f32 and f64 across CPU/GPU. The kernel runs in f32 by
  default (Apple Silicon Metal restricts; CUDA prefers); convert at the
  boundary.
- **Do not** ship a GPU sampler without the bit-equality test. The §1.4
  audit caught a mismatch where the GPU FBM was correct *on average*
  but had a different seed-mapping; the test made the bug obvious.

## 7. Reference impls

- `noise/fgn/cuda_native.rs` — Philox + scale fused kernel; the
  reference for the §3 / §5 patterns.
- `noise/fgn/metal.rs` — Metal Performance Shaders alternative.
- `noise/fgn/gpu.rs` — `wgpu`-backed cross-platform fallback (lower
  priority than cuda_native).

## Related SKILLs

- `feature-flag-management` — for the `cuda` / `metal` propagation
  through workspace.
- `add-fractional-process` — the natural consumer of GPU FGN.
- `bench-writing` — for `[[bench]] required-features = ["cuda"]` gating.

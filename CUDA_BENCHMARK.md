# fGN sampling: PC (i9-285K + RTX 4070 SUPER)

`benches/fgn_cuda_compare.rs`, same parameters as the Apple M4 Max table so the
machines are directly comparable. Times are criterion medians; **lower is
better**. `n` = path length, `m` = number of paths.

Build with the `mimalloc` global allocator — the batch path allocates one
`Array1` per path, so the default Windows allocator otherwise bottlenecks it:

```bash
cargo bench --bench fgn_cuda_compare --features "cuda-native,gpu-cuda,mimalloc"
```

## Single path (`sample`, m = 1)

| Backend | n=1k | n=4k | n=16k | n=64k |
|---|--:|--:|--:|--:|
| CPU (i9-285K) | **7.9 µs** | **34.2 µs** | 150 µs | 601 µs |
| cuFFT (cudarc) | 51 µs | 83 µs | **102 µs** | **219 µs** |
| cubecl | 140 µs | 196 µs | 248 µs | 486 µs |

## Batch (`sample_par`, n × m)

| Backend | 1k×1k | 4k×16k | 16k×16k |
|---|--:|--:|--:|
| CPU (i9-285K) | **556 µs** | **33.7 ms** | **143 ms** |
| cuFFT (cudarc) | 716 µs | 53.6 ms | 481 ms |
| cubecl | 889 µs | 81.6 ms | — (OOM > 12 GB) |

## PC vs Apple M4 Max (batch, lower = better)

| n × m | PC CPU (i9) | Apple CPU | PC best GPU | Apple best GPU |
|---|--:|--:|--:|--:|
| 1k×1k | **556 µs** | 1.46 ms | cuFFT 716 µs | Metal 2.08 ms |
| 4k×16k | **33.7 ms** | 80.8 ms | cuFFT 53.6 ms | Metal 160 ms |
| 16k×16k | **143 ms** | 339 ms | cuFFT 481 ms | Metal 628 ms |

The PC wins every cell; the M4 Max's edge in the old run was purely the missing
fast allocator.

## When is the GPU worth it?

- **Single long path:** the crossover is **n ≈ 16k**. Below it the CPU wins
  (kernel-launch + transfer overhead dominates); above it cuFFT pulls ahead —
  ~3× at n = 64k. Use cuFFT for one big FFT.
- **Batches:** the **CPU wins at every size here**, because a 24-core i9
  parallelises `m` independent FFTs perfectly (rayon + per-core Ziggurat RNG,
  cache-friendly, no transfer), while the GPU pays for (a) generating all
  Gaussians on-device via Box–Muller (`ln`/`cos`/`sin` → SFU-bound, slower than
  Ziggurat), and (b) copying the full `m·n` result back over PCIe (~1 GB at
  16k×16k). This is **not** CUDA-specific — on the M4 Max, Metal batch (628 ms)
  is also slower than its CPU (339 ms) for the same reasons.
- **So the GPU pays off only when the generated paths stay on the device** for a
  downstream GPU step (Fourier pricing, Greeks, RL rollouts) — eliminating the
  device→host copy. For "generate then copy back to host", the CPU is the better
  choice on this hardware.

## cubecl note (updated)

The updated cubecl backend is **dramatically faster** than the previous version:

| case | old | new | speedup |
|---|--:|--:|--:|
| single n=16k | 509 µs | 248 µs | ~2× |
| single n=64k | 1.71 ms | 486 µs | ~3.5× |
| batch 1k×1k | 25.9 ms | 889 µs | ~29× |
| batch 4k×16k | 2.29 s | 81.6 ms | ~28× |

cubecl is no longer the outlier it was — it now lands within ~1.5–2× of cuFFT
across the grid and scales with problem size like a real batched FFT (the old
build was kernel-launch-bound at a roughly constant ~30–45 Melem/s regardless of
size). It is still a step behind cuFFT (vendor-tuned plans) and behind the CPU
for batches, for the same on-device-RNG + device→host-copy reasons above.

16k×16k cubecl is still skipped: cuFFT's ~5.4 GB device buffers plus cubecl's
own pools would exceed the 12 GB card (the grid runs largest-first, so cuFFT
allocates before cubecl), and cubecl-CUDA has **no** 256 MB buffer cap (that is a
wgpu limit, not a cubecl-CUDA one).

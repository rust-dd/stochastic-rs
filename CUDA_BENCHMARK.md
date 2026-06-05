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
| CPU (i9-285K) | **7.9 µs** | **35.8 µs** | 148 µs | 599 µs |
| cuFFT (cudarc) | 52 µs | 76 µs | **111 µs** | **190 µs** |
| cubecl | 145 µs | 210 µs | 509 µs | 1.71 ms |

## Batch (`sample_par`, n × m)

| Backend | 1k×1k | 4k×16k | 16k×16k |
|---|--:|--:|--:|
| CPU (i9-285K) | **549 µs** | **35.1 ms** | **155 ms** |
| cuFFT (cudarc) | 719 µs | 53.9 ms | 449 ms |
| cubecl | 25.9 ms | 2.29 s | — (OOM > 12 GB) |

## PC vs Apple M4 Max (batch, lower = better)

| n × m | PC CPU (i9) | Apple CPU | PC best GPU | Apple best GPU |
|---|--:|--:|--:|--:|
| 1k×1k | **549 µs** | 1.46 ms | cuFFT 719 µs | Metal 2.08 ms |
| 4k×16k | **35.1 ms** | 80.8 ms | cuFFT 53.9 ms | Metal 160 ms |
| 16k×16k | **155 ms** | 339 ms | cuFFT 449 ms | Metal 628 ms |

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

## cubecl note

cubecl is slow because of its hand-written radix-2 shared-memory FFT, not a
buffer-size limit: it is already ~25 ms at 1k×1k (a few-MB buffer) and its
throughput is roughly constant (~30–45 Melem/s) across all sizes — it is
kernel-launch-bound. cubecl-CUDA has **no** 256 MB buffer cap (that is a wgpu
limit). 16k×16k cubecl is skipped because cuFFT's ~5.4 GB device buffers plus
cubecl's would exceed the 12 GB card (the grid runs largest-first so cuFFT
allocates before cubecl pools VRAM).

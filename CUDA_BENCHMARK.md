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
| cuFFT (cudarc) | 686 µs | 51.4 ms | 253 ms |
| cubecl | 889 µs | 81.6 ms | — (OOM > 12 GB) |

cuFFT batch 16k×16k was **481 ms** before the device→host transfer was optimised
(see below) — now **253 ms**.

## PC vs Apple M4 Max (batch, lower = better)

| n × m | PC CPU (i9) | Apple CPU | PC best GPU | Apple best GPU |
|---|--:|--:|--:|--:|
| 1k×1k | **556 µs** | 1.46 ms | cuFFT 686 µs | Metal 2.08 ms |
| 4k×16k | **33.7 ms** | 80.8 ms | cuFFT 51.4 ms | Metal 160 ms |
| 16k×16k | **143 ms** | 339 ms | cuFFT 253 ms | Metal 628 ms |

The PC wins every cell; the M4 Max's edge in the old run was purely the missing
fast allocator.

## cuFFT batch pipeline breakdown

Per-call breakdown of `cuda_native` (cuFFT) `sample_par`, measured in-process
with `STOCHASTIC_RS_CUDA_PROFILE=1` (env-gated phase timing in the sampler).
`compute` = on-device RNG + batched FFT + extract; `dtoh` = device→host transfer
+ the parallel host copy.

| n × m | compute | dtoh | pipeline total |
|---|--:|--:|--:|
| 1k×1k | 83 µs | 303 µs | 386 µs |
| 4k×16k | 9.3 ms | 19.7 ms | 29 ms |
| 16k×16k | 56 ms | **79 ms** | 135 ms |

At 16k×16k the result is ~1 GB of `f32`, and moving it to the host dominates —
`dtoh` is ~58 % of the pipeline. The end-to-end criterion median (253 ms) is
higher than the 135 ms pipeline total because it also includes host-side
allocation and freeing of the 1 GB result `Array2` around each call, which is
outside the GPU pipeline.

### How the device→host transfer was optimised (280 ms → 79 ms)

The naive path — `clone_dtoh` straight into a fresh pageable `Vec` — ran at
**~3.9 GB/s** (≈280 ms for 1 GB). Three things were wrong, fixed in order:

1. **Pageable memory caps the DMA.** A device→host copy into ordinary (pageable)
   host memory is staged through a small driver bounce buffer in chunks, so it
   never reaches link bandwidth no matter the PCIe gen. Copying instead into a
   **page-locked (pinned) staging buffer** (cached in the sized context, pinned
   once per parameter set) lets the driver DMA directly at **~24 GB/s** (≈41 ms
   for 1 GB) — see `examples/cuda_d2h_bw.rs`.
2. **The staging→output copy was serial.** The result still has to land in an
   owned `Vec` for the `Array2`. A single-threaded copy of a *fresh* 1 GB buffer
   is dominated by first-touch page faults (~3.7 GB/s). A **rayon-parallel copy**
   spreads those faults across all 24 cores.
3. **The page faults were on the critical path.** The output `Vec` is now
   allocated and **pre-faulted while the GPU kernels are still running** (the
   launches are async), so the faulting overlaps `compute` and the post-transfer
   copy is pure bandwidth. This took `dtoh` from ~100 ms to **~79 ms**.

Small transfers (< 32 MB) skip all of this and use a plain `clone_dtoh` — the
staging round-trip and rayon overhead aren't worth it there.

### How PCIe Gen4 bandwidth is actually reached

The link **idles at Gen1** (PCIe ASPM power saving) and **auto-ramps to Gen4
under sustained load** — confirmed by polling `nvidia-smi
--query-gpu=pcie.link.gen.current` during the bench (it steps 1 → 4 while the
kernels run, then decays back to 1 when idle). No BIOS/driver change forces it.

The key point: the earlier ~3.6 GB/s was *not* the link sitting at Gen1 — even
with the link at Gen4 the pageable bounce-buffer copy stays ~3.9 GB/s. **Pinned
memory is what unlocks the Gen4 bandwidth that was already there.** The RTX 4070
SUPER is Gen4-max (the board's host is Gen5-capable), so ~24 GB/s / ~41 ms per
GB is the hardware ceiling for this transfer.

### What could reduce `dtoh` further

- **Double-buffer the transfer.** The remaining `dtoh` is ~41 ms DMA + ~38 ms
  bandwidth-bound copy, run back-to-back. Splitting the result into chunks with
  two pinned staging slots so chunk *k*'s copy overlaps chunk *k+1*'s DMA would
  bring `dtoh` toward the **~45 ms DMA floor**. (Not done — more moving parts;
  the current win is the bulk of it.)
- **Move less data.** Below ~41 ms is impossible while returning 1 GB of `f32`.
  `f16` output would halve it (~22 ms) at a precision cost; otherwise the only
  way to skip the copy entirely is to **keep the paths on the device** for a
  downstream GPU step (pricing, Greeks, RL rollouts).
- **A faster bus** (PCIe 5.0) is not available — the GPU is Gen4-max.

## When is the GPU worth it?

- **Single long path:** the crossover is **n ≈ 16k**. Below it the CPU wins
  (kernel-launch + transfer overhead dominates); above it cuFFT pulls ahead —
  ~3× at n = 64k. Use cuFFT for one big FFT.
- **Batches:** the **CPU still wins at every size here** (143 ms vs cuFFT's
  253 ms at 16k×16k), but the gap is now ~1.8× rather than ~3×. A 24-core i9
  parallelises `m` independent FFTs perfectly (rayon + per-core Ziggurat RNG,
  cache-friendly, no transfer), while the GPU still pays for (a) on-device
  Box–Muller RNG (SFU-bound) and (b) copying the full `m·n` result back over
  PCIe — even at full Gen4 bandwidth, ~41 ms of unavoidable transfer for 1 GB.
- **So the GPU pays off when the generated paths stay on the device** for a
  downstream GPU step — eliminating the device→host copy. For "generate then
  copy back to host", the CPU is still the better choice on this hardware.

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

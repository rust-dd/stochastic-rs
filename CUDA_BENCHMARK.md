# CUDA benchmarks

Two machines, two workloads. The fGN sections below are the FFT pipeline on a
desktop RTX 4070 SUPER; the last section is the Euler engine and fGN together
on a Colab Tesla T4.

## fGN sampling: PC (i9-285K + RTX 4070 SUPER)

Measured with the `fgn_cuda_compare` bench (since removed; `fgn_cuda` carries
the cuFFT legs), same parameters as the Apple M4 Max table so the machines are
directly comparable. Times are criterion medians; **lower is better**. `n` =
path length, `m` = number of paths.

The portable CubeCL backend was removed in 3.0.0-rc.2: it duplicated the native
CUDA and Metal kernels and was slower on the same hardware.

Build with the `mimalloc` global allocator — the batch path allocates one
`Array1` per path, so the default Windows allocator otherwise bottlenecks it:

```bash
cargo bench --bench fgn_cuda --features "cuda,mimalloc"
```

### Single path (`sample`, m = 1)

| Backend | n=1k | n=4k | n=16k | n=64k |
|---|--:|--:|--:|--:|
| CPU (i9-285K) | **7.9 µs** | **34.2 µs** | 150 µs | 601 µs |
| cuFFT (cudarc) | 51 µs | 83 µs | **102 µs** | **219 µs** |

### Batch (`sample_par`, n × m)

| Backend | 1k×1k | 4k×16k | 16k×16k |
|---|--:|--:|--:|
| CPU (i9-285K) | **556 µs** | **33.7 ms** | **143 ms** |
| cuFFT (cudarc) | 686 µs | 51.4 ms | 253 ms |

cuFFT batch 16k×16k was **481 ms** before the device→host transfer was optimised
(see below) — now **253 ms**.

### PC vs Apple M4 Max (batch, lower = better)

| n × m | PC CPU (i9) | Apple CPU | PC best GPU | Apple best GPU |
|---|--:|--:|--:|--:|
| 1k×1k | **556 µs** | 1.46 ms | cuFFT 686 µs | Metal 2.08 ms |
| 4k×16k | **33.7 ms** | 80.8 ms | cuFFT 51.4 ms | Metal 160 ms |
| 16k×16k | **143 ms** | 339 ms | cuFFT 253 ms | Metal 628 ms |

The PC wins every cell; the M4 Max's edge in the old run was purely the missing
fast allocator.

### cuFFT batch pipeline breakdown

Per-call breakdown of `cuda` (cuFFT) `sample_par`, measured in-process
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

#### How the device→host transfer was optimised (280 ms → 79 ms)

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

#### How PCIe Gen4 bandwidth is actually reached

The link **idles at Gen1** (PCIe ASPM power saving) and **auto-ramps to Gen4
under sustained load** — confirmed by polling `nvidia-smi
--query-gpu=pcie.link.gen.current` during the bench (it steps 1 → 4 while the
kernels run, then decays back to 1 when idle). No BIOS/driver change forces it.

The key point: the earlier ~3.6 GB/s was *not* the link sitting at Gen1 — even
with the link at Gen4 the pageable bounce-buffer copy stays ~3.9 GB/s. **Pinned
memory is what unlocks the Gen4 bandwidth that was already there.** The RTX 4070
SUPER is Gen4-max (the board's host is Gen5-capable), so ~24 GB/s / ~41 ms per
GB is the hardware ceiling for this transfer.

#### What could reduce `dtoh` further

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

### When is the GPU worth it?

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

## Euler engine: Colab Tesla T4

The recursion, not the FFT: one CUDA thread per path, the whole step loop in
the kernel. GBM through `sample_map`, best of five, from
`cargo run --release --example cuda_engine_profile --features cuda`, measured
2026-09-07. The host is a Colab VM — its CPU column runs at roughly a tenth
of an M4 Max's, so the ratios are the reading, not the absolutes.

The kernel as the driver reports it, at 256 threads a block:

| dtype | registers/thread | local/thread | shared | blocks/SM | occupancy |
|---|--:|--:|--:|--:|--:|
| `float` | **31** | **0 B** | 0 B | 4 | **100 %** |
| `double` | 40 | 48 B | 0 B | 4 | **100 %** |

Turing holds **1024** threads a multiprocessor, not the 2048 of most other
generations, so 4 blocks × 256 is the whole machine. The example used to
divide by a hard-coded 2048 and print that as 50 %; it now reads
`CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR` from the device.

### Steps per second (paths × grid points ÷ wall time)

| paths | steps | dtype | cpu ms | cuda ms | CPU | CUDA | speed-up |
|---:|---:|---|--:|--:|--:|--:|--:|
| 10 000 | 1 024 | `f32` | 48.0 | **4.4** | 0.21 G | **2.35 G** | 11.01× |
| 50 000 | 1 024 | `f32` | 242.7 | **67.0** | 0.21 G | **0.76 G** | 3.62× |
| 10 000 | 4 096 | `f32` | 192.5 | **16.7** | 0.21 G | **2.45 G** | 11.49× |
| 10 000 | 1 024 | `f64` | 48.6 | **15.6** | 0.21 G | **0.66 G** | 3.12× |
| 50 000 | 1 024 | `f64` | 362.3 | **81.9** | 0.14 G | **0.63 G** | 4.43× |
| 10 000 | 4 096 | `f64` | 192.3 | **51.5** | 0.21 G | **0.80 G** | 3.74× |

The 50 000-path `f32` row is the one to distrust: the run before this one, on
the same code and the same kind of runtime, read 31.6 ms and 1.62 G. Colab's
GPU is shared and that row moves the most bytes of the six — 205 MB — so it
is the most exposed to a neighbour. Read it as somewhere between 0.8 and 1.6
rather than as a regression.

### The fold in the kernel

The same shapes, `sample_map_view` against `sample_reduce`, which returns one
value a path instead of the whole grid:

| paths | steps | mapped | reduced | | reduced |
|---:|---:|-------:|--------:|---:|--------:|
| 10 000 | 1 024 | 4.3 ms | **0.2 ms** | 18.2× | 43.3 G steps/s |
| 50 000 | 1 024 | 62.5 ms | **0.6 ms** | 102.0× | 83.5 G steps/s |
| 10 000 | 4 096 | 16.7 ms | **0.8 ms** | 21.9× | 53.6 G steps/s |

**This is what makes the card worth its bus.** Every other entry point sends
`paths × steps` four-byte values back across PCIe; this one sends `paths`, so
the crossing shrinks by the grid factor — 205 MB becomes 200 KB in the middle
row. The card stops waiting on the bus and starts running its kernel, which
is why this is 18× to 102× where lending the buffer was 5× to 8×.

Numbers that steep are worth checking against the hardware rather than
believing. A T4 issues one instruction per warp scheduler per cycle over a
warp of 32 threads: 4 × 32 × 40 SM × 1.59 GHz = **8.1 × 10¹² thread
instructions a second**. At 85 G steps/s that is 95 instructions a step, at
51 G it is 159. A GBM step — the counter hash, a Box-Muller pair, the Euler
update — is roughly 40 to 50 with `__logf`/`__cosf`, so these sit at 30 to
50 % of peak issue. Steep, and possible; the fast transcendentals are what
put it there, since the accurate `logf` and `cosf` are tens of instructions
each on their own.

**The per-shape kernel did what it was for.** The monolithic body cost this
same card 80 registers and **3 552 bytes of local memory per thread**, three
blocks a multiprocessor, 37 % occupancy and 0.15 G steps/s against 0.21 on
its host. Rendering one kernel per launch shape took the local memory to
**zero**, the registers to **31** and the occupancy to a full multiprocessor.
That alone doubled the throughput; everything past it was the road home.

**`f64` is not slower than this machine's own CPU, and this corrects what
this file said before.** At 0.59–0.86× the natural reading was the T4's 1/32
double-precision rate, so the hardware had spoken. It had not: `f64` moves
twice the bytes and was paying twice the host-side copy. The evidence was in
the table — the `f64` rows took 2.1–2.3× the `f32` rows, where a kernel-bound
`f64` run on this card would have taken thirty. With the copy gone `f64` runs
at **3.1–4.4×** its host. Prefer `f32` on a consumer card for the bytes, not
for the ALU.

### fGN on the same machine, for contrast

The FFT pipeline, not the engine: `sample_par`, `f32`, H = 0.7, best of five.

| paths | points | cpu ms | cuda ms | speed-up |
|---:|---:|--:|--:|--:|
| 1 000 | 4 096 | 83.6 | 20.6 | **4.07×** |
| 1 000 | 16 384 | 360.4 | 117.7 | **3.06×** |
| 10 000 | 4 096 | 891.5 | 301.4 | **2.96×** |

fGN sits at **3.0–4.1×** and none of the engine's changes touched it — the
Metal pipeline was rewritten today, the CUDA one was not. An FFT does
O(n log n) device work per byte returned where a Euler step does O(1), which
is why it led for so long; the engine has now overtaken it, because a folded
batch returns almost no bytes at all.

### Where the T4's time actually goes

*Written when the card measured 0.31 G steps/s, and kept because the
diagnosis is what produced everything above it. Read it as history: the
host-side items of the ranked list at the end have landed and were worth
5–8×, not the 2–2.8× estimated here — the estimates assumed a desktop-class
host, where a two-vCPU VM makes every host-side byte dearer than the bus.
Folding in the kernel, which the list did name as the largest remaining
item, then took the shapes it applies to another 18× to 102×.*

An M4 Max reached **10.10** G steps/s on the same kernel where the T4 read
**0.31** — 33×, where the two cards are within about 2× of each other in raw `f32`
throughput (T4 ≈ 8.1 TFLOPS, M4 Max GPU ≈ 16). Hardware explains almost none
of it. The numbers say where the rest is.

**Cost per cell is flat, and flat rules the kernel out.** Wall time divided
by paths × steps:

| shape | `f32` ns/cell | `f64` ns/cell | `f64` − `f32` |
|---|--:|--:|--:|
| 10 000 × 1 024 | 3.21 | 7.29 | 4.08 |
| 50 000 × 1 024 | 3.63 | 8.02 | 4.39 |
| 10 000 × 4 096 | 3.46 | 8.17 | 4.71 |

10 000 paths is 10 000 threads against 40 SMs × 1024 slots — **24 %** of the
card's resident capacity. 50 000 oversubscribes it. Going from a quarter of
the thread slots to all of them changes the cost per cell by 13 %, and in the
wrong direction. A
kernel-bound run gets *cheaper* per cell as occupancy fills; this one does
not, so whatever dominates scales with cells and not with occupancy.

**The kernel is at most a per cent or two of the `f32` wall**, by two
independent routes.

1. *From the two precisions.* A T4's FP64 is 1/32 of its FP32, and `double`
   transcendentals are worse still, so if the `f32` kernel costs *K* per cell
   the `f64` kernel costs at least 32*K*. Switching precision costs a
   measured 4.08–4.71 ns per cell, and that buys 31*K* plus four more bytes
   through the copy. Those four bytes cost at most the whole `f32` wall of
   the same shape, 3.21–3.63 ns. Taking the loosest of the three shapes,
   31*K* ≤ 1.25 ns and **K ≤ 0.04 ns/cell** — under 1.5 % of the `f32` wall,
   and the `f64` kernel under 20 % of the `f64` wall.
2. *From the instruction count.* A GBM step is on the order of 90
   instructions — about 30 of integer hashing for the two counter-hashed
   uniforms, about 50 for `logf` + `cosf` + `sqrtf` as NVRTC's IEEE-accurate
   routines, and under 10 for the Euler update and the store. That split is
   read off the rendered source and the usual cost of those routines, not
   counted in PTX. A T4 issues 4 warp instructions per SM per cycle at
   1.59 GHz — 8.1 × 10¹² thread-instructions/s — so 90 instructions a step is
   a ceiling near **90 G steps/s**, or 0.011 ns/cell. The bound above puts
   the kernel at 0.011–0.04 ns/cell, which is 28–100 % of peak issue: a
   normal figure for a dependent transcendental loop, and 290× away from the
   0.31 G steps/s measured.

**3.5 ns per four bytes is 1.1 GB/s, which is a transfer number.** Three
things on the host account for it, and none of them is on the CPU column's
path:

- **The copy back is pageable and synchronous.** `euler_kernel` ends in
  `clone_dtoh` into a freshly allocated `Vec`. That is the same naive path
  this file measured at **~3.9 GB/s** on a 24-core desktop; a two-vCPU Colab
  VM will not beat it. At 3.9 GB/s the three launches move 40.96, 204.8 and 163.8 MB
  in ~10.5, ~52 and ~42 ms — **28–32 %** of a 32.9, 186.1 and 141.8 ms wall.
  `clone_dtoh` also synchronises, so kernel and copy never overlap.
- **The destination is faulted in on the critical path.** A fresh `Vec` of
  40.96 MB is 10 000 first-touch page faults and 204.8 MB is 50 000, taken
  serially by the driver's write. This file already names that cost and
  pre-faults around it — for fGN. The engine does not.
- **Every element is copied a second time.** `try_euler_paths_map` on a
  device backend materialises the whole `m × n` matrix and then runs
  `buffer.assign(&row)` per path into a per-worker `Array1`. The host
  backend's `sample_map_chunked` does neither: it generates straight into one
  reused buffer per rayon worker and never allocates `m × n`. The CPU column
  is measured on a path that touches `workers × n` bytes of host memory and
  the CUDA column on one that allocates `m × n` and traverses it twice.

**What Metal gets for free.** Apple's memory is unified: the kernel writes
into a buffer the CPU already addresses, so there is no PCIe hop, no staging
and no fresh allocation — the engine's kept output buffer is the whole
transfer. Metal's *total* is 0.099 ns/cell, which is still 2.5–9× the T4's
kernel bound above; the T4's kernel is not the slower of the two. The 33× is
the copy and the assembly around it.

**And the Box-Muller asymmetry is real but not yet the bill.** The engine's
NVRTC call is a plain `compile_ptx` with no options, so `logf` and `cosf`
compile to the IEEE-accurate routines — tens of instructions apiece, with an
SFU op inside, on a Turing whose SFUs run at a quarter of its FP32 rate. The
Apple measurement puts the same three functions at 0–7 % of the step, which
is hard to square with an accurate `log`, so the two back-ends are probably
not compiling to the same strictness. Neither side pins it: the Metal
path passes `MTLCompileOptions::new()`, which is Apple's default math mode and
has changed across OS versions. Whichever way it is pinned, the CUDA half is
still half of a kernel that is under 2 % of the wall, so it explains none of
the gap today — and becomes the top kernel-side item the moment the copy
stops dominating.

### What is left, ranked

The two host-side items that led this list have landed; what follows is what
they left behind. Estimates are from arithmetic, not a profiler.

1. **Reduce on the device and return the reduction.** The wall is now the bus,
   and the bus cannot be made faster: 41 MB at 9.3 GB/s is 4.4 ms, and every
   step's four bytes have to cross. This shape is capped near **3 G steps/s**
   on a PCIe 3 card, and it already runs at 78–83 % of that. The only way
   past is to stop returning `m · n` scalars — a payoff, a running maximum, a
   Greek — computed in the kernel. *Design work, and the largest remaining
   item by far.*
2. **Overlap the copy with the launch.** `run` still synchronises between the
   kernel and the `dtoh`; `pipelined_paths` already alternates two streams,
   but only above the 1 GiB batch budget. Widening that gate would hide the
   crossing behind the next chunk's compute. Est. **1.2–1.4×** on the shapes
   here. *Needs a CUDA device to measure.*
3. **Pin the float-math mode on both back-ends, and give NVRTC the
   `__logf`/`__cosf` intrinsics or `--use_fast_math`** — est. **1.5–1.9× on
   the kernel**. Worth roughly nothing while the bus binds, and it costs the
   seed-for-seed agreement between the two back-ends, so it wants a revised
   parity tolerance rather than a flag flip. *Needs a CUDA device to measure.*
4. **Cache the pinned staging buffers in `pipelined_paths`.** Done for the
   single launch; the pipeline still calls `PinnedHost::alloc` twice per call,
   which at the default budget is a gigabyte pinned twice. Off every shape
   measured here. *Verifiable by reading; the cost of `cuMemHostAlloc` needs
   a device.*
5. **A coalesced output layout** — est. **under 1.1×**, and Apple measured
   the time-major alternative as *slower*. Lowest expected return.
   *Needs a CUDA device to measure.*

**When is the T4 worth it?** For `f32` and ten thousand paths or more,
**7–12×**. For `f64`, **3.2–3.8×** — the card is worth using in double
precision after all, which was not true a day ago. For fGN, 2.8–3.9×. The
number that no longer moves is the bus: at ten thousand paths over a thousand
steps the crossing alone is 4.4 ms of the 4.4 ms wall.

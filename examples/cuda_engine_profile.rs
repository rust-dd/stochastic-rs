//! What the Euler engine costs on a CUDA device, and why.
//!
//! ```text
//! cargo run --release --example cuda_engine_profile --features cuda
//! ```
//!
//! Prints three things, in the order they answer the question *is the GPU
//! worth using for this engine*:
//!
//! 1. **The kernel's own shape** — registers and local memory per thread, and
//!    the blocks per multiprocessor that follow. The engine compiles one
//!    monolithic kernel that carries every family's step, every optional
//!    frame block and the scratch they need, so a launch of any one family
//!    pays for all of them. If occupancy here is low, that is the reason, and
//!    a per-family kernel is the fix.
//! 2. **Throughput against the host**, in steps per second, over a few batch
//!    shapes and both precisions. On an Apple M4 Max the same measurement now
//!    gives 5–10 G steps/s on Metal against 2.3–2.7 on the CPU, where before
//!    the per-family kernels and the kept output buffer it gave 0.7–1.3.
//! 3. **The fractional pipeline for contrast** — fGN is FFT-shaped rather
//!    than recursion-shaped, and it is where the GPU already wins.
//!
//! Every timing is the best of five runs: a single run on a shared machine
//! measures the neighbours as much as the kernel.

use std::time::Instant;

use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stochastic::device::Cpu;
use stochastic_rs::stochastic::device::Cuda;
use stochastic_rs::stochastic::diffusion::gbm::Gbm;
use stochastic_rs::stochastic::euler::cuda::kernel_profile;
use stochastic_rs::stochastic::noise::fgn::Fgn;
use stochastic_rs::traits::ProcessExt;

/// Threads per block the engine launches with.
const BLOCK: u32 = 256;

/// The best of `reps` runs, in seconds.
fn best<T>(reps: usize, mut run: impl FnMut() -> T) -> f64 {
  let mut best = f64::MAX;
  for _ in 0..reps {
    let start = Instant::now();
    let _ = run();
    best = best.min(start.elapsed().as_secs_f64());
  }
  best
}

/// Steps per second for a batch of `m` paths over `n` points.
fn throughput(cells: f64, seconds: f64) -> f64 {
  cells / seconds / 1e9
}

fn main() {
  match stochastic_rs::stochastic::device::Backend::probe(&Cuda::default()) {
    Ok(info) => println!("device: {} ({:?})\n", info.name, info.precisions),
    Err(e) => {
      eprintln!("no CUDA device: {e}");
      return;
    }
  }

  println!("kernel as the driver sees it, at {BLOCK} threads per block");
  for real in ["float", "double"] {
    match kernel_profile(0, real, BLOCK) {
      Ok(p) => println!(
        "  {real:6}  {:4} registers/thread  {:6} B local/thread  {:5} B shared  \
         max block {:4}  {:2} blocks/SM  ({:3}% of a 2048-thread SM)",
        p.registers,
        p.local_bytes,
        p.shared_bytes,
        p.max_threads_per_block,
        p.blocks_per_multiprocessor,
        p.blocks_per_multiprocessor * BLOCK * 100 / 2048,
      ),
      Err(e) => println!("  {real}: {e}"),
    }
  }

  println!("\nEuler engine — GBM, best of five");
  println!(
    "  {:>7} {:>7} {:>6} {:>10} {:>10} {:>13} {:>13} {:>9}",
    "paths", "steps", "dtype", "cpu ms", "cuda ms", "cpu Gsteps/s", "gpu Gsteps/s", "speed-up"
  );
  for (m, n) in [(10_000usize, 1_024usize), (50_000, 1_024), (10_000, 4_096)] {
    let cells = (m * n) as f64;

    let build32 =
      || Gbm::<f32, _>::new(0.05, 0.2, n, Some(100.0), Some(1.0), Deterministic::new(7));
    let _ = build32().on::<Cuda>().sample_par(8);
    let host = best(5, || {
      build32().on::<Cpu>().sample_map(m, |p| p[p.len() - 1])
    });
    let device = best(5, || {
      build32().on::<Cuda>().sample_map(m, |p| p[p.len() - 1])
    });
    println!(
      "  {m:>7} {n:>7} {:>6} {:>10.1} {:>10.1} {:>13.2} {:>13.2} {:>8.2}x",
      "f32",
      host * 1e3,
      device * 1e3,
      throughput(cells, host),
      throughput(cells, device),
      host / device
    );

    // NVIDIA has native double precision, so the engine renders a `double`
    // kernel too — the ratio between the two says how much of the card's
    // f64 rate this workload actually uses.
    let build64 =
      || Gbm::<f64, _>::new(0.05, 0.2, n, Some(100.0), Some(1.0), Deterministic::new(7));
    let _ = build64().on::<Cuda>().sample_par(8);
    let host = best(5, || {
      build64().on::<Cpu>().sample_map(m, |p| p[p.len() - 1])
    });
    let device = best(5, || {
      build64().on::<Cuda>().sample_map(m, |p| p[p.len() - 1])
    });
    println!(
      "  {m:>7} {n:>7} {:>6} {:>10.1} {:>10.1} {:>13.2} {:>13.2} {:>8.2}x",
      "f64",
      host * 1e3,
      device * 1e3,
      throughput(cells, host),
      throughput(cells, device),
      host / device
    );
  }

  println!("\nfractional noise — fGN, best of five (the FFT pipeline, not the engine)");
  println!(
    "  {:>7} {:>7} {:>10} {:>10} {:>9}",
    "paths", "points", "cpu ms", "cuda ms", "speed-up"
  );
  for (m, n) in [(1_000usize, 4_096usize), (1_000, 16_384), (10_000, 4_096)] {
    let build = || Fgn::<f32, _>::new(0.7, n, Some(1.0), Deterministic::new(11));
    let _ = build().on::<Cuda>().sample_par(8);
    let host = best(5, || build().on::<Cpu>().sample_par(m));
    let device = best(5, || build().on::<Cuda>().sample_par(m));
    println!(
      "  {m:>7} {n:>7} {:>10.1} {:>10.1} {:>8.2}x",
      host * 1e3,
      device * 1e3,
      host / device
    );
  }

  println!(
    "\nWhat to read: local memory per thread should now be zero for a plain \
     diffusion — the kernel is rendered for one family and declares no scratch \
     it cannot reach — and the blocks per multiprocessor should be what the \
     registers alone allow. A T4 reported 80 registers and 3552 bytes of local \
     memory before that change, at 37% occupancy and 0.15 G steps/s against \
     0.21 on its host. The same three changes took an M4 Max from 0.70 to \
     10.10 G steps/s at 200k paths, 3.7x its CPU."
  );
}

//! `Accelerate` is reproducible-effort only: seed consumption ignores the thread count, but
//! `vDSP_fft_zip` is not bit-stable under core contention (21 of 400 runs, worst 2.08e-3 relative).

#[cfg(accelerate_backend)]
mod accelerate_reproducibility {
  use std::sync::Arc;
  use std::sync::atomic::AtomicBool;
  use std::sync::atomic::Ordering;

  use ndarray::Array1;
  use rayon::ThreadPoolBuilder;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_stochastic::device::Accelerate;
  use stochastic_rs_stochastic::device::Cpu;
  use stochastic_rs_stochastic::noise::fgn::Fgn;
  use stochastic_rs_stochastic::traits::ProcessExt;

  const SEED: u64 = 42;

  /// Relative-difference bound: above the worst scheduling noise measured (`2.08e-3`), orders of
  /// magnitude below what a seeding bug such as two paths swapping bases would produce.
  const VDSP_NOISE_TOLERANCE: f64 = 1e-2;

  fn fgn_accelerate(seed: u64, n: usize) -> Fgn<f32, Deterministic, Accelerate> {
    Fgn::<f32, _>::new(0.7, n, Some(1.0), Deterministic::new(seed)).on::<Accelerate>()
  }

  fn fgn_cpu(seed: u64, n: usize) -> Fgn<f32, Deterministic, Cpu> {
    Fgn::<f32, _>::new(0.7, n, Some(1.0), Deterministic::new(seed))
  }

  fn to_vecs(paths: Vec<Array1<f32>>) -> Vec<Vec<f32>> {
    paths.into_iter().map(|p| p.to_vec()).collect()
  }

  /// Max relative difference and count of bit-differing elements between
  /// two equal-shaped path sets.
  fn max_rel_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> (f64, usize, usize) {
    let mut max_rel = 0.0f64;
    let mut diff_count = 0usize;
    let mut total = 0usize;
    for (pa, pb) in a.iter().zip(b.iter()) {
      for (&x, &y) in pa.iter().zip(pb.iter()) {
        total += 1;
        if x.to_bits() != y.to_bits() {
          diff_count += 1;
        }
        let denom = (x.abs() as f64).max(y.abs() as f64).max(1e-9);
        let rel = ((x - y).abs() as f64) / denom;
        max_rel = max_rel.max(rel);
      }
    }
    (max_rel, diff_count, total)
  }

  /// Runs `f` while several threads spin on floating-point work, so the scheduler dispatches
  /// across core types; a quiet system never reproduced the divergence, contention did.
  fn under_core_contention<R>(f: impl FnOnce() -> R) -> R {
    let stop = Arc::new(AtomicBool::new(false));
    let handles: Vec<_> = (0..12)
      .map(|_| {
        let stop = stop.clone();
        std::thread::spawn(move || {
          let mut x = 1.0f64;
          while !stop.load(Ordering::Relaxed) {
            for _ in 0..1000 {
              x = x.sin().cos() + 1.000_000_1;
            }
          }
          std::hint::black_box(x);
        })
      })
      .collect();
    let out = f();
    stop.store(true, Ordering::Relaxed);
    for h in handles {
      let _ = h.join();
    }
    out
  }

  /// Seed consumption, which derived basis feeds which path, is thread-count independent within
  /// the tolerance vDSP's nondeterminism costs; a seeding regression exceeds it by far.
  #[test]
  fn accelerate_seed_consumption_is_thread_count_independent_within_vdsp_tolerance() {
    let m = 64;
    let n = 1024;
    let pool1 = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let pool8 = ThreadPoolBuilder::new().num_threads(8).build().unwrap();

    let (a, b) = under_core_contention(|| {
      let a = to_vecs(pool1.install(|| fgn_accelerate(SEED, n).sample_par(m)));
      let b = to_vecs(pool8.install(|| fgn_accelerate(SEED, n).sample_par(m)));
      (a, b)
    });

    let (max_rel, diffs, total) = max_rel_diff(&a, &b);
    assert!(
      max_rel < VDSP_NOISE_TOLERANCE,
      "Accelerate sample_par diverged between 1 and 8 threads far beyond vDSP's own \
       measured noise floor (see module doc): max relative difference {max_rel}, \
       {diffs} of {total} elements differ"
    );
  }

  /// The same over the `(n, m)` sweep that exposed the divergence: two identically seeded
  /// objects on one pool under core contention, which isolates vDSP's run-to-run stability.
  #[test]
  fn accelerate_repeated_calls_agree_within_vdsp_tolerance_under_core_contention() {
    let mut worst = 0.0f64;
    under_core_contention(|| {
      for i in 0..40 {
        let n = 256 + (i % 7) * 128;
        let m = 8 + (i % 5) * 16;
        let a = to_vecs(fgn_accelerate(SEED, n).sample_par(m));
        let b = to_vecs(fgn_accelerate(SEED, n).sample_par(m));
        let (max_rel, _diffs, _total) = max_rel_diff(&a, &b);
        worst = worst.max(max_rel);
      }
    });
    assert!(
      worst < VDSP_NOISE_TOLERANCE,
      "Accelerate's own repeated-call divergence exceeded the tolerance vDSP's \
       measured hardware nondeterminism costs: worst max relative difference {worst}"
    );
  }

  /// Control: `Cpu` under the same contention harness and sweep stays bit-identical, so the
  /// divergence is vDSP's and not the harness's.
  #[test]
  fn cpu_repeated_calls_are_bit_identical_under_the_same_core_contention() {
    under_core_contention(|| {
      for i in 0..40 {
        let n = 256 + (i % 7) * 128;
        let m = 8 + (i % 5) * 16;
        let a = to_vecs(fgn_cpu(SEED, n).sample_par(m));
        let b = to_vecs(fgn_cpu(SEED, n).sample_par(m));
        let (max_rel, diffs, _total) = max_rel_diff(&a, &b);
        assert_eq!(
          (max_rel, diffs),
          (0.0, 0),
          "Cpu diverged under core contention at n={n}, m={m} — this would mean the \
           measurement harness itself is unreliable, not that Cpu regressed"
        );
      }
    });
  }
}

//! RNG-behaviour tests for the sampler refactor: `Deterministic`
//! reproducibility, `Unseeded` variability, and path distinctness ("the seed
//! never gets stuck") — including across the thread-local block-seed boundary.

use std::collections::HashSet;

use ndarray::Array1;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::simd_rng::Unseeded;
use stochastic_rs::stochastic::diffusion::gbm::Gbm;
use stochastic_rs::stochastic::volatility::HestonPow;
use stochastic_rs::stochastic::volatility::heston::Heston;
use stochastic_rs::traits::PathSampler;
use stochastic_rs::traits::ProcessExt;

fn key(p: &Array1<f64>) -> Vec<u64> {
  p.iter().map(|x| x.to_bits()).collect()
}

fn gbm<S: stochastic_rs::simd_rng::SeedExt>(n: usize, seed: S) -> Gbm<f64, S> {
  Gbm::<f64, _>::new(0.05, 0.2, n, Some(1.0), Some(1.0), seed)
}

#[test]
fn deterministic_single_path_reproducible() {
  // Same seed ⇒ identical path.
  let a = gbm(48, Deterministic::new(123));
  let b = gbm(48, Deterministic::new(123));
  assert_eq!(key(&a.sample()), key(&b.sample()));
  // Different seed ⇒ different path.
  let c = gbm(48, Deterministic::new(124));
  assert_ne!(key(&a.sample()), key(&c.sample()));
}

#[test]
fn deterministic_serial_stream_reproducible() {
  // Two identically-seeded processes produce the SAME sequence of paths when
  // sampled serially (each `sample()` advances the deterministic seed).
  let a = gbm(32, Deterministic::new(7));
  let b = gbm(32, Deterministic::new(7));
  let seq_a: Vec<_> = (0..6).map(|_| key(&a.sample())).collect();
  let seq_b: Vec<_> = (0..6).map(|_| key(&b.sample())).collect();
  assert_eq!(seq_a, seq_b);
  // …and within that sequence every path is distinct (the seed advances, it
  // does not get stuck on one stream).
  let distinct: HashSet<_> = seq_a.iter().cloned().collect();
  assert_eq!(distinct.len(), 6, "deterministic serial paths repeated");
}

#[test]
fn deterministic_sampler_reuse_paths_distinct() {
  // A reused sampler must yield a fresh, distinct path on every call.
  let g = gbm(32, Deterministic::new(99));
  let mut s = g.sampler();
  let mut seen = HashSet::new();
  let mut buf = s.sample();
  seen.insert(key(&buf));
  for _ in 0..999 {
    s.sample_into(&mut buf);
    assert!(seen.insert(key(&buf)), "reused sampler repeated a path");
  }
  assert_eq!(seen.len(), 1000);
}

#[test]
fn unseeded_paths_vary() {
  // Unseeded: consecutive single samples differ, and two independent
  // instances differ.
  let g = gbm(48, Unseeded);
  let p1 = key(&g.sample());
  let p2 = key(&g.sample());
  assert_ne!(p1, p2, "unseeded repeated a path");
  let h = gbm(48, Unseeded);
  assert_ne!(p1, key(&h.sample()), "two unseeded instances coincided");
}

#[test]
fn sample_par_paths_all_distinct() {
  // Across rayon workers (each with its own derived seed) every kept path is
  // distinct — no two workers share a stream.
  let g = gbm(32, Unseeded);
  let paths = g.sample_par(4000);
  let set: HashSet<_> = paths.iter().map(key).collect();
  assert_eq!(set.len(), 4000, "sample_par produced duplicate paths");
}

#[test]
fn sample_map_paths_all_distinct() {
  // The buffer-reusing fold must still visit 4000 distinct realisations.
  let g = gbm(32, Unseeded);
  let keys = g.sample_map(4000, key);
  let set: HashSet<_> = keys.into_iter().collect();
  assert_eq!(set.len(), 4000, "sample_map produced duplicate paths");
}

#[test]
fn seed_not_stuck_across_block_boundary() {
  // SEED_BLOCK_LEN = 1<<18 = 262_144. Constructing more auto-seeded RNGs than
  // one block on a single thread forces a re-reservation; every path must
  // still be unique (the block scheme hands out disjoint seeds, it never
  // sticks or repeats at the boundary).
  let count = 300_000usize;
  let g = gbm(4, Unseeded);
  let mut seen = HashSet::with_capacity(count);
  for _ in 0..count {
    assert!(
      seen.insert(key(&g.sample())),
      "auto-seed collided / got stuck"
    );
  }
  assert_eq!(seen.len(), count);
}

#[test]
fn multistate_sample_par_paths_distinct() {
  // The same guarantees on a 2-state model (Heston): distinct asset paths.
  let h = Heston::<f64, _>::new(
    Some(1.0),
    Some(0.04),
    2.0,
    0.04,
    0.3,
    -0.7,
    0.05,
    32,
    Some(1.0),
    HestonPow::Sqrt,
    None,
    Unseeded,
  );
  let runs = h.sample_par(2000);
  let set: HashSet<_> = runs.iter().map(|[s, _v]| key(s)).collect();
  assert_eq!(
    set.len(),
    2000,
    "Heston sample_par produced duplicate asset paths"
  );
}

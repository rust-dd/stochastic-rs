//! The bit-identity promise, held under threads fighting for the machine.
//!
//! Twice this week a comparison involving the fractional OU disagreed with
//! itself — the same seed, the same backend, two calls that should be bit
//! identical, differing in one or two of four thousand paths — and only while
//! the machine was busy. Neither sighting reproduced on an idle machine, and
//! the suites that saw it are expensive: the one that failed runs a hundred
//! and sixty comparisons and takes minutes on a small runtime, so "run it a
//! hundred times" is not a diagnosis anyone can afford.
//!
//! This is the cheap version of the same question. Several threads each
//! resample the same process from the same seed, over and over, and every
//! result is compared against a reference taken once. That is the contention
//! the sightings had — rayon workers competing, and for the fractional
//! processes a nested rayon region inside each of them, since the fGN
//! pipeline's FFT is itself parallel — without the cost of the suite around
//! it.
//!
//! A failure here is a real bug in `ProcessExt`'s guarantee. Passing does not
//! prove the guarantee holds; it removes the cheapest explanation for the two
//! sightings, which is what a test can do.

use std::sync::Arc;
use std::thread;

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::fou::Fou;
use stochastic_rs_stochastic::diffusion::gbm::Gbm;
use stochastic_rs_stochastic::diffusion::ou::Ou;
use stochastic_rs_stochastic::euler::Reduce;
use stochastic_rs_stochastic::noise::fgn::Fgn;
use stochastic_rs_stochastic::traits::ProcessExt;

/// Paths a round samples. Small enough that a round is milliseconds, large
/// enough that it crosses the chunk boundaries `chunk_count` draws.
const M: usize = 512;

/// Grid points a path takes.
const N: usize = 253;

/// Threads competing. More than the cores of a small runtime on purpose: the
/// sightings were under oversubscription, not under a tidy one-per-core.
const THREADS: usize = 8;

/// Rounds each thread runs.
const ROUNDS: usize = 12;

/// Every thread, every round, reproduces `reference` exactly.
fn holds<T>(name: &'static str, build: T, reference: Vec<Array1<f32>>)
where
  T: Fn() -> Vec<Array1<f32>> + Send + Sync + 'static,
{
  let build = Arc::new(build);
  let reference = Arc::new(reference);
  let workers: Vec<_> = (0..THREADS)
    .map(|worker| {
      let build = Arc::clone(&build);
      let reference = Arc::clone(&reference);
      thread::spawn(move || {
        for round in 0..ROUNDS {
          let paths = build();
          assert_eq!(
            paths.len(),
            reference.len(),
            "{name}: worker {worker} round {round} returned {} paths, not {}",
            paths.len(),
            reference.len()
          );
          for (i, (got, want)) in paths.iter().zip(reference.iter()).enumerate() {
            assert_eq!(
              got, want,
              "{name}: worker {worker} round {round} path {i} is not the seed's path"
            );
          }
        }
      })
    })
    .collect();
  for worker in workers {
    worker.join().expect("a worker panicked");
  }
}

/// The fractional OU, which is what both sightings involved.
///
/// Its host sampler draws through the fGN pipeline, whose FFT is a rayon
/// parallel call made from inside whatever rayon region the batch is already
/// in — the one piece of nested parallelism in the crate's host path, and the
/// first place to look for an effect that only appears under load.
#[test]
fn the_fractional_ou_reproduces_its_seed_under_contention() {
  let build = || {
    Fou::<f32, _>::new(
      0.7,
      2.0,
      1.0,
      0.3,
      N,
      Some(0.0),
      Some(1.0),
      Deterministic::new(9),
    )
  };
  let reference = build().sample_par(M);
  holds("fractional OU", move || build().sample_par(M), reference);
}

/// The pipeline underneath it, on its own.
#[test]
fn the_fractional_noise_reproduces_its_seed_under_contention() {
  let build = || Fgn::<f32, _>::new(0.7, N, Some(1.0), Deterministic::new(11));
  let reference = build().sample_par(M);
  holds("fGN", move || build().sample_par(M), reference);
}

/// A plain diffusion, as the control: it shares the chunking and the rayon
/// region but draws its noise inline, with no nested parallel call. If this
/// one moves too, the cause is in the chunking rather than in the pipeline.
#[test]
fn a_plain_diffusion_reproduces_its_seed_under_contention() {
  let build = || Gbm::<f32, _>::new(0.05, 0.2, N, Some(100.0), Some(1.0), Deterministic::new(3));
  let reference = build().sample_par(M);
  holds("GBM", move || build().sample_par(M), reference);
}

/// The mean-reverting diffusion, which is the fractional one's host recursion
/// without the fractional noise.
#[test]
fn the_mean_reverting_diffusion_reproduces_its_seed_under_contention() {
  let build = || {
    Ou::<f32, _>::new(
      2.0,
      1.0,
      0.3,
      N,
      Some(0.0),
      Some(1.0),
      Deterministic::new(9),
    )
  };
  let reference = build().sample_par(M);
  holds("OU", move || build().sample_par(M), reference);
}

/// The folded batch, which is the call the second sighting was inside.
///
/// A fold is one value a path rather than a whole grid, so a difference here
/// is a difference in the paths behind it — the same question asked where it
/// was actually seen to fail.
#[test]
fn a_folded_batch_reproduces_its_seed_under_contention() {
  let build = || {
    Fou::<f32, _>::new(
      0.7,
      2.0,
      1.0,
      0.3,
      N,
      Some(0.0),
      Some(1.0),
      Deterministic::new(9),
    )
  };
  let reference = build().sample_reduce(M, Reduce::Terminal);
  let build = Arc::new(build);
  let reference = Arc::new(reference);
  let workers: Vec<_> = (0..THREADS)
    .map(|worker| {
      let build = Arc::clone(&build);
      let reference = Arc::clone(&reference);
      thread::spawn(move || {
        for round in 0..ROUNDS {
          assert_eq!(
            build().sample_reduce(M, Reduce::Terminal),
            *reference,
            "folded batch: worker {worker} round {round} is not the seed's fold"
          );
        }
      })
    })
    .collect();
  for worker in workers {
    worker.join().expect("a worker panicked");
  }
}

//! The bit-identity promise under contention: several threads resample one process from one seed
//! against a reference taken once. A failure is a bug in `ProcessExt`'s guarantee; a pass no proof.

use std::sync::Arc;
use std::thread;

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::fou::Fou;
use stochastic_rs_stochastic::diffusion::gbm::Gbm;
use stochastic_rs_stochastic::diffusion::ou::Ou;
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

/// The fractional OU: its host sampler draws through the fGN pipeline, whose FFT is a rayon call
/// made inside the batch's rayon region, the one nested parallelism in the crate's host path.
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

/// A plain diffusion as the control: the same chunking and rayon region, noise drawn inline. If
/// this one moves too, the cause is in the chunking, not in the pipeline.
#[test]
fn a_plain_diffusion_reproduces_its_seed_under_contention() {
  let build = || Gbm::<f32, _>::new(0.05, 0.2, N, 100.0, Some(1.0), Deterministic::new(3));
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

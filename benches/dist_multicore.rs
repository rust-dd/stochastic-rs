use std::hint::black_box;
use std::time::Instant;

use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;
use stochastic_rs::distributions::DistributionSampler;
use stochastic_rs::distributions::exp::SimdExp;
use stochastic_rs::distributions::normal::SimdNormal;
use stochastic_rs::distributions::poisson::SimdPoisson;

fn median_ms(samples: &mut [f64]) -> f64 {
  samples.sort_by(f64::total_cmp);
  samples[samples.len() / 2]
}

fn bench_pool<T, D>(
  pool: &ThreadPool,
  dist: &D,
  m: usize,
  n: usize,
  warmup: usize,
  runs: usize,
) -> f64
where
  D: DistributionSampler<T> + Clone + Send,
  T: Default + Clone + Send,
{
  for _ in 0..warmup {
    let dist_run = dist.clone();
    pool.install(move || {
      let out = dist_run.sample_matrix(m, n);
      black_box(out);
    });
  }

  let mut times_ms = Vec::with_capacity(runs);
  for _ in 0..runs {
    let dist_run = dist.clone();
    let t0 = Instant::now();
    pool.install(move || {
      let out = dist_run.sample_matrix(m, n);
      black_box(out);
    });
    times_ms.push(t0.elapsed().as_secs_f64() * 1_000.0);
  }
  median_ms(&mut times_ms)
}

fn run_case<T, D>(name: &str, dist: &D, m: usize, n: usize, single: &ThreadPool, multi: &ThreadPool)
where
  D: DistributionSampler<T> + Clone + Send,
  T: Default + Clone + Send,
{
  let warmup = 2;
  let runs = 7;
  let t1 = bench_pool(single, dist, m, n, warmup, runs);
  let tn = bench_pool(multi, dist, m, n, warmup, runs);
  let speedup = t1 / tn;
  let values = m * n;
  println!(
    "{name:>12} | m={m:<5} n={n:<5} values={values:<10} | 1T={t1:>8.2} ms | MT={tn:>8.2} ms | speedup={speedup:>5.2}x"
  );
}

fn main() {
  let threads = std::thread::available_parallelism()
    .map(|v| v.get())
    .unwrap_or(1);
  let mt_threads = threads.max(2);
  let single = ThreadPoolBuilder::new()
    .num_threads(1)
    .build()
    .expect("failed to build single-thread pool");
  let multi = ThreadPoolBuilder::new()
    .num_threads(mt_threads)
    .build()
    .expect("failed to build multi-thread pool");

  println!("Distribution sample_matrix benchmark");
  println!("Using MT threads: {mt_threads}");
  println!();

  run_case(
    "Normal<f64>",
    &SimdNormal::<f64>::new(0.0, 1.0),
    2048,
    2048,
    &single,
    &multi,
  );
  run_case(
    "Exp<f64>",
    &SimdExp::<f64>::new(1.5),
    2048,
    2048,
    &single,
    &multi,
  );
  run_case(
    "Poisson<i64>",
    &SimdPoisson::<i64>::new(1.5),
    2048,
    2048,
    &single,
    &multi,
  );
}

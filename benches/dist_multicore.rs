use std::hint::black_box;
use std::time::Instant;

use rand_distr::Distribution;
use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;
use stochastic_rs::distributions::DistributionSampler;
use stochastic_rs::distributions::alpha_stable::SimdAlphaStable;
use stochastic_rs::distributions::beta::SimdBeta;
use stochastic_rs::distributions::binomial::SimdBinomial;
use stochastic_rs::distributions::cauchy::SimdCauchy;
use stochastic_rs::distributions::chi_square::SimdChiSquared;
use stochastic_rs::distributions::exp::SimdExp;
use stochastic_rs::distributions::gamma::SimdGamma;
use stochastic_rs::distributions::geometric::SimdGeometric;
use stochastic_rs::distributions::hypergeometric::SimdHypergeometric;
use stochastic_rs::distributions::inverse_gauss::SimdInverseGauss;
use stochastic_rs::distributions::lognormal::SimdLogNormal;
use stochastic_rs::distributions::normal::SimdNormal;
use stochastic_rs::distributions::normal_inverse_gauss::SimdNormalInverseGauss;
use stochastic_rs::distributions::pareto::SimdPareto;
use stochastic_rs::distributions::poisson::SimdPoisson;
use stochastic_rs::distributions::studentt::SimdStudentT;
use stochastic_rs::distributions::uniform::SimdUniform;
use stochastic_rs::distributions::weibull::SimdWeibull;
use stochastic_rs::simd_rng::SimdRng;

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

fn bench_normal_fill_slice(n: usize, warmup: usize, runs: usize) -> (f64, f64, f64) {
  let simd = SimdNormal::<f64>::new(0.0, 1.0);
  let rand_distr = rand_distr::Normal::<f64>::new(0.0, 1.0).expect("valid normal params");
  let mut out = vec![0.0f64; n];
  let iters = (262_144 / n.max(1)).clamp(1, 16_384);

  let mut simd_rng = SimdRng::new();
  for _ in 0..warmup {
    for _ in 0..iters {
      simd.fill_slice(&mut simd_rng, &mut out);
    }
    black_box(&out);
  }
  let mut simd_times_ms = Vec::with_capacity(runs);
  for _ in 0..runs {
    let t0 = Instant::now();
    for _ in 0..iters {
      simd.fill_slice(&mut simd_rng, &mut out);
    }
    black_box(&out);
    simd_times_ms.push(t0.elapsed().as_secs_f64() * 1_000.0 / iters as f64);
  }

  let mut base_rng = SimdRng::new();
  for _ in 0..warmup {
    for _ in 0..iters {
      for x in &mut out {
        *x = rand_distr.sample(&mut base_rng);
      }
    }
    black_box(&out);
  }
  let mut base_times_ms = Vec::with_capacity(runs);
  for _ in 0..runs {
    let t0 = Instant::now();
    for _ in 0..iters {
      for x in &mut out {
        *x = rand_distr.sample(&mut base_rng);
      }
    }
    black_box(&out);
    base_times_ms.push(t0.elapsed().as_secs_f64() * 1_000.0 / iters as f64);
  }

  let mut base_thread_rng = rand::rng();
  for _ in 0..warmup {
    for _ in 0..iters {
      for x in &mut out {
        *x = rand_distr.sample(&mut base_thread_rng);
      }
    }
    black_box(&out);
  }
  let mut base_thread_times_ms = Vec::with_capacity(runs);
  for _ in 0..runs {
    let t0 = Instant::now();
    for _ in 0..iters {
      for x in &mut out {
        *x = rand_distr.sample(&mut base_thread_rng);
      }
    }
    black_box(&out);
    base_thread_times_ms.push(t0.elapsed().as_secs_f64() * 1_000.0 / iters as f64);
  }

  (
    median_ms(&mut simd_times_ms),
    median_ms(&mut base_times_ms),
    median_ms(&mut base_thread_times_ms),
  )
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
    "{name:>18} | m={m:<5} n={n:<5} values={values:<10} | 1T={t1:>8.2} ms | MT={tn:>8.2} ms | speedup={speedup:>5.2}x"
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
    "Normal<f64>(ref)",
    &SimdNormal::<f64>::new(0.0, 1.0),
    2048,
    2048,
    &single,
    &multi,
  );
  run_case(
    "Exp<f64>(ref)",
    &SimdExp::<f64>::new(1.5),
    2048,
    2048,
    &single,
    &multi,
  );
  run_case(
    "Poisson<i64>(ref)",
    &SimdPoisson::<i64>::new(1.5),
    2048,
    2048,
    &single,
    &multi,
  );

  println!();
  println!("All built-in distributions sample_matrix benchmark");
  println!("(sizes tuned to keep runtime practical for heavier discrete samplers)");
  let fm = 1024usize;
  let fnn = 1024usize;
  let im = 512usize;
  let inn = 512usize;

  run_case(
    "Normal<f64>",
    &SimdNormal::<f64>::new(0.0, 1.0),
    fm,
    fnn,
    &single,
    &multi,
  );
  run_case(
    "Exp<f64>",
    &SimdExp::<f64>::new(1.5),
    fm,
    fnn,
    &single,
    &multi,
  );
  run_case(
    "Uniform<f64>",
    &SimdUniform::<f64>::new(0.0, 1.0),
    fm,
    fnn,
    &single,
    &multi,
  );
  run_case(
    "Cauchy<f64>",
    &SimdCauchy::<f64>::new(0.0, 1.0),
    fm,
    fnn,
    &single,
    &multi,
  );
  run_case(
    "LogNormal<f64>",
    &SimdLogNormal::<f64>::new(0.2, 0.8),
    fm,
    fnn,
    &single,
    &multi,
  );
  run_case(
    "Gamma<f64>",
    &SimdGamma::<f64>::new(2.0, 2.0),
    fm,
    fnn,
    &single,
    &multi,
  );
  run_case(
    "ChiSq<f64>",
    &SimdChiSquared::<f64>::new(5.0),
    fm,
    fnn,
    &single,
    &multi,
  );
  run_case(
    "StudentT<f64>",
    &SimdStudentT::<f64>::new(5.0),
    fm,
    fnn,
    &single,
    &multi,
  );
  run_case(
    "Beta<f64>",
    &SimdBeta::<f64>::new(2.0, 2.0),
    fm,
    fnn,
    &single,
    &multi,
  );
  run_case(
    "Weibull<f64>",
    &SimdWeibull::<f64>::new(1.0, 1.5),
    fm,
    fnn,
    &single,
    &multi,
  );
  run_case(
    "Pareto<f64>",
    &SimdPareto::<f64>::new(1.0, 1.5),
    fm,
    fnn,
    &single,
    &multi,
  );
  run_case(
    "InvGauss<f64>",
    &SimdInverseGauss::<f64>::new(1.0, 2.0),
    fm,
    fnn,
    &single,
    &multi,
  );
  run_case(
    "NIG<f64>",
    &SimdNormalInverseGauss::<f64>::new(2.0, 0.5, 1.0, 0.0),
    fm,
    fnn,
    &single,
    &multi,
  );
  run_case(
    "AlphaStable<f64>",
    &SimdAlphaStable::<f64>::new(1.7, 0.3, 1.0, 0.0),
    fm,
    fnn,
    &single,
    &multi,
  );

  run_case(
    "Poisson<i64>",
    &SimdPoisson::<i64>::new(2.5),
    fm,
    fnn,
    &single,
    &multi,
  );
  run_case(
    "Geometric<u64>",
    &SimdGeometric::<u64>::new(0.3),
    fm,
    fnn,
    &single,
    &multi,
  );
  run_case(
    "Binomial<u32>",
    &SimdBinomial::<u32>::new(32, 0.3),
    im,
    inn,
    &single,
    &multi,
  );
  run_case(
    "Hypergeo<u32>",
    &SimdHypergeometric::<u32>::new(500, 80, 32),
    im,
    inn,
    &single,
    &multi,
  );

  println!();
  println!("Normal fill_slice benchmark (single-thread)");
  println!("Reference A: rand_distr + SimdRng (fair algorithm compare)");
  println!("Reference B: rand_distr + rand::rng() (out-of-box baseline)");
  for &n in &[4usize, 8, 16, 64, 256, 4096, 65_536] {
    let (simd_ms, base_simd_ms, base_thread_ms) = bench_normal_fill_slice(n, 2, 9);
    let speedup_a = base_simd_ms / simd_ms;
    let speedup_b = base_thread_ms / simd_ms;
    let simd_us = simd_ms * 1_000.0;
    let base_simd_us = base_simd_ms * 1_000.0;
    let base_thread_us = base_thread_ms * 1_000.0;
    println!(
      "{:>12} | n={n:<6} | simd={simd_us:>9.3} us | rd+simd_rng={base_simd_us:>9.3} us ({speedup_a:>5.2}x) | rd+rand_rng={base_thread_us:>9.3} us ({speedup_b:>5.2}x)",
      "Normal<f64>"
    );
  }
}

use std::hint::black_box;
use std::time::Duration;

use criterion::Criterion;
use rand::distr::Distribution;
use stochastic_rs::distributions::DistributionSampler;
use stochastic_rs::distributions::SimdDistribution;
use stochastic_rs::distributions::truncated::SimdTruncatedBeta;
use stochastic_rs::distributions::truncated::SimdTruncatedExp;
use stochastic_rs::distributions::truncated::SimdTruncatedGamma;
use stochastic_rs::distributions::truncated::SimdTruncatedNormal;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::simd_rng::SimdRng;
use stochastic_rs::traits::DistributionExt;

/// The truncated laws' rows of `DistributionExt`: the per-point cost of a density, cdf, quantile, mgf or cf.
pub fn bench_distribution_ext(c: &mut Criterion) {
  let mut group = c.benchmark_group("DistributionExt");
  group.measurement_time(Duration::from_secs(3));
  group.warm_up_time(Duration::from_millis(500));
  let xs = (0..4096)
    .map(|i| -4.0 + 8.0 * i as f64 / 4095.0)
    .collect::<Vec<_>>();
  let ps = (1..=4096).map(|i| i as f64 / 4097.0).collect::<Vec<_>>();
  let truncated_normal = SimdTruncatedNormal::<f64>::new(0.5, 2.0, -1.0, 2.0);
  let truncated_gamma = SimdTruncatedGamma::<f64>::new(2.5, 1.2, 0.8, 4.0);
  let truncated_beta = SimdTruncatedBeta::<f64>::new(2.0, 3.0, 0.2, 0.7);
  let truncated_exp = SimdTruncatedExp::<f64>::new(1.5, 0.5, 2.5);
  // `xs` mapped onto the law's `[-1, 2]`.
  group.bench_function("pdf/truncated_normal", |b| {
    b.iter(|| {
      black_box(
        xs.iter()
          .map(|&x| truncated_normal.pdf(0.5 + 0.375 * x).unwrap())
          .sum::<f64>(),
      )
    })
  });
  group.bench_function("cdf/truncated_normal", |b| {
    b.iter(|| {
      black_box(
        xs.iter()
          .map(|&x| truncated_normal.cdf(0.5 + 0.375 * x).unwrap())
          .sum::<f64>(),
      )
    })
  });
  group.bench_function("quantile/truncated_normal", |b| {
    b.iter(|| {
      black_box(
        ps.iter()
          .map(|&p| truncated_normal.quantile(p).unwrap())
          .sum::<f64>(),
      )
    })
  });
  // `xs` mapped onto `t ∈ [−2, 2]`.
  group.bench_function("mgf/truncated_normal", |b| {
    b.iter(|| {
      black_box(
        xs.iter()
          .map(|&x| {
            truncated_normal
              .moment_generating_function(0.5 * x)
              .unwrap()
          })
          .sum::<f64>(),
      )
    })
  });
  // `xs` mapped onto `t ∈ [−2, 2]` on the bounded TE(1.5, 0.5, 2.5).
  group.bench_function("cf/truncated_exp", |b| {
    b.iter(|| {
      black_box(
        xs.iter()
          .map(|&x| truncated_exp.characteristic_function(0.5 * x).unwrap().re)
          .sum::<f64>(),
      )
    })
  });
  // `xs` mapped onto `[0.8, 4]` and `[0.2, 0.7]`.
  group.bench_function("pdf/truncated_gamma", |b| {
    b.iter(|| {
      black_box(
        xs.iter()
          .map(|&x| truncated_gamma.pdf(2.4 + 0.4 * x).unwrap())
          .sum::<f64>(),
      )
    })
  });
  group.bench_function("pdf/truncated_beta", |b| {
    b.iter(|| {
      black_box(
        xs.iter()
          .map(|&x| truncated_beta.pdf(0.45 + 0.0625 * x).unwrap())
          .sum::<f64>(),
      )
    })
  });
  group.finish();
}

pub fn bench_truncated_exp(c: &mut Criterion) {
  let mut group = c.benchmark_group("TruncatedExp");
  let law = SimdTruncatedExp::<f64>::new(2.0, 0.0, 1.5);
  group.bench_function("seeded/f64", |b| {
    let mut s = law.seeded(&Deterministic::new(7));
    b.iter(|| black_box(s.sample()))
  });
  group.bench_function("fill_slice/f64/4096", |b| {
    let mut s = law.seeded(&Deterministic::new(7));
    let mut out = vec![0.0_f64; 4096];
    b.iter(|| {
      s.fill_slice(&mut out);
      black_box(&out);
    })
  });
  group.finish();
}

/// The rejection regime: `[1, 5]` holds 0.70 of the Gamma(2, 1) mass; `[45, 60]` holds 9e-15 of Gamma(2.5, 1.2)'s.
pub fn bench_truncated_gamma(c: &mut Criterion) {
  let mut group = c.benchmark_group("TruncatedGamma");
  let law = SimdTruncatedGamma::<f64>::new(2.0, 1.0, 1.0, 5.0);
  group.bench_function("seeded/f64", |b| {
    let mut s = law.clone().seeded(&Deterministic::new(7));
    b.iter(|| black_box(s.sample()))
  });
  let far = SimdTruncatedGamma::<f64>::new(2.5, 1.2, 45.0, 60.0);
  group.bench_function("inversion/f64", |b| {
    let mut s = far.clone().seeded(&Deterministic::new(7));
    b.iter(|| black_box(s.sample()))
  });
  group.finish();
}

/// The inverse-cdf regime: `[0.99, 0.999]` holds 4e-6 of the Beta(2, 3) mass.
pub fn bench_truncated_beta(c: &mut Criterion) {
  let mut group = c.benchmark_group("TruncatedBeta");
  let far = SimdTruncatedBeta::<f64>::new(2.0, 3.0, 0.99, 0.999);
  group.bench_function("inversion/f64", |b| {
    let mut s = far.clone().seeded(&Deterministic::new(7));
    b.iter(|| black_box(s.sample()))
  });
  group.finish();
}

/// A law built for every draw, as a Gibbs step builds it, drawn once from the caller's rng.
pub fn bench_truncated_construct(c: &mut Criterion) {
  let mut group = c.benchmark_group("TruncatedConstruct");
  let mut rng = SimdRng::from_seed(7);
  group.bench_function("normal", |b| {
    b.iter(|| {
      let law = SimdTruncatedNormal::<f64>::new(black_box(0.0), 1.0, -1.0, 2.0);
      black_box(law.sample(&mut rng))
    })
  });
  group.bench_function("exp", |b| {
    b.iter(|| {
      let law = SimdTruncatedExp::<f64>::new(black_box(2.0), 0.0, 1.5);
      black_box(law.sample(&mut rng))
    })
  });
  group.bench_function("gamma", |b| {
    b.iter(|| {
      let law = SimdTruncatedGamma::<f64>::new(black_box(2.0), 1.0, 1.0, 5.0);
      black_box(law.sample(&mut rng))
    })
  });
  group.bench_function("beta", |b| {
    b.iter(|| {
      let law = SimdTruncatedBeta::<f64>::new(black_box(2.0), 2.0, 0.2, 0.8);
      black_box(law.sample(&mut rng))
    })
  });
  group.finish();
}

pub fn bench_truncated_normal(c: &mut Criterion) {
  let mut group = c.benchmark_group("TruncatedNormal");
  for (label, lo, hi) in [
    ("seeded/f64/bulk", -1.0, 2.0),
    ("seeded/f64/exp_tail", 3.0, 6.0),
    ("seeded/f64/short_tail", 3.0, 3.4),
    ("seeded/f64/uniform_tail", 3.0, 3.2),
  ] {
    let law = SimdTruncatedNormal::<f64>::new(0.0, 1.0, lo, hi);
    group.bench_function(label, |b| {
      let mut s = law.clone().seeded(&Deterministic::new(7));
      b.iter(|| black_box(s.sample()))
    });
  }
  group.finish();
}

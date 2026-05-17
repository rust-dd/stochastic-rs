//! Benchmarks for the realized-volatility module.
//!
//! Hot paths measured:
//! - Realized variance, semivariance, skewness on 50k intraday returns.
//! - Bipower variation, MedRV, BNS jump test.
//! - Realized kernel (Parzen) at the rule-of-thumb bandwidth.
//! - Pre-averaging and TSRV / MSRV on 50k-tick noisy paths.

use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use ndarray::Array1;
use stochastic_rs::distributions::normal::SimdNormal;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stats::realized::KernelType;
use stochastic_rs::stats::realized::bipower_variation;
use stochastic_rs::stats::realized::bns_jump_test;
use stochastic_rs::stats::realized::kernel::parzen_default_bandwidth;
use stochastic_rs::stats::realized::medrv;
use stochastic_rs::stats::realized::multi_scale_rv;
use stochastic_rs::stats::realized::pre_averaged_variance;
use stochastic_rs::stats::realized::realized_kernel;
use stochastic_rs::stats::realized::realized_kurtosis;
use stochastic_rs::stats::realized::realized_semivariance;
use stochastic_rs::stats::realized::realized_skewness;
use stochastic_rs::stats::realized::realized_variance;
use stochastic_rs::stats::realized::two_scale_rv;

fn iid_returns(n: usize, std: f64, seed: u64) -> Array1<f64> {
  let dist = SimdNormal::<f64>::new(0.0, std, &Deterministic::new(seed));
  let mut out = Array1::<f64>::zeros(n);
  dist.fill_slice_fast(out.as_slice_mut().unwrap());
  out
}

fn noisy_price_path(n: usize, sigma: f64, omega: f64, seed: u64) -> Array1<f64> {
  let dx = SimdNormal::<f64>::new(0.0, sigma, &Deterministic::new(seed));
  let dn = SimdNormal::<f64>::new(0.0, omega, &Deterministic::new(seed.wrapping_add(1)));
  let mut steps = vec![0.0_f64; n];
  dx.fill_slice_fast(&mut steps);
  let mut noise = vec![0.0_f64; n + 1];
  dn.fill_slice_fast(&mut noise);
  let mut x = vec![0.0_f64; n + 1];
  for i in 1..=n {
    x[i] = x[i - 1] + steps[i - 1];
  }
  let y: Vec<f64> = x
    .iter()
    .zip(noise.iter())
    .map(|(&xv, &nv)| xv + nv)
    .collect();
  Array1::from(y)
}

fn bench_variance(c: &mut Criterion) {
  let r = iid_returns(50_000, 0.001, 1);
  c.bench_function("realized_variance_50k", |b| {
    b.iter(|| std::hint::black_box(realized_variance(r.view())));
  });
  c.bench_function("realized_semivariance_50k", |b| {
    b.iter(|| std::hint::black_box(realized_semivariance(r.view(), 0.0)));
  });
  c.bench_function("realized_skewness_50k", |b| {
    b.iter(|| std::hint::black_box(realized_skewness(r.view())));
  });
  c.bench_function("realized_kurtosis_50k", |b| {
    b.iter(|| std::hint::black_box(realized_kurtosis(r.view())));
  });
}

fn bench_bipower(c: &mut Criterion) {
  let r = iid_returns(50_000, 0.001, 3);
  c.bench_function("bipower_variation_50k", |b| {
    b.iter(|| std::hint::black_box(bipower_variation(r.view())));
  });
  c.bench_function("medrv_50k", |b| {
    b.iter(|| std::hint::black_box(medrv(r.view())));
  });
  c.bench_function("bns_jump_test_50k", |b| {
    b.iter(|| std::hint::black_box(bns_jump_test(r.view(), 0.05)));
  });
}

fn bench_kernel(c: &mut Criterion) {
  let r = iid_returns(50_000, 0.001, 5);
  let h = parzen_default_bandwidth(50_000, 0.5);
  c.bench_function("realized_kernel_parzen_50k", |b| {
    b.iter(|| std::hint::black_box(realized_kernel(r.view(), KernelType::Parzen, h)));
  });
}

fn bench_noise_robust(c: &mut Criterion) {
  let y = noisy_price_path(50_000, 0.0005, 0.0001, 7);
  c.bench_function("pre_averaged_variance_50k", |b| {
    b.iter(|| {
      let dy = Array1::from_iter((1..y.len()).map(|i| y[i] - y[i - 1]));
      std::hint::black_box(pre_averaged_variance(dy.view(), 1.0 / 3.0))
    });
  });
  c.bench_function("two_scale_rv_50k", |b| {
    b.iter(|| std::hint::black_box(two_scale_rv(y.view(), 20)));
  });
  c.bench_function("multi_scale_rv_50k", |b| {
    b.iter(|| std::hint::black_box(multi_scale_rv(y.view(), 12)));
  });
}

criterion_group!(
  benches,
  bench_variance,
  bench_bipower,
  bench_kernel,
  bench_noise_robust
);
criterion_main!(benches);

//! Benchmarks for the `stats::econometrics` module.

use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use ndarray::Array1;
#[cfg(feature = "openblas")]
use ndarray::Array2;
use stochastic_rs::distributions::normal::SimdNormal;
#[cfg(feature = "openblas")]
use stochastic_rs::stats::econometrics::GaussianHmm;
use stochastic_rs::stats::econometrics::cusum;
#[cfg(feature = "openblas")]
use stochastic_rs::stats::econometrics::engle_granger_test;
#[cfg(feature = "openblas")]
use stochastic_rs::stats::econometrics::granger_causality;
#[cfg(feature = "openblas")]
use stochastic_rs::stats::econometrics::johansen_test;
use stochastic_rs::stats::econometrics::pelt;

#[cfg(feature = "openblas")]
fn random_walk(seed: u64, n: usize, sigma: f64) -> Array1<f64> {
  let dist = SimdNormal::<f64>::new(0.0, sigma, &stochastic_rs_core::simd_rng::Deterministic::new(seed));
  let mut steps = vec![0.0_f64; n];
  dist.fill_slice_fast(&mut steps);
  let mut out = Array1::<f64>::zeros(n);
  for i in 1..n {
    out[i] = out[i - 1] + steps[i];
  }
  out
}

fn iid_normal(seed: u64, n: usize) -> Array1<f64> {
  let dist = SimdNormal::<f64>::new(0.0, 1.0, &stochastic_rs_core::simd_rng::Deterministic::new(seed));
  let mut buf = vec![0.0_f64; n];
  dist.fill_slice_fast(&mut buf);
  Array1::from(buf)
}

fn bench_changepoint(c: &mut Criterion) {
  let s = iid_normal(7, 5_000);
  c.bench_function("cusum_5k", |b| {
    b.iter(|| std::hint::black_box(cusum(s.view(), 0.5, 5.0)));
  });
  c.bench_function("pelt_5k", |b| {
    b.iter(|| std::hint::black_box(pelt(s.view(), 5.0, 10)));
  });
}

#[cfg(feature = "openblas")]
fn bench_cointegration(c: &mut Criterion) {
  let x = random_walk(11, 1_000, 1.0);
  let dist = SimdNormal::<f64>::new(0.0, 0.05, &stochastic_rs_core::simd_rng::Deterministic::new(13));
  let mut eps = vec![0.0_f64; 1_000];
  dist.fill_slice_fast(&mut eps);
  let y: Array1<f64> = (0..1_000)
    .map(|i| 1.0 + 1.2 * x[i] + eps[i])
    .collect::<Vec<_>>()
    .into();
  c.bench_function("engle_granger_1k", |b| {
    b.iter(|| std::hint::black_box(engle_granger_test(y.view(), x.view())));
  });
  let mut s = Array2::<f64>::zeros((500, 3));
  for i in 0..500 {
    s[[i, 0]] = x[i];
    s[[i, 1]] = y[i];
    s[[i, 2]] = random_walk(17, 500, 1.0)[i];
  }
  c.bench_function("johansen_500x3", |b| {
    b.iter(|| std::hint::black_box(johansen_test(s.view(), 2)));
  });
}

#[cfg(feature = "openblas")]
fn bench_granger(c: &mut Criterion) {
  let x = iid_normal(21, 2_000);
  let y = iid_normal(23, 2_000);
  c.bench_function("granger_2k_lag4", |b| {
    b.iter(|| std::hint::black_box(granger_causality(y.view(), x.view(), 4, 0.05)));
  });
}

#[cfg(feature = "openblas")]
fn bench_hmm(c: &mut Criterion) {
  let dist = SimdNormal::<f64>::new(0.0, 1.0, &stochastic_rs_core::simd_rng::Deterministic::new(31));
  let mut buf = vec![0.0_f64; 1_000];
  dist.fill_slice_fast(&mut buf);
  let obs = Array1::from(buf);
  c.bench_function("hmm_baum_welch_2state_1k", |b| {
    b.iter(|| {
      let mut hmm = GaussianHmm::new(
        Array1::from(vec![0.5, 0.5]),
        Array2::from_shape_vec((2, 2), vec![0.9, 0.1, 0.1, 0.9]).unwrap(),
        Array1::from(vec![-1.0, 1.0]),
        Array1::from(vec![1.0, 1.0]),
      );
      std::hint::black_box(hmm.baum_welch(obs.view(), 20, 1e-6))
    });
  });
}

#[cfg(feature = "openblas")]
criterion_group!(
  benches,
  bench_changepoint,
  bench_cointegration,
  bench_granger,
  bench_hmm
);
#[cfg(not(feature = "openblas"))]
criterion_group!(benches, bench_changepoint);
criterion_main!(benches);

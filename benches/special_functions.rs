use std::hint::black_box;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use stochastic_rs::distributions::special::bessel_ie;
use stochastic_rs::distributions::special::gamma;
use stochastic_rs::distributions::special::ln_bessel_ie;

fn bench_gamma(c: &mut Criterion) {
  let mut group = c.benchmark_group("special/gamma");
  group.measurement_time(Duration::from_secs(3));
  for x in [0.4, 1.4, -0.4, 30.5] {
    group.bench_with_input(BenchmarkId::from_parameter(x), &x, |b, &x| {
      b.iter(|| gamma(black_box(x)))
    });
  }
  group.finish();
}

fn bench_bessel_i(c: &mut Criterion) {
  let mut group = c.benchmark_group("special/bessel_ie");
  group.measurement_time(Duration::from_secs(3));
  for (region, nu, x) in [
    ("series", 0.78, 1.0),
    ("temme", 0.78, 40.0),
    ("hankel", 4.0, 5_065.0),
    ("uniform", 1_999.0, 2_452.0),
    ("reflection", -0.78, 4.3),
  ] {
    group.bench_function(region, |b| {
      b.iter(|| bessel_ie(black_box(nu), black_box(x)))
    });
  }
  group.bench_function("ln_uniform", |b| {
    b.iter(|| ln_bessel_ie(black_box(1_999.0), black_box(89.5)))
  });
  group.finish();
}

criterion_group!(benches, bench_gamma, bench_bessel_i);
criterion_main!(benches);

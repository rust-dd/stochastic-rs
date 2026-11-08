use std::hint::black_box;
use std::time::Duration;

use criterion::Criterion;
use ndarray::array;
use stochastic_rs::distributions::SimdDistribution;
use stochastic_rs::distributions::wishart::SimdWishart;
use stochastic_rs::simd_rng::Unseeded;

/// One Bartlett draw of a 3 × 3 Wishart matrix from the buffered stream.
pub fn bench_wishart(c: &mut Criterion) {
  let mut group = c.benchmark_group("Wishart");
  group.measurement_time(Duration::from_secs(3));
  group.warm_up_time(Duration::from_millis(500));
  let scale = array![[2.0, 0.5, 0.1], [0.5, 1.0, 0.2], [0.1, 0.2, 1.5]];
  group.bench_function("sample_f64", |b| {
    let mut stream = SimdWishart::<f64>::new(8.0, scale.clone()).seeded(&Unseeded);
    b.iter(|| black_box(stream.sample()))
  });
  group.finish();
}

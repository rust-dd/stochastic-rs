use std::hint::black_box;
use std::time::Duration;

use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;

#[path = "../tests/common/lm_calibration.rs"]
mod fixtures;

fn calibration(c: &mut Criterion) {
  let mut group = c.benchmark_group("calibration");
  group
    .sample_size(20)
    .measurement_time(Duration::from_secs(5));
  for case in fixtures::cases() {
    let (converged, rmse) = (case.run)();
    assert!(
      converged != Some(false) && rmse < case.max_rmse,
      "{}: converged={converged:?}, RMSE={rmse}",
      case.name
    );
    group.bench_function(case.name, |b| b.iter(|| black_box((case.run)())));
  }
  group.finish();
}

criterion_group!(benches, calibration);
criterion_main!(benches);

use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs::quant::pricing::slv::HestonSlvParams;
use stochastic_rs::quant::pricing::slv::HestonSlvPricer;
use stochastic_rs::quant::pricing::slv::ParticleMethod;
use stochastic_rs::quant::pricing::slv::calibrate_leverage;
use stochastic_rs::traits::Grid2D;
use stochastic_rs::traits::ModelPricer;

fn params() -> HestonSlvParams {
  HestonSlvParams {
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
    v0: 0.04,
    eta: 1.0,
  }
}

fn flat_local_vol() -> Grid2D<f64> {
  Grid2D::new(
    Array1::from_vec(vec![0.1, 0.25, 0.5]),
    Array1::linspace(70.0, 130.0, 11),
    Array2::from_elem((3, 11), 0.2),
  )
}

fn bench_calibrate_leverage(c: &mut Criterion) {
  let params = params();
  let lv = flat_local_vol();
  let mut group = c.benchmark_group("slv_calibrate_leverage");
  for &n in &[1_000usize, 10_000] {
    let method = ParticleMethod::default()
      .with_particles(n)
      .with_steps_per_year(100)
      .with_seed(42);
    group.bench_with_input(criterion::BenchmarkId::from_parameter(n), &n, |b, _| {
      b.iter(|| calibrate_leverage(&params, 100.0, 0.05, 0.0, &lv, &[0.25, 0.5], &method).unwrap())
    });
  }
  group.finish();
}

fn bench_slv_price_call(c: &mut Criterion) {
  let params = params();
  let lv = flat_local_vol();
  let method = ParticleMethod::default()
    .with_particles(2_000)
    .with_steps_per_year(100)
    .with_seed(42);
  let run = calibrate_leverage(&params, 100.0, 0.05, 0.0, &lv, &[0.25, 0.5], &method).unwrap();
  let pricer = HestonSlvPricer::new(params, run.leverage, 0.05, 0.0)
    .with_paths(10_000)
    .with_steps_per_year(100)
    .with_seed(42);

  c.bench_function("slv_price_call_10k_paths", |b| {
    b.iter(|| pricer.price_call(100.0, 100.0, 0.05, 0.0, 0.5))
  });
}

criterion_group!(benches, bench_calibrate_leverage, bench_slv_price_call);
criterion_main!(benches);

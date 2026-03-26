use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use ndarray::Array1;
use stochastic_rs::quant::pricing::slv::HestonSlvParams;
use stochastic_rs::quant::pricing::slv::HestonSlvPricer;
use stochastic_rs::quant::pricing::slv::calibrate_leverage;
use stochastic_rs::traits::ModelPricer;

fn bench_calibrate_leverage(c: &mut Criterion) {
  let params = HestonSlvParams {
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
    v0: 0.04,
    eta: 1.0,
  };
  let spots = Array1::linspace(70.0, 130.0, 11);
  let times = Array1::from_vec(vec![0.1, 0.25, 0.5]);
  let lv = ndarray::Array2::from_elem((3, 11), 0.2);

  c.bench_function("slv_calibrate_1k_particles", |b| {
    b.iter(|| {
      calibrate_leverage(
        &params, 100.0, 0.05, 0.0, &spots, &times, &lv, &spots, &times, 1000, 42,
      )
    })
  });
}

fn bench_slv_price_call(c: &mut Criterion) {
  let params = HestonSlvParams {
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
    v0: 0.04,
    eta: 1.0,
  };
  let spots = Array1::linspace(70.0, 130.0, 11);
  let times = Array1::from_vec(vec![0.1, 0.25, 0.5]);
  let lv = ndarray::Array2::from_elem((3, 11), 0.2);

  let leverage = calibrate_leverage(
    &params, 100.0, 0.05, 0.0, &spots, &times, &lv, &spots, &times, 2000, 42,
  );

  let pricer = HestonSlvPricer::new(params, leverage, 0.05, 0.0)
    .with_paths(10_000)
    .with_steps_per_year(100)
    .with_seed(42);

  c.bench_function("slv_price_call_10k_paths", |b| {
    b.iter(|| pricer.price_call(100.0, 100.0, 0.05, 0.0, 0.5))
  });
}

criterion_group!(benches, bench_calibrate_leverage, bench_slv_price_call);
criterion_main!(benches);

use criterion::{Criterion, criterion_group, criterion_main};
use stochastic_rs::quant::pricing::cgmysv::{CgmysvParams, CgmysvPathGen, CgmysvPricer};
use stochastic_rs::quant::OptionType;

fn bench_path_generation(c: &mut Criterion) {
  let params = CgmysvParams {
    alpha: 0.52,
    lambda_plus: 25.46,
    lambda_minus: 4.604,
    kappa: 1.003,
    eta: 0.0711,
    zeta: 0.3443,
    rho: -2.0280,
    v0: 0.0064,
  };

  let gen = CgmysvPathGen {
    params: params.clone(),
    n_steps: 100,
    n_jumps: 1024,
    t: 28.0 / 365.0,
  };

  c.bench_function("cgmysv_path_gen_1000", |b| {
    b.iter(|| gen.generate(1000));
  });
}

fn bench_european_pricing(c: &mut Criterion) {
  let pricer = CgmysvPricer {
    params: CgmysvParams {
      alpha: 0.52,
      lambda_plus: 25.46,
      lambda_minus: 4.604,
      kappa: 1.003,
      eta: 0.0711,
      zeta: 0.3443,
      rho: -2.0280,
      v0: 0.0064,
    },
    s: 2488.0,
    r: 0.0121,
    q: 0.0188,
    n_paths: 1000,
    n_steps: 100,
    n_jumps: 1024,
  };

  c.bench_function("cgmysv_european_call_1000", |b| {
    b.iter(|| pricer.price_european(2500.0, 28.0 / 365.0, OptionType::Call));
  });
}

criterion_group!(benches, bench_path_generation, bench_european_pricing);
criterion_main!(benches);

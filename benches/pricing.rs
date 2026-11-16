//! Hot paths of the quant pricers: Heston, the Fourier pricers, Black–Scholes, Hagan SABR,
//! Black inversion, the caplet.

use std::hint::black_box;
use std::time::Duration;

use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use stochastic_rs::quant::OptionType;
use stochastic_rs::quant::instruments::option::caplet::black_forward_caplet;
use stochastic_rs::quant::pricing::bsm::BSMCoc;
use stochastic_rs::quant::pricing::bsm::BSMPricer;
use stochastic_rs::quant::pricing::fourier::CarrMadanPricer;
use stochastic_rs::quant::pricing::fourier::CosEngine;
use stochastic_rs::quant::pricing::fourier::FourierModelExt;
use stochastic_rs::quant::pricing::fourier::GilPelaezPricer;
use stochastic_rs::quant::pricing::fourier::HestonFourier;
use stochastic_rs::quant::pricing::fourier::KouFourier;
use stochastic_rs::quant::pricing::fourier::LewisPricer;
use stochastic_rs::quant::pricing::heston::HestonPricer;
use stochastic_rs::quant::pricing::sabr::SabrPricer;
use stochastic_rs::quant::pricing::sabr::hagan_implied_vol;
use stochastic_rs::traits::ModelPricer;

const S: f64 = 100.0;
const R: f64 = 0.05;
const Q: f64 = 0.02;
const STRIKES: [f64; 9] = [70.0, 80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0, 130.0];

fn heston_pricer(c: &mut Criterion) {
  let mut group = c.benchmark_group("pricing/heston_pricer");
  group
    .sample_size(30)
    .measurement_time(Duration::from_secs(5));
  let model = HestonPricer::new(0.04, -0.7, 1.5, 0.04, 0.3, None);
  group.bench_function("call_grid_9x1y", |b| {
    b.iter(|| {
      STRIKES
        .iter()
        .map(|&k| model.price_call(S, k, R, Q, 1.0))
        .sum::<f64>()
    })
  });
  group.bench_function("call_short_dated_9x0.02y", |b| {
    b.iter(|| {
      STRIKES
        .iter()
        .map(|&k| model.price_call(S, k, R, Q, 0.02))
        .sum::<f64>()
    })
  });
  group.bench_function("implied_vol_9", |b| {
    let prices = STRIKES.map(|k| model.price_call(S, k, R, Q, 1.0));
    b.iter(|| {
      STRIKES
        .iter()
        .zip(prices)
        .map(|(&k, p)| model.implied_volatility(p, S, k, R, Q, 1.0, OptionType::Call))
        .sum::<f64>()
    })
  });
  group.finish();
}

fn fourier(c: &mut Criterion) {
  let mut group = c.benchmark_group("pricing/fourier_heston");
  group
    .sample_size(30)
    .measurement_time(Duration::from_secs(5));
  let heston = HestonFourier {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
    r: R,
    q: Q,
  };
  group.bench_function("gil_pelaez_9", |b| {
    b.iter(|| {
      STRIKES
        .iter()
        .map(|&k| GilPelaezPricer::price_call(&heston, S, k, R, Q, 1.0))
        .sum::<f64>()
    })
  });
  group.bench_function("lewis_9", |b| {
    b.iter(|| {
      STRIKES
        .iter()
        .map(|&k| LewisPricer::price_call(&heston, S, k, R, Q, 1.0))
        .sum::<f64>()
    })
  });
  group.bench_function("carr_madan_surface", |b| {
    let pricer = CarrMadanPricer::default();
    b.iter(|| black_box(pricer.price_call_surface(&heston, S, R, 1.0)))
  });
  group.bench_function("cos_9", |b| {
    let cos = CosEngine::default();
    b.iter(|| {
      STRIKES
        .iter()
        .map(|&k| cos.price(&heston, S, k, R, Q, 1.0, OptionType::Call))
        .sum::<f64>()
    })
  });
  group.bench_function("cumulants", |b| b.iter(|| black_box(heston.cumulants(1.0))));
  group.finish();

  let mut group = c.benchmark_group("pricing/fourier_levy");
  group
    .sample_size(30)
    .measurement_time(Duration::from_secs(5));
  let kou = KouFourier {
    sigma: 0.2,
    lambda: 0.8,
    p_up: 0.4,
    eta1: 10.0,
    eta2: 5.0,
    r: R,
    q: Q,
  };
  group.bench_function("kou_gil_pelaez_9", |b| {
    b.iter(|| {
      STRIKES
        .iter()
        .map(|&k| GilPelaezPricer::price_call(&kou, S, k, R, Q, 1.0))
        .sum::<f64>()
    })
  });
  group.finish();
}

fn bsm_and_sabr(c: &mut Criterion) {
  let mut group = c.benchmark_group("pricing/bsm");
  group
    .sample_size(50)
    .measurement_time(Duration::from_secs(4));
  let bsm = BSMPricer::new(0.2, BSMCoc::Merton1973);
  group.bench_function("call_put_9", |b| {
    b.iter(|| {
      STRIKES
        .iter()
        .map(|&k| bsm.call_put(S, k, R, Q, 1.0).0)
        .sum::<f64>()
    })
  });
  group.bench_function("greeks_9", |b| {
    b.iter(|| {
      STRIKES
        .iter()
        .map(|&k| bsm.greeks(S, k, R, Q, 1.0, OptionType::Call).vega)
        .sum::<f64>()
    })
  });
  group.finish();

  let mut group = c.benchmark_group("pricing/sabr");
  group
    .sample_size(50)
    .measurement_time(Duration::from_secs(4));
  let sabr = SabrPricer::new(0.2, 1.0, 0.6, -0.3);
  group.bench_function("hagan_vol_9", |b| {
    b.iter(|| {
      STRIKES
        .iter()
        .map(|&k| hagan_implied_vol(k, S, 1.0, 0.2, 1.0, 0.6, -0.3))
        .sum::<f64>()
    })
  });
  group.bench_function("call_put_9", |b| {
    b.iter(|| {
      STRIKES
        .iter()
        .map(|&k| sabr.call_put(S, k, R, Q, 1.0).0)
        .sum::<f64>()
    })
  });
  group.finish();

  let mut group = c.benchmark_group("pricing/caplet");
  group
    .sample_size(50)
    .measurement_time(Duration::from_secs(4));
  group.bench_function("black_forward_caplet_1000", |b| {
    b.iter(|| {
      (0..1000)
        .map(|i| black_forward_caplet(0.03 + 1e-5 * i as f64, 0.025, 1.0, 0.25))
        .sum::<f64>()
    })
  });
  group.finish();
}

criterion_group!(benches, heston_pricer, fourier, bsm_and_sabr);
criterion_main!(benches);

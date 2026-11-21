//! Hot paths of the quant pricers: Heston, Fourier, Black–Scholes, Hagan SABR, Black inversion,
//! the caplet, the Monte Carlo exotics and the Malliavin–Thalmaier delta.

use std::hint::black_box;
use std::time::Duration;

use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs::quant::OptionType;
use stochastic_rs::quant::instruments::option::caplet::black_forward_caplet;
use stochastic_rs::quant::pricing::AutocallablePricer;
use stochastic_rs::quant::pricing::BasketAverageType;
use stochastic_rs::quant::pricing::KnockInStyle;
use stochastic_rs::quant::pricing::McBasketPricer;
use stochastic_rs::quant::pricing::McCliquetPricer;
use stochastic_rs::quant::pricing::barrier::BarrierPricer;
use stochastic_rs::quant::pricing::barrier::BarrierType;
use stochastic_rs::quant::pricing::barrier::DoubleBarrierPricer;
use stochastic_rs::quant::pricing::barrier::McBarrierPricer;
use stochastic_rs::quant::pricing::bsm::BsmCoc;
use stochastic_rs::quant::pricing::bsm::BsmPricer;
use stochastic_rs::quant::pricing::chooser::ChooserPricer;
use stochastic_rs::quant::pricing::cliquet::CliquetPricer;
use stochastic_rs::quant::pricing::compound::CompoundPricer;
use stochastic_rs::quant::pricing::compound::CompoundType;
use stochastic_rs::quant::pricing::exotic::BarrierOption;
use stochastic_rs::quant::pricing::exotic::CliquetOption;
use stochastic_rs::quant::pricing::exotic::ComplexChooserOption;
use stochastic_rs::quant::pricing::exotic::CompoundOption;
use stochastic_rs::quant::pricing::exotic::DoubleBarrierOption;
use stochastic_rs::quant::pricing::exotic::LookbackOption;
use stochastic_rs::quant::pricing::fourier::CarrMadanPricer;
use stochastic_rs::quant::pricing::fourier::CosEngine;
use stochastic_rs::quant::pricing::fourier::FourierModelExt;
use stochastic_rs::quant::pricing::fourier::GilPelaezPricer;
use stochastic_rs::quant::pricing::fourier::HestonFourier;
use stochastic_rs::quant::pricing::fourier::KouFourier;
use stochastic_rs::quant::pricing::fourier::LewisPricer;
use stochastic_rs::quant::pricing::heston::HestonParams;
use stochastic_rs::quant::pricing::heston::HestonPricer;
use stochastic_rs::quant::pricing::lookback::LookbackPricer;
use stochastic_rs::quant::pricing::malliavin_thalmaier::AssetParams;
use stochastic_rs::quant::pricing::malliavin_thalmaier::MtGreeks;
use stochastic_rs::quant::pricing::malliavin_thalmaier::MtPayoff;
use stochastic_rs::quant::pricing::malliavin_thalmaier::MultiHestonParams;
use stochastic_rs::quant::pricing::merton_jump::Merton1976Pricer;
use stochastic_rs::quant::pricing::sabr::SabrParams;
use stochastic_rs::quant::pricing::sabr::SabrPricer;
use stochastic_rs::quant::pricing::sabr::hagan_implied_vol;
use stochastic_rs::quant::pricing::spread::McSpreadPricer;
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
  let model = HestonPricer::new(HestonParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
  });
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
      let model = black_box(&model);
      black_box(&STRIKES)
        .iter()
        .zip(black_box(prices))
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
  let heston = HestonFourier::new(HestonParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
  });
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
  group.bench_function("blanket_9", |b| {
    b.iter(|| {
      STRIKES
        .iter()
        .map(|&k| heston.price_call(S, k, R, Q, 1.0))
        .sum::<f64>()
    })
  });
  group.bench_function("carr_madan_surface", |b| {
    let pricer = CarrMadanPricer::default();
    b.iter(|| black_box(pricer.price_call_surface(&heston, S, R, Q, 1.0)))
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
  group.bench_function("cumulants", |b| {
    b.iter(|| black_box(&heston).cumulants(black_box(1.0)))
  });
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
  let bsm = BsmPricer::new(0.2, BsmCoc::Merton1973);
  group.bench_function("price_call_9", |b| {
    b.iter(|| {
      STRIKES
        .iter()
        .map(|&k| bsm.price_call(S, k, R, Q, 1.0))
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

  let mut group = c.benchmark_group("pricing/merton_jump");
  group
    .sample_size(50)
    .measurement_time(Duration::from_secs(4));
  let merton = Merton1976Pricer::new(0.25, 1.0, 0.4, 50, BsmCoc::Merton1973);
  group.bench_function("price_call_9x50_terms", |b| {
    b.iter(|| {
      let merton = black_box(&merton);
      STRIKES
        .iter()
        .map(|&k| merton.price_call(S, k, R, Q, 1.0))
        .sum::<f64>()
    })
  });
  group.finish();

  let mut group = c.benchmark_group("pricing/sabr");
  group
    .sample_size(50)
    .measurement_time(Duration::from_secs(4));
  let params = SabrParams {
    alpha: 0.2,
    beta: 1.0,
    rho: -0.3,
    nu: 0.6,
  };
  let sabr = SabrPricer::new(params);
  group.bench_function("hagan_vol_9", |b| {
    b.iter(|| {
      let params = black_box(&params);
      black_box(&STRIKES)
        .iter()
        .map(|&k| hagan_implied_vol(k, S, 1.0, params))
        .sum::<f64>()
    })
  });
  group.bench_function("price_call_9", |b| {
    b.iter(|| {
      STRIKES
        .iter()
        .map(|&k| sabr.price_call(S, k, R, Q, 1.0))
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

fn bench_exotics(c: &mut Criterion) {
  let mut group = c.benchmark_group("pricing/exotics");
  group
    .sample_size(50)
    .measurement_time(Duration::from_secs(4));
  let barrier = BarrierPricer::new(0.25);
  let up_out = BarrierOption {
    strike: 100.0,
    barrier: 130.0,
    rebate: 0.0,
    barrier_type: BarrierType::UpAndOut,
    option_type: OptionType::Call,
  };
  group.bench_function("barrier_up_out_call", |b| {
    b.iter(|| black_box(&barrier).price(black_box(&up_out), S, R, Q, 1.0))
  });
  let double = DoubleBarrierPricer::new(0.25).with_terms(50);
  let corridor = DoubleBarrierOption {
    strike: 100.0,
    lower: 70.0,
    upper: 130.0,
    option_type: OptionType::Call,
  };
  group.bench_function("double_barrier_50_terms", |b| {
    b.iter(|| black_box(&double).price(black_box(&corridor), S, R, Q, 1.0))
  });
  let lookback = LookbackPricer::new(0.25);
  let floating = LookbackOption {
    strike: None,
    running_min: None,
    running_max: None,
    option_type: OptionType::Call,
  };
  group.bench_function("lookback_floating", |b| {
    b.iter(|| black_box(&lookback).price(black_box(&floating), S, R, Q, 1.0))
  });
  let chooser = ChooserPricer::new(0.3);
  let complex = ComplexChooserOption {
    call_strike: 105.0,
    call_tau: 1.0,
    put_strike: 95.0,
    put_tau: 0.75,
    choice_time: 0.4,
  };
  group.bench_function("chooser_complex", |b| {
    b.iter(|| black_box(&chooser).price_complex(black_box(&complex), S, R, Q))
  });
  let compound = CompoundPricer::new(0.25);
  let call_on_call = CompoundOption {
    outer_strike: 5.0,
    outer_tau: 0.25,
    inner_strike: 100.0,
    compound_type: CompoundType::CallOnCall,
  };
  group.bench_function("compound_call_on_call", |b| {
    b.iter(|| black_box(&compound).price(black_box(&call_on_call), S, R, Q, 1.0))
  });
  let cliquet = CliquetPricer::new(0.25);
  let collar = CliquetOption {
    notional: 100.0,
    periods: 12,
    local_floor: Some(0.0),
    local_cap: Some(0.05),
    global_floor: None,
    global_cap: None,
  };
  group.bench_function("cliquet_12_periods", |b| {
    b.iter(|| black_box(&cliquet).price(black_box(&collar), R, Q, 1.0))
  });
  group.finish();
}

fn bench_mc_exotics(c: &mut Criterion) {
  let mut group = c.benchmark_group("pricing/mc_exotics");
  group
    .sample_size(10)
    .measurement_time(Duration::from_secs(10));
  let barrier = McBarrierPricer {
    n_paths: 20_000,
    n_steps: 252,
  };
  group.bench_function("mc_barrier_20k_paths_252_steps", |b| {
    b.iter(|| {
      black_box(&barrier).price(
        black_box(S),
        black_box(100.0),
        black_box(120.0),
        black_box(R),
        black_box(0.2),
        black_box(1.0),
        black_box(BarrierType::UpAndOut),
        black_box(OptionType::Call),
      )
    })
  });
  let cliquet = McCliquetPricer {
    s: S,
    notional: 100.0,
    m: 12,
    tau: 1.0,
    r: R,
    q: Q,
    sigma: 0.2,
    local_floor: Some(0.0),
    local_cap: Some(0.08),
    global_floor: Some(0.0),
    global_cap: Some(0.5),
    n_paths: 20_000,
  };
  group.bench_function("mc_cliquet_20k", |b| b.iter(|| black_box(&cliquet).price()));
  let autocallable = AutocallablePricer {
    s: S,
    notional: 100.0,
    observation_times: Array1::from_iter((1..=8).map(|i| 0.25 * i as f64)),
    autocall_barrier: 1.0,
    coupon_barrier: 0.75,
    coupon: 0.025,
    knock_in_barrier: 0.65,
    knock_in_style: KnockInStyle::Continuous,
    memory: false,
    r: 0.03,
    q: 0.01,
    sigma: 0.3,
    n_paths: 20_000,
    steps_per_period: 8,
  };
  group.bench_function("autocallable_20k_continuous", |b| {
    b.iter(|| black_box(&autocallable).price())
  });
  let spread = McSpreadPricer::new(0.3, 0.2, 0.5, 50_000);
  group.bench_function("mc_spread_50k", |b| {
    b.iter(|| {
      black_box(&spread).price_call(
        black_box(110.0),
        black_box(100.0),
        black_box(5.0),
        black_box(0.03),
        black_box(0.0),
        black_box(0.0),
        black_box(1.0),
      )
    })
  });
  let rho = Array2::from_shape_fn((3, 3), |(i, j)| if i == j { 1.0 } else { 0.5 });
  let basket = McBasketPricer::new(
    Array1::from_elem(3, 1.0 / 3.0),
    BasketAverageType::Arithmetic,
    Array1::from_vec(vec![0.2, 0.25, 0.3]),
    rho,
    50_000,
  );
  let spots = Array1::from_elem(3, S);
  let yields = Array1::from_elem(3, Q);
  group.bench_function("mc_basket_3_assets_50k", |b| {
    b.iter(|| {
      black_box(&basket).price_call(
        black_box(spots.view()),
        black_box(100.0),
        black_box(0.03),
        black_box(yields.view()),
        black_box(1.0),
      )
    })
  });
  group.finish();
}

/// The two-asset set of Kohatsu-Higa & Yasuda (2008), RIMS Kokyuroku 1580, §7.
fn mt_paper_params() -> MultiHestonParams<f64> {
  let asset = |vol: f64| AssetParams {
    s: 100.0,
    v0: vol * vol,
    kappa: 1.0,
    theta: vol * vol,
    xi: 0.0,
    rho: 0.0,
  };
  let mut cross_corr = Array2::<f64>::eye(2);
  cross_corr[[0, 1]] = 0.2;
  cross_corr[[1, 0]] = 0.2;
  MultiHestonParams {
    assets: vec![asset(0.25), asset(0.20)],
    cross_corr,
    r: 0.0,
    tau: 1.0,
    n_steps: 9,
  }
}

fn bench_malliavin_thalmaier(c: &mut Criterion) {
  let mut group = c.benchmark_group("pricing/malliavin_thalmaier");
  group
    .sample_size(10)
    .measurement_time(Duration::from_secs(10));
  let engine = MtGreeks::new(mt_paper_params(), 0.01, 2_000);
  let payoff = MtPayoff::DigitalPut2D {
    strikes: [100.0, 100.0],
  };
  group.bench_function("mt_delta_2k_paths", |b| {
    b.iter(|| {
      black_box(&engine)
        .try_delta_with_seed(black_box(&payoff), black_box(0), black_box(0x1580_0011))
        .unwrap()
    })
  });
  group.finish();
}

criterion_group!(
  benches,
  heston_pricer,
  fourier,
  bsm_and_sabr,
  bench_exotics,
  bench_mc_exotics,
  bench_malliavin_thalmaier
);
criterion_main!(benches);

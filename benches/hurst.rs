use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use ndarray::Array1;
use stochastic_rs::stochastic::noise::fgn::Fgn;
use stochastic_rs::stochastic::process::fbm::Fbm;
use stochastic_rs::stochastic::traits::ProcessExt;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stats::fractal_dim::{FractalDimEstimator, Higuchi, Variogram};
use stochastic_rs_stats::hurst::{
  Dfa, Gph, HurstEstimator, RescaledRange, VariationKind, Variations, Wavelet,
};

fn make_fbm_path(h: f64, n: usize, seed: u64) -> Array1<f64> {
  Fbm::<f64, _>::new(h, n, Some(1.0), Deterministic::new(seed)).sample()
}

fn make_fgn_path(h: f64, n: usize, seed: u64) -> Array1<f64> {
  Fgn::<f64, _>::new(h, n, Some(1.0), Deterministic::new(seed)).sample()
}

fn bench_hurst_estimators(c: &mut Criterion) {
  let n = 4096usize;
  let h_true = 0.7_f64;
  let fbm = make_fbm_path(h_true, n, 0xC0FFEE);
  let fgn = make_fgn_path(h_true, n, 0xC0FFEE);

  let mut group = c.benchmark_group(format!("hurst_n{n}_h{}", (h_true * 10.0) as u32));

  let rs = RescaledRange::default();
  group.bench_function("rescaled_range", |b| b.iter(|| rs.estimate(fbm.view())));

  let dfa = Dfa::default();
  group.bench_function("dfa_1", |b| b.iter(|| dfa.estimate(fbm.view())));

  let gph = Gph::default();
  group.bench_function("gph", |b| b.iter(|| gph.estimate(fbm.view())));

  let wavelet = Wavelet::default();
  group.bench_function("wavelet_d4", |b| b.iter(|| wavelet.estimate(fbm.view())));

  let higuchi = Higuchi { kmax: 32 };
  group.bench_function("higuchi", |b| b.iter(|| higuchi.estimate(fbm.view())));

  let variogram = Variogram { p: 2.0 };
  group.bench_function("variogram", |b| b.iter(|| variogram.estimate(fbm.view())));

  let variations = Variations {
    kind: VariationKind::CentralDiff,
  };
  group.bench_function("variations_central_diff", |b| {
    b.iter(|| variations.estimate(fbm.view()))
  });

  let variations_daub = Variations {
    kind: VariationKind::Daubechies,
  };
  group.bench_function("variations_daubechies", |b| {
    b.iter(|| variations_daub.estimate(fbm.view()))
  });

  group.finish();

  let mut group_fgn = c.benchmark_group(format!("hurst_fgn_n{n}"));
  let wavelet_fgn = Wavelet {
    take_differences: false,
    ..Default::default()
  };
  group_fgn.bench_function("wavelet_d4_fgn", |b| {
    b.iter(|| wavelet_fgn.estimate(fgn.view()))
  });
  group_fgn.finish();
}

criterion_group!(benches, bench_hurst_estimators);
criterion_main!(benches);

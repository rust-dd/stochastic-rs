//! Copula hot paths: conditional-inversion sampling (closed form, Brent, the Marshall–Olkin atom), the
//! elliptical and vine samplers and cdfs, the fits, the pairwise τ matrix and the bootstrap goodness-of-fit test.

use std::time::Duration;

use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use ndarray::Array2;
use stochastic_rs::copulas::bivariate::clayton::Clayton;
use stochastic_rs::copulas::bivariate::frank::Frank;
use stochastic_rs::copulas::bivariate::gaussian::GaussianCopula;
use stochastic_rs::copulas::bivariate::marshall_olkin::MarshallOlkin;
use stochastic_rs::copulas::correlation::kendall_tau;
use stochastic_rs::copulas::gof::gof_cramer_von_mises;
use stochastic_rs::copulas::gof::pseudo_observations;
use stochastic_rs::copulas::multivariate::dvine::DVine;
use stochastic_rs::copulas::multivariate::dvine::PairCopula;
use stochastic_rs::copulas::multivariate::fit::PairFamily;
use stochastic_rs::copulas::multivariate::fit::SelectionCriterion;
use stochastic_rs::copulas::multivariate::fit::VineStructure;
use stochastic_rs::copulas::multivariate::fit::fit_vine;
use stochastic_rs::copulas::multivariate::gaussian::GaussianMultivariate;
use stochastic_rs::copulas::multivariate::t::TMultivariate;
use stochastic_rs::traits::BivariateExt;
use stochastic_rs::traits::MultivariateExt;

fn corr(d: usize) -> Array2<f64> {
  Array2::from_shape_fn((d, d), |(i, j)| {
    if i == j {
      1.0
    } else {
      0.5_f64.powi((i as i32 - j as i32).abs())
    }
  })
}

fn queries(d: usize) -> Array2<f64> {
  Array2::from_shape_fn((16, d), |(i, j)| 0.1 + 0.05 * ((i + 3 * j) % 16) as f64)
}

fn dvine(d: usize) -> DVine {
  let trees = (0..d - 1)
    .map(|m| {
      let pair = if m == 0 {
        PairCopula::Gaussian { rho: 0.5 }
      } else {
        PairCopula::Independence
      };
      vec![pair; d - 1 - m]
    })
    .collect();
  DVine::new(d, trees).unwrap()
}

fn bivariate_sample(c: &mut Criterion) {
  let mut group = c.benchmark_group("copulas/bivariate_sample");
  let clayton = Clayton {
    theta: Some(2.0),
    ..Clayton::new()
  };
  group.bench_function("clayton_n10k", |b| {
    b.iter(|| clayton.sample_with_seed(10_000, 42).unwrap())
  });
  // At theta >= 2 the Brent h-inverse exceeds its 50-iteration cap on some of 2k draws, and the unwrap panics.
  let frank = Frank::new(Some(1.0), None);
  group.bench_function("frank_brent_n2k", |b| {
    b.iter(|| frank.sample_with_seed(2_000, 42).unwrap())
  });
  let gaussian = GaussianCopula {
    theta: Some(0.5),
    ..GaussianCopula::new()
  };
  group.bench_function("gaussian_n10k", |b| {
    b.iter(|| gaussian.sample_with_seed(10_000, 42).unwrap())
  });
  let mo = MarshallOlkin::with_alpha_beta(0.3, 0.6);
  group.bench_function("marshall_olkin_n10k", |b| {
    b.iter(|| mo.sample_with_seed(10_000, 42).unwrap())
  });
  group.finish();
}

fn multivariate_sample(c: &mut Criterion) {
  let mut group = c.benchmark_group("copulas/multivariate_sample");
  let gaussian = GaussianMultivariate::new_with_corr(corr(5)).unwrap();
  group.bench_function("gaussian_d5_n10k", |b| {
    b.iter(|| gaussian.sample_with_seed(10_000, 42).unwrap())
  });
  let t = TMultivariate::new_with(corr(5), 5.0).unwrap();
  group.bench_function("t_d5_n10k", |b| {
    b.iter(|| t.sample_with_seed(10_000, 42).unwrap())
  });
  let vine = dvine(4);
  group.bench_function("dvine_d4_n10k", |b| {
    b.iter(|| vine.sample_with_seed(10_000, 42).unwrap())
  });
  group.finish();
}

fn multivariate_cdf(c: &mut Criterion) {
  let mut group = c.benchmark_group("copulas/multivariate_cdf");
  group.sample_size(20);
  let q = queries(3);
  let gaussian = GaussianMultivariate::new_with_corr(corr(3)).unwrap();
  group.bench_function("gaussian_d3_16q", |b| b.iter(|| gaussian.cdf(&q).unwrap()));
  let t = TMultivariate::new_with(corr(3), 5.0).unwrap();
  group.bench_function("t_d3_16q", |b| b.iter(|| t.cdf(&q).unwrap()));
  let vine = dvine(3);
  group.bench_function("dvine_d3_16q", |b| b.iter(|| vine.cdf(&q).unwrap()));
  group.finish();
}

fn fit(c: &mut Criterion) {
  let mut group = c.benchmark_group("copulas/fit");
  group.sample_size(20);
  group.measurement_time(Duration::from_secs(9));
  let clayton = Clayton {
    theta: Some(2.0),
    ..Clayton::new()
  };
  let pairs = pseudo_observations(&clayton.sample_with_seed(2_000, 1).unwrap());
  group.bench_function("clayton_fit_n2k", |b| {
    b.iter(|| {
      let mut fitted = Clayton::new();
      fitted.fit(&pairs).unwrap()
    })
  });
  let data = pseudo_observations(&dvine(4).sample_with_seed(1_000, 1).unwrap());
  group.bench_function("vine_fit_d4_n1k", |b| {
    b.iter(|| {
      fit_vine(
        &data,
        VineStructure::DVine,
        &PairFamily::ALL,
        SelectionCriterion::Aic,
      )
      .unwrap()
    })
  });
  group.bench_function("kendall_tau_d4_n1k", |b| b.iter(|| kendall_tau(&data)));
  group.finish();
}

fn gof(c: &mut Criterion) {
  let mut group = c.benchmark_group("copulas/gof");
  group.sample_size(10);
  let clayton = Clayton {
    theta: Some(2.0),
    ..Clayton::new()
  };
  let pairs = pseudo_observations(&clayton.sample_with_seed(300, 1).unwrap());
  let mut fitted = Clayton::new();
  fitted.fit(&pairs).unwrap();
  group.bench_function("clayton_cvm_n300_r10", |b| {
    b.iter(|| gof_cramer_von_mises(&fitted, &pairs, 10, 100, |c, x| c.fit(x)).unwrap())
  });
  group.finish();
}

criterion_group!(
  benches,
  bivariate_sample,
  multivariate_sample,
  multivariate_cdf,
  fit,
  gof
);
criterion_main!(benches);

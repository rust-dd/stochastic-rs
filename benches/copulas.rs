//! Copula hot paths: conditional-inversion sampling (closed form, Brent, the Marshall–Olkin atom), the
//! elliptical and vine samplers and cdfs, the fits, the pairwise τ matrix and the bootstrap goodness-of-fit test.

use std::time::Duration;

use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use ndarray::Array2;
use stochastic_rs::copulas::bivariate::BivariateFamily;
use stochastic_rs::copulas::bivariate::amh::Amh;
use stochastic_rs::copulas::bivariate::clayton::Clayton;
use stochastic_rs::copulas::bivariate::fgm::Fgm;
use stochastic_rs::copulas::bivariate::frank::Frank;
use stochastic_rs::copulas::bivariate::galambos::Galambos;
use stochastic_rs::copulas::bivariate::gaussian::GaussianCopula;
use stochastic_rs::copulas::bivariate::gumbel::Gumbel;
use stochastic_rs::copulas::bivariate::marshall_olkin::MarshallOlkin;
use stochastic_rs::copulas::bivariate::plackett::Plackett;
use stochastic_rs::copulas::bivariate::t_copula::TCopula;
use stochastic_rs::copulas::correlation::kendall_tau;
use stochastic_rs::copulas::gof::GofConfig;
use stochastic_rs::copulas::gof::gof_cramer_von_mises;
use stochastic_rs::copulas::gof::pseudo_observations;
use stochastic_rs::copulas::multivariate::dvine::DVine;
use stochastic_rs::copulas::multivariate::dvine::PairCopula;
use stochastic_rs::copulas::multivariate::fit::SelectionCriterion;
use stochastic_rs::copulas::multivariate::fit::VineStructure;
use stochastic_rs::copulas::multivariate::fit::fit_vine;
use stochastic_rs::copulas::multivariate::gaussian::GaussianMultivariate;
use stochastic_rs::copulas::multivariate::t::TMultivariate;
use stochastic_rs::simd_rng::Deterministic;
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
  let clayton = Clayton::new(2.0);
  group.bench_function("clayton_n10k", |b| {
    b.iter(|| clayton.sample(10_000, &Deterministic::new(42)).unwrap())
  });
  let frank = Frank::new(4.0);
  group.bench_function("frank_n2k", |b| {
    b.iter(|| frank.sample(2_000, &Deterministic::new(42)).unwrap())
  });
  let gaussian = GaussianCopula::new(0.5);
  group.bench_function("gaussian_n10k", |b| {
    b.iter(|| gaussian.sample(10_000, &Deterministic::new(42)).unwrap())
  });
  let gumbel = Gumbel::new(2.0);
  group.bench_function("gumbel_brent_n2k", |b| {
    b.iter(|| gumbel.sample(2_000, &Deterministic::new(42)).unwrap())
  });
  let mo = MarshallOlkin::new(0.3, 0.6);
  group.bench_function("marshall_olkin_n10k", |b| {
    b.iter(|| mo.sample(10_000, &Deterministic::new(42)).unwrap())
  });
  let amh = Amh::new(0.5);
  group.bench_function("amh_n10k", |b| {
    b.iter(|| amh.sample(10_000, &Deterministic::new(42)).unwrap())
  });
  let fgm = Fgm::new(0.5);
  group.bench_function("fgm_n10k", |b| {
    b.iter(|| fgm.sample(10_000, &Deterministic::new(42)).unwrap())
  });
  let plackett = Plackett::new(3.0);
  group.bench_function("plackett_n10k", |b| {
    b.iter(|| plackett.sample(10_000, &Deterministic::new(42)).unwrap())
  });
  let t = TCopula::new(0.5, 4.0);
  group.bench_function("t_copula_n256", |b| {
    b.iter(|| t.sample(256, &Deterministic::new(42)).unwrap())
  });
  group.finish();
}

fn multivariate_sample(c: &mut Criterion) {
  let mut group = c.benchmark_group("copulas/multivariate_sample");
  let gaussian = GaussianMultivariate::new(corr(5));
  group.bench_function("gaussian_d5_n10k", |b| {
    b.iter(|| gaussian.sample(10_000, &Deterministic::new(42)).unwrap())
  });
  let t = TMultivariate::new(corr(5), 5.0);
  group.bench_function("t_d5_n10k", |b| {
    b.iter(|| t.sample(10_000, &Deterministic::new(42)).unwrap())
  });
  let vine = dvine(4);
  group.bench_function("dvine_d4_n10k", |b| {
    b.iter(|| vine.sample(10_000, &Deterministic::new(42)).unwrap())
  });
  group.finish();
}

fn multivariate_cdf(c: &mut Criterion) {
  let mut group = c.benchmark_group("copulas/multivariate_cdf");
  group.sample_size(20);
  let q = queries(3);
  let gaussian = GaussianMultivariate::new(corr(3));
  group.bench_function("gaussian_d3_16q", |b| {
    b.iter(|| gaussian.cdf(q.view()).unwrap())
  });
  let t = TMultivariate::new(corr(3), 5.0);
  group.bench_function("t_d3_16q", |b| b.iter(|| t.cdf(q.view()).unwrap()));
  let vine = dvine(3);
  group.bench_function("dvine_d3_16q", |b| b.iter(|| vine.cdf(q.view()).unwrap()));
  group.finish();
}

fn fit(c: &mut Criterion) {
  let mut group = c.benchmark_group("copulas/fit");
  group.sample_size(20);
  group.measurement_time(Duration::from_secs(9));
  let clayton = Clayton::new(2.0);
  let pairs = pseudo_observations(
    clayton
      .sample(2_000, &Deterministic::new(1))
      .unwrap()
      .view(),
  );
  group.bench_function("clayton_fit_n2k", |b| {
    b.iter(|| {
      let mut fitted = Clayton::default();
      fitted.fit(pairs.view()).unwrap()
    })
  });
  let plackett = Plackett::new(3.0);
  let plackett_pairs = pseudo_observations(
    plackett
      .sample(2_000, &Deterministic::new(1))
      .unwrap()
      .view(),
  );
  group.bench_function("plackett_fit_n2k", |b| {
    b.iter(|| {
      let mut fitted = Plackett::default();
      fitted.fit(plackett_pairs.view()).unwrap()
    })
  });
  let frank_pairs = pseudo_observations(
    Frank::new(4.0)
      .sample(2_000, &Deterministic::new(1))
      .unwrap()
      .view(),
  );
  group.bench_function("frank_fit_n2k", |b| {
    b.iter(|| {
      let mut fitted = Frank::default();
      fitted.fit(frank_pairs.view()).unwrap()
    })
  });
  let galambos_pairs = pseudo_observations(
    Galambos::new(1.5)
      .sample(2_000, &Deterministic::new(1))
      .unwrap()
      .view(),
  );
  group.bench_function("galambos_fit_n2k", |b| {
    b.iter(|| {
      let mut fitted = Galambos::default();
      fitted.fit(galambos_pairs.view()).unwrap()
    })
  });
  let data = pseudo_observations(
    dvine(4)
      .sample(1_000, &Deterministic::new(1))
      .unwrap()
      .view(),
  );
  group.bench_function("vine_fit_d4_n1k", |b| {
    b.iter(|| {
      fit_vine(
        data.view(),
        VineStructure::DVine,
        BivariateFamily::VINE_CANDIDATES,
        SelectionCriterion::Aic,
      )
      .unwrap()
    })
  });
  group.bench_function("kendall_tau_d4_n1k", |b| {
    b.iter(|| kendall_tau(data.view()))
  });
  group.bench_function("gaussian_fit_d4_n1k", |b| {
    b.iter(|| {
      let mut fitted = GaussianMultivariate::default();
      fitted.fit(data.view()).unwrap()
    })
  });
  group.bench_function("t_fit_d4_n1k", |b| {
    b.iter(|| {
      let mut fitted = TMultivariate::default();
      fitted.fit(data.view()).unwrap()
    })
  });
  group.finish();
}

fn gof(c: &mut Criterion) {
  let mut group = c.benchmark_group("copulas/gof");
  group.sample_size(10);
  let clayton = Clayton::new(2.0);
  let pairs = pseudo_observations(clayton.sample(300, &Deterministic::new(1)).unwrap().view());
  let mut fitted = Clayton::default();
  fitted.fit(pairs.view()).unwrap();
  group.bench_function("clayton_cvm_n300_r10", |b| {
    b.iter(|| {
      gof_cramer_von_mises(
        &fitted,
        pairs.view(),
        GofConfig {
          replications: 10,
          alpha: 0.05,
        },
        &Deterministic::new(100),
        |c, x| c.fit(x),
      )
      .unwrap()
    })
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

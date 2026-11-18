use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use stochastic_rs::quant::calibration::BlackKarasinskiParams;
use stochastic_rs::quant::calibration::G2ppParams;
use stochastic_rs::quant::calibration::HullWhiteParams;
use stochastic_rs::quant::lattice::BlackKarasinskiTree;
use stochastic_rs::quant::lattice::BlackKarasinskiTreeModel;
use stochastic_rs::quant::lattice::G2ppTree;
use stochastic_rs::quant::lattice::G2ppTreeModel;
use stochastic_rs::quant::lattice::HullWhiteTree;
use stochastic_rs::quant::lattice::HullWhiteTreeModel;

fn bench_hull_white(c: &mut Criterion) {
  let tree = HullWhiteTree::new(
    HullWhiteTreeModel::new(HullWhiteParams {
      mean_reversion: 0.4,
      sigma: 0.02,
      initial_rate: 0.03,
      theta: 0.05,
    }),
    5.0,
    400,
  );
  c.bench_function("lattice_hull_white_zcb", |b| {
    b.iter(|| tree.zero_coupon_bond_price())
  });
}

fn bench_black_karasinski(c: &mut Criterion) {
  let tree = BlackKarasinskiTree::new(
    BlackKarasinskiTreeModel::new(BlackKarasinskiParams {
      mean_reversion: 0.6,
      sigma: 0.015,
      initial_rate: 0.03,
      long_run_rate: 0.05,
    }),
    5.0,
    400,
  );
  c.bench_function("lattice_black_karasinski_zcb", |b| {
    b.iter(|| tree.zero_coupon_bond_price())
  });
}

fn bench_g2pp(c: &mut Criterion) {
  let tree = G2ppTree::new(
    G2ppTreeModel::new(
      0.01,
      0.015,
      G2ppParams {
        mean_reversion_x: 0.4,
        mean_reversion_y: 0.8,
        sigma_x: 0.02,
        sigma_y: 0.015,
        rho: -0.3,
        phi: 0.005,
      },
    ),
    5.0,
    240,
  );
  c.bench_function("lattice_g2pp_zcb", |b| {
    b.iter(|| tree.zero_coupon_bond_price())
  });
}

criterion_group!(
  benches,
  bench_hull_white,
  bench_black_karasinski,
  bench_g2pp
);
criterion_main!(benches);

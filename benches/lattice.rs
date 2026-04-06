use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
use stochastic_rs::quant::lattice::BlackKarasinskiTree;
use stochastic_rs::quant::lattice::BlackKarasinskiTreeModel;
use stochastic_rs::quant::lattice::G2ppTree;
use stochastic_rs::quant::lattice::G2ppTreeModel;
use stochastic_rs::quant::lattice::HullWhiteTree;
use stochastic_rs::quant::lattice::HullWhiteTreeModel;

fn bench_hull_white(c: &mut Criterion) {
  let tree = HullWhiteTree::new(HullWhiteTreeModel::new(0.03, 0.4, 0.05, 0.02), 5.0, 400);
  c.bench_function("lattice_hull_white_zcb", |b| {
    b.iter(|| tree.zero_coupon_bond_price())
  });
}

fn bench_black_karasinski(c: &mut Criterion) {
  let tree = BlackKarasinskiTree::new(
    BlackKarasinskiTreeModel::new(0.03, 0.6, 0.05, 0.015),
    5.0,
    400,
  );
  c.bench_function("lattice_black_karasinski_zcb", |b| {
    b.iter(|| tree.zero_coupon_bond_price())
  });
}

fn bench_g2pp(c: &mut Criterion) {
  let tree = G2ppTree::new(
    G2ppTreeModel::new(0.01, 0.015, 0.005, 0.4, 0.8, 0.02, 0.015, -0.3),
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

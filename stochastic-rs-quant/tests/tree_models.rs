//! Short-rate tree models built from the calibrators' parameter structs, and what prices on them.

use stochastic_rs_quant::calibration::hw_swaption::HullWhiteParams;
use stochastic_rs_quant::calibration::tree_swaption::BlackKarasinskiParams;
use stochastic_rs_quant::calibration::tree_swaption::G2ppParams;
use stochastic_rs_quant::lattice::BlackKarasinskiTree;
use stochastic_rs_quant::lattice::BlackKarasinskiTreeModel;
use stochastic_rs_quant::lattice::G2ppTree;
use stochastic_rs_quant::lattice::G2ppTreeModel;
use stochastic_rs_quant::lattice::HullWhiteTree;
use stochastic_rs_quant::lattice::HullWhiteTreeModel;

const HW: HullWhiteParams = HullWhiteParams {
  mean_reversion: 0.1,
  sigma: 0.012,
  initial_rate: 0.03,
  theta: 0.05,
};

const BK: BlackKarasinskiParams = BlackKarasinskiParams {
  mean_reversion: 0.6,
  sigma: 0.15,
  initial_rate: 0.03,
  long_run_rate: 0.05,
};

const G2: G2ppParams = G2ppParams {
  mean_reversion_x: 0.1,
  mean_reversion_y: 0.5,
  sigma_x: 0.01,
  sigma_y: 0.005,
  rho: -0.3,
  phi: 0.04,
};

fn hw_tree() -> HullWhiteTree<f64> {
  HullWhiteTree::new(HullWhiteTreeModel::new(HW), 5.0, 20)
}

fn bk_tree() -> BlackKarasinskiTree<f64> {
  BlackKarasinskiTree::new(BlackKarasinskiTreeModel::new(BK), 5.0, 20)
}

fn g2_tree() -> G2ppTree<f64> {
  G2ppTree::new(G2ppTreeModel::new(0.002, -0.001, G2), 5.0, 20)
}

#[test]
fn a_tree_model_reads_each_field_of_its_params() {
  let hw = HullWhiteTreeModel::<f64>::new(HW);
  assert_eq!(
    (hw.initial_rate, hw.mean_reversion, hw.theta, hw.sigma),
    (0.03, 0.1, 0.05, 0.012)
  );
  let bk = BlackKarasinskiTreeModel::<f64>::new(BK);
  assert_eq!(
    (bk.initial_rate, bk.mean_reversion, bk.theta_log, bk.sigma),
    (0.03, 0.6, 0.05_f64.ln(), 0.15)
  );
  let g2 = G2ppTreeModel::<f64>::new(0.002, -0.001, G2);
  assert_eq!((g2.initial_x, g2.initial_y, g2.phi), (0.002, -0.001, 0.04));
  assert_eq!((g2.mean_reversion_x, g2.mean_reversion_y), (0.1, 0.5));
  assert_eq!((g2.sigma_x, g2.sigma_y, g2.rho), (0.01, 0.005, -0.3));
  let single = HullWhiteTreeModel::<f32>::new(HW);
  assert_eq!((single.initial_rate, single.sigma), (0.03_f32, 0.012_f32));
}

/// The values the positional constructors gave for the same numbers, bit for bit.
#[test]
fn the_trees_price_a_zero_coupon_bond_as_before() {
  assert_eq!(hw_tree().zero_coupon_bond_price(), HW_ZCB);
  assert_eq!(bk_tree().zero_coupon_bond_price(), BK_ZCB);
  assert_eq!(g2_tree().zero_coupon_bond_price(), G2_ZCB);
}

const HW_ZCB: f64 = 0.844_900_258_875_831_1;
const BK_ZCB: f64 = 0.805_747_726_783_313_6;
const G2_ZCB: f64 = 0.814_796_346_157_607;

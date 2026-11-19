//! Short-rate tree models built from the calibrators' parameter structs, and what prices on them.

use chrono::NaiveDate;
use stochastic_rs_quant::calendar::DayCountConvention;
use stochastic_rs_quant::calibration::hw_swaption::HullWhiteParams;
use stochastic_rs_quant::calibration::tree_swaption::BlackKarasinskiParams;
use stochastic_rs_quant::calibration::tree_swaption::G2ppParams;
use stochastic_rs_quant::instruments::BermudanDates;
use stochastic_rs_quant::instruments::BermudanSwaption;
use stochastic_rs_quant::instruments::BermudanSwaptionValuation;
use stochastic_rs_quant::instruments::BermudanTree;
use stochastic_rs_quant::instruments::ExerciseSchedule;
use stochastic_rs_quant::instruments::SwaptionDirection;
use stochastic_rs_quant::instruments::TreeCouponSchedule;
use stochastic_rs_quant::instruments::option::bermudan::one_factor_valuation;
use stochastic_rs_quant::lattice::BlackKarasinskiTree;
use stochastic_rs_quant::lattice::BlackKarasinskiTreeModel;
use stochastic_rs_quant::lattice::CurveFittedHullWhiteTree;
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

fn swaption(direction: SwaptionDirection) -> BermudanSwaption<f64> {
  let levels: Vec<usize> = (4..=20).step_by(4).collect();
  BermudanSwaption::new(
    direction,
    0.045,
    100.0,
    ExerciseSchedule::new(levels[..4].to_vec()),
    TreeCouponSchedule::new(levels, vec![1.0; 5]),
  )
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

/// rc.4's values, to the last bits libm may change between platforms.
fn assert_pinned(got: f64, pinned: f64) {
  let close = (got - pinned).abs() <= 1e-13 * pinned.abs().max(1.0);
  assert!(close, "{got} against {pinned}");
}

#[test]
fn the_trees_price_a_zero_coupon_bond_as_before() {
  assert_pinned(hw_tree().zero_coupon_bond_price(), HW_ZCB);
  assert_pinned(bk_tree().zero_coupon_bond_price(), BK_ZCB);
  assert_pinned(g2_tree().zero_coupon_bond_price(), G2_ZCB);
}

/// Every tree answers one `valuation`, at rc.4's prices.
#[test]
fn a_bermudan_swaption_prices_on_any_bermudan_tree() {
  let pins = [
    (SwaptionDirection::Payer, HW_PAYER, BK_PAYER, G2_PAYER),
    (
      SwaptionDirection::Receiver,
      HW_RECEIVER,
      BK_RECEIVER,
      G2_RECEIVER,
    ),
  ];
  for (direction, hw, bk, g2) in pins {
    let swaption = swaption(direction);
    assert_pinned(swaption.npv(&hw_tree()), hw);
    assert_pinned(swaption.npv(&bk_tree()), bk);
    let on_g2 = swaption.npv(&g2_tree());
    assert_pinned(on_g2, g2);
    let valuation = swaption.valuation(&g2_tree());
    assert_eq!((valuation.npv, valuation.exercise_count), (on_g2, 4));
  }
}

#[test]
fn a_curve_fitted_tree_and_a_generic_caller_price_through_the_trait() {
  fn npv_on(tree: &impl BermudanTree<f64>, swaption: &BermudanSwaption<f64>) -> f64 {
    tree.bermudan_valuation(swaption).npv
  }
  let payer = swaption(SwaptionDirection::Payer);
  assert_eq!(npv_on(&hw_tree(), &payer), payer.npv(&hw_tree()));
  assert_pinned(npv_on(&hw_tree(), &payer), HW_PAYER);
  let curve = stochastic_rs_quant::curves::DiscountCurve::from_zero_rates(
    ndarray::array![1.0, 5.0].view(),
    ndarray::array![0.03, 0.035].view(),
    stochastic_rs_quant::curves::InterpolationMethod::LinearOnZeroRates,
  );
  let fitted = CurveFittedHullWhiteTree::new(0.1, 0.012, &curve, 5.0, 20);
  assert_pinned(payer.npv(&fitted), 0.1040235044511398);
}

struct OwnTree(HullWhiteTree<f64>);

impl BermudanTree<f64> for OwnTree {
  fn bermudan_valuation(&self, swaption: &BermudanSwaption<f64>) -> BermudanSwaptionValuation<f64> {
    one_factor_valuation(swaption, &self.0.tree, &self.0.model)
  }
}

#[test]
fn a_callers_own_tree_prices_through_the_one_factor_helper() {
  let payer = swaption(SwaptionDirection::Payer);
  assert_eq!(payer.npv(&OwnTree(hw_tree())), payer.npv(&hw_tree()));
}

#[test]
fn a_bermudan_swaption_snaps_its_dates_to_tree_levels() {
  let d = |y, m, day| NaiveDate::from_ymd_opt(y, m, day).unwrap();
  let dates = BermudanDates {
    valuation_date: d(2024, 1, 15),
    day_count: DayCountConvention::Actual365Fixed,
    exercise_dates: (1..=4).map(|i| d(2024 + i, 1, 15)).collect(),
    coupon_dates: (1..=5).map(|i| d(2024 + i, 1, 15)).collect(),
    accrual_factors: vec![1.0; 5],
  };
  let swaption = BermudanSwaption::from_dates(SwaptionDirection::Payer, 0.045, 100.0, &dates, 0.25);
  assert_eq!(swaption.exercise_schedule.levels, [4, 8, 12, 16]);
  assert_eq!(swaption.coupon_schedule.levels, [4, 8, 12, 16, 20]);
  let on_levels = crate::swaption(SwaptionDirection::Payer);
  assert_eq!(swaption.npv(&hw_tree()), on_levels.npv(&hw_tree()));
  assert_pinned(swaption.npv(&hw_tree()), HW_PAYER);
}

const HW_ZCB: f64 = 0.844_900_258_875_831_1;
const BK_ZCB: f64 = 0.805_747_726_783_313_6;
const G2_ZCB: f64 = 0.814_796_346_157_607;
const HW_PAYER: f64 = 0.063_431_582_574_042_65;
const HW_RECEIVER: f64 = 8.160_819_002_420_86;
const BK_PAYER: f64 = 5.792_960_501_350_106_4e-5;
const BK_RECEIVER: f64 = 3.971_411_064_801_325;
const G2_PAYER: f64 = 0.061_143_868_557_397_83;
const G2_RECEIVER: f64 = 5.949_552_713_380_301;

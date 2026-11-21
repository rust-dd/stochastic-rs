//! The short-rate trees converge to their models' closed forms: Vasicek's bond and Jamshidian's
//! swaption for Hull–White, the closed-form bond for G2++, Jamshidian on the fitted curve.

use ndarray::Array1;
use stochastic_rs_quant::bonds::Vasicek;
use stochastic_rs_quant::calibration::G2ppParams;
use stochastic_rs_quant::calibration::HullWhiteParams;
use stochastic_rs_quant::curves::DiscountCurve;
use stochastic_rs_quant::curves::InterpolationMethod;
use stochastic_rs_quant::instruments::BermudanSwaption;
use stochastic_rs_quant::instruments::ExerciseSchedule;
use stochastic_rs_quant::instruments::SwaptionDirection;
use stochastic_rs_quant::instruments::TreeCouponSchedule;
use stochastic_rs_quant::instruments::option::jamshidian::price_jamshidian_hull_white;
use stochastic_rs_quant::lattice::CurveFittedHullWhiteTree;
use stochastic_rs_quant::lattice::G2ppTree;
use stochastic_rs_quant::lattice::G2ppTreeModel;
use stochastic_rs_quant::lattice::HullWhiteTree;
use stochastic_rs_quant::lattice::HullWhiteTreeModel;
use stochastic_rs_quant::traits::ShortRatePricer;

const HW: HullWhiteParams = HullWhiteParams {
  mean_reversion: 0.1,
  sigma: 0.012,
  initial_rate: 0.03,
  theta: 0.05,
};

/// Each refinement halves the error of a first-order tree.
fn assert_first_order(errors: &[f64], last: f64) {
  for pair in errors.windows(2) {
    let ratio = pair[0] / pair[1];
    assert!((1.9..2.1).contains(&ratio), "{errors:?}");
  }
  let end = errors[errors.len() - 1];
  assert!(end.abs() < last, "{errors:?}");
}

/// The tree discounts at each step's start, so its bond is `O(Δt)` off: `+7.2e-4` at 20 steps.
#[test]
fn the_hull_white_bond_converges_at_first_order_to_vasiceks() {
  let vasicek = Vasicek {
    theta: 0.1,
    mu: 0.05,
    sigma: 0.012,
  }
  .zero_coupon_price(0.03, 5.0);
  let errors = [20, 40, 80, 160].map(|steps| {
    let tree = HullWhiteTree::new(HullWhiteTreeModel::new(HW), 5.0, steps);
    tree.zero_coupon_bond_price() - vasicek
  });
  assert_first_order(&errors, 1e-4);
}

/// `exp(−φT − B_a x₀ − B_b y₀ + V/2)` for a constant `φ`, `V = Var ∫₀ᵀ (x + y) dt`.
fn g2pp_bond(p: &G2ppParams, x0: f64, y0: f64, t: f64) -> f64 {
  let (a, b, s, e) = (p.mean_reversion_x, p.mean_reversion_y, p.sigma_x, p.sigma_y);
  let one = |k: f64| t + 2.0 / k * (-k * t).exp() - (-2.0 * k * t).exp() / (2.0 * k) - 1.5 / k;
  let cross = t + (-a * t).exp_m1() / a + (-b * t).exp_m1() / b - (-(a + b) * t).exp_m1() / (a + b);
  let v =
    s * s / (a * a) * one(a) + e * e / (b * b) * one(b) + 2.0 * p.rho * s * e / (a * b) * cross;
  let (ba, bb) = (-(-a * t).exp_m1() / a, -(-b * t).exp_m1() / b);
  (-p.phi * t - ba * x0 - bb * y0 + v / 2.0).exp()
}

#[test]
fn the_g2pp_bond_converges_at_first_order_to_brigo_mercurios() {
  let params = G2ppParams {
    mean_reversion_x: 0.1,
    mean_reversion_y: 0.5,
    sigma_x: 0.01,
    sigma_y: 0.005,
    rho: -0.3,
    phi: 0.04,
  };
  let closed = g2pp_bond(&params, 0.002, -0.001, 5.0);
  let errors = [20, 40, 80, 160].map(|steps| {
    let tree = G2ppTree::new(G2ppTreeModel::new(0.002, -0.001, params), 5.0, steps);
    tree.zero_coupon_bond_price() - closed
  });
  assert_first_order(&errors, 1e-5);
}

/// A payer exercised at year 1 into a swap paying `0.5·K` at years 1.5 to 3, the tree's horizon.
fn european(steps_per_year: usize, strike: f64) -> BermudanSwaption<f64> {
  let level = |t: f64| (t * steps_per_year as f64).round() as usize;
  BermudanSwaption::new(
    SwaptionDirection::Payer,
    strike,
    100.0,
    ExerciseSchedule::new(vec![level(1.0)]),
    TreeCouponSchedule::new([1.5, 2.0, 2.5, 3.0].map(level).to_vec(), vec![0.5; 4]),
  )
}

/// At-the-money strike and Jamshidian's price on `curve`, the reference of the tree's European.
fn jamshidian_atm(curve: &DiscountCurve<f64>, a: f64, sigma: f64) -> (f64, f64) {
  let coupons = [1.5, 2.0, 2.5, 3.0];
  let annuity = coupons
    .iter()
    .map(|&t| 0.5 * curve.discount_factor(t))
    .sum::<f64>();
  let strike = (curve.discount_factor(1.0) - curve.discount_factor(3.0)) / annuity;
  let price = price_jamshidian_hull_white(
    SwaptionDirection::Payer,
    strike,
    100.0,
    1.0,
    &coupons,
    &[0.5; 4],
    a,
    sigma,
    curve,
  );
  (strike, price)
}

/// On Vasicek's own curve Jamshidian prices the constant-`θ` model: `−6.4e-2` at 4 steps a year.
#[test]
fn the_hull_white_european_swaption_converges_to_jamshidians() {
  let vasicek = Vasicek {
    theta: 0.1,
    mu: 0.05,
    sigma: 0.012,
  };
  let times = [0.25, 1.0, 1.5, 2.0, 2.5, 3.0];
  let rates = times.map(|t| -vasicek.zero_coupon_price(0.03, t).ln() / t);
  let curve = DiscountCurve::from_zero_rates(
    Array1::from_vec(times.to_vec()).view(),
    Array1::from_vec(rates.to_vec()).view(),
    InterpolationMethod::LogLinearOnDiscountFactors,
  );
  let (strike, closed) = jamshidian_atm(&curve, 0.1, 0.012);
  let errors = [4, 8, 16, 32, 64].map(|per_year| {
    let tree = HullWhiteTree::new(HullWhiteTreeModel::new(HW), 3.0, 3 * per_year);
    european(per_year, strike).npv(&tree) - closed
  });
  assert!(
    errors.windows(2).all(|w| w[1].abs() < w[0].abs()),
    "{errors:?}"
  );
  assert!(errors[4].abs() < 3e-3, "{errors:?}");
}

#[test]
fn the_curve_fitted_european_swaption_converges_to_jamshidians_on_its_curve() {
  let curve = DiscountCurve::from_zero_rates(
    ndarray::array![1.0, 5.0].view(),
    ndarray::array![0.03, 0.035].view(),
    InterpolationMethod::LinearOnZeroRates,
  );
  let (strike, closed) = jamshidian_atm(&curve, 0.1, 0.012);
  let errors = [4, 8, 16, 32, 64].map(|per_year| {
    let tree = CurveFittedHullWhiteTree::new(0.1, 0.012, &curve, 3.0, 3 * per_year);
    european(per_year, strike).npv(&tree) - closed
  });
  assert!(
    errors.windows(2).all(|w| w[1].abs() < w[0].abs()),
    "{errors:?}"
  );
  assert!(errors[4].abs() < 2e-3, "{errors:?}");
}

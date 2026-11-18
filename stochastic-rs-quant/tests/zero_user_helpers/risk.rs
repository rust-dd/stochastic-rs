use ndarray::array;
use stochastic_rs_quant::credit::survival_curve::HazardInterpolation;
use stochastic_rs_quant::credit::survival_curve::SurvivalCurve;
use stochastic_rs_quant::risk::PnlOrLoss;
use stochastic_rs_quant::risk::Scenario;
use stochastic_rs_quant::risk::drawdown::max_peak_to_recovery_duration;
use stochastic_rs_quant::risk::expected_shortfall::gaussian_var_es;
use stochastic_rs_quant::risk::forward_difference;
use stochastic_rs_quant::risk::gaussian_es;
use stochastic_rs_quant::risk::gaussian_var;
use stochastic_rs_quant::risk::greeks::second_difference;
use stochastic_rs_quant::risk::probability_of_default_before;

/// On `x²` the forward difference is `2x + h`, its first-order error in full; on a line it is
/// the slope.
#[test]
fn forward_difference_is_the_slope_plus_h_on_a_parabola() {
  let (x, h) = (3.0_f64, 0.5);
  assert_eq!(forward_difference(|v| v * v, x, h), 2.0 * x + h);
  assert_eq!(forward_difference(|v| 2.0 * v + 1.0, x, h), 2.0);
}

/// `((x+h)³ − 2x³ + (x−h)³) / h² = 6x`: the second difference has no error on a cubic.
#[test]
fn second_difference_is_exact_on_a_cubic() {
  let (x, h) = (2.0_f64, 0.5);
  assert_eq!(second_difference(|v| v * v * v, x, h), 6.0 * x);
}

#[test]
fn gaussian_var_es_is_the_pair_of_its_parts() {
  let pnl = array![0.012, -0.034, 0.005, -0.021, 0.017, -0.002, 0.009];
  for orientation in [PnlOrLoss::Pnl, PnlOrLoss::Loss] {
    let (var, es) = gaussian_var_es(pnl.view(), 0.99, orientation).unwrap();
    assert_eq!(var, gaussian_var(pnl.view(), 0.99, orientation).unwrap());
    assert_eq!(es, gaussian_es(pnl.view(), 0.99, orientation).unwrap());
  }
}

/// At a flat hazard `h` the default probability before `t` is `1 − e^{−ht}`.
#[test]
fn probability_of_default_before_is_one_minus_survival() {
  let hazard = 0.02_f64;
  let curve = SurvivalCurve::from_hazard_rates(
    &array![1.0, 5.0],
    &array![hazard, hazard],
    HazardInterpolation::PiecewiseConstantHazard,
  );
  for t in [0.5, 1.0, 3.0, 5.0] {
    let pd = probability_of_default_before(&curve, t);
    assert_eq!(pd, 1.0 - curve.survival_probability(t));
    assert!(
      (pd - (1.0 - (-hazard * t).exp())).abs() < 1e-14,
      "t = {t}: {pd}"
    );
  }
}

#[test]
fn tagged_appends_its_tag_and_an_untagged_scenario_has_none() {
  assert!(Scenario::<f64>::new("base").tags.is_empty());
  let scenario = Scenario::<f64>::new("stress").tagged("EBA").tagged("2020");
  assert_eq!(scenario.tags, ["EBA", "2020"]);
}

/// A peak at index 0 regained at index 4 is four observations; an unrecovered tail counts
/// `n − 1 − peak_index`, and a rising series has no drawdown.
#[test]
fn max_peak_to_recovery_duration_counts_from_the_peak_to_its_recovery() {
  let duration = |equity: ndarray::Array1<f64>| max_peak_to_recovery_duration(equity.view());
  assert_eq!(duration(array![100.0, 90.0, 80.0, 95.0, 101.0]).unwrap(), 4);
  assert_eq!(
    duration(array![90.0, 100.0, 90.0, 80.0]).unwrap(),
    4 - 1 - 1
  );
  assert_eq!(duration(array![1.0, 2.0, 3.0]).unwrap(), 0);
}

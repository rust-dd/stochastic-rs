use ndarray::Array1;
use ndarray::s;
use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::calibration::bsm::BSMCalibrator;
use stochastic_rs_quant::calibration::bsm::BSMParams;
use stochastic_rs_quant::calibration::hkde::HKDECalibrator;
use stochastic_rs_quant::calibration::least_squares::LeastSquaresProblem;
use stochastic_rs_quant::pricing::bsm::BSMCoc;
use stochastic_rs_quant::pricing::bsm::BSMPricer;
use stochastic_rs_quant::traits::Calibrator;
use stochastic_rs_quant::traits::ModelPricer;

fn strided(values: impl IntoIterator<Item = f64>) -> Array1<f64> {
  Array1::from_iter(values.into_iter().flat_map(|v| [v, f64::NAN])).slice_move(s![..;2])
}

fn reversed(values: impl IntoIterator<Item = f64>) -> Array1<f64> {
  let values = values.into_iter().collect::<Vec<_>>();
  Array1::from_iter(values.into_iter().rev()).slice_move(s![..;-1])
}

#[test]
fn bsm_calibration_accepts_strided_quotes() {
  let strikes = strided([90.0, 100.0, 110.0]);
  let pricer = BSMPricer::new(0.25, BSMCoc::Bsm1973);
  let market = strided(
    strikes
      .iter()
      .map(|&k| pricer.price_call(100.0, k, 0.03, 0.0, 1.0)),
  );
  assert!(market.as_slice().is_none());
  let model = BSMCalibrator::new(
    BSMParams { v: 0.3 },
    market,
    strided([100.0; 3]),
    strikes,
    0.03,
    None,
    None,
    None,
    1.0,
    OptionType::Call,
  );
  let result = model.calibrate(None).unwrap();
  assert!((result.v - 0.25).abs() < 1e-6);
}

#[test]
fn bsm_calibration_preserves_reversed_quote_order() {
  let pricer = BSMPricer::new(0.25, BSMCoc::Bsm1973);
  let strikes = Array1::from_vec(vec![90.0, 100.0, 110.0]);
  let market = reversed(
    strikes
      .iter()
      .map(|&k| pricer.price_call(100.0, k, 0.03, 0.0, 1.0)),
  );
  assert!(market.as_slice().is_none());
  let model = BSMCalibrator::new(
    BSMParams { v: 0.3 },
    market,
    Array1::from_elem(3, 100.0),
    strikes,
    0.03,
    None,
    None,
    None,
    1.0,
    OptionType::Call,
  );
  let result = model.calibrate(None).unwrap();
  assert!((result.v - 0.25).abs() < 1e-6);
  assert!(result.loss.get(stochastic_rs_quant::LossMetric::Rmse) < 1e-6);
}

#[test]
fn hkde_weights_accept_strided_quotes() {
  let model = HKDECalibrator::new(
    None,
    strided([15.0, 9.0, 4.0]),
    strided([100.0; 3]),
    strided([90.0, 100.0, 110.0]),
    0.03,
    None,
    1.0,
    OptionType::Call,
    true,
  );
  assert_eq!(model.c_market.len(), 3);
  assert!(model.residuals().unwrap().iter().all(|x| x.is_finite()));
}

#[test]
fn hkde_weights_preserve_reversed_quote_order() {
  let contiguous = HKDECalibrator::new(
    None,
    Array1::from_vec(vec![15.0, 9.0, 4.0]),
    Array1::from_elem(3, 100.0),
    Array1::from_vec(vec![90.0, 100.0, 110.0]),
    0.03,
    None,
    1.0,
    OptionType::Call,
    true,
  );
  let reversed = HKDECalibrator::new(
    None,
    reversed([15.0, 9.0, 4.0]),
    reversed([100.0; 3]),
    reversed([90.0, 100.0, 110.0]),
    0.03,
    None,
    1.0,
    OptionType::Call,
    true,
  );
  assert_eq!(reversed.sqrt_weights, contiguous.sqrt_weights);
  assert_eq!(reversed.residuals(), contiguous.residuals());
}

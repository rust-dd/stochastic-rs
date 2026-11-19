use ndarray::Array1;
use ndarray::ArrayView1;
use stochastic_rs_stats::realized::KernelType;
use stochastic_rs_stats::realized::bipower_variation;
use stochastic_rs_stats::realized::medrv;
use stochastic_rs_stats::realized::minrv;
use stochastic_rs_stats::realized::multi_scale_rv;
use stochastic_rs_stats::realized::pre_averaged_variance;
use stochastic_rs_stats::realized::pre_averaging::pre_averaged_variance_with_block;
use stochastic_rs_stats::realized::realized_kernel;
use stochastic_rs_stats::realized::realized_kurtosis;
use stochastic_rs_stats::realized::realized_quarticity;
use stochastic_rs_stats::realized::realized_semivariance;
use stochastic_rs_stats::realized::realized_skewness;
use stochastic_rs_stats::realized::realized_variance;
use stochastic_rs_stats::realized::realized_volatility;
use stochastic_rs_stats::realized::tripower_quarticity;
use stochastic_rs_stats::realized::two_scale_rv;

type Measure = Box<dyn Fn(ArrayView1<f64>) -> f64>;

fn measure(f: impl Fn(ArrayView1<f64>) -> f64 + 'static) -> Measure {
  Box::new(f)
}

/// Every pure realized measure with the fewest inputs it answers a number on (prices for the two
/// subsampling estimators, returns otherwise).
fn realized_measures() -> Vec<(&'static str, usize, Measure)> {
  vec![
    ("realized_variance", 1, measure(realized_variance)),
    (
      "realized_volatility",
      1,
      measure(|x| realized_volatility(x, 252.0)),
    ),
    (
      "semivariance_down",
      1,
      measure(|x| realized_semivariance(x, 0.0).0),
    ),
    (
      "semivariance_up",
      1,
      measure(|x| realized_semivariance(x, 0.0).1),
    ),
    ("realized_quarticity", 1, measure(realized_quarticity)),
    ("realized_skewness", 2, measure(realized_skewness)),
    ("realized_kurtosis", 2, measure(realized_kurtosis)),
    ("bipower_variation", 2, measure(bipower_variation)),
    ("minrv", 2, measure(minrv)),
    (
      "realized_kernel",
      2,
      measure(|x| realized_kernel(x, KernelType::Parzen, 1)),
    ),
    ("medrv", 3, measure(medrv)),
    ("tripower_quarticity", 3, measure(tripower_quarticity)),
    (
      "pre_averaged_variance",
      4,
      measure(|x| pre_averaged_variance(x, 0.5)),
    ),
    (
      "pre_averaged_with_block",
      4,
      measure(|x| pre_averaged_variance_with_block(x, 4)),
    ),
    ("two_scale_rv", 4, measure(|x| two_scale_rv(x, 2))),
    ("multi_scale_rv", 4, measure(|x| multi_scale_rv(x, 2))),
  ]
}

#[test]
fn realized_measures_answer_nan_below_their_minimum_length_and_on_a_nan() {
  let returns = Array1::<f64>::linspace(-0.01, 0.0137, 50);
  let mut with_nan = returns.clone();
  with_nan[3] = f64::NAN;
  for (name, min_len, f) in realized_measures() {
    let below = f(returns.slice(ndarray::s![..min_len - 1]));
    assert!(below.is_nan(), "{name}: {below} below {min_len} inputs");
    let at = f(returns.slice(ndarray::s![..min_len]));
    assert!(at.is_finite(), "{name}: {at} at {min_len} inputs");
    assert!(
      f(with_nan.view()).is_nan(),
      "{name}: a NaN input must give NaN"
    );
  }
  let zeros = Array1::<f64>::zeros(5);
  assert!(realized_skewness(zeros.view()).is_nan() && realized_kurtosis(zeros.view()).is_nan());
}

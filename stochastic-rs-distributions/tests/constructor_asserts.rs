//! A rejected argument is named in its own assert: ``<arg> must satisfy `<pred>`, got <arg> = …``.

use std::panic::UnwindSafe;
use std::panic::catch_unwind;

use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_distributions::FloatExt;
use stochastic_rs_distributions::beta::SimdBeta;
use stochastic_rs_distributions::gamma::SimdGamma;
use stochastic_rs_distributions::generalized_hyperbolic::SimdGeneralizedHyperbolic;
use stochastic_rs_distributions::generalized_inverse_gauss::SimdGig;
use stochastic_rs_distributions::inverse_gauss::SimdInverseGauss;
use stochastic_rs_distributions::johnson_su::SimdJohnsonSu;
use stochastic_rs_distributions::normal_inverse_gauss::SimdNormalInverseGauss;
use stochastic_rs_distributions::pareto::SimdPareto;
use stochastic_rs_distributions::skew_t::SimdSkewT;
use stochastic_rs_distributions::tempered_stable::SimdTemperedStable;
use stochastic_rs_distributions::traits::Grid2D;
use stochastic_rs_distributions::truncated::SimdTruncatedBeta;
use stochastic_rs_distributions::uniform::SimdUniform;
use stochastic_rs_distributions::variance_gamma::SimdVarianceGamma;
use stochastic_rs_distributions::weibull::SimdWeibull;

fn panic_text<R>(f: impl FnOnce() -> R + UnwindSafe) -> String {
  let payload = catch_unwind(f)
    .err()
    .expect("the call accepted an invalid argument");
  payload
    .downcast_ref::<String>()
    .cloned()
    .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
    .expect("a text panic payload")
}

fn grid(ts: Vec<f64>, xs: Vec<f64>, rows: usize, cols: usize) -> String {
  panic_text(move || {
    Grid2D::new(
      Array1::from_vec(ts),
      Array1::from_vec(xs),
      Array2::zeros((rows, cols)),
    )
  })
}

#[test]
fn every_rejected_argument_is_named_with_its_value() {
  let cases = [
    (
      panic_text(|| SimdGamma::<f64>::new(0.0, 1.0)),
      "alpha must satisfy `alpha > T::zero()`, got alpha = 0.0",
    ),
    (
      panic_text(|| SimdGamma::<f64>::new(2.0, 0.0)),
      "scale must satisfy `scale > T::zero()`, got scale = 0.0",
    ),
    (
      panic_text(|| SimdBeta::<f64>::new(1.0, -1.0)),
      "beta must satisfy `beta > T::zero()`, got beta = -1.0",
    ),
    (
      panic_text(|| SimdPareto::<f64>::new(1.0, 0.0)),
      "alpha must satisfy `alpha > T::zero()`, got alpha = 0.0",
    ),
    (
      panic_text(|| SimdWeibull::<f64>::new(1.0, 0.0)),
      "k must satisfy `k > T::zero()`, got k = 0.0",
    ),
    (
      panic_text(|| SimdInverseGauss::<f64>::new(1.0, 0.0)),
      "lambda must satisfy `lambda > T::zero()`, got lambda = 0.0",
    ),
    (
      panic_text(|| SimdUniform::<f64>::new(f64::NAN, 1.0)),
      "low must satisfy `low.is_finite()`, got low = NaN",
    ),
    (
      panic_text(|| SimdUniform::<f64>::new(0.0, f64::INFINITY)),
      "high must satisfy `high.is_finite()`, got high = inf",
    ),
    (
      panic_text(|| SimdUniform::<f64>::new(1.0, 0.0)),
      "low must satisfy `low < high`, got low = 1.0, high = 0.0",
    ),
    (
      panic_text(|| SimdGig::<f64>::new(0.5, 0.0, 1.0)),
      "chi must satisfy `chi > T::zero()`, got chi = 0.0",
    ),
    (
      panic_text(|| SimdGig::<f64>::new(0.5, 1.0, 0.0)),
      "psi must satisfy `psi > T::zero()`, got psi = 0.0",
    ),
    (
      panic_text(|| SimdGeneralizedHyperbolic::<f64>::new(0.5, 1.0, 1.5, 1.0, 0.0)),
      "alpha must satisfy `alpha > beta.abs()`, got alpha = 1.0, beta = 1.5",
    ),
    (
      panic_text(|| SimdGeneralizedHyperbolic::<f64>::new(0.5, 2.0, 0.5, 0.0, 0.0)),
      "delta must satisfy `delta > T::zero()`, got delta = 0.0",
    ),
    (
      panic_text(|| SimdJohnsonSu::<f64>::new(0.0, 0.0, 0.0, 1.0)),
      "delta must satisfy `delta > T::zero()`, got delta = 0.0",
    ),
    (
      panic_text(|| SimdJohnsonSu::<f64>::new(0.0, 1.0, 0.0, 0.0)),
      "lambda must satisfy `lambda > T::zero()`, got lambda = 0.0",
    ),
    (
      panic_text(|| SimdNormalInverseGauss::<f64>::new(1.0, -1.0, 1.0, 0.0)),
      "alpha must satisfy `alpha > beta.abs()`, got alpha = 1.0, beta = -1.0",
    ),
    (
      panic_text(|| SimdNormalInverseGauss::<f64>::new(2.0, 0.5, 0.0, 0.0)),
      "delta must satisfy `delta > T::zero()`, got delta = 0.0",
    ),
    (
      panic_text(|| SimdSkewT::<f64>::new(2.0, 0.0)),
      "eta must satisfy `eta > 2.0`, got eta = 2.0",
    ),
    (
      panic_text(|| SimdSkewT::<f64>::new(5.0, 1.0)),
      "lambda must satisfy `lambda.abs() < 1.0`, got lambda = 1.0",
    ),
    (
      panic_text(|| SimdTemperedStable::<f64>::new(1.0, 1.0, 1.0)),
      "alpha must satisfy `alpha > 0.0 && alpha < 1.0`, got alpha = 1.0",
    ),
    (
      panic_text(|| SimdTemperedStable::<f64>::new(0.5, -1.0, 1.0)),
      "lambda must satisfy `lambda >= 0.0`, got lambda = -1.0",
    ),
    (
      panic_text(|| SimdTemperedStable::<f64>::new(0.5, 1.0, 0.0)),
      "theta must satisfy `theta > 0.0`, got theta = 0.0",
    ),
    (
      panic_text(|| SimdVarianceGamma::<f64>::new(0.0, 1.0, 0.0, 0.0)),
      "sigma must satisfy `sigma > T::zero()`, got sigma = 0.0",
    ),
    (
      panic_text(|| SimdVarianceGamma::<f64>::new(0.2, 0.0, 0.0, 0.0)),
      "nu must satisfy `nu > T::zero()`, got nu = 0.0",
    ),
    (
      panic_text(|| SimdTruncatedBeta::<f64>::new(2.0, 2.0, -0.1, 0.5)),
      "lower must satisfy `0.0 <= lower <= 1.0`, got lower = -0.1",
    ),
    (
      panic_text(|| SimdTruncatedBeta::<f64>::new(2.0, 2.0, 0.1, 1.5)),
      "upper must satisfy `0.0 <= upper <= 1.0`, got upper = 1.5",
    ),
    (
      panic_text(|| <f64 as FloatExt>::normal_array(4, 0.0, 0.0)),
      "std_dev must satisfy `std_dev > 0.0`, got std_dev = 0.0",
    ),
    (
      panic_text(|| <f32 as FloatExt>::normal_array(4, 0.0, -1.0)),
      "std_dev must satisfy `std_dev > 0.0`, got std_dev = -1.0",
    ),
    (
      grid(vec![], vec![1.0], 0, 1),
      "ts must satisfy `!ts.is_empty()`, got ts.len() = 0",
    ),
    (
      grid(vec![0.5], vec![], 1, 0),
      "xs must satisfy `!xs.is_empty()`, got xs.len() = 0",
    ),
    (
      grid(vec![0.0, 0.5, 0.25], vec![1.0], 3, 1),
      "ts must satisfy `ts[j] < ts[j + 1]`, got ts[1] = 0.5, ts[2] = 0.25",
    ),
    (
      grid(vec![0.5], vec![1.0, 1.0], 1, 2),
      "xs must satisfy `xs[i] < xs[i + 1]`, got xs[0] = 1.0, xs[1] = 1.0",
    ),
    (
      grid(vec![0.25, 0.5], vec![1.0, 2.0], 2, 3),
      "values must satisfy `values.dim() == (ts.len(), xs.len())`, got values.dim() = (2, 3), ts.len() = 2, xs.len() = 2",
    ),
  ];
  let wrong = cases
    .iter()
    .filter(|(got, want)| got != want)
    .map(|(got, want)| format!("got  {got:?}\nwant {want:?}"))
    .collect::<Vec<_>>();
  assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

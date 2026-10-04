//! A rejected argument is named in its own assert: ``<arg> must satisfy `<pred>`, got <arg> = …``.

use std::panic::UnwindSafe;
use std::panic::catch_unwind;

use ndarray::Array1;
use ndarray::Array2;
use ndarray::array;
use stochastic_rs_distributions::FloatExt;
use stochastic_rs_distributions::alpha_stable::SimdAlphaStable;
use stochastic_rs_distributions::beta::SimdBeta;
use stochastic_rs_distributions::binomial::SimdBinomial;
use stochastic_rs_distributions::cauchy::SimdCauchy;
use stochastic_rs_distributions::chi_square::SimdChiSquared;
use stochastic_rs_distributions::dirichlet::SimdDirichlet;
use stochastic_rs_distributions::exp::SimdExp;
use stochastic_rs_distributions::gamma::SimdGamma;
use stochastic_rs_distributions::ged::SimdGed;
use stochastic_rs_distributions::generalized_hyperbolic::SimdGeneralizedHyperbolic;
use stochastic_rs_distributions::generalized_inverse_gauss::SimdGig;
use stochastic_rs_distributions::geometric::SimdGeometric;
use stochastic_rs_distributions::gev::SimdGev;
use stochastic_rs_distributions::gpd::SimdGpd;
use stochastic_rs_distributions::inverse_gauss::SimdInverseGauss;
use stochastic_rs_distributions::johnson_su::SimdJohnsonSu;
use stochastic_rs_distributions::lognormal::SimdLogNormal;
use stochastic_rs_distributions::non_central_chi_squared::SimdNonCentralChiSquared;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::normal_inverse_gauss::SimdNormalInverseGauss;
use stochastic_rs_distributions::pareto::SimdPareto;
use stochastic_rs_distributions::simd_rng::Deterministic;
use stochastic_rs_distributions::skew_t::SimdSkewT;
use stochastic_rs_distributions::studentt::SimdStudentT;
use stochastic_rs_distributions::tempered_stable::SimdTemperedStable;
use stochastic_rs_distributions::traits::Grid2D;
use stochastic_rs_distributions::truncated::SimdTruncatedBeta;
use stochastic_rs_distributions::truncated::SimdTruncatedExp;
use stochastic_rs_distributions::truncated::SimdTruncatedGamma;
use stochastic_rs_distributions::truncated::SimdTruncatedNormal;
use stochastic_rs_distributions::uniform::SimdUniform;
use stochastic_rs_distributions::variance_gamma::SimdVarianceGamma;
use stochastic_rs_distributions::weibull::SimdWeibull;
use stochastic_rs_distributions::wishart::SimdWishart;

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
    (
      panic_text(|| SimdWishart::<f64>::new(3.0, array![[1.0, f64::NAN], [f64::NAN, 1.0]])),
      "scale must satisfy `scale[[i, j]].is_finite()`, got scale[[0, 1]] = NaN",
    ),
    (
      panic_text(|| SimdWishart::<f64>::new(3.0, array![[f64::INFINITY, 0.0], [0.0, 1.0]])),
      "scale must satisfy `scale[[i, j]].is_finite()`, got scale[[0, 0]] = inf",
    ),
    (
      panic_text(|| SimdWishart::<f64>::new(3.0, array![[1.0, 2.0], [2.0, 1.0]])),
      "scale must satisfy `pivot[i] > 0.0` (positive definite), got pivot[1] = -3.0",
    ),
    (
      panic_text(|| SimdDirichlet::<f64>::new(vec![1.0, f64::INFINITY])),
      "alpha must satisfy `alpha[k].is_finite()`, got alpha[1] = inf",
    ),
    (
      panic_text(|| SimdGed::<f64>::new(0.0, 1.0, 1e-320)),
      "beta must satisfy `(1 / beta).is_finite()`, got beta = 1e-320",
    ),
    (
      panic_text(|| SimdVarianceGamma::<f64>::new(0.2, 1e-320, 0.0, 0.0)),
      "nu must satisfy `(1 / nu).is_finite()`, got nu = 1e-320",
    ),
    (
      panic_text(|| SimdGeneralizedHyperbolic::<f64>::new(0.5, 2.0, 0.5, 1e200, 0.0)),
      "delta must satisfy `0 < delta * delta < ∞`, got delta = 1e200",
    ),
    (
      panic_text(|| SimdGeneralizedHyperbolic::<f64>::new(0.5, 1e200, 0.5, 1.0, 0.0)),
      "alpha must satisfy `0 < alpha * alpha - beta * beta < ∞`, got alpha = 1e200, beta = 0.5",
    ),
    (
      panic_text(|| SimdNormalInverseGauss::<f64>::new(1e-170, 0.0, 1.0, 0.0)),
      "delta must satisfy `0 < delta / (alpha * alpha - beta * beta).sqrt() < ∞`, got delta = 1.0, alpha = 1e-170, beta = 0.0",
    ),
    (
      panic_text(|| SimdNormalInverseGauss::<f64>::new(2.0, 0.5, 1e200, 0.0)),
      "delta must satisfy `0 < delta * delta < ∞`, got delta = 1e200",
    ),
    (
      panic_text(|| SimdChiSquared::<f64>::new(5e-324)),
      "k must satisfy `k / 2 > 0`, got k = 5e-324",
    ),
    (
      panic_text(|| SimdStudentT::<f64>::new(5e-324)),
      "nu must satisfy `nu / 2 > 0`, got nu = 5e-324",
    ),
    (
      panic_text(|| SimdNonCentralChiSquared::<f64>::new(5e-324)),
      "df must satisfy `df / 2 > 0`, got df = 5e-324",
    ),
    (
      panic_text(|| {
        stochastic_rs_distributions::non_central_chi_squared::sample(
          -1.0_f64,
          1.0,
          &Deterministic::new(1),
        )
      }),
      "df must satisfy `df > T::zero()`, got df = -1.0",
    ),
    (
      panic_text(|| SimdWishart::<f64>::new(5e-324, array![[1.0]])),
      "nu must satisfy `0 < (nu - j) / 2 < ∞`, got nu = 5e-324, j = 0",
    ),
    (
      panic_text(|| SimdWishart::<f32>::new(1e39, array![[1.0]])),
      "nu must satisfy `0 < (nu - j) / 2 < ∞`, got nu = 1e39, j = 0",
    ),
  ];
  let wrong = cases
    .iter()
    .filter(|(got, want)| got != want)
    .map(|(got, want)| format!("got  {got:?}\nwant {want:?}"))
    .collect::<Vec<_>>();
  assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// One constructor driven through its float parameters: their names in order, a valid point, the call.
struct Ctor {
  names: &'static [&'static str],
  base: &'static [f64],
  build: fn(&[f64]),
}

macro_rules! ctor {
  ($names:expr, $base:expr, |$p:ident| $call:expr) => {
    Ctor {
      names: &$names,
      base: &$base,
      build: |$p: &[f64]| {
        let _ = $call;
      },
    }
  };
}

fn ctors() -> Vec<Ctor> {
  vec![
    ctor!(
      ["alpha", "beta", "scale", "location"],
      [1.5, 0.3, 1.0, 0.0],
      |p| { SimdAlphaStable::<f64>::new(p[0], p[1], p[2], p[3]) }
    ),
    ctor!(["alpha", "beta"], [2.0, 3.0], |p| SimdBeta::<f64>::new(
      p[0], p[1]
    )),
    ctor!(["p"], [0.3], |p| SimdBinomial::<u32>::new(10, p[0])),
    ctor!(["x0", "gamma"], [0.0, 1.0], |p| SimdCauchy::<f64>::new(
      p[0], p[1]
    )),
    ctor!(["k"], [3.0], |p| SimdChiSquared::<f64>::new(p[0])),
    ctor!(["lambda"], [1.0], |p| SimdExp::<f64>::new(p[0])),
    ctor!(["alpha", "scale"], [2.0, 1.5], |p| SimdGamma::<f64>::new(
      p[0], p[1]
    )),
    ctor!(["mu", "alpha", "beta"], [0.0, 1.0, 1.5], |p| {
      SimdGed::<f64>::new(p[0], p[1], p[2])
    }),
    ctor!(
      ["lambda", "alpha", "beta", "delta", "mu"],
      [1.0, 2.0, 0.5, 1.0, 0.0],
      |p| { SimdGeneralizedHyperbolic::<f64>::new(p[0], p[1], p[2], p[3], p[4]) }
    ),
    ctor!(["lambda", "chi", "psi"], [0.5, 1.0, 2.0], |p| {
      SimdGig::<f64>::new(p[0], p[1], p[2])
    }),
    ctor!(["p"], [0.3], |p| SimdGeometric::<u32>::new(p[0])),
    ctor!(["mu", "sigma", "xi"], [0.0, 1.0, 0.1], |p| {
      SimdGev::<f64>::new(p[0], p[1], p[2])
    }),
    ctor!(["mu", "sigma", "xi"], [0.0, 1.0, 0.1], |p| {
      SimdGpd::<f64>::new(p[0], p[1], p[2])
    }),
    ctor!(["mu", "lambda"], [1.0, 2.0], |p| {
      SimdInverseGauss::<f64>::new(p[0], p[1])
    }),
    ctor!(
      ["gamma", "delta", "xi", "lambda"],
      [0.5, 1.5, 0.0, 1.0],
      |p| { SimdJohnsonSu::<f64>::new(p[0], p[1], p[2], p[3]) }
    ),
    ctor!(["mu", "sigma"], [0.0, 0.5], |p| SimdLogNormal::<f64>::new(
      p[0], p[1]
    )),
    ctor!(["df"], [2.0], |p| SimdNonCentralChiSquared::<f64>::new(
      p[0]
    )),
    ctor!(["mean", "std_dev"], [0.0, 1.0], |p| SimdNormal::<f64>::new(
      p[0], p[1]
    )),
    ctor!(
      ["alpha", "beta", "delta", "mu"],
      [2.0, 0.5, 1.0, 0.0],
      |p| { SimdNormalInverseGauss::<f64>::new(p[0], p[1], p[2], p[3]) }
    ),
    ctor!(["x_m", "alpha"], [1.0, 1.5], |p| SimdPareto::<f64>::new(
      p[0], p[1]
    )),
    ctor!(["eta", "lambda"], [5.0, 0.2], |p| SimdSkewT::<f64>::new(
      p[0], p[1]
    )),
    ctor!(["nu"], [5.0], |p| SimdStudentT::<f64>::new(p[0])),
    ctor!(["alpha", "lambda", "theta"], [0.5, 1.0, 1.0], |p| {
      SimdTemperedStable::<f64>::new(p[0], p[1], p[2])
    }),
    ctor!(["sigma", "nu", "theta", "mu"], [0.2, 0.5, -0.1, 0.0], |p| {
      SimdVarianceGamma::<f64>::new(p[0], p[1], p[2], p[3])
    }),
    ctor!(["lambda", "k"], [1.0, 1.5], |p| SimdWeibull::<f64>::new(
      p[0], p[1]
    )),
    ctor!(["nu"], [3.0], |p| {
      SimdWishart::<f64>::new(p[0], array![[1.0, 0.0], [0.0, 1.0]])
    }),
    ctor!(
      ["alpha", "beta", "lower", "upper"],
      [2.0, 2.0, 0.1, 0.9],
      |p| { SimdTruncatedBeta::<f64>::new(p[0], p[1], p[2], p[3]) }
    ),
    ctor!(["lambda", "lower", "upper"], [1.0, 0.0, 1.0], |p| {
      SimdTruncatedExp::<f64>::new(p[0], p[1], p[2])
    }),
    ctor!(
      ["shape", "scale", "lower", "upper"],
      [2.0, 1.0, 0.0, 1.0],
      |p| { SimdTruncatedGamma::<f64>::new(p[0], p[1], p[2], p[3]) }
    ),
    ctor!(
      ["mu", "sigma", "lower", "upper"],
      [0.0, 1.0, -1.0, 1.0],
      |p| { SimdTruncatedNormal::<f64>::new(p[0], p[1], p[2], p[3]) }
    ),
    ctor!(["low", "high"], [0.0, 1.0], |p| SimdUniform::<f64>::new(
      p[0], p[1]
    )),
  ]
}

const NON_FINITE: [f64; 3] = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];

/// Every finiteness assert names its parameter and value: NaN and ±∞ at each float parameter (NaN only at a
/// truncation bound, which may be infinite) and at each entry of Wishart's scale and Dirichlet's concentrations.
#[test]
fn every_finiteness_assert_names_its_parameter() {
  let mut params = 0;
  for c in ctors() {
    for (i, &name) in c.names.iter().enumerate() {
      let bound = name == "lower" || name == "upper";
      let (pred, bad) = if bound {
        (format!("!{name}.is_nan()"), &NON_FINITE[..1])
      } else {
        (format!("{name}.is_finite()"), &NON_FINITE[..])
      };
      for &v in bad {
        let mut p = c.base.to_vec();
        p[i] = v;
        let build = c.build;
        let want = format!("{name} must satisfy `{pred}`, got {name} = {v:?}");
        assert_eq!(panic_text(move || build(&p)), want);
      }
      params += 1;
    }
  }
  assert_eq!(
    params, 78,
    "the 76 asserts the policy added plus SimdUniform's two"
  );
  for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
    for v in NON_FINITE {
      let mut scale = array![[1.0, 0.0], [0.0, 1.0]];
      scale[[i, j]] = v;
      let want =
        format!("scale must satisfy `scale[[i, j]].is_finite()`, got scale[[{i}, {j}]] = {v:?}");
      assert_eq!(
        panic_text(move || SimdWishart::<f64>::new(3.0, scale)),
        want
      );
    }
  }
  for k in 0..3 {
    for v in NON_FINITE {
      let mut alpha = vec![1.0, 2.0, 3.0];
      alpha[k] = v;
      let want = format!("alpha must satisfy `alpha[k].is_finite()`, got alpha[{k}] = {v:?}");
      assert_eq!(panic_text(move || SimdDirichlet::<f64>::new(alpha)), want);
    }
  }
  assert!(
    SimdTruncatedExp::<f64>::new(1.0, 0.0, f64::INFINITY)
      .upper()
      .is_infinite()
  );
}

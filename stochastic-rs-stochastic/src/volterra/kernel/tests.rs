use ndarray::Array1;

use super::ExponentialKernel;
use super::GammaKernel;
use super::RlKernel;
use super::SumOfExponentials;
use super::VolterraKernel;

/// The three implementors in one enum, so one loop drives them through [`VolterraKernel`].
#[derive(Clone)]
enum AnyKernel {
  Exponential(ExponentialKernel<f64>),
  Gamma(GammaKernel<f64>),
  SumOfExp(SumOfExponentials<f64>),
}

impl VolterraKernel<f64> for AnyKernel {
  fn nodes(&self) -> &Array1<f64> {
    match self {
      AnyKernel::Exponential(k) => k.nodes(),
      AnyKernel::Gamma(k) => k.nodes(),
      AnyKernel::SumOfExp(k) => k.nodes(),
    }
  }

  fn weights(&self) -> &Array1<f64> {
    match self {
      AnyKernel::Exponential(k) => k.weights(),
      AnyKernel::Gamma(k) => k.weights(),
      AnyKernel::SumOfExp(k) => k.weights(),
    }
  }

  fn evaluate(&self, t: f64) -> f64 {
    match self {
      AnyKernel::Exponential(k) => k.evaluate(t),
      AnyKernel::Gamma(k) => k.evaluate(t),
      AnyKernel::SumOfExp(k) => k.evaluate(t),
    }
  }

  fn integral_from_zero(&self, dt: f64) -> f64 {
    match self {
      AnyKernel::Exponential(k) => k.integral_from_zero(dt),
      AnyKernel::Gamma(k) => k.integral_from_zero(dt),
      AnyKernel::SumOfExp(k) => k.integral_from_zero(dt),
    }
  }
}

/// One instance of each implementor. `GammaKernel`'s degree 100 clears the 5e-3 tolerance down
/// to `t=1e-2`: its sum is the RL sum, and `degree=40` misses by about 5x there.
fn kernels_under_test() -> Vec<(&'static str, AnyKernel)> {
  vec![
    (
      "ExponentialKernel",
      AnyKernel::Exponential(ExponentialKernel::new(1.7, 2.3)),
    ),
    (
      "GammaKernel",
      AnyKernel::Gamma(GammaKernel::new(0.3, 1.5, 100)),
    ),
    (
      "SumOfExponentials",
      AnyKernel::SumOfExp(SumOfExponentials::new(
        Array1::from_vec(vec![0.5, 2.0, 6.0]),
        Array1::from_vec(vec![0.3, 0.5, 0.9]),
      )),
    ),
  ]
}

/// `ExponentialKernel` is one exponential, so its sum reproduces `evaluate` to floating-point
/// noise, not to a quadrature tolerance.
#[test]
fn exponential_kernel_is_exact_at_degree_one() {
  let k = ExponentialKernel::<f64>::new(1.7, 2.3);
  assert_eq!(k.degree(), 1);
  for &t in &[1e-3, 0.1, 0.5, 1.0, 5.0] {
    let approx = k.weights()[0] * (-k.nodes()[0] * t).exp();
    assert!((approx - k.evaluate(t)).abs() < 1e-15, "t={t}");
  }
}

/// `integral_from_zero` agrees with a fine quadrature of `evaluate` for every kernel: the term
/// that generalises `MarkovLift`'s `dt^{H+1/2} / Γ(H+3/2)`.
#[test]
fn integral_from_zero_matches_numerical_quadrature() {
  let dt = 0.01_f64;
  let n = 200_000;
  for (name, k) in kernels_under_test() {
    let h = dt / n as f64;
    // midpoint rule; the integrand is singular at 0 for fractional kernels,
    // so the midpoint (not left endpoint) is what keeps this finite.
    let mut acc = 0.0;
    for i in 0..n {
      acc += k.evaluate((i as f64 + 0.5) * h);
    }
    acc *= h;
    let closed = k.integral_from_zero(dt);
    let rel = (acc - closed).abs() / closed.abs().max(1e-300);
    assert!(
      rel < 1e-4,
      "{name}: quadrature={acc} closed={closed} rel={rel}"
    );
  }
}

/// The exponential-sum fit approximates its kernel within `rel < 5e-3`, the tolerance
/// `rough/kernel.rs` asserts for the RL kernel on the same kind of grid.
#[test]
fn exponential_sum_approximates_the_kernel() {
  for (name, k) in kernels_under_test() {
    for &t in &[1e-2, 0.1, 0.5, 1.0] {
      let approx: f64 = (0..k.degree())
        .map(|l| k.weights()[l] * (-k.nodes()[l] * t).exp())
        .sum();
      let truth = k.evaluate(t);
      let rel = (approx - truth).abs() / truth.abs();
      assert!(
        rel < 5e-3,
        "{name} t={t}: approx={approx} truth={truth} rel={rel}"
      );
    }
  }
}

#[test]
#[should_panic(expected = "beta must satisfy `beta > T::zero()`, got beta = 0.0")]
fn exponential_kernel_rejects_non_positive_beta() {
  let _ = ExponentialKernel::<f64>::new(0.0, 1.0);
}

#[test]
#[should_panic(expected = "lambda must satisfy `lambda > T::zero()`, got lambda = 0.0")]
fn gamma_kernel_rejects_non_positive_lambda() {
  let _ = GammaKernel::<f64>::new(0.3, 0.0, 10);
}

/// `GammaKernel::new` checks `lambda` itself; the Hurst range is enforced by the `RlKernel::new`
/// inside it and surfaces here as a panic, not as a `NaN` from an out-of-range Laguerre `alpha`.
#[test]
#[should_panic(expected = "hurst must satisfy `hurst > 0.0 && hurst < 0.5`, got hurst = 0.7")]
fn gamma_kernel_rejects_hurst_out_of_range() {
  let _ = GammaKernel::<f64>::new(0.7, 1.0, 10);
}

/// `degree` is not re-validated inside `GammaKernel::new`; this locks in
/// that `RlKernel::new`'s stability ceiling still fires through it.
#[test]
#[should_panic(expected = "degree must satisfy `degree <= Self::MAX_STABLE_DEGREE`, got degree = ")]
fn gamma_kernel_rejects_degree_above_stability_ceiling() {
  let _ = GammaKernel::<f64>::new(0.3, 1.0, RlKernel::<f64>::MAX_STABLE_DEGREE + 1);
}

#[test]
#[should_panic(
  expected = "nodes.len() must satisfy `nodes.len() == weights.len()`, got nodes.len() = 2, weights.len() = 1"
)]
fn sum_of_exponentials_rejects_mismatched_lengths() {
  let _ = SumOfExponentials::new(
    Array1::from_vec(vec![1.0, 2.0]),
    Array1::from_vec(vec![1.0]),
  );
}

#[test]
#[should_panic(expected = "nodes must satisfy `!nodes.is_empty()`, got nodes = []")]
fn sum_of_exponentials_rejects_empty() {
  let _ = SumOfExponentials::new(Array1::<f64>::zeros(0), Array1::<f64>::zeros(0));
}

#[test]
#[should_panic(expected = "nodes[1] must satisfy `nodes[i] > T::zero()`, got nodes[1] = -0.5")]
fn sum_of_exponentials_rejects_non_positive_node() {
  let _ = SumOfExponentials::new(
    Array1::from_vec(vec![1.0, -0.5]),
    Array1::from_vec(vec![1.0, 1.0]),
  );
}

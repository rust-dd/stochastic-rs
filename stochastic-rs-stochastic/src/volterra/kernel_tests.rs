use ndarray::Array1;

use super::ExponentialKernel;
use super::GammaKernel;
use super::SumOfExponentials;
use super::VolterraKernel;

/// Test-only union of the three implementors so a single loop can drive all
/// of them through the shared [`VolterraKernel`] interface, exactly as a
/// generic caller would.
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

/// One representative instance of each implementor defined in this module.
/// `GammaKernel`'s degree (100) is chosen empirically to clear the 5e-3
/// fit tolerance down to `t=1e-2`, the smallest `t` exercised below: its
/// exponential sum is the RL sum for the same Hurst, so it inherits the RL
/// fit's own convergence — `degree=40` misses by ~5x at `t=1e-2`.
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

/// `ExponentialKernel` is representable exactly by one exponential, so its
/// approximation must reproduce `evaluate` to floating-point noise, not to a
/// quadrature tolerance.
#[test]
fn exponential_kernel_is_exact_at_degree_one() {
  let k = ExponentialKernel::<f64>::new(1.7, 2.3);
  assert_eq!(k.degree(), 1);
  for &t in &[1e-3, 0.1, 0.5, 1.0, 5.0] {
    let approx = k.weights()[0] * (-k.nodes()[0] * t).exp();
    assert!((approx - k.evaluate(t)).abs() < 1e-15, "t={t}");
  }
}

/// `integral_from_zero` must agree with a fine numerical quadrature of
/// `evaluate`, for every kernel — this is the term that replaces
/// `MarkovLift`'s hard-wired `dt^{H+1/2} / Γ(H+3/2)`.
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

/// The exponential-sum fit must actually approximate the kernel it claims to.
/// Reference for the tolerance: the existing `rough/kernel.rs` test asserts
/// `rel < 5e-3` for the RL kernel on the same kind of grid.
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

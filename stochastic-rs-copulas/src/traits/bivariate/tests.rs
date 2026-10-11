use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;

use super::*;
use crate::bivariate::amh::Amh;
use crate::bivariate::clayton::Clayton;
use crate::bivariate::galambos::Galambos;

/// Overrides no provided method, so every default body answers for it; `pdf` and `cdf` fail.
struct DummyNonArchimedean {
  theta: Option<f64>,
}

impl BivariateExt for DummyNonArchimedean {
  fn r#type(&self) -> BivariateCopulaType {
    BivariateCopulaType::Fgm
  }

  fn tau(&self) -> Option<f64> {
    None
  }

  fn set_tau(&mut self, _tau: f64) {}

  fn theta(&self) -> Option<f64> {
    self.theta
  }

  fn theta_bounds(&self) -> (f64, f64) {
    (-1.0, 1.0)
  }

  fn invalid_thetas(&self) -> Vec<f64> {
    vec![]
  }

  fn set_theta(&mut self, _theta: f64) {}

  fn compute_theta(&self) -> f64 {
    0.0
  }

  fn tail_dependence(&self) -> TailDependence<f64> {
    TailDependence {
      lower: 0.0,
      upper: 0.0,
    }
  }

  fn pdf(&self, _x: ndarray::ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    Err(CopulaError::Unsupported(
      "not implemented for DummyNonArchimedean".into(),
    ))
  }

  fn cdf(&self, _x: ndarray::ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    Err(CopulaError::Unsupported(
      "not implemented for DummyNonArchimedean".into(),
    ))
  }
}

/// Parameterised outside `theta`, like `MarshallOlkin::with_alpha_beta`; the h-function is `u`.
struct ParameterisedWithoutTheta;

impl BivariateExt for ParameterisedWithoutTheta {
  fn r#type(&self) -> BivariateCopulaType {
    BivariateCopulaType::MarshallOlkin
  }

  fn tau(&self) -> Option<f64> {
    None
  }

  fn set_tau(&mut self, _tau: f64) {}

  fn theta(&self) -> Option<f64> {
    None
  }

  fn theta_bounds(&self) -> (f64, f64) {
    (0.0, 1.0)
  }

  fn invalid_thetas(&self) -> Vec<f64> {
    vec![]
  }

  fn set_theta(&mut self, _theta: f64) {}

  fn compute_theta(&self) -> f64 {
    0.0
  }

  fn tail_dependence(&self) -> TailDependence<f64> {
    TailDependence {
      lower: 0.0,
      upper: 0.0,
    }
  }

  fn check_fit(&self) -> Result<(), CopulaError> {
    Ok(())
  }

  fn pdf(&self, x: ndarray::ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    Ok(Array1::ones(x.nrows()))
  }

  fn cdf(&self, x: ndarray::ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    Ok(&x.column(0) * &x.column(1))
  }

  fn partial_derivative(&self, x: ndarray::ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    Ok(x.column(0).to_owned())
  }
}

/// A type with no `generator` override reaches the trait default, which names its `r#type()`.
#[test]
fn generator_default_returns_anchored_not_archimedean_err() {
  let dummy = DummyNonArchimedean { theta: None };
  assert_eq!(
    dummy.generator(array![0.5_f64, 0.8].view()).unwrap_err(),
    CopulaError::Unsupported("Fgm is not Archimedean: no generator".into())
  );
}

/// An unfitted copula is `NotFitted` before any solve, and a failing `partial_derivative` keeps
/// its own variant: neither becomes a panic or a fabricated `EPSILON` quantile.
#[test]
fn percent_point_numerical_propagates_inner_errors_instead_of_panicking() {
  let (y, v) = (array![0.5_f64], array![0.5_f64]);
  let unfitted = DummyNonArchimedean { theta: None };
  assert_eq!(
    unfitted
      .percent_point_numerical(y.view(), v.view())
      .unwrap_err(),
    CopulaError::NotFitted
  );
  let fitted = DummyNonArchimedean { theta: Some(0.5) };
  assert_eq!(
    fitted
      .percent_point_numerical(y.view(), v.view())
      .unwrap_err(),
    CopulaError::Unsupported("not implemented for DummyNonArchimedean".into())
  );
}

/// A target below the bracket floor is a saturation, not a failure: the
/// h-function of any copula at `u = EPSILON` is at least `y = 1e-300`, so
/// the quantile is below resolution and the floor itself is the honest
/// answer.
#[test]
fn percent_point_numerical_saturates_at_the_bracket_floor() {
  let mut amh = Amh::new();
  amh.set_tau(0.25);
  amh._compute_theta();
  let out = amh
    .percent_point_numerical(array![1e-300_f64].view(), array![0.5_f64].view())
    .expect("saturation is a value");
  assert_eq!(
    out[0],
    f64::EPSILON,
    "expected the bracket floor, got {}",
    out[0]
  );
}

/// A conditional law with a point mass at `u = 1` keeps `h` below `y` on the whole bracket: the ceiling answers.
#[test]
fn percent_point_numerical_saturates_at_the_bracket_ceiling() {
  let mut galambos = Galambos::new();
  galambos.set_theta(1.5);
  let out = galambos
    .percent_point_numerical(array![0.3_f64].view(), array![1.0_f64].view())
    .expect("saturation is a value");
  assert_eq!(out[0], 1.0);
}

/// Level 0 answers the floor even where rounding takes `h(EPSILON | v)` below zero, as Galambos's does near `v = 1`.
#[test]
fn percent_point_numerical_answers_the_floor_at_level_zero() {
  let mut galambos = Galambos::new();
  galambos.set_theta(1.5);
  for v in [1.0 - 1e-10, 1.0 - 1e-12, 1.0 - f64::EPSILON / 2.0] {
    let h = galambos.partial_derivative_scalar(f64::EPSILON, v).unwrap();
    assert!(h < 0.0, "the premise: h(EPSILON | {v}) = {h}");
    let u = galambos.percent_point_numerical(array![0.0].view(), array![v].view());
    assert_eq!(u, Ok(array![f64::EPSILON]), "v = {v}");
  }
}

/// Levels far below `√f64::MIN_POSITIVE` reach Brent's `zero`, whose sign test never multiplies two gaps.
#[test]
fn percent_point_numerical_solves_levels_below_the_product_underflow() {
  let mut galambos = Galambos::new();
  galambos.set_theta(1.5);
  for y in [1e-300, 1e-310, 5e-324] {
    for v in [1.0 - 1e-10, 1.0 - f64::EPSILON / 2.0] {
      let u = galambos
        .percent_point_numerical(array![y].view(), array![v].view())
        .unwrap_or_else(|e| panic!("y = {y}, v = {v}: {e}"))[0];
      assert!((f64::EPSILON..=1.0).contains(&u), "y = {y}, v = {v}: {u}");
    }
  }
}

/// `percent_point_numerical` reaches the Brent body whatever `percent_point` the family overrides.
#[test]
fn percent_point_numerical_round_trips_through_the_h_function() {
  let mut amh = Amh::new();
  amh.set_tau(0.25);
  amh._compute_theta();
  for &(y, v) in &[(0.1_f64, 0.3_f64), (0.5, 0.5), (0.9, 0.7), (0.05, 0.95)] {
    let u = amh
      .percent_point_numerical(array![y].view(), array![v].view())
      .expect("solve")[0];
    let back = amh.partial_derivative_scalar(u, v).expect("h eval");
    assert!(
      (back - y).abs() < 1e-9,
      "round trip failed: h(h_inv({y}|{v})|{v}) = {back}"
    );
  }
}

/// `#[doc(hidden)]` hides `_compute_theta` / `check_marginal` /
/// `partial_derivative_scalar` from rendered docs but must not restrict
/// who can call them — they stay reachable in-crate exactly like
/// `sample_with_uniform` already was. Compile-guard test: this would
/// stop compiling if any of the three ever became `pub(crate)` (or
/// otherwise less visible) by mistake.
#[test]
fn doc_hidden_methods_remain_callable_in_crate() {
  let mut c = Clayton::new();
  c.set_tau(0.5);
  c._compute_theta();
  assert!(c.theta().is_some());

  let u = array![0.1_f64, 0.5, 0.9];
  assert!(c.check_marginal(u.view()).is_ok());

  let scalar = c.partial_derivative_scalar(0.3, 0.6).unwrap();
  assert!((0.0..=1.0).contains(&scalar));
}

#[test]
fn sampling_needs_theta_not_tau() {
  let c = Clayton {
    theta: Some(2.0),
    ..Clayton::new()
  };
  let uv = c
    .sample(1_000, &Deterministic::new(7))
    .expect("theta alone is enough to sample");
  assert_eq!(uv.dim(), (1_000, 2));

  assert_eq!(
    Clayton::new()
      .sample(10, &Deterministic::new(7))
      .unwrap_err(),
    CopulaError::NotFitted
  );
}

#[test]
fn sampling_gates_on_the_family_check_fit() {
  let uv = ParameterisedWithoutTheta
    .sample(64, &Deterministic::new(7))
    .expect("check_fit passes without a theta");
  assert_eq!(uv.dim(), (64, 2));
  assert!(uv.iter().all(|x| (0.0..=1.0).contains(x)));
}

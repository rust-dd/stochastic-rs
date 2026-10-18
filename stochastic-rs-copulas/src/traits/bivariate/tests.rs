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
  fn family(&self) -> BivariateFamily {
    BivariateFamily::Fgm
  }

  fn theta(&self) -> Option<f64> {
    self.theta
  }

  fn theta_bounds(&self) -> (f64, f64) {
    (-1.0, 1.0)
  }

  fn invalid_thetas(&self) -> Vec<f64> {
    vec![]
  }

  fn set_theta(&mut self, theta: f64) {
    self.validate_theta(theta).unwrap_or_else(|e| panic!("{e}"));
    self.theta = Some(theta);
  }

  fn tau_from_theta(&self, _theta: f64) -> f64 {
    0.0
  }

  fn theta_from_tau(&self, _tau: f64) -> Result<f64, CopulaError> {
    Ok(0.0)
  }

  fn tail_dependence(&self) -> Result<TailDependence<f64>, CopulaError> {
    Ok(TailDependence {
      lower: 0.0,
      upper: 0.0,
    })
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

/// Parameterised outside `theta`, like an asymmetric `MarshallOlkin`; the h-function is `u`.
struct ParameterisedWithoutTheta;

impl BivariateExt for ParameterisedWithoutTheta {
  fn family(&self) -> BivariateFamily {
    BivariateFamily::MarshallOlkin
  }

  fn theta(&self) -> Option<f64> {
    None
  }

  fn theta_bounds(&self) -> (f64, f64) {
    (0.0, 1.0)
  }

  fn invalid_thetas(&self) -> Vec<f64> {
    vec![]
  }

  fn set_theta(&mut self, theta: f64) {
    self.validate_theta(theta).unwrap_or_else(|e| panic!("{e}"));
  }

  fn tau_from_theta(&self, _theta: f64) -> f64 {
    0.0
  }

  fn theta_from_tau(&self, _tau: f64) -> Result<f64, CopulaError> {
    Ok(0.0)
  }

  fn tail_dependence(&self) -> Result<TailDependence<f64>, CopulaError> {
    Ok(TailDependence {
      lower: 0.0,
      upper: 0.0,
    })
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

/// Overrides `percent_point` alone, as an implementor outside the crate may: half the level, so its draws show it.
struct HalvesTheLevel;

impl BivariateExt for HalvesTheLevel {
  fn family(&self) -> BivariateFamily {
    BivariateFamily::MarshallOlkin
  }

  fn theta(&self) -> Option<f64> {
    None
  }

  fn theta_bounds(&self) -> (f64, f64) {
    (0.0, 1.0)
  }

  fn invalid_thetas(&self) -> Vec<f64> {
    vec![]
  }

  fn set_theta(&mut self, theta: f64) {
    self.validate_theta(theta).unwrap_or_else(|e| panic!("{e}"));
  }

  fn tau_from_theta(&self, _theta: f64) -> f64 {
    0.0
  }

  fn theta_from_tau(&self, _tau: f64) -> Result<f64, CopulaError> {
    Ok(0.0)
  }

  fn tail_dependence(&self) -> Result<TailDependence<f64>, CopulaError> {
    Ok(TailDependence {
      lower: 0.0,
      upper: 0.0,
    })
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

  fn percent_point(
    &self,
    y: ArrayView1<f64>,
    _v: ArrayView1<f64>,
  ) -> Result<Array1<f64>, CopulaError> {
    Ok(y.mapv(|y| 0.5 * y))
  }
}

/// A type with no `generator` override reaches the trait default, which names its `family()`.
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

/// The bracket starts at 0, so a level far below `h(EPSILON | v)` finds its own root, below `EPSILON`.
#[test]
fn percent_point_numerical_solves_below_epsilon() {
  let amh = Amh::from_tau(0.25).unwrap();
  let out = amh
    .percent_point_numerical(array![1e-300_f64].view(), array![0.5_f64].view())
    .expect("a root");
  let closed = amh
    .percent_point(array![1e-300_f64].view(), array![0.5_f64].view())
    .unwrap();
  assert!(
    (out[0] - closed[0]).abs() <= 1e-12 * closed[0],
    "{} vs the closed form {}",
    out[0],
    closed[0]
  );
}

/// A conditional law with a point mass at `u = 1`, Galambos's at `v = 1`, answers the top of the bracket within Brent's
/// `6 macheps |x| + 2t`.
#[test]
fn percent_point_numerical_answers_a_point_mass_at_one() {
  let mut galambos = Galambos::default();
  galambos.set_theta(1.5);
  let out = galambos
    .percent_point_numerical(array![0.3_f64].view(), array![1.0_f64].view())
    .expect("a root");
  assert!(1.0 - out[0] <= 3.0 * f64::EPSILON, "{}", out[0]);
}

/// Level 0 answers 0, and near `v = 1` Galambos's `h(EPSILON | v)` stays at or above 0.
#[test]
fn percent_point_numerical_answers_zero_at_level_zero() {
  let mut galambos = Galambos::default();
  galambos.set_theta(1.5);
  for v in [1.0 - 1e-10, 1.0 - 1e-12, 1.0 - f64::EPSILON / 2.0] {
    let h = galambos.partial_derivative_scalar(f64::EPSILON, v).unwrap();
    assert!(h >= 0.0, "h(EPSILON | {v}) = {h}");
    let u = galambos.percent_point_numerical(array![0.0].view(), array![v].view());
    assert_eq!(u, Ok(array![0.0]), "v = {v}");
  }
}

/// Levels far below `√f64::MIN_POSITIVE` reach Brent's `zero`, whose sign test never multiplies two gaps.
#[test]
fn percent_point_numerical_solves_levels_below_the_product_underflow() {
  let mut galambos = Galambos::default();
  galambos.set_theta(1.5);
  for y in [1e-300, 1e-310, 5e-324] {
    for v in [1.0 - 1e-10, 1.0 - f64::EPSILON / 2.0] {
      let u = galambos
        .percent_point_numerical(array![y].view(), array![v].view())
        .unwrap_or_else(|e| panic!("y = {y}, v = {v}: {e}"))[0];
      assert!((0.0..=1.0).contains(&u), "y = {y}, v = {v}: {u}");
    }
  }
}

/// `percent_point_numerical` reaches the Brent body whatever `percent_point` the family overrides.
#[test]
fn percent_point_numerical_round_trips_through_the_h_function() {
  let amh = Amh::from_tau(0.25).unwrap();
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

/// `#[doc(hidden)]` hides `partial_derivative_scalar` from the docs, not from callers: this stops compiling if it, or the
/// public `check_marginal`, becomes less visible.
#[test]
fn doc_hidden_methods_remain_callable_in_crate() {
  let c = Clayton::from_tau(0.5).unwrap();
  assert!(c.theta().is_some());

  let u = array![0.1_f64, 0.5, 0.9];
  assert!(c.check_marginal(u.view()).is_ok());

  let scalar = c.partial_derivative_scalar(0.3, 0.6).unwrap();
  assert!((0.0..=1.0).contains(&scalar));
}

#[test]
fn sampling_needs_theta_not_tau() {
  let c = Clayton::new(2.0);
  let uv = c
    .sample(1_000, &Deterministic::new(7))
    .expect("theta alone is enough to sample");
  assert_eq!(uv.dim(), (1_000, 2));

  assert_eq!(
    Clayton::default()
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

/// The default `percent_point_unchecked` calls `percent_point`, so a sampler draws through an override of it alone, and the
/// default `percent_point` solves numerically without calling back.
#[test]
fn a_sampler_draws_through_an_override_of_percent_point_alone() {
  let own = HalvesTheLevel.sample(64, &Deterministic::new(7)).unwrap();
  let solved = ParameterisedWithoutTheta
    .sample(64, &Deterministic::new(7))
    .unwrap();
  assert_eq!(own.column(1), solved.column(1));
  for (a, b) in own.column(0).iter().zip(solved.column(0)) {
    assert!((a - 0.5 * b).abs() <= 1e-15, "{a} vs {b}");
  }
}

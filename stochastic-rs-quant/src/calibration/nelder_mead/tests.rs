use std::convert::Infallible;

use super::*;

struct Sphere;

impl CostFunction for Sphere {
  type Param = Vec<f64>;
  type Output = f64;
  type Error = Infallible;

  fn cost(&self, x: &Self::Param) -> Result<Self::Output, Self::Error> {
    Ok(x.iter().map(|value| value.powi(2)).sum())
  }
}

/// McKinnon's function with `(τ, θ, φ)`, from Kelley's (1999, §4) simplex. At (2, 6, 60) plain
/// Nelder–Mead contracts onto the origin, where ∂f/∂y = 1; the minimum is (0, −1/2).
struct McKinnon(f64, f64, f64);

const SMOOTH: McKinnon = McKinnon(2.0, 6.0, 60.0);

impl CostFunction for McKinnon {
  type Param = Vec<f64>;
  type Output = f64;
  type Error = Infallible;

  fn cost(&self, p: &Self::Param) -> Result<f64, Infallible> {
    let Self(tau, theta, phi) = *self;
    let (x, y) = (p[0], p[1]);
    let scale = if x <= 0.0 { theta * phi } else { theta };
    Ok(scale * x.abs().powf(tau) + y + y * y)
  }
}

#[test]
fn a_nelder_mead_trail_records_market_minus_model() {
  let options = CalibrationOptions::default().with_record_history(true);
  let simplex = vec![vec![2.0], vec![3.0]];
  let fit = nelder_mead_fit(Sphere, simplex, &options, 400, |x| {
    (x[0], vec![1.0], vec![1.0 + x[0]])
  })
  .unwrap();
  assert!(fit.history.len() > 1);
  for entry in &fit.history {
    assert_eq!(entry.residuals[0], 1.0 - (1.0 + entry.params));
  }
}

#[test]
fn a_simplex_without_volume_is_not_convergence() {
  let options = CalibrationOptions::default();
  let simplex = vec![vec![1.0, 1.0], vec![2.0, 2.0], vec![3.0, 3.0]];
  let fit = nelder_mead_fit(SMOOTH, simplex, &options, 400, |x| {
    (x.to_vec(), vec![0.0], vec![x[0]])
  })
  .unwrap();
  assert!(!fit.converged);
  assert_eq!(fit.message.as_deref(), Some("no numerical progress"));
}

#[test]
fn a_nelder_mead_fit_names_its_stop() {
  let simplex = vec![vec![2.0], vec![3.0]];
  let price = |x: &[f64]| (x[0], vec![1.0], vec![1.0 + x[0]]);
  let options = CalibrationOptions::default();
  let fit = nelder_mead_fit(Sphere, simplex.clone(), &options, 400, price).unwrap();
  assert!(fit.converged);
  assert_eq!(fit.message.as_deref(), Some("converged"));
  let capped = nelder_mead_fit(Sphere, simplex, &options, 3, price).unwrap();
  assert!(!capped.converged);
  assert_eq!(capped.message.as_deref(), Some("iteration cap reached"));
}

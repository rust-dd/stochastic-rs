//! The basket closed forms check the shape of their query.

use ndarray::array;
use stochastic_rs_quant::pricing::basket::ArithmeticBasketLevyPricer;
use stochastic_rs_quant::pricing::basket::GeometricBasketPricer;

fn levy() -> ArithmeticBasketLevyPricer {
  let rho = array![[1.0, 0.4], [0.4, 1.0]];
  ArithmeticBasketLevyPricer::new(array![0.5, 0.5], array![0.25, 0.2], rho)
}

fn geometric() -> GeometricBasketPricer {
  let rho = array![[1.0, 0.4], [0.4, 1.0]];
  GeometricBasketPricer::new(array![0.5, 0.5], array![0.25, 0.2], rho)
}

/// One spot of a two-asset basket has no price, at expiry as before it.
#[test]
#[should_panic(expected = "s must satisfy `s.len() == sigma.len()`, got s.len() = 1")]
fn a_levy_basket_with_a_spot_too_few_is_a_precondition() {
  levy().price_put(
    array![110.0].view(),
    100.0,
    0.05,
    array![0.0, 0.0].view(),
    0.0,
  );
}

#[test]
#[should_panic(expected = "s must satisfy `s.len() == sigma.len()`, got s.len() = 3")]
fn a_levy_basket_with_a_spot_too_many_is_a_precondition() {
  let spots = array![110.0, 100.0, 90.0];
  levy().price_call(spots.view(), 100.0, 0.05, array![0.0, 0.0].view(), 1.0);
}

#[test]
#[should_panic(expected = "q must satisfy `q.len() == sigma.len()`, got q.len() = 1")]
fn a_levy_basket_with_a_yield_too_few_is_a_precondition() {
  levy().price_call(
    array![110.0, 100.0].view(),
    100.0,
    0.05,
    array![0.0].view(),
    1.0,
  );
}

#[test]
#[should_panic(expected = "s must satisfy `s.len() == sigma.len()`, got s.len() = 1")]
fn a_geometric_basket_with_a_spot_too_few_is_a_precondition() {
  geometric().price_put(array![110.0].view(), 100.0, 0.05, array![0.0].view(), 1.0);
}

#[test]
#[should_panic(expected = "q must satisfy `q.len() == sigma.len()`, got q.len() = 3")]
fn a_geometric_basket_with_a_yield_too_many_is_a_precondition() {
  let q = array![0.0, 0.0, 0.0];
  geometric().price_call(array![110.0, 100.0].view(), 100.0, 0.05, q.view(), 1.0);
}

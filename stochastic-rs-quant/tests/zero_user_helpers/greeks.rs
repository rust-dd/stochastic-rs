use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::pricing::BsmCoc;
use stochastic_rs_quant::pricing::BsmPricer;

const QUERY: (f64, f64, f64, f64, f64) = (120.0, 110.0, 0.03, 0.01, 0.75);

/// `S·Γ/100`, the change in delta for a 1 % move in spot: delta's central difference at ±0.5 %.
#[test]
fn gamma_percent_is_the_delta_change_for_a_one_percent_move() {
  let (s, k, r, q, tau) = QUERY;
  let bsm = BsmPricer::new(0.25, BsmCoc::Merton1973);
  let delta = |spot: f64| bsm.delta(spot, k, r, q, tau, OptionType::Call);
  let one_percent = delta(1.005 * s) - delta(0.995 * s);
  let got = bsm.gamma_percent(s, k, r, q, tau);
  assert!(
    (got / one_percent - 1.0).abs() < 1e-3,
    "{got} vs {one_percent}"
  );
}

/// `S·zomma/100` is the volatility derivative of `gamma_percent`, by a central difference.
#[test]
fn zomma_percent_is_the_vol_derivative_of_gamma_percent() {
  let (s, k, r, q, tau) = QUERY;
  let (sigma, h) = (0.25, 1e-5);
  let gamma_percent =
    |sigma: f64| BsmPricer::new(sigma, BsmCoc::Merton1973).gamma_percent(s, k, r, q, tau);
  let difference = (gamma_percent(sigma + h) - gamma_percent(sigma - h)) / (2.0 * h);
  let got = BsmPricer::new(sigma, BsmCoc::Merton1973).zomma_percent(s, k, r, q, tau);
  assert!(
    (got / difference - 1.0).abs() < 1e-7,
    "{got} vs {difference}"
  );
}

//! The exact European against an independent quadrature, its identities, and the variance kernel.

use super::*;

fn params(a: f64, b: f64, sigma: f64, eta: f64, rho: f64) -> G2ppParams {
  G2ppParams {
    mean_reversion_x: a,
    mean_reversion_y: b,
    sigma_x: sigma,
    sigma_y: eta,
    rho,
    phi: 0.03,
  }
}

fn coupons(expiry: f64, tenor: f64) -> Vec<f64> {
  let n = (tenor * 2.0).round() as usize;
  (1..=n).map(|k| expiry + 0.5 * k as f64).collect()
}

fn price(
  p: &G2ppParams,
  direction: SwaptionDirection,
  strike: f64,
  expiry: f64,
  tenor: f64,
) -> f64 {
  g2pp_european_swaption(p, direction, strike, expiry, &coupons(expiry, tenor), 0.5)
}

/// Payers by a first-principles quadrature: `y` past the exercise boundary in closed form under the
/// risk-neutral measure, `x(T)` by a 40,000-point midpoint rule over twelve standard deviations.
#[test]
fn the_payers_match_a_first_principles_quadrature() {
  let rows = [
    (
      (0.5, 0.05, 0.01, 0.006, -0.99),
      0.035,
      1.0,
      2.0,
      8.602997225776935e-9,
    ),
    (
      (0.5, 0.05, 0.01, 0.006, -0.7),
      0.02,
      2.0,
      4.0,
      0.0358378062467403,
    ),
    (
      (0.1, 0.5, 0.01, 0.005, -0.3),
      0.05,
      10.0,
      10.0,
      0.0018538162073690062,
    ),
    (
      (0.1, 0.5, 0.01, 0.005, 0.5),
      0.035,
      5.0,
      5.0,
      0.013549482715557757,
    ),
    (
      (2.0, 0.1, 0.02, 0.01, -0.9),
      0.05,
      2.0,
      4.0,
      0.0002956000918283389,
    ),
    (
      (2.0, 0.1, 0.02, 0.01, 0.9),
      0.02,
      1.0,
      2.0,
      0.020416422294373876,
    ),
    (
      (0.05, 0.3, 0.015, 0.008, 0.0),
      0.02,
      5.0,
      5.0,
      0.0556625209834736,
    ),
    (
      (0.05, 0.3, 0.015, 0.008, 0.9),
      0.05,
      10.0,
      10.0,
      0.017429173012660185,
    ),
  ];
  for ((a, b, sigma, eta, rho), strike, expiry, tenor, want) in rows {
    let p = params(a, b, sigma, eta, rho);
    let got = price(&p, SwaptionDirection::Payer, strike, expiry, tenor);
    assert!(
      (got - want).abs() < 1e-14,
      "rho {rho}, {expiry}y x {tenor}y at {strike}: {got} against {want}"
    );
  }
}

/// A vol at the calibrator's `1e-6` floor, or `|ρ_xy| → 1`, folds the payoff onto a kink; the
/// references take an unsplit rule refined to `1e-17` and agree with a finer one to `2e-18`.
#[test]
fn a_kinked_payoff_integrates_to_its_reference() {
  let rows = [
    (
      (0.5, 0.05, 0.01, 1e-6, 0.0),
      SwaptionDirection::Payer,
      5.0,
      10.0,
      7.383455520750805e-5,
    ),
    (
      (0.5, 0.05, 0.01, 1e-6, -0.99),
      SwaptionDirection::Payer,
      2.0,
      4.0,
      0.0008613633250147597,
    ),
    (
      (0.5, 0.05, 1e-6, 0.006, 0.99),
      SwaptionDirection::Receiver,
      1.0,
      2.0,
      0.010211465774381265,
    ),
  ];
  for ((a, b, sigma, eta, rho), direction, expiry, tenor, want) in rows {
    let got = price(
      &params(a, b, sigma, eta, rho),
      direction,
      0.035,
      expiry,
      tenor,
    );
    assert!((got - want).abs() < 1e-14, "{got} against {want}");
  }
  let near_one = price(
    &params(0.5, 0.5, 0.01, 0.01, 0.999),
    SwaptionDirection::Payer,
    0.02,
    2.0,
    4.0,
  );
  assert!((near_one - 0.03577528477410794).abs() < 1e-14, "{near_one}");
}

/// Where the forward bond cannot pass par, a zero-strike payer is the spread of two bonds.
#[test]
fn a_zero_strike_payer_is_the_spread_of_two_bonds() {
  for p in [
    params(0.5, 0.05, 0.01, 0.006, -0.99),
    params(0.5, 0.05, 0.01, 0.006, -0.7),
    params(0.5, 0.05, 0.002, 0.0012, 0.0),
    params(0.5, 0.05, 0.002, 0.0012, 0.9),
  ] {
    let got = price(&p, SwaptionDirection::Payer, 0.0, 1.0, 2.0);
    let want = bond(&p, 1.0) - bond(&p, 3.0);
    assert!(
      (got / want - 1.0).abs() < 1e-13,
      "rho {}: {got} against {want}",
      p.rho
    );
  }
}

#[test]
fn a_payer_less_its_receiver_is_the_forward_swap() {
  for (p, strike, expiry, tenor) in [
    (params(0.5, 0.05, 0.01, 0.006, -0.7), 0.03, 1.0, 2.0),
    (params(0.1, 0.5, 0.01, 0.005, 0.5), 0.045, 5.0, 10.0),
    (params(0.5, 0.05, 0.01, 1e-6, -0.99), 0.035, 2.0, 4.0),
  ] {
    let times = coupons(expiry, tenor);
    let fixed = times
      .iter()
      .map(|&t| strike * 0.5 * bond(&p, t))
      .sum::<f64>();
    let swap = bond(&p, expiry) - bond(&p, *times.last().unwrap()) - fixed;
    let payer = price(&p, SwaptionDirection::Payer, strike, expiry, tenor);
    let receiver = price(&p, SwaptionDirection::Receiver, strike, expiry, tenor);
    assert!(
      (payer - receiver - swap).abs() < 1e-14,
      "{payer} − {receiver} against {swap}"
    );
  }
}

#[test]
fn a_one_payment_swaption_is_its_bond_option() {
  for direction in [SwaptionDirection::Payer, SwaptionDirection::Receiver] {
    for rho in [-0.99, -0.7, 0.0, 0.9] {
      let p = params(0.5, 0.05, 0.01, 0.006, rho);
      let closed = one_period(&p, direction, 0.03, 1.0, 1.5, 0.5);
      let integrated = integral(&p, direction, 0.03, 1.0, &[1.5], 0.5);
      assert!(
        (closed - integrated).abs() < 1e-14,
        "{closed} against {integrated}"
      );
    }
  }
}

/// `u, v` from the calibrator's `1e-6` floor to `a·τ = 90`, across the three branches.
#[test]
fn the_variance_kernel_is_its_integral_on_every_branch() {
  let rule = GaussLegendre::new(64.try_into().expect("64 > 0"));
  let f = |x: f64| if x == 0.0 { 1.0 } else { -(-x).exp_m1() / x };
  for (u, v) in [
    (1e-6, 1e-6),
    (1e-5, 3e-3),
    (0.3, 1.2),
    (1.0, 1.0),
    (2.5, 0.7),
    (5.0, 0.2),
    (40.0, 1e-3),
    (90.0, 30.0),
  ] {
    let want = rule.integrate(0.0, 1.0, |s| s * s * f(u * s) * f(v * s));
    let got = variance_kernel(u, v);
    assert!(
      (got / want - 1.0).abs() < 1e-13,
      "({u}, {v}): {got} against {want}"
    );
  }
}

use super::*;

/// Put-call parity has no volatility dependence, including for a truncated series.
#[test]
fn the_parity_spread_is_independent_of_volatility() {
  let spread_at = |v| {
    let m = merton_at(v, 0.5, 0.4, 10);
    let CallPut { call, put } = m.price_call_put(S, K, R, Q, TAU);
    call - put
  };
  let spread = spread_at(0.2);
  for v in [0.1, 0.19998, 0.20002, 0.4] {
    assert!((spread_at(v) - spread).abs() < 1e-12);
  }
}

/// Against 60-digit mpmath derivatives of the ten-term price (erfc for the CDF), where the `n = 3`
/// term crosses `d₂ = 0` and differencing an approximate CDF is unstable.
#[test]
fn volga_is_stable_when_a_term_crosses_zero_d2() {
  let m = merton_at(0.2, 0.5, 0.3, 10);
  for (strike, expected) in [
    (110.0, 11.328_499_405_604_37),
    (110.00001, 11.328_452_359_669_114),
    (109.99999, 11.328_546_451_612_872),
  ] {
    let actual = m.volga(110.0, strike, R, Q, 1.0);
    assert!((actual - expected).abs() < 1e-11, "K={strike}: {actual}");
  }
}

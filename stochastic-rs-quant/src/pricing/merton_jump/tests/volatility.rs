use super::*;

#[test]
fn zero_jump_variance_matches_black_scholes_near_zero_volatility() {
  for v in [1e-10, 1e-8, 5e-7, 1e-6, 1e-5, 0.2] {
    let m = merton_at(v, 0.5, 0.0, 50);
    let bs = BSMPricer::new(v, BSMCoc::Bsm1973);
    let cases = [
      (m.vega(S, S, 0.0, 0.0, 1.0), bs.vega(S, S, 0.0, 0.0, 1.0)),
      (m.vanna(S, S, 0.0, 0.0, 1.0), bs.vanna(S, S, 0.0, 0.0, 1.0)),
      (m.volga(S, S, 0.0, 0.0, 1.0), bs.vomma(S, S, 0.0, 0.0, 1.0)),
      (
        m.veta(S, S, 0.0, 0.0, 1.0),
        bs.dvega_dtime(S, S, 0.0, 0.0, 1.0),
      ),
    ];
    for (actual, expected) in cases {
      assert!(
        (actual - expected).abs() < 1e-8,
        "v={v}: {actual} vs {expected}"
      );
    }
  }
}

#[test]
fn zero_volatility_greeks_are_their_right_limits() {
  for gamma in [0.0, 0.4, 1.0] {
    let frozen = merton_at(0.0, 0.5, gamma, 50);
    let small = merton_at(1e-8, 0.5, gamma, 50);
    for (actual, expected) in [
      (
        frozen.vega(S, S, 0.0, 0.0, 1.0),
        small.vega(S, S, 0.0, 0.0, 1.0),
      ),
      (
        frozen.vanna(S, S, 0.0, 0.0, 1.0),
        small.vanna(S, S, 0.0, 0.0, 1.0),
      ),
      (
        frozen.volga(S, S, 0.0, 0.0, 1.0),
        small.volga(S, S, 0.0, 0.0, 1.0),
      ),
    ] {
      assert!(
        (actual - expected).abs() < 1e-6,
        "gamma={gamma}: {actual} vs {expected}"
      );
    }
  }
}

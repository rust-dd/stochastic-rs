//! The Greeks outside the closed-form families at and past expiry.

use stochastic_rs_quant::pricing::bachelier::BachelierPricer;
use stochastic_rs_quant::pricing::spread::MargrabePricer;

const K: f64 = 100.0;
const R: f64 = 0.05;
const Q: f64 = 0.02;
const NAN: f64 = f64::NAN;

#[test]
fn the_bachelier_vega_and_the_margrabe_deltas_are_nan_at_and_past_expiry() {
  let bachelier = BachelierPricer::new(20.0);
  let margrabe = MargrabePricer::new(0.25, 0.2, 0.4);
  for s in [90.0, 100.0, 110.0] {
    for tau in [0.0, -1.0, NAN] {
      let vega = bachelier.vega(s, K, R, Q, tau);
      let deltas = (
        margrabe.delta1(s, K, Q, 0.01, tau),
        margrabe.delta2(s, K, Q, 0.01, tau),
      );
      assert!(vega.is_nan(), "vega at s = {s}, tau = {tau}: {vega}");
      assert!(
        deltas.0.is_nan() && deltas.1.is_nan(),
        "s = {s}, tau = {tau}: {deltas:?}"
      );
    }
    assert!(bachelier.vega(s, K, R, Q, 1.0) > 0.0);
    let (d1, d2) = (
      margrabe.delta1(s, K, Q, 0.01, 1.0),
      margrabe.delta2(s, K, Q, 0.01, 1.0),
    );
    assert!(
      0.0 < d1 && d1 < 1.0 && -1.0 < d2 && d2 < 0.0,
      "s = {s}: {d1}, {d2}"
    );
  }
}

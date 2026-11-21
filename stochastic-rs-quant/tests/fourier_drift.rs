//! A Fourier model is drift-free: the same model prices the call at the carry the query supplies.

use num_complex::Complex64;
use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::pricing::bsm::BsmCoc;
use stochastic_rs_quant::pricing::bsm::BsmPricer;
use stochastic_rs_quant::pricing::fourier::BsmFourier;
use stochastic_rs_quant::pricing::fourier::CarrMadanPricer;
use stochastic_rs_quant::pricing::fourier::CosEngine;
use stochastic_rs_quant::pricing::fourier::FourierModelExt;
use stochastic_rs_quant::pricing::fourier::FrftCarrMadanPricer;
use stochastic_rs_quant::pricing::fourier::GilPelaezPricer;
use stochastic_rs_quant::pricing::fourier::HestonFourier;
use stochastic_rs_quant::pricing::fourier::LewisPricer;
use stochastic_rs_quant::pricing::heston::HestonParams;
use stochastic_rs_quant::traits::ModelPricer;

const CARRIES: [(f64, f64); 4] = [(0.0, 0.0), (0.05, 0.0), (0.05, 0.02), (-0.01, 0.03)];

const STRIKES: [f64; 3] = [80.0, 100.0, 125.0];

/// Heston calls at `STRIKES` per carry by Lewis's integral, mpmath at 30 digits.
const HESTON_CALLS: [[f64; 3]; 4] = [
  [21.699492971722037, 7.540266638818706, 0.5918210997105707],
  [25.09517801643501, 10.36186902096611, 1.2835974366788698],
  [23.269959905307026, 9.011278407541728, 0.9377998954923663],
  [18.41938643020191, 5.394964061465379, 0.2817199189917139],
];

fn heston() -> HestonFourier {
  HestonFourier::new(HestonParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
  })
}

#[test]
fn the_chf_is_a_martingale_density_and_c1_carries_no_drift() {
  let m = heston();
  let one = m.chf(1.0, Complex64::new(0.0, -1.0));
  assert!(
    (one.re - 1.0).abs() < 1e-12 && one.im.abs() < 1e-12,
    "E[e^X] = {one}"
  );
  let c = m.cumulants(1.0);
  assert!(
    (c.c1 - (-0.02)).abs() < 1e-12,
    "c1 = {} is the mpmath value -0.02 at (1.5, 0.04, 0.3, -0.7, 0.04, 1)",
    c.c1
  );
}

#[test]
fn the_pricers_apply_the_carry_from_the_query() {
  let m = heston();
  let bsm = BsmFourier { sigma: 0.2 };
  let closed = BsmPricer::new(0.2, BsmCoc::Merton1973);
  let cos = CosEngine::default();
  for (r, q) in CARRIES {
    for k in STRIKES {
      let gp = GilPelaezPricer::price_call(&bsm, 100.0, k, r, q, 1.0);
      let lw = LewisPricer::price_call(&bsm, 100.0, k, r, q, 1.0);
      let cs = cos.price(&bsm, 100.0, k, r, q, 1.0, OptionType::Call);
      let cm = CarrMadanPricer::default().price_call(&bsm, 100.0, k, r, q, 1.0);
      let frft =
        FrftCarrMadanPricer::cumulant_sized(&bsm, 1.0, 12.0).price_call(&bsm, 100.0, k, r, q, 1.0);
      let want = closed.price_call(100.0, k, r, q, 1.0);
      // Gil–Pelaez's band is its quadrature error.
      for (name, got, tol) in [
        ("gil-pelaez", gp, 2e-6),
        ("lewis", lw, 1e-9),
        ("cos", cs, 1e-8),
        ("carr-madan", cm, 5e-3),
        ("frft", frft, 5e-3),
      ] {
        assert!(
          (got - want).abs() < tol,
          "{name} at (r, q) = ({r}, {q}), k = {k}: {got} vs {want}"
        );
      }
      let call = m.price_call(100.0, k, r, q, 1.0);
      let put = cos.price(&m, 100.0, k, r, q, 1.0, OptionType::Put);
      let parity = 100.0 * (-q * 1.0_f64).exp() - k * (-r * 1.0_f64).exp();
      // COS(256, L = 10) spans `√c4` and meets parity to 1e-10; the rest, ≤ 4e-8, is Gil–Pelaez's.
      assert!(
        (call - put - parity).abs() < 1e-7,
        "parity at ({r}, {q}), k = {k}: {call} - {put} vs {parity}"
      );
    }
  }
}

#[test]
fn heston_gil_pelaez_calls_match_the_carry_in_the_exponent() {
  let m = heston();
  for ((r, q), row) in CARRIES.iter().zip(HESTON_CALLS) {
    for (k, want) in STRIKES.iter().zip(row) {
      let got = GilPelaezPricer::price_call(&m, 100.0, *k, *r, *q, 1.0);
      assert!(
        ((got - want) / want).abs() < 1e-10,
        "(r, q) = ({r}, {q}), k = {k}: {got} vs {want}"
      );
    }
  }
}

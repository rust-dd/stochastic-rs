// docs: processes#heston-stochastic-local-volatility
//! Backs the Heston SLV example on the processes catalog page.

use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stochastic::volatility::heston_slv::HestonSlv;
use stochastic_rs::traits::Expr;
use stochastic_rs::traits::Grid2D;
use stochastic_rs::traits::ProcessExt;

#[test]
fn heston_slv_two_factor_sample_under_an_expression_leverage() {
  // L(t, S) = 1.2 − 0.003 S, a leverage that falls with the spot, written as
  // an expression so the process can also run on a device; eta = 0.6 scales
  // the vol-of-vol.
  let leverage = Expr::lit(1.2) - Expr::x() * 0.003;
  let p = HestonSlv::<f64, _>::new(
    Some(100.0),
    Some(0.04),
    2.0,
    0.04,
    0.3,
    -0.7,
    0.03,
    0.6,
    leverage,
    1_000,
    Some(1.0),
    Deterministic::new(7),
  );
  assert!(p.device_ready());
  let [s_path, v_path] = p.sample();
  assert_eq!(s_path.len(), 1_000);
  assert!(s_path.iter().all(|&s| s > 0.0));
  assert!(v_path.iter().all(|&v| v >= 0.0));

  // A tabulated leverage — what a calibrated `LeverageSurface` converts
  // into — drives the same sampler on the host.
  let grid = Grid2D::new(
    Array1::from_vec(vec![0.0, 1.0]),
    Array1::from_vec(vec![50.0, 150.0]),
    Array2::from_elem((2, 2), 1.0),
  );
  let tabulated = p.with_leverage(grid);
  assert!(!tabulated.device_ready());
  let [s_grid, _] = tabulated.sample();
  assert_eq!(s_grid.len(), 1_000);
}

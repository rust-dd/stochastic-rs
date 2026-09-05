//! The processes whose coefficients vary with time. The host tabulates one
//! value per grid point and the kernel binds it per step, so what these cases
//! pin is that the device reads the same entry the host's own loop reads: an
//! off-by-one in that indexing shifts the whole term structure and shows up
//! as a different terminal law.

use stochastic_rs_core::simd_rng::Deterministic;
use ndarray::Array1;
use stochastic_rs_distributions::traits::Fn1D;
use stochastic_rs_stochastic::diffusion::gbm_ih::GbmIh;
use stochastic_rs_stochastic::interest::black_karasinski::BlackKarasinski;
use stochastic_rs_stochastic::interest::cir_2f::Cir2F;
use stochastic_rs_stochastic::interest::cir_pp::CirPlusPlus;
use stochastic_rs_stochastic::interest::ho_lee::HoLee;
use stochastic_rs_stochastic::interest::hull_white::HullWhite;
use stochastic_rs_stochastic::traits::ProcessExt;

use super::common::Device;
use super::common::M;
use super::common::agrees;
use super::common::all_finite;
use super::common::terminal_mean;
use super::common::within;

const N: usize = 253;

/// A level that climbs across the year, so reading the wrong entry moves the
/// terminal mean rather than cancelling out.
fn rising() -> Fn1D<f32> {
  Fn1D::Native(|t: f32| 0.02 + 0.03 * t)
}

#[test]
fn hull_white_agrees_with_the_cpu_law() {
  let build = || {
    HullWhite::<f32, _>::new(
      rising(),
      1.0,
      0.01,
      N,
      Some(0.02),
      Some(1.0),
      Deterministic::new(41),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "Hull-White");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.05,
    "Hull-White terminal mean",
  );
}

#[test]
fn ho_lee_agrees_with_the_cpu_law() {
  let build = || HoLee::<f32, _>::new(None, Some(0.03), 0.01, N, Some(1.0), Deterministic::new(43));
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "Ho-Lee");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.05,
    "Ho-Lee terminal mean",
  );
}

/// Black-Karasinski steps an exact Ornstein-Uhlenbeck transition in log
/// space, so the rate it reports is positive by construction.
#[test]
fn black_karasinski_stays_positive_and_agrees() {
  let build = || {
    BlackKarasinski::<f32, _>::new(
      rising(),
      1.0,
      0.2,
      N,
      Some(0.03),
      Some(1.0),
      Deterministic::new(47),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  within(&device, 0.0, f32::INFINITY, "Black-Karasinski");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.08,
    "Black-Karasinski terminal mean",
  );
}

/// The CIR++ shift is added in the report, so the reported path must clear
/// the shift's own floor even where the square-root state is at zero.
#[test]
fn cir_plus_plus_agrees_with_the_cpu_law() {
  let build = || {
    CirPlusPlus::<f32, _>::new(
      2.0,
      0.04,
      0.2,
      rising(),
      N,
      Some(0.04),
      Some(1.0),
      Some(false),
      Deterministic::new(53),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  within(&device, 0.0, f32::INFINITY, "CIR++");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.04,
    "CIR++ terminal mean",
  );
}

/// A term structure of volatilities: the device reads the same entry per step
/// the host's own loop reads, which a flat curve could not tell apart.
#[test]
fn inhomogeneous_gbm_agrees_with_the_cpu_law() {
  let sigmas: Array1<f32> = (0..N - 1)
    .map(|i| 0.1 + 0.2 * i as f32 / N as f32)
    .collect();
  let build = || {
    GbmIh::<f32, _>::new(
      0.05,
      0.2,
      N,
      Some(100.0),
      Some(1.0),
      Some(sigmas.clone()),
      Deterministic::new(59),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "inhomogeneous GBM");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.03,
    "inhomogeneous GBM terminal mean",
  );
}

/// The two-factor CIR model reports the shifted sum of its factors, which is
/// the first plane the launch writes.
#[test]
fn two_factor_cir_agrees_with_the_cpu_law() {
  use stochastic_rs_stochastic::diffusion::cir::Cir;
  let build = || {
    Cir2F::<f32, _>::new(
      Cir::new(2.0, 0.03, 0.1, N, Some(0.03), Some(1.0), None, Deterministic::new(2)),
      Cir::new(1.0, 0.01, 0.05, N, Some(0.01), Some(1.0), None, Deterministic::new(3)),
      rising(),
      Deterministic::new(73),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "two-factor CIR");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.04,
    "two-factor CIR terminal rate",
  );
}

/// The bridge pins its endpoint: the curve's last entry makes the step's own
/// variance ratio zero and its drift the whole remaining gap, so the device
/// lands on `xt` for the same reason the host assigns it.
#[test]
fn brownian_bridge_lands_on_its_endpoint() {
  use stochastic_rs_stochastic::process::brownian_bridge::BrownianBridge;
  let build = || {
    BrownianBridge::<f32, _>::new(0.3, N, Some(0.0), Some(1.0), Some(1.0), Deterministic::new(79))
  };
  let device = build().on::<Device>().sample_par(M);
  assert!(
    device.iter().all(|p| (p[N - 1] - 1.0).abs() < 1e-4),
    "a device bridge missed its endpoint"
  );
  agrees(
    terminal_mean(&build().sample_par(M)) + 1.0,
    terminal_mean(&device) + 1.0,
    1e-4,
    "Brownian bridge endpoint",
  );
}

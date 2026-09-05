//! The processes whose coefficients vary with time. The host tabulates one
//! value per grid point and the kernel binds it per step, so what these cases
//! pin is that the device reads the same entry the host's own loop reads: an
//! off-by-one in that indexing shifts the whole term structure and shows up
//! as a different terminal law.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::traits::Fn1D;
use stochastic_rs_stochastic::diffusion::gbm_ih::GbmIh;
use stochastic_rs_stochastic::interest::black_karasinski::BlackKarasinski;
use stochastic_rs_stochastic::interest::cir_2f::Cir2F;
use stochastic_rs_stochastic::interest::cir_pp::CirPlusPlus;
use stochastic_rs_stochastic::interest::hjm::Hjm;
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
      Cir::new(
        2.0,
        0.03,
        0.1,
        N,
        Some(0.03),
        Some(1.0),
        None,
        Deterministic::new(2),
      ),
      Cir::new(
        1.0,
        0.01,
        0.05,
        N,
        Some(0.01),
        Some(1.0),
        None,
        Deterministic::new(3),
      ),
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
    BrownianBridge::<f32, _>::new(
      0.3,
      N,
      Some(0.0),
      Some(1.0),
      Some(1.0),
      Deterministic::new(79),
    )
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

/// Heath-Jarrow-Morton is the first family to bind six curves at once — the
/// short rate, the bond price and the forward rate each take a drift and a
/// diffusion that vary with time — so beyond the usual agreement this case
/// pins that every slot reached the kernel: each coefficient is given a
/// distinct time dependence and each row's terminal mean is compared, since a
/// slot read from its neighbour would move one row and leave the others.
#[test]
fn hjm_agrees_with_the_cpu_law() {
  fn a(t: f32) -> f32 {
    0.02 + 0.03 * t
  }
  fn b(t: f32) -> f32 {
    0.01 + 0.01 * t
  }
  fn p(t: f32, _t_max: f32) -> f32 {
    1.0 - 0.2 * t
  }
  fn q(t: f32, _t_max: f32) -> f32 {
    -0.05 + 0.02 * t
  }
  fn v(t: f32, _t_max: f32) -> f32 {
    0.03 + 0.02 * t
  }
  fn alpha(t: f32, _t_max: f32) -> f32 {
    0.01 * t
  }
  fn sigma(t: f32, _t_max: f32) -> f32 {
    0.02 + 0.01 * t
  }
  let build = || {
    Hjm::<f32, _>::new(
      a as fn(f32) -> f32,
      b as fn(f32) -> f32,
      p as fn(f32, f32) -> f32,
      q as fn(f32, f32) -> f32,
      v as fn(f32, f32) -> f32,
      alpha as fn(f32, f32) -> f32,
      sigma as fn(f32, f32) -> f32,
      N,
      Some(0.03),
      Some(1.0),
      Some(0.04),
      Some(1.0),
      Deterministic::new(97),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  let host = build().sample_par(M);
  assert_eq!(device.len(), M);
  assert_eq!(device[0][0].len(), N);
  assert_eq!(device[0][0][0], 0.03, "the short rate starts at r0");
  assert_eq!(device[0][1][0], 1.0, "the bond price starts at p0");
  assert_eq!(device[0][2][0], 0.04, "the forward rate starts at f0");
  assert!(
    device
      .iter()
      .all(|p| p.iter().all(|row| row.iter().all(|v| v.is_finite()))),
    "HJM: a device path left the reals"
  );
  let row_mean = |paths: &[[Array1<f32>; 3]], c: usize| {
    let last = paths[0][c].len() - 1;
    paths.iter().map(|p| p[c][last] as f64).sum::<f64>() / paths.len() as f64
  };
  let row_spread = |paths: &[[Array1<f32>; 3]], c: usize| {
    let last = paths[0][c].len() - 1;
    let n = paths.len() as f64;
    let mean = paths.iter().map(|p| p[c][last] as f64).sum::<f64>() / n;
    (paths
      .iter()
      .map(|p| (p[c][last] as f64 - mean).powi(2))
      .sum::<f64>()
      / n)
      .sqrt()
  };
  for (c, name) in ["short rate", "bond price", "forward rate"]
    .iter()
    .enumerate()
  {
    let (h, d) = (row_mean(&host, c), row_mean(&device, c));
    assert!(
      (h - d).abs() < 0.01,
      "HJM {name} terminal mean: host {h}, device {d}"
    );
    agrees(
      row_spread(&host, c),
      row_spread(&device, c),
      0.08,
      &format!("HJM {name} terminal spread"),
    );
  }
  // The three rows take three independent shocks, so their terminal values
  // are uncorrelated. A kernel feeding one hashed component to every row
  // keeps each marginal above exactly right and fails only here.
  let row_corr = |paths: &[[Array1<f32>; 3]], a: usize, b: usize| {
    let last = paths[0][a].len() - 1;
    let (ma, mb) = (row_mean(paths, a), row_mean(paths, b));
    let (mut sab, mut saa, mut sbb) = (0.0f64, 0.0f64, 0.0f64);
    for p in paths {
      let (x, y) = (p[a][last] as f64 - ma, p[b][last] as f64 - mb);
      sab += x * y;
      saa += x * x;
      sbb += y * y;
    }
    sab / (saa * sbb).sqrt()
  };
  for (a, b) in [(0, 1), (1, 2), (0, 2)] {
    let c = row_corr(&device, a, b);
    assert!(
      c.abs() < 0.08,
      "HJM rows {a} and {b} are correlated on the device ({c}): the rows share a shock"
    );
  }
}

//! The processes whose families clamp, truncate or reflect. For these the
//! boundary is as much of the contract as the law, so each case asserts the
//! interval the family promises alongside the terminal statistic.

use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::correlation::teng::TengSCP;
use stochastic_rs_stochastic::correlation::van_emmerich::VanEmmerich;
use stochastic_rs_stochastic::diffusion::bessel::Bessel;
use stochastic_rs_stochastic::diffusion::bessel::SquaredBessel;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::diffusion::displaced_diffusion::DisplacedDiffusion;
use stochastic_rs_stochastic::diffusion::feller::FellerLogistic;
use stochastic_rs_stochastic::diffusion::feller_root::FellerRoot;
use stochastic_rs_stochastic::diffusion::gompertz::Gompertz;
use stochastic_rs_stochastic::diffusion::jacobi::Jacobi;
use stochastic_rs_stochastic::diffusion::kimura::Kimura;
use stochastic_rs_stochastic::diffusion::pearson::Pearson;
use stochastic_rs_stochastic::diffusion::verhulst::Verhulst;
use stochastic_rs_stochastic::traits::ProcessExt;

use super::common::Device;
use super::common::M;
use super::common::agrees;
use super::common::all_finite;
use super::common::terminal_mean;
use super::common::within;

#[test]
fn cir_stays_nonnegative_and_agrees() {
  let build = || {
    Cir::<f32, _>::new(
      2.0,
      0.04,
      0.2,
      253,
      Some(0.04),
      Some(1.0),
      None,
      Deterministic::new(3),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  within(&device, 0.0, f32::INFINITY, "CIR");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.03,
    "CIR terminal mean",
  );
}

#[test]
fn jacobi_stays_in_the_unit_interval() {
  let build = || {
    Jacobi::<f32, _>::new(
      0.3,
      0.6,
      0.2,
      253,
      Some(0.5),
      Some(1.0),
      Deterministic::new(3),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  within(&device, 0.0, 1.0, "Jacobi");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.05,
    "Jacobi terminal mean",
  );
}

#[test]
fn kimura_stays_in_the_unit_interval() {
  let build = || Kimura::<f32, _>::new(0.5, 0.2, 253, Some(0.5), Some(1.0), Deterministic::new(3));
  let device = build().on::<Device>().sample_par(M);
  within(&device, 0.0, 1.0, "Kimura");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.05,
    "Kimura terminal mean",
  );
}

/// Clamping is a property of the process, not of the step, so it picks the
/// family: the clamped form must honour `[0, K]` exactly.
#[test]
fn verhulst_honours_its_carrying_capacity() {
  let build = || {
    Verhulst::<f32, _>::new(
      1.0,
      2.0,
      0.3,
      253,
      Some(0.5),
      Some(1.0),
      Some(true),
      Deterministic::new(3),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  within(&device, 0.0, 2.0, "clamped Verhulst");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.05,
    "clamped Verhulst terminal mean",
  );
}

#[test]
fn unclamped_verhulst_agrees_with_the_cpu_law() {
  let build = || {
    Verhulst::<f32, _>::new(
      1.0,
      2.0,
      0.3,
      253,
      Some(0.5),
      Some(1.0),
      Some(false),
      Deterministic::new(3),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "unclamped Verhulst");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.05,
    "unclamped Verhulst terminal mean",
  );
}

#[test]
fn feller_logistic_stays_nonnegative() {
  let build = || {
    FellerLogistic::<f32, _>::new(
      1.0,
      1.0,
      0.3,
      253,
      Some(0.5),
      Some(1.0),
      Some(false),
      Deterministic::new(3),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  within(&device, 0.0, f32::INFINITY, "Feller logistic");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.05,
    "Feller logistic terminal mean",
  );
}

#[test]
fn feller_root_agrees_with_the_cpu_law() {
  let build = || {
    FellerRoot::<f32, _>::new(
      0.5,
      0.3,
      0.2,
      253,
      Some(0.5),
      Some(1.0),
      Deterministic::new(3),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "Feller root");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.06,
    "Feller root terminal mean",
  );
}

#[test]
fn squared_bessel_stays_nonnegative() {
  let build =
    || SquaredBessel::<f32, _>::new(3.0, 253, Some(1.0), Some(1.0), None, Deterministic::new(3));
  let device = build().on::<Device>().sample_par(M);
  within(&device, 0.0, f32::INFINITY, "squared Bessel");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.03,
    "squared Bessel terminal mean",
  );
}

/// The Bessel process runs its recursion in squared space and reports the
/// root, so a device path is non-negative for the same reason the host's is.
#[test]
fn bessel_stays_nonnegative() {
  let build = || Bessel::<f32, _>::new(3.0, 253, Some(1.0), Some(1.0), None, Deterministic::new(3));
  let device = build().on::<Device>().sample_par(M);
  within(&device, 0.0, f32::INFINITY, "Bessel");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.03,
    "Bessel terminal mean",
  );
}

#[test]
fn gompertz_stays_positive() {
  let build = || {
    Gompertz::<f32, _>::new(
      0.5,
      0.3,
      0.2,
      253,
      Some(1.0),
      Some(1.0),
      Deterministic::new(3),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  within(&device, 0.0, f32::INFINITY, "Gompertz");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.05,
    "Gompertz terminal mean",
  );
}

#[test]
fn pearson_agrees_with_the_cpu_law() {
  let build = || {
    Pearson::<f32, _>::new(
      1.0,
      0.3,
      0.0,
      0.0,
      0.01,
      253,
      Some(0.3),
      Some(1.0),
      Deterministic::new(3),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "Pearson");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.05,
    "Pearson terminal mean",
  );
}

/// The displaced diffusion steps the shifted variable and reports the shift
/// back out, so what the device must reproduce is the reported level, not the
/// state it stepped.
#[test]
fn displaced_diffusion_agrees_with_the_cpu_law() {
  let build = || {
    DisplacedDiffusion::<f32, _>::new(
      0.05,
      0.2,
      20.0,
      253,
      Some(100.0),
      Some(1.0),
      Deterministic::new(3),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "displaced diffusion");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.02,
    "displaced diffusion terminal mean",
  );
}

/// Teng's process is stepped on an unbounded variable and reported through a
/// `tanh`, so the reported correlation cannot leave `(−1, 1)` however the
/// state wanders.
#[test]
fn teng_correlation_stays_in_range() {
  let build = || TengSCP::<f32, _>::new(1.0, 0.3, 0.4, 0.2, 253, Some(1.0), Deterministic::new(3));
  let device = build().on::<Device>().sample_par(M);
  within(&device, -1.0, 1.0, "Teng correlation");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.10,
    "Teng terminal mean",
  );
}

#[test]
fn van_emmerich_correlation_stays_in_range() {
  let build =
    || VanEmmerich::<f32, _>::new(1.0, 0.3, 0.2, 0.2, 253, Some(1.0), Deterministic::new(3));
  let device = build().on::<Device>().sample_par(M);
  within(&device, -0.9999, 0.9999, "Van Emmerich correlation");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.10,
    "Van Emmerich terminal mean",
  );
}

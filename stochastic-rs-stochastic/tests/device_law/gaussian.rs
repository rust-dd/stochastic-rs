//! The unbounded single-state diffusions: nothing in their families clamps,
//! so the terminal mean — or, where a process reverts to zero, the terminal
//! standard deviation — is the whole comparison.

use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::ait_sahalia::AitSahalia;
use stochastic_rs_stochastic::diffusion::cev::Cev;
use stochastic_rs_stochastic::diffusion::ckls::Ckls;
use stochastic_rs_stochastic::diffusion::gbm_log::GbmLog;
use stochastic_rs_stochastic::diffusion::hyperbolic::Hyperbolic;
use stochastic_rs_stochastic::diffusion::hyperbolic2::Hyperbolic2;
use stochastic_rs_stochastic::diffusion::linear_sde::LinearSDE;
use stochastic_rs_stochastic::diffusion::logistic::Logistic;
use stochastic_rs_stochastic::diffusion::modified_cir::ModifiedCIR;
use stochastic_rs_stochastic::diffusion::nonlinear_sde::NonLinearSDE;
use stochastic_rs_stochastic::diffusion::quadratic::Quadratic;
use stochastic_rs_stochastic::diffusion::radial_ou::RadialOU;
use stochastic_rs_stochastic::diffusion::three_half::ThreeHalf;
use stochastic_rs_stochastic::interest::vasicek::Vasicek;
use stochastic_rs_stochastic::jump::mjd_log::MjdLog;
use stochastic_rs_stochastic::process::bm::Bm;
use stochastic_rs_stochastic::traits::ProcessExt;

use super::common::Device;
use super::common::M;
use super::common::agrees;
use super::common::all_finite;
use super::common::terminal_mean;
use super::common::terminal_std;

#[test]
fn cev_agrees_with_the_cpu_law() {
  let build = || {
    Cev::<f32, _>::new(
      0.05,
      0.2,
      0.8,
      253,
      Some(100.0),
      Some(1.0),
      Deterministic::new(5),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "CEV");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.02,
    "CEV terminal mean",
  );
}

#[test]
fn ckls_agrees_with_the_cpu_law() {
  let build = || {
    Ckls::<f32, _>::new(
      0.06,
      -1.5,
      0.3,
      0.5,
      253,
      Some(0.04),
      Some(1.0),
      Deterministic::new(5),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "CKLS");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.05,
    "CKLS terminal mean",
  );
}

#[test]
fn log_gbm_agrees_with_the_cpu_law() {
  let build = || {
    GbmLog::<f32, _>::new(
      Some(0.05),
      None,
      None,
      None,
      0.2,
      253,
      Some(100.0),
      Some(1.0),
      Deterministic::new(7),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "log-GBM");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.02,
    "log-GBM terminal mean",
  );
}

#[test]
fn logistic_agrees_with_the_cpu_law() {
  let build =
    || Logistic::<f32, _>::new(0.5, 0.2, 253, Some(1.0), Some(1.0), Deterministic::new(7));
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "logistic");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.03,
    "logistic terminal mean",
  );
}

#[test]
fn three_half_agrees_with_the_cpu_law() {
  let build = || {
    ThreeHalf::<f32, _>::new(
      2.0,
      0.04,
      0.3,
      253,
      Some(0.04),
      Some(1.0),
      Deterministic::new(7),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "3/2");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.05,
    "3/2 terminal mean",
  );
}

#[test]
fn radial_ou_agrees_with_the_cpu_law() {
  let build =
    || RadialOU::<f32, _>::new(1.0, 0.3, 253, Some(1.0), Some(1.0), Deterministic::new(7));
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "radial OU");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.05,
    "radial OU terminal mean",
  );
}

#[test]
fn vasicek_agrees_with_the_cpu_law() {
  let build = || {
    Vasicek::<f32, _>::new(
      0.5,
      0.04,
      0.02,
      253,
      Some(0.03),
      Some(1.0),
      Deterministic::new(7),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "Vasicek");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.05,
    "Vasicek terminal mean",
  );
}

#[test]
fn linear_sde_agrees_with_the_cpu_law() {
  let build = || {
    LinearSDE::<f32, _>::new(
      0.02,
      0.3,
      0.2,
      253,
      Some(1.0),
      Some(1.0),
      Deterministic::new(9),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "linear SDE");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.03,
    "linear SDE terminal mean",
  );
}

#[test]
fn quadratic_agrees_with_the_cpu_law() {
  let build = || {
    Quadratic::<f32, _>::new(
      0.02,
      0.1,
      -0.05,
      0.2,
      253,
      Some(1.0),
      Some(1.0),
      Deterministic::new(9),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "quadratic");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.03,
    "quadratic terminal mean",
  );
}

/// The hyperbolic drift pulls the state to zero, so the terminal mean is a
/// ratio of two near-zero numbers and the spread is the stable statistic.
#[test]
fn hyperbolic_agrees_with_the_cpu_law() {
  let build =
    || Hyperbolic::<f32, _>::new(1.0, 0.3, 253, Some(0.5), Some(1.0), Deterministic::new(9));
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "hyperbolic");
  agrees(
    terminal_std(&build().sample_par(M)),
    terminal_std(&device),
    0.05,
    "hyperbolic terminal spread",
  );
}

/// Also zero-reverting; see [`hyperbolic_agrees_with_the_cpu_law`].
#[test]
fn modified_cir_agrees_with_the_cpu_law() {
  let build =
    || ModifiedCIR::<f32, _>::new(1.0, 0.2, 253, Some(0.5), Some(1.0), Deterministic::new(9));
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "modified CIR");
  agrees(
    terminal_std(&build().sample_par(M)),
    terminal_std(&device),
    0.05,
    "modified CIR terminal spread",
  );
}

#[test]
fn hyperbolic_diffusion_agrees_with_the_cpu_law() {
  let build = || {
    Hyperbolic2::<f32, _>::new(
      0.5,
      1.0,
      1.0,
      0.0,
      0.3,
      253,
      Some(0.5),
      Some(1.0),
      Deterministic::new(9),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "hyperbolic diffusion");
  agrees(
    terminal_std(&build().sample_par(M)),
    terminal_std(&device),
    0.05,
    "hyperbolic diffusion terminal spread",
  );
}

/// Parametrised as a short rate reverting to 5% at speed 3, where the
/// terminal mean is a statistic two independent streams can agree on. The
/// model's `a₋₁/X` drift makes a weakly-reverting parametrisation dominated
/// by near-zero excursions, whose terminal mean is heavy-tailed enough that
/// no tolerance would mean anything.
#[test]
fn ait_sahalia_agrees_with_the_cpu_law() {
  let build = || {
    AitSahalia::<f32, _>::new(
      0.0001,
      0.15,
      -3.0,
      0.0,
      0.0004,
      0.0,
      0.05,
      1.5,
      253,
      Some(0.05),
      Some(1.0),
      Deterministic::new(11),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "Ait-Sahalia");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.05,
    "Ait-Sahalia terminal mean",
  );
}

#[test]
fn nonlinear_sde_agrees_with_the_cpu_law() {
  let build = || {
    NonLinearSDE::<f32, _>::new(
      0.0001,
      0.15,
      -3.0,
      0.0,
      0.0,
      0.0,
      0.2,
      1.0,
      253,
      Some(0.05),
      Some(1.0),
      Deterministic::new(11),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "non-linear SDE");
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.05,
    "non-linear SDE terminal mean",
  );
}

/// Brownian motion is the additive family with Gaussian increments: the mean
/// is zero by construction, so the spread is what carries the law.
#[test]
fn brownian_motion_agrees_with_the_cpu_law() {
  let build = || Bm::<f32, _>::new(253, Some(1.0), Deterministic::new(13));
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "Brownian motion");
  agrees(
    terminal_std(&build().sample_par(M)),
    terminal_std(&device),
    0.05,
    "Brownian motion terminal spread",
  );
}

/// Merton's jump diffusion: the device draws its own Poisson count per step
/// and aggregates the jump sizes into one normal, as the host sampler does,
/// so the terminal mean carries both the diffusion and the jump compensator.
#[test]
fn merton_jump_diffusion_agrees_with_the_cpu_law() {
  let build = || {
    MjdLog::<f32, _>::new(
      Some(0.05),
      None,
      None,
      None,
      0.2,
      3.0,
      -0.05,
      0.1,
      253,
      Some(100.0),
      Some(1.0),
      Deterministic::new(89),
    )
  };
  let device = build().on::<Device>().sample_par(M);
  all_finite(&device, "Merton jump diffusion");
  assert!(
    device.iter().all(|p| p.iter().all(|&v| v > 0.0)),
    "the log-price form let the spot reach zero"
  );
  agrees(
    terminal_mean(&build().sample_par(M)),
    terminal_mean(&device),
    0.03,
    "Merton jump diffusion terminal mean",
  );
}

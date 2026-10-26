use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::Unseeded;

use super::*;
use crate::diffusion::cir::Cir;
use crate::diffusion::gbm::Gbm;
use crate::diffusion::ou::Ou;

/// `on::<Cpu>()` is the identity on a process's output, and the `Cpu`
/// backend's `euler_paths` is the process's own `sample_par` — bit for bit.
#[test]
fn cpu_backend_is_the_process_sampler() {
  // `Deterministic` advances as it samples, so every comparison starts from a
  // freshly constructed process with the same seed.
  let gbm = || {
    Gbm::new(
      0.05,
      0.2,
      64,
      Some(100.0),
      Some(1.0),
      Deterministic::new(42),
    )
  };
  let plain = gbm().sample_par(16);
  let switched = gbm().on::<Cpu>().sample_par(16);
  let through_trait = Cpu.euler_paths(&gbm(), 16);
  assert_eq!(plain.len(), 16);
  for i in 0..16 {
    assert_eq!(plain[i].to_vec(), switched[i].to_vec());
    assert_eq!(plain[i].to_vec(), through_trait[i].to_vec());
  }

  const {}
  let ou = || {
    Ou::new(
      2.0,
      1.0,
      0.5,
      33,
      Some(1.0),
      Some(2.0),
      Deterministic::new(7),
    )
  };
  assert_eq!(ou().on::<Cpu>().sample().to_vec(), ou().sample().to_vec());
  let cir = || {
    Cir::new(
      1.5,
      0.04,
      0.3,
      33,
      Some(0.09),
      Some(1.0),
      None,
      Deterministic::new(7),
    )
  };
  assert_eq!(
    cir().on::<Cpu>().sample_map(4, |p| p[32]),
    cir().sample_map(4, |p| p[32])
  );
  assert_eq!(gbm().on::<Cpu>().sample_par(0).len(), 0);
}

/// The body keys noise on `first_path + path` and rows on `path * increments`, both in 32 bits,
/// so a launch past that range is refused rather than handed a wrapped value.
#[test]
fn launch_scalars_refuse_a_wrapped_index() {
  let last = u32::MAX as usize;
  assert_eq!(launch_scalars(253, 7, 3, 0), Ok((253, 3, 7)));
  assert!(launch_scalars(253, last - 2, 2, 0).is_ok());
  assert!(launch_scalars(253, last - 1, 2, 0).is_err());
  assert!(launch_scalars(253, 0, last / 2 + 1, 2).is_err());
  assert!(launch_scalars(last + 1, 0, 1, 0).is_err());
  assert!(launch_scalars(253, usize::MAX, 1, 0).is_err());
  assert!(launch_scalars(253, 1, usize::MAX, 2).is_err());
}

#[test]
fn switching_the_backend_keeps_the_parameters() {
  let gbm = Gbm::new(0.05, 0.2, 64, Some(100.0), Some(2.0), Unseeded).on::<Cpu>();
  assert_eq!(
    (gbm.mu, gbm.sigma, gbm.n, gbm.x0, gbm.t),
    (0.05, 0.2, 64, Some(100.0), Some(2.0))
  );
  let cir = Cir::new(
    1.5,
    0.04,
    0.3,
    10,
    Some(0.09),
    Some(2.0),
    Some(true),
    Unseeded,
  )
  .on::<Cpu>();
  assert_eq!(cir.use_sym, Some(true));
  assert_eq!(cir.euler_spec().encode().0, 2);
  assert_eq!(
    (cir.initial_value(), cir.grid_points(), cir.horizon()),
    (0.09, 10, 2.0)
  );
  assert_eq!(gbm.euler_spec().encode().0, 0);
  let (code, params) = Ou::new(2.0, 1.0, 0.5, 5, None, None, Unseeded)
    .euler_spec()
    .encode();
  assert_eq!(code, 1);
  assert_eq!(&params[..3], &[2.0, 1.0, 0.5]);
  assert!(
    params[3..].iter().all(|p| *p == 0.0),
    "unused slots are zero"
  );
}

#[test]
fn try_sample_par_matches_sample_par_on_the_cpu() {
  let gbm = || Gbm::new(0.05, 0.2, 64, Some(100.0), Some(1.0), Deterministic::new(9));
  assert_eq!(gbm().try_sample_par(8).expect("cpu"), gbm().sample_par(8));
  let ou = || {
    Ou::new(
      2.0,
      1.0,
      0.5,
      64,
      Some(1.0),
      Some(1.0),
      Deterministic::new(9),
    )
  };
  assert_eq!(ou().try_sample_par(8).expect("cpu"), ou().sample_par(8));
  let cir = || {
    Cir::new(
      1.5,
      0.04,
      0.3,
      64,
      Some(0.09),
      Some(1.0),
      None,
      Deterministic::new(9),
    )
  };
  assert_eq!(cir().try_sample_par(8).expect("cpu"), cir().sample_par(8));
  let fgn = || crate::noise::fgn::Fgn::<f64, _>::new(0.7, 64, Some(1.0), Deterministic::new(9));
  assert_eq!(fgn().try_sample_par(4).expect("cpu"), fgn().sample_par(4));
}

#[test]
fn host_map_and_chunk_calls_match_the_process_sampler() {
  let gbm = || Gbm::new(0.05, 0.2, 32, Some(100.0), Some(1.0), Deterministic::new(4));
  let direct: Vec<f64> = gbm().sample_map(6, |p| p[31]);
  let through: Vec<f64> = Cpu.try_euler_paths_map(&gbm(), 6, |p| p[31]).expect("cpu");
  assert_eq!(direct, through);
  assert_eq!(
    Cpu.try_euler_paths(&gbm(), 6).expect("cpu"),
    gbm().sample_par(6),
    "the host stream is the process's own"
  );
}

#[test]
fn matrix_form_matches_the_rows() {
  let gbm = || Gbm::new(0.05, 0.2, 32, Some(100.0), Some(1.0), Deterministic::new(6));
  let rows = gbm().sample_par(5);
  let matrix = gbm().try_sample_matrix(5).expect("cpu");
  assert_eq!(matrix.dim(), (5, 32));
  for (i, row) in rows.iter().enumerate() {
    assert_eq!(matrix.row(i), row.view());
  }
}

#[test]
fn device_seed_follows_the_seed_source() {
  let a = Gbm::new(0.05, 0.2, 8, None, None, Deterministic::new(3));
  let b = Gbm::new(0.05, 0.2, 8, None, None, Deterministic::new(3));
  assert_eq!(a.device_seed(), b.device_seed());
  assert_ne!(
    a.device_seed(),
    Gbm::new(0.05, 0.2, 8, None, None, Deterministic::new(4)).device_seed()
  );
  let u = Gbm::new(0.05, 0.2, 8, None, None, Unseeded);
  assert_ne!(u.device_seed(), u.device_seed());
}

/// Every compiled device back-end reproduces the lognormal moments and is
/// deterministic in its seed; the device kernels share one integer hash for
/// their uniforms, so two device back-ends agree seed for seed up to the
/// `f32` libm rounding of Box–Muller.
#[cfg(any(
  all(feature = "metal", target_os = "macos", target_arch = "aarch64"),
  feature = "cuda"
))]
mod devices;

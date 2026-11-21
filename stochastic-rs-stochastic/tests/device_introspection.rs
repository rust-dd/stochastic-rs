//! What a process says about where it runs before it runs there: which handle it holds (the type
//! records the marker, not the ordinal or budget) and why a configuration falls back to the host.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use ndarray::Array2;
use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::exp::SimdExp;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_stochastic::autoregressive::ar::Ar;
use stochastic_rs_stochastic::device::Cpu;
use stochastic_rs_stochastic::diffusion::cir::Cir;
use stochastic_rs_stochastic::diffusion::gbm::Gbm;
use stochastic_rs_stochastic::diffusion::wishart::Wishart;
use stochastic_rs_stochastic::interest::black_karasinski::BlackKarasinski;
use stochastic_rs_stochastic::interest::duffie_kan_jump_exp::DuffieKanJumpExp;
use stochastic_rs_stochastic::interest::vasicek::Vasicek;
use stochastic_rs_stochastic::jump::bates::Bates;
use stochastic_rs_stochastic::jump::jump_fou::JumpFou;
use stochastic_rs_stochastic::jump::jump_fou_custom::JumpFouCustom;
use stochastic_rs_stochastic::jump::kou::Kou;
use stochastic_rs_stochastic::jump::levy_diffusion::LevyDiffusion;
use stochastic_rs_stochastic::jump::merton::Merton;
use stochastic_rs_stochastic::jump::mjd_log::MjdLog;
use stochastic_rs_stochastic::noise::fgn::Fgn;
use stochastic_rs_stochastic::params::Drift;
use stochastic_rs_stochastic::params::VarianceBoundary;
use stochastic_rs_stochastic::process::subordinator::ctrw::Ctrw;
use stochastic_rs_stochastic::process::subordinator::ctrw::CtrwJumpLaw;
use stochastic_rs_stochastic::process::subordinator::ctrw::CtrwWaitingLaw;
use stochastic_rs_stochastic::process::subordinator::poisson_subordinator::PoissonSubordinator;
use stochastic_rs_stochastic::process::subordinator::tempered_stable::TemperedStableSubordinator;
use stochastic_rs_stochastic::traits::ProcessExt;
use stochastic_rs_stochastic::volatility::bns::Bns;
use stochastic_rs_stochastic::volatility::fbates_svj::FbatesSvj;
use stochastic_rs_stochastic::volatility::hkde::Hkde;

/// A process hands back the handle it holds, and the handle opens.
#[test]
fn a_process_reports_the_backend_it_holds() {
  let gbm = Gbm::<f64, _>::new(0.05, 0.2, 256, 100.0, None, Deterministic::new(1));
  assert_eq!(*gbm.backend(), Cpu);
  let info = gbm.probe().expect("the CPU device always opens");
  assert_eq!(info.backend, "Cpu");
  assert!(info.precisions.contains(&"f64"));
}

/// The fractional processes carry their backend inside the noise pipeline, not in a field of
/// their own; the accessor is the same either way.
#[test]
fn a_fractional_process_reports_its_backend_too() {
  let fgn = Fgn::<f64, _>::new(0.7, 128, Some(1.0), Deterministic::new(2));
  assert_eq!(*fgn.backend(), Cpu);
  assert_eq!(fgn.probe().expect("cpu").backend, "Cpu");
}

/// A configuration the kernels carry whole says nothing; one they cannot says why, so a caller
/// can print the reason instead of finding the fallback in a profile.
#[test]
fn a_configuration_past_the_kernels_names_its_reason() {
  let first = Ar::<f64, _>::new(array![0.5], 1.0, 128, None, Deterministic::new(3));
  assert!(first.device_ready());
  assert_eq!(first.device_fallback(), None);

  let second = Ar::<f64, _>::new(array![0.5, 0.2], 1.0, 128, None, Deterministic::new(3));
  assert!(!second.device_ready());
  let why = second
    .device_fallback()
    .expect("an order above one has a reason");
  assert!(
    why.contains("order above one"),
    "the reason should name the order, got {why:?}"
  );
}

/// The two answers cannot disagree: `device_ready` is defined as the absence
/// of a reason, and every override states the reason rather than the bool.
#[test]
fn readiness_is_exactly_the_absence_of_a_reason() {
  let identity = Array2::<f64>::eye(3);
  let wide = Wishart::<f64, _>::new(
    4.0,
    identity.clone(),
    identity.clone(),
    identity,
    64,
    Some(1.0),
    Deterministic::new(5),
  );
  assert_eq!(wide.device_ready(), wide.device_fallback().is_none());
  assert!(
    wide.device_fallback().is_some(),
    "a three-dimensional Wishart is past the kernel's two"
  );

  let plain = Vasicek::<f64, _>::new(1.5, 0.04, 0.3, 128, None, None, Deterministic::new(7));
  assert_eq!(plain.device_ready(), plain.device_fallback().is_none());
  assert_eq!(plain.device_fallback(), None);
}

fn flat(_t: f64) -> f64 {
  0.05
}

/// A Black–Karasinski reversion of exactly zero reaches the kernels, which step its limit.
#[test]
fn a_zero_reversion_black_karasinski_reaches_the_device() {
  let build = |a: f64| {
    BlackKarasinski::<f64, _>::new(
      flat as fn(f64) -> f64,
      a,
      0.1,
      64,
      Some(0.03),
      Some(1.0),
      Deterministic::new(11),
    )
  };
  assert!(build(0.0).device_ready() && build(0.0).device_fallback().is_none());
  assert!(build(0.1).device_ready());
}

/// Every type in `src` that implements `jump_intensity`, so its kernel reads the step's count; the
/// engine's trait defaults and its test probes are not processes.
fn count_readers() -> BTreeSet<String> {
  let mut found = BTreeSet::new();
  let mut dirs = vec![Path::new(env!("CARGO_MANIFEST_DIR")).join("src")];
  while let Some(dir) = dirs.pop() {
    let entries = fs::read_dir(&dir).expect("the crate's own source tree is readable");
    for path in entries.flatten().map(|entry| entry.path()) {
      if path.is_dir() {
        dirs.push(path);
        continue;
      }
      if path.extension().is_none_or(|e| e != "rs")
        || path.to_string_lossy().contains("family_parity")
      {
        continue;
      }
      let text = fs::read_to_string(&path).expect("a source file is readable");
      for (at, _) in text.match_indices("fn jump_intensity") {
        let header = &text[text[..at].rfind("\nimpl").unwrap_or(0)..at];
        if header.contains("\npub trait ") {
          continue;
        }
        let (_, implementor) = header
          .split_once(" for ")
          .unwrap_or_else(|| panic!("{}: an impl names its type", path.display()));
        found.insert(
          implementor
            .trim_start()
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect(),
        );
      }
    }
  }
  found
}

/// A device draws a step's Poisson count only up to 64 events, exact to `f32` up to 25 expected:
/// every process whose kernel reads the count names that limit past 25 and keeps to the host.
#[test]
fn a_dense_count_keeps_every_count_reading_process_on_the_host() {
  const N: usize = 65;
  fn normal() -> SimdNormal<f64> {
    SimdNormal::new(0.0, 0.01)
  }
  type Fallback = Box<dyn Fn(f64) -> Option<&'static str>>;
  let cases: [(&str, Fallback); 14] = [
    (
      "Kou",
      Box::new(|rate| {
        Kou::new(
          0.03,
          0.2,
          rate,
          0.0,
          normal(),
          N,
          Some(0.0),
          Some(1.0),
          Unseeded,
        )
        .device_fallback()
      }),
    ),
    (
      "Merton",
      Box::new(|rate| {
        Merton::new(
          0.03,
          0.2,
          rate,
          0.0,
          normal(),
          N,
          Some(0.0),
          Some(1.0),
          Unseeded,
        )
        .device_fallback()
      }),
    ),
    (
      "LevyDiffusion",
      Box::new(|rate| {
        LevyDiffusion::new(0.01, 0.2, rate, normal(), N, Some(0.0), Some(1.0), Unseeded)
          .device_fallback()
      }),
    ),
    (
      "JumpFou",
      Box::new(|rate| {
        JumpFou::new(
          0.65,
          1.5,
          0.0,
          0.2,
          rate,
          normal(),
          N,
          Some(0.0),
          Some(1.0),
          Unseeded,
        )
        .device_fallback()
      }),
    ),
    (
      "JumpFouCustom",
      Box::new(|rate| {
        JumpFouCustom::new(
          0.65,
          1.5,
          0.0,
          0.2,
          N,
          Some(0.0),
          Some(1.0),
          SimdExp::<f64>::new(rate),
          normal(),
          Unseeded,
        )
        .device_fallback()
      }),
    ),
    (
      "Bates",
      Box::new(|rate| {
        Bates::new(
          Drift::Mu(0.05),
          rate,
          0.0,
          0.04,
          1.5,
          0.3,
          -0.6,
          normal(),
          N,
          100.0,
          0.04,
          Some(1.0),
          VarianceBoundary::Floor,
          Unseeded,
        )
        .device_fallback()
      }),
    ),
    (
      "FbatesSvj",
      Box::new(|rate| {
        FbatesSvj::new(
          0.7,
          0.05,
          100.0,
          0.04,
          0.04,
          1.5,
          0.3,
          -0.6,
          rate,
          0.0,
          0.01,
          N,
          Some(1.0),
          Unseeded,
        )
        .device_fallback()
      }),
    ),
    (
      "Bns",
      Box::new(|rate| {
        Bns::new(100.0, 0.04, rate, 0.05, 1.0, 0.01, N, Some(1.0), Unseeded).device_fallback()
      }),
    ),
    (
      "DuffieKanJumpExp",
      Box::new(|rate| {
        DuffieKanJumpExp::new(
          0.5,
          0.5,
          0.1,
          0.3,
          -0.5,
          0.1,
          0.02,
          0.1,
          -0.5,
          0.1,
          0.02,
          0.1,
          rate,
          0.01,
          N,
          Some(0.03),
          Some(0.01),
          Some(1.0),
          Unseeded,
        )
        .device_fallback()
      }),
    ),
    (
      "Hkde",
      Box::new(|rate| {
        Hkde::new(
          0.05,
          1.5,
          0.04,
          0.3,
          -0.6,
          0.04,
          rate,
          0.4,
          10.0,
          5.0,
          N,
          100.0,
          Some(1.0),
          VarianceBoundary::Floor,
          Unseeded,
        )
        .device_fallback()
      }),
    ),
    (
      "PoissonSubordinator",
      Box::new(|rate| {
        PoissonSubordinator::new(rate, N, None, Some(1.0), Unseeded).device_fallback()
      }),
    ),
    (
      "MjdLog",
      Box::new(|rate| {
        MjdLog::new(
          Drift::Mu(0.05),
          0.2,
          rate,
          0.0,
          0.01,
          N,
          100.0,
          Some(1.0),
          Unseeded,
        )
        .device_fallback()
      }),
    ),
    (
      "TemperedStableSubordinator",
      // c/α · ε^{−α} = rate at α = 0.5 and ε = 1e-4.
      Box::new(|rate| {
        TemperedStableSubordinator::new(0.5, rate / 200.0, 1.0, 1e-4, N, None, Some(1.0), Unseeded)
          .device_fallback()
      }),
    ),
    (
      "Ctrw",
      Box::new(|rate| {
        Ctrw::<f64, _>::new(
          CtrwWaitingLaw::Exponential { rate },
          CtrwJumpLaw::Normal {
            mean: 1.0,
            std: 1e-6,
          },
          N,
          None,
          Some(1.0),
          Unseeded,
        )
        .device_fallback()
      }),
    ),
  ];
  assert_eq!(
    cases
      .iter()
      .map(|(name, _)| name.to_string())
      .collect::<BTreeSet<_>>(),
    count_readers(),
    "the table must hold every process whose kernel reads the step's count"
  );
  for (name, fallback) in cases {
    assert!(
      fallback(100.0 * 64.0).is_some_and(|why| why.contains("64")),
      "{name}: a hundred expected events a step must name the kernels' count"
    );
    assert_eq!(
      fallback(20.0 * 64.0),
      None,
      "{name}: twenty a step run on the device"
    );
  }
}

/// A setter that rebuilds no cache is defined for every backend, so a process can be adjusted after
/// it has moved to a device.
#[test]
fn a_setter_is_defined_for_every_backend() {
  fn retune<B>(c: Cir<f64, Unseeded, B>) -> Cir<f64, Unseeded, B> {
    c.with_sigma(0.3).with_points(128)
  }
  let c = retune(Cir::<f64>::default());
  assert_eq!((c.sigma, c.n), (0.3, 128));
}

#[cfg(metal_backend)]
#[test]
fn a_device_backed_process_takes_its_setters() {
  use stochastic_rs_stochastic::device::Metal;
  let c = Cir::<f32>::default()
    .on::<Metal>()
    .with_sigma(0.3)
    .with_points(128);
  assert_eq!((c.sigma, c.n), (0.3, 128));
}

/// One macro writes both device handles: the same constructors, the same one-byte budget floor.
#[cfg(device_backend)]
macro_rules! handle_surface {
  ($test:ident, $handle:ident) => {
    #[test]
    fn $test() {
      use stochastic_rs_stochastic::device::{DEFAULT_BATCH_BUDGET_BYTES, DeviceError, $handle};
      let h = $handle::new(0).with_batch_budget(1 << 20);
      assert_eq!((h.ordinal, h.batch_budget), (0, 1 << 20));
      assert_eq!($handle::default().batch_budget, DEFAULT_BATCH_BUDGET_BYTES);
      assert_eq!($handle::new(1).with_batch_budget(0).batch_budget, 1);
      let _: &dyn Fn() -> Result<$handle, DeviceError> = &$handle::from_env;
    }
  };
}

#[cfg(metal_backend)]
handle_surface!(
  the_metal_handle_has_the_same_surface_as_the_cuda_handle,
  Metal
);

#[cfg(feature = "cuda")]
handle_surface!(
  the_cuda_handle_has_the_same_surface_as_the_metal_handle,
  Cuda
);

/// A release frees caches, not a seed's stream: two identically built processes, one sampled before
/// the release and one after, draw the same paths.
#[cfg(metal_backend)]
#[test]
fn a_released_metal_cache_rebuilds_on_the_next_launch() {
  use stochastic_rs_stochastic::device::{Metal, release_host_scratch};

  let gbm = || {
    Gbm::<f32, Deterministic>::new(0.05, 0.2, 64, 100.0, Some(1.0), Deterministic::new(1))
      .on::<Metal>()
  };
  let before = gbm().sample_par(8);
  Metal::default().release();
  release_host_scratch();
  assert_eq!(before, gbm().sample_par(8));
}

//! Every family, launched on a device without a process that happens to use
//! it.
//!
//! The kernels the CUDA and Metal back-ends run are generated from the family
//! declarations, so a family that compiles is a family those two can step.
//! The CubeCL kernel is different: its dispatch is written by hand, because
//! `#[cube]` cannot look through a macro call. A family added without its
//! dispatch line still compiles, still runs, and quietly returns the state
//! unchanged — a flat path rather than an error.
//!
//! What closes that gap is [`Probe`], a process that is nothing but a family,
//! and [`family_name`], whose `match` carries no wildcard: a new
//! [`EulerSpec`] variant fails to compile here until it is named, which is
//! the prompt to add it to [`every_family`] as well. From there the tests
//! launch all of them and compare the two kernels point for point.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::normal::SimdNormal;

use super::*;
use crate::traits::PathSampler;

/// Points per probe path. Long enough for a wrong step to diverge visibly,
/// short enough that launching every family stays quick.
const N: usize = 64;

/// A process that is only a family: the engine's own entry point, so a family
/// can be exercised before — or without — any process reaching for it.
pub(crate) struct Probe {
  spec: EulerSpec<f32>,
  x0: f32,
}

/// The host stream for a [`Probe`]: this crate's Gaussian generator feeding
/// the family's own generated host step, which is the same expression the
/// kernels run.
pub(crate) struct ProbeSampler {
  spec: EulerSpec<f32>,
  x0: f32,
  dt: f32,
  normal: SimdNormal<f32>,
}

impl PathSampler<f32> for ProbeSampler {
  type Output = Array1<f32>;

  fn sample_into(&mut self, out: &mut Array1<f32>) {
    let slice = out.as_slice_mut().expect("probe output is contiguous");
    let (family, params) = self.spec.encode();
    let family = super::families::Family::from_code(family).expect("a declared family");
    if slice.is_empty() {
      return;
    }
    let mut state = [self.x0, 0.0, 0.0, 0.0];
    let mut out = [0.0f32; 4];
    super::families::host_report(family, &state, &params, &mut out);
    slice[0] = out[0];
    if slice.len() == 1 {
      return;
    }
    let tail = &mut slice[1..];
    self.normal.fill_slice(tail);
    for z in tail.iter_mut() {
      let mut next = [0.0f32; 4];
      super::families::host_step(family, &state, &params, self.dt, &[*z, 0.0, 0.0, 0.0], &mut next);
      state = next;
      super::families::host_report(family, &state, &params, &mut out);
      *z = out[0];
    }
  }

  fn sample(&mut self) -> Array1<f32> {
    let mut out = Array1::zeros(N);
    self.sample_into(&mut out);
    out
  }
}

impl ProcessExt<f32> for Probe {
  type Output = Array1<f32>;
  type Sampler<'s>
    = ProbeSampler
  where
    Self: 's;

  fn sampler(&self) -> ProbeSampler {
    let dt = 1.0 / (N - 1) as f32;
    ProbeSampler {
      spec: self.spec,
      x0: self.x0,
      dt,
      normal: SimdNormal::<f32>::new(0.0, dt.sqrt(), &Deterministic::new(7)),
    }
  }
}

impl EulerCoefficients<f32> for Probe {
  fn euler_spec(&self) -> EulerSpec<f32> {
    self.spec
  }

  fn initial_value(&self) -> f32 {
    self.x0
  }

  fn grid_points(&self) -> usize {
    N
  }

  fn horizon(&self) -> f32 {
    1.0
  }

  fn device_seed(&self) -> u64 {
    11
  }

  fn host_sample(&self) -> Array1<f32> {
    <Self as ProcessExt<f32>>::sampler(self).sample()
  }
}

/// The name of a family, matched without a wildcard on purpose: a new
/// [`EulerSpec`] variant fails to compile here until it is named, which is
/// the prompt to give it an entry in [`every_family`] and a dispatch line in
/// the CubeCL kernel.
fn family_name(spec: &EulerSpec<f32>) -> &'static str {
  match spec {
    EulerSpec::GeometricBrownian { .. } => "GeometricBrownian",
    EulerSpec::OrnsteinUhlenbeck { .. } => "OrnsteinUhlenbeck",
    EulerSpec::SquareRoot { .. } => "SquareRoot",
    EulerSpec::Additive => "Additive",
    EulerSpec::ReflectedSquareRoot { .. } => "ReflectedSquareRoot",
    EulerSpec::MirroredSquareRoot { .. } => "MirroredSquareRoot",
    EulerSpec::Jacobi { .. } => "Jacobi",
    EulerSpec::ConstantElasticity { .. } => "ConstantElasticity",
    EulerSpec::Ckls { .. } => "Ckls",
    EulerSpec::Logistic { .. } => "Logistic",
    EulerSpec::ThreeHalf { .. } => "ThreeHalf",
    EulerSpec::LogGeometric { .. } => "LogGeometric",
    EulerSpec::RadialOrnsteinUhlenbeck { .. } => "RadialOrnsteinUhlenbeck",
    EulerSpec::LinearSde { .. } => "LinearSde",
    EulerSpec::Hyperbolic { .. } => "Hyperbolic",
    EulerSpec::ModifiedSquareRoot { .. } => "ModifiedSquareRoot",
    EulerSpec::FellerRoot { .. } => "FellerRoot",
    EulerSpec::AitSahalia { .. } => "AitSahalia",
    EulerSpec::Gompertz { .. } => "Gompertz",
    EulerSpec::Kimura { .. } => "Kimura",
    EulerSpec::Quadratic { .. } => "Quadratic",
    EulerSpec::Pearson { .. } => "Pearson",
    EulerSpec::Verhulst { .. } => "Verhulst",
    EulerSpec::VerhulstClamped { .. } => "VerhulstClamped",
    EulerSpec::FellerLogistic { .. } => "FellerLogistic",
    EulerSpec::FellerLogisticReflected { .. } => "FellerLogisticReflected",
    EulerSpec::SquaredBesselState { .. } => "SquaredBesselState",
    EulerSpec::SquaredBesselStateReflected { .. } => "SquaredBesselStateReflected",
    EulerSpec::BesselFromSquared { .. } => "BesselFromSquared",
    EulerSpec::BesselFromSquaredReflected { .. } => "BesselFromSquaredReflected",
    EulerSpec::HyperbolicDiffusion { .. } => "HyperbolicDiffusion",
    EulerSpec::NonLinear { .. } => "NonLinear",
    EulerSpec::Displaced { .. } => "Displaced",
    EulerSpec::TanhOrnsteinUhlenbeck { .. } => "TanhOrnsteinUhlenbeck",
    EulerSpec::BoundedCorrelation { .. } => "BoundedCorrelation",
  }
}

/// One probe per family, parametrised so the recursion stays in a regime
/// where a step that is merely wrong looks different from a step that is
/// right: away from a boundary the family would clamp to anyway, and with a
/// diffusion large enough to move the state over 64 points.
fn every_family() -> Vec<Probe> {
  let p = |spec, x0| Probe { spec, x0 };
  vec![
    p(EulerSpec::GeometricBrownian { mu: 0.05, sigma: 0.2 }, 100.0),
    p(EulerSpec::OrnsteinUhlenbeck { theta: 2.0, mu: 0.05, sigma: 0.1 }, 0.03),
    p(EulerSpec::SquareRoot { kappa: 2.0, theta: 0.04, sigma: 0.2 }, 0.04),
    p(EulerSpec::Additive, 0.0),
    p(EulerSpec::ReflectedSquareRoot { theta: 2.0, mu: 0.04, sigma: 0.2 }, 0.04),
    p(EulerSpec::MirroredSquareRoot { theta: 2.0, mu: 0.04, sigma: 0.2 }, 0.04),
    p(EulerSpec::Jacobi { alpha: 0.3, beta: 0.6, sigma: 0.2 }, 0.5),
    p(EulerSpec::ConstantElasticity { mu: 0.05, sigma: 0.2, gamma: 0.8 }, 100.0),
    p(
      EulerSpec::Ckls { theta1: 0.06, theta2: -1.5, theta3: 0.3, theta4: 0.5 },
      0.04,
    ),
    p(EulerSpec::Logistic { a: 0.5, b: 0.2 }, 1.0),
    p(EulerSpec::ThreeHalf { kappa: 2.0, mu: 0.04, sigma: 0.3 }, 0.04),
    p(EulerSpec::LogGeometric { drift_ln: 0.0001, sigma: 0.2 }, 100.0),
    p(EulerSpec::RadialOrnsteinUhlenbeck { kappa: 1.0, sigma: 0.3 }, 1.0),
    p(EulerSpec::LinearSde { a: 0.02, b: 0.3, c: 0.2 }, 1.0),
    p(EulerSpec::Hyperbolic { kappa: 1.0, sigma: 0.3 }, 0.5),
    p(EulerSpec::ModifiedSquareRoot { kappa: 1.0, sigma: 0.2 }, 0.5),
    p(EulerSpec::FellerRoot { theta1: 0.5, decay: -0.142, theta3: 0.2 }, 0.5),
    p(
      EulerSpec::AitSahalia {
        am1: 0.0001,
        a0: 0.15,
        a1: -3.0,
        a2: 0.0,
        b0: 0.0004,
        b1: 0.0,
        b2: 0.05,
        b3: 1.5,
      },
      0.05,
    ),
    p(EulerSpec::Gompertz { a: 0.5, b: 0.3, sigma: 0.2 }, 1.0),
    p(EulerSpec::Kimura { a: 0.5, sigma: 0.2 }, 0.5),
    p(
      EulerSpec::Quadratic { alpha: 0.02, beta: 0.1, gamma: -0.05, sigma: 0.2 },
      1.0,
    ),
    p(
      EulerSpec::Pearson { kappa: 3.0, mu: 0.05, a: 0.0, b: 0.0, c: 0.01, two_kappa: 6.0 },
      0.05,
    ),
    p(EulerSpec::Verhulst { r: 1.0, k: 2.0, sigma: 0.3 }, 0.5),
    p(EulerSpec::VerhulstClamped { r: 1.0, k: 2.0, sigma: 0.3 }, 0.5),
    p(EulerSpec::FellerLogistic { kappa: 1.0, theta: 1.0, sigma: 0.3 }, 0.5),
    p(
      EulerSpec::FellerLogisticReflected { kappa: 1.0, theta: 1.0, sigma: 0.3 },
      0.5,
    ),
    p(EulerSpec::SquaredBesselState { delta: 3.0, two: 2.0 }, 1.0),
    p(EulerSpec::SquaredBesselStateReflected { delta: 3.0, two: 2.0 }, 1.0),
    p(EulerSpec::BesselFromSquared { delta: 3.0, two: 2.0 }, 1.0),
    p(EulerSpec::BesselFromSquaredReflected { delta: 3.0, two: 2.0 }, 1.0),
    p(
      EulerSpec::HyperbolicDiffusion {
        beta: 0.5,
        gamma: 1.0,
        delta: 1.0,
        mu: 0.0,
        sigma: 0.3,
        half_var: 0.045,
      },
      0.5,
    ),
    p(
      EulerSpec::NonLinear {
        am1: 0.0001,
        a0: 0.15,
        a1: -3.0,
        a2: 0.0,
        b0: 0.0,
        b1: 0.0,
        b2: 0.2,
        b3: 1.0,
      },
      0.05,
    ),
    p(EulerSpec::Displaced { mu: 0.05, sigma: 0.2, beta: 20.0 }, 120.0),
    p(EulerSpec::TanhOrnsteinUhlenbeck { kappa: 1.0, mu: 0.3, sigma: 0.2 }, 0.4),
    p(EulerSpec::BoundedCorrelation { kappa: 1.0, mu: 0.3, sigma: 0.2 }, 0.2),
  ]
}

/// Every declared family has a probe, so the parity tests below cover all of
/// them rather than whichever ones happened to be listed.
#[test]
fn every_family_has_a_probe() {
  let probes = every_family();
  let mut names: Vec<_> = probes.iter().map(|p| family_name(&p.spec)).collect();
  names.sort_unstable();
  names.dedup();
  assert_eq!(
    names.len(),
    probes.len(),
    "two probes name the same family: {names:?}"
  );
  for probe in &probes {
    let (code, _) = probe.spec.encode();
    assert!(
      super::families::Family::from_code(code).is_some(),
      "{} encodes to an undeclared code {code}",
      family_name(&probe.spec)
    );
  }
}

/// The kernels the native back-ends run are generated, so what this checks is
/// that each generated body compiles into a launchable kernel and produces a
/// path that stays in the reals.
#[cfg(any(feature = "metal", feature = "cuda"))]
#[test]
fn every_family_runs_on_the_device() {
  #[cfg(feature = "cuda")]
  type Device = crate::device::Cuda;
  #[cfg(all(feature = "metal", not(feature = "cuda")))]
  type Device = crate::device::Metal;

  for probe in every_family() {
    let name = family_name(&probe.spec);
    let paths = Device::default().euler_paths(&probe, 8);
    assert_eq!(paths.len(), 8, "{name}");
    assert_eq!(paths[0].len(), N, "{name}");
    assert!(
      paths.iter().all(|p| p.iter().all(|v| v.is_finite())),
      "{name}: a device path left the reals"
    );
  }
}

/// The CubeCL dispatch is written by hand, so a family missing from it
/// returns the state unchanged. Comparing against the generated Metal kernel
/// point for point is what turns that silence into a failure.
#[cfg(all(
  feature = "metal",
  any(feature = "cubecl-cuda", feature = "cubecl-wgpu")
))]
#[test]
fn the_cubecl_kernel_matches_the_generated_one() {
  #[cfg(feature = "cubecl-wgpu")]
  type Cube = crate::device::Cubecl<crate::device::WgpuRuntime>;
  #[cfg(all(feature = "cubecl-cuda", not(feature = "cubecl-wgpu")))]
  type Cube = crate::device::Cubecl<crate::device::CudaRuntime>;

  for probe in every_family() {
    let name = family_name(&probe.spec);
    let native = crate::device::Metal::default().euler_paths(&probe, 8);
    let cube = Cube::default().euler_paths(&probe, 8);
    for (a, b) in native.iter().zip(&cube) {
      for (x, y) in a.iter().zip(b.iter()) {
        assert!(
          (x - y).abs() < 1e-3 * y.abs().max(1.0),
          "{name}: native {x} vs cubecl {y}"
        );
      }
    }
  }
}

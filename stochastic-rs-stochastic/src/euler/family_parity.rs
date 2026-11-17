//! Every family, launched on a device without a process that happens to use
//! it.
//!
//! The kernels the CUDA and Metal back-ends run are generated from the family
//! declarations, so a family that compiles is a family those two can step;
//! what is left to check is that the declaration means on the device what it
//! means on the host.
//!
//! What closes that gap is [`Probe`], a process that is nothing but a family,
//! and [`family_name`], whose `match` carries no wildcard: a new
//! [`EulerSpec`] variant fails to compile here until it is named, which is
//! the prompt to add it to [`every_family`] as well. From there the tests
//! launch all of them and compare the two kernels point for point.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::traits::Expr;
use stochastic_rs_distributions::traits::Program;

use super::*;

#[cfg(metal_backend)]
mod migration;
mod one_component;
mod probe;
mod system_probe;
mod three_four_component;
mod two_component;

use one_component::every_family;
#[cfg(metal_backend)]
use system_probe::SystemProbeSampler;
use three_four_component::every_four_component_family;
use three_four_component::every_three_component_family;
use two_component::every_two_component_family;

/// Points per probe path. Long enough for a wrong step to diverge visibly,
/// short enough that launching every family stays quick.
const N: usize = 64;

/// A process that is only a family: the engine's own entry point, so a family
/// can be exercised before — or without — any process reaching for it.
pub(crate) struct Probe {
  spec: EulerSpec<f32>,
  x0: f32,
  /// The Markov lift a lifted family steps under, `None` for the rest.
  lift: Option<ProbeLift>,
  /// The programs a family reading `pv` / `pv2` runs, `None` for the rest.
  program: Option<ProbePrograms>,
}

/// The two programs the probes hand a family that reads `pv` and `pv2`:
/// compiled expressions of the step's `ct` and first state slot.
#[derive(Clone)]
pub(crate) struct ProbePrograms {
  first: Program,
  second: Option<Program>,
}

impl ProbePrograms {
  /// A drift and a diffusion of time and state with every node kind in use,
  /// so the kernels' interpreters are checked opcode by opcode.
  fn pair() -> Self {
    Self {
      first: Program::compile(&(Expr::x() * -0.5 + Expr::t() * 0.1 - Expr::lit(0.02)))
        .expect("the probe program fits the kernel bounds"),
      second: Some(
        Program::compile(
          &((Expr::lit(0.3) + Expr::x().abs() * 0.2)
            .min(Expr::lit(0.6))
            .max(0.05)
            * (-Expr::x() * Expr::x()).exp()
            + Expr::x().powf(2.0).sqrt().tanh() / 4.0
            + (Expr::lit(1.0) + Expr::t()).ln() * 0.1),
        )
        .expect("the probe program fits the kernel bounds"),
      ),
    }
  }

  fn spec(&self) -> crate::euler::ProgramSpec<'_> {
    crate::euler::ProgramSpec {
      first: &self.first,
      second: self.second.as_ref(),
    }
  }

  /// The two values at `(t, x)`, `pv2` zero without a second program.
  fn values(&self, t: f32, x: f32) -> (f32, f32) {
    (
      self.first.eval(t, x),
      self.second.as_ref().map_or(0.0, |s| s.eval(t, x)),
    )
  }
}

/// A small Markov lift for the probes: a Riemann-Liouville kernel at
/// `H = 0.3` on twelve nodes, its per-node constants and boundary terms taken
/// from the same `VolterraLift` the host samplers step with.
#[derive(Clone)]
pub(crate) struct ProbeLift {
  decay: Vec<f32>,
  weight: Vec<f32>,
  drift_scale: Vec<f32>,
  db: f32,
  fb: f32,
  x0: f32,
}

fn probe_lift(dt: f32) -> ProbeLift {
  let lift =
    crate::volterra::lift::VolterraLift::new(crate::rough::RlKernel::<f32>::new(0.3, 12), dt);
  ProbeLift {
    decay: lift.exp_neg_x_dt.to_vec(),
    weight: lift.we.to_vec(),
    drift_scale: lift.one_minus_e_over_x.to_vec(),
    db: lift.drift_boundary,
    fb: lift.diffusion_boundary,
    x0: 0.0,
  }
}

/// The curve every probe carries: a family that reads none ignores it, and
/// one that does gets a value that varies along the path, so a kernel binding
/// the wrong step would show up as a different law.
/// The jump intensity every probe declares: enough that a step sees a jump
/// now and then, low enough that the count stays small.
/// The series terms every series probe draws per path: a few per cell on
/// the probes' short grid, so a cell that gets none is the exception.
const PROBE_TERMS: u32 = 64;

/// The table points every table probe builds per path.
const PROBE_TABLE: u32 = 64;

const PROBE_INTENSITY: f32 = 3.0;

/// The size law every probe declares. Double-exponential rather than normal
/// because it is the one the kernel sums in a loop, so the loop runs on both
/// kernels whether or not the family under test reads the sum.
/// The Gamma draws every probe declares. Two of them, with the first shape
/// below one so the boost branch runs and the second above it so the plain
/// rejection loop does.
const PROBE_GAMMAS: GammaDraws<f32> = GammaDraws {
  first: (0.4, 1.5, 0.0),
  second: Some((2.5, 0.8, 0.0)),
};

const PROBE_SIZES: JumpSizes<f32> = JumpSizes::DoubleExponential {
  p_up: 0.4,
  eta_up: 25.0,
  eta_down: 20.0,
};

/// The size law a probe's jump sum is drawn under: the shared double-exponential unless the family
/// is held to another law, so every `js` branch of the kernels has a family that reaches it.
fn probe_sizes(spec: &EulerSpec<f32>) -> JumpSizes<f32> {
  match spec {
    EulerSpec::KouJumpHestonReflected { .. } => JumpSizes::Exponential { rate: 20.0 },
    EulerSpec::BatesJumpReflected { .. } => JumpSizes::ExponentialProduct { rate: 20.0 },
    EulerSpec::FractionalBatesMemory { .. } => JumpSizes::OneExponential { rate: 20.0 },
    EulerSpec::TemperedStableSubordinator { .. } => JumpSizes::TemperedStable {
      eps: 0.05,
      neg_inv_alpha: -1.0 / 0.7,
      mu: 1.0,
    },
    EulerSpec::Bates1996 { .. } => JumpSizes::NormalProduct {
      mean: -0.02,
      sd: 0.05,
    },
    EulerSpec::Bates1996Reflected { .. } => JumpSizes::DoubleExponentialProduct {
      p_up: 0.4,
      eta_up: 20.0,
      eta_down: 25.0,
    },
    EulerSpec::CompoundPoissonEvents { .. } => JumpSizes::OneDoubleExponential {
      p_up: 0.4,
      eta_up: 20.0,
      eta_down: 25.0,
    },
    EulerSpec::JumpFractionalOu { .. } => JumpSizes::OneNormal {
      mean: 0.01,
      sd: 0.05,
    },
    // The aggregated normal is the production law of most jump models, so
    // one probe holds it too rather than leaving the busiest branch unchecked.
    EulerSpec::AdditiveJumpDiffusion { .. } => JumpSizes::Normal {
      mean: -0.02,
      sd: 0.1,
    },
    // Two more families that read the sum, so the Rademacher and the stable
    // laws are compared kernel to kernel rather than merely executed.
    EulerSpec::MertonJumpLog { .. } => JumpSizes::SymmetricStable {
      alpha: 1.7,
      scale: 0.02,
    },
    EulerSpec::DuffieKanJump { .. } => JumpSizes::Rademacher { scale: 0.02 },
    _ => PROBE_SIZES,
  }
}

/// One ramp per curve slot, each with its own level and slope, so every slot
/// the kernels bind carries a value no other slot carries and a family that
/// reads `ct3` is checked against the host reading `ct3` and nothing else.
/// The series sum a step reads: the filed cell for a preamble-sized family,
/// the terms of the cell sized against the current state for a live one.
fn cell_sum(
  family: super::families::Family,
  state: &[f32],
  params: &[f32],
  dt: f32,
  terms: &[(f32, f32, f32, f32, usize)],
  series: &[f32],
  point: usize,
) -> f32 {
  if family.series_live() {
    terms
      .iter()
      .filter(|term| term.4 == point)
      .map(|&(gj, ej, uj, uv, _)| {
        super::families::host_series(family, state, params, dt, gj, ej, uj, uv)
          .expect("a family with a series clause sizes its terms")
      })
      .sum()
  } else {
    series[point]
  }
}

/// One series term's gap, exponential, two uniforms and arrival time, drawn for path 0 under
/// launch seed 11 from the Philox port exactly as the frame's preamble draws them.
fn series_draws(term: u32) -> (f32, f32, f32, f32, f32) {
  use super::philox::{
    SITE_SERIES, SITE_SERIES_MARKS, SITE_SERIES_TIME, key, philox4x32_10, u01_f32,
  };
  use super::tests::philox::exp1_f32;

  let words = |site| philox4x32_10([0, 0, site, term], key(11));
  let (draws, marks) = (words(SITE_SERIES), words(SITE_SERIES_MARKS));
  (
    exp1_f32(draws[0], draws[1]),
    exp1_f32(draws[2], draws[3]),
    u01_f32(marks[0]),
    u01_f32(marks[2]),
    u01_f32(words(SITE_SERIES_TIME)[0]),
  )
}

fn probe_curves() -> Vec<Vec<f32>> {
  (0..crate::euler::CURVE_SLOTS)
    .map(|k| {
      let (level, slope) = (0.02 + 0.003 * k as f32, 0.001 + 0.0002 * k as f32);
      (0..N).map(|i| level + slope * i as f32).collect()
    })
    .collect()
}

/// The name of a family, matched without a wildcard on purpose: a new
/// [`EulerSpec`] variant fails to compile here until it is named, which is
/// the prompt to give it an entry in [`every_family`].
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
    EulerSpec::Heston { .. } => "Heston",
    EulerSpec::HestonReflected { .. } => "HestonReflected",
    EulerSpec::Sabr { .. } => "Sabr",
    EulerSpec::Bergomi { .. } => "Bergomi",
    EulerSpec::TwoScaleOrnsteinUhlenbeck { .. } => "TwoScaleOrnsteinUhlenbeck",
    EulerSpec::LogHeston { .. } => "LogHeston",
    EulerSpec::LogHestonReflected { .. } => "LogHestonReflected",
    EulerSpec::DoubleHeston { .. } => "DoubleHeston",
    EulerSpec::DoubleHestonReflected { .. } => "DoubleHestonReflected",
    EulerSpec::StochasticCorrelationHeston { .. } => "StochasticCorrelationHeston",
    EulerSpec::HullWhite { .. } => "HullWhite",
    EulerSpec::CurveDrift { .. } => "CurveDrift",
    EulerSpec::LogMeanReverting { .. } => "LogMeanReverting",
    EulerSpec::ShiftedSquareRoot { .. } => "ShiftedSquareRoot",
    EulerSpec::ShiftedSquareRootMirrored { .. } => "ShiftedSquareRootMirrored",
    EulerSpec::TimeVaryingGeometricBrownian { .. } => "TimeVaryingGeometricBrownian",
    EulerSpec::CorrelatedBrownian { .. } => "CorrelatedBrownian",
    EulerSpec::BrownianBridge { .. } => "BrownianBridge",
    EulerSpec::TwoFactorHullWhite { .. } => "TwoFactorHullWhite",
    EulerSpec::TwoFactorSquareRoot { .. } => "TwoFactorSquareRoot",
    EulerSpec::DuffieKan { .. } => "DuffieKan",
    EulerSpec::TwoAssetHeston { .. } => "TwoAssetHeston",
    EulerSpec::TwoAssetHestonReflected { .. } => "TwoAssetHestonReflected",
    EulerSpec::MertonJumpLog { .. } => "MertonJumpLog",
    EulerSpec::BatesJump { .. } => "BatesJump",
    EulerSpec::BatesJumpReflected { .. } => "BatesJumpReflected",
    EulerSpec::AndersenQe { .. } => "AndersenQe",
    EulerSpec::CountingProcess => "CountingProcess",
    EulerSpec::InverseGaussianSubordinator { .. } => "InverseGaussianSubordinator",
    EulerSpec::NormalInverseGaussian { .. } => "NormalInverseGaussian",
    EulerSpec::StableSubordinator { .. } => "StableSubordinator",
    EulerSpec::KouJumpHeston { .. } => "KouJumpHeston",
    EulerSpec::KouJumpHestonReflected { .. } => "KouJumpHestonReflected",
    EulerSpec::DuffieKanJump { .. } => "DuffieKanJump",
    EulerSpec::HawkesJumpDiffusion { .. } => "HawkesJumpDiffusion",
    EulerSpec::Garch { .. } => "Garch",
    EulerSpec::ThresholdGarch { .. } => "ThresholdGarch",
    EulerSpec::ExponentialGarch { .. } => "ExponentialGarch",
    EulerSpec::Innovation { .. } => "Innovation",
    EulerSpec::CorrelatedInnovation { .. } => "CorrelatedInnovation",
    EulerSpec::Autoregressive { .. } => "Autoregressive",
    EulerSpec::MovingAverage { .. } => "MovingAverage",
    EulerSpec::GammaSubordinator => "GammaSubordinator",
    EulerSpec::VarianceGamma { .. } => "VarianceGamma",
    EulerSpec::BilateralGamma => "BilateralGamma",
    EulerSpec::BilateralGammaMotion { .. } => "BilateralGammaMotion",
    EulerSpec::TemperedStableSubordinator { .. } => "TemperedStableSubordinator",
    EulerSpec::BarndorffNielsenShephard { .. } => "BarndorffNielsenShephard",
    EulerSpec::CorrelatedFractionalMotion { .. } => "CorrelatedFractionalMotion",
    EulerSpec::ComplexFractionalOu { .. } => "ComplexFractionalOu",
    EulerSpec::TransformedOrnsteinUhlenbeck { .. } => "TransformedOrnsteinUhlenbeck",
    EulerSpec::PoissonArrivals { .. } => "PoissonArrivals",
    EulerSpec::DynamicSabr => "DynamicSabr",
    EulerSpec::HeathJarrowMorton => "HeathJarrowMorton",
    EulerSpec::AffineDiffusionGaussian { .. } => "AffineDiffusionGaussian",
    EulerSpec::WuZhang { .. } => "WuZhang",
    EulerSpec::CorrelatedGeometric4 { .. } => "CorrelatedGeometric4",
    EulerSpec::CorrelatedNoises4 { .. } => "CorrelatedNoises4",
    EulerSpec::RegimeSwitching { .. } => "RegimeSwitching",
    EulerSpec::RiemannLiouville => "RiemannLiouville",
    EulerSpec::RiemannLiouvilleOu { .. } => "RiemannLiouvilleOu",
    EulerSpec::RiemannLiouvilleBlackScholes { .. } => "RiemannLiouvilleBlackScholes",
    EulerSpec::RiemannLiouvilleHeston { .. } => "RiemannLiouvilleHeston",
    EulerSpec::AdditiveJumpDiffusion { .. } => "AdditiveJumpDiffusion",
    EulerSpec::Bates1996 { .. } => "Bates1996",
    EulerSpec::Bates1996Reflected { .. } => "Bates1996Reflected",
    EulerSpec::CompoundPoissonEvents { .. } => "CompoundPoissonEvents",
    EulerSpec::JumpFractionalOu { .. } => "JumpFractionalOu",
    EulerSpec::VolterraSquareRoot { .. } => "VolterraSquareRoot",
    EulerSpec::GaussianPolynomialVolatility { .. } => "GaussianPolynomialVolatility",
    EulerSpec::RoughHestonMemory { .. } => "RoughHestonMemory",
    EulerSpec::FractionalBatesMemory { .. } => "FractionalBatesMemory",
    EulerSpec::RoughBergomiMemory { .. } => "RoughBergomiMemory",
    EulerSpec::LinearFractionalStable { .. } => "LinearFractionalStable",
    EulerSpec::MovingAverageFilter { .. } => "MovingAverageFilter",
    EulerSpec::TemperedStableSeries { .. } => "TemperedStableSeries",
    EulerSpec::VolterraReference => "VolterraReference",
    EulerSpec::HawkesEvents { .. } => "HawkesEvents",
    EulerSpec::HawkesEvents2 { .. } => "HawkesEvents2",
    EulerSpec::CheyetteLocalVol { .. } => "CheyetteLocalVol",
    EulerSpec::HestonSlv { .. } => "HestonSlv",
    EulerSpec::VolterraProgram => "VolterraProgram",
    EulerSpec::LiborMarket4 { .. } => "LiborMarket4",
    EulerSpec::InverseStableSubordinator { .. } => "InverseStableSubordinator",
    EulerSpec::StochasticVolatilityCgmy { .. } => "StochasticVolatilityCgmy",
    EulerSpec::WishartTwo { .. } => "WishartTwo",
  }
}

/// A system that is only a family: what [`Probe`] is for a one-component
/// family, for one with `D` of them.
pub(crate) struct SystemProbe<const D: usize> {
  spec: EulerSpec<f32>,
  x0: [f32; D],
  /// The Markov lift a lifted family steps under, `None` for the rest.
  lift: Option<ProbeLift>,
  /// The programs a family reading `pv` / `pv2` runs, `None` for the rest.
  program: Option<ProbePrograms>,
}

/// Every declared family has a probe of the right arity, so the parity tests
/// below cover all of them rather than whichever ones happened to be listed.
#[test]
fn every_family_has_a_probe() {
  let mut names: Vec<&'static str> = every_family()
    .iter()
    .map(|p| family_name(&p.spec))
    .collect();
  names.extend(
    every_two_component_family()
      .iter()
      .map(|p| family_name(&p.spec)),
  );
  names.extend(
    every_three_component_family()
      .iter()
      .map(|p| family_name(&p.spec)),
  );
  names.extend(
    every_four_component_family()
      .iter()
      .map(|p| family_name(&p.spec)),
  );
  let total = names.len();
  names.sort_unstable();
  names.dedup();
  assert_eq!(names.len(), total, "two probes name the same family");

  // Codes are dense and small, so walking the space is how the lists above
  // are held to the declarations rather than to whoever last edited them.
  let declared = (0..256u32)
    .filter(|c| super::families::Family::from_code(*c).is_some())
    .count();
  assert_eq!(
    declared, total,
    "{declared} families are declared but {total} have probes"
  );

  let arity = |spec: &EulerSpec<f32>| {
    let (code, _) = spec.encode();
    super::families::Family::from_code(code)
      .expect("a declared family")
      .components()
  };
  for probe in &every_family() {
    assert_eq!(arity(&probe.spec), 1, "{}", family_name(&probe.spec));
  }
  for probe in &every_two_component_family() {
    assert_eq!(arity(&probe.spec), 2, "{}", family_name(&probe.spec));
  }
  for probe in &every_three_component_family() {
    assert_eq!(arity(&probe.spec), 3, "{}", family_name(&probe.spec));
  }
  for probe in &every_four_component_family() {
    assert_eq!(arity(&probe.spec), 4, "{}", family_name(&probe.spec));
  }
}

/// The kernels the native back-ends run are generated, so what this checks is
/// that each generated body compiles into a launchable kernel and produces a
/// path that stays in the reals.
#[cfg(device_backend)]
#[test]
fn every_family_runs_on_the_device() {
  #[cfg(feature = "cuda")]
  type Device = crate::device::Cuda;
  #[cfg(all(metal_backend, not(feature = "cuda")))]
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
  for probe in every_two_component_family() {
    let name = family_name(&probe.spec);
    let paths = Device::default().system_paths(&probe, 8);
    assert_eq!(paths.len(), 8, "{name}");
    assert!(
      paths.iter().all(|c| c
        .iter()
        .all(|p| p.len() == N && p.iter().all(|v| v.is_finite()))),
      "{name}: a device path left the reals"
    );
  }
  for probe in every_three_component_family() {
    let name = family_name(&probe.spec);
    let paths = Device::default().system_paths(&probe, 8);
    assert_eq!(paths.len(), 8, "{name}");
    assert!(
      paths.iter().all(|c| c
        .iter()
        .all(|p| p.len() == N && p.iter().all(|v| v.is_finite()))),
      "{name}: a device path left the reals"
    );
  }
  for probe in every_four_component_family() {
    let name = family_name(&probe.spec);
    let paths = Device::default().system_paths(&probe, 8);
    assert_eq!(paths.len(), 8, "{name}");
    assert!(
      paths.iter().all(|c| c
        .iter()
        .all(|p| p.len() == N && p.iter().all(|v| v.is_finite()))),
      "{name}: a device path left the reals"
    );
  }
}

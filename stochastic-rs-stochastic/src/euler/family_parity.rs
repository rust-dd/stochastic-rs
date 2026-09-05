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
  /// The Markov lift a lifted family steps under, `None` for the rest.
  lift: Option<ProbeLift>,
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

/// One ramp per curve slot, each with its own level and slope, so every slot
/// the kernels bind carries a value no other slot carries and a family that
/// reads `ct3` is checked against the host reading `ct3` and nothing else.
fn probe_curves() -> Vec<Vec<f32>> {
  (0..crate::euler::CURVE_SLOTS)
    .map(|k| {
      let (level, slope) = (0.02 + 0.003 * k as f32, 0.001 + 0.0002 * k as f32);
      (0..N).map(|i| level + slope * i as f32).collect()
    })
    .collect()
}

/// The host stream for a [`Probe`]: this crate's Gaussian generator feeding
/// the family's own generated host step, which is the same expression the
/// kernels run.
pub(crate) struct ProbeSampler {
  spec: EulerSpec<f32>,
  x0: f32,
  dt: f32,
  normal: SimdNormal<f32>,
  lift: Option<ProbeLift>,
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
    let curves = probe_curves();
    let mut state = [self.x0, 0.0, 0.0, 0.0];
    let mut out = [0.0f32; 4];
    super::families::host_report(
      family,
      &state,
      &params,
      curves[0][0],
      curves[1][0],
      curves[2][0],
      curves[3][0],
      curves[4][0],
      curves[5][0],
      curves[6][0],
      curves[7][0],
      0.0,
      0.0,
      0.7,
      1.3,
      0.5,
      0.5,
      0.0,
      &mut out,
    );
    slice[0] = out[0];
    if slice.len() == 1 {
      return;
    }
    let tail = &mut slice[1..];
    self.normal.fill_slice(tail);
    // The lift's two state vectors, advanced per step exactly as the frame
    // advances them, from the family's own drift, diffusion and shock.
    let nodes = self.lift.as_ref().map_or(0, |l| l.decay.len());
    let (mut lh, mut lj) = (vec![0.0f32; nodes], vec![0.0f32; nodes]);
    for (i, z) in tail.iter_mut().enumerate() {
      let noise = [*z, 0.0, 0.0, 0.0];
      let (mut lv, mut coefficients) = (0.0f32, [0.0f32; 3]);
      if let Some(lift) = &self.lift {
        coefficients = super::families::host_lift(family, &state, &params, self.dt, &noise);
        let [lf, lg, lsh] = coefficients;
        let hist: f32 = (0..nodes).map(|l| lift.weight[l] * (lh[l] + lj[l])).sum();
        lv = lift.x0 + lift.db * lf + hist + lift.fb * lg * lsh;
      }
      let mut next = [0.0f32; 4];
      super::families::host_step(
        family,
        &state,
        &params,
        self.dt,
        curves[0][i + 1],
        curves[1][i + 1],
        curves[2][i + 1],
        curves[3][i + 1],
        curves[4][i + 1],
        curves[5][i + 1],
        curves[6][i + 1],
        curves[7][i + 1],
        0.0,
        0.0,
        0.7,
        1.3,
        0.5,
        0.5,
        lv,
        &noise,
        &mut next,
      );
      state = next;
      if let Some(lift) = &self.lift {
        let [lf, lg, lsh] = coefficients;
        for l in 0..nodes {
          lh[l] = lift.decay[l] * lh[l] + lift.drift_scale[l] * lf;
          lj[l] = lift.decay[l] * (lj[l] + lg * lsh);
        }
      }
      super::families::host_report(
        family,
        &state,
        &params,
        curves[0][i + 1],
        curves[1][i + 1],
        curves[2][i + 1],
        curves[3][i + 1],
        curves[4][i + 1],
        curves[5][i + 1],
        curves[6][i + 1],
        curves[7][i + 1],
        0.0,
        0.0,
        0.7,
        1.3,
        0.5,
        0.5,
        0.0,
        &mut out,
      );
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
      lift: self.lift.clone(),
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

  fn curves(&self) -> Option<Vec<Vec<f32>>> {
    Some(probe_curves())
  }

  fn lift_spec(&self) -> Option<crate::euler::LiftSpec<'_, f32>> {
    self.lift.as_ref().map(|l| crate::euler::LiftSpec {
      decay: &l.decay,
      weight: &l.weight,
      drift_scale: &l.drift_scale,
      drift_boundary: l.db,
      diffusion_boundary: l.fb,
      x0: l.x0,
    })
  }

  /// Every probe takes jumps, so the count's own hash stream runs on both
  /// kernels whether or not the family under test reads it.
  fn jump_intensity(&self) -> Option<f32> {
    Some(PROBE_INTENSITY)
  }

  fn jump_sizes(&self) -> Option<JumpSizes<f32>> {
    Some(PROBE_SIZES)
  }

  fn gamma_draws(&self) -> Option<GammaDraws<f32>> {
    Some(PROBE_GAMMAS)
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
  }
}

/// One probe per family, parametrised so the recursion stays in a regime
/// where a step that is merely wrong looks different from a step that is
/// right: away from a boundary the family would clamp to anyway, and with a
/// diffusion large enough to move the state over 64 points.
fn every_family() -> Vec<Probe> {
  let p = |spec, x0| Probe {
    spec,
    x0,
    lift: None,
  };
  // A lifted family runs under the probes' small Riemann-Liouville lift.
  let p_lift = |spec, x0| Probe {
    spec,
    x0,
    lift: Some(probe_lift(1.0 / (N - 1) as f32)),
  };
  vec![
    p(
      EulerSpec::GeometricBrownian {
        mu: 0.05,
        sigma: 0.2,
      },
      100.0,
    ),
    p(
      EulerSpec::OrnsteinUhlenbeck {
        theta: 2.0,
        mu: 0.05,
        sigma: 0.1,
      },
      0.03,
    ),
    p(
      EulerSpec::SquareRoot {
        kappa: 2.0,
        theta: 0.04,
        sigma: 0.2,
      },
      0.04,
    ),
    p(EulerSpec::Additive, 0.0),
    p(
      EulerSpec::ReflectedSquareRoot {
        theta: 2.0,
        mu: 0.04,
        sigma: 0.2,
      },
      0.04,
    ),
    p(
      EulerSpec::MirroredSquareRoot {
        theta: 2.0,
        mu: 0.04,
        sigma: 0.2,
      },
      0.04,
    ),
    p(
      EulerSpec::Jacobi {
        alpha: 0.3,
        beta: 0.6,
        sigma: 0.2,
      },
      0.5,
    ),
    p(
      EulerSpec::ConstantElasticity {
        mu: 0.05,
        sigma: 0.2,
        gamma: 0.8,
      },
      100.0,
    ),
    p(
      EulerSpec::Ckls {
        theta1: 0.06,
        theta2: -1.5,
        theta3: 0.3,
        theta4: 0.5,
      },
      0.04,
    ),
    p(EulerSpec::Logistic { a: 0.5, b: 0.2 }, 1.0),
    p(
      EulerSpec::ThreeHalf {
        kappa: 2.0,
        mu: 0.04,
        sigma: 0.3,
      },
      0.04,
    ),
    p(
      EulerSpec::LogGeometric {
        drift_ln: 0.0001,
        sigma: 0.2,
      },
      100.0,
    ),
    p(
      EulerSpec::RadialOrnsteinUhlenbeck {
        kappa: 1.0,
        sigma: 0.3,
      },
      1.0,
    ),
    p(
      EulerSpec::LinearSde {
        a: 0.02,
        b: 0.3,
        c: 0.2,
      },
      1.0,
    ),
    p(
      EulerSpec::Hyperbolic {
        kappa: 1.0,
        sigma: 0.3,
      },
      0.5,
    ),
    p(
      EulerSpec::ModifiedSquareRoot {
        kappa: 1.0,
        sigma: 0.2,
      },
      0.5,
    ),
    p(
      EulerSpec::FellerRoot {
        theta1: 0.5,
        decay: -0.142,
        theta3: 0.2,
      },
      0.5,
    ),
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
    p(
      EulerSpec::Gompertz {
        a: 0.5,
        b: 0.3,
        sigma: 0.2,
      },
      1.0,
    ),
    p(EulerSpec::Kimura { a: 0.5, sigma: 0.2 }, 0.5),
    p(
      EulerSpec::Quadratic {
        alpha: 0.02,
        beta: 0.1,
        gamma: -0.05,
        sigma: 0.2,
      },
      1.0,
    ),
    p(
      EulerSpec::Pearson {
        kappa: 3.0,
        mu: 0.05,
        a: 0.0,
        b: 0.0,
        c: 0.01,
        two_kappa: 6.0,
      },
      0.05,
    ),
    p(
      EulerSpec::Verhulst {
        r: 1.0,
        k: 2.0,
        sigma: 0.3,
      },
      0.5,
    ),
    p(
      EulerSpec::VerhulstClamped {
        r: 1.0,
        k: 2.0,
        sigma: 0.3,
      },
      0.5,
    ),
    p(
      EulerSpec::FellerLogistic {
        kappa: 1.0,
        theta: 1.0,
        sigma: 0.3,
      },
      0.5,
    ),
    p(
      EulerSpec::FellerLogisticReflected {
        kappa: 1.0,
        theta: 1.0,
        sigma: 0.3,
      },
      0.5,
    ),
    p(
      EulerSpec::SquaredBesselState {
        delta: 3.0,
        two: 2.0,
      },
      1.0,
    ),
    p(
      EulerSpec::SquaredBesselStateReflected {
        delta: 3.0,
        two: 2.0,
      },
      1.0,
    ),
    p(
      EulerSpec::BesselFromSquared {
        delta: 3.0,
        two: 2.0,
      },
      1.0,
    ),
    p(
      EulerSpec::BesselFromSquaredReflected {
        delta: 3.0,
        two: 2.0,
      },
      1.0,
    ),
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
    p(
      EulerSpec::Displaced {
        mu: 0.05,
        sigma: 0.2,
        beta: 20.0,
      },
      120.0,
    ),
    p(
      EulerSpec::TanhOrnsteinUhlenbeck {
        kappa: 1.0,
        mu: 0.3,
        sigma: 0.2,
      },
      0.4,
    ),
    p(EulerSpec::PoissonArrivals { lambda: 2.5 }, 0.0),
    p_lift(EulerSpec::RiemannLiouville, 0.0),
    p(EulerSpec::AffineDiffusionGaussian { sigma: 0.02 }, 0.03),
    p(
      EulerSpec::TransformedOrnsteinUhlenbeck {
        kappa: 1.0,
        mu: 0.3,
        sigma: 0.2,
        arctan: 1.0,
        half_pi: std::f32::consts::FRAC_PI_2,
      },
      0.4,
    ),
    p(
      EulerSpec::BoundedCorrelation {
        kappa: 1.0,
        mu: 0.3,
        sigma: 0.2,
      },
      0.2,
    ),
    p(
      EulerSpec::HullWhite {
        alpha: 0.5,
        sigma: 0.02,
      },
      0.03,
    ),
    p(EulerSpec::CurveDrift { sigma: 0.02 }, 0.0),
    p(
      EulerSpec::LogMeanReverting {
        decay: 0.99,
        a: 0.5,
        sigma_eff: 0.02,
      },
      (0.03_f32).ln(),
    ),
    p(
      EulerSpec::ShiftedSquareRoot {
        theta: 2.0,
        mu: 0.04,
        sigma: 0.2,
      },
      0.04,
    ),
    p(
      EulerSpec::ShiftedSquareRootMirrored {
        theta: 2.0,
        mu: 0.04,
        sigma: 0.2,
      },
      0.04,
    ),
    p(EulerSpec::TimeVaryingGeometricBrownian { mu: 0.05 }, 100.0),
    p(
      EulerSpec::BrownianBridge {
        xt: 1.0,
        sigma: 0.2,
      },
      0.0,
    ),
    p(EulerSpec::CountingProcess, 0.0),
    p(EulerSpec::GammaSubordinator, 0.0),
    p(
      EulerSpec::VarianceGamma {
        mu: -0.05,
        sigma: 0.2,
      },
      0.0,
    ),
    p(EulerSpec::BilateralGamma, 0.0),
    p(EulerSpec::BilateralGammaMotion { sigma: 0.1 }, 0.0),
    p(EulerSpec::TemperedStableSubordinator { drift: 0.01 }, 0.0),
    p(
      EulerSpec::Innovation {
        mean: 0.01,
        sd: 0.2,
      },
      0.0,
    ),
    p(
      EulerSpec::Autoregressive {
        phi: 0.6,
        sigma: 0.2,
      },
      0.0,
    ),
    p(
      EulerSpec::StableSubordinator {
        alpha: 0.7,
        inv_alpha: 1.0 / 0.7,
        one_minus_alpha: 0.3,
        tail_exp: 0.3 / 0.7,
        scale: 0.05,
        pi: std::f32::consts::PI,
      },
      0.0,
    ),
    p(
      EulerSpec::InverseGaussianSubordinator {
        mu_ig: 0.02,
        two_lam: 0.0008,
        four_mu_lam: 0.000032,
      },
      0.0,
    ),
    p(
      EulerSpec::NormalInverseGaussian {
        theta: -0.1,
        sigma: 0.2,
        mu_ig: 0.02,
        two_lam: 0.0008,
        four_mu_lam: 0.000032,
      },
      0.0,
    ),
    p(
      EulerSpec::MertonJumpLog {
        drift_ln: 0.0001,
        sigma: 0.2,
      },
      100.0,
    ),
  ]
}

/// A system that is only a family: what [`Probe`] is for a one-component
/// family, for one with `D` of them.
pub(crate) struct SystemProbe<const D: usize> {
  spec: EulerSpec<f32>,
  x0: [f32; D],
  /// The Markov lift a lifted family steps under, `None` for the rest.
  lift: Option<ProbeLift>,
}

/// The host stream for a [`SystemProbe`]: independent normals per noise
/// component through the family's own generated host step.
pub(crate) struct SystemProbeSampler<const D: usize> {
  spec: EulerSpec<f32>,
  x0: [f32; D],
  dt: f32,
  normal: SimdNormal<f32>,
  lift: Option<ProbeLift>,
}

impl<const D: usize> PathSampler<f32> for SystemProbeSampler<D> {
  type Output = [Array1<f32>; D];

  fn sample_into(&mut self, out: &mut [Array1<f32>; D]) {
    let (code, params) = self.spec.encode();
    let family = super::families::Family::from_code(code).expect("a declared family");
    let noises = family.noises();
    let curves = probe_curves();
    let mut state = [0.0f32; 4];
    state[..D].copy_from_slice(&self.x0);
    let mut reported = [0.0f32; 4];
    super::families::host_report(
      family,
      &state,
      &params,
      curves[0][0],
      curves[1][0],
      curves[2][0],
      curves[3][0],
      curves[4][0],
      curves[5][0],
      curves[6][0],
      curves[7][0],
      0.0,
      0.0,
      0.7,
      1.3,
      0.5,
      0.5,
      0.0,
      &mut reported,
    );
    for (c, path) in out.iter_mut().enumerate() {
      path[0] = reported[c];
    }
    let mut draw = vec![0.0f32; noises];
    // The lift's two state vectors, advanced per step exactly as the frame
    // advances them, from the family's own drift, diffusion and shock.
    let nodes = self.lift.as_ref().map_or(0, |l| l.decay.len());
    let (mut lh, mut lj) = (vec![0.0f32; nodes], vec![0.0f32; nodes]);
    for i in 1..N {
      let mut noise = [0.0f32; 4];
      self.normal.fill_slice(&mut draw);
      noise[..noises].copy_from_slice(&draw);
      let (mut lv, mut coefficients) = (0.0f32, [0.0f32; 3]);
      if let Some(lift) = &self.lift {
        coefficients = super::families::host_lift(family, &state, &params, self.dt, &noise);
        let [lf, lg, lsh] = coefficients;
        let hist: f32 = (0..nodes).map(|l| lift.weight[l] * (lh[l] + lj[l])).sum();
        lv = lift.x0 + lift.db * lf + hist + lift.fb * lg * lsh;
      }
      let mut next = [0.0f32; 4];
      super::families::host_step(
        family,
        &state,
        &params,
        self.dt,
        curves[0][i],
        curves[1][i],
        curves[2][i],
        curves[3][i],
        curves[4][i],
        curves[5][i],
        curves[6][i],
        curves[7][i],
        0.0,
        0.0,
        0.7,
        1.3,
        0.5,
        0.5,
        lv,
        &noise,
        &mut next,
      );
      state = next;
      if let Some(lift) = &self.lift {
        let [lf, lg, lsh] = coefficients;
        for l in 0..nodes {
          lh[l] = lift.decay[l] * lh[l] + lift.drift_scale[l] * lf;
          lj[l] = lift.decay[l] * (lj[l] + lg * lsh);
        }
      }
      super::families::host_report(
        family,
        &state,
        &params,
        curves[0][i],
        curves[1][i],
        curves[2][i],
        curves[3][i],
        curves[4][i],
        curves[5][i],
        curves[6][i],
        curves[7][i],
        0.0,
        0.0,
        0.7,
        1.3,
        0.5,
        0.5,
        0.0,
        &mut reported,
      );
      for (c, path) in out.iter_mut().enumerate() {
        path[i] = reported[c];
      }
    }
  }

  fn sample(&mut self) -> [Array1<f32>; D] {
    let mut out = std::array::from_fn(|_| Array1::zeros(N));
    self.sample_into(&mut out);
    out
  }
}

impl<const D: usize> ProcessExt<f32> for SystemProbe<D> {
  type Output = [Array1<f32>; D];
  type Sampler<'s>
    = SystemProbeSampler<D>
  where
    Self: 's;

  fn sampler(&self) -> SystemProbeSampler<D> {
    let dt = 1.0 / (N - 1) as f32;
    SystemProbeSampler {
      spec: self.spec,
      x0: self.x0,
      dt,
      normal: SimdNormal::<f32>::new(0.0, dt.sqrt(), &Deterministic::new(7)),
      lift: self.lift.clone(),
    }
  }
}

impl<const D: usize> EulerSystem<f32, D> for SystemProbe<D> {
  fn euler_spec(&self) -> EulerSpec<f32> {
    self.spec
  }

  fn initial_state(&self) -> [f32; 4] {
    let mut slots = [0.0; 4];
    slots[..D].copy_from_slice(&self.x0);
    slots
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

  fn curves(&self) -> Option<Vec<Vec<f32>>> {
    Some(probe_curves())
  }

  fn lift_spec(&self) -> Option<crate::euler::LiftSpec<'_, f32>> {
    self.lift.as_ref().map(|l| crate::euler::LiftSpec {
      decay: &l.decay,
      weight: &l.weight,
      drift_scale: &l.drift_scale,
      drift_boundary: l.db,
      diffusion_boundary: l.fb,
      x0: l.x0,
    })
  }

  fn jump_intensity(&self) -> Option<f32> {
    Some(PROBE_INTENSITY)
  }

  fn jump_sizes(&self) -> Option<JumpSizes<f32>> {
    Some(PROBE_SIZES)
  }

  fn gamma_draws(&self) -> Option<GammaDraws<f32>> {
    Some(PROBE_GAMMAS)
  }

  fn host_sample(&self) -> [Array1<f32>; D] {
    <Self as ProcessExt<f32>>::sampler(self).sample()
  }
}

/// One probe per two-component family.
fn every_two_component_family() -> Vec<SystemProbe<2>> {
  let heston = |rho, pow_v| (0.03, 2.0, 0.04, 0.3, rho, pow_v);
  let (mu, kappa, theta, sigma, rho, pow_v) = heston(-0.7, 0.5);
  vec![
    SystemProbe {
      spec: EulerSpec::Heston {
        mu,
        kappa,
        theta,
        sigma,
        rho,
        pow_v,
      },
      x0: [100.0, 0.04],
      lift: None,
    },
    SystemProbe {
      spec: EulerSpec::HestonReflected {
        mu,
        kappa,
        theta,
        sigma,
        rho,
        pow_v,
      },
      x0: [100.0, 0.04],
      lift: None,
    },
    SystemProbe {
      spec: EulerSpec::Sabr {
        beta: 0.5,
        rho: -0.4,
        nu: 0.4,
        half_nu_sq: 0.08,
      },
      x0: [100.0, 0.2],
      lift: None,
    },
    SystemProbe {
      spec: EulerSpec::TwoScaleOrnsteinUhlenbeck {
        kappa: 1.0,
        theta: 0.0,
        eps: 0.3,
        alpha: 0.0,
        eps_inv: 4.0,
        sqrt_eps_inv: 2.0,
      },
      x0: [0.5, 0.5],
      lift: None,
    },
    SystemProbe {
      spec: EulerSpec::LogHeston {
        drift: 0.03,
        kappa: 2.0,
        theta: 0.04,
        xi: 0.3,
        rho: -0.7,
      },
      x0: [100.0, 0.04],
      lift: None,
    },
    SystemProbe {
      spec: EulerSpec::KouJumpHeston {
        drift_c: 0.02,
        kappa: 2.0,
        theta: 0.04,
        sigma_v: 0.3,
        rho: -0.7,
      },
      x0: [100.0, 0.04],
      lift: None,
    },
    SystemProbe {
      spec: EulerSpec::KouJumpHestonReflected {
        drift_c: 0.02,
        kappa: 2.0,
        theta: 0.04,
        sigma_v: 0.3,
        rho: -0.7,
      },
      x0: [100.0, 0.04],
      lift: None,
    },
    SystemProbe {
      spec: EulerSpec::AndersenQe {
        theta: 0.04,
        e_kd: 0.9689,
        c1: 0.0436,
        k0: -0.00022,
        c2: 0.0009,
        k1: -2.3355,
        k2: 2.3323,
        k34: 0.001008,
        mu: 0.02,
      },
      x0: [(100.0_f32).ln(), 0.04],
      lift: None,
    },
    SystemProbe {
      spec: EulerSpec::BatesJump {
        drift_c: 0.02,
        alpha: 0.08,
        beta: 2.0,
        sigma: 0.3,
        rho: -0.7,
      },
      x0: [100.0, 0.04],
      lift: None,
    },
    SystemProbe {
      spec: EulerSpec::BatesJumpReflected {
        drift_c: 0.02,
        alpha: 0.08,
        beta: 2.0,
        sigma: 0.3,
        rho: -0.7,
      },
      x0: [100.0, 0.04],
      lift: None,
    },
    SystemProbe {
      spec: EulerSpec::HawkesJumpDiffusion {
        drift_c: 0.02,
        sigma: 0.2,
        alpha: 0.5,
        beta: 2.0,
        mu_lambda: 1.0,
        jump_mu: -0.02,
        jump_sigma: 0.05,
      },
      x0: [0.0, 1.0],
      lift: None,
    },
    SystemProbe {
      spec: EulerSpec::BarndorffNielsenShephard {
        decay: 0.99,
        mu: 0.02,
      },
      x0: [100.0, 0.04],
      lift: None,
    },
    SystemProbe {
      spec: EulerSpec::RiemannLiouvilleOu {
        kappa: 2.0,
        mu: 0.05,
        nu: 0.3,
      },
      x0: [0.02, 0.0],
      lift: Some(probe_lift(1.0 / (N - 1) as f32)),
    },
    SystemProbe {
      spec: EulerSpec::RiemannLiouvilleBlackScholes {
        s0: 100.0,
        sigma: 0.2,
      },
      x0: [100.0, 0.0],
      lift: Some(probe_lift(1.0 / (N - 1) as f32)),
    },
    SystemProbe {
      spec: EulerSpec::CorrelatedInnovation { rho: -0.5 },
      x0: [0.0, 0.0],
      lift: None,
    },
    SystemProbe {
      spec: EulerSpec::CorrelatedFractionalMotion { rho: 0.3 },
      x0: [0.0, 0.0],
      lift: None,
    },
    SystemProbe {
      spec: EulerSpec::DynamicSabr,
      x0: [0.04, 0.3],
      lift: None,
    },
    SystemProbe {
      spec: EulerSpec::WuZhang {
        alpha: 0.04,
        beta: 1.5,
        nu: 0.3,
        lambda: 0.8,
      },
      x0: [0.03, 0.04],
      lift: None,
    },
    SystemProbe {
      spec: EulerSpec::RegimeSwitching {
        mu: 0.03,
        sigma: [0.1, 0.25, 0.4, 0.0],
        thresholds: [
          [0.9, 0.97, 1.0],
          [0.05, 0.92, 1.0],
          [0.02, 0.1, 1.0],
          [1.0, 1.0, 1.0],
        ],
      },
      x0: [100.0, 1.0],
      lift: None,
    },
    SystemProbe {
      spec: EulerSpec::ComplexFractionalOu {
        lambda: 1.5,
        omega: 0.8,
        scale: 0.4,
      },
      x0: [0.1, -0.1],
      lift: None,
    },
    SystemProbe {
      spec: EulerSpec::MovingAverage {
        theta: 0.4,
        sigma: 0.2,
      },
      x0: [0.0, 0.0],
      lift: None,
    },
    SystemProbe {
      spec: EulerSpec::CorrelatedBrownian { rho: -0.5 },
      x0: [0.0, 0.0],
      lift: None,
    },
    SystemProbe {
      spec: EulerSpec::TwoFactorHullWhite {
        a: 1.0,
        b: 0.5,
        sigma1: 0.01,
        sigma2: 0.005,
        rho: -0.4,
      },
      x0: [0.02, 0.0],
      lift: None,
    },
    SystemProbe {
      spec: EulerSpec::TwoFactorSquareRoot {
        theta1: 2.0,
        mu1: 0.03,
        sigma1: 0.1,
        theta2: 1.0,
        mu2: 0.01,
        sigma2: 0.05,
        sym1: 0.0,
        sym2: 1.0,
      },
      x0: [0.03, 0.01],
      lift: None,
    },
    SystemProbe {
      spec: EulerSpec::DuffieKanJump {
        a1: -0.5,
        b1: 0.1,
        c1: 0.02,
        sigma1: 0.1,
        a2: 0.05,
        b2: -0.3,
        c2: 0.01,
        sigma2: 0.08,
        alpha: 0.5,
        beta: 0.2,
        gamma: 0.1,
        rho: -0.3,
      },
      x0: [0.03, 0.01],
      lift: None,
    },
    SystemProbe {
      spec: EulerSpec::DuffieKan {
        a1: -0.5,
        b1: 0.1,
        c1: 0.02,
        sigma1: 0.1,
        a2: 0.05,
        b2: -0.3,
        c2: 0.01,
        sigma2: 0.08,
        alpha: 0.5,
        beta: 0.2,
        gamma: 0.1,
        rho: -0.3,
      },
      x0: [0.03, 0.01],
      lift: None,
    },
    SystemProbe {
      spec: EulerSpec::LogHestonReflected {
        drift: 0.03,
        kappa: 2.0,
        theta: 0.04,
        xi: 0.3,
        rho: -0.7,
      },
      x0: [100.0, 0.04],
      lift: None,
    },
  ]
}

/// One probe per three-component family.
fn every_three_component_family() -> Vec<SystemProbe<3>> {
  let hjm = SystemProbe {
    spec: EulerSpec::HeathJarrowMorton,
    x0: [0.03, 1.0, 0.04],
    lift: None,
  };
  let garch = SystemProbe {
    spec: EulerSpec::Garch {
      omega: 0.00001,
      alpha: 0.1,
      beta: 0.85,
    },
    x0: [0.0, 0.0002, 0.0],
    lift: None,
  };
  let threshold = SystemProbe {
    spec: EulerSpec::ThresholdGarch {
      omega: 0.00001,
      alpha: 0.05,
      gamma: 0.1,
      beta: 0.85,
    },
    x0: [0.0, 0.0002, 0.0],
    lift: None,
  };
  let double = |sym| {
    let spec = if sym {
      EulerSpec::DoubleHestonReflected {
        mu: 0.03,
        kappa1: 2.0,
        theta1: 0.04,
        sigma1: 0.3,
        rho1: -0.7,
        kappa2: 1.0,
        theta2: 0.02,
        sigma2: 0.2,
        rho2: -0.3,
      }
    } else {
      EulerSpec::DoubleHeston {
        mu: 0.03,
        kappa1: 2.0,
        theta1: 0.04,
        sigma1: 0.3,
        rho1: -0.7,
        kappa2: 1.0,
        theta2: 0.02,
        sigma2: 0.2,
        rho2: -0.3,
      }
    };
    SystemProbe {
      spec,
      x0: [100.0, 0.04, 0.02],
      lift: None,
    }
  };
  let exponential = SystemProbe {
    spec: EulerSpec::ExponentialGarch {
      omega: -0.2,
      alpha: 0.1,
      gamma: -0.05,
      beta: 0.95,
      e_abs_z: std::f32::consts::FRAC_2_PI.sqrt(),
    },
    x0: [0.0, -0.2, 0.0],
    lift: None,
  };
  vec![
    hjm,
    garch,
    threshold,
    exponential,
    double(false),
    double(true),
    SystemProbe {
      spec: EulerSpec::StochasticCorrelationHeston {
        kappa_r: 1.0,
        mu_r: -0.3,
        sigma_r: 0.2,
        kappa_v: 2.0,
        mu_v: 0.04,
        sigma_v: 0.3,
        r: 0.02,
        rho2: 0.1,
      },
      x0: [100.0, 0.04, -0.3],
      lift: None,
    },
  ]
}

/// One probe per four-component family. The Bergomi variance is a function of
/// the running sum of its own increments, so that sum and the elapsed time
/// travel as two further components; the probe compares all four.
fn every_four_component_family() -> Vec<SystemProbe<4>> {
  let two_asset = |sym| {
    let (mu1, mu2) = (0.03, 0.02);
    let (kappa1, theta1, sigma1) = (2.0, 0.04, 0.3);
    let (kappa2, theta2, sigma2) = (1.5, 0.03, 0.25);
    let (l11, l21, l22) = (1.0, -0.6, 0.8);
    let (l31, l32, l33) = (0.2, 0.1, 0.97);
    let (l41, l42, l43, l44) = (0.1, -0.2, 0.3, 0.92);
    let spec = if sym {
      EulerSpec::TwoAssetHestonReflected {
        mu1,
        mu2,
        kappa1,
        theta1,
        sigma1,
        kappa2,
        theta2,
        sigma2,
        l11,
        l21,
        l22,
        l31,
        l32,
        l33,
        l41,
        l42,
        l43,
        l44,
      }
    } else {
      EulerSpec::TwoAssetHeston {
        mu1,
        mu2,
        kappa1,
        theta1,
        sigma1,
        kappa2,
        theta2,
        sigma2,
        l11,
        l21,
        l22,
        l31,
        l32,
        l33,
        l41,
        l42,
        l43,
        l44,
      }
    };
    SystemProbe {
      spec,
      x0: [4.6, 0.04, 4.6, 0.03],
      lift: None,
    }
  };
  vec![
    SystemProbe {
      spec: EulerSpec::CorrelatedGeometric4 {
        mu: [0.03, 0.02, 0.01, 0.04],
        sigma: [0.2, 0.3, 0.25, 0.15],
        l: [1.0, -0.6, 0.8, 0.2, 0.1, 0.97, 0.1, -0.2, 0.3, 0.92],
      },
      x0: [100.0, 50.0, 80.0, 60.0],
      lift: None,
    },
    SystemProbe {
      spec: EulerSpec::CorrelatedNoises4 {
        l: [1.0, -0.6, 0.8, 0.2, 0.1, 0.97, 0.1, -0.2, 0.3, 0.92],
      },
      x0: [0.0, 0.0, 0.0, 0.0],
      lift: None,
    },
    two_asset(false),
    two_asset(true),
    SystemProbe {
      spec: EulerSpec::Bergomi {
        r: 0.02,
        nu: 0.5,
        half_nu_sq: 0.125,
        v0_sq: 0.04,
        rho: -0.6,
      },
      x0: [100.0, 0.04, 0.0, 0.0],
      lift: None,
    },
  ]
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

  /// One part in a thousand of the value, with a floor of `1e-2` so a state
  /// that sits near zero is still held to a scale rather than to an absolute
  /// `1e-3` that would be a few percent of it. Relative rather than exact
  /// because a family with a branch — the inverse-Gaussian draw's accept
  /// test — can take the other side of a boundary on one runtime when two
  /// `f32` roundings land a hair apart, and the path then differs by one
  /// draw; both sides are draws of the same law, so that is not a defect.
  fn agree(name: &str, native: &Array1<f32>, cube: &Array1<f32>) {
    for (x, y) in native.iter().zip(cube.iter()) {
      assert!(
        (x - y).abs() < 1e-3 * y.abs().max(1e-2),
        "{name}: native {x} vs cubecl {y}"
      );
    }
  }

  for probe in every_family() {
    let name = family_name(&probe.spec);
    let native = crate::device::Metal::default().euler_paths(&probe, 8);
    let cube = Cube::default().euler_paths(&probe, 8);
    for (a, b) in native.iter().zip(&cube) {
      agree(name, a, b);
    }
  }
  for probe in every_two_component_family() {
    let name = family_name(&probe.spec);
    let native = crate::device::Metal::default().system_paths(&probe, 8);
    let cube = Cube::default().system_paths(&probe, 8);
    for (a, b) in native.iter().zip(&cube) {
      for (x, y) in a.iter().zip(b.iter()) {
        agree(name, x, y);
      }
    }
  }
  for probe in every_three_component_family() {
    let name = family_name(&probe.spec);
    let native = crate::device::Metal::default().system_paths(&probe, 8);
    let cube = Cube::default().system_paths(&probe, 8);
    for (a, b) in native.iter().zip(&cube) {
      for (x, y) in a.iter().zip(b.iter()) {
        agree(name, x, y);
      }
    }
  }
  for probe in every_four_component_family() {
    let name = family_name(&probe.spec);
    let native = crate::device::Metal::default().system_paths(&probe, 8);
    let cube = Cube::default().system_paths(&probe, 8);
    for (a, b) in native.iter().zip(&cube) {
      for (x, y) in a.iter().zip(b.iter()) {
        agree(name, x, y);
      }
    }
  }
}

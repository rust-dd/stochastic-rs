use super::*;

/// One probe per family, in a regime where a wrong step looks different from a right one: away
/// from a boundary the family clamps to, with a diffusion that moves the state over 64 points.
pub(super) fn every_family() -> Vec<Probe> {
  let p = |spec, x0| Probe {
    spec,
    x0,
    lift: None,
    program: None,
  };
  // A lifted family runs under the probes' small Riemann-Liouville lift.
  let p_lift = |spec, x0| Probe {
    spec,
    x0,
    lift: Some(probe_lift(1.0 / (N - 1) as f32)),
    program: None,
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
    p(
      EulerSpec::AdditiveJumpDiffusion {
        drift_dt: 0.0002,
        sigma: 0.2,
      },
      1.0,
    ),
    p(
      EulerSpec::JumpFractionalOu {
        theta: 1.5,
        mu: 0.02,
        sigma: 0.3,
      },
      0.0,
    ),
    p_lift(
      EulerSpec::VolterraSquareRoot {
        kappa: 2.0,
        theta: 0.04,
        nu: 0.3,
      },
      0.04,
    ),
    p_lift(
      EulerSpec::GaussianPolynomialVolatility {
        coefficients: [0.2, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
      },
      0.0,
    ),
    p(
      EulerSpec::LinearFractionalStable {
        alpha: 1.7,
        inv_alpha: 1.0 / 1.7,
        tail_exp: (1.0 - 1.7) / 1.7,
        b: 0.1,
        scale_s: 0.05,
        pi: std::f32::consts::PI,
        half_pi: std::f32::consts::FRAC_PI_2,
      },
      0.0,
    ),
    p(EulerSpec::MovingAverageFilter { sigma: 0.3 }, 0.0),
    p(
      EulerSpec::TemperedStableSeries {
        b_t: 0.01,
        rate: 0.25,
        inv_alpha: 2.0,
        e_scale: 1.0,
        e_pow: 1.0,
        w_plus: 0.5,
        lambda_plus: 3.0,
        lambda_minus: 3.0,
      },
      0.0,
    ),
    p(EulerSpec::VolterraReference, 0.0),
    p(
      EulerSpec::HawkesEvents {
        mu: 1.0,
        alpha: 0.5,
        beta: 1.5,
      },
      0.0,
    ),
    p(
      EulerSpec::InverseStableSubordinator {
        alpha: 0.7,
        c: 1.0,
        inv_alpha: 1.0 / 0.7,
        one_minus_alpha: 0.3,
        tail_exp: 0.3 / 0.7,
        pi: std::f32::consts::PI,
      },
      0.0,
    ),
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
        pull: 0.02,
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
        log_scale: 0.05_f32.ln(),
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
    // A Volterra equation whose drift and diffusion are programs, under the
    // probes' lift.
    Probe {
      spec: EulerSpec::VolterraProgram,
      x0: 0.1,
      lift: Some(probe_lift(1.0 / (N - 1) as f32)),
      program: Some(ProbePrograms::pair()),
    },
  ]
}

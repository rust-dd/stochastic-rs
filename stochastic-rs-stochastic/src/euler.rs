//! # Euler engine
//!
//! $$
//! X_{i+1} = X_i + b(X_i)\,\Delta t + \sigma(X_i)\sqrt{\Delta t}\,Z_i,\qquad
//! Z_i \sim \mathcal N(0, 1)\ \text{i.i.d.}
//! $$
//!
//! Device-side path generation for the diffusions whose coefficients are a
//! handful of scalars. The backend is a type parameter of the process, as
//! for the fGN-driven types: `Gbm<T, S, B = Cpu>`, `Ou`, `Cir` are switched
//! with `.on::<B>()` and then sampled through [`ProcessExt`] as usual —
//! `gbm.on::<Metal>().sample_par(m)`.
//!
//! - [`Cpu`] (and `Accelerate`, a CPU device) is **the process's own
//!   sampler**, so nothing is re-implemented on the host: GBM keeps its exact
//!   log-normal scheme, OU and CIR their SIMD Euler steppers.
//! - The GPU back-ends run one device thread per path with the whole
//!   Euler–Maruyama recursion in the kernel and Box–Muller normals from a
//!   counter hash of `(path, step, seed)`: `Cubecl` (its CUDA runtime, or
//!   Metal / Vulkan / WebGPU through wgpu, `f32`),
//!   `Cuda` (feature `cuda`: cudarc + NVRTC, `f32` or `f64` after
//!   `T`) and `Metal` (feature `metal`: hand-written MSL, `f32`).
//!   `sample_par` is one launch for all `m` paths; `sample` launches one path.
//!
//! The device seed is drawn from the process's own seed source, so the same
//! `Deterministic` seed value gives the same device paths, consecutive calls
//! advance the stream and an `Unseeded` process draws fresh entropy, exactly
//! as on the host. The device
//! kernels share one integer hash, so the device back-ends agree with each
//! other seed for seed up to libm rounding; the host path is the process's
//! own stream, so CPU and device paths agree in distribution, not bit for bit.
//!
//! A process joins the engine by describing its coefficients as an
//! [`EulerSpec`] through [`EulerCoefficients`].
//!
//! References: Kloeden, P. E. & Platen, E. (1992), *Numerical Solution of
//! Stochastic Differential Equations*, Springer, §10.2 (Euler–Maruyama);
//! Lord, R., Koekkoek, R. & van Dijk, D. (2010), *A comparison of biased
//! simulation schemes for stochastic volatility models*, Quantitative Finance
//! 10(2), 177–194 (full truncation, used by the device kernels for CIR).

use ndarray::Array1;
use ndarray::Array2;
use ndarray::Array3;
use stochastic_rs_core::simd_rng::SeedExt;

use crate::device::Backend;
use crate::device::Cpu;
use crate::device::DeviceError;
use crate::diffusion::cir::Cir;
use crate::diffusion::gbm::Gbm;
use crate::diffusion::ou::Ou;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;
use crate::traits::process::sample_map_chunked;
use crate::traits::process::sample_par_chunked;

/// How many scalar parameters one family may carry. The layout is the
/// kernels' ABI and stays inside the crate, so it can widen again without a
/// breaking change.
/// The streams a correlated four-component family can carry; a model with
/// more stays on its host sampler whatever the backend.
pub(crate) const CORRELATED_STREAMS: usize = 4;

/// A lower Cholesky factor of at most [`CORRELATED_STREAMS`] rows, packed
/// row-major into the ten lower-triangle slots the correlated families take
/// (`l00, l10, l11, l20, ...`). Rows the factor does not have get a unit
/// diagonal, which is what leaves a padded component equal to its own shock
/// and, with zero drift and volatility, constant.
pub(crate) fn pack_cholesky<T: FloatExt>(chol: &ndarray::Array2<T>) -> [T; 10] {
  let k = chol.nrows();
  assert!(
    k <= CORRELATED_STREAMS,
    "a correlated family carries at most {CORRELATED_STREAMS} streams, not {k}"
  );
  let mut out = [T::zero(); 10];
  let mut at = 0;
  for i in 0..CORRELATED_STREAMS {
    for j in 0..=i {
      out[at] = if i < k {
        chol[(i, j)]
      } else if i == j {
        T::one()
      } else {
        T::zero()
      };
      at += 1;
    }
  }
  out
}

/// How many time-varying coefficients a kernel binds per launch, which is the
/// most a process's `curves()` may return. Public because that hook is: an
/// out-of-tree process needs the cap it is held to. A family
/// names them `ct` and `ct1` through `ct7`; a launch pays one buffer read per
/// declared curve per step, so declaring fewer costs less.
pub const CURVE_SLOTS: usize = 8;

/// The curve buffer a launch binds: the declared curves laid end to end, each
/// padded to `n` values, so the kernel reads curve `k` at step `i` from
/// `curve[k * n + i]`. Returns the flattened values and how many curves they
/// hold; an empty buffer and zero when the family declares none.
///
/// A curve shorter than the grid is extended with its last value rather than
/// read out of bounds — a host tabulation that stops one short is a
/// declaration slip, not a reason to fault a kernel.
#[cfg_attr(
  not(any(
    feature = "cuda",
    feature = "metal",
    feature = "cubecl-cuda",
    feature = "cubecl-wgpu"
  )),
  allow(dead_code)
)]
pub(crate) fn flatten_curves<T: FloatExt>(curves: Option<Vec<Vec<T>>>, n: usize) -> (Vec<T>, u32) {
  let Some(curves) = curves else {
    return (Vec::new(), 0);
  };
  assert!(
    curves.len() <= CURVE_SLOTS,
    "a family may declare at most {CURVE_SLOTS} curves, not {}",
    curves.len()
  );
  let mut flat = Vec::with_capacity(curves.len() * n);
  for curve in &curves {
    let last = curve.last().copied().unwrap_or_else(T::zero);
    flat.extend((0..n).map(|i| curve.get(i).copied().unwrap_or(last)));
  }
  (flat, curves.len() as u32)
}

pub(crate) const PARAM_SLOTS: usize = 20;

/// Scalar drift / diffusion families the device kernels know how to step.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum EulerSpec<T: FloatExt> {
  /// `dX = μX dt + σX dW`.
  GeometricBrownian { mu: T, sigma: T },
  /// `dX = θ(μ − X) dt + σ dW`.
  OrnsteinUhlenbeck { theta: T, mu: T, sigma: T },
  /// `dX = κ(θ − X) dt + σ√X dW`, stepped with full truncation (Lord,
  /// Koekkoek & van Dijk 2010): the recursion runs on an auxiliary process
  /// whose positive part enters drift, diffusion and the reported path.
  SquareRoot { kappa: T, theta: T, sigma: T },
  /// `dX = dW`: the increment accumulates. With fractional increments this is
  /// fractional Brownian motion.
  Additive,
  /// `dX = θ(μ − X) dt + σ√|X| dW`, clamped at zero after the step — the
  /// fractional CIR recursion, which truncates the result rather than
  /// stepping on a truncated state.
  ReflectedSquareRoot { theta: T, mu: T, sigma: T },
  /// The same with the symmetric reflection: the step's absolute value.
  MirroredSquareRoot { theta: T, mu: T, sigma: T },
  /// `dX = (α − βX) dt + σ√(X(1−X)) dW` on the unit interval, absorbing at
  /// both ends.
  Jacobi { alpha: T, beta: T, sigma: T },
  /// `dX = μX dt + σ|X|^γ dW`: constant elasticity of variance.
  ConstantElasticity { mu: T, sigma: T, gamma: T },
  /// `dX = (θ₁ + θ₂X) dt + θ₃|X|^θ₄ dW`: Chan–Karolyi–Longstaff–Sanders.
  Ckls {
    theta1: T,
    theta2: T,
    theta3: T,
    theta4: T,
  },
  /// `dX = X(1 − aX) dt + bX dW`: logistic growth with multiplicative noise.
  Logistic { a: T, b: T },
  /// `dX = κX(μ − X) dt + σ|X|^{3/2} dW`: the 3/2 model.
  ThreeHalf { kappa: T, mu: T, sigma: T },
  /// Geometric Brownian motion stepped in logs, with the log drift per step
  /// already formed on the host.
  LogGeometric { drift_ln: T, sigma: T },
  /// `dX = (κ/X − X) dt + σ dW`: radial Ornstein–Uhlenbeck.
  RadialOrnsteinUhlenbeck { kappa: T, sigma: T },
  /// `dX = (a + bX) dt + cX dW`: the linear scalar SDE.
  LinearSde { a: T, b: T, c: T },
  /// `dX = −κX/√(1+X²) dt + σ dW`.
  Hyperbolic { kappa: T, sigma: T },
  /// `dX = −κX dt + σ√(1+X²) dW`.
  ModifiedSquareRoot { kappa: T, sigma: T },
  /// `dX = X(θ₁ − X·d) dt + θ₃|X|^{3/2} dW`, with `d = θ₃³ − θ₁θ₂` folded on
  /// the host.
  FellerRoot { theta1: T, decay: T, theta3: T },
  /// The Aït-Sahalia short-rate model's eight coefficients.
  AitSahalia {
    am1: T,
    a0: T,
    a1: T,
    a2: T,
    b0: T,
    b1: T,
    b2: T,
    b3: T,
  },
  /// `dX = (a − b·ln X)X dt + σX dW`, floored at `1e-12`.
  Gompertz { a: T, b: T, sigma: T },
  /// `dX = aX(1−X) dt + σ√(X(1−X)) dW` on `[0, 1]`.
  Kimura { a: T, sigma: T },
  /// `dX = (α + βX + γX²) dt + σX dW`.
  Quadratic {
    alpha: T,
    beta: T,
    gamma: T,
    sigma: T,
  },
  /// `dX = κ(μ − X) dt + √|2κ(aX² + bX + c)| dW`, with `2κ` folded on the host.
  Pearson {
    kappa: T,
    mu: T,
    a: T,
    b: T,
    c: T,
    two_kappa: T,
  },
  /// `dX = rX(1 − X/K) dt + σX dW`, unclamped.
  Verhulst { r: T, k: T, sigma: T },
  /// [`Verhulst`](EulerSpec::Verhulst) confined to `[0, K]`.
  VerhulstClamped { r: T, k: T, sigma: T },
  /// `dX = κ(θ − X)X dt + σ√X dW`, truncated at zero.
  FellerLogistic { kappa: T, theta: T, sigma: T },
  /// [`FellerLogistic`](EulerSpec::FellerLogistic) reflected at zero.
  FellerLogisticReflected { kappa: T, theta: T, sigma: T },
  /// `dX = δ dt + 2√|X| dW`, truncated at zero.
  SquaredBesselState { delta: T, two: T },
  /// [`SquaredBesselState`](EulerSpec::SquaredBesselState) reflected at zero.
  SquaredBesselStateReflected { delta: T, two: T },
  /// The squared-Bessel recursion reporting `√X`.
  BesselFromSquared { delta: T, two: T },
  /// [`BesselFromSquared`](EulerSpec::BesselFromSquared) reflected at zero.
  BesselFromSquaredReflected { delta: T, two: T },
  /// `dX = ½σ²(β − γ(X−μ)/√(δ² + (X−μ)²)) dt + σ dW`, with `½σ²` folded on the host.
  HyperbolicDiffusion {
    beta: T,
    gamma: T,
    delta: T,
    mu: T,
    sigma: T,
    half_var: T,
  },
  /// The Aït-Sahalia drift with the diffusion left unsquared.
  NonLinear {
    am1: T,
    a0: T,
    a1: T,
    a2: T,
    b0: T,
    b1: T,
    b2: T,
    b3: T,
  },
  /// Geometric Brownian motion on `Y = S + β`, reported as `Y − β`.
  Displaced { mu: T, sigma: T, beta: T },
  /// `dX = κ(μ − tanh X) dt + σ dW` reported as `tanh X`.
  TanhOrnsteinUhlenbeck { kappa: T, mu: T, sigma: T },
  /// `dρ = κ(μ − ρ) dt + σ√(1 − ρ²) dW` confined to `[−0.9999, 0.9999]`.
  BoundedCorrelation { kappa: T, mu: T, sigma: T },
  /// The Heston model's Euler scheme, variance truncated at zero.
  Heston {
    mu: T,
    kappa: T,
    theta: T,
    sigma: T,
    rho: T,
    pow_v: T,
  },
  /// [`Heston`](EulerSpec::Heston) with the variance reflected at zero.
  HestonReflected {
    mu: T,
    kappa: T,
    theta: T,
    sigma: T,
    rho: T,
    pow_v: T,
  },
  /// SABR under an exact log-normal volatility step.
  Sabr {
    beta: T,
    rho: T,
    nu: T,
    half_nu_sq: T,
  },
  /// The Bergomi model, whose variance is a function of the running sum of
  /// its own increments; that sum and the elapsed time travel as state.
  Bergomi {
    r: T,
    nu: T,
    half_nu_sq: T,
    v0_sq: T,
    rho: T,
  },
  /// A slow and a fast Ornstein-Uhlenbeck factor on one clock.
  TwoScaleOrnsteinUhlenbeck {
    kappa: T,
    theta: T,
    eps: T,
    alpha: T,
    eps_inv: T,
    sqrt_eps_inv: T,
  },
  /// The Heston model stepped in log-price, variance truncated.
  LogHeston {
    drift: T,
    kappa: T,
    theta: T,
    xi: T,
    rho: T,
  },
  /// [`LogHeston`](EulerSpec::LogHeston) with the variance reflected.
  LogHestonReflected {
    drift: T,
    kappa: T,
    theta: T,
    xi: T,
    rho: T,
  },
  /// Two variance factors driving one spot, both truncated.
  DoubleHeston {
    mu: T,
    kappa1: T,
    theta1: T,
    sigma1: T,
    rho1: T,
    kappa2: T,
    theta2: T,
    sigma2: T,
    rho2: T,
  },
  /// [`DoubleHeston`](EulerSpec::DoubleHeston) with both variances reflected.
  DoubleHestonReflected {
    mu: T,
    kappa1: T,
    theta1: T,
    sigma1: T,
    rho1: T,
    kappa2: T,
    theta2: T,
    sigma2: T,
    rho2: T,
  },
  /// A Heston spot whose correlation to its variance is itself stochastic.
  StochasticCorrelationHeston {
    kappa_r: T,
    mu_r: T,
    sigma_r: T,
    kappa_v: T,
    mu_v: T,
    sigma_v: T,
    r: T,
    rho2: T,
  },
  /// Hull-White, whose mean-reversion level is the launch's curve.
  HullWhite { alpha: T, sigma: T },
  /// A drift that is entirely the curve, which is Ho-Lee.
  CurveDrift { sigma: T },
  /// The exact Ornstein-Uhlenbeck transition in log space: Black-Karasinski.
  LogMeanReverting { decay: T, a: T, sigma_eff: T },
  /// A square-root diffusion shifted by a curve: CIR++.
  ShiftedSquareRoot { theta: T, mu: T, sigma: T },
  /// [`ShiftedSquareRoot`](EulerSpec::ShiftedSquareRoot) reflected at zero.
  ShiftedSquareRootMirrored { theta: T, mu: T, sigma: T },
  /// Geometric Brownian motion over a term structure of volatilities.
  TimeVaryingGeometricBrownian { mu: T },
  /// Two Brownian motions correlated by `ρ`.
  CorrelatedBrownian { rho: T },
  /// A Brownian bridge stepped by its exact conditional law.
  BrownianBridge { xt: T, sigma: T },
  /// The two-factor Hull-White model.
  TwoFactorHullWhite {
    a: T,
    b: T,
    sigma1: T,
    sigma2: T,
    rho: T,
  },
  /// Two square-root factors whose shifted sum is the reported rate.
  TwoFactorSquareRoot {
    theta1: T,
    mu1: T,
    sigma1: T,
    theta2: T,
    mu2: T,
    sigma2: T,
    sym1: T,
    sym2: T,
  },
  /// The Duffie-Kan two-factor affine model.
  DuffieKan {
    a1: T,
    b1: T,
    c1: T,
    sigma1: T,
    a2: T,
    b2: T,
    c2: T,
    sigma2: T,
    alpha: T,
    beta: T,
    gamma: T,
    rho: T,
  },
  /// Two Heston assets under one 4x4 Cholesky factor, variances truncated.
  TwoAssetHeston {
    mu1: T,
    mu2: T,
    kappa1: T,
    theta1: T,
    sigma1: T,
    kappa2: T,
    theta2: T,
    sigma2: T,
    l11: T,
    l21: T,
    l22: T,
    l31: T,
    l32: T,
    l33: T,
    l41: T,
    l42: T,
    l43: T,
    l44: T,
  },
  /// [`TwoAssetHeston`](EulerSpec::TwoAssetHeston) with both variances reflected.
  TwoAssetHestonReflected {
    mu1: T,
    mu2: T,
    kappa1: T,
    theta1: T,
    sigma1: T,
    kappa2: T,
    theta2: T,
    sigma2: T,
    l11: T,
    l21: T,
    l22: T,
    l31: T,
    l32: T,
    l33: T,
    l41: T,
    l42: T,
    l43: T,
    l44: T,
  },
  /// Merton's jump diffusion in log-price.
  MertonJumpLog { drift_ln: T, sigma: T },
  /// The Bates stochastic-volatility jump model, variance truncated.
  BatesJump {
    drift_c: T,
    alpha: T,
    beta: T,
    sigma: T,
    rho: T,
  },
  /// [`BatesJump`](EulerSpec::BatesJump) with the variance reflected.
  BatesJumpReflected {
    drift_c: T,
    alpha: T,
    beta: T,
    sigma: T,
    rho: T,
  },
  /// Andersen's quadratic-exponential Heston step.
  AndersenQe {
    theta: T,
    e_kd: T,
    c1: T,
    c2: T,
    k0: T,
    k1: T,
    k2: T,
    k34: T,
    mu: T,
  },
  /// A Poisson counting process on the grid.
  CountingProcess,
  /// An inverse-Gaussian subordinator.
  InverseGaussianSubordinator {
    mu_ig: T,
    two_lam: T,
    four_mu_lam: T,
  },
  /// Brownian motion under an inverse-Gaussian clock.
  NormalInverseGaussian {
    theta: T,
    sigma: T,
    mu_ig: T,
    two_lam: T,
    four_mu_lam: T,
  },
  /// A positive-stable subordinator by the Chambers-Mallows-Stuck transform.
  StableSubordinator {
    alpha: T,
    inv_alpha: T,
    one_minus_alpha: T,
    tail_exp: T,
    scale: T,
    pi: T,
  },
  /// A Heston variance under a Kou-jumping log-price, variance truncated.
  KouJumpHeston {
    drift_c: T,
    kappa: T,
    theta: T,
    sigma_v: T,
    rho: T,
  },
  /// [`KouJumpHeston`](EulerSpec::KouJumpHeston) with the variance reflected.
  KouJumpHestonReflected {
    drift_c: T,
    kappa: T,
    theta: T,
    sigma_v: T,
    rho: T,
  },
  /// [`DuffieKan`](EulerSpec::DuffieKan) with a compound-Poisson jump.
  DuffieKanJump {
    a1: T,
    b1: T,
    c1: T,
    sigma1: T,
    a2: T,
    b2: T,
    c2: T,
    sigma2: T,
    alpha: T,
    beta: T,
    gamma: T,
    rho: T,
  },
  /// A jump diffusion whose intensity is excited by its own jumps.
  HawkesJumpDiffusion {
    drift_c: T,
    sigma: T,
    alpha: T,
    beta: T,
    mu_lambda: T,
    jump_mu: T,
    jump_sigma: T,
  },
  /// GARCH(1,1), and at zero persistence ARCH(1).
  Garch { omega: T, alpha: T, beta: T },
  /// GARCH(1,1) with a threshold term on negative returns.
  ThresholdGarch {
    omega: T,
    alpha: T,
    gamma: T,
    beta: T,
  },
  /// EGARCH(1,1), whose variance recursion runs in log space.
  ExponentialGarch {
    omega: T,
    alpha: T,
    gamma: T,
    beta: T,
    e_abs_z: T,
  },
  /// One draw per grid point, with no recursion over them.
  Innovation { mean: T, sd: T },
  /// A correlated pair of innovations.
  CorrelatedInnovation { rho: T },
  /// A first-order autoregression.
  Autoregressive { phi: T, sigma: T },
  /// A first-order moving average.
  MovingAverage { theta: T, sigma: T },
  /// A gamma subordinator.
  GammaSubordinator,
  /// Brownian motion under a gamma clock.
  VarianceGamma { mu: T, sigma: T },
  /// The difference of two gamma processes.
  BilateralGamma,
  /// [`BilateralGamma`](EulerSpec::BilateralGamma) with a Brownian part.
  BilateralGammaMotion { sigma: T },
  /// A tempered-stable subordinator: a deterministic small-jump drift plus
  /// the step's own thinned jumps.
  TemperedStableSubordinator { drift: T },
  /// The Barndorff-Nielsen-Shephard model: a gamma-driven variance and a
  /// log-Euler asset over it.
  BarndorffNielsenShephard { decay: T, mu: T },
  /// A correlated pair of fractional motions: both rows accumulate their own
  /// stream out of one embedding, correlated by `rho` in the step.
  CorrelatedFractionalMotion { rho: T },
  /// The complex fractional Ornstein-Uhlenbeck process, real and imaginary
  /// parts, under one complex mean reversion `lambda - i·omega`.
  ComplexFractionalOu { lambda: T, omega: T, scale: T },
  /// An Ornstein-Uhlenbeck process reported through a bounded map onto
  /// `(-1, 1)`: `tanh` at `arctan = 0`, `(2/pi) arctan(pi x / 2)` at one.
  TransformedOrnsteinUhlenbeck {
    kappa: T,
    mu: T,
    sigma: T,
    arctan: T,
    half_pi: T,
  },
  /// The arrival times of a Poisson process on a fixed count: the running sum
  /// of exponential inter-arrival times.
  PoissonArrivals { lambda: T },
  /// The dynamic SABR, whose three coefficients all travel as curves rather
  /// than parameters.
  DynamicSabr,
  /// Heath-Jarrow-Morton's three rows, with all six coefficients as curves.
  HeathJarrowMorton,
  /// One factor of the affine-diffusion Gaussian model: a mean reversion
  /// under two curves, reported through a quadratic map of three more.
  AffineDiffusionGaussian { sigma: T },
  /// One forward-rate / square-root-variance pair of the Wu-Zhang model.
  WuZhang { alpha: T, beta: T, nu: T, lambda: T },
  /// Up to four correlated geometric Brownian motions under one lower
  /// Cholesky factor `l` (row-major, lower triangle), padded for fewer.
  CorrelatedGeometric4 {
    mu: [T; 4],
    sigma: [T; 4],
    l: [T; 10],
  },
  /// Up to four correlated Gaussian noises under one lower Cholesky factor.
  CorrelatedNoises4 { l: [T; 10] },
}

/// Widens a family's parameter list to the kernels' fixed slot count.
fn pad<T: FloatExt, const N: usize>(values: [T; N]) -> [T; PARAM_SLOTS] {
  let mut slots = [T::zero(); PARAM_SLOTS];
  slots[..N].copy_from_slice(&values);
  slots
}

impl<T: FloatExt> EulerSpec<T> {
  /// Family code and the four parameter slots the device kernels read. The
  /// layout is the kernels' ABI and stays inside the crate, so it can widen
  /// for a new family without a breaking change. Only the device kernels
  /// read it, so a build without any device feature has no caller.
  #[cfg_attr(
    not(any(
      feature = "metal",
      feature = "cuda",
      feature = "cubecl-cuda",
      feature = "cubecl-wgpu"
    )),
    allow(dead_code)
  )]
  pub(crate) fn encode(&self) -> (u32, [T; PARAM_SLOTS]) {
    use families::Family;
    match *self {
      EulerSpec::GeometricBrownian { mu, sigma } => {
        (Family::GeometricBrownian.code(), pad([mu, sigma]))
      }
      EulerSpec::OrnsteinUhlenbeck { theta, mu, sigma } => {
        (Family::OrnsteinUhlenbeck.code(), pad([theta, mu, sigma]))
      }
      EulerSpec::SquareRoot {
        kappa,
        theta,
        sigma,
      } => (Family::SquareRoot.code(), pad([kappa, theta, sigma])),
      EulerSpec::Additive => (Family::Additive.code(), pad([])),
      EulerSpec::ReflectedSquareRoot { theta, mu, sigma } => {
        (Family::ReflectedSquareRoot.code(), pad([theta, mu, sigma]))
      }
      EulerSpec::MirroredSquareRoot { theta, mu, sigma } => {
        (Family::MirroredSquareRoot.code(), pad([theta, mu, sigma]))
      }
      EulerSpec::Jacobi { alpha, beta, sigma } => {
        (Family::Jacobi.code(), pad([alpha, beta, sigma]))
      }
      EulerSpec::ConstantElasticity { mu, sigma, gamma } => {
        (Family::ConstantElasticity.code(), pad([mu, sigma, gamma]))
      }
      EulerSpec::Ckls {
        theta1,
        theta2,
        theta3,
        theta4,
      } => (Family::Ckls.code(), pad([theta1, theta2, theta3, theta4])),
      EulerSpec::Logistic { a, b } => (Family::Logistic.code(), pad([a, b])),
      EulerSpec::ThreeHalf { kappa, mu, sigma } => {
        (Family::ThreeHalf.code(), pad([kappa, mu, sigma]))
      }
      EulerSpec::LogGeometric { drift_ln, sigma } => {
        (Family::LogGeometric.code(), pad([drift_ln, sigma]))
      }
      EulerSpec::RadialOrnsteinUhlenbeck { kappa, sigma } => {
        (Family::RadialOrnsteinUhlenbeck.code(), pad([kappa, sigma]))
      }
      EulerSpec::LinearSde { a, b, c } => (Family::LinearSde.code(), pad([a, b, c])),
      EulerSpec::Hyperbolic { kappa, sigma } => (Family::Hyperbolic.code(), pad([kappa, sigma])),
      EulerSpec::ModifiedSquareRoot { kappa, sigma } => {
        (Family::ModifiedSquareRoot.code(), pad([kappa, sigma]))
      }
      EulerSpec::FellerRoot {
        theta1,
        decay,
        theta3,
      } => (Family::FellerRoot.code(), pad([theta1, decay, theta3])),
      EulerSpec::AitSahalia {
        am1,
        a0,
        a1,
        a2,
        b0,
        b1,
        b2,
        b3,
      } => (
        Family::AitSahalia.code(),
        pad([am1, a0, a1, a2, b0, b1, b2, b3]),
      ),
      EulerSpec::Gompertz { a, b, sigma } => (Family::Gompertz.code(), pad([a, b, sigma])),
      EulerSpec::Kimura { a, sigma } => (Family::Kimura.code(), pad([a, sigma])),
      EulerSpec::Quadratic {
        alpha,
        beta,
        gamma,
        sigma,
      } => (Family::Quadratic.code(), pad([alpha, beta, gamma, sigma])),
      EulerSpec::Pearson {
        kappa,
        mu,
        a,
        b,
        c,
        two_kappa,
      } => (Family::Pearson.code(), pad([kappa, mu, a, b, c, two_kappa])),
      EulerSpec::Verhulst { r, k, sigma } => (Family::Verhulst.code(), pad([r, k, sigma])),
      EulerSpec::VerhulstClamped { r, k, sigma } => {
        (Family::VerhulstClamped.code(), pad([r, k, sigma]))
      }
      EulerSpec::FellerLogistic {
        kappa,
        theta,
        sigma,
      } => (Family::FellerLogistic.code(), pad([kappa, theta, sigma])),
      EulerSpec::FellerLogisticReflected {
        kappa,
        theta,
        sigma,
      } => (
        Family::FellerLogisticReflected.code(),
        pad([kappa, theta, sigma]),
      ),
      EulerSpec::SquaredBesselState { delta, two } => {
        (Family::SquaredBesselState.code(), pad([delta, two]))
      }
      EulerSpec::SquaredBesselStateReflected { delta, two } => (
        Family::SquaredBesselStateReflected.code(),
        pad([delta, two]),
      ),
      EulerSpec::BesselFromSquared { delta, two } => {
        (Family::BesselFromSquared.code(), pad([delta, two]))
      }
      EulerSpec::BesselFromSquaredReflected { delta, two } => {
        (Family::BesselFromSquaredReflected.code(), pad([delta, two]))
      }
      EulerSpec::HyperbolicDiffusion {
        beta,
        gamma,
        delta,
        mu,
        sigma,
        half_var,
      } => (
        Family::HyperbolicDiffusion.code(),
        pad([beta, gamma, delta, mu, sigma, half_var]),
      ),
      EulerSpec::NonLinear {
        am1,
        a0,
        a1,
        a2,
        b0,
        b1,
        b2,
        b3,
      } => (
        Family::NonLinear.code(),
        pad([am1, a0, a1, a2, b0, b1, b2, b3]),
      ),
      EulerSpec::Displaced { mu, sigma, beta } => {
        (Family::Displaced.code(), pad([mu, sigma, beta]))
      }
      EulerSpec::TanhOrnsteinUhlenbeck { kappa, mu, sigma } => (
        Family::TanhOrnsteinUhlenbeck.code(),
        pad([kappa, mu, sigma]),
      ),
      EulerSpec::BoundedCorrelation { kappa, mu, sigma } => {
        (Family::BoundedCorrelation.code(), pad([kappa, mu, sigma]))
      }
      EulerSpec::Heston {
        mu,
        kappa,
        theta,
        sigma,
        rho,
        pow_v,
      } => (
        Family::Heston.code(),
        pad([mu, kappa, theta, sigma, rho, pow_v]),
      ),
      EulerSpec::HestonReflected {
        mu,
        kappa,
        theta,
        sigma,
        rho,
        pow_v,
      } => (
        Family::HestonReflected.code(),
        pad([mu, kappa, theta, sigma, rho, pow_v]),
      ),
      EulerSpec::Sabr {
        beta,
        rho,
        nu,
        half_nu_sq,
      } => (Family::Sabr.code(), pad([beta, rho, nu, half_nu_sq])),
      EulerSpec::Bergomi {
        r,
        nu,
        half_nu_sq,
        v0_sq,
        rho,
      } => (Family::Bergomi.code(), pad([r, nu, half_nu_sq, v0_sq, rho])),
      EulerSpec::TwoScaleOrnsteinUhlenbeck {
        kappa,
        theta,
        eps,
        alpha,
        eps_inv,
        sqrt_eps_inv,
      } => (
        Family::TwoScaleOrnsteinUhlenbeck.code(),
        pad([kappa, theta, eps, alpha, eps_inv, sqrt_eps_inv]),
      ),
      EulerSpec::LogHeston {
        drift,
        kappa,
        theta,
        xi,
        rho,
      } => (
        Family::LogHeston.code(),
        pad([drift, kappa, theta, xi, rho]),
      ),
      EulerSpec::LogHestonReflected {
        drift,
        kappa,
        theta,
        xi,
        rho,
      } => (
        Family::LogHestonReflected.code(),
        pad([drift, kappa, theta, xi, rho]),
      ),
      EulerSpec::DoubleHeston {
        mu,
        kappa1,
        theta1,
        sigma1,
        rho1,
        kappa2,
        theta2,
        sigma2,
        rho2,
      } => (
        Family::DoubleHeston.code(),
        pad([
          mu, kappa1, theta1, sigma1, rho1, kappa2, theta2, sigma2, rho2,
        ]),
      ),
      EulerSpec::DoubleHestonReflected {
        mu,
        kappa1,
        theta1,
        sigma1,
        rho1,
        kappa2,
        theta2,
        sigma2,
        rho2,
      } => (
        Family::DoubleHestonReflected.code(),
        pad([
          mu, kappa1, theta1, sigma1, rho1, kappa2, theta2, sigma2, rho2,
        ]),
      ),
      EulerSpec::StochasticCorrelationHeston {
        kappa_r,
        mu_r,
        sigma_r,
        kappa_v,
        mu_v,
        sigma_v,
        r,
        rho2,
      } => (
        Family::StochasticCorrelationHeston.code(),
        pad([kappa_r, mu_r, sigma_r, kappa_v, mu_v, sigma_v, r, rho2]),
      ),
      EulerSpec::HullWhite { alpha, sigma } => (Family::HullWhite.code(), pad([alpha, sigma])),
      EulerSpec::CurveDrift { sigma } => (Family::CurveDrift.code(), pad([sigma])),
      EulerSpec::LogMeanReverting {
        decay,
        a,
        sigma_eff,
      } => (Family::LogMeanReverting.code(), pad([decay, a, sigma_eff])),
      EulerSpec::ShiftedSquareRoot { theta, mu, sigma } => {
        (Family::ShiftedSquareRoot.code(), pad([theta, mu, sigma]))
      }
      EulerSpec::ShiftedSquareRootMirrored { theta, mu, sigma } => (
        Family::ShiftedSquareRootMirrored.code(),
        pad([theta, mu, sigma]),
      ),
      EulerSpec::TimeVaryingGeometricBrownian { mu } => {
        (Family::TimeVaryingGeometricBrownian.code(), pad([mu]))
      }
      EulerSpec::CorrelatedBrownian { rho } => (Family::CorrelatedBrownian.code(), pad([rho])),
      EulerSpec::BrownianBridge { xt, sigma } => (Family::BrownianBridge.code(), pad([xt, sigma])),
      EulerSpec::TwoFactorHullWhite {
        a,
        b,
        sigma1,
        sigma2,
        rho,
      } => (
        Family::TwoFactorHullWhite.code(),
        pad([a, b, sigma1, sigma2, rho]),
      ),
      EulerSpec::TwoFactorSquareRoot {
        theta1,
        mu1,
        sigma1,
        theta2,
        mu2,
        sigma2,
        sym1,
        sym2,
      } => (
        Family::TwoFactorSquareRoot.code(),
        pad([theta1, mu1, sigma1, theta2, mu2, sigma2, sym1, sym2]),
      ),
      EulerSpec::DuffieKan {
        a1,
        b1,
        c1,
        sigma1,
        a2,
        b2,
        c2,
        sigma2,
        alpha,
        beta,
        gamma,
        rho,
      } => (
        Family::DuffieKan.code(),
        pad([
          a1, b1, c1, sigma1, a2, b2, c2, sigma2, alpha, beta, gamma, rho,
        ]),
      ),
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
      } => (
        Family::TwoAssetHeston.code(),
        pad([
          mu1, mu2, kappa1, theta1, sigma1, kappa2, theta2, sigma2, l11, l21, l22, l31, l32, l33,
          l41, l42, l43, l44,
        ]),
      ),
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
      } => (
        Family::TwoAssetHestonReflected.code(),
        pad([
          mu1, mu2, kappa1, theta1, sigma1, kappa2, theta2, sigma2, l11, l21, l22, l31, l32, l33,
          l41, l42, l43, l44,
        ]),
      ),
      EulerSpec::MertonJumpLog { drift_ln, sigma } => {
        (Family::MertonJumpLog.code(), pad([drift_ln, sigma]))
      }
      EulerSpec::BatesJump {
        drift_c,
        alpha,
        beta,
        sigma,
        rho,
      } => (
        Family::BatesJump.code(),
        pad([drift_c, alpha, beta, sigma, rho]),
      ),
      EulerSpec::BatesJumpReflected {
        drift_c,
        alpha,
        beta,
        sigma,
        rho,
      } => (
        Family::BatesJumpReflected.code(),
        pad([drift_c, alpha, beta, sigma, rho]),
      ),
      EulerSpec::AndersenQe {
        theta,
        e_kd,
        c1,
        c2,
        k0,
        k1,
        k2,
        k34,
        mu,
      } => (
        Family::AndersenQe.code(),
        pad([theta, e_kd, c1, c2, k0, k1, k2, k34, mu]),
      ),
      EulerSpec::CountingProcess => (Family::CountingProcess.code(), pad([])),
      EulerSpec::InverseGaussianSubordinator {
        mu_ig,
        two_lam,
        four_mu_lam,
      } => (
        Family::InverseGaussianSubordinator.code(),
        pad([mu_ig, two_lam, four_mu_lam]),
      ),
      EulerSpec::NormalInverseGaussian {
        theta,
        sigma,
        mu_ig,
        two_lam,
        four_mu_lam,
      } => (
        Family::NormalInverseGaussian.code(),
        pad([theta, sigma, mu_ig, two_lam, four_mu_lam]),
      ),
      EulerSpec::StableSubordinator {
        alpha,
        inv_alpha,
        one_minus_alpha,
        tail_exp,
        scale,
        pi,
      } => (
        Family::StableSubordinator.code(),
        pad([alpha, inv_alpha, one_minus_alpha, tail_exp, scale, pi]),
      ),
      EulerSpec::KouJumpHeston {
        drift_c,
        kappa,
        theta,
        sigma_v,
        rho,
      } => (
        Family::KouJumpHeston.code(),
        pad([drift_c, kappa, theta, sigma_v, rho]),
      ),
      EulerSpec::KouJumpHestonReflected {
        drift_c,
        kappa,
        theta,
        sigma_v,
        rho,
      } => (
        Family::KouJumpHestonReflected.code(),
        pad([drift_c, kappa, theta, sigma_v, rho]),
      ),
      EulerSpec::DuffieKanJump {
        a1,
        b1,
        c1,
        sigma1,
        a2,
        b2,
        c2,
        sigma2,
        alpha,
        beta,
        gamma,
        rho,
      } => (
        Family::DuffieKanJump.code(),
        pad([
          a1, b1, c1, sigma1, a2, b2, c2, sigma2, alpha, beta, gamma, rho,
        ]),
      ),
      EulerSpec::HawkesJumpDiffusion {
        drift_c,
        sigma,
        alpha,
        beta,
        mu_lambda,
        jump_mu,
        jump_sigma,
      } => (
        Family::HawkesJumpDiffusion.code(),
        pad([drift_c, sigma, alpha, beta, mu_lambda, jump_mu, jump_sigma]),
      ),
      EulerSpec::Garch { omega, alpha, beta } => (Family::Garch.code(), pad([omega, alpha, beta])),
      EulerSpec::ThresholdGarch {
        omega,
        alpha,
        gamma,
        beta,
      } => (
        Family::ThresholdGarch.code(),
        pad([omega, alpha, gamma, beta]),
      ),
      EulerSpec::ExponentialGarch {
        omega,
        alpha,
        gamma,
        beta,
        e_abs_z,
      } => (
        Family::ExponentialGarch.code(),
        pad([omega, alpha, gamma, beta, e_abs_z]),
      ),
      EulerSpec::Innovation { mean, sd } => (Family::Innovation.code(), pad([mean, sd])),
      EulerSpec::CorrelatedInnovation { rho } => (Family::CorrelatedInnovation.code(), pad([rho])),
      EulerSpec::Autoregressive { phi, sigma } => {
        (Family::Autoregressive.code(), pad([phi, sigma]))
      }
      EulerSpec::MovingAverage { theta, sigma } => {
        (Family::MovingAverage.code(), pad([theta, sigma]))
      }
      EulerSpec::GammaSubordinator => (Family::GammaSubordinator.code(), pad([])),
      EulerSpec::VarianceGamma { mu, sigma } => (Family::VarianceGamma.code(), pad([mu, sigma])),
      EulerSpec::BilateralGamma => (Family::BilateralGamma.code(), pad([])),
      EulerSpec::BilateralGammaMotion { sigma } => {
        (Family::BilateralGammaMotion.code(), pad([sigma]))
      }
      EulerSpec::TemperedStableSubordinator { drift } => {
        (Family::TemperedStableSubordinator.code(), pad([drift]))
      }
      EulerSpec::BarndorffNielsenShephard { decay, mu } => {
        (Family::BarndorffNielsenShephard.code(), pad([decay, mu]))
      }
      EulerSpec::CorrelatedFractionalMotion { rho } => {
        (Family::CorrelatedFractionalMotion.code(), pad([rho]))
      }
      EulerSpec::ComplexFractionalOu {
        lambda,
        omega,
        scale,
      } => (
        Family::ComplexFractionalOu.code(),
        pad([lambda, omega, scale]),
      ),
      EulerSpec::TransformedOrnsteinUhlenbeck {
        kappa,
        mu,
        sigma,
        arctan,
        half_pi,
      } => (
        Family::TransformedOrnsteinUhlenbeck.code(),
        pad([kappa, mu, sigma, arctan, half_pi]),
      ),
      EulerSpec::PoissonArrivals { lambda } => (Family::PoissonArrivals.code(), pad([lambda])),
      EulerSpec::DynamicSabr => (Family::DynamicSabr.code(), pad([])),
      EulerSpec::HeathJarrowMorton => (Family::HeathJarrowMorton.code(), pad([])),
      EulerSpec::AffineDiffusionGaussian { sigma } => {
        (Family::AffineDiffusionGaussian.code(), pad([sigma]))
      }
      EulerSpec::WuZhang {
        alpha,
        beta,
        nu,
        lambda,
      } => (Family::WuZhang.code(), pad([alpha, beta, nu, lambda])),
      EulerSpec::CorrelatedGeometric4 { mu, sigma, l } => {
        let mut values = [T::zero(); 18];
        values[..4].copy_from_slice(&mu);
        values[4..8].copy_from_slice(&sigma);
        values[8..].copy_from_slice(&l);
        (Family::CorrelatedGeometric4.code(), pad(values))
      }
      EulerSpec::CorrelatedNoises4 { l } => (Family::CorrelatedNoises4.code(), pad(l)),
    }
  }
}

/// A process the device kernels can run: its coefficients, initial value,
/// grid, horizon and the seed the launch derives from the process's seed
/// source.
pub trait EulerCoefficients<T: FloatExt>: ProcessExt<T, Output = Array1<T>> {
  fn euler_spec(&self) -> EulerSpec<T>;

  /// The value the reported path starts from.
  fn initial_value(&self) -> T;

  /// The state each path starts from, in the engine's four slots. The
  /// default puts [`initial_value`](Self::initial_value) in slot zero, which
  /// is what a one-component family needs; a process whose family carries
  /// further components — a sum of two factors reported as one path, say —
  /// overrides this.
  fn initial_state(&self) -> [T; 4] {
    [self.initial_value(), T::zero(), T::zero(), T::zero()]
  }
  /// Number of grid points including `t = 0`.
  fn grid_points(&self) -> usize;
  fn horizon(&self) -> T;
  /// One draw from the process's seed source: reproducible for
  /// `Deterministic`, fresh entropy for `Unseeded`.
  fn device_seed(&self) -> u64;

  /// The time step the recursion advances by. The default is the grid's own
  /// spacing, `horizon / (grid_points - 1)`; a process whose host sampler
  /// divides the horizon differently states that here, so the device
  /// reproduces that process's law rather than a neighbouring one.
  fn time_step(&self) -> T {
    self.horizon() / T::from_usize_(self.grid_points().max(2) - 1)
  }

  /// A time-varying coefficient, one value per grid point, or `None` when the
  /// family reads none. It reaches the step as `ct`: a short-rate model's
  /// `θ(t)`, a term structure of volatilities, anything the host can tabulate
  /// on the same grid the recursion walks.
  ///
  /// This is the one-curve convenience. A family that reads several — a
  /// dynamic-SABR term structure, a Heath-Jarrow-Morton coefficient set —
  /// overrides [`curves`](Self::curves) instead, which is what the engine
  /// actually reads.
  fn curve(&self) -> Option<Vec<T>> {
    None
  }

  /// Every time-varying coefficient the family reads, one `Vec` per curve and
  /// one value per grid point, in the order the step names them: `ct`, then
  /// `ct1` through `ct7`. Defaults to lifting [`curve`](Self::curve), so a
  /// one-curve process implements that and nothing else; override this one
  /// instead — never both — when the step reads more than one.
  ///
  /// At most [`CURVE_SLOTS`] curves, since that is what the kernels bind.
  fn curves(&self) -> Option<Vec<Vec<T>>> {
    self.curve().map(|c| vec![c])
  }

  /// The jump intensity per unit time, or `None` when the family has no jump
  /// term. The kernel draws a Poisson count with mean `intensity · dt` once
  /// per step and offers it to the step as `nj`.
  fn jump_intensity(&self) -> Option<T> {
    None
  }

  /// How big each jump is, or `None` when the family reads only the count.
  /// The step sees the sum as `js`.
  fn jump_sizes(&self) -> Option<JumpSizes<T>> {
    None
  }

  /// Whether the first point written is a step rather than the initial state.
  /// A conditional-variance model's series starts at `σ₀ z₀`, so its first
  /// point is a draw; every diffusion here starts at a level, which is the
  /// default.
  fn step_first(&self) -> bool {
    false
  }

  /// The Gamma draws the step reads as `gm` and `gm2`, or `None` when it
  /// reads none.
  fn gamma_draws(&self) -> Option<GammaDraws<T>> {
    None
  }

  /// One path from the process's own sampler, the host stream.
  fn host_sample(&self) -> Array1<T>;

  /// The fGN pipeline's inputs when this process's increments come from one,
  /// or `None` to let the kernel hash its own Gaussian increments from
  /// `(path, step, seed)`.
  ///
  /// A device runs the pipeline itself and keeps the increments in the buffer
  /// it wrote them to, so they never travel through host memory between the
  /// two kernels. That is why this reports the pipeline's inputs rather than
  /// the increments: handing over an array would be the round trip.
  fn fgn_spec(&self) -> Option<FgnSpec<'_, T>> {
    None
  }
}

/// What a device needs to run the fGN pipeline for a process whose increments
/// are fractional: the precomputed circulant eigenvalues and the grid they
/// were built for.
pub struct FgnSpec<'a, T> {
  /// Square roots of the circulant embedding's eigenvalues.
  pub sqrt_eigenvalues: &'a [T],
  /// The padded grid the eigenvalues belong to.
  pub n: usize,
  /// How many leading samples the pipeline drops.
  pub offset: usize,
  /// The Hurst exponent.
  pub hurst: f64,
  /// The horizon the increments are scaled to.
  pub t: f64,
  /// How many independent fGN streams the family's noise components read.
  /// One feeds `noise[0]`; a second feeds `noise[1]`, which is what a
  /// correlated fractional pair needs — its two rows share a Hurst
  /// exponent, so they share this embedding and the device draws
  /// `streams * m` paths in the one batched call rather than running the
  /// pipeline twice.
  pub streams: usize,
}

/// The device primitive of the Euler engine: one launch under one seed.
/// Implement it for a device handle and [`EulerBackend`] follows through
/// `kernel_euler_backend!`; the host handles implement [`EulerBackend`]
/// directly.
pub trait EulerKernel<T: FloatExt>: Backend {
  /// Paths `first .. first + m` of the launch stream seeded by `seed`, as an
  /// `m × n` matrix whose column 0 is the initial value. The kernels hash
  /// `(first + path, step, seed)`, so a batch produced in chunks under one
  /// seed is bit-identical to one launch of the whole batch.
  fn euler_kernel<P: EulerCoefficients<T>>(
    &self,
    process: &P,
    first: usize,
    m: usize,
    seed: u64,
  ) -> Result<Array2<T>, DeviceError>;

  /// Paths `first .. first + m` of a system's launch stream, as a
  /// `components × m × n` array.
  fn euler_system_kernel<const D: usize, P: EulerSystem<T, D>>(
    &self,
    process: &P,
    first: usize,
    m: usize,
    seed: u64,
  ) -> Result<Array3<T>, DeviceError>;

  /// Bytes of path data one launch may hold.
  fn batch_budget(&self) -> usize;

  /// The whole system batch under `seed`, chunked to the budget.
  fn euler_system_batch<const D: usize, P: EulerSystem<T, D>>(
    &self,
    process: &P,
    m: usize,
    seed: u64,
  ) -> Result<Array3<T>, DeviceError> {
    let n = process.grid_points();
    let rows = crate::device::chunk_rows(self.batch_budget(), n * D, std::mem::size_of::<T>());
    if m <= rows {
      return self.euler_system_kernel(process, 0, m, seed);
    }
    let mut out = Array3::<T>::zeros((D, m, n));
    let mut first = 0;
    while first < m {
      let len = rows.min(m - first);
      let chunk = self.euler_system_kernel(process, first, len, seed)?;
      out
        .slice_mut(ndarray::s![.., first..first + len, ..])
        .assign(&chunk);
      first += len;
    }
    Ok(out)
  }

  /// The whole batch under `seed`, chunked to the budget. A device may
  /// override it to pipeline the chunks; the result must stay bit-identical.
  fn euler_kernel_batch<P: EulerCoefficients<T>>(
    &self,
    process: &P,
    m: usize,
    seed: u64,
  ) -> Result<Array2<T>, DeviceError> {
    let n = process.grid_points();
    let rows = crate::device::chunk_rows(self.batch_budget(), n, std::mem::size_of::<T>());
    if m <= rows {
      return self.euler_kernel(process, 0, m, seed);
    }
    let mut out = Array2::<T>::zeros((m, n));
    let mut first = 0;
    while first < m {
      let len = rows.min(m - first);
      let chunk = self.euler_kernel(process, first, len, seed)?;
      out
        .slice_mut(ndarray::s![first..first + len, ..])
        .assign(&chunk);
      first += len;
    }
    Ok(out)
  }
}

/// How a backend handle produces Euler paths for the processes it serves:
/// the CPU handles run the process's own sampler, a device handle runs its
/// [`EulerKernel`]. The `try_*` methods report a device failure as a
/// [`DeviceError`]; the plain ones panic with it.
/// How big a jump is, when a family declares jumps that carry a size.
///
/// The normal case aggregates exactly: the sum of `n` normal jump sizes is
/// itself normal with mean `n·mean` and standard deviation `sd·√n`, so the
/// kernel draws it once however many jumps the step saw. The
/// double-exponential case has no such aggregation, so the kernel sums the
/// sizes in a bounded loop.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum JumpSizes<T: FloatExt> {
  /// Normal jump sizes, as Merton's and Bates's models take them.
  Normal { mean: T, sd: T },
  /// Kou's double-exponential sizes: up with probability `p_up` at rate
  /// `eta_up`, down otherwise at rate `eta_down`.
  DoubleExponential { p_up: T, eta_up: T, eta_down: T },
  /// A tempered-stable subordinator's sizes: a candidate `ε u^{−1/α}` kept
  /// with probability `exp(−μ·candidate)`, so the sum is over the accepted
  /// ones. `neg_inv_alpha` is `−1/α`, which depends on `α` alone.
  TemperedStable { eps: T, neg_inv_alpha: T, mu: T },
}

impl<T: FloatExt> JumpSizes<T> {
  /// The law code and the three scalars the kernels read it through. Public
  /// because [`EulerKernel`] is: a device implemented outside this crate has
  /// to pass the same five values its launch takes.
  pub fn encode(&self) -> (u32, T, T, T) {
    match *self {
      JumpSizes::Normal { mean, sd } => (1, mean, sd, T::zero()),
      JumpSizes::DoubleExponential {
        p_up,
        eta_up,
        eta_down,
      } => (2, p_up, eta_up, eta_down),
      JumpSizes::TemperedStable {
        eps,
        neg_inv_alpha,
        mu,
      } => (3, eps, neg_inv_alpha, mu),
    }
  }
}

/// The Gamma draws a family reads as `gm` and `gm2`, when its increment is a
/// gamma variate rather than a Gaussian one.
///
/// Each is one Marsaglia-Tsang draw of `Gamma(shape, scale)`, with the
/// shape-below-one boost that method prescribes. A process that needs one
/// draw declares [`second`](Self::second) as `None`; the bilateral gamma
/// processes, whose increment is the difference of two, declare both.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GammaDraws<T: FloatExt> {
  /// The shape, scale and per-jump shape of `gm`. The shape the kernel uses
  /// is `shape + per_jump · nj`, which is what a compound sum of gamma jumps
  /// needs: the sum of `k` of them is one draw whose shape is `k` times the
  /// single jump's. A process whose shape does not depend on the jump count
  /// leaves the third term at zero.
  pub first: (T, T, T),
  /// The same three for `gm2`, when the family reads a second draw.
  pub second: Option<(T, T, T)>,
}

impl<T: FloatExt> GammaDraws<T> {
  /// How many draws to take, and the two shape/scale/per-jump triples the
  /// kernels read. Public for the same reason [`JumpSizes::encode`] is.
  pub fn encode(&self) -> (u32, T, T, T, T, T, T) {
    let (s1, c1, p1) = self.first;
    match self.second {
      Some((s2, c2, p2)) => (2, s1, c1, p1, s2, c2, p2),
      None => (1, s1, c1, p1, T::zero(), T::zero(), T::zero()),
    }
  }
}

/// A process whose device recursion carries more than one state component:
/// the same families, the same kernels and the same launch, with `D` arrays
/// back instead of one.
///
/// The engine's kernels always carry four state and four noise slots, so a
/// system costs nothing beyond the components it declares. What a system adds
/// over [`EulerCoefficients`] is only the shape of the answer: `D` paths per
/// draw rather than one, which is what a stochastic-volatility or two-factor
/// model returns.
pub trait EulerSystem<T: FloatExt, const D: usize>: ProcessExt<T, Output = [Array1<T>; D]> {
  /// The family this system steps. Its component count must be `D`.
  fn euler_spec(&self) -> EulerSpec<T>;

  /// The state each path starts from, in the engine's four slots. A family
  /// that steps fewer components leaves the rest at zero.
  ///
  /// The slots are the family's own state, which may be larger than the `D`
  /// paths the process returns: a model that carries an accumulator steps it
  /// alongside the components it reports. The reported ones come first.
  fn initial_state(&self) -> [T; 4];

  /// Number of grid points including `t = 0`.
  fn grid_points(&self) -> usize;

  fn horizon(&self) -> T;

  /// One draw from the process's seed source: reproducible for
  /// `Deterministic`, fresh entropy for `Unseeded`.
  fn device_seed(&self) -> u64;

  /// The time step the recursion advances by. The default is the grid's own
  /// spacing, `horizon / (grid_points - 1)`; a process whose host sampler
  /// divides the horizon differently states that here, so the device
  /// reproduces that process's law rather than a neighbouring one.
  fn time_step(&self) -> T {
    self.horizon() / T::from_usize_(self.grid_points().max(2) - 1)
  }

  /// A time-varying coefficient, one value per grid point, or `None` when the
  /// family reads none. It reaches the step as `ct`: a short-rate model's
  /// `θ(t)`, a term structure of volatilities, anything the host can tabulate
  /// on the same grid the recursion walks.
  ///
  /// This is the one-curve convenience. A family that reads several — a
  /// dynamic-SABR term structure, a Heath-Jarrow-Morton coefficient set —
  /// overrides [`curves`](Self::curves) instead, which is what the engine
  /// actually reads.
  fn curve(&self) -> Option<Vec<T>> {
    None
  }

  /// Every time-varying coefficient the family reads, one `Vec` per curve and
  /// one value per grid point, in the order the step names them: `ct`, then
  /// `ct1` through `ct7`. Defaults to lifting [`curve`](Self::curve), so a
  /// one-curve process implements that and nothing else; override this one
  /// instead — never both — when the step reads more than one.
  ///
  /// At most [`CURVE_SLOTS`] curves, since that is what the kernels bind.
  fn curves(&self) -> Option<Vec<Vec<T>>> {
    self.curve().map(|c| vec![c])
  }

  /// The jump intensity per unit time, or `None` when the family has no jump
  /// term. The kernel draws a Poisson count with mean `intensity · dt` once
  /// per step and offers it to the step as `nj`.
  fn jump_intensity(&self) -> Option<T> {
    None
  }

  /// How big each jump is, or `None` when the family reads only the count.
  /// The step sees the sum as `js`.
  fn jump_sizes(&self) -> Option<JumpSizes<T>> {
    None
  }

  /// Whether the first point written is a step rather than the initial state.
  /// A conditional-variance model's series starts at `σ₀ z₀`, so its first
  /// point is a draw; every diffusion here starts at a level, which is the
  /// default.
  fn step_first(&self) -> bool {
    false
  }

  /// The Gamma draws the step reads as `gm` and `gm2`, or `None` when it
  /// reads none.
  fn gamma_draws(&self) -> Option<GammaDraws<T>> {
    None
  }

  /// The fGN pipeline's inputs when this system's noise components come from
  /// one, or `None` to let the kernel hash its own Gaussian increments. The
  /// same contract as [`EulerCoefficients::fgn_spec`], with
  /// [`FgnSpec::streams`] saying how many components the pipeline fills.
  fn fgn_spec(&self) -> Option<FgnSpec<'_, T>> {
    None
  }

  /// One draw from the process's own sampler, the host stream.
  fn host_sample(&self) -> [Array1<T>; D];
}

/// The arity check that a system's `D` paths are components its family
/// actually steps. A mismatch is a declaration error rather than a runtime
/// condition, so it fails loudly here instead of silently returning planes
/// the kernel never wrote.
///
/// Public because [`EulerKernel`] is: a device implemented outside this crate
/// needs it, and [`system_row`], to satisfy `euler_system_kernel`.
pub fn check_arity<T: FloatExt>(spec: &EulerSpec<T>, d: usize) {
  let (code, _) = spec.encode();
  let family = families::Family::from_code(code).expect("a declared family");
  assert!(
    family.components() >= d,
    "{family:?} steps {} components but the process returns {d} arrays",
    family.components()
  );
}

/// The `D` paths of one launch row, taken out of the `components × m × n`
/// array a kernel returns. Public for the same reason [`check_arity`] is.
pub fn system_row<T: FloatExt, const D: usize>(planes: &Array3<T>, row: usize) -> [Array1<T>; D] {
  std::array::from_fn(|c| planes.slice(ndarray::s![c, row, ..]).to_owned())
}

pub trait EulerBackend<T: FloatExt>: Backend {
  /// One path.
  fn try_sample<P: EulerCoefficients<T>>(&self, process: &P) -> Result<Array1<T>, DeviceError>;

  /// `m` paths.
  fn try_euler_paths<P: EulerCoefficients<T>>(
    &self,
    process: &P,
    m: usize,
  ) -> Result<Vec<Array1<T>>, DeviceError>;

  /// `f` over `m` paths, mapped as they are produced, so the batch never has
  /// to fit in memory at once.
  fn try_euler_paths_map<P: EulerCoefficients<T>, R: Send>(
    &self,
    process: &P,
    m: usize,
    f: impl Fn(&Array1<T>) -> R + Sync,
  ) -> Result<Vec<R>, DeviceError>;

  /// The batch as one `m × n` matrix; on a device the launch buffer itself.
  fn try_euler_matrix<P: EulerCoefficients<T>>(
    &self,
    process: &P,
    m: usize,
  ) -> Result<Array2<T>, DeviceError>;

  /// [`try_sample`](Self::try_sample), panicking with the device's error.
  fn euler_sample<P: EulerCoefficients<T>>(&self, process: &P) -> Array1<T> {
    self
      .try_sample(process)
      .unwrap_or_else(crate::device::device_panic)
  }

  /// [`try_euler_paths`](Self::try_euler_paths), panicking with the device's
  /// error; [`Backend::probe`] first turns that failure into a `Result`.
  fn euler_paths<P: EulerCoefficients<T>>(&self, process: &P, m: usize) -> Vec<Array1<T>> {
    self
      .try_euler_paths(process, m)
      .unwrap_or_else(crate::device::device_panic)
  }

  /// [`try_euler_paths_map`](Self::try_euler_paths_map), panicking with the
  /// device's error.
  fn euler_paths_map<P: EulerCoefficients<T>, R: Send>(
    &self,
    process: &P,
    m: usize,
    f: impl Fn(&Array1<T>) -> R + Sync,
  ) -> Vec<R> {
    self
      .try_euler_paths_map(process, m, f)
      .unwrap_or_else(crate::device::device_panic)
  }

  /// One draw of a multi-component system.
  fn try_system_sample<const D: usize, P: EulerSystem<T, D>>(
    &self,
    process: &P,
  ) -> Result<[Array1<T>; D], DeviceError>;

  /// `m` draws of a multi-component system.
  fn try_system_paths<const D: usize, P: EulerSystem<T, D>>(
    &self,
    process: &P,
    m: usize,
  ) -> Result<Vec<[Array1<T>; D]>, DeviceError>;

  /// `f` over `m` draws of a system, mapped as they are produced.
  fn try_system_paths_map<const D: usize, P: EulerSystem<T, D>, R: Send>(
    &self,
    process: &P,
    m: usize,
    f: impl Fn(&[Array1<T>; D]) -> R + Sync,
  ) -> Result<Vec<R>, DeviceError>;

  /// [`try_system_sample`](Self::try_system_sample), panicking with the
  /// device's error.
  fn system_sample<const D: usize, P: EulerSystem<T, D>>(&self, process: &P) -> [Array1<T>; D] {
    self
      .try_system_sample(process)
      .unwrap_or_else(crate::device::device_panic)
  }

  /// [`try_system_paths`](Self::try_system_paths), panicking with the
  /// device's error.
  fn system_paths<const D: usize, P: EulerSystem<T, D>>(
    &self,
    process: &P,
    m: usize,
  ) -> Vec<[Array1<T>; D]> {
    self
      .try_system_paths(process, m)
      .unwrap_or_else(crate::device::device_panic)
  }

  /// [`try_system_paths_map`](Self::try_system_paths_map), panicking with the
  /// device's error.
  fn system_paths_map<const D: usize, P: EulerSystem<T, D>, R: Send>(
    &self,
    process: &P,
    m: usize,
    f: impl Fn(&[Array1<T>; D]) -> R + Sync,
  ) -> Vec<R> {
    self
      .try_system_paths_map(process, m, f)
      .unwrap_or_else(crate::device::device_panic)
  }
}

/// A host handle samples through the process's own sampler, chunked the way
/// `ProcessExt` chunks, so its streams are those of the process.
macro_rules! host_euler_backend {
  ($handle:ty) => {
    impl<T: FloatExt> EulerBackend<T> for $handle {
      fn try_sample<P: EulerCoefficients<T>>(&self, process: &P) -> Result<Array1<T>, DeviceError> {
        Ok(process.host_sample())
      }

      fn try_euler_paths<P: EulerCoefficients<T>>(
        &self,
        process: &P,
        m: usize,
      ) -> Result<Vec<Array1<T>>, DeviceError> {
        Ok(sample_par_chunked(process, m))
      }

      fn try_euler_paths_map<P: EulerCoefficients<T>, R: Send>(
        &self,
        process: &P,
        m: usize,
        f: impl Fn(&Array1<T>) -> R + Sync,
      ) -> Result<Vec<R>, DeviceError> {
        Ok(sample_map_chunked(process, m, f))
      }

      fn try_euler_matrix<P: EulerCoefficients<T>>(
        &self,
        process: &P,
        m: usize,
      ) -> Result<Array2<T>, DeviceError> {
        let rows = sample_par_chunked(process, m);
        let n = rows.first().map_or(process.grid_points(), |r| r.len());
        let mut out = Array2::<T>::zeros((m, n));
        for (i, row) in rows.iter().enumerate() {
          out.row_mut(i).assign(row);
        }
        Ok(out)
      }

      fn try_system_sample<const D: usize, P: EulerSystem<T, D>>(
        &self,
        process: &P,
      ) -> Result<[Array1<T>; D], DeviceError> {
        Ok(process.host_sample())
      }

      fn try_system_paths<const D: usize, P: EulerSystem<T, D>>(
        &self,
        process: &P,
        m: usize,
      ) -> Result<Vec<[Array1<T>; D]>, DeviceError> {
        Ok(sample_par_chunked(process, m))
      }

      fn try_system_paths_map<const D: usize, P: EulerSystem<T, D>, R: Send>(
        &self,
        process: &P,
        m: usize,
        f: impl Fn(&[Array1<T>; D]) -> R + Sync,
      ) -> Result<Vec<R>, DeviceError> {
        Ok(sample_map_chunked(process, m, f))
      }
    }
  };
}

host_euler_backend!(Cpu);
#[cfg(feature = "accelerate")]
host_euler_backend!(crate::device::Accelerate);

/// A device kernel is an Euler backend: one seed per call, chunks to the
/// handle's budget, the map applied per chunk in parallel. One impl per
/// handle rather than a blanket one, which coherence would not allow beside
/// the host impls above.
#[cfg(any(feature = "cuda", feature = "metal", feature = "cubecl"))]
macro_rules! kernel_euler_backend {
  ($handle:ty, [$($gen:tt)*] $scalar:ty) => {
    impl<$($gen)*> EulerBackend<$scalar> for $handle {
    fn try_sample<P: EulerCoefficients<$scalar>>(&self, process: &P) -> Result<Array1<$scalar>, DeviceError> {
      let seed = process.device_seed();
      Ok(<Self as EulerKernel<$scalar>>::euler_kernel(self, process, 0, 1, seed)?.row(0).to_owned())
    }

    fn try_euler_paths<P: EulerCoefficients<$scalar>>(
      &self,
      process: &P,
      m: usize,
    ) -> Result<Vec<Array1<$scalar>>, DeviceError> {
      let seed = process.device_seed();
      Ok(
        <Self as EulerKernel<$scalar>>::euler_kernel_batch(self, process, m, seed)?
          .outer_iter()
          .map(|row| row.to_owned())
          .collect(),
      )
    }

    fn try_euler_paths_map<P: EulerCoefficients<$scalar>, R: Send>(
      &self,
      process: &P,
      m: usize,
      f: impl Fn(&Array1<$scalar>) -> R + Sync,
    ) -> Result<Vec<R>, DeviceError> {
      use rayon::prelude::*;
      let seed = process.device_seed();
      let rows = crate::device::chunk_rows(
        <Self as EulerKernel<$scalar>>::batch_budget(self),
        process.grid_points(),
        std::mem::size_of::<$scalar>(),
      );
      let mut out = Vec::with_capacity(m);
      let mut first = 0;
      while first < m {
        let len = rows.min(m - first);
        let chunk: Vec<Array1<$scalar>> = <Self as EulerKernel<$scalar>>::euler_kernel(self, process, first, len, seed)?
          .outer_iter()
          .map(|row| row.to_owned())
          .collect();
        out.extend(chunk.par_iter().map(&f).collect::<Vec<R>>());
        first += len;
      }
      Ok(out)
    }

    fn try_euler_matrix<P: EulerCoefficients<$scalar>>(
      &self,
      process: &P,
      m: usize,
    ) -> Result<Array2<$scalar>, DeviceError> {
      <Self as EulerKernel<$scalar>>::euler_kernel_batch(self, process, m, process.device_seed())
    }

    fn try_system_sample<const D: usize, P: EulerSystem<$scalar, D>>(
      &self,
      process: &P,
    ) -> Result<[Array1<$scalar>; D], DeviceError> {
      let seed = process.device_seed();
      let planes = <Self as EulerKernel<$scalar>>::euler_system_kernel(self, process, 0, 1, seed)?;
      Ok(system_row(&planes, 0))
    }

    fn try_system_paths<const D: usize, P: EulerSystem<$scalar, D>>(
      &self,
      process: &P,
      m: usize,
    ) -> Result<Vec<[Array1<$scalar>; D]>, DeviceError> {
      let seed = process.device_seed();
      let planes = <Self as EulerKernel<$scalar>>::euler_system_batch(self, process, m, seed)?;
      Ok((0..m).map(|row| system_row(&planes, row)).collect())
    }

    fn try_system_paths_map<const D: usize, P: EulerSystem<$scalar, D>, R: Send>(
      &self,
      process: &P,
      m: usize,
      f: impl Fn(&[Array1<$scalar>; D]) -> R + Sync,
    ) -> Result<Vec<R>, DeviceError> {
      use rayon::prelude::*;
      let seed = process.device_seed();
      let rows = crate::device::chunk_rows(
        <Self as EulerKernel<$scalar>>::batch_budget(self),
        process.grid_points() * D,
        std::mem::size_of::<$scalar>(),
      );
      let mut out = Vec::with_capacity(m);
      let mut first = 0;
      while first < m {
        let len = rows.min(m - first);
        let planes =
          <Self as EulerKernel<$scalar>>::euler_system_kernel(self, process, first, len, seed)?;
        let chunk: Vec<[Array1<$scalar>; D]> =
          (0..len).map(|row| system_row(&planes, row)).collect();
        out.extend(chunk.par_iter().map(&f).collect::<Vec<R>>());
        first += len;
      }
      Ok(out)
    }
    }
  };
}

#[cfg(feature = "metal")]
kernel_euler_backend!(crate::device::Metal, [] f32);
#[cfg(any(feature = "cubecl-cuda", feature = "cubecl-wgpu"))]
kernel_euler_backend!(crate::device::Cubecl<Rt>, [Rt: crate::euler::cubecl::CubeclRuntime] f32);
#[cfg(feature = "cuda")]
kernel_euler_backend!(crate::device::Cuda, [T: FloatExt] T);

/// One launch seed from a process's own seed source: reproducible for
/// `Deterministic`, fresh entropy for `Unseeded`, and advancing either way so
/// two launches from one process do not replay.
///
/// This is [`SeedExt::seed_value`] and nothing else — the workspace draws its
/// randomness from its own generator, never from `rand`. Public because
/// `EulerCoefficients` is: an out-of-tree process needs it to answer
/// `device_seed`.
pub fn draw_seed<S: SeedExt>(seed: &S) -> u64 {
  seed.seed_value()
}

impl<T: FloatExt, S: SeedExt, B: EulerBackend<T>> EulerCoefficients<T> for Gbm<T, S, B> {
  fn euler_spec(&self) -> EulerSpec<T> {
    EulerSpec::GeometricBrownian {
      mu: self.mu,
      sigma: self.sigma,
    }
  }

  fn initial_value(&self) -> T {
    self.x0.unwrap_or(T::one())
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  fn device_seed(&self) -> u64 {
    draw_seed(&self.seed)
  }

  fn host_sample(&self) -> Array1<T> {
    let out = self.sampler().sample();
    self.advance_chunk_seed();
    out
  }
}

impl<T: FloatExt, S: SeedExt, B: EulerBackend<T>> EulerCoefficients<T> for Ou<T, S, B> {
  fn euler_spec(&self) -> EulerSpec<T> {
    EulerSpec::OrnsteinUhlenbeck {
      theta: self.theta,
      mu: self.mu,
      sigma: self.sigma,
    }
  }

  fn initial_value(&self) -> T {
    self.x0.unwrap_or(T::zero())
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  fn device_seed(&self) -> u64 {
    draw_seed(&self.seed)
  }

  fn host_sample(&self) -> Array1<T> {
    let out = self.sampler().sample();
    self.advance_chunk_seed();
    out
  }
}

impl<T: FloatExt, S: SeedExt, B: EulerBackend<T>> EulerCoefficients<T> for Cir<T, S, B> {
  fn euler_spec(&self) -> EulerSpec<T> {
    EulerSpec::SquareRoot {
      kappa: self.theta,
      theta: self.mu,
      sigma: self.sigma,
    }
  }

  fn initial_value(&self) -> T {
    self.x0.unwrap_or(T::zero())
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  fn device_seed(&self) -> u64 {
    draw_seed(&self.seed)
  }

  fn host_sample(&self) -> Array1<T> {
    let out = self.sampler().sample();
    self.advance_chunk_seed();
    out
  }
}

macro_rules! try_sample_matrix {
  ($ty:ident) => {
    impl<T: FloatExt, S: SeedExt, B: EulerBackend<T>> $ty<T, S, B> {
      /// The batch as one `m × n` matrix: on a device back-end the launch
      /// buffer itself, without a re-layout into rows. The row form is
      /// [`ProcessExt::try_sample_par`], the single path
      /// [`ProcessExt::try_sample`].
      pub fn try_sample_matrix(&self, m: usize) -> Result<Array2<T>, DeviceError> {
        self.backend.try_euler_matrix(self, m)
      }
    }
  };
}

try_sample_matrix!(Gbm);
try_sample_matrix!(Ou);
try_sample_matrix!(Cir);

#[cfg(any(feature = "cubecl-cuda", feature = "cubecl-wgpu"))]
pub mod cubecl;
#[cfg(feature = "cuda")]
pub mod cuda;
// The generated C artifacts have no consumer until `cuda` or `metal` renders
// a kernel from them; the declarations, the family codes and the host step
// stay compiled either way, so a family is checked without a GPU.
#[cfg_attr(not(any(feature = "cuda", feature = "metal")), allow(dead_code))]
pub(crate) mod families;
#[cfg(any(feature = "cuda", feature = "metal"))]
pub(crate) mod kernel;
#[cfg(feature = "metal")]
pub mod metal;

#[cfg(test)]
mod family_parity;
#[cfg(test)]
mod tests;

/// A single-precision device refuses an `f64` process at compile time.
///
/// ```compile_fail,E0277
/// use stochastic_rs_core::simd_rng::Unseeded;
/// use stochastic_rs_stochastic::device::Metal;
/// use stochastic_rs_stochastic::diffusion::gbm::Gbm;
/// use stochastic_rs_stochastic::traits::ProcessExt;
///
/// let gbm = Gbm::<f64, _>::new(0.05, 0.2, 16, None, None, Unseeded);
/// let _ = gbm.on::<Metal>().sample();
/// ```
#[cfg(feature = "metal")]
pub mod precision_guard {}

#[cfg(feature = "python")]
pub mod python {
  //! Python surface of the device layer: probing a device and choosing the
  //! ordinal. Sampling on a device goes through the process classes'
  //! `device=` argument.

  use pyo3::prelude::*;

  /// Opens the named device (`"cpu"`, `"accelerate"`, `"cuda"`, `"metal"`,
  /// `"cubecl-cuda"`, `"cubecl-wgpu"`, optionally with `:ordinal`) and describes it
  /// as a dict with `backend`, `name`, `precisions` and `ordinal`; raises
  /// `RuntimeError` with the device's own message when it cannot be used,
  /// `ValueError` for a device this build does not carry.
  #[pyfunction]
  pub fn probe_device<'py>(
    py: Python<'py>,
    device: &str,
  ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let info = crate::python_device::Device::parse_name(device)?.probe()?;
    let d = pyo3::types::PyDict::new(py);
    d.set_item("backend", info.backend)?;
    d.set_item("name", info.name)?;
    d.set_item("precisions", info.precisions.to_vec())?;
    d.set_item("ordinal", info.ordinal)?;
    Ok(d)
  }
}

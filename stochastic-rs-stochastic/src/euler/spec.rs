use super::*;

mod encode;

/// Scalar drift / diffusion families the device kernels know how to step.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
#[doc(hidden)]
pub enum EulerSpec<T: FloatExt> {
  /// `dX = μX dt + σX dW`.
  GeometricBrownian { mu: T, sigma: T },
  /// `dX = θ(μ − X) dt + σ dW`.
  OrnsteinUhlenbeck { theta: T, mu: T, sigma: T },
  /// `dX = κ(θ − X) dt + σ√X dW` with full truncation (Lord, Koekkoek & van Dijk 2010): an
  /// auxiliary process whose positive part enters drift, diffusion and the reported path.
  SquareRoot { kappa: T, theta: T, sigma: T },
  /// `dX = dW`: the increment accumulates. With fractional increments this is
  /// fractional Brownian motion.
  Additive,
  /// `dX = θ(μ − X) dt + σ√|X| dW`, clamped at zero after the step: the fractional CIR
  /// recursion, which truncates the result, not the state it steps on.
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
  /// `dX = κ(θ − X)X dt + σ√X dW`, absorbed at zero.
  FellerLogistic { kappa: T, theta: T, sigma: T },
  /// [`FellerLogistic`](EulerSpec::FellerLogistic) reflected at zero.
  FellerLogisticReflected { kappa: T, theta: T, sigma: T },
  /// `dX = δ dt + 2√|X| dW`, absorbed at zero.
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
  /// The Heston model's Euler scheme, variance absorbed at zero.
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
  /// The Heston model stepped in log-price, variance absorbed.
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
  /// Two variance factors driving one spot, both absorbed.
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
  LogMeanReverting { decay: T, pull: T, sigma_eff: T },
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
  /// Two Heston assets under one 4x4 Cholesky factor, variances absorbed.
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
  /// The Bates stochastic-volatility jump model, variance absorbed.
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
  /// A positive-stable subordinator by the Chambers-Mallows-Stuck transform. The scale is passed
  /// as `ln((c·dt)^{1/α})`: the step sums logs, and `(c·dt)^{1/α}` underflows for small `α`.
  StableSubordinator {
    alpha: T,
    inv_alpha: T,
    one_minus_alpha: T,
    tail_exp: T,
    log_scale: T,
    pi: T,
  },
  /// A Heston variance under a Kou-jumping log-price, variance absorbed.
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
  /// A log-normal spot under a Markov chain of up to four volatility regimes: one volatility
  /// per regime and, per transition-matrix row, three cumulative thresholds for the successor.
  RegimeSwitching {
    mu: T,
    sigma: [T; 4],
    thresholds: [[T; 3]; 4],
  },
  /// Riemann-Liouville fractional Brownian motion, or any rough Volterra
  /// kernel, under the Markov lift the launch's `lift_spec()` supplies.
  RiemannLiouville,
  /// A mean-reverting process driven by the increments of Riemann-Liouville
  /// fBm under the launch's Markov lift.
  RiemannLiouvilleOu { kappa: T, mu: T, nu: T },
  /// Black-Scholes under Riemann-Liouville fBm in closed form, the
  /// deterministic exponent tabulated as the curve `ct`.
  RiemannLiouvilleBlackScholes { s0: T, sigma: T },
  /// The rough Heston model: the variance under the launch's Markov lift, the
  /// spot by Euler, the two shocks correlated by `rho`.
  RiemannLiouvilleHeston {
    mu: T,
    kappa: T,
    theta: T,
    nu: T,
    rho: T,
  },
  /// A diffusion with additive compound-Poisson jumps, the drift handed over
  /// already compensated and multiplied by `dt`.
  AdditiveJumpDiffusion { drift_dt: T, sigma: T },
  /// Bates (1996) stepped by Euler in the spot, the jump multiplicative under
  /// a product size law, the variance absorbed at zero.
  Bates1996 {
    drift_c: T,
    alpha: T,
    beta: T,
    sigma: T,
    rho: T,
  },
  /// [`Bates1996`](EulerSpec::Bates1996) with the variance reflected.
  Bates1996Reflected {
    drift_c: T,
    alpha: T,
    beta: T,
    sigma: T,
    rho: T,
  },
  /// An event-indexed compound Poisson path (arrival times, cumulative jumps and the jumps),
  /// one arrival per step, the size under a one-per-step law.
  CompoundPoissonEvents { lambda: T },
  /// A fractional Ornstein-Uhlenbeck process with compound-Poisson jumps.
  JumpFractionalOu { theta: T, mu: T, sigma: T },
  /// A square-root Volterra process under the launch's Markov lift.
  VolterraSquareRoot { kappa: T, theta: T, nu: T },
  /// A polynomial of a Gaussian Volterra process under the launch's Markov
  /// lift, up to eight coefficients in rising order, the unused ones zero.
  GaussianPolynomialVolatility { coefficients: [T; 8] },
  /// Rough Heston in the two-term rational approximation of its kernel; the history block keeps
  /// the local factor's memory integral exactly, against the weights in the launch's first curve.
  RoughHestonMemory {
    mu: T,
    theta: T,
    ek: T,
    nu: T,
    c1: T,
    c2: T,
    inv_g: T,
    rho: T,
  },
  /// The fractional Bates model: [`RoughHestonMemory`](EulerSpec::RoughHestonMemory) with unit
  /// calibration coefficients, a jump-compensated drift and normal jumps in the log-spot.
  FractionalBatesMemory {
    mu_c: T,
    theta: T,
    ek: T,
    xi: T,
    inv_g: T,
    rho: T,
  },
  /// Rough Bergomi under the hybrid scheme: the Volterra driver is the history block's
  /// convolution (first curve) plus an independent residual; the exponent is the second curve.
  RoughBergomiMemory {
    r: T,
    a: T,
    v0sq: T,
    residual_sd: T,
    rho: T,
  },
  /// Linear fractional stable motion: a moving average of α-stable draws
  /// under the weights the launch tabulates as its first curve.
  LinearFractionalStable {
    alpha: T,
    inv_alpha: T,
    tail_exp: T,
    b: T,
    scale_s: T,
    pi: T,
    half_pi: T,
  },
  /// A moving-average filter of white noise whose weights — the impulse
  /// response of the host's linear recursion — travel as the first curve.
  MovingAverageFilter { sigma: T },
  /// A tempered-stable process by its shot-noise series. `rate` scales the arrival bound, the
  /// exponential enters as `e_scale E^{e_pow}`, and a side is positive with probability `w_plus`.
  TemperedStableSeries {
    b_t: T,
    rate: T,
    inv_alpha: T,
    e_scale: T,
    e_pow: T,
    w_plus: T,
    lambda_plus: T,
    lambda_minus: T,
    /// The diffusion that stands in for the terms past the last; their drift is in `b_t`.
    rem_sd: T,
  },
  /// A Gaussian Volterra process by the reference quadrature, the kernel on
  /// the grid travelling as the launch's first curve.
  VolterraReference,
  /// A Hawkes process with exponential excitation, one event per step, by the
  /// exact two-uniform recursion.
  HawkesEvents { mu: T, alpha: T, beta: T },
  /// A bivariate Hawkes process, one event per step, by superposing each target's exact excess
  /// clock and the baselines' Poisson clock. `alpha` is row-major, `α_ij` onto `i` from `j`.
  HawkesEvents2 {
    mu: [T; 2],
    alpha: [T; 4],
    beta: [T; 2],
  },
  /// The Cheyette quasi-Gaussian state `(x, y)` under a local volatility the
  /// launch's first program computes from `(t, x)`.
  CheyetteLocalVol { kappa: T },
  /// A stochastic Volterra equation under the Markov lift of its kernel, the
  /// drift and diffusion the launch's two programs compute from `(t, x)`.
  VolterraProgram,
  /// The inverse of an α-stable subordinator by the table block: the subordinator's
  /// Chambers–Mallows–Stuck constants, with the table's extent and resolution from `table_spec()`.
  InverseStableSubordinator {
    alpha: T,
    c: T,
    inv_alpha: T,
    one_minus_alpha: T,
    tail_exp: T,
    pi: T,
  },
  /// The Heston stochastic-local volatility model `(s, v)` under the leverage the launch's first
  /// program computes from `(t, s)`, the vol-of-vol already mixed by `η`; full truncation.
  HestonSlv {
    mu: T,
    kappa: T,
    theta: T,
    sigma: T,
    rho: T,
  },
  /// CGMY under an exactly stepped CIR variance, each term sized in its cell's step against the
  /// variance there. `rate0 = α / (2 C T)` is the arrival bound's rate at unit variance.
  StochasticVolatilityCgmy {
    rate0: T,
    inv_alpha: T,
    lambda_plus: T,
    lambda_minus: T,
    bcoef: T,
    twoc: T,
    ek: T,
    rho: T,
  },
  /// The two-dimensional Wishart process by its exact step: row-major `m = exp(Δt b)`, the
  /// extended Cholesky factor `θ` of `q / Δt` and its inverse, and whether the noise has rank two.
  WishartTwo {
    m: [T; 4],
    theta: [T; 4],
    theta_inv: [T; 4],
    two: T,
  },
  /// Up to four forward LIBOR rates under the spot measure's drift coupling: volatilities,
  /// accruals and the correlation's lower Cholesky factor; the reset index is the first curve.
  LiborMarket4 {
    sigma: [T; 4],
    delta: [T; 4],
    l: [T; 10],
  },
  /// `Heston` under full truncation, its parameters in the same order.
  HestonFullTruncation([T; 6]),
  /// `LogHeston` under full truncation, its parameters in the same order.
  LogHestonFullTruncation([T; 5]),
  /// `DoubleHeston` under full truncation, its parameters in the same order.
  DoubleHestonFullTruncation([T; 9]),
  /// `TwoAssetHeston` under full truncation, its parameters in the same order.
  TwoAssetHestonFullTruncation([T; 18]),
  /// `BatesJump` under full truncation, its parameters in the same order.
  BatesJumpFullTruncation([T; 5]),
  /// `Bates1996` under full truncation, its parameters in the same order.
  Bates1996FullTruncation([T; 5]),
  /// `KouJumpHeston` under full truncation, its parameters in the same order.
  KouJumpHestonFullTruncation([T; 5]),
  /// `ShiftedSquareRoot` under full truncation, its parameters in the same order.
  ShiftedSquareRootFullTruncation([T; 3]),
  /// `TwoFactorSquareRoot`'s six parameters, then a scheme per factor: `0` full truncation, `1`
  /// absorption, `2` reflection.
  TwoFactorSquareRootSchemes([T; 8]),
  /// `SquaredBesselState` under full truncation, its parameters in the same order.
  SquaredBesselFullTruncation([T; 2]),
  /// `BesselFromSquared` under full truncation, its parameters in the same order.
  BesselFromSquaredFullTruncation([T; 2]),
  /// Heston on two-factor Hull-White rates by Euler: `l` is the lower Cholesky factor of the
  /// `(x, v, r, u)` correlation, row-major; `theta_r` travels as the first curve.
  HestonHullWhite2F {
    kappa: T,
    theta: T,
    sigma: T,
    a: T,
    b: T,
    sigma1: T,
    sigma2: T,
    l: [T; 10],
    s0: T,
  },
  /// [`HestonHullWhite2F`](Self::HestonHullWhite2F) under the quadratic-exponential variance, its
  /// constants as [`AndersenQe`](Self::AndersenQe)'s; `l` is the `(v, r, u, x)` factor past `v`.
  HestonHullWhite2FQe {
    theta: T,
    e_kd: T,
    c1: T,
    c2: T,
    k0: T,
    k1: T,
    k2: T,
    a: T,
    b: T,
    sigma1: T,
    sigma2: T,
    l: [T; 6],
    s0: T,
  },
}

/// Widens a family's parameter list to the kernels' fixed slot count.
fn pad<T: FloatExt, const N: usize>(values: [T; N]) -> [T; PARAM_SLOTS] {
  let mut slots = [T::zero(); PARAM_SLOTS];
  slots[..N].copy_from_slice(&values);
  slots
}

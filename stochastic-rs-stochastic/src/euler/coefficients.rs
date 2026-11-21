use stochastic_rs_core::simd_rng::SeedExt;

use super::*;
use crate::params::VarianceBoundary;

/// A process the device kernels can run: its coefficients, initial value, grid, horizon and the
/// seed the launch derives from the process's seed source.
#[doc(hidden)]
pub trait EulerCoefficients<T: FloatExt>: ProcessExt<T, Output = Array1<T>> {
  fn euler_spec(&self) -> EulerSpec<T>;

  /// The value the reported path starts from.
  fn initial_value(&self) -> T;

  /// The state each path starts from, in the engine's four slots. The default puts
  /// `initial_value` in slot zero; a family with further components overrides this.
  fn initial_state(&self) -> [T; 4] {
    [self.initial_value(), T::zero(), T::zero(), T::zero()]
  }
  /// Number of grid points including `t = 0`.
  fn grid_points(&self) -> usize;
  fn horizon(&self) -> T;
  /// One draw from the process's seed source: reproducible for
  /// `Deterministic`, fresh entropy for `Unseeded`.
  fn device_seed(&self) -> u64;

  /// The time step of the recursion, by default `horizon / (grid_points - 1)`. A process whose
  /// host sampler divides the horizon differently states that here, so the device has its law.
  fn time_step(&self) -> T {
    self.horizon() / T::from_usize_(self.grid_points().max(2) - 1)
  }

  /// A time-varying coefficient, one value per grid point, read by the step as `ct`, or `None`.
  /// A family that reads several overrides [`curves`](Self::curves), which the engine reads.
  fn curve(&self) -> Option<Vec<T>> {
    None
  }

  /// Every time-varying coefficient, one `Vec` per curve in the order `ct`, `ct1`, … `ct7`, at
  /// most [`CURVE_SLOTS`]. Defaults to lifting [`curve`](Self::curve); override one, never both.
  fn curves(&self) -> Option<Vec<Vec<T>>> {
    self.curve().map(|c| vec![c])
  }

  /// The jump intensity per unit time, or `None` for a family with no jump term. The kernel
  /// draws a Poisson count with mean `intensity · dt` once per step and offers it as `nj`.
  fn jump_intensity(&self) -> Option<T> {
    None
  }

  /// How big each jump is, or `None` when the family reads only the count.
  /// The step sees the sum as `js`.
  fn jump_sizes(&self) -> Option<JumpSizes<T>> {
    None
  }

  /// Whether the first point written is a step, not the initial state: a conditional-variance
  /// series starts at the draw `σ₀ z₀`, while every diffusion starts at a level (the default).
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

  /// The fGN pipeline's inputs when the increments come from one, `None` for the kernel's own
  /// normals: inputs, not increments, since the device keeps its increments in its own buffer.
  fn fgn_spec(&self) -> Option<FgnSpec<'_, T>> {
    None
  }

  /// The Markov lift this process's family steps under, or `None` without a `lift` clause. The
  /// device keeps the lift's state per path and offers the lifted value to the step as `lv`.
  fn lift_spec(&self) -> Option<LiftSpec<'_, T>> {
    None
  }

  /// How many terms a family with a `series` clause draws per path before
  /// the steps; `None` for a family without one.
  fn series_terms(&self) -> Option<u32> {
    None
  }

  /// The table a family with a `table` clause has built per path before the
  /// steps; `None` for a family without one.
  fn table_spec(&self) -> Option<TableSpec<T>> {
    None
  }

  /// The programs the kernel interprets for this process, coefficients written as expressions
  /// that the family reads as `pv` and `pv2`, or `None`.
  fn program_spec(&self) -> Option<ProgramSpec<'_>> {
    None
  }
}

/// A process whose device recursion carries `D` state components: the same families, kernels and
/// launch, with `D` arrays back. The kernels always carry four slots, so a system costs no more.
#[doc(hidden)]
pub trait EulerSystem<T: FloatExt, const D: usize>:
  ProcessExt<T, Output: From<[Array1<T>; D]>>
{
  /// The family this system steps. Its component count must be `D`.
  fn euler_spec(&self) -> EulerSpec<T>;

  /// The state each path starts from, in the engine's four slots, the `D` reported ones first.
  /// A family may step more than `D`, an accumulator say; unused slots stay zero.
  fn initial_state(&self) -> [T; 4];

  /// Number of grid points including `t = 0`.
  fn grid_points(&self) -> usize;

  fn horizon(&self) -> T;

  /// One draw from the process's seed source: reproducible for
  /// `Deterministic`, fresh entropy for `Unseeded`.
  fn device_seed(&self) -> u64;

  /// The time step of the recursion, by default `horizon / (grid_points - 1)`. A process whose
  /// host sampler divides the horizon differently states that here, so the device has its law.
  fn time_step(&self) -> T {
    self.horizon() / T::from_usize_(self.grid_points().max(2) - 1)
  }

  /// A time-varying coefficient, one value per grid point, read by the step as `ct`, or `None`.
  /// A family that reads several overrides [`curves`](Self::curves), which the engine reads.
  fn curve(&self) -> Option<Vec<T>> {
    None
  }

  /// Every time-varying coefficient, one `Vec` per curve in the order `ct`, `ct1`, … `ct7`, at
  /// most [`CURVE_SLOTS`]. Defaults to lifting [`curve`](Self::curve); override one, never both.
  fn curves(&self) -> Option<Vec<Vec<T>>> {
    self.curve().map(|c| vec![c])
  }

  /// The jump intensity per unit time, or `None` for a family with no jump term. The kernel
  /// draws a Poisson count with mean `intensity · dt` once per step and offers it as `nj`.
  fn jump_intensity(&self) -> Option<T> {
    None
  }

  /// How big each jump is, or `None` when the family reads only the count.
  /// The step sees the sum as `js`.
  fn jump_sizes(&self) -> Option<JumpSizes<T>> {
    None
  }

  /// Whether the first point written is a step, not the initial state: a conditional-variance
  /// series starts at the draw `σ₀ z₀`, while every diffusion starts at a level (the default).
  fn step_first(&self) -> bool {
    false
  }

  /// The Gamma draws the step reads as `gm` and `gm2`, or `None` when it
  /// reads none.
  fn gamma_draws(&self) -> Option<GammaDraws<T>> {
    None
  }

  /// [`EulerCoefficients::fgn_spec`] for a system, [`FgnSpec::streams`] saying how many noise
  /// components the pipeline fills.
  fn fgn_spec(&self) -> Option<FgnSpec<'_, T>> {
    None
  }

  /// The Markov lift this process's family steps under, or `None` without a `lift` clause. The
  /// device keeps the lift's state per path and offers the lifted value to the step as `lv`.
  fn lift_spec(&self) -> Option<LiftSpec<'_, T>> {
    None
  }

  /// How many terms a family with a `series` clause draws per path before
  /// the steps; `None` for a family without one.
  fn series_terms(&self) -> Option<u32> {
    None
  }

  /// The table a family with a `table` clause has built per path before the
  /// steps; `None` for a family without one.
  fn table_spec(&self) -> Option<TableSpec<T>> {
    None
  }

  /// The programs the kernel interprets for this process, coefficients written as expressions
  /// that the family reads as `pv` and `pv2`, or `None`.
  fn program_spec(&self) -> Option<ProgramSpec<'_>> {
    None
  }

  /// One draw from the process's own sampler, the host stream.
  fn host_sample(&self) -> Self::Output;
}

/// Panics unless the family steps at least the `D` components the system returns: a declaration
/// error, refused here so no plane the kernel never wrote is returned. Public as `EulerKernel` is.
#[doc(hidden)]
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
#[doc(hidden)]
pub fn system_row<T: FloatExt, const D: usize>(planes: &Array3<T>, row: usize) -> [Array1<T>; D] {
  std::array::from_fn(|c| planes.slice(ndarray::s![c, row, ..]).to_owned())
}

/// One launch seed from a process's seed source, [`SeedExt::next_seed`]: reproducible for
/// `Deterministic`, fresh for `Unseeded`, advancing either way. Public as `EulerCoefficients` is.
#[doc(hidden)]
pub fn draw_seed<S: SeedExt>(seed: &S) -> u64 {
  seed.next_seed()
}

impl<T: FloatExt, S: SeedExt, B: EulerBackend<T>> EulerCoefficients<T> for Gbm<T, S, B> {
  fn euler_spec(&self) -> EulerSpec<T> {
    EulerSpec::GeometricBrownian {
      mu: self.mu,
      sigma: self.sigma,
    }
  }

  fn initial_value(&self) -> T {
    self.x0
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
    let (theta, mu, sigma) = (self.theta, self.mu, self.sigma);
    match self.boundary {
      VarianceBoundary::FullTruncation => EulerSpec::SquareRoot {
        kappa: theta,
        theta: mu,
        sigma,
      },
      VarianceBoundary::Absorption => EulerSpec::ReflectedSquareRoot { theta, mu, sigma },
      VarianceBoundary::Reflect => EulerSpec::MirroredSquareRoot { theta, mu, sigma },
    }
  }

  fn initial_value(&self) -> T {
    self.boundary.carried(self.x0.unwrap_or(T::zero()))
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
      /// The batch as one `m × n` matrix: on a device back-end the launch buffer itself, with no
      /// re-layout into rows. [`ProcessExt::try_sample_par`] is the row form.
      pub fn try_sample_matrix(&self, m: usize) -> Result<Array2<T>, DeviceError> {
        self.backend.try_euler_matrix(self, m)
      }
    }
  };
}

try_sample_matrix!(Gbm);
try_sample_matrix!(Ou);
try_sample_matrix!(Cir);

use stochastic_rs_core::simd_rng::SeedExt;

use super::*;

/// A process the device kernels can run: its coefficients, initial value,
/// grid, horizon and the seed the launch derives from the process's seed
/// source.
#[doc(hidden)]
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

  /// The fGN pipeline's inputs when the increments come from one, `None` for the kernel's own
  /// normals: inputs, not increments, since the device keeps its increments in its own buffer.
  fn fgn_spec(&self) -> Option<FgnSpec<'_, T>> {
    None
  }

  /// The Markov lift this process's family steps under, or `None` when it
  /// declares no `lift` clause. The device keeps the lift's state per path
  /// and offers the lifted value to the step as `lv`.
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

  /// The programs the kernel interprets for this process — coefficients
  /// written as expressions, read by the family as `pv` and `pv2` — or
  /// `None` when the family reads none.
  fn program_spec(&self) -> Option<ProgramSpec<'_>> {
    None
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
#[doc(hidden)]
pub trait EulerSystem<T: FloatExt, const D: usize>:
  ProcessExt<T, Output: From<[Array1<T>; D]>>
{
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

  /// [`EulerCoefficients::fgn_spec`] for a system, [`FgnSpec::streams`] saying how many noise
  /// components the pipeline fills.
  fn fgn_spec(&self) -> Option<FgnSpec<'_, T>> {
    None
  }

  /// The Markov lift this process's family steps under, or `None` when it
  /// declares no `lift` clause. The device keeps the lift's state per path
  /// and offers the lifted value to the step as `lv`.
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

  /// The programs the kernel interprets for this process — coefficients
  /// written as expressions, read by the family as `pv` and `pv2` — or
  /// `None` when the family reads none.
  fn program_spec(&self) -> Option<ProgramSpec<'_>> {
    None
  }

  /// One draw from the process's own sampler, the host stream.
  fn host_sample(&self) -> Self::Output;
}

/// The arity check that a system's `D` paths are components its family
/// actually steps. A mismatch is a declaration error rather than a runtime
/// condition, so it fails loudly here instead of silently returning planes
/// the kernel never wrote.
///
/// Public because [`EulerKernel`] is: a device implemented outside this crate
/// needs it, and [`system_row`], to satisfy `euler_system_kernel`.
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

/// One launch seed from a process's own seed source: reproducible for
/// `Deterministic`, fresh entropy for `Unseeded`, and advancing either way so
/// two launches from one process do not replay.
///
/// This is [`SeedExt::next_seed`] and nothing else — the workspace draws its
/// randomness from its own generator, never from `rand`. Public because
/// `EulerCoefficients` is: an out-of-tree process needs it to answer
/// `device_seed`.
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

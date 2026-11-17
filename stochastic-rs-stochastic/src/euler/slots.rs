use super::*;

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

/// The most nodes a Markov lift may carry on a device: one past
/// [`crate::rough::RlKernel::MAX_STABLE_DEGREE`], since the kernels hold the
/// lift's two state vectors in fixed per-thread arrays of this length.
#[doc(hidden)]
pub const LIFT_SLOTS: usize = 176;

/// What a device needs to run the Markov lift of a rough Volterra kernel for a
/// process: the per-node constants the host precomputed and the two boundary
/// terms, exactly the quantities [`crate::volterra::lift::VolterraLift`] steps
/// with. The family declares the drift, the diffusion and the driving shock;
/// the frame keeps the `2 · degree` lift states per path and hands the family
/// the lifted value as `lv` each step.
#[doc(hidden)]
#[derive(Debug)]
pub struct LiftSpec<'a, T> {
  /// `exp(-x_l dt)` per node.
  pub decay: &'a [T],
  /// `w_l exp(-x_l dt)` per node: the weight the history sum applies.
  pub weight: &'a [T],
  /// `(1 - exp(-x_l dt)) / x_l` per node: the drift's per-node integral.
  pub drift_scale: &'a [T],
  /// The kernel's integral over one step, multiplying the drift.
  pub drift_boundary: T,
  /// The kernel's value at one step, multiplying the diffusion.
  pub diffusion_boundary: T,
  /// The lifted component's starting value, which every step adds back.
  pub x0: T,
}

/// The lift tables and scalars a launch binds: `[decay, weight, drift_scale]`
/// (empty when the process has no lift), the `has_lift` flag, the node count
/// and the two boundary terms plus the start the lift adds back each step.
#[cfg_attr(not(device_backend), allow(dead_code))]
pub(crate) fn encode_lift<'a, T: FloatExt>(
  lift: Option<&LiftSpec<'a, T>>,
) -> ([&'a [T]; 3], u32, u32, T, T, T) {
  match lift {
    Some(l) => {
      let n = l.decay.len();
      assert!(
        n <= LIFT_SLOTS && l.weight.len() == n && l.drift_scale.len() == n,
        "a lift carries at most {LIFT_SLOTS} nodes and three tables of one length, not {n}, {}, {}",
        l.weight.len(),
        l.drift_scale.len()
      );
      (
        [l.decay, l.weight, l.drift_scale],
        1,
        n as u32,
        l.drift_boundary,
        l.diffusion_boundary,
        l.x0,
      )
    }
    None => ([&[], &[], &[]], 0, 0, T::zero(), T::zero(), T::zero()),
  }
}

/// The floats a launch's program buffer holds: the two programs' lengths,
/// then each program's `(opcode, constant)` pairs — room for two programs of
/// [`Program::MAX_OPS`] operations each.
#[doc(hidden)]
pub const PROGRAM_SLOTS: usize = 2 + 4 * Program::MAX_OPS;

/// The programs a launch interprets in its kernel: a coefficient of time and
/// state written as an [`Expr`](stochastic_rs_distributions::traits::Expr)
/// and compiled, and an optional second one. Before every step the kernel
/// evaluates each at the launch's first curve `ct` and the first state slot
/// and hands the values to the family as `pv` and `pv2` — the local
/// volatility of a Cheyette model, the drift and diffusion of a Volterra
/// equation.
#[doc(hidden)]
#[derive(Debug)]
pub struct ProgramSpec<'a> {
  /// The first program, read as `pv`.
  pub first: &'a Program,
  /// The second program, read as `pv2`; `None` leaves `pv2` at zero.
  pub second: Option<&'a Program>,
}

/// The program buffer a launch binds — `[len₁, len₂, pairs…]` in
/// [`PROGRAM_SLOTS`] floats — and how many programs it holds.
#[cfg_attr(not(device_backend), allow(dead_code))]
pub(crate) fn encode_programs<T: FloatExt>(spec: Option<&ProgramSpec<'_>>) -> (Vec<T>, u32) {
  let mut out = vec![T::zero(); PROGRAM_SLOTS];
  let Some(spec) = spec else {
    return (out, 0);
  };
  let mut at = 2usize;
  let mut write = |program: &Program, slot: usize| {
    out[slot] = T::from_usize_(program.len());
    for &(code, value) in program.ops() {
      out[at] = T::from_usize_(code as usize);
      out[at + 1] = T::from_f64_fast(value);
      at += 2;
    }
  };
  write(spec.first, 0);
  if let Some(second) = spec.second {
    write(second, 1);
  }
  (out, if spec.second.is_some() { 2 } else { 1 })
}

/// How many time-varying coefficients a kernel binds per launch, which is the
/// most a process's `curves()` may return. Public because that hook is: an
/// out-of-tree process needs the cap it is held to. A family
/// names them `ct` and `ct1` through `ct7`; a launch pays one buffer read per
/// declared curve per step, so declaring fewer costs less.
#[doc(hidden)]
pub const CURVE_SLOTS: usize = 8;

/// The most grid points a family with a `history` clause steps on a device:
/// the kernels keep each path's pushed values in an array of this many slots
/// and convolve them with the family's weight curve every step. A longer grid
/// stays on the host, which the process's own guard decides.
#[doc(hidden)]
pub const HISTORY_SLOTS: usize = 512;

/// The curve slot the launch's family reads its history weights from, or the
/// kernels' "none" sentinel; a history family whose grid exceeds
/// [`HISTORY_SLOTS`] never reaches a launch through `ProcessExt`, so an
/// oversize grid here is a caller bypassing that guard.
#[cfg_attr(not(device_backend), allow(dead_code))]
pub(crate) fn history_slot(family: u32, n: usize) -> u32 {
  match families::Family::from_code(family).and_then(families::Family::history_slot) {
    Some(slot) => {
      assert!(
        n <= HISTORY_SLOTS,
        "a history family steps at most {HISTORY_SLOTS} grid points on a device, not {n}"
      );
      slot
    }
    None => u32::MAX,
  }
}

/// The most grid points a family with a `series` clause steps on a device:
/// the kernels sum each path's terms into an array of this many cells. A
/// longer grid stays on the host, which the process's own guard decides.
#[doc(hidden)]
pub const SERIES_SLOTS: usize = 512;

/// How many series terms the launch draws per path, or zero for a family
/// without a `series` clause. A family with one is never launched without a
/// count and a family without one never with — either would be a process
/// declaring the wrong family — and a grid beyond [`SERIES_SLOTS`] never
/// reaches a launch through `ProcessExt`, so an oversize one here is a caller
/// bypassing that guard.
#[cfg_attr(not(device_backend), allow(dead_code))]
pub(crate) fn series_terms(family: u32, n: usize, terms: Option<u32>) -> u32 {
  let family = families::Family::from_code(family);
  let has_series = family.is_some_and(families::Family::has_series);
  assert!(
    has_series == terms.is_some(),
    "a family declares a series clause exactly when its process names a term count"
  );
  if let Some(count) = terms {
    if family.is_some_and(families::Family::series_live) {
      assert!(
        count as usize <= SERIES_SLOTS,
        "a live series family keeps at most {SERIES_SLOTS} terms per path on a device, not {count}"
      );
    } else {
      assert!(
        n <= SERIES_SLOTS,
        "a series family steps at most {SERIES_SLOTS} grid points on a device, not {n}"
      );
    }
  }
  terms.unwrap_or(0)
}

/// Whether the launch's family sizes its series terms in their own step
/// rather than in the preamble, as the kernels read it.
#[cfg_attr(not(device_backend), allow(dead_code))]
pub(crate) fn series_live(family: u32) -> u32 {
  u32::from(families::Family::from_code(family).is_some_and(families::Family::series_live))
}

/// The most points a family with a `table` clause builds per path on a
/// device: the kernels keep the table in an array of this many entries. A
/// finer table stays on the host, which the process's own guard decides.
#[doc(hidden)]
pub const TABLE_SLOTS: usize = 512;

/// What a family with a `table` clause has the frame build per path before
/// the steps: a monotone table of `points` values over `[0, u_max]` — a
/// non-positive `u_max` meaning the horizon, one at least — grown by doubling
/// until it reaches the horizon.
#[derive(Clone, Copy, Debug, PartialEq)]
#[doc(hidden)]
pub struct TableSpec<T> {
  pub points: u32,
  pub u_max: T,
}

/// The table's point count and starting extent for a launch, zero and zero
/// for a family without a `table` clause. A family with one is never launched
/// without a spec and a family without one never with, and a table beyond
/// [`TABLE_SLOTS`] never reaches a launch through `ProcessExt`.
#[cfg_attr(not(device_backend), allow(dead_code))]
pub(crate) fn table_terms<T: FloatExt>(family: u32, spec: Option<TableSpec<T>>) -> (u32, T) {
  let has_table = families::Family::from_code(family).is_some_and(families::Family::has_table);
  assert!(
    has_table == spec.is_some(),
    "a family declares a table clause exactly when its process names a table"
  );
  match spec {
    Some(spec) => {
      assert!(
        spec.points as usize <= TABLE_SLOTS && spec.points >= 2,
        "a table family builds between 2 and {TABLE_SLOTS} points on a device, not {}",
        spec.points
      );
      (spec.points, spec.u_max)
    }
    None => (0, T::zero()),
  }
}

/// The curve buffer a launch binds: the declared curves laid end to end, each
/// padded to `n` values, so the kernel reads curve `k` at step `i` from
/// `curve[k * n + i]`. Returns the flattened values and how many curves they
/// hold; an empty buffer and zero when the family declares none.
///
/// A curve shorter than the grid is extended with its last value rather than
/// read out of bounds — a host tabulation that stops one short is a
/// declaration slip, not a reason to fault a kernel.
#[cfg_attr(not(device_backend), allow(dead_code))]
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

/// What a device needs to run the fGN pipeline for a process whose increments
/// are fractional: the precomputed circulant eigenvalues and the grid they
/// were built for.
#[doc(hidden)]
#[derive(Debug)]
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

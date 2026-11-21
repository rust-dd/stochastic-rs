use super::*;

/// The streams a correlated four-component family can carry; a model with more stays on its
/// host sampler whatever the backend.
pub(crate) const CORRELATED_STREAMS: usize = 4;

/// A lower Cholesky factor of at most [`CORRELATED_STREAMS`] rows, packed row-major into ten
/// lower-triangle slots. A missing row gets a unit diagonal, so a padded component is constant.
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

/// The most nodes a Markov lift may carry on a device, one past `RlKernel::MAX_STABLE_DEGREE`:
/// the kernels hold the lift's two state vectors in fixed per-thread arrays of this length.
#[doc(hidden)]
pub const LIFT_SLOTS: usize = 176;

/// What a device needs to run a rough kernel's Markov lift: the per-node constants and the two
/// boundary terms `VolterraLift` steps with. The frame hands the family the lifted value as `lv`.
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

/// The lift tables and scalars a launch binds: `[decay, weight, drift_scale]` (empty without a
/// lift), the `has_lift` flag, the node count, the two boundary terms and the start added back.
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

/// The floats a launch's program buffer holds: the two programs' lengths, then each program's
/// `(opcode, constant)` pairs, with room for two programs of [`Program::MAX_OPS`] operations.
#[doc(hidden)]
pub const PROGRAM_SLOTS: usize = 2 + 4 * Program::MAX_OPS;

/// The one or two compiled `Expr` programs a launch interprets. Before every step the kernel
/// evaluates each at the first curve `ct` and the first state slot and passes `pv` and `pv2`.
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

/// How many time-varying coefficients a kernel binds per launch, the most `curves()` may
/// return (`ct`, `ct1` … `ct7`); each declared curve costs one buffer read per step.
#[doc(hidden)]
pub const CURVE_SLOTS: usize = 8;

/// The most grid points a family with a `history` clause steps on a device: the kernels keep
/// each path's pushed values in an array this long. A longer grid stays on the host.
#[doc(hidden)]
pub const HISTORY_SLOTS: usize = 512;

/// The curve slot the family reads its history weights from, or the kernels' "none" sentinel.
/// `ProcessExt` keeps a grid past [`HISTORY_SLOTS`] on the host, so one here bypassed that guard.
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

/// The most grid points a family with a `series` clause steps on a device: the kernels sum
/// each path's terms into an array of this many cells. A longer grid stays on the host.
#[doc(hidden)]
pub const SERIES_SLOTS: usize = 512;

/// How many series terms the launch draws per path, zero for a family without a `series`
/// clause. A count without the clause, the reverse, or a grid past [`SERIES_SLOTS`] is refused.
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

/// The most points a family with a `table` clause builds per path on a device: the kernels
/// keep the table in an array of this many entries. A finer table stays on the host.
#[doc(hidden)]
pub const TABLE_SLOTS: usize = 512;

/// What a `table` family has the frame build per path before the steps: a monotone table of
/// `points` values over `[0, u_max]` (non-positive means the horizon), doubled until it reaches it.
#[derive(Clone, Copy, Debug, PartialEq)]
#[doc(hidden)]
pub struct TableSpec<T> {
  pub points: u32,
  pub u_max: T,
}

/// The table's point count and starting extent for a launch, zeros for a family without a
/// `table` clause. A spec without the clause, the reverse, or one past [`TABLE_SLOTS`] is refused.
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

/// The declared curves laid end to end, each padded to `n` values with its last one, so the
/// kernel reads curve `k` at step `i` from `curve[k * n + i]`; returns the values and the count.
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

/// What a device needs to run the fGN pipeline for a process with fractional increments: the
/// precomputed circulant eigenvalues and the grid they were built for.
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
  /// How many independent fGN streams the noise components read: one feeds `noise[0]`, a second
  /// `noise[1]`. A correlated pair shares one embedding, drawn as `streams * m` paths in one call.
  pub streams: usize,
}

//! The generator behind the family table: one `macro_rules!` that turns each
//! declaration into the host step and report, the C statements the CUDA and
//! Metal kernels render, the CubeCL functions, and the `Family` enum the
//! kernels dispatch on. The declarations it consumes are in the parent
//! module, which pulls the macro in with `#[macro_use]` because the
//! generated modules recurse into it; the vocabulary the generated code
//! calls is in the sibling.

/// Declares the Euler engine's families: one entry per family, from which the
/// host step, the host report and the kernels' C body are generated.
///
/// Each entry names its parameters in the order the parameter buffer carries
/// them, its state components and its noise components, then gives one step
/// expression per state component and one report expression per component.
/// A family with a single component is the common case and reads as it did
/// before the engine learned about systems; the comma-separated forms are
/// what a stochastic-volatility or two-factor model needs.
macro_rules! euler_families {
  (
    step_inputs($params:ident, $dt:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $ln:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, state $sxs:tt, noise $sds:tt, select($component:ident, $produced:ident));
    $(
    $(#[$meta:meta])*
    $code:literal => $name:ident { $($param:ident),* $(,)? }
      state ($($state:ident),+ $(,)?)
      noise ($($noise:ident),+ $(,)?)
      step { $($step:tt)* }
      report { $($report:tt)* }
      $(lift $lift:tt)?
      $(history $hist:tt)?
      $(series $ser:tt)?
      $(table $tab:tt)?
  ),* $(,)?) => {
    /// The family codes the kernels dispatch on, in declaration order.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    #[repr(u32)]
    pub(crate) enum Family {
      $(
        $(#[$meta])*
        $name = $code,
      )*
    }

    impl Family {
      /// Every declared family, in declaration order. What a check over all
      /// of them iterates, so a family added without one is not silently
      /// skipped.
      #[allow(dead_code)]
      pub(crate) const ALL: &'static [Family] = &[$(Family::$name),*];

      /// The code the kernels compare `family` against.
      #[allow(dead_code)]
      pub(crate) fn code(self) -> u32 {
        self as u32
      }

      /// The family a code names, or `None` when no family carries it. The
      /// inverse of [`code`](Self::code), which is what lets a caller holding
      /// an encoded spec run the generated host step for it.
      #[allow(dead_code)]
      pub(crate) fn from_code(code: u32) -> Option<Self> {
        match code {
          $( $code => Some(Family::$name), )*
          _ => None,
        }
      }

      /// How many state components the family steps, which is how many paths
      /// a launch writes and how many arrays a process built on it returns.
      /// How many values the family reports per grid point — the planes a
      /// launch writes. A family may keep more state slots than it reports.
      #[allow(dead_code)]
      pub(crate) fn components(self) -> usize {
        match self {
          $( Family::$name => euler_families!(@count_exprs $($report)*), )*
        }
      }

      /// How many state slots the family steps — at least as many as it
      /// reports, more when it keeps auxiliary state the path never records.
      #[allow(dead_code)]
      pub(crate) fn slots(self) -> usize {
        match self {
          $( Family::$name => [$(stringify!($state)),*].len(), )*
        }
      }

      /// The curve slot a family's `history` clause reads its weights from,
      /// `None` for a family without one.
      #[allow(dead_code)]
      pub(crate) fn history_slot(self) -> Option<u32> {
        match self {
          $( Family::$name => euler_families!(@history_slot $($hist)?), )*
        }
      }

      /// Whether the family declares a `series` clause: terms drawn per path
      /// before the steps and summed into the grid cells they fall in.
      #[allow(dead_code)]
      pub(crate) fn has_series(self) -> bool {
        match self {
          $( Family::$name => euler_families!(@has_series $($ser)?), )*
        }
      }

      /// Whether the family declares a `table` clause: a monotone table built
      /// per path before the steps, whose inverse each step reads at its time.
      #[allow(dead_code)]
      pub(crate) fn has_table(self) -> bool {
        match self {
          $( Family::$name => euler_families!(@has_table $($tab)?), )*
        }
      }

      /// How many independent noise components a step draws. A model that
      /// wants correlated noise draws independent components and correlates
      /// them in its own step, which is what the host samplers do.
      #[allow(dead_code)]
      pub(crate) fn noises(self) -> usize {
        match self {
          $( Family::$name => [$(stringify!($noise)),*].len(), )*
        }
      }
    }

    /// One step of `family` on the host, from the same expressions the
    /// kernels run. The parameter buffer is read in declaration order, the
    /// state and the noise by position, and every component is computed from
    /// the state as it stood before the step.
    #[allow(dead_code, unused_variables, unused_mut, unused_assignments)]
    pub(crate) fn host_step<T: FloatExt>(
      family: Family,
      state: &[T],
      $params: &[T],
      $dt: T,
      $ct: T,
      $ct1: T,
      $ct2: T,
      $ct3: T,
      $ct4: T,
      $ct5: T,
      $ct6: T,
      $ct7: T,
      $nj: T,
      $js: T,
      $gm: T,
      $gm2: T,
      $u: T,
      $u2: T,
      $ln: T,
      $cv: T,
      $sj: T,
      $gj: T,
      $ej: T,
      $uj: T,
      $uv: T,
      $iv: T,
      $tv: T,
      noise: &[T],
      out: &mut [T],
    ) {
      #[allow(unused_imports)]
      use ops::*;
      match family {
        $(
          Family::$name => {
            let mut slot = 0;
            $(
              let $param = $params[slot];
              slot += 1;
            )*
            let mut at = 0;
            $(
              let $state = state[at];
              at += 1;
            )*
            let mut of = 0;
            $(
              let $noise = noise[of];
              of += 1;
            )*
            euler_families!(@host_assign out, $($step)*)
          }
        )*
      }
    }

    /// What `family` reports for a state, on the host. The parameters and the
    /// state components are bound as [`host_step`] binds them, so a report may
    /// name either.
    #[allow(dead_code, unused_variables, unused_mut, unused_assignments)]
    pub(crate) fn host_report<T: FloatExt>(
      family: Family,
      state: &[T],
      $params: &[T],
      $ct: T,
      $ct1: T,
      $ct2: T,
      $ct3: T,
      $ct4: T,
      $ct5: T,
      $ct6: T,
      $ct7: T,
      $nj: T,
      $js: T,
      $gm: T,
      $gm2: T,
      $u: T,
      $u2: T,
      $ln: T,
      $cv: T,
      $sj: T,
      $gj: T,
      $ej: T,
      $uj: T,
      $uv: T,
      $iv: T,
      $tv: T,
      out: &mut [T],
    ) {
      #[allow(unused_imports)]
      use ops::*;
      match family {
        $(
          Family::$name => {
            let mut slot = 0;
            $(
              let $param = $params[slot];
              slot += 1;
            )*
            let mut at = 0;
            $(
              let $state = state[at];
              at += 1;
            )*
            euler_families!(@host_assign out, $($report)*)
          }
        )*
      }
    }

    /// One `#[cube]` function per family, from the same expressions the host
    /// and the C kernels run. Each takes the four state and four noise
    /// scalars plus the component to produce, so the hand-written dispatcher
    /// calls every family the same way. The bindings and the per-component
    /// branches are peeled into the body before the `#[cube]` attribute sees
    /// it, which the attribute cannot look through.
    #[cfg(feature = "cubecl")]
    #[allow(non_snake_case)]
    pub(crate) mod cube {
      #[allow(unused_imports)]
      use super::cube_ops::*;
      #[allow(unused_imports)]
      use cubecl::prelude::*;

      $(
        euler_families!(@cube_step
          $(#[$meta])* $name, $params, $dt, $ct, $ct1, $ct2, $ct3, $ct4, $ct5, $ct6, $ct7, $nj, $js, $gm, $gm2, $u, $u2, $ln, $cv, $sj, $gj, $ej, $uj, $uv, $iv, $tv, $component, $produced,
          sig $sxs $sds,
          place $sxs [$($state)*] $sds [$($noise)*],
          params [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19] [$($param)*],
          bound {}, arms {}, at [0u32 1u32 2u32 3u32], body {$($step)*}
        );
      )*
    }

    #[cfg(feature = "cubecl")]
    #[allow(non_snake_case)]
    pub(crate) mod cube_lift {
      #[allow(unused_imports)]
      use super::cube_ops::*;
      #[allow(unused_imports)]
      use cubecl::prelude::*;

      $(
        euler_families!(@cube_lift
          $(#[$meta])* $name, $params, $dt, $ct, $ct1, $ct2, $ct3, $ct4, $ct5, $ct6, $ct7, $nj, $js, $gm, $gm2, $u, $u2, $ln, $cv, $sj, $gj, $ej, $uj, $uv, $iv, $tv, $component, $produced,
          sig $sxs $sds,
          place $sxs [$($state)*] $sds [$($noise)*],
          params [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19] [$($param)*],
          $($lift)?
        );
      )*
    }

    /// One `#[cube]` push function per history family, shaped like [`cube`]
    /// with the pushed value as its only component.
    #[cfg(feature = "cubecl")]
    #[allow(non_snake_case)]
    pub(crate) mod cube_history {
      #[allow(unused_imports)]
      use super::cube_ops::*;
      #[allow(unused_imports)]
      use cubecl::prelude::*;

      $(
        euler_families!(@cube_history
          $(#[$meta])* $name, $params, $dt, $ct, $ct1, $ct2, $ct3, $ct4, $ct5, $ct6, $ct7, $nj, $js, $gm, $gm2, $u, $u2, $ln, $cv, $sj, $gj, $ej, $uj, $uv, $iv, $tv, $component, $produced,
          sig $sxs $sds,
          place $sxs [$($state)*] $sds [$($noise)*],
          params [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19] [$($param)*],
          $($hist)?
        );
      )*
    }

    /// One `#[cube]` size function per series family, shaped like [`cube`]
    /// with the term size as its only component.
    #[cfg(feature = "cubecl")]
    #[allow(non_snake_case)]
    pub(crate) mod cube_series {
      #[allow(unused_imports)]
      use super::cube_ops::*;
      #[allow(unused_imports)]
      use cubecl::prelude::*;

      $(
        euler_families!(@cube_series
          $(#[$meta])* $name, $params, $dt, $ct, $ct1, $ct2, $ct3, $ct4, $ct5, $ct6, $ct7, $nj, $js, $gm, $gm2, $u, $u2, $ln, $cv, $sj, $gj, $ej, $uj, $uv, $iv, $tv, $component, $produced,
          sig $sxs $sds,
          place $sxs [$($state)*] $sds [$($noise)*],
          params [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19] [$($param)*],
          $($ser)?
        );
      )*
    }

    /// One `#[cube]` increment function per table family, shaped like [`cube`]
    /// with the increment as its only component.
    #[cfg(feature = "cubecl")]
    #[allow(non_snake_case)]
    pub(crate) mod cube_table {
      #[allow(unused_imports)]
      use super::cube_ops::*;
      #[allow(unused_imports)]
      use cubecl::prelude::*;

      $(
        euler_families!(@cube_table
          $(#[$meta])* $name, $params, $dt, $ct, $ct1, $ct2, $ct3, $ct4, $ct5, $ct6, $ct7, $nj, $js, $gm, $gm2, $u, $u2, $ln, $cv, $sj, $gj, $ej, $uj, $uv, $iv, $tv, $component, $produced,
          sig $sxs $sds,
          place $sxs [$($state)*] $sds [$($noise)*],
          params [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19] [$($param)*],
          $($tab)?
        );
      )*
    }

    /// One `#[cube]` report function per family, shaped like [`cube`].
    #[cfg(feature = "cubecl")]
    #[allow(non_snake_case)]
    pub(crate) mod cube_report {
      #[allow(unused_imports)]
      use super::cube_ops::*;
      #[allow(unused_imports)]
      use cubecl::prelude::*;

      $(
        euler_families!(@cube_report
          $(#[$meta])* $name, $params, $ct, $ct1, $ct2, $ct3, $ct4, $ct5, $ct6, $ct7, $nj, $js, $gm, $gm2, $u, $u2, $ln, $cv, $sj, $gj, $ej, $uj, $uv, $iv, $tv, $component, $produced,
          sig $sxs,
          place $sxs [$($state)*],
          params [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19] [$($param)*],
          bound {}, arms {}, at [0u32 1u32 2u32 3u32], body {$($report)*}
        );
      )*
    }

    /// The C statements that step the state, one guarded block per family.
    pub(crate) const C_STEP: &str = concat!($(
      "        if (family == ", stringify!($code), "u) {\n",
      euler_families!(@bind_params [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19] $($param)*),
      euler_families!(@bind_slots "state" [0 1 2 3] $($state)*),
      euler_families!(@bind_slots "noise" [0 1 2 3] $($noise)*),
      euler_families!(@c_body "state", [0 1 2 3], [$($state)*], $($step)*),
      "        }\n",
    )*);

    /// The C statements that evaluate a lifted family's three coefficients —
    /// its drift, its diffusion and the shock that drives the lift — into
    /// `lift[0..3]`, one block per family that declares a `lift` clause. A
    /// family without one contributes nothing: the launch has no lift.
    pub(crate) const C_LIFT: &str = concat!($(
      euler_families!(@c_lift $code, [$($param)*], [$($state)*], [$($noise)*], $($lift)?),
    )*);

    /// The host evaluation of a lifted family's `[drift, diffusion, shock]`
    /// from the state and noise of one step, `[0, 0, 0]` for a family that
    /// declares no lift. What the parity probes and the tests run against.
    #[allow(dead_code, clippy::too_many_arguments, unused_variables)]
    pub(crate) fn host_lift<T: FloatExt>(
      family: Family,
      state: &[T],
      $params: &[T],
      $dt: T,
      noise: &[T],
    ) -> [T; 3] {
      #[allow(unused_imports)]
      use ops::*;
      match family {
        $(
          Family::$name => {
            euler_families!(@host_lift $params, $dt, state, noise, [$($param)*], [$($state)*], [$($noise)*], $($lift)?)
          }
        )*
      }
    }

    /// The C statements that evaluate a history family's pushed value into
    /// `hist_in[0]`, one block per family that declares a `history` clause;
    /// the frame appends it to the path's history and convolves that history
    /// with the family's weight curve into `cv`. A family without a clause
    /// contributes nothing.
    pub(crate) const C_HISTORY: &str = concat!($(
      euler_families!(@c_history $code, [$($param)*], [$($state)*], [$($noise)*], $($hist)?),
    )*);

    /// The host evaluation of a history family's pushed value from the state,
    /// noise and uniforms of one step, `None` for a family that declares no
    /// history. What the parity probes and the tests run against.
    #[allow(dead_code, clippy::too_many_arguments, unused_variables)]
    pub(crate) fn host_history<T: FloatExt>(
      family: Family,
      state: &[T],
      $params: &[T],
      $dt: T,
      noise: &[T],
      $u: T,
      $u2: T,
    ) -> Option<T> {
      #[allow(unused_imports)]
      use ops::*;
      match family {
        $(
          Family::$name => {
            euler_families!(@host_history $params, $dt, $u, $u2, state, noise, [$($param)*], [$($state)*], [$($noise)*], $($hist)?)
          }
        )*
      }
    }

    /// The C statements that size one term of a series family into
    /// `series_size[0]` from the draws the frame's preamble made, one block
    /// per family that declares a `series` clause. A family without one
    /// contributes nothing.
    pub(crate) const C_SERIES: &str = concat!($(
      euler_families!(@c_series $code, [$($param)*], $($ser)?),
    )*);

    /// The host evaluation of a series family's term size from one set of
    /// draws, `None` for a family that declares no series. What the parity
    /// probes and the tests run against.
    #[allow(dead_code, clippy::too_many_arguments, unused_variables)]
    pub(crate) fn host_series<T: FloatExt>(
      family: Family,
      $params: &[T],
      $dt: T,
      $gj: T,
      $ej: T,
      $uj: T,
      $uv: T,
    ) -> Option<T> {
      #[allow(unused_imports)]
      use ops::*;
      match family {
        $(
          Family::$name => {
            euler_families!(@host_series $params, $dt, $gj, $ej, $uj, $uv, [$($param)*], $($ser)?)
          }
        )*
      }
    }

    /// The C statements that size one increment of a table family into
    /// `table_inc[0]` from the draws the frame's preamble made and the
    /// table's spacing, one block per family that declares a `table` clause.
    /// A family without one contributes nothing.
    pub(crate) const C_TABLE: &str = concat!($(
      euler_families!(@c_table $code, [$($param)*], $($tab)?),
    )*);

    /// The host evaluation of a table family's increment from one set of
    /// draws and the spacing, `None` for a family that declares no table.
    /// What the parity probes and the tests run against.
    #[allow(dead_code, clippy::too_many_arguments, unused_variables)]
    pub(crate) fn host_table<T: FloatExt>(
      family: Family,
      $params: &[T],
      $dt: T,
      $uj: T,
      $uv: T,
      $tv: T,
    ) -> Option<T> {
      #[allow(unused_imports)]
      use ops::*;
      match family {
        $(
          Family::$name => {
            euler_families!(@host_table $params, $dt, $uj, $uv, $tv, [$($param)*], $($tab)?)
          }
        )*
      }
    }

    /// The C statements that set the reported values, one block per family.
    pub(crate) const C_REPORT: &str = concat!($(
      "        if (family == ", stringify!($code), "u) {\n",
      euler_families!(@bind_params [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19] $($param)*),
      euler_families!(@bind_slots "state" [0 1 2 3] $($state)*),
      euler_families!(@c_body "reported", [0 1 2 3], [$($state)*], $($report)*),
      "        }\n",
    )*);
  };

  (@c_lift $code:literal, [$($param:ident)*], [$($state:ident)*], [$($noise:ident)*],) => { "" };

  (@c_lift $code:literal, [$($param:ident)*], [$($state:ident)*], [$($noise:ident)*],
    { drift ($($ld:tt)*) diffusion ($($lg:tt)*) shock ($($lsh:tt)*) }) => {
    concat!(
      "        if (family == ", stringify!($code), "u) {\n",
      euler_families!(@bind_params [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19] $($param)*),
      euler_families!(@bind_slots "state" [0 1 2 3] $($state)*),
      euler_families!(@bind_slots "noise" [0 1 2 3] $($noise)*),
      euler_families!(@c_body "lift", [0 1 2], [lf lg lsh], $($ld)*, $($lg)*, $($lsh)*),
      "        }\n",
    )
  };

  (@host_lift $params:ident, $dt:ident, $state_in:ident, $noise_in:ident, [$($param:ident)*], [$($state:ident)*], [$($noise:ident)*],) => {
    [T::zero(); 3]
  };

  (@host_lift $params:ident, $dt:ident, $state_in:ident, $noise_in:ident, [$($param:ident)*], [$($state:ident)*], [$($noise:ident)*],
    { drift ($($ld:tt)*) diffusion ($($lg:tt)*) shock ($($lsh:tt)*) }) => {{
    #[allow(unused_mut, unused_variables)]
    let mut slot = 0;
    $(
      let $param = $params[slot];
      slot += 1;
    )*
    let _ = slot;
    #[allow(unused_mut, unused_variables)]
    let mut at = 0;
    $(
      let $state = $state_in[at];
      at += 1;
    )*
    let _ = at;
    #[allow(unused_mut, unused_variables)]
    let mut at = 0;
    $(
      let $noise = $noise_in[at];
      at += 1;
    )*
    let _ = at;
    let mut lift = [T::zero(); 3];
    euler_families!(@host_assign lift, $($ld)*, $($lg)*, $($lsh)*);
    lift
  }};

  (@cube_lift
    $(#[$meta:meta])* $name:ident, $params:ident, $dt:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $ln:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $component:ident, $produced:ident,
    sig $sxs:tt $sds:tt,
    place $psx:tt [$($state:ident)*] $psd:tt [$($noise:ident)*],
    params [$($idx:literal)*] [$($param:ident)*],
  ) => {};

  (@cube_lift
    $(#[$meta:meta])* $name:ident, $params:ident, $dt:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $ln:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $component:ident, $produced:ident,
    sig $sxs:tt $sds:tt,
    place $psx:tt [$($state:ident)*] $psd:tt [$($noise:ident)*],
    params [$($idx:literal)*] [$($param:ident)*],
    { drift ($($ld:tt)*) diffusion ($($lg:tt)*) shock ($($lsh:tt)*) }
  ) => {
    euler_families!(@cube_step
      $(#[$meta])* $name, $params, $dt, $ct, $ct1, $ct2, $ct3, $ct4, $ct5, $ct6, $ct7, $nj, $js, $gm, $gm2, $u, $u2, $ln, $cv, $sj, $gj, $ej, $uj, $uv, $iv, $tv, $component, $produced,
      sig $sxs $sds,
      place $psx [$($state)*] $psd [$($noise)*],
      params [$($idx)*] [$($param)*],
      bound {}, arms {}, at [0u32 1u32 2u32 3u32], body {$($ld)*, $($lg)*, $($lsh)*}
    );
  };

  (@count_exprs bind $n:ident = $e:expr; $($rest:tt)*) => {
    euler_families!(@count_exprs $($rest)*)
  };

  (@count_exprs $($e:expr),* $(,)?) => { [$(stringify!($e)),*].len() };

  (@history_slot) => { None };

  (@history_slot { push ($($hp:tt)*) weights (ct) }) => { Some(0u32) };

  (@history_slot { push ($($hp:tt)*) weights (ct1) }) => { Some(1u32) };

  (@history_slot { push ($($hp:tt)*) weights (ct2) }) => { Some(2u32) };

  (@history_slot { push ($($hp:tt)*) weights (ct3) }) => { Some(3u32) };

  (@history_slot { push ($($hp:tt)*) weights (ct4) }) => { Some(4u32) };

  (@history_slot { push ($($hp:tt)*) weights (ct5) }) => { Some(5u32) };

  (@history_slot { push ($($hp:tt)*) weights (ct6) }) => { Some(6u32) };

  (@history_slot { push ($($hp:tt)*) weights (ct7) }) => { Some(7u32) };

  (@c_history $code:literal, [$($param:ident)*], [$($state:ident)*], [$($noise:ident)*],) => { "" };

  (@c_history $code:literal, [$($param:ident)*], [$($state:ident)*], [$($noise:ident)*],
    { push ($($hp:tt)*) weights ($w:ident) }) => {
    concat!(
      "        if (family == ", stringify!($code), "u) {\n",
      euler_families!(@bind_params [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19] $($param)*),
      euler_families!(@bind_slots "state" [0 1 2 3] $($state)*),
      euler_families!(@bind_slots "noise" [0 1 2 3] $($noise)*),
      euler_families!(@c_body "hist_in", [0], [hp], $($hp)*),
      "        }\n",
    )
  };

  (@host_history $params:ident, $dt:ident, $u:ident, $u2:ident, $state_in:ident, $noise_in:ident, [$($param:ident)*], [$($state:ident)*], [$($noise:ident)*],) => {
    None
  };

  (@host_history $params:ident, $dt:ident, $u:ident, $u2:ident, $state_in:ident, $noise_in:ident, [$($param:ident)*], [$($state:ident)*], [$($noise:ident)*],
    { push ($($hp:tt)*) weights ($w:ident) }) => {{
    #[allow(unused_mut, unused_variables)]
    let mut slot = 0;
    $(
      let $param = $params[slot];
      slot += 1;
    )*
    let _ = slot;
    #[allow(unused_mut, unused_variables)]
    let mut at = 0;
    $(
      let $state = $state_in[at];
      at += 1;
    )*
    let _ = at;
    #[allow(unused_mut, unused_variables)]
    let mut at = 0;
    $(
      let $noise = $noise_in[at];
      at += 1;
    )*
    let _ = at;
    let mut pushed = [T::zero(); 1];
    euler_families!(@host_assign pushed, $($hp)*);
    Some(pushed[0])
  }};

  (@cube_history
    $(#[$meta:meta])* $name:ident, $params:ident, $dt:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $ln:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $component:ident, $produced:ident,
    sig $sxs:tt $sds:tt,
    place $psx:tt [$($state:ident)*] $psd:tt [$($noise:ident)*],
    params [$($idx:literal)*] [$($param:ident)*],
  ) => {};

  (@cube_history
    $(#[$meta:meta])* $name:ident, $params:ident, $dt:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $ln:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $component:ident, $produced:ident,
    sig $sxs:tt $sds:tt,
    place $psx:tt [$($state:ident)*] $psd:tt [$($noise:ident)*],
    params [$($idx:literal)*] [$($param:ident)*],
    { push ($($hp:tt)*) weights ($w:ident) }
  ) => {
    euler_families!(@cube_step
      $(#[$meta])* $name, $params, $dt, $ct, $ct1, $ct2, $ct3, $ct4, $ct5, $ct6, $ct7, $nj, $js, $gm, $gm2, $u, $u2, $ln, $cv, $sj, $gj, $ej, $uj, $uv, $iv, $tv, $component, $produced,
      sig $sxs $sds,
      place $psx [$($state)*] $psd [$($noise)*],
      params [$($idx)*] [$($param)*],
      bound {}, arms {}, at [0u32 1u32 2u32 3u32], body {$($hp)*}
    );
  };

  (@has_series) => { false };

  (@has_series { size ($($sz:tt)*) }) => { true };

  (@c_series $code:literal, [$($param:ident)*],) => { "" };

  (@c_series $code:literal, [$($param:ident)*], { size ($($sz:tt)*) }) => {
    concat!(
      "            if (family == ", stringify!($code), "u) {\n",
      euler_families!(@bind_params [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19] $($param)*),
      euler_families!(@c_body "series_size", [0], [sz], $($sz)*),
      "            }\n",
    )
  };

  (@host_series $params:ident, $dt:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, [$($param:ident)*],) => {
    None
  };

  (@host_series $params:ident, $dt:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, [$($param:ident)*],
    { size ($($sz:tt)*) }) => {{
    #[allow(unused_mut, unused_variables)]
    let mut slot = 0;
    $(
      let $param = $params[slot];
      slot += 1;
    )*
    let _ = slot;
    let mut size = [T::zero(); 1];
    euler_families!(@host_assign size, $($sz)*);
    Some(size[0])
  }};

  (@cube_series
    $(#[$meta:meta])* $name:ident, $params:ident, $dt:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $ln:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $component:ident, $produced:ident,
    sig $sxs:tt $sds:tt,
    place $psx:tt [$($state:ident)*] $psd:tt [$($noise:ident)*],
    params [$($idx:literal)*] [$($param:ident)*],
  ) => {};

  (@cube_series
    $(#[$meta:meta])* $name:ident, $params:ident, $dt:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $ln:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $component:ident, $produced:ident,
    sig $sxs:tt $sds:tt,
    place $psx:tt [$($state:ident)*] $psd:tt [$($noise:ident)*],
    params [$($idx:literal)*] [$($param:ident)*],
    { size ($($sz:tt)*) }
  ) => {
    euler_families!(@cube_step
      $(#[$meta])* $name, $params, $dt, $ct, $ct1, $ct2, $ct3, $ct4, $ct5, $ct6, $ct7, $nj, $js, $gm, $gm2, $u, $u2, $ln, $cv, $sj, $gj, $ej, $uj, $uv, $iv, $tv, $component, $produced,
      sig $sxs $sds,
      place $psx [$($state)*] $psd [$($noise)*],
      params [$($idx)*] [$($param)*],
      bound {}, arms {}, at [0u32 1u32 2u32 3u32], body {$($sz)*}
    );
  };

  (@has_table) => { false };

  (@has_table { increment ($($ti:tt)*) }) => { true };

  (@c_table $code:literal, [$($param:ident)*],) => { "" };

  (@c_table $code:literal, [$($param:ident)*], { increment ($($ti:tt)*) }) => {
    concat!(
      "                if (family == ", stringify!($code), "u) {\n",
      euler_families!(@bind_params [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19] $($param)*),
      euler_families!(@c_body "table_inc", [0], [ti], $($ti)*),
      "                }\n",
    )
  };

  (@host_table $params:ident, $dt:ident, $uj:ident, $uv:ident, $tv:ident, [$($param:ident)*],) => {
    None
  };

  (@host_table $params:ident, $dt:ident, $uj:ident, $uv:ident, $tv:ident, [$($param:ident)*],
    { increment ($($ti:tt)*) }) => {{
    #[allow(unused_mut, unused_variables)]
    let mut slot = 0;
    $(
      let $param = $params[slot];
      slot += 1;
    )*
    let _ = slot;
    let mut inc = [T::zero(); 1];
    euler_families!(@host_assign inc, $($ti)*);
    Some(inc[0])
  }};

  (@cube_table
    $(#[$meta:meta])* $name:ident, $params:ident, $dt:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $ln:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $component:ident, $produced:ident,
    sig $sxs:tt $sds:tt,
    place $psx:tt [$($state:ident)*] $psd:tt [$($noise:ident)*],
    params [$($idx:literal)*] [$($param:ident)*],
  ) => {};

  (@cube_table
    $(#[$meta:meta])* $name:ident, $params:ident, $dt:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $ln:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $component:ident, $produced:ident,
    sig $sxs:tt $sds:tt,
    place $psx:tt [$($state:ident)*] $psd:tt [$($noise:ident)*],
    params [$($idx:literal)*] [$($param:ident)*],
    { increment ($($ti:tt)*) }
  ) => {
    euler_families!(@cube_step
      $(#[$meta])* $name, $params, $dt, $ct, $ct1, $ct2, $ct3, $ct4, $ct5, $ct6, $ct7, $nj, $js, $gm, $gm2, $u, $u2, $ln, $cv, $sj, $gj, $ej, $uj, $uv, $iv, $tv, $component, $produced,
      sig $sxs $sds,
      place $psx [$($state)*] $psd [$($noise)*],
      params [$($idx)*] [$($param)*],
      bound {}, arms {}, at [0u32 1u32 2u32 3u32], body {$($ti)*}
    );
  };

  (@host_assign $out:ident, bind $n:ident = $e:expr; $($rest:tt)*) => {{
    let $n = $e;
    euler_families!(@host_assign $out, $($rest)*)
  }};

  (@host_assign $out:ident, $($e:expr),* $(,)?) => {{
    let mut at = 0;
    $(
      $out[at] = $e;
      at += 1;
    )*
    let _ = at;
  }};

  (@cube_step
    $(#[$meta:meta])* $name:ident, $params:ident, $dt:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $ln:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $component:ident, $produced:ident,
    sig ($($sigx:ident),*) ($($sigd:ident),*),
    place ($slot:ident $(, $restx:ident)*) [$head:ident $($restname:ident)*] ($($pd:ident),*) [$($nd:ident)*],
    params [$($idx:literal)*] [$($p:ident)*],
    bound {$($bound:tt)*}, arms {$($arms:tt)*}, at [$($at:literal)*], body {$($body:tt)*}
  ) => {
    euler_families!(@cube_step
      $(#[$meta])* $name, $params, $dt, $ct, $ct1, $ct2, $ct3, $ct4, $ct5, $ct6, $ct7, $nj, $js, $gm, $gm2, $u, $u2, $ln, $cv, $sj, $gj, $ej, $uj, $uv, $iv, $tv, $component, $produced,
      sig ($($sigx),*) ($($sigd),*),
      place ($($restx),*) [$($restname)*] ($($pd),*) [$($nd)*],
      params [$($idx)*] [$($p)*],
      bound {$($bound)* let $head = $slot;}, arms {$($arms)*}, at [$($at)*], body {$($body)*}
    );
  };

  (@cube_step
    $(#[$meta:meta])* $name:ident, $params:ident, $dt:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $ln:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $component:ident, $produced:ident,
    sig ($($sigx:ident),*) ($($sigd:ident),*),
    place ($($px:ident),*) [] ($slot:ident $(, $restd:ident)*) [$head:ident $($restname:ident)*],
    params [$($idx:literal)*] [$($p:ident)*],
    bound {$($bound:tt)*}, arms {$($arms:tt)*}, at [$($at:literal)*], body {$($body:tt)*}
  ) => {
    euler_families!(@cube_step
      $(#[$meta])* $name, $params, $dt, $ct, $ct1, $ct2, $ct3, $ct4, $ct5, $ct6, $ct7, $nj, $js, $gm, $gm2, $u, $u2, $ln, $cv, $sj, $gj, $ej, $uj, $uv, $iv, $tv, $component, $produced,
      sig ($($sigx),*) ($($sigd),*),
      place ($($px),*) [] ($($restd),*) [$($restname)*],
      params [$($idx)*] [$($p)*],
      bound {$($bound)* let $head = $slot;}, arms {$($arms)*}, at [$($at)*], body {$($body)*}
    );
  };

  (@cube_step
    $(#[$meta:meta])* $name:ident, $params:ident, $dt:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $ln:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $component:ident, $produced:ident,
    sig ($($sigx:ident),*) ($($sigd:ident),*),
    place ($($px:ident),*) [] ($($pd:ident),*) [],
    params [$i:literal $($restidx:literal)*] [$head:ident $($restname:ident)*],
    bound {$($bound:tt)*}, arms {$($arms:tt)*}, at [$($at:literal)*], body {$($body:tt)*}
  ) => {
    euler_families!(@cube_step
      $(#[$meta])* $name, $params, $dt, $ct, $ct1, $ct2, $ct3, $ct4, $ct5, $ct6, $ct7, $nj, $js, $gm, $gm2, $u, $u2, $ln, $cv, $sj, $gj, $ej, $uj, $uv, $iv, $tv, $component, $produced,
      sig ($($sigx),*) ($($sigd),*),
      place ($($px),*) [] ($($pd),*) [],
      params [$($restidx)*] [$($restname)*],
      bound {$($bound)* let $head = $params[$i];}, arms {$($arms)*}, at [$($at)*], body {$($body)*}
    );
  };

  (@cube_step
    $(#[$meta:meta])* $name:ident, $params:ident, $dt:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $ln:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $component:ident, $produced:ident,
    sig ($($sigx:ident),*) ($($sigd:ident),*),
    place ($($px:ident),*) [] ($($pd:ident),*) [],
    params [$($idx:literal)*] [],
    bound {$($bound:tt)*}, arms {$($arms:tt)*}, at [$($at:literal)*],
    body {bind $n:ident = $e:expr; $($body:tt)*}
  ) => {
    euler_families!(@cube_step
      $(#[$meta])* $name, $params, $dt, $ct, $ct1, $ct2, $ct3, $ct4, $ct5, $ct6, $ct7, $nj, $js, $gm, $gm2, $u, $u2, $ln, $cv, $sj, $gj, $ej, $uj, $uv, $iv, $tv, $component, $produced,
      sig ($($sigx),*) ($($sigd),*),
      place ($($px),*) [] ($($pd),*) [],
      params [$($idx)*] [],
      bound {$($bound)* let $n = $e;}, arms {$($arms)*}, at [$($at)*], body {$($body)*}
    );
  };

  (@cube_step
    $(#[$meta:meta])* $name:ident, $params:ident, $dt:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $ln:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $component:ident, $produced:ident,
    sig ($($sigx:ident),*) ($($sigd:ident),*),
    place ($($px:ident),*) [] ($($pd:ident),*) [],
    params [$($idx:literal)*] [],
    bound {$($bound:tt)*}, arms {$($arms:tt)*}, at [$a:literal $($at:literal)*],
    body {$e:expr $(, $rest:expr)+ $(,)?}
  ) => {
    euler_families!(@cube_step
      $(#[$meta])* $name, $params, $dt, $ct, $ct1, $ct2, $ct3, $ct4, $ct5, $ct6, $ct7, $nj, $js, $gm, $gm2, $u, $u2, $ln, $cv, $sj, $gj, $ej, $uj, $uv, $iv, $tv, $component, $produced,
      sig ($($sigx),*) ($($sigd),*),
      place ($($px),*) [] ($($pd),*) [],
      params [$($idx)*] [],
      bound {$($bound)*},
      arms {$($arms)* if $component == $a { $produced = $e; }},
      at [$($at)*], body {$($rest),+}
    );
  };

  (@cube_step
    $(#[$meta:meta])* $name:ident, $params:ident, $dt:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $ln:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $component:ident, $produced:ident,
    sig ($($sigx:ident),*) ($($sigd:ident),*),
    place ($($px:ident),*) [] ($($pd:ident),*) [],
    params [$($idx:literal)*] [],
    bound {$($bound:tt)*}, arms {$($arms:tt)*}, at [$a:literal $($at:literal)*],
    body {$e:expr $(,)?}
  ) => {
    $(#[$meta])*
    #[cube]
    #[allow(non_snake_case, unused_variables)]
    pub(crate) fn $name(
      $component: u32,
      $($sigx: f32,)*
      $params: &Array<f32>,
      $dt: f32,
      $ct: f32,
      $ct1: f32,
      $ct2: f32,
      $ct3: f32,
      $ct4: f32,
      $ct5: f32,
      $ct6: f32,
      $ct7: f32,
      $nj: f32,
      $js: f32,
      $gm: f32,
      $gm2: f32,
      $u: f32,
      $u2: f32,
      $ln: f32,
      $cv: f32,
      $sj: f32,
      $gj: f32,
      $ej: f32,
      $uj: f32,
      $uv: f32,
      $iv: f32,
      $tv: f32,
      $($sigd: f32,)*
    ) -> f32 {
      $($bound)*
      let mut $produced = 0.0f32;
      $($arms)*
      if $component == $a {
        $produced = $e;
      }
      $produced
    }
  };

  (@cube_report
    $(#[$meta:meta])* $name:ident, $params:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $ln:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $component:ident, $produced:ident,
    sig ($($sigx:ident),*),
    place ($slot:ident $(, $restx:ident)*) [$head:ident $($restname:ident)*],
    params [$($idx:literal)*] [$($p:ident)*],
    bound {$($bound:tt)*}, arms {$($arms:tt)*}, at [$($at:literal)*], body {$($body:tt)*}
  ) => {
    euler_families!(@cube_report
      $(#[$meta])* $name, $params, $ct, $ct1, $ct2, $ct3, $ct4, $ct5, $ct6, $ct7, $nj, $js, $gm, $gm2, $u, $u2, $ln, $cv, $sj, $gj, $ej, $uj, $uv, $iv, $tv, $component, $produced,
      sig ($($sigx),*),
      place ($($restx),*) [$($restname)*],
      params [$($idx)*] [$($p)*],
      bound {$($bound)* let $head = $slot;}, arms {$($arms)*}, at [$($at)*], body {$($body)*}
    );
  };

  (@cube_report
    $(#[$meta:meta])* $name:ident, $params:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $ln:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $component:ident, $produced:ident,
    sig ($($sigx:ident),*),
    place ($($px:ident),*) [],
    params [$i:literal $($restidx:literal)*] [$head:ident $($restname:ident)*],
    bound {$($bound:tt)*}, arms {$($arms:tt)*}, at [$($at:literal)*], body {$($body:tt)*}
  ) => {
    euler_families!(@cube_report
      $(#[$meta])* $name, $params, $ct, $ct1, $ct2, $ct3, $ct4, $ct5, $ct6, $ct7, $nj, $js, $gm, $gm2, $u, $u2, $ln, $cv, $sj, $gj, $ej, $uj, $uv, $iv, $tv, $component, $produced,
      sig ($($sigx),*),
      place ($($px),*) [],
      params [$($restidx)*] [$($restname)*],
      bound {$($bound)* let $head = $params[$i];}, arms {$($arms)*}, at [$($at)*], body {$($body)*}
    );
  };

  (@cube_report
    $(#[$meta:meta])* $name:ident, $params:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $ln:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $component:ident, $produced:ident,
    sig ($($sigx:ident),*),
    place ($($px:ident),*) [],
    params [$($idx:literal)*] [],
    bound {$($bound:tt)*}, arms {$($arms:tt)*}, at [$($at:literal)*],
    body {bind $n:ident = $e:expr; $($body:tt)*}
  ) => {
    euler_families!(@cube_report
      $(#[$meta])* $name, $params, $ct, $ct1, $ct2, $ct3, $ct4, $ct5, $ct6, $ct7, $nj, $js, $gm, $gm2, $u, $u2, $ln, $cv, $sj, $gj, $ej, $uj, $uv, $iv, $tv, $component, $produced,
      sig ($($sigx),*),
      place ($($px),*) [],
      params [$($idx)*] [],
      bound {$($bound)* let $n = $e;}, arms {$($arms)*}, at [$($at)*], body {$($body)*}
    );
  };

  (@cube_report
    $(#[$meta:meta])* $name:ident, $params:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $ln:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $component:ident, $produced:ident,
    sig ($($sigx:ident),*),
    place ($($px:ident),*) [],
    params [$($idx:literal)*] [],
    bound {$($bound:tt)*}, arms {$($arms:tt)*}, at [$a:literal $($at:literal)*],
    body {$e:expr $(, $rest:expr)+ $(,)?}
  ) => {
    euler_families!(@cube_report
      $(#[$meta])* $name, $params, $ct, $ct1, $ct2, $ct3, $ct4, $ct5, $ct6, $ct7, $nj, $js, $gm, $gm2, $u, $u2, $ln, $cv, $sj, $gj, $ej, $uj, $uv, $iv, $tv, $component, $produced,
      sig ($($sigx),*),
      place ($($px),*) [],
      params [$($idx)*] [],
      bound {$($bound)*},
      arms {$($arms)* if $component == $a { $produced = $e; }},
      at [$($at)*], body {$($rest),+}
    );
  };

  (@cube_report
    $(#[$meta:meta])* $name:ident, $params:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $ln:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $component:ident, $produced:ident,
    sig ($($sigx:ident),*),
    place ($($px:ident),*) [],
    params [$($idx:literal)*] [],
    bound {$($bound:tt)*}, arms {$($arms:tt)*}, at [$a:literal $($at:literal)*],
    body {$e:expr $(,)?}
  ) => {
    $(#[$meta])*
    #[cube]
    #[allow(non_snake_case, unused_variables)]
    pub(crate) fn $name(
      $component: u32,
      $($sigx: f32,)*
      $params: &Array<f32>,
      $ct: f32,
      $ct1: f32,
      $ct2: f32,
      $ct3: f32,
      $ct4: f32,
      $ct5: f32,
      $ct6: f32,
      $ct7: f32,
      $nj: f32,
      $js: f32,
      $gm: f32,
      $gm2: f32,
      $u: f32,
      $u2: f32,
      $ln: f32,
      $cv: f32,
      $sj: f32,
      $gj: f32,
      $ej: f32,
      $uj: f32,
      $uv: f32,
      $iv: f32,
      $tv: f32,
    ) -> f32 {
      $($bound)*
      let mut $produced = 0.0f32;
      $($arms)*
      if $component == $a {
        $produced = $e;
      }
      $produced
    }
  };

  (@bind_params [$($idx:literal)*]) => { "" };

  (@bind_params [$i:literal $($rest_idx:literal)*] $head:ident $($rest:ident)*) => {
    concat!(
      "            const REAL ", stringify!($head), " = params[", stringify!($i), "];\n",
      euler_families!(@bind_params [$($rest_idx)*] $($rest)*)
    )
  };

  (@bind_slots $buf:literal [$($idx:literal)*]) => { "" };

  (@bind_slots $buf:literal [$i:literal $($rest_idx:literal)*] $head:ident $($rest:ident)*) => {
    concat!(
      "            const REAL ", stringify!($head), " = ", $buf, "[", stringify!($i), "];\n",
      euler_families!(@bind_slots $buf [$($rest_idx)*] $($rest)*)
    )
  };

  (@c_body $lhs:literal, [$($idx:literal)*], [$($rem:ident)*], bind $n:ident = $e:expr; $($rest:tt)*) => {
    concat!(
      "            const REAL ", stringify!($n), " = ", stringify!($e), ";\n",
      euler_families!(@c_body $lhs, [$($idx)*], [$($rem)*], $($rest)*)
    )
  };

  (@c_body $lhs:literal, [$($idx:literal)*], [$($rem:ident)*], $($e:expr),* $(,)?) => {
    concat!(
      euler_families!(@c_temps [$($idx)*] $($e),*),
      euler_families!(@c_store $lhs, [$($idx)*] $($e),*)
    )
  };

  (@c_temps [$($idx:literal)*]) => { "" };

  (@c_temps [$i:literal $($rest_idx:literal)*] $head:expr $(, $rest:expr)*) => {
    concat!(
      "            const REAL __n", stringify!($i), " = ", stringify!($head), ";\n",
      euler_families!(@c_temps [$($rest_idx)*] $($rest),*)
    )
  };

  (@c_store $lhs:literal, [$($idx:literal)*]) => { "" };

  // One store per expression, not per declared name: a family may keep more
  // state slots than it reports, and the temporaries exist per expression.
  (@c_store $lhs:literal, [$i:literal $($rest_idx:literal)*] $head:expr $(, $rest:expr)*) => {
    concat!(
      "            ", $lhs, "[", stringify!($i), "] = __n", stringify!($i), ";\n",
      euler_families!(@c_store $lhs, [$($rest_idx)*] $($rest),*)
    )
  };
}

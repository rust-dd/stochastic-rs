//! Generates host family operations and CUDA source from the shared declarations.

#[macro_use]
mod helpers;

macro_rules! euler_families {
  (
    step_inputs($params:ident, $dt:ident, $ct:ident, $ct1:ident, $ct2:ident, $ct3:ident, $ct4:ident, $ct5:ident, $ct6:ident, $ct7:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $ln:ident, $cv:ident, $sj:ident, $gj:ident, $ej:ident, $uj:ident, $uv:ident, $iv:ident, $tv:ident, $pv:ident, $pv2:ident, state $sxs:tt, noise $sds:tt, select($component:ident, $produced:ident));
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
      /// Every declared family, in code order.
      #[allow(dead_code)]
      pub(crate) const ALL: &'static [Family] = &[$(Family::$name),*];

      /// The code the kernels compare `family` against.
      #[allow(dead_code)]
      pub(crate) fn code(self) -> u32 {
        self as u32
      }

      /// Returns the family with this code, or `None` for an unknown code.
      #[allow(dead_code)]
      pub(crate) fn from_code(code: u32) -> Option<Self> {
        match code {
          $( $code => Some(Family::$name), )*
          _ => None,
        }
      }

      /// Output planes per path; auxiliary state slots need not be reported.
      #[allow(dead_code)]
      pub(crate) fn components(self) -> usize {
        match self {
          $( Family::$name => euler_family_exprs!(@count_exprs $($report)*), )*
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
          $( Family::$name => euler_family_exprs!(@history_slot $($hist)?), )*
        }
      }

      /// Whether the family declares a `series` clause: terms drawn per path
      /// before the steps and summed into the grid cells they fall in.
      #[allow(dead_code)]
      pub(crate) fn has_series(self) -> bool {
        match self {
          $( Family::$name => euler_family_exprs!(@has_series $($ser)?), )*
        }
      }

      /// Whether the family's `series` clause sizes its terms in the step of
      /// their cell, where the state is known, rather than in the preamble.
      #[allow(dead_code)]
      pub(crate) fn series_live(self) -> bool {
        match self {
          $( Family::$name => euler_family_exprs!(@series_live $($ser)?), )*
        }
      }

      /// Whether the family declares a `table` clause: a monotone table built
      /// per path before the steps, whose inverse each step reads at its time.
      #[allow(dead_code)]
      pub(crate) fn has_table(self) -> bool {
        match self {
          $( Family::$name => euler_family_exprs!(@has_table $($tab)?), )*
        }
      }

      /// Independent noise components per step, before the family's correlation transform.
      #[allow(dead_code)]
      pub(crate) fn noises(self) -> usize {
        match self {
          $( Family::$name => [$(stringify!($noise)),*].len(), )*
        }
      }

      /// Whether the jump and the gamma launch shapes change what the family
      /// reads; shapes that differ only in unread draws share one Metal kernel.
      #[allow(dead_code)]
      pub(crate) fn draws(self) -> (bool, bool) {
        match self {
          $( Family::$name => {
            let source = stringify!($($step)* $($report)* $($lift)? $($hist)? $($ser)? $($tab)?);
            let gamma = definitions::reads(source, "gm")
              || definitions::reads(source, "gm2");
            let count = definitions::reads(source, "nj")
              || definitions::reads(source, "js");
            (count || gamma, gamma)
          } )*
        }
      }
    }

    /// Evaluates every component from the pre-step state using the shared formula.
    /// Parameters, state and noise follow declaration order.
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
      $pv: T,
      $pv2: T,
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
            euler_family_exprs!(@host_assign out, $($step)*)
          }
        )*
      }
    }

    /// Evaluates reports from state and parameters in declaration order.
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
      $pv: T,
      $pv2: T,
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
            euler_family_exprs!(@host_assign out, $($report)*)
          }
        )*
      }
    }

    /// The C statements that step the state, one guarded block per family.
    #[cfg(test)]
    pub(crate) const C_STEP: &str = concat!($(
      "        if (family == ", stringify!($code), "u) {\n",
      euler_family_exprs!(@bind_params [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19] $($param)*),
      euler_family_exprs!(@bind_slots "state" [0 1 2 3] $($state)*),
      euler_family_exprs!(@bind_slots "noise" [0 1 2 3] $($noise)*),
      euler_family_exprs!(@c_body "state", [0 1 2 3], [$($state)*], $($step)*),
      "        }\n",
    )*);

    /// Guarded CUDA blocks writing each lifted family's drift, diffusion and shock.
    #[cfg(test)]
    pub(crate) const C_LIFT: &str = concat!($(
      euler_family_exprs!(@c_lift $code, [$($param)*], [$($state)*], [$($noise)*], $($lift)?),
    )*);

    /// Evaluates `[drift, diffusion, shock]`, or zeros for a family without a lift.
    #[allow(dead_code, clippy::too_many_arguments, unused_variables)]
    pub(crate) fn host_lift<T: FloatExt>(
      family: Family,
      state: &[T],
      $params: &[T],
      $dt: T,
      $pv: T,
      $pv2: T,
      noise: &[T],
    ) -> [T; 3] {
      #[allow(unused_imports)]
      use ops::*;
      match family {
        $(
          Family::$name => {
            euler_family_exprs!(@host_lift $params, $dt, state, noise, [$($param)*], [$($state)*], [$($noise)*], $($lift)?)
          }
        )*
      }
    }

    /// Guarded CUDA blocks writing the value appended to each family's history.
    #[cfg(test)]
    pub(crate) const C_HISTORY: &str = concat!($(
      euler_family_exprs!(@c_history $code, [$($param)*], [$($state)*], [$($noise)*], $($hist)?),
    )*);

    /// Evaluates the next history value, or `None` for a family without history.
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
            euler_family_exprs!(@host_history $params, $dt, $u, $u2, state, noise, [$($param)*], [$($state)*], [$($noise)*], $($hist)?)
          }
        )*
      }
    }

    /// Guarded CUDA blocks sizing a term from each series family's draws.
    #[cfg(test)]
    pub(crate) const C_SERIES: &str = concat!($(
      euler_family_exprs!(@c_series $code, [$($param)*], [$($state)*], $($ser)?),
    )*);

    /// Evaluates one series term, or `None` for a family without a series.
    #[allow(dead_code, clippy::too_many_arguments, unused_variables)]
    pub(crate) fn host_series<T: FloatExt>(
      family: Family,
      state: &[T],
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
            euler_family_exprs!(@host_series $params, $dt, $gj, $ej, $uj, $uv, state, [$($param)*], [$($state)*], $($ser)?)
          }
        )*
      }
    }

    /// Guarded CUDA blocks sizing a table increment from draws and grid spacing.
    #[cfg(test)]
    pub(crate) const C_TABLE: &str = concat!($(
      euler_family_exprs!(@c_table $code, [$($param)*], $($tab)?),
    )*);

    /// Evaluates one table increment, or `None` for a family without a table.
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
            euler_family_exprs!(@host_table $params, $dt, $uj, $uv, $tv, [$($param)*], $($tab)?)
          }
        )*
      }
    }

    /// One family's CUDA step without a runtime family-code branch.
    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn c_step_for(family: Family) -> &'static str {
      match family {
        $( Family::$name => concat!(
          euler_family_exprs!(@bind_params [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19] $($param)*),
          euler_family_exprs!(@bind_slots "state" [0 1 2 3] $($state)*),
          euler_family_exprs!(@bind_slots "noise" [0 1 2 3] $($noise)*),
          euler_family_exprs!(@c_body "state", [0 1 2 3], [$($state)*], $($step)*),
        ), )*
      }
    }

    /// One family's report, as [`c_step_for`] is its step.
    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn c_report_for(family: Family) -> &'static str {
      match family {
        $( Family::$name => concat!(
          euler_family_exprs!(@bind_params [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19] $($param)*),
          euler_family_exprs!(@bind_slots "state" [0 1 2 3] $($state)*),
          euler_family_exprs!(@c_body "reported", [0 1 2 3], [$($state)*], $($report)*),
        ), )*
      }
    }

    /// One family's CUDA lift coefficients, or `""` for a family without a lift.
    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn c_lift_for(family: Family) -> &'static str {
      match family {
        $( Family::$name => euler_family_exprs!(@bare_lift [$($param)*], [$($state)*], [$($noise)*], $($lift)?), )*
      }
    }

    /// One family's pushed history value, or `""` where it declares none.
    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn c_history_for(family: Family) -> &'static str {
      match family {
        $( Family::$name => euler_family_exprs!(@bare_history [$($param)*], [$($state)*], [$($noise)*], $($hist)?), )*
      }
    }

    /// One family's series term, or `""` where it declares none.
    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn c_series_for(family: Family) -> &'static str {
      match family {
        $( Family::$name => euler_family_exprs!(@bare_series [$($param)*], [$($state)*], $($ser)?), )*
      }
    }

    /// One family's table increment, or `""` where it declares none.
    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn c_table_for(family: Family) -> &'static str {
      match family {
        $( Family::$name => euler_family_exprs!(@bare_table [$($param)*], $($tab)?), )*
      }
    }

    #[cfg(test)]
    pub(crate) const C_REPORT: &str = concat!($(
      "        if (family == ", stringify!($code), "u) {\n",
      euler_family_exprs!(@bind_params [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19] $($param)*),
      euler_family_exprs!(@bind_slots "state" [0 1 2 3] $($state)*),
      euler_family_exprs!(@c_body "reported", [0 1 2 3], [$($state)*], $($report)*),
      "        }\n",
    )*);
  };
}

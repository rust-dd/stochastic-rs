//! The drift / diffusion families of the Euler engine, declared once.
//!
//! A family is written in a subset that is simultaneously a valid Rust
//! expression and a valid C expression: infix arithmetic, parentheses and a
//! small vocabulary of functions. [`euler_families!`] takes that one
//! declaration and emits every form the engine needs — the host step, and the
//! C body the native CUDA and Metal kernels render — from the same tokens, so
//! the operation order cannot drift between them.
//!
//! The CubeCL kernel is written by hand instead: its `#[cube]` attribute
//! cannot see through a macro expansion, and calls into a generic helper need
//! a turbofish the shared tokens must not carry. What keeps it honest is
//! `euler::family_parity`, which launches every declared family on both
//! kernels and compares them point for point, so a family missing from the
//! hand-written dispatch fails a test rather than quietly returning a flat
//! path.
//!
//! A step or report may open with `bind name = expr;` lines before its final
//! expression. Each becomes a `let` on the host and in a CubeCL kernel and a
//! `const REAL` in the emitted C, which is how a family names a clamped or
//! guarded state once and then reads like the host sampler it came from.
//!
//! A step or report may read `u` and `u2`, two uniforms in `[0, 1)` for the
//! step, `nj`, the number of jumps it saw, `js`, the sum of their sizes, `gm`
//! and `gm2`, one or two Gamma draws, and `ct`, the step's value of a time-varying
//! coefficient the host supplies as one value per grid point. A family that
//! never names it costs nothing for it.
//!
//! The CubeCL functions take the four state and four noise slots as
//! parameters and bind the family's own names from them, so those parameters
//! are named `slot_a`..`slot_d` and `shock_a`..`shock_d`: a family whose
//! state were called `x1` would otherwise shadow the slot it was being read
//! from, and every later binding would read the shadowed value.
//!
//! The names a step may use are fixed: `x` for the state, `dt` for the step
//! size, `dz` for the step's noise **increment** — `sqrt_dt · z` for Gaussian
//! noise, the fractional increment itself for fGN, which is what lets one
//! declaration serve both — and the family's own
//! parameters, which the generated code binds as locals from the parameter
//! buffer in declaration order. The `report` expression maps the state to what
//! the path records and is evaluated at `t = 0` as well, where no noise exists.
//!
//! The function vocabulary is `sqrt`, `exp`, `ln`, `pow`, `abs`, `negate`,
//! `tanh`, `atan`, `sin`, `recip`, `positive`, `max`, `min`, the literal
//! `lit`, the
//! comparisons
//! `less`, `leq` and `geq`, and the branch-free `pick`. Each has a host
//! implementation in [`ops`] and a C definition in [`C_PRELUDE`]; anything
//! outside it fails to compile on the host, which is the intended way to find
//! out that a kernel could not have run it either.

use crate::traits::FloatExt;

/// Host implementations of the function vocabulary a family step may use, so
/// `sqrt(v)` means on the host what the `#define` in [`C_PRELUDE`] makes it
/// mean in a kernel. The whole vocabulary is defined whether or not a family
/// declared so far happens to use it, which is what lets a new family reach
/// for one without touching this module.
#[allow(dead_code)]
pub(crate) mod ops {
  use crate::traits::FloatExt;

  /// `√v`
  #[inline(always)]
  pub(crate) fn sqrt<T: FloatExt>(v: T) -> T {
    v.sqrt()
  }

  /// `exp v`
  #[inline(always)]
  pub(crate) fn exp<T: FloatExt>(v: T) -> T {
    v.exp()
  }

  /// `ln v`
  #[inline(always)]
  pub(crate) fn ln<T: FloatExt>(v: T) -> T {
    v.ln()
  }

  /// `a^b`
  #[inline(always)]
  pub(crate) fn pow<T: FloatExt>(a: T, b: T) -> T {
    a.powf(b)
  }

  /// `|v|`
  #[inline(always)]
  pub(crate) fn abs<T: FloatExt>(v: T) -> T {
    v.abs()
  }

  /// `−v`. A literal may never sit on the left of an operator — the compiler
  /// cannot infer its type there — so a step that needs `c − f(x)` writes
  /// `negate(f(x) − lit(c))`, which is the same value in IEEE arithmetic.
  /// The name avoids `neg`, which `cubecl::prelude` already exports.
  #[inline(always)]
  pub(crate) fn negate<T: FloatExt>(v: T) -> T {
    T::zero() - v
  }

  /// `tanh v`
  #[inline(always)]
  pub(crate) fn tanh<T: FloatExt>(v: T) -> T {
    v.tanh()
  }

  /// `arctan v`
  #[inline(always)]
  pub(crate) fn atan<T: FloatExt>(v: T) -> T {
    v.atan()
  }

  /// `sin v`
  #[inline(always)]
  pub(crate) fn sin<T: FloatExt>(v: T) -> T {
    v.sin()
  }

  /// `1/v`. A literal may not sit on the left of an operator, so a family
  /// that needs `c / f(x)` writes `recip(f(x)) * lit(c)`.
  #[inline(always)]
  pub(crate) fn recip<T: FloatExt>(v: T) -> T {
    T::one() / v
  }

  /// The positive part, the truncation a square-root diffusion steps on.
  #[inline(always)]
  pub(crate) fn positive<T: FloatExt>(v: T) -> T {
    if v > T::zero() { v } else { T::zero() }
  }

  /// `max(a, b)`
  #[inline(always)]
  pub(crate) fn max<T: FloatExt>(a: T, b: T) -> T {
    if a > b { a } else { b }
  }

  /// `min(a, b)`
  #[inline(always)]
  pub(crate) fn min<T: FloatExt>(a: T, b: T) -> T {
    if a < b { a } else { b }
  }

  /// A numeric literal. Each target spells one differently — `T::from_f64_fast`
  /// on the host, `F::new` in a CubeCL kernel, a cast in C — so a family writes
  /// `lit(0.5)` and the emitters agree on what it means.
  #[inline(always)]
  pub(crate) fn lit<T: FloatExt>(v: f64) -> T {
    T::from_f64_fast(v)
  }

  /// `1` when `a < b`, `0` otherwise: a strict comparison, kept distinct from
  /// [`leq`] so a step can reproduce a host guard's boundary exactly.
  #[inline(always)]
  pub(crate) fn less<T: FloatExt>(a: T, b: T) -> T {
    if a < b { T::one() } else { T::zero() }
  }

  /// `1` when `a <= b`, `0` otherwise: a condition as a number, so a step
  /// stays one expression on every target.
  #[inline(always)]
  pub(crate) fn leq<T: FloatExt>(a: T, b: T) -> T {
    if a <= b { T::one() } else { T::zero() }
  }

  /// `1` when `a >= b`, `0` otherwise.
  #[inline(always)]
  pub(crate) fn geq<T: FloatExt>(a: T, b: T) -> T {
    if a >= b { T::one() } else { T::zero() }
  }

  /// `a` when `cond` is non-zero, `b` otherwise. Both arms are evaluated, so
  /// an arm that could produce a NaN guards itself.
  #[inline(always)]
  pub(crate) fn pick<T: FloatExt>(cond: T, a: T, b: T) -> T {
    if cond != T::zero() { a } else { b }
  }
}

/// The C definitions of the function vocabulary, in terms of the precision
/// placeholders the kernel renderer fills in. Each name expands to a distinct
/// intrinsic, so no definition refers to itself.
pub(crate) const C_PRELUDE: &str = r#"#define sqrt(v) STOCH_SQRT(v)
#define exp(v) STOCH_EXP(v)
#define ln(v) STOCH_LOG(v)
#define pow(a, b) STOCH_POW(a, b)
#define abs(v) STOCH_ABS(v)
#define negate(v) (-(v))
#define tanh(v) STOCH_TANH(v)
#define atan(v) STOCH_ATAN(v)
#define recip(v) ((REAL)1 / (v))
#define sin(v) STOCH_SIN(v)
#define positive(v) ((v) > (REAL)0 ? (v) : (REAL)0)
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define less(a, b) ((a) < (b) ? (REAL)1 : (REAL)0)
#define leq(a, b) ((a) <= (b) ? (REAL)1 : (REAL)0)
#define geq(a, b) ((a) >= (b) ? (REAL)1 : (REAL)0)
#define pick(c, a, b) ((c) != (REAL)0 ? (a) : (b))
#define lit(v) ((REAL)(v))
"#;

/// The CubeCL implementations of the same vocabulary. They are concrete on
/// `f32` — the only precision the CubeCL kernels compute in — so a call from a
/// generated body has nothing left to infer, which a generic helper would.
/// `max` and `min` are not defined here: `cubecl::prelude` already exports
/// them as free functions with these names.
#[cfg(feature = "cubecl")]
#[allow(dead_code)]
pub(crate) mod cube_ops {
  use cubecl::prelude::*;

  /// `√v`
  #[cube]
  pub(crate) fn sqrt(v: f32) -> f32 {
    Sqrt::sqrt(v)
  }

  /// `exp v`
  #[cube]
  pub(crate) fn exp(v: f32) -> f32 {
    Exp::exp(v)
  }

  /// `ln v`
  #[cube]
  pub(crate) fn ln(v: f32) -> f32 {
    Log::ln(v)
  }

  /// `a^b`
  #[cube]
  pub(crate) fn pow(a: f32, b: f32) -> f32 {
    Powf::powf(a, b)
  }

  /// `|v|`
  #[cube]
  pub(crate) fn abs(v: f32) -> f32 {
    Abs::abs(v)
  }

  /// `−v`.
  #[cube]
  pub(crate) fn negate(v: f32) -> f32 {
    0.0f32 - v
  }

  /// `tanh v`
  #[cube]
  pub(crate) fn tanh(v: f32) -> f32 {
    Tanh::tanh(v)
  }

  /// `arctan v`
  #[cube]
  pub(crate) fn atan(v: f32) -> f32 {
    ArcTan::atan(v)
  }

  /// `1/v`.
  #[cube]
  pub(crate) fn recip(v: f32) -> f32 {
    1.0f32 / v
  }

  /// `sin v`
  #[cube]
  pub(crate) fn sin(v: f32) -> f32 {
    Sin::sin(v)
  }

  /// The positive part, the truncation a square-root diffusion steps on.
  #[cube]
  pub(crate) fn positive(v: f32) -> f32 {
    max(v, 0.0f32)
  }

  /// A numeric literal.
  #[cube]
  pub(crate) fn lit(v: f32) -> f32 {
    v
  }

  /// `1` when `a < b`, `0` otherwise.
  #[cube]
  pub(crate) fn less(a: f32, b: f32) -> f32 {
    select(a < b, 1.0f32, 0.0f32)
  }

  /// `1` when `a <= b`, `0` otherwise.
  #[cube]
  pub(crate) fn leq(a: f32, b: f32) -> f32 {
    select(a <= b, 1.0f32, 0.0f32)
  }

  /// `1` when `a >= b`, `0` otherwise.
  #[cube]
  pub(crate) fn geq(a: f32, b: f32) -> f32 {
    select(a >= b, 1.0f32, 0.0f32)
  }

  /// `a` when `cond` is non-zero, `b` otherwise.
  #[cube]
  pub(crate) fn pick(cond: f32, a: f32, b: f32) -> f32 {
    select(cond != 0.0f32, a, b)
  }
}

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
    step_inputs($params:ident, $dt:ident, $ct:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, state $sxs:tt, noise $sds:tt, select($component:ident, $produced:ident));
    $(
    $(#[$meta:meta])*
    $code:literal => $name:ident { $($param:ident),* $(,)? }
      state ($($state:ident),+ $(,)?)
      noise ($($noise:ident),+ $(,)?)
      step { $($step:tt)* }
      report { $($report:tt)* }
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
      #[allow(dead_code)]
      pub(crate) fn components(self) -> usize {
        match self {
          $( Family::$name => [$(stringify!($state)),*].len(), )*
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
      $nj: T,
      $js: T,
      $gm: T,
      $gm2: T,
      $u: T,
      $u2: T,
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
      $nj: T,
      $js: T,
      $gm: T,
      $gm2: T,
      $u: T,
      $u2: T,
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
          $(#[$meta])* $name, $params, $dt, $ct, $nj, $js, $gm, $gm2, $u, $u2, $component, $produced,
          sig $sxs $sds,
          place $sxs [$($state)*] $sds [$($noise)*],
          params [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19] [$($param)*],
          bound {}, arms {}, at [0u32 1u32 2u32 3u32], body {$($step)*}
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
          $(#[$meta])* $name, $params, $ct, $nj, $js, $gm, $gm2, $u, $u2, $component, $produced,
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

    /// The C statements that set the reported values, one block per family.
    pub(crate) const C_REPORT: &str = concat!($(
      "        if (family == ", stringify!($code), "u) {\n",
      euler_families!(@bind_params [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19] $($param)*),
      euler_families!(@bind_slots "state" [0 1 2 3] $($state)*),
      euler_families!(@c_body "reported", [0 1 2 3], [$($state)*], $($report)*),
      "        }\n",
    )*);
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
    $(#[$meta:meta])* $name:ident, $params:ident, $dt:ident, $ct:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $component:ident, $produced:ident,
    sig ($($sigx:ident),*) ($($sigd:ident),*),
    place ($slot:ident $(, $restx:ident)*) [$head:ident $($restname:ident)*] ($($pd:ident),*) [$($nd:ident)*],
    params [$($idx:literal)*] [$($p:ident)*],
    bound {$($bound:tt)*}, arms {$($arms:tt)*}, at [$($at:literal)*], body {$($body:tt)*}
  ) => {
    euler_families!(@cube_step
      $(#[$meta])* $name, $params, $dt, $ct, $nj, $js, $gm, $gm2, $u, $u2, $component, $produced,
      sig ($($sigx),*) ($($sigd),*),
      place ($($restx),*) [$($restname)*] ($($pd),*) [$($nd)*],
      params [$($idx)*] [$($p)*],
      bound {$($bound)* let $head = $slot;}, arms {$($arms)*}, at [$($at)*], body {$($body)*}
    );
  };

  (@cube_step
    $(#[$meta:meta])* $name:ident, $params:ident, $dt:ident, $ct:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $component:ident, $produced:ident,
    sig ($($sigx:ident),*) ($($sigd:ident),*),
    place ($($px:ident),*) [] ($slot:ident $(, $restd:ident)*) [$head:ident $($restname:ident)*],
    params [$($idx:literal)*] [$($p:ident)*],
    bound {$($bound:tt)*}, arms {$($arms:tt)*}, at [$($at:literal)*], body {$($body:tt)*}
  ) => {
    euler_families!(@cube_step
      $(#[$meta])* $name, $params, $dt, $ct, $nj, $js, $gm, $gm2, $u, $u2, $component, $produced,
      sig ($($sigx),*) ($($sigd),*),
      place ($($px),*) [] ($($restd),*) [$($restname)*],
      params [$($idx)*] [$($p)*],
      bound {$($bound)* let $head = $slot;}, arms {$($arms)*}, at [$($at)*], body {$($body)*}
    );
  };

  (@cube_step
    $(#[$meta:meta])* $name:ident, $params:ident, $dt:ident, $ct:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $component:ident, $produced:ident,
    sig ($($sigx:ident),*) ($($sigd:ident),*),
    place ($($px:ident),*) [] ($($pd:ident),*) [],
    params [$i:literal $($restidx:literal)*] [$head:ident $($restname:ident)*],
    bound {$($bound:tt)*}, arms {$($arms:tt)*}, at [$($at:literal)*], body {$($body:tt)*}
  ) => {
    euler_families!(@cube_step
      $(#[$meta])* $name, $params, $dt, $ct, $nj, $js, $gm, $gm2, $u, $u2, $component, $produced,
      sig ($($sigx),*) ($($sigd),*),
      place ($($px),*) [] ($($pd),*) [],
      params [$($restidx)*] [$($restname)*],
      bound {$($bound)* let $head = $params[$i];}, arms {$($arms)*}, at [$($at)*], body {$($body)*}
    );
  };

  (@cube_step
    $(#[$meta:meta])* $name:ident, $params:ident, $dt:ident, $ct:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $component:ident, $produced:ident,
    sig ($($sigx:ident),*) ($($sigd:ident),*),
    place ($($px:ident),*) [] ($($pd:ident),*) [],
    params [$($idx:literal)*] [],
    bound {$($bound:tt)*}, arms {$($arms:tt)*}, at [$($at:literal)*],
    body {bind $n:ident = $e:expr; $($body:tt)*}
  ) => {
    euler_families!(@cube_step
      $(#[$meta])* $name, $params, $dt, $ct, $nj, $js, $gm, $gm2, $u, $u2, $component, $produced,
      sig ($($sigx),*) ($($sigd),*),
      place ($($px),*) [] ($($pd),*) [],
      params [$($idx)*] [],
      bound {$($bound)* let $n = $e;}, arms {$($arms)*}, at [$($at)*], body {$($body)*}
    );
  };

  (@cube_step
    $(#[$meta:meta])* $name:ident, $params:ident, $dt:ident, $ct:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $component:ident, $produced:ident,
    sig ($($sigx:ident),*) ($($sigd:ident),*),
    place ($($px:ident),*) [] ($($pd:ident),*) [],
    params [$($idx:literal)*] [],
    bound {$($bound:tt)*}, arms {$($arms:tt)*}, at [$a:literal $($at:literal)*],
    body {$e:expr $(, $rest:expr)+ $(,)?}
  ) => {
    euler_families!(@cube_step
      $(#[$meta])* $name, $params, $dt, $ct, $nj, $js, $gm, $gm2, $u, $u2, $component, $produced,
      sig ($($sigx),*) ($($sigd),*),
      place ($($px),*) [] ($($pd),*) [],
      params [$($idx)*] [],
      bound {$($bound)*},
      arms {$($arms)* if $component == $a { $produced = $e; }},
      at [$($at)*], body {$($rest),+}
    );
  };

  (@cube_step
    $(#[$meta:meta])* $name:ident, $params:ident, $dt:ident, $ct:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $component:ident, $produced:ident,
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
      $nj: f32,
      $js: f32,
      $gm: f32,
      $gm2: f32,
      $u: f32,
      $u2: f32,
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
    $(#[$meta:meta])* $name:ident, $params:ident, $ct:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $component:ident, $produced:ident,
    sig ($($sigx:ident),*),
    place ($slot:ident $(, $restx:ident)*) [$head:ident $($restname:ident)*],
    params [$($idx:literal)*] [$($p:ident)*],
    bound {$($bound:tt)*}, arms {$($arms:tt)*}, at [$($at:literal)*], body {$($body:tt)*}
  ) => {
    euler_families!(@cube_report
      $(#[$meta])* $name, $params, $ct, $nj, $js, $gm, $gm2, $u, $u2, $component, $produced,
      sig ($($sigx),*),
      place ($($restx),*) [$($restname)*],
      params [$($idx)*] [$($p)*],
      bound {$($bound)* let $head = $slot;}, arms {$($arms)*}, at [$($at)*], body {$($body)*}
    );
  };

  (@cube_report
    $(#[$meta:meta])* $name:ident, $params:ident, $ct:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $component:ident, $produced:ident,
    sig ($($sigx:ident),*),
    place ($($px:ident),*) [],
    params [$i:literal $($restidx:literal)*] [$head:ident $($restname:ident)*],
    bound {$($bound:tt)*}, arms {$($arms:tt)*}, at [$($at:literal)*], body {$($body:tt)*}
  ) => {
    euler_families!(@cube_report
      $(#[$meta])* $name, $params, $ct, $nj, $js, $gm, $gm2, $u, $u2, $component, $produced,
      sig ($($sigx),*),
      place ($($px),*) [],
      params [$($restidx)*] [$($restname)*],
      bound {$($bound)* let $head = $params[$i];}, arms {$($arms)*}, at [$($at)*], body {$($body)*}
    );
  };

  (@cube_report
    $(#[$meta:meta])* $name:ident, $params:ident, $ct:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $component:ident, $produced:ident,
    sig ($($sigx:ident),*),
    place ($($px:ident),*) [],
    params [$($idx:literal)*] [],
    bound {$($bound:tt)*}, arms {$($arms:tt)*}, at [$($at:literal)*],
    body {bind $n:ident = $e:expr; $($body:tt)*}
  ) => {
    euler_families!(@cube_report
      $(#[$meta])* $name, $params, $ct, $nj, $js, $gm, $gm2, $u, $u2, $component, $produced,
      sig ($($sigx),*),
      place ($($px),*) [],
      params [$($idx)*] [],
      bound {$($bound)* let $n = $e;}, arms {$($arms)*}, at [$($at)*], body {$($body)*}
    );
  };

  (@cube_report
    $(#[$meta:meta])* $name:ident, $params:ident, $ct:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $component:ident, $produced:ident,
    sig ($($sigx:ident),*),
    place ($($px:ident),*) [],
    params [$($idx:literal)*] [],
    bound {$($bound:tt)*}, arms {$($arms:tt)*}, at [$a:literal $($at:literal)*],
    body {$e:expr $(, $rest:expr)+ $(,)?}
  ) => {
    euler_families!(@cube_report
      $(#[$meta])* $name, $params, $ct, $nj, $js, $gm, $gm2, $u, $u2, $component, $produced,
      sig ($($sigx),*),
      place ($($px),*) [],
      params [$($idx)*] [],
      bound {$($bound)*},
      arms {$($arms)* if $component == $a { $produced = $e; }},
      at [$($at)*], body {$($rest),+}
    );
  };

  (@cube_report
    $(#[$meta:meta])* $name:ident, $params:ident, $ct:ident, $nj:ident, $js:ident, $gm:ident, $gm2:ident, $u:ident, $u2:ident, $component:ident, $produced:ident,
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
      $nj: f32,
      $js: f32,
      $gm: f32,
      $gm2: f32,
      $u: f32,
      $u2: f32,
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
      euler_families!(@c_store $lhs, [$($idx)*] $($rem)*)
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

  (@c_store $lhs:literal, [$i:literal $($rest_idx:literal)*] $head:ident $($rest:ident)*) => {
    concat!(
      "            ", $lhs, "[", stringify!($i), "] = __n", stringify!($i), ";\n",
      euler_families!(@c_store $lhs, [$($rest_idx)*] $($rest)*)
    )
  };
}

euler_families! {
  step_inputs(
    params, dt, ct, nj, js, gm, gm2, u, u2,
    state(slot_a, slot_b, slot_c, slot_d),
    noise(shock_a, shock_b, shock_c, shock_d),
    select(component, produced)
  );

  /// `dX = μX dt + σX dW`.
  0 => GeometricBrownian { mu, sigma }
    state (x)
    noise (dz)
    step { x + mu * x * dt + sigma * x * dz }
    report { x },

  /// `dX = θ(μ − X) dt + σ dW`.
  1 => OrnsteinUhlenbeck { theta, mu, sigma }
    state (x)
    noise (dz)
    step { x + theta * (mu - x) * dt + sigma * dz }
    report { x },

  /// `dX = κ(θ − X) dt + σ√X dW`, stepped with full truncation (Lord,
  /// Koekkoek & van Dijk 2010): the recursion runs on an auxiliary state
  /// whose positive part enters drift, diffusion and the reported path.
  /// `dX = dW`: the increment accumulates, which is fractional Brownian
  /// motion when the increments come from an fGN pipeline.
  3 => Additive { }
    state (x)
    noise (dz)
    step { x + dz }
    report { x },

  /// `dX = θ(μ − X) dt + σ√|X| dW`, clamped at zero after the step — the
  /// fractional CIR recursion, which truncates the *result* rather than
  /// stepping on a truncated state.
  4 => ReflectedSquareRoot { theta, mu, sigma }
    state (x)
    noise (dz)
    step { positive(x + theta * (mu - x) * dt + sigma * sqrt(abs(x)) * dz) }
    report { x },

  /// The same with the symmetric reflection: the step's absolute value.
  5 => MirroredSquareRoot { theta, mu, sigma }
    state (x)
    noise (dz)
    step { abs(x + theta * (mu - x) * dt + sigma * sqrt(abs(x)) * dz) }
    report { x },

  /// `dX = (α − βX) dt + σ√(X(1−X)) dW` on the unit interval, absorbing at
  /// both ends — the fractional Jacobi recursion. `X(1−X)` is written
  /// `x - x * x`, which needs no literal and so no type to infer, and both
  /// arms of a `pick` are evaluated, so the diffusion guards its own root.
  6 => Jacobi { alpha, beta, sigma }
    state (x)
    noise (dz)
    step {
      pick(
        leq(x, lit(0.0)),
        lit(0.0),
        pick(
          geq(x, lit(1.0)),
          lit(1.0),
          x + (alpha - beta * x) * dt + sigma * sqrt(positive(x - x * x)) * dz
        )
      )
    }
    report { x },

  /// `dX = μX dt + σ|X|^γ dW`: constant elasticity of variance, the power
  /// taken off `|X|` so a negative excursion stays defined.
  7 => ConstantElasticity { mu, sigma, gamma }
    state (x)
    noise (dz)
    step { x + mu * x * dt + sigma * pow(abs(x), gamma) * dz }
    report { x },

  /// `dX = (θ₁ + θ₂X) dt + θ₃|X|^θ₄ dW`: the Chan–Karolyi–Longstaff–Sanders
  /// family, whose four parameters nest most one-factor short-rate models.
  8 => Ckls { theta1, theta2, theta3, theta4 }
    state (x)
    noise (dz)
    step { x + (theta1 + theta2 * x) * dt + theta3 * pow(abs(x), theta4) * dz }
    report { x },

  /// `dX = X(1 − aX) dt + bX dW`: logistic growth with multiplicative noise.
  /// `X(1 − aX)` is written `x - a * x * x`, which needs no literal.
  9 => Logistic { a, b }
    state (x)
    noise (dz)
    step { x + (x - a * x * x) * dt + b * x * dz }
    report { x },

  /// `dX = κX(μ − X) dt + σ|X|^{3/2} dW`: the 3/2 model, whose variance
  /// mean-reverts faster than a square-root diffusion.
  10 => ThreeHalf { kappa, mu, sigma }
    state (x)
    noise (dz)
    step { x + kappa * x * (mu - x) * dt + sigma * pow(abs(x), lit(1.5)) * dz }
    report { x },

  /// Geometric Brownian motion stepped in logs: `X ← X·exp(m + σ dW)`, with
  /// `m = (μ − σ²/2)Δt` computed once on the host, so the kernel needs no
  /// literal and the exponential is exact rather than a first-order step.
  11 => LogGeometric { drift_ln, sigma }
    state (x)
    noise (dz)
    step { x * exp(drift_ln + sigma * dz) }
    report { x },

  /// `dX = (κ/X − X) dt + σ dW`: the radial Ornstein–Uhlenbeck process, whose
  /// drift is guarded away from the origin exactly as the host sampler guards
  /// it.
  12 => RadialOrnsteinUhlenbeck { kappa, sigma }
    state (x)
    noise (dz)
    step {
      x + (kappa / pick(leq(abs(x), lit(1e-12)), lit(1e-12), x) - x) * dt + sigma * dz
    }
    report { x },

  /// `dX = (a + bX) dt + cX dW`: the linear scalar SDE.
  13 => LinearSde { a, b, c }
    state (x)
    noise (dz)
    step { x + (a + b * x) * dt + c * x * dz }
    report { x },

  /// `dX = −κX/√(1+X²) dt + σ dW`: a hyperbolic drift, bounded in `X`.
  14 => Hyperbolic { kappa, sigma }
    state (x)
    noise (dz)
    step { x - kappa * x / sqrt(x * x + lit(1.0)) * dt + sigma * dz }
    report { x },

  /// `dX = −κX dt + σ√(1+X²) dW`: the modified CIR process, whose diffusion
  /// never vanishes.
  15 => ModifiedSquareRoot { kappa, sigma }
    state (x)
    noise (dz)
    step { x - kappa * x * dt + sigma * sqrt(x * x + lit(1.0)) * dz }
    report { x },

  /// `dX = X(θ₁ − X(θ₃³ − θ₁θ₂)) dt + θ₃|X|^{3/2} dW`: the Feller root
  /// process, with the drift's constant folded on the host.
  16 => FellerRoot { theta1, decay, theta3 }
    state (x)
    noise (dz)
    step { x + x * (theta1 - x * decay) * dt + theta3 * pow(abs(x), lit(1.5)) * dz }
    report { x },

  /// `dX = (a₋₁/X + a₀ + a₁X + a₂X²) dt + √|b₀ + b₁X + b₂|X|^{b₃}| dW`: the
  /// Aït-Sahalia short-rate model, whose drift is guarded away from the origin
  /// exactly as the host sampler guards it.
  17 => AitSahalia { am1, a0, a1, a2, b0, b1, b2, b3 }
    state (x)
    noise (dz)
    step {
      x + (am1 / pick(less(abs(x), lit(1e-12)), lit(1e-12), x)
        + a0 + a1 * x + a2 * x * x) * dt
        + sqrt(abs(b0 + b1 * x + b2 * pow(abs(x), b3))) * dz
    }
    report { x },

  /// `dX = (a − b·ln X)X dt + σX dW`, floored at `1e-12`. The step's own
  /// floor makes the state positive from the first step on, and the process
  /// floors `X₀` the same way, so the guard the host applies to every
  /// coefficient is already true of `x` here.
  18 => Gompertz { a, b, sigma }
    state (x)
    noise (dz)
    step { max(x + (a - b * ln(x)) * x * dt + sigma * x * dz, lit(1e-12)) }
    report { x },

  /// `dX = aX(1−X) dt + σ√(X(1−X)) dW` on `[0, 1]`: the Kimura diffusion of
  /// population genetics. As with [`Gompertz`](Family::Gompertz) the step's
  /// own clamp is what keeps the coefficients in range.
  19 => Kimura { a, sigma }
    state (x)
    noise (dz)
    step {
      bind xi = min(max(x, lit(0.0)), lit(1.0));
      min(
        max(
          xi + a * xi * negate(xi - lit(1.0)) * dt
            + sigma * sqrt(xi * negate(xi - lit(1.0))) * dz,
          lit(0.0)
        ),
        lit(1.0)
      )
    }
    report { x },

  /// `dX = (α + βX + γX²) dt + σX dW`: a quadratic drift with proportional
  /// noise.
  20 => Quadratic { alpha, beta, gamma, sigma }
    state (x)
    noise (dz)
    step { x + (alpha + beta * x + gamma * x * x) * dt + sigma * x * dz }
    report { x },

  /// `dX = κ(μ − X) dt + √|2κ(aX² + bX + c)| dW`: the Pearson diffusion
  /// family. `2κ` is folded on the host so the step needs no literal.
  21 => Pearson { kappa, mu, a, b, c, two_kappa }
    state (x)
    noise (dz)
    step {
      x + kappa * (mu - x) * dt + sqrt(abs(two_kappa * (a * x * x + b * x + c))) * dz
    }
    report { x },

  /// `dX = rX(1 − X/K) dt + σX dW`: logistic growth in its Verhulst
  /// parametrisation, run unclamped.
  22 => Verhulst { r, k, sigma }
    state (x)
    noise (dz)
    step { x + r * x * ((k - x) / k) * dt + sigma * x * dz }
    report { x },

  /// [`Verhulst`](Family::Verhulst) with the state confined to `[0, K]`.
  23 => VerhulstClamped { r, k, sigma }
    state (x)
    noise (dz)
    step { min(max(x + r * x * ((k - x) / k) * dt + sigma * x * dz, lit(0.0)), k) }
    report { x },

  /// `dX = κ(θ − X)X dt + σ√X dW`: Feller's logistic diffusion, truncated at
  /// zero.
  24 => FellerLogistic { kappa, theta, sigma }
    state (x)
    noise (dz)
    step {
      bind xi = positive(x);
      positive(xi + kappa * (theta - xi) * xi * dt + sigma * sqrt(xi) * dz)
    }
    report { x },

  /// [`FellerLogistic`](Family::FellerLogistic) reflected at zero instead of
  /// truncated.
  25 => FellerLogisticReflected { kappa, theta, sigma }
    state (x)
    noise (dz)
    step {
      bind xi = positive(x);
      abs(xi + kappa * (theta - xi) * xi * dt + sigma * sqrt(xi) * dz)
    }
    report { x },

  /// `dX = δ dt + 2√|X| dW`: the squared-Bessel recursion, truncated at zero.
  26 => SquaredBesselState { delta, two }
    state (x)
    noise (dz)
    step { positive(x + delta * dt + two * sqrt(abs(x)) * dz) }
    report { x },

  /// [`SquaredBesselState`](Family::SquaredBesselState) reflected at zero.
  27 => SquaredBesselStateReflected { delta, two }
    state (x)
    noise (dz)
    step { abs(x + delta * dt + two * sqrt(abs(x)) * dz) }
    report { x },

  /// [`SquaredBesselState`](Family::SquaredBesselState) reporting `√X`: the
  /// Bessel process itself, stepped in squared space so the `(δ−1)/2X`
  /// singularity never enters the recursion.
  28 => BesselFromSquared { delta, two }
    state (x)
    noise (dz)
    step { positive(x + delta * dt + two * sqrt(abs(x)) * dz) }
    report { sqrt(x) },

  /// [`BesselFromSquared`](Family::BesselFromSquared) reflected at zero.
  29 => BesselFromSquaredReflected { delta, two }
    state (x)
    noise (dz)
    step { abs(x + delta * dt + two * sqrt(abs(x)) * dz) }
    report { sqrt(x) },

  /// `dX = ½σ²(β − γ(X−μ)/√(δ² + (X−μ)²)) dt + σ dW`: the hyperbolic
  /// diffusion whose stationary law is the hyperbolic distribution. `½σ²` is
  /// folded on the host.
  30 => HyperbolicDiffusion { beta, gamma, delta, mu, sigma, half_var }
    state (x)
    noise (dz)
    step {
      x + half_var * (beta - gamma * x / sqrt(delta * delta + (x - mu) * (x - mu))) * dt
        + sigma * dz
    }
    report { x },

  /// `dX = (a₋₁/X + a₀ + a₁X + a₂X²) dt + (b₀ + b₁X + b₂|X|^{b₃}) dW`: the
  /// Aït-Sahalia drift with the diffusion left unsquared, guarded away from
  /// the origin exactly as the host sampler guards it.
  31 => NonLinear { am1, a0, a1, a2, b0, b1, b2, b3 }
    state (x)
    noise (dz)
    step {
      x + (am1 / pick(less(abs(x), lit(1e-12)), lit(1e-12), x)
        + a0 + a1 * x + a2 * x * x) * dt
        + (b0 + b1 * x + b2 * pow(abs(x), b3)) * dz
    }
    report { x },

  /// Geometric Brownian motion on the shifted variable `Y = S + β`, reported
  /// as `Y − β`: the displaced diffusion. The shift lives in the report, so
  /// the step is the geometric one term for term.
  32 => Displaced { mu, sigma, beta }
    state (x)
    noise (dz)
    step { x + mu * x * dt + sigma * x * dz }
    report { x - beta },

  /// `dX = κ(μ − tanh X) dt + σ dW` reported as `tanh X`: Teng's stochastic
  /// correlation process, stepped on the unbounded variable so the reported
  /// correlation stays in `(−1, 1)` by construction.
  33 => TanhOrnsteinUhlenbeck { kappa, mu, sigma }
    state (x)
    noise (dz)
    step { x + kappa * (mu - tanh(x)) * dt + sigma * dz }
    report { tanh(x) },

  /// `dρ = κ(μ − ρ) dt + σ√(1 − ρ²) dW` confined to `[−0.9999, 0.9999]`: the
  /// Van Emmerich stochastic correlation process.
  34 => BoundedCorrelation { kappa, mu, sigma }
    state (x)
    noise (dz)
    step {
      min(
        max(
          x + kappa * (mu - x) * dt + sigma * sqrt(positive(negate(x * x - lit(1.0)))) * dz,
          lit(-0.9999)
        ),
        lit(0.9999)
      )
    }
    report { x },

  /// `dS = μS dt + S√V dW`, `dV = κ(θ − V) dt + σV^p dB` with `corr(W, B) = ρ`
  /// and the variance truncated at zero: the Heston model under its Euler
  /// scheme. The two noise components are drawn independently and correlated
  /// here, which is what the host sampler does with its own pair.
  35 => Heston { mu, kappa, theta, sigma, rho, pow_v }
    state (s, v)
    noise (dw, dz)
    step {
      bind vp = positive(v);
      bind db = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dz;
      s + mu * s * dt + s * sqrt(vp) * dw,
      positive(v + kappa * (theta - vp) * dt + sigma * pow(vp, pow_v) * db)
    }
    report { s, v },

  /// [`Heston`](Family::Heston) with the variance reflected at zero instead
  /// of truncated.
  36 => HestonReflected { mu, kappa, theta, sigma, rho, pow_v }
    state (s, v)
    noise (dw, dz)
    step {
      bind vp = positive(v);
      bind db = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dz;
      s + mu * s * dt + s * sqrt(vp) * dw,
      abs(v + kappa * (theta - vp) * dt + sigma * pow(vp, pow_v) * db)
    }
    report { s, v },

  /// `dF = α F^β dW`, `dα = ν α dB` with `corr(W, B) = ρ`: SABR, with the
  /// volatility stepped by its exact log-normal solution so it stays
  /// positive. `½ν²` is folded on the host.
  37 => Sabr { beta, rho, nu, half_nu_sq }
    state (f, v)
    noise (dw, dz)
    step {
      bind fp = positive(f);
      bind vp = positive(v);
      bind db = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dz;
      f + vp * pow(fp, beta) * dw,
      vp * exp(nu * db - half_nu_sq * dt)
    }
    report { f, v },

  /// `dS = rS dt + S√V dW`, `V_t = V₀ exp(ν Z_t − ½ν² t)` with `Z` the
  /// running sum of the correlated increments: the Bergomi variance is a
  /// function of that sum, so the sum and the elapsed time are stepped as
  /// state of their own rather than recomputed from the whole history.
  38 => Bergomi { r, nu, half_nu_sq, v0_sq, rho }
    state (s, v, z, elapsed)
    noise (dw, dq)
    step {
      bind db = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq;
      s + r * s * dt + sqrt(v) * s * dw,
      v0_sq * exp(nu * (z + db) - half_nu_sq * (elapsed + dt)),
      z + db,
      elapsed + dt
    }
    report { s, v, z, elapsed },

  /// Two Ornstein–Uhlenbeck factors on one clock, a slow one and a fast one:
  /// the Fouque–Papanicolaou–Sircar two-scale volatility driver.
  39 => TwoScaleOrnsteinUhlenbeck { kappa, theta, eps, alpha, eps_inv, sqrt_eps_inv }
    state (x, y)
    noise (dx, dy)
    step {
      x + kappa * (theta - x) * dt + eps * dx,
      y + eps_inv * (alpha - y) * dt + sqrt_eps_inv * dy
    }
    report { x, y },

  /// The Heston model stepped in log-price: `S` advances by the exponential
  /// of its log increment, so it stays positive whatever the variance does.
  /// The variance is truncated at zero and, unlike the arithmetic form, the
  /// truncated value is what the next step starts from.
  40 => LogHeston { drift, kappa, theta, xi, rho }
    state (s, v)
    noise (dw, dq)
    step {
      bind vp = positive(v);
      bind sv = sqrt(vp);
      bind dwv = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq;
      s * exp((drift - vp / lit(2.0)) * dt + sv * dw),
      positive(vp + kappa * (theta - vp) * dt + xi * sv * dwv)
    }
    report { s, v },

  /// [`LogHeston`](Family::LogHeston) with the variance reflected at zero.
  41 => LogHestonReflected { drift, kappa, theta, xi, rho }
    state (s, v)
    noise (dw, dq)
    step {
      bind vp = abs(v);
      bind sv = sqrt(vp);
      bind dwv = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq;
      s * exp((drift - vp / lit(2.0)) * dt + sv * dw),
      abs(vp + kappa * (theta - vp) * dt + xi * sv * dwv)
    }
    report { s, v },

  /// Two independent square-root variance factors driving one spot, each with
  /// its own correlation to the spot's own shock: the double Heston model,
  /// with both variances truncated at zero.
  42 => DoubleHeston {
    mu, kappa1, theta1, sigma1, rho1, kappa2, theta2, sigma2, rho2
  }
    state (s, v1, v2)
    noise (ds1, dq1, ds2, dq2)
    step {
      bind p1 = positive(v1);
      bind p2 = positive(v2);
      bind dv1 = rho1 * ds1 + sqrt(negate(rho1 * rho1 - lit(1.0))) * dq1;
      bind dv2 = rho2 * ds2 + sqrt(negate(rho2 * rho2 - lit(1.0))) * dq2;
      s + mu * s * dt + s * sqrt(p1) * ds1 + s * sqrt(p2) * ds2,
      positive(v1 + kappa1 * (theta1 - p1) * dt + sigma1 * sqrt(p1) * dv1),
      positive(v2 + kappa2 * (theta2 - p2) * dt + sigma2 * sqrt(p2) * dv2)
    }
    report { s, v1, v2 },

  /// [`DoubleHeston`](Family::DoubleHeston) with both variances reflected.
  43 => DoubleHestonReflected {
    mu, kappa1, theta1, sigma1, rho1, kappa2, theta2, sigma2, rho2
  }
    state (s, v1, v2)
    noise (ds1, dq1, ds2, dq2)
    step {
      bind p1 = positive(v1);
      bind p2 = positive(v2);
      bind dv1 = rho1 * ds1 + sqrt(negate(rho1 * rho1 - lit(1.0))) * dq1;
      bind dv2 = rho2 * ds2 + sqrt(negate(rho2 * rho2 - lit(1.0))) * dq2;
      s + mu * s * dt + s * sqrt(p1) * ds1 + s * sqrt(p2) * ds2,
      abs(v1 + kappa1 * (theta1 - p1) * dt + sigma1 * sqrt(p1) * dv1),
      abs(v2 + kappa2 * (theta2 - p2) * dt + sigma2 * sqrt(p2) * dv2)
    }
    report { s, v1, v2 },

  /// A Heston spot whose correlation to its own variance is itself a
  /// mean-reverting process, stepped on the unbounded variable and reported
  /// through a `tanh`. The log increment reads the correlation *after* its
  /// own step, so the third component is computed once and used by both.
  44 => StochasticCorrelationHeston {
    kappa_r, mu_r, sigma_r, kappa_v, mu_v, sigma_v, r, rho2
  }
    state (s, v, x)
    noise (dv_w, drho, dx_w)
    step {
      bind vp = positive(v);
      bind sv = sqrt(vp);
      bind xc = x + kappa_r * (mu_r - tanh(x)) * dt + sigma_r * drho;
      bind rt = tanh(xc);
      bind indep = sqrt(positive(negate(rt * rt + rho2 * rho2 - lit(1.0))));
      s * exp(
        (r - vp / lit(2.0)) * dt
          + rt * sv * dv_w
          + rho2 * sv * drho
          + indep * sv * dx_w
      ),
      positive(vp + kappa_v * (mu_v - vp) * dt + sigma_v * sv * dv_w),
      xc
    }
    report { s, v, tanh(x) },

  /// `dr = (θ(t) − αr) dt + σ dW`: Hull-White, whose mean-reversion level is
  /// the time-varying coefficient the launch carries.
  45 => HullWhite { alpha, sigma }
    state (x)
    noise (dz)
    step { x + (ct - alpha * x) * dt + sigma * dz }
    report { x },

  /// `dr = θ(t) dt + σ dW`: a drift that is entirely the curve, which is Ho-Lee
  /// under either of its two drift forms.
  46 => CurveDrift { sigma }
    state (x)
    noise (dz)
    step { x + ct * dt + sigma * dz }
    report { x },

  /// The exact one-step Ornstein–Uhlenbeck transition in log space, reported
  /// exponentiated: Black-Karasinski. `decay` is `exp(−a·dt)` and the noise
  /// scale folds the exact transition standard deviation, both of which
  /// depend on `dt` alone.
  47 => LogMeanReverting { decay, a, sigma_eff }
    state (y)
    noise (dz)
    step { y * decay + (ct / a) * negate(decay - lit(1.0)) + sigma_eff * dz }
    report { exp(y) },

  /// A square-root diffusion shifted by a deterministic curve, truncated at
  /// zero before the shift: CIR++.
  48 => ShiftedSquareRoot { theta, mu, sigma }
    state (x)
    noise (dz)
    step { positive(x + theta * (mu - x) * dt + sigma * sqrt(abs(x)) * dz) }
    report { x + ct },

  /// [`ShiftedSquareRoot`](Family::ShiftedSquareRoot) reflected at zero.
  49 => ShiftedSquareRootMirrored { theta, mu, sigma }
    state (x)
    noise (dz)
    step { abs(x + theta * (mu - x) * dt + sigma * sqrt(abs(x)) * dz) }
    report { x + ct },

  /// `dX = μX dt + σ(t)X dW`: geometric Brownian motion over a term
  /// structure of volatilities.
  50 => TimeVaryingGeometricBrownian { mu }
    state (x)
    noise (dz)
    step { x + mu * x * dt + ct * x * dz }
    report { x },

  /// Two Brownian motions correlated by `ρ`: the pair every two-factor model
  /// here draws its shocks from, as a process in its own right.
  51 => CorrelatedBrownian { rho }
    state (a, b)
    noise (dw, dq)
    step {
      a + dw,
      b + rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq
    }
    report { a, b },

  /// A Brownian bridge from `X₀` to `xt`, stepped by the exact conditional
  /// law rather than by Euler's own variance. The curve carries `1/(T − s)`,
  /// from which both the drift and the per-step variance ratio follow; at the
  /// last step that ratio is zero and the drift is the whole remaining gap,
  /// so the path lands on `xt` exactly rather than by a diffusion kick.
  52 => BrownianBridge { xt, sigma }
    state (x)
    noise (dz)
    step {
      x + (xt - x) * ct * dt + sigma * sqrt(positive(negate(dt * ct - lit(1.0)))) * dz
    }
    report { x },

  /// The two-factor Hull-White model: a short rate pulled toward the curve
  /// and a second, zero-reverting factor added to its drift.
  53 => TwoFactorHullWhite { a, b, sigma1, sigma2, rho }
    state (x, u)
    noise (dw, dq)
    step {
      bind du = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq;
      x + (ct + u - a * x) * dt + sigma1 * dw,
      u - b * u * dt + sigma2 * du
    }
    report { x, u },

  /// Two independent square-root factors whose sum, shifted by the curve, is
  /// the reported short rate: the two-factor CIR model. Each factor chooses
  /// reflection or truncation through a flag rather than a family of its own,
  /// since the pair would otherwise need four.
  54 => TwoFactorSquareRoot {
    theta1, mu1, sigma1, theta2, mu2, sigma2, sym1, sym2
  }
    state (a, b)
    noise (dw, dq)
    step {
      bind v1 = a + theta1 * (mu1 - a) * dt + sigma1 * sqrt(abs(a)) * dw;
      bind v2 = b + theta2 * (mu2 - b) * dt + sigma2 * sqrt(abs(b)) * dq;
      pick(sym1, abs(v1), positive(v1)),
      pick(sym2, abs(v2), positive(v2))
    }
    report { a + b + ct, b },

  /// The Duffie-Kan two-factor affine model: both factors drift affinely in
  /// the pair and share one affine volatility.
  55 => DuffieKan {
    a1, b1, c1, sigma1, a2, b2, c2, sigma2, alpha, beta, gamma, rho
  }
    state (r, x)
    noise (dw, dq)
    step {
      bind dx = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq;
      bind vol = alpha * r + beta * x + gamma;
      r + (a1 * r + b1 * x + c1) * dt + sigma1 * vol * dw,
      x + (a2 * r + b2 * x + c2) * dt + sigma2 * vol * dx
    }
    report { r, x },

  /// Two Heston assets under one 4×4 Cholesky factor: both log-prices and
  /// both variances step together, so every cross-correlation the factor
  /// encodes is present in one launch. The variances are truncated at zero.
  56 => TwoAssetHeston {
    mu1, mu2, kappa1, theta1, sigma1, kappa2, theta2, sigma2,
    l11, l21, l22, l31, l32, l33, l41, l42, l43, l44
  }
    state (x1, v1, x2, v2)
    noise (e1, e2, e3, e4)
    step {
      bind dz1 = l11 * e1;
      bind dz2 = l21 * e1 + l22 * e2;
      bind dw1 = l31 * e1 + l32 * e2 + l33 * e3;
      bind dw2 = l41 * e1 + l42 * e2 + l43 * e3 + l44 * e4;
      bind p1 = positive(v1);
      bind p2 = positive(v2);
      x1 + (mu1 - p1 / lit(2.0)) * dt + sqrt(p1) * dw1,
      positive(v1 + kappa1 * (theta1 - p1) * dt + sigma1 * sqrt(p1) * dz1),
      x2 + (mu2 - p2 / lit(2.0)) * dt + sqrt(p2) * dw2,
      positive(v2 + kappa2 * (theta2 - p2) * dt + sigma2 * sqrt(p2) * dz2)
    }
    report { x1, v1, x2, v2 },

  /// [`TwoAssetHeston`](Family::TwoAssetHeston) with both variances
  /// reflected at zero.
  57 => TwoAssetHestonReflected {
    mu1, mu2, kappa1, theta1, sigma1, kappa2, theta2, sigma2,
    l11, l21, l22, l31, l32, l33, l41, l42, l43, l44
  }
    state (x1, v1, x2, v2)
    noise (e1, e2, e3, e4)
    step {
      bind dz1 = l11 * e1;
      bind dz2 = l21 * e1 + l22 * e2;
      bind dw1 = l31 * e1 + l32 * e2 + l33 * e3;
      bind dw2 = l41 * e1 + l42 * e2 + l43 * e3 + l44 * e4;
      bind p1 = positive(v1);
      bind p2 = positive(v2);
      x1 + (mu1 - p1 / lit(2.0)) * dt + sqrt(p1) * dw1,
      abs(v1 + kappa1 * (theta1 - p1) * dt + sigma1 * sqrt(p1) * dz1),
      x2 + (mu2 - p2 / lit(2.0)) * dt + sqrt(p2) * dw2,
      abs(v2 + kappa2 * (theta2 - p2) * dt + sigma2 * sqrt(p2) * dz2)
    }
    report { x1, v1, x2, v2 },

  /// `d ln S = (μ − λκ − ½σ²) dt + σ dW + Σ Y_i`, the jump sizes lognormal:
  /// Merton's jump diffusion in log-price. The jump sum is the kernel's, from
  /// the size law the process declares.
  58 => MertonJumpLog { drift_ln, sigma }
    state (x)
    noise (dw)
    step { x * exp(drift_ln + sigma * dw + js) }
    report { x },

  /// A Heston variance under a log-price that also jumps: the Bates
  /// stochastic-volatility jump model. The compensated drift is folded on the
  /// host, and the jump sizes aggregate into one normal draw as they do for
  /// [`MertonJumpLog`](Family::MertonJumpLog). The variance is truncated at
  /// zero and the truncated value is what the next step starts from.
  59 => BatesJump { drift_c, alpha, beta, sigma, rho }
    state (s, v)
    noise (dw, dq)
    step {
      bind vp = positive(v);
      bind sv = sqrt(vp);
      bind dv = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq;
      s * exp((drift_c - vp / lit(2.0)) * dt + sv * dw + js),
      positive(vp + (alpha - beta * vp) * dt + sigma * sv * dv)
    }
    report { s, v },

  /// [`BatesJump`](Family::BatesJump) with the variance reflected at zero.
  60 => BatesJumpReflected { drift_c, alpha, beta, sigma, rho }
    state (s, v)
    noise (dw, dq)
    step {
      bind vp = abs(v);
      bind sv = sqrt(vp);
      bind dv = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq;
      s * exp((drift_c - vp / lit(2.0)) * dt + sv * dw + js),
      abs(vp + (alpha - beta * vp) * dt + sigma * sv * dv)
    }
    report { s, v },

  /// Andersen's quadratic-exponential Heston step: the variance is drawn from
  /// a moment-matched law rather than stepped, so the spot needs no correlated
  /// Brownian pair — the correlation lives in the `k` constants, all of which
  /// depend on the parameters and `dt` alone and so are folded on the host.
  ///
  /// Both branches are evaluated and one is selected, which is what a kernel
  /// can do. The quadratic branch's square root is guarded because it is
  /// evaluated on the exponential branch's side too, where its argument is
  /// negative; the selected value is never the guarded one.
  61 => AndersenQe { theta, e_kd, c1, c2, k0, k1, k2, k34, mu }
    state (y, v)
    noise (dzv, dz)
    step {
      bind omekd = negate(e_kd - lit(1.0));
      bind m = theta + (v - theta) * e_kd;
      bind s2 = v * c1 * omekd + c2 * omekd * omekd;
      bind psi = s2 / (m * m);
      bind inv = recip(psi) * lit(2.0);
      bind b2 = inv - lit(1.0) + sqrt(positive(inv * (inv - lit(1.0))));
      bind zv = dzv / sqrt(dt);
      bind quad = m / (b2 + lit(1.0)) * (sqrt(b2) + zv) * (sqrt(b2) + zv);
      bind p = (psi - lit(1.0)) / (psi + lit(1.0));
      bind tail = ln(negate(p - lit(1.0)) / negate(u - lit(1.0))) * m
        * recip(negate(p - lit(1.0)));
      bind vn = pick(leq(psi, lit(1.5)), quad, pick(leq(u, p), lit(0.0), tail));
      y + mu * dt + k0 + k1 * v + k2 * vn + sqrt(positive(k34 * (v + vn))) * (dz / sqrt(dt)),
      vn
    }
    report { exp(y), v },

  /// A Poisson counting process on the grid: the step adds the jumps it saw.
  62 => CountingProcess { }
    state (x)
    noise (dz)
    step { x + nj }
    report { x },

  /// An inverse-Gaussian subordinator: each increment is one Michael-Schucany-
  /// Haas draw, which needs a standard normal and a uniform and no rejection,
  /// so it is one expression. `2λ` and `4μλ` depend on the parameters and `dt`
  /// alone and are folded on the host.
  63 => InverseGaussianSubordinator { mu_ig, two_lam, four_mu_lam }
    state (x)
    noise (dz)
    step {
      bind w = (dz / sqrt(dt)) * (dz / sqrt(dt));
      bind rad = sqrt(four_mu_lam * w + mu_ig * mu_ig * w * w);
      bind xr = mu_ig + mu_ig * mu_ig * w / two_lam - mu_ig / two_lam * rad;
      x + pick(less(u, mu_ig / (mu_ig + xr)), xr, mu_ig * mu_ig / xr)
    }
    report { x },

  /// Brownian motion subordinated by an inverse-Gaussian clock: the normal
  /// inverse Gaussian process. The clock's draw is the same one
  /// [`InverseGaussianSubordinator`](Family::InverseGaussianSubordinator)
  /// takes, and the second noise component is the Brownian shock it scales.
  64 => NormalInverseGaussian { theta, sigma, mu_ig, two_lam, four_mu_lam }
    state (x)
    noise (dz, dq)
    step {
      bind w = (dz / sqrt(dt)) * (dz / sqrt(dt));
      bind rad = sqrt(four_mu_lam * w + mu_ig * mu_ig * w * w);
      bind xr = mu_ig + mu_ig * mu_ig * w / two_lam - mu_ig / two_lam * rad;
      bind ig = pick(less(u, mu_ig / (mu_ig + xr)), xr, mu_ig * mu_ig / xr);
      x + theta * ig + sigma * sqrt(ig) * (dq / sqrt(dt))
    }
    report { x },

  /// A positive-stable subordinator by the Chambers-Mallows-Stuck transform:
  /// one uniform on `(0, π)` and one exponential, both from the step's own
  /// uniforms, with no rejection. The two exponents depend on `α` alone and
  /// are folded on the host, as are the scale `(c·dt)^{1/α}` and `π`.
  ///
  /// The uniforms are clamped into the open interval at a bound `f32` can
  /// hold below one: at exactly one the angle is `π`, whose sine is a small
  /// *negative* in single precision, and raising that to a fractional power
  /// is a NaN. The sines are floored for the same reason the clamp exists.
  65 => StableSubordinator { alpha, inv_alpha, one_minus_alpha, tail_exp, scale, pi }
    state (x)
    noise (dz)
    step {
      bind uu = min(max(u, lit(1e-7)), lit(0.9999999)) * pi;
      bind w = negate(ln(min(max(u2, lit(1e-7)), lit(0.9999999))));
      bind s1 = sin(alpha * uu) / pow(max(sin(uu), lit(1e-20)), inv_alpha);
      bind s2 = pow(max(sin(one_minus_alpha * uu), lit(1e-20)) / w, tail_exp);
      x + scale * s1 * s2
    }
    report { x },

  /// A Heston variance under a log-price whose jumps are Kou's
  /// double-exponential: the sum has no closed form, so the kernel sums the
  /// sizes in a bounded loop and the step reads that sum. The variance is
  /// truncated at zero.
  66 => KouJumpHeston { drift_c, kappa, theta, sigma_v, rho }
    state (s, v)
    noise (dw, dq)
    step {
      bind vp = positive(v);
      bind sv = sqrt(vp);
      bind dv = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq;
      s * exp((drift_c - vp / lit(2.0)) * dt + sv * dw + js),
      positive(vp + kappa * (theta - vp) * dt + sigma_v * sv * dv)
    }
    report { s, v },

  /// [`KouJumpHeston`](Family::KouJumpHeston) with the variance reflected.
  67 => KouJumpHestonReflected { drift_c, kappa, theta, sigma_v, rho }
    state (s, v)
    noise (dw, dq)
    step {
      bind vp = abs(v);
      bind sv = sqrt(vp);
      bind dv = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq;
      s * exp((drift_c - vp / lit(2.0)) * dt + sv * dw + js),
      abs(vp + kappa * (theta - vp) * dt + sigma_v * sv * dv)
    }
    report { s, v },

  /// [`DuffieKan`](Family::DuffieKan) with a compound-Poisson jump on the
  /// second factor. The host walks its jump times sequentially; the waiting
  /// time is memoryless, so the number of jumps a step sees is Poisson with
  /// mean `λ·dt` and their normal sizes aggregate into the kernel's own sum.
  68 => DuffieKanJump {
    a1, b1, c1, sigma1, a2, b2, c2, sigma2, alpha, beta, gamma, rho
  }
    state (r, x)
    noise (dw, dq)
    step {
      bind dx = rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq;
      bind vol = alpha * r + beta * x + gamma;
      r + (a1 * r + b1 * x + c1) * dt + sigma1 * vol * dw,
      x + (a2 * r + b2 * x + c2) * dt + sigma2 * vol * dx + js
    }
    report { r, x },

  /// A jump diffusion whose jump intensity is itself excited by its jumps:
  /// the Hawkes jump diffusion. At most one jump per step, as the host's own
  /// Bernoulli test takes it, so the intensity is a state component the step
  /// excites and then mean-reverts. The process reports the path alone; the
  /// intensity is the second component it carries.
  69 => HawkesJumpDiffusion {
    drift_c, sigma, alpha, beta, mu_lambda, jump_mu, jump_sigma
  }
    state (x, lam)
    noise (dw, dj)
    step {
      bind fired = less(u, lam * dt);
      bind size = pick(fired, jump_mu + jump_sigma * (dj / sqrt(dt)), lit(0.0));
      bind excited = lam + pick(fired, alpha, lit(0.0));
      x + drift_c * dt + sigma * dw + size,
      positive(excited + beta * (mu_lambda - excited) * dt)
    }
    report { x, lam },

  /// `σ²_t = ω + αX²_{t−1} + βσ²_{t−1}`, `X_t = σ_t z_t`: GARCH(1,1), and at
  /// `β = 0` ARCH(1). The series starts at `σ₀ z₀` with `σ₀²` the
  /// unconditional variance, so the launch steps before writing its first
  /// point and the third component marks whether that first step has been
  /// taken — until it has, the variance stays at the level the host seeds it
  /// with rather than running the recursion on a state that does not exist
  /// yet.
  70 => Garch { omega, alpha, beta }
    state (x, s2, warm)
    noise (dz)
    step {
      bind v = pick(warm, omega + alpha * x * x + beta * s2, s2);
      sqrt(max(v, lit(1e-12))) * (dz / sqrt(dt)),
      v,
      lit(1.0)
    }
    report { x, s2, warm },

  /// [`Garch`](Family::Garch) with a threshold term: `γX²_{t−1}` enters only
  /// when the previous return was negative, which is the GJR asymmetry and,
  /// under the other name its author gave it, the asymmetric GARCH.
  71 => ThresholdGarch { omega, alpha, gamma, beta }
    state (x, s2, warm)
    noise (dz)
    step {
      bind lev = pick(less(x, lit(0.0)), gamma * x * x, lit(0.0));
      bind v = pick(warm, omega + alpha * x * x + lev + beta * s2, s2);
      sqrt(max(v, lit(1e-12))) * (dz / sqrt(dt)),
      v,
      lit(1.0)
    }
    report { x, s2, warm },

  /// `ln σ²_t = ω + α(|z_{t−1}| − E|z|) + γ z_{t−1} + β ln σ²_{t−1}`,
  /// `X_t = σ_t z_t`: EGARCH(1,1). The lagged standardised residual is the
  /// previous return over the previous standard deviation, both of which the
  /// state carries, so the step recovers it rather than keeping a third
  /// series. `E|z| = √(2/π)` is folded on the host.
  72 => ExponentialGarch { omega, alpha, gamma, beta, e_abs_z }
    state (x, ls2, warm)
    noise (dz)
    step {
      bind sd = sqrt(exp(ls2));
      bind zl = x / sd;
      bind shock = alpha * (abs(zl) - e_abs_z) + gamma * zl;
      bind v = pick(warm, omega + shock + beta * ls2, ls2);
      sqrt(exp(v)) * (dz / sqrt(dt)),
      v,
      lit(1.0)
    }
    report { x, ls2, warm },

  /// One draw per grid point: the innovations themselves, with no recursion
  /// over them. White noise takes a mean and a standard deviation; Gaussian
  /// noise is the same family at zero mean and `√dt`.
  73 => Innovation { mean, sd }
    state (x)
    noise (dz)
    step { mean + sd * (dz / sqrt(dt)) }
    report { x },

  /// A correlated pair of innovations: the second is drawn independently and
  /// correlated in the step, which is what every two-factor model here does
  /// with its own shocks.
  74 => CorrelatedInnovation { rho }
    state (a, b)
    noise (dw, dq)
    step {
      dw,
      rho * dw + sqrt(negate(rho * rho - lit(1.0))) * dq
    }
    report { a, b },

  /// `X_t = φX_{t−1} + σz_t`: a first-order autoregression.
  75 => Autoregressive { phi, sigma }
    state (x)
    noise (dz)
    step { phi * x + sigma * (dz / sqrt(dt)) }
    report { x },

  /// `X_t = σz_t + θσz_{t−1}`: a first-order moving average. The lagged
  /// innovation is state of its own, since the step cannot see the previous
  /// draw any other way.
  76 => MovingAverage { theta, sigma }
    state (x, e)
    noise (dz)
    step {
      bind now = sigma * (dz / sqrt(dt));
      now + theta * e,
      now
    }
    report { x, e },

  /// A gamma subordinator: each increment is one Gamma draw.
  77 => GammaSubordinator { }
    state (x)
    noise (dz)
    step { x + gm }
    report { x },

  /// Brownian motion under a gamma clock: the variance gamma process.
  78 => VarianceGamma { mu, sigma }
    state (x)
    noise (dz)
    step { x + mu * gm + sigma * sqrt(gm) * (dz / sqrt(dt)) }
    report { x },

  /// The difference of two gamma processes: the bilateral gamma process.
  79 => BilateralGamma { }
    state (x)
    noise (dz)
    step { x + gm - gm2 }
    report { x },

  /// [`BilateralGamma`](Family::BilateralGamma) with a Brownian part.
  80 => BilateralGammaMotion { sigma }
    state (x)
    noise (dz)
    step { x + sigma * dz + gm - gm2 }
    report { x },

  /// A tempered-stable subordinator: the deterministic drift the small jumps
  /// below the truncation contribute, plus the step's own thinned jumps.
  81 => TemperedStableSubordinator { drift }
    state (x)
    noise (dz)
    step { x + drift + js }
    report { x },

  /// `dσ² = −λσ² dt + dZ` with `Z` a compound-Poisson subordinator of gamma
  /// jumps, and a log-Euler asset over it: the Barndorff-Nielsen-Shephard
  /// model. The variance step is exact in the decay, as the host takes it,
  /// and the jump sum is one gamma draw whose shape is the step's own jump
  /// count times a single jump's.
  82 => BarndorffNielsenShephard { decay, mu }
    state (s, v)
    noise (dw)
    step {
      s * exp((mu - v / lit(2.0)) * dt + sqrt(v) * dw),
      decay * v + gm
    }
    report { s, v },

  /// Two fractional rows out of one embedding: the first reads the increment
  /// buffer's leading `paths` rows, the second the next block, and the step
  /// correlates them exactly as `CorrelatedInnovation` correlates two
  /// Brownian shocks. Both rows share a Hurst exponent, which is what lets a
  /// single embedding feed the pair.
  83 => CorrelatedFractionalMotion { rho }
    state (a, b)
    noise (dz1, dz2)
    step {
      a + dz1,
      b + rho * dz1 + sqrt(negate(rho * rho - lit(1.0))) * dz2
    }
    report { a, b },

  /// The complex fractional Ornstein-Uhlenbeck process in its real and
  /// imaginary parts: one complex mean reversion `lambda - i·omega` acting on
  /// `x1 + i·x2`, driven by a complex fractional noise whose two parts are
  /// the pair of streams the one embedding produces. `scale` is the noise
  /// intensity `sqrt(a / 2)`, folded in on the host so the step carries no
  /// square root of its own.
  84 => ComplexFractionalOu { lambda, omega, scale }
    state (x1, x2)
    noise (dz1, dz2)
    step {
      x1 - (lambda * x1 + omega * x2) * dt + scale * dz1,
      x2 - (lambda * x2 - omega * x1) * dt + scale * dz2
    }
    report { x1, x2 },

  /// An Ornstein-Uhlenbeck process reported through a bounded map onto
  /// `(-1, 1)`, which is how a stochastic correlation is built from an
  /// unbounded state. `arctan` selects the map: at zero it is `tanh x`, and
  /// at one the shallower `(2/pi) arctan(pi x / 2)`, whose `pi / 2` arrives
  /// as `half_pi` rather than as a literal the kernel would carry at the
  /// wrong precision — `2 / pi` is its reciprocal. Both branches are
  /// evaluated and one is picked, since neither can fault.
  85 => TransformedOrnsteinUhlenbeck { kappa, mu, sigma, arctan, half_pi }
    state (x)
    noise (dz)
    step { x + kappa * (mu - x) * dt + sigma * dz }
    report {
      pick(
        geq(arctan, lit(0.5)),
        atan(x * half_pi) * recip(half_pi),
        tanh(x)
      )
    },

  /// The arrival times of a Poisson process sampled to a fixed count: a
  /// running sum of exponential inter-arrival times, each drawn by inverse
  /// CDF from the step's own uniform. The uniform is floored before the
  /// logarithm because the hash stream can land on exactly zero, which is a
  /// clamp of probability `1e-7` rather than a change of law.
  86 => PoissonArrivals { lambda }
    state (x)
    noise (dz)
    step { x + negate(ln(max(u, lit(1.0e-7)))) * recip(lambda) }
    report { x },

  2 => SquareRoot { kappa, theta, sigma }
    state (x)
    noise (dz)
    step { x + kappa * (theta - positive(x)) * dt + sigma * sqrt(positive(x)) * dz }
    report { positive(x) },
}

#[cfg(test)]
mod tests;

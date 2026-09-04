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
//! `euler::tests`, which pins every family's CubeCL output against the
//! generated Metal kernel, so a formula that drifts from the declaration
//! fails a test rather than a review.
//!
//! The names a step may use are fixed: `x` for the state, `dt` for the step
//! size, `dz` for the step's noise **increment** — `sqrt_dt · z` for Gaussian
//! noise, the fractional increment itself for fGN, which is what lets one
//! declaration serve both — and the family's own
//! parameters, which the generated code binds as locals from the parameter
//! buffer in declaration order. The `report` expression maps the state to what
//! the path records and is evaluated at `t = 0` as well, where no noise exists.
//!
//! The function vocabulary is `sqrt`, `exp`, `ln`, `pow`, `positive`, `max`
//! and `min`. Each has a host implementation in [`ops`] and a C definition in
//! [`C_PRELUDE`]; anything outside it fails to compile on the host, which is
//! the intended way to find out that a kernel could not have run it either.

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
}

/// The C definitions of the function vocabulary, in terms of the precision
/// placeholders the kernel renderer fills in. Each name expands to a distinct
/// intrinsic, so no definition refers to itself.
pub(crate) const C_PRELUDE: &str = r#"#define sqrt(v) STOCH_SQRT(v)
#define exp(v) STOCH_EXP(v)
#define ln(v) STOCH_LOG(v)
#define pow(a, b) STOCH_POW(a, b)
#define positive(v) ((v) > (REAL)0 ? (v) : (REAL)0)
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
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

  /// The positive part, the truncation a square-root diffusion steps on.
  #[cube]
  pub(crate) fn positive(v: f32) -> f32 {
    max(v, 0.0f32)
  }
}

/// Declares the Euler engine's families: one entry per family, from which the
/// host step, the host report and the kernels' C body are generated.
///
/// Each entry names its parameters in the order the parameter buffer carries
/// them, gives the step as an expression over `x`, `dt`, `sqrt_dt`, `z` and
/// those parameters, and the reported value as an expression over `x`.
macro_rules! euler_families {
  (
    step_inputs($x:ident, $params:ident, $dt:ident, $dz:ident);
    $(
    $(#[$meta:meta])*
    $code:literal => $name:ident { $($param:ident),* $(,)? }
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
    }

    /// One step of `family` on the host, from the same expression the kernels
    /// run. The parameter buffer is read in declaration order.
    #[allow(dead_code, unused_variables, unused_mut, unused_assignments)]
    pub(crate) fn host_step<T: FloatExt>(
      family: Family,
      $x: T,
      $params: &[T],
      $dt: T,
      $dz: T,
    ) -> T {
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
            $($step)*
          }
        )*
      }
    }

    /// What `family` reports for a state `x`, on the host.
    #[allow(dead_code, unused_variables)]
    pub(crate) fn host_report<T: FloatExt>(family: Family, $x: T) -> T {
      #[allow(unused_imports)]
      use ops::*;
      match family {
        $(
          Family::$name => { $($report)* }
        )*
      }
    }

    /// One `#[cube]` function per family, from the same expression the host
    /// and the C kernels run, plus the dispatch that picks one. The parameter
    /// bindings are peeled into the body before the `#[cube]` attribute sees
    /// it, so the emitted item contains no macro call — which the attribute
    /// cannot look through.
    #[cfg(feature = "cubecl")]
    pub(crate) mod cube {
      #[allow(unused_imports)]
      use super::cube_ops::*;
      #[allow(unused_imports)]
      use cubecl::prelude::*;

      $(
        euler_families!(@cube_step
          $(#[$meta])* $name, $x, $params, $dt, $dz,
          [0 1 2 3 4 5 6 7], [$($param)*], {}, {$($step)*}
        );
      )*

    }

    /// What each family reports for a state, in a CubeCL kernel.
    #[cfg(feature = "cubecl")]
    pub(crate) mod cube_report {
      #[allow(unused_imports)]
      use super::cube_ops::*;
      #[allow(unused_imports)]
      use cubecl::prelude::*;

      $(
        $(#[$meta])*
        #[cube]
        #[allow(non_snake_case)]
        pub(crate) fn $name($x: f32) -> f32 {
          $($report)*
        }
      )*
    }

    /// The C statements that step `x`, one guarded block per family.
    pub(crate) const C_STEP: &str = concat!($(
      "        if (family == ", stringify!($code), "u) {\n",
      euler_families!(@bind [0 1 2 3 4 5 6 7] $($param)*),
      "            x = ", stringify!($($step)*), ";\n",
      "        }\n",
    )*);

    /// The C statements that set `reported` from `x`, one block per family.
    pub(crate) const C_REPORT: &str = concat!($(
      "        if (family == ", stringify!($code), "u) {\n",
      "            reported = ", stringify!($($report)*), ";\n",
      "        }\n",
    )*);
  };

  (@cube_step
    $(#[$meta:meta])* $name:ident, $x:ident, $params:ident, $dt:ident, $dz:ident,
    [$($idx:literal)*], [], {$($bound:tt)*}, {$($step:tt)*}
  ) => {
    $(#[$meta])*
    #[cube]
    #[allow(non_snake_case)]
    pub(crate) fn $name(
      $x: f32,
      $params: &Array<f32>,
      $dt: f32,
      $dz: f32,
    ) -> f32 {
      $($bound)*
      $($step)*
    }
  };

  (@cube_step
    $(#[$meta:meta])* $name:ident, $x:ident, $params:ident, $dt:ident, $dz:ident,
    [$i:literal $($rest_idx:literal)*], [$head:ident $($rest:ident)*],
    {$($bound:tt)*}, {$($step:tt)*}
  ) => {
    euler_families!(@cube_step
      $(#[$meta])* $name, $x, $params, $dt, $dz,
      [$($rest_idx)*], [$($rest)*],
      {$($bound)* let $head = $params[$i];}, {$($step)*}
    );
  };

  (@bind [$($idx:literal)*]) => { "" };

  (@bind [$i:literal $($rest_idx:literal)*] $head:ident $($rest:ident)*) => {
    concat!(
      "            const REAL ", stringify!($head), " = params[", stringify!($i), "];\n",
      euler_families!(@bind [$($rest_idx)*] $($rest)*)
    )
  };
}

euler_families! {
  step_inputs(x, params, dt, dz);

  /// `dX = μX dt + σX dW`.
  0 => GeometricBrownian { mu, sigma }
    step { x + mu * x * dt + sigma * x * dz }
    report { x },

  /// `dX = θ(μ − X) dt + σ dW`.
  1 => OrnsteinUhlenbeck { theta, mu, sigma }
    step { x + theta * (mu - x) * dt + sigma * dz }
    report { x },

  /// `dX = κ(θ − X) dt + σ√X dW`, stepped with full truncation (Lord,
  /// Koekkoek & van Dijk 2010): the recursion runs on an auxiliary state
  /// whose positive part enters drift, diffusion and the reported path.
  2 => SquareRoot { kappa, theta, sigma }
    step { x + kappa * (theta - positive(x)) * dt + sigma * sqrt(positive(x)) * dz }
    report { positive(x) },
}

#[cfg(test)]
mod tests;

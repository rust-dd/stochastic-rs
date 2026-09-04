//! The drift / diffusion families of the Euler engine, declared once.
//!
//! A family is written in a subset that is simultaneously a valid Rust
//! expression and a valid C expression: infix arithmetic, parentheses and a
//! small vocabulary of functions. [`euler_families!`] takes that one
//! declaration and emits every form the engine needs — the host step, and the
//! C body the native CUDA and Metal kernels render — from the same tokens, so
//! the operation order cannot drift between them.
//!
//! The names a step may use are fixed: `x` for the state, `dt`, `sqrt_dt` and
//! `z` for the grid and the step's normal draw, and the family's own
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
"#;

/// Declares the Euler engine's families: one entry per family, from which the
/// host step, the host report and the kernels' C body are generated.
///
/// Each entry names its parameters in the order the parameter buffer carries
/// them, gives the step as an expression over `x`, `dt`, `sqrt_dt`, `z` and
/// those parameters, and the reported value as an expression over `x`.
macro_rules! euler_families {
  (
    step_inputs($x:ident, $dt:ident, $sqrt_dt:ident, $z:ident);
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
      params: &[T],
      $dt: T,
      $sqrt_dt: T,
      $z: T,
    ) -> T {
      #[allow(unused_imports)]
      use ops::*;
      match family {
        $(
          Family::$name => {
            let mut slot = 0;
            $(
              let $param = params[slot];
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

  (@bind [$($idx:literal)*]) => { "" };

  (@bind [$i:literal $($rest_idx:literal)*] $head:ident $($rest:ident)*) => {
    concat!(
      "            const REAL ", stringify!($head), " = params[", stringify!($i), "];\n",
      euler_families!(@bind [$($rest_idx)*] $($rest)*)
    )
  };
}

euler_families! {
  step_inputs(x, dt, sqrt_dt, z);

  /// `dX = μX dt + σX dW`.
  0 => GeometricBrownian { mu, sigma }
    step { x + mu * x * dt + sigma * x * sqrt_dt * z }
    report { x },

  /// `dX = θ(μ − X) dt + σ dW`.
  1 => OrnsteinUhlenbeck { theta, mu, sigma }
    step { x + theta * (mu - x) * dt + sigma * sqrt_dt * z }
    report { x },

  /// `dX = κ(θ − X) dt + σ√X dW`, stepped with full truncation (Lord,
  /// Koekkoek & van Dijk 2010): the recursion runs on an auxiliary state
  /// whose positive part enters drift, diffusion and the reported path.
  2 => SquareRoot { kappa, theta, sigma }
    step { x + kappa * (theta - positive(x)) * dt + sigma * sqrt(positive(x)) * sqrt_dt * z }
    report { positive(x) },
}

#[cfg(test)]
mod tests;

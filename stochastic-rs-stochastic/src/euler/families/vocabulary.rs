//! The function vocabulary a family step may use, in its three
//! implementations: the host functions in [`ops`], the C `#define`s in
//! [`C_PRELUDE`] the kernel renderer completes with each language's
//! intrinsics. A new name is added to both, and the parity run over every
//! family is what holds them to one meaning.

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
  /// on the host, a cast in C — so a family writes
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
/// placeholders the kernel renderer fills in. Several expand to an intrinsic
/// of the same name — MSL renders `#define sqrt(v) sqrt(v)` — which is not a
/// loop: the preprocessor does not re-expand a macro inside its own
/// expansion, so the definition resolves to the intrinsic exactly once.
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

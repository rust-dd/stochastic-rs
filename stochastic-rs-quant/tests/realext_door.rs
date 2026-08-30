//! The acceptance proof for the `RealExt` / `FloatExt` split: a scalar that
//! implements **only** `RealExt` — no SIMD lanes, no RNG fills, so it could
//! never implement `FloatExt` — flows through the generic analytic pricing
//! path and reproduces `f64` bit-for-bit.
//!
//! `R64` is a plain delegating newtype over `f64`. Its point is what it does
//! *not* implement: `SimdFloatExt` and `FloatExt` are absent, so every call
//! below compiles only because the code under test is bounded on `RealExt`
//! alone. An AAD dual number or a higher-precision scalar takes exactly this
//! route in.

use std::iter::Sum;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Rem;
use std::ops::Sub;
use std::ops::SubAssign;

use ndarray::ScalarOperand;
use num_traits::Float;
use num_traits::FloatConst;
use num_traits::FromPrimitive;
use num_traits::Num;
use num_traits::NumCast;
use num_traits::One;
use num_traits::Signed;
use num_traits::ToPrimitive;
use num_traits::Zero;
use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::lattice::equity::CrrModel;
use stochastic_rs_quant::traits::RealExt;

#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
struct R64(f64);

macro_rules! delegate_binop {
  ($($trait_:ident :: $method:ident),* $(,)?) => {
    $(
      impl $trait_ for R64 {
        type Output = R64;
        fn $method(self, rhs: R64) -> R64 {
          R64(self.0.$method(rhs.0))
        }
      }
    )*
  };
}

delegate_binop!(Add::add, Sub::sub, Mul::mul, Div::div, Rem::rem);

impl Neg for R64 {
  type Output = R64;
  fn neg(self) -> R64 {
    R64(-self.0)
  }
}

impl AddAssign for R64 {
  fn add_assign(&mut self, rhs: R64) {
    self.0 += rhs.0;
  }
}

impl SubAssign for R64 {
  fn sub_assign(&mut self, rhs: R64) {
    self.0 -= rhs.0;
  }
}

impl Zero for R64 {
  fn zero() -> Self {
    R64(0.0)
  }

  fn is_zero(&self) -> bool {
    self.0 == 0.0
  }
}

impl One for R64 {
  fn one() -> Self {
    R64(1.0)
  }
}

impl Num for R64 {
  type FromStrRadixErr = num_traits::ParseFloatError;

  fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
    f64::from_str_radix(str, radix).map(R64)
  }
}

impl ToPrimitive for R64 {
  fn to_i64(&self) -> Option<i64> {
    self.0.to_i64()
  }

  fn to_u64(&self) -> Option<u64> {
    self.0.to_u64()
  }

  fn to_f64(&self) -> Option<f64> {
    Some(self.0)
  }
}

impl NumCast for R64 {
  fn from<T: ToPrimitive>(n: T) -> Option<Self> {
    n.to_f64().map(R64)
  }
}

impl FromPrimitive for R64 {
  fn from_i64(n: i64) -> Option<Self> {
    f64::from_i64(n).map(R64)
  }

  fn from_u64(n: u64) -> Option<Self> {
    f64::from_u64(n).map(R64)
  }

  fn from_f64(n: f64) -> Option<Self> {
    Some(R64(n))
  }
}

impl Signed for R64 {
  fn abs(&self) -> Self {
    R64(self.0.abs())
  }

  fn abs_sub(&self, other: &Self) -> Self {
    R64(Signed::abs_sub(&self.0, &other.0))
  }

  fn signum(&self) -> Self {
    R64(self.0.signum())
  }

  fn is_positive(&self) -> bool {
    self.0.is_sign_positive()
  }

  fn is_negative(&self) -> bool {
    self.0.is_sign_negative()
  }
}

macro_rules! delegate_nullary {
  ($($method:ident),* $(,)?) => {
    $(
      fn $method() -> Self {
        R64(f64::$method())
      }
    )*
  };
}

macro_rules! delegate_unary {
  ($($method:ident),* $(,)?) => {
    $(
      fn $method(self) -> Self {
        R64(self.0.$method())
      }
    )*
  };
}

impl Float for R64 {
  delegate_nullary!(
    nan,
    infinity,
    neg_infinity,
    neg_zero,
    min_value,
    min_positive_value,
    max_value,
  );

  delegate_unary!(
    floor, ceil, round, trunc, fract, abs, signum, recip, sqrt, exp, exp2, ln, log2, log10, cbrt,
    sin, cos, tan, asin, acos, atan, exp_m1, ln_1p, sinh, cosh, tanh, asinh, acosh, atanh,
  );

  fn is_nan(self) -> bool {
    self.0.is_nan()
  }

  fn is_infinite(self) -> bool {
    self.0.is_infinite()
  }

  fn is_finite(self) -> bool {
    self.0.is_finite()
  }

  fn is_normal(self) -> bool {
    self.0.is_normal()
  }

  fn classify(self) -> std::num::FpCategory {
    self.0.classify()
  }

  fn is_sign_positive(self) -> bool {
    self.0.is_sign_positive()
  }

  fn is_sign_negative(self) -> bool {
    self.0.is_sign_negative()
  }

  fn mul_add(self, a: Self, b: Self) -> Self {
    R64(self.0.mul_add(a.0, b.0))
  }

  fn powi(self, n: i32) -> Self {
    R64(self.0.powi(n))
  }

  fn powf(self, n: Self) -> Self {
    R64(self.0.powf(n.0))
  }

  fn log(self, base: Self) -> Self {
    R64(self.0.log(base.0))
  }

  fn max(self, other: Self) -> Self {
    R64(self.0.max(other.0))
  }

  fn min(self, other: Self) -> Self {
    R64(self.0.min(other.0))
  }

  fn abs_sub(self, other: Self) -> Self {
    R64((self.0 - other.0).max(0.0))
  }

  fn hypot(self, other: Self) -> Self {
    R64(self.0.hypot(other.0))
  }

  fn atan2(self, other: Self) -> Self {
    R64(self.0.atan2(other.0))
  }

  fn sin_cos(self) -> (Self, Self) {
    let (s, c) = self.0.sin_cos();
    (R64(s), R64(c))
  }

  fn integer_decode(self) -> (u64, i16, i8) {
    self.0.integer_decode()
  }
}

impl FloatConst for R64 {
  delegate_nullary!(
    E,
    FRAC_1_PI,
    FRAC_1_SQRT_2,
    FRAC_2_PI,
    FRAC_2_SQRT_PI,
    FRAC_PI_2,
    FRAC_PI_3,
    FRAC_PI_4,
    FRAC_PI_6,
    FRAC_PI_8,
    LN_10,
    LN_2,
    LOG10_E,
    LOG2_E,
    PI,
    SQRT_2,
  );
}

impl Sum for R64 {
  fn sum<I: Iterator<Item = R64>>(iter: I) -> R64 {
    R64(iter.map(|x| x.0).sum())
  }
}

impl ScalarOperand for R64 {}

impl RealExt for R64 {
  fn from_usize_(n: usize) -> Self {
    R64(n as f64)
  }

  fn from_f64_fast(v: f64) -> Self {
    R64(v)
  }

  fn pi() -> Self {
    R64(std::f64::consts::PI)
  }

  fn two_pi() -> Self {
    R64(2.0 * std::f64::consts::PI)
  }

  fn min_positive_val() -> Self {
    R64(f64::MIN_POSITIVE)
  }
}

/// The CRR binomial rollback is a genuine pricer that is generic over the
/// scalar, so it is the door: if `R64` prices it, any `RealExt`-only scalar
/// can. Delegation is exact, so the two runs must agree to the bit.
#[test]
fn a_realext_only_scalar_prices_the_crr_lattice_bit_for_bit() {
  let queries = [
    (100.0, 100.0, 0.05, 0.00, 1.00, OptionType::Call),
    (100.0, 110.0, 0.05, 0.02, 0.50, OptionType::Put),
    (80.0, 100.0, 0.01, 0.00, 2.00, OptionType::Put),
  ];

  for &(s, k, r, q, tau, ty) in &queries {
    let via_f64 = CrrModel::<f64>::new(0.2, 64).price_american(s, k, r, q, tau, ty);
    let via_r64 = CrrModel::<R64>::new(R64(0.2), 64)
      .price_american(R64(s), R64(k), R64(r), R64(q), R64(tau), ty)
      .0;
    assert!(
      via_f64.to_bits() == via_r64.to_bits(),
      "CRR through R64 diverged from f64: {via_r64} vs {via_f64} at \
       (s={s}, k={k}, r={r}, q={q}, tau={tau}, {ty:?})"
    );
    assert!(via_f64 > 0.0, "degenerate query, price = {via_f64}");
  }
}

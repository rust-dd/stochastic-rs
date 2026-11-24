//! A scalar with only `RealExt` (`R64`, a delegating newtype with no SIMD or RNG, so never
//! `FloatExt`) flows through the generic analytic pricing path and reproduces `f64` bit for bit.

use std::cell::Cell;
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
use stochastic_rs_quant::calendar::Frequency;
use stochastic_rs_quant::curves::BootstrapInstrument;
use stochastic_rs_quant::curves::BootstrapSpec;
use stochastic_rs_quant::curves::CurvePoint;
use stochastic_rs_quant::curves::DiscountCurve;
use stochastic_rs_quant::curves::InterpolationMethod;
use stochastic_rs_quant::curves::bootstrap;
use stochastic_rs_quant::lattice::equity::CrrModel;
use stochastic_rs_quant::traits::RealExt;

#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
struct R64(f64);

thread_local! {
  /// Calls to `exp` and `ln` on this thread: a computation's cost, counted without a clock.
  static TRANSCENDENTALS: Cell<usize> = const { Cell::new(0) };
}

fn counted(x: f64) -> R64 {
  TRANSCENDENTALS.with(|n| n.set(n.get() + 1));
  R64(x)
}

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
    epsilon,
  );

  delegate_unary!(
    floor, ceil, round, trunc, fract, abs, signum, recip, sqrt, exp2, log2, log10, cbrt, sin, cos,
    tan, asin, acos, atan, exp_m1, ln_1p, sinh, cosh, tanh, asinh, acosh, atanh,
  );

  fn exp(self) -> Self {
    counted(self.0.exp())
  }

  fn ln(self) -> Self {
    counted(self.0.ln())
  }

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

  fn machine_epsilon() -> Self {
    R64(f64::EPSILON)
  }
}

/// The CRR rollback is generic over the scalar, so `R64` pricing it opens the door to any
/// `RealExt`-only scalar; delegation is exact, so the runs agree to the bit.
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

/// The `market` bench's seven quotes at their day-count year fractions; the swaps pay past the
/// last pillar.
fn bench_ladder<T: RealExt>() -> Vec<BootstrapInstrument<T>> {
  let c = T::from_f64_fast;
  let swap = |days: f64, rate: f64| BootstrapInstrument::Swap {
    maturity: c(days / 365.0),
    rate: c(rate),
    frequency: Frequency::SemiAnnual,
  };
  vec![
    BootstrapInstrument::Deposit {
      maturity: c(31.0 / 360.0),
      rate: c(0.03),
    },
    BootstrapInstrument::Fra {
      start: c(31.0 / 360.0),
      end: c(181.0 / 360.0),
      rate: c(0.032),
    },
    BootstrapInstrument::Fra {
      start: c(181.0 / 360.0),
      end: c(365.0 / 360.0),
      rate: c(0.035),
    },
    swap(730.0, 0.038),
    swap(1826.0, 0.04),
    swap(3652.0, 0.042),
    swap(10957.0, 0.043),
  ]
}

fn assert_same_pillars(via_f64: &DiscountCurve<f64>, via_r64: &DiscountCurve<R64>) {
  for (a, b) in via_f64.points().iter().zip(via_r64.points()) {
    assert!(
      a.discount_factor.to_bits() == b.discount_factor.0.to_bits(),
      "t = {}: {} vs {}",
      a.time,
      a.discount_factor,
      b.discount_factor.0
    );
  }
}

/// `f64`'s curve bit for bit, its four gapped swaps solved in a few reads of their payments past
/// the last pillar: 439 calls to `exp` and `ln` in all.
#[test]
fn a_realext_only_scalar_bootstraps_the_bench_ladder_in_a_few_hundred_exp_and_ln_calls() {
  let method = InterpolationMethod::LogLinearOnDiscountFactors;
  let via_f64 = bootstrap(BootstrapSpec::new(&bench_ladder::<f64>(), method)).unwrap();
  TRANSCENDENTALS.with(|n| n.set(0));
  let via_r64 = bootstrap(BootstrapSpec::new(&bench_ladder::<R64>(), method)).unwrap();
  let calls = TRANSCENDENTALS.with(Cell::get);
  assert_same_pillars(&via_f64, &via_r64);
  assert!(calls <= 500, "{calls} calls to exp and ln");
}

/// A 3M forecast curve 50 bp over a flat 2 % OIS curve, a few trial curves per swap: 870 calls to
/// `exp` and `ln` in all.
#[test]
fn a_realext_only_scalar_bootstraps_a_forecast_curve_in_a_few_trial_curves_per_swap() {
  fn ois<T: RealExt>() -> DiscountCurve<T> {
    let points = [0.25, 1.0, 5.0, 10.0]
      .map(|t: f64| CurvePoint {
        time: T::from_f64_fast(t),
        discount_factor: T::from_f64_fast((-0.02 * t).exp()),
      })
      .to_vec();
    DiscountCurve::new(points, InterpolationMethod::LogLinearOnDiscountFactors)
  }
  fn quotes<T: RealExt>() -> Vec<BootstrapInstrument<T>> {
    let (c, ois, tenor) = (
      T::from_f64_fast,
      |t: f64| (-0.02 * t).exp(),
      |t: f64| (-0.025 * t).exp(),
    );
    let mut quotes = vec![BootstrapInstrument::Deposit {
      maturity: c(0.25),
      rate: c((1.0 / tenor(0.25) - 1.0) / 0.25),
    }];
    for years in [1_usize, 2, 3, 5] {
      let float: Vec<f64> = (1..=4 * years).map(|i| 0.25 * i as f64).collect();
      let (mut floating, mut prev) = (0.0, 0.0);
      for &t in &float {
        floating += ois(t) * (tenor(prev) / tenor(t) - 1.0);
        prev = t;
      }
      let annuity: f64 = (1..=years).map(|i| ois(i as f64)).sum();
      quotes.push(BootstrapInstrument::SwapWithSchedule {
        rate: c(floating / annuity),
        fixed_times: (1..=years).map(|i| c(i as f64)).collect(),
        float_times: float.into_iter().map(c).collect(),
      });
    }
    quotes
  }
  let method = InterpolationMethod::LogLinearOnDiscountFactors;
  let (ois_f64, ois_r64) = (ois::<f64>(), ois::<R64>());
  let (f64_quotes, r64_quotes) = (quotes::<f64>(), quotes::<R64>());
  let via_f64 =
    bootstrap(BootstrapSpec::new(&f64_quotes, method).discounting_with(&ois_f64)).unwrap();
  TRANSCENDENTALS.with(|n| n.set(0));
  let via_r64 =
    bootstrap(BootstrapSpec::new(&r64_quotes, method).discounting_with(&ois_r64)).unwrap();
  let calls = TRANSCENDENTALS.with(Cell::get);
  assert_same_pillars(&via_f64, &via_r64);
  assert!(calls <= 1000, "{calls} calls to exp and ln");
}

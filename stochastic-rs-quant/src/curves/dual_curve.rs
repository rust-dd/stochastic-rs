//! # Dual-curve bootstrap
//!
//! Post-crisis a tenor's forward curve is bootstrapped **against an exogenous
//! OIS discount curve**. The pseudo-discount factors `P_x(t)` of tenor `x`
//! only project its forwards, `F_x(t_{i−1}, t_i) = (P_x(t_{i−1}) / P_x(t_i) − 1) / δ_i`,
//! while every cash flow is discounted with `D(t)`. A deposit on the tenor
//! fixes `P_x(τ) = 1 / (1 + r τ)`, a FRA extends it multiplicatively, and a
//! par swap against the tenor pins the pseudo-discount factor at its last
//! floating date through
//!
//! $$
//! S \sum_j \delta_j D(t_j) = \sum_i D(t_i)\left(\frac{P_x(t_{i-1})}{P_x(t_i)} - 1\right),
//! $$
//!
//! solved by bisection because the floating leg no longer telescopes once
//! discounting and forecasting curves differ.
//!
//! References: Ametrano, F. M. & Bianchetti, M. (2013), *Everything You Always
//! Wanted to Know About Multiple Interest Rate Curve Bootstrapping but Were
//! Afraid to Ask*, SSRN 2219548, §4–5; Bianchetti, M. (2010), *Two Curves, One
//! Price*, Risk 23(8), 66–72.

use super::discount_curve::DiscountCurve;
use super::interpolation::interpolate_discount_factor;
use super::types::CurvePoint;
use super::types::InterpolationMethod;
use crate::traits::RealExt;

/// Quote on the tenor being bootstrapped; times are year fractions from the
/// curve origin.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ForecastInstrument<T: RealExt> {
  /// Tenor deposit: `P_x(maturity) = 1 / (1 + rate · maturity)`.
  Deposit { maturity: T, rate: T },
  /// FRA on the tenor: `P_x(end) = P_x(start) / (1 + rate · (end − start))`.
  Fra { start: T, end: T, rate: T },
  /// Par swap against the tenor: `fixed_times` are the fixed-leg payment
  /// times, `float_times` the floating-leg (tenor-spaced) payment times;
  /// both end at the swap maturity.
  Swap {
    rate: T,
    fixed_times: Vec<T>,
    float_times: Vec<T>,
  },
}

impl<T: RealExt> ForecastInstrument<T> {
  /// Maturity (last payment time) of the quote.
  pub fn maturity(&self) -> T {
    match self {
      Self::Deposit { maturity, .. } => *maturity,
      Self::Fra { end, .. } => *end,
      Self::Swap { float_times, .. } => float_times.last().copied().unwrap_or_else(T::zero),
    }
  }
}

/// Bootstraps the tenor's pseudo-discount curve against the exogenous
/// `discount` curve; instruments are processed by increasing maturity and
/// each adds one node.
pub fn bootstrap_forecast<T: RealExt>(
  instruments: &[ForecastInstrument<T>],
  discount: &DiscountCurve<T>,
  method: InterpolationMethod,
) -> DiscountCurve<T> {
  let mut sorted: Vec<&ForecastInstrument<T>> = instruments.iter().collect();
  sorted.sort_by(|a, b| {
    a.maturity()
      .partial_cmp(&b.maturity())
      .expect("finite maturities")
  });
  let mut points: Vec<CurvePoint<T>> = vec![CurvePoint {
    time: T::zero(),
    discount_factor: T::one(),
  }];
  for inst in sorted {
    match inst {
      ForecastInstrument::Deposit { maturity, rate } => {
        points.push(CurvePoint {
          time: *maturity,
          discount_factor: T::one() / (T::one() + *rate * *maturity),
        });
      }
      ForecastInstrument::Fra { start, end, rate } => {
        let p_start = interpolate_discount_factor(&points, *start, method);
        points.push(CurvePoint {
          time: *end,
          discount_factor: p_start / (T::one() + *rate * (*end - *start)),
        });
      }
      ForecastInstrument::Swap {
        rate,
        fixed_times,
        float_times,
      } => {
        assert!(
          !fixed_times.is_empty() && !float_times.is_empty(),
          "swap schedules must not be empty"
        );
        let maturity = *float_times.last().expect("non-empty");
        assert!(
          (maturity - *fixed_times.last().expect("non-empty")).abs() <= T::from_f64_fast(1e-8),
          "float and fixed schedules must end at the same maturity"
        );
        let fixed_pv = fixed_leg_pv(*rate, fixed_times, discount);
        let residual = |p: T| {
          let mut trial = points.clone();
          trial.push(CurvePoint {
            time: maturity,
            discount_factor: p,
          });
          floating_leg_pv(float_times, &trial, discount, method) - fixed_pv
        };
        let p = solve_last_pseudo_discount(
          residual,
          points.last().expect("origin present").discount_factor,
        );
        points.push(CurvePoint {
          time: maturity,
          discount_factor: p,
        });
      }
    }
  }
  DiscountCurve::new(points, method)
}

/// `S Σ_j δ_j D(t_j)` with `t_0 = 0`.
fn fixed_leg_pv<T: RealExt>(rate: T, fixed_times: &[T], discount: &DiscountCurve<T>) -> T {
  let mut pv = T::zero();
  let mut prev = T::zero();
  for t in fixed_times {
    pv += (*t - prev) * discount.discount_factor(*t);
    prev = *t;
  }
  rate * pv
}

/// `Σ_i D(t_i) (P_x(t_{i−1}) / P_x(t_i) − 1)` with the pseudo-discount
/// factors read off `points`.
fn floating_leg_pv<T: RealExt>(
  float_times: &[T],
  points: &[CurvePoint<T>],
  discount: &DiscountCurve<T>,
  method: InterpolationMethod,
) -> T {
  let mut pv = T::zero();
  let mut prev_p = T::one();
  for t in float_times {
    let p = interpolate_discount_factor(points, *t, method);
    pv += discount.discount_factor(*t) * (prev_p / p - T::one());
    prev_p = p;
  }
  pv
}

/// Bisection on the last pseudo-discount factor: the floating leg is
/// decreasing in it, so the residual changes sign once on the bracket.
fn solve_last_pseudo_discount<T: RealExt>(residual: impl Fn(T) -> T, previous: T) -> T {
  let two = T::from_f64_fast(2.0);
  let mut lo = previous * T::from_f64_fast(1e-6);
  let mut hi = previous * two;
  let mut f_lo = residual(lo);
  let mut f_hi = residual(hi);
  let mut widen = 0;
  while f_lo.signum() == f_hi.signum() && widen < 60 {
    lo = lo / two;
    hi = hi * two;
    f_lo = residual(lo);
    f_hi = residual(hi);
    widen += 1;
  }
  assert!(
    f_lo.signum() != f_hi.signum(),
    "dual-curve bootstrap could not bracket the pseudo-discount factor; the swap quote is inconsistent with the earlier nodes"
  );
  let tol = T::from_f64_fast(1e-15);
  for _ in 0..200 {
    let mid = (lo + hi) / two;
    let f_mid = residual(mid);
    if f_mid == T::zero() || (hi - lo).abs() <= tol * hi.abs() {
      return mid;
    }
    if f_mid.signum() == f_lo.signum() {
      lo = mid;
      f_lo = f_mid;
    } else {
      hi = mid;
    }
  }
  (lo + hi) / two
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;

  use super::*;
  use crate::curves::multi_curve::MultiCurve;

  fn flat(rate: f64) -> DiscountCurve<f64> {
    DiscountCurve::from_zero_rates(
      &Array1::from_vec(vec![0.25, 1.0, 5.0, 10.0]),
      &Array1::from_vec(vec![rate; 4]),
      InterpolationMethod::LogLinearOnDiscountFactors,
    )
  }

  /// Par rate of a swap against the tenor from the true curves.
  fn par_rate(
    discount: &DiscountCurve<f64>,
    tenor_df: impl Fn(f64) -> f64,
    fixed: &[f64],
    float: &[f64],
  ) -> f64 {
    let mut float_pv = 0.0;
    let mut prev = 0.0;
    for t in float {
      float_pv += discount.discount_factor(*t) * (tenor_df(prev) / tenor_df(*t) - 1.0);
      prev = *t;
    }
    let mut annuity = 0.0;
    prev = 0.0;
    for t in fixed {
      annuity += (t - prev) * discount.discount_factor(*t);
      prev = *t;
    }
    float_pv / annuity
  }

  fn quotes(
    discount: &DiscountCurve<f64>,
    tenor_df: impl Fn(f64) -> f64 + Copy,
  ) -> Vec<ForecastInstrument<f64>> {
    let mut q = vec![
      ForecastInstrument::Deposit {
        maturity: 0.25,
        rate: (1.0 / tenor_df(0.25) - 1.0) / 0.25,
      },
      ForecastInstrument::Fra {
        start: 0.25,
        end: 0.5,
        rate: (tenor_df(0.25) / tenor_df(0.5) - 1.0) / 0.25,
      },
    ];
    for years in [1_usize, 2, 3, 5, 7] {
      let fixed: Vec<f64> = (1..=years).map(|i| i as f64).collect();
      let float: Vec<f64> = (1..=4 * years).map(|i| 0.25 * i as f64).collect();
      q.push(ForecastInstrument::Swap {
        rate: par_rate(discount, tenor_df, &fixed, &float),
        fixed_times: fixed,
        float_times: float,
      });
    }
    q
  }

  #[test]
  fn single_curve_quotes_reproduce_the_discount_curve() {
    let ois = flat(0.02);
    let df = |t: f64| (-0.02_f64 * t).exp();
    let forecast = bootstrap_forecast(
      &quotes(&ois, df),
      &ois,
      InterpolationMethod::LogLinearOnDiscountFactors,
    );
    for t in [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0] {
      assert!(
        (forecast.discount_factor(t) - df(t)).abs() < 1e-10,
        "t = {t}: {}",
        forecast.discount_factor(t)
      );
    }
  }

  /// A tenor trading 50 bp above OIS: the bootstrap recovers its
  /// pseudo-discount factors at every pillar and the multi-curve basis reads
  /// the spread back.
  #[test]
  fn recovers_a_tenor_curve_with_basis_over_ois() {
    let ois = flat(0.02);
    let tenor = |t: f64| (-0.025_f64 * t).exp();
    let forecast = bootstrap_forecast(
      &quotes(&ois, tenor),
      &ois,
      InterpolationMethod::LogLinearOnDiscountFactors,
    );
    for t in [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0] {
      assert!(
        (forecast.discount_factor(t) - tenor(t)).abs() < 1e-9,
        "t = {t}: {}",
        forecast.discount_factor(t)
      );
    }
    let mut multi = MultiCurve::new(ois);
    multi.add_forecast("3M", forecast);
    let basis = multi
      .basis_spread("3M", 1.0, 1.25)
      .expect("tenor registered");
    assert!((basis - 0.005).abs() < 2e-4, "basis {basis}");
  }

  #[test]
  fn deposit_and_fra_extend_the_short_end_multiplicatively() {
    let ois = flat(0.02);
    let q = [
      ForecastInstrument::Deposit {
        maturity: 0.5,
        rate: 0.03,
      },
      ForecastInstrument::Fra {
        start: 0.5,
        end: 1.0,
        rate: 0.032,
      },
    ];
    let forecast = bootstrap_forecast(&q, &ois, InterpolationMethod::LogLinearOnDiscountFactors);
    let p_half = 1.0 / (1.0 + 0.03 * 0.5);
    assert!((forecast.discount_factor(0.5) - p_half).abs() < 1e-14);
    assert!((forecast.discount_factor(1.0) - p_half / (1.0 + 0.032 * 0.5)).abs() < 1e-14);
  }

  #[test]
  #[should_panic(expected = "must end at the same maturity")]
  fn rejects_swap_schedules_with_different_maturities() {
    let ois = flat(0.02);
    let q = [ForecastInstrument::Swap {
      rate: 0.02,
      fixed_times: vec![1.0, 2.0],
      float_times: vec![0.5, 1.0, 1.5],
    }];
    let _ = bootstrap_forecast(&q, &ois, InterpolationMethod::LogLinearOnDiscountFactors);
  }
}

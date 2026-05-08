//! Yield curve bootstrapping from market instruments.
//!
//! Reference: Hagan & West, "Methods for Constructing a Yield Curve",
//! Wilmott Magazine (2006).
//!
//! Reference: Ametrano & Bianchetti, "Everything You Always Wanted to Know About
//! Multiple Interest Rate Curve Bootstrapping but Were Afraid to Ask",
//! SSRN 2219548 (2013).
//!
//! Bootstrapping algorithm:
//! 1. **Short end** (deposits): $D(t) = 1/(1 + r\,\delta)$
//! 2. **Middle** (FRAs / futures): $D(t_2) = D(t_1)/(1 + R\,\delta)$
//! 3. **Long end** (swaps): $D(t_n) = \frac{1 - S\sum_{i=1}^{n-1}\delta_i\,D(t_i)}{1 + S\,\delta_n}$

use super::discount_curve::DiscountCurve;
use super::interpolation;
use super::types::CurvePoint;
use super::types::Instrument;
use super::types::InterpolationMethod;
use crate::traits::FloatExt;

/// Build a discount curve by bootstrapping from a set of market instruments.
///
/// Instruments are processed in order of increasing maturity. Each instrument
/// provides one equation for one unknown discount factor.
pub fn bootstrap<T: FloatExt>(
  instruments: &[Instrument<T>],
  method: InterpolationMethod,
) -> DiscountCurve<T> {
  let mut sorted: Vec<&Instrument<T>> = instruments.iter().collect();
  sorted.sort_by(|a, b| a.maturity().partial_cmp(&b.maturity()).unwrap());

  let mut points: Vec<CurvePoint<T>> = vec![CurvePoint {
    time: T::zero(),
    discount_factor: T::one(),
  }];

  for inst in &sorted {
    match inst {
      Instrument::Deposit { maturity, rate } => {
        let df = T::one() / (T::one() + *rate * *maturity);
        points.push(CurvePoint {
          time: *maturity,
          discount_factor: df,
        });
      }
      Instrument::Fra { start, end, rate } => {
        let d_start = interpolation::interpolate_discount_factor(&points, *start, method);
        let delta = *end - *start;
        let df = d_start / (T::one() + *rate * delta);
        points.push(CurvePoint {
          time: *end,
          discount_factor: df,
        });
      }
      Instrument::Future {
        start,
        end,
        price,
        sigma,
      } => {
        let d_start = interpolation::interpolate_discount_factor(&points, *start, method);
        let delta = *end - *start;
        let hundred = T::from_f64_fast(100.0);
        let futures_rate = (hundred - *price) / hundred;
        let half = T::from_f64_fast(0.5);
        let convexity_adj = half * *sigma * *sigma * *start * *end;
        let fra_rate = futures_rate - convexity_adj;
        let df = d_start / (T::one() + fra_rate * delta);
        points.push(CurvePoint {
          time: *end,
          discount_factor: df,
        });
      }
      Instrument::Swap {
        maturity,
        rate,
        frequency,
      } => {
        let delta = T::one() / T::from_f64_fast(*frequency as f64);
        let n_payments = (*maturity * T::from_f64_fast(*frequency as f64))
          .round()
          .to_f64()
          .unwrap() as usize;

        let mut annuity = T::zero();
        for i in 1..n_payments {
          let t_i = T::from_f64_fast(i as f64) * delta;
          let d_i = interpolation::interpolate_discount_factor(&points, t_i, method);
          annuity += delta * d_i;
        }

        let df_n = (T::one() - *rate * annuity) / (T::one() + *rate * delta);
        points.push(CurvePoint {
          time: *maturity,
          discount_factor: df_n,
        });
      }
    }
  }

  DiscountCurve::new(points, method)
}

/// Iterative bootstrapping with root-finding for instruments that don't align with nodes.
///
/// Uses the bisection method to solve for the unknown discount factor at each step.
pub fn bootstrap_iterative<T: FloatExt>(
  instruments: &[Instrument<T>],
  method: InterpolationMethod,
  tol: T,
  max_iter: usize,
) -> DiscountCurve<T> {
  let mut sorted: Vec<&Instrument<T>> = instruments.iter().collect();
  sorted.sort_by(|a, b| a.maturity().partial_cmp(&b.maturity()).unwrap());

  let mut points: Vec<CurvePoint<T>> = vec![CurvePoint {
    time: T::zero(),
    discount_factor: T::one(),
  }];

  for inst in &sorted {
    match inst {
      Instrument::Deposit { maturity, rate } => {
        let df = T::one() / (T::one() + *rate * *maturity);
        points.push(CurvePoint {
          time: *maturity,
          discount_factor: df,
        });
      }
      Instrument::Fra { start, end, rate } => {
        let d_start = interpolation::interpolate_discount_factor(&points, *start, method);
        let delta = *end - *start;
        let df = d_start / (T::one() + *rate * delta);
        points.push(CurvePoint {
          time: *end,
          discount_factor: df,
        });
      }
      Instrument::Future {
        start,
        end,
        price,
        sigma,
      } => {
        let d_start = interpolation::interpolate_discount_factor(&points, *start, method);
        let delta = *end - *start;
        let hundred = T::from_f64_fast(100.0);
        let futures_rate = (hundred - *price) / hundred;
        let half = T::from_f64_fast(0.5);
        let convexity_adj = half * *sigma * *sigma * *start * *end;
        let fra_rate = futures_rate - convexity_adj;
        let df = d_start / (T::one() + fra_rate * delta);
        points.push(CurvePoint {
          time: *end,
          discount_factor: df,
        });
      }
      Instrument::Swap {
        maturity,
        rate,
        frequency,
      } => {
        let df_n = solve_swap_df(&points, *maturity, *rate, *frequency, method, tol, max_iter);
        points.push(CurvePoint {
          time: *maturity,
          discount_factor: df_n,
        });
      }
    }
  }

  DiscountCurve::new(points, method)
}

/// Solve for the swap's terminal discount factor using bisection.
fn solve_swap_df<T: FloatExt>(
  existing_points: &[CurvePoint<T>],
  maturity: T,
  swap_rate: T,
  frequency: u32,
  method: InterpolationMethod,
  tol: T,
  max_iter: usize,
) -> T {
  let delta = T::one() / T::from_f64_fast(frequency as f64);
  let n_payments = (maturity * T::from_f64_fast(frequency as f64))
    .round()
    .to_f64()
    .unwrap() as usize;

  let mut annuity_known = T::zero();
  for i in 1..n_payments {
    let t_i = T::from_f64_fast(i as f64) * delta;
    let d_i = interpolation::interpolate_discount_factor(existing_points, t_i, method);
    annuity_known += delta * d_i;
  }

  let df_analytic = (T::one() - swap_rate * annuity_known) / (T::one() + swap_rate * delta);
  if df_analytic > T::zero() && df_analytic < T::one() {
    return df_analytic;
  }

  let mut lo = T::from_f64_fast(1e-6);
  let mut hi = T::one();
  let half = T::from_f64_fast(0.5);

  for _ in 0..max_iter {
    let mid = half * (lo + hi);
    let annuity_total = annuity_known + delta * mid;
    let implied_rate = (T::one() - mid) / annuity_total;
    let err = implied_rate - swap_rate;

    if err.abs() < tol {
      return mid;
    }
    if err > T::zero() {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  half * (lo + hi)
}

#[cfg(test)]
mod tests {
  use super::super::types::Instrument;
  use super::super::types::InterpolationMethod;
  use super::*;

  #[test]
  fn bootstrap_single_deposit() {
    let inst: Vec<Instrument<f64>> = vec![Instrument::Deposit {
      maturity: 1.0,
      rate: 0.05,
    }];
    let curve = bootstrap(&inst, InterpolationMethod::LinearOnZeroRates);
    assert!(
      !curve.is_empty(),
      "bootstrap should produce at least one point"
    );
  }

  #[test]
  fn bootstrap_iterative_swap() {
    let inst: Vec<Instrument<f64>> = vec![
      Instrument::Deposit {
        maturity: 0.25,
        rate: 0.04,
      },
      Instrument::Swap {
        maturity: 1.0,
        rate: 0.045,
        frequency: 2,
      },
    ];
    let curve = bootstrap_iterative(&inst, InterpolationMethod::LinearOnZeroRates, 1e-10, 50);
    assert!(!curve.is_empty());
  }
}

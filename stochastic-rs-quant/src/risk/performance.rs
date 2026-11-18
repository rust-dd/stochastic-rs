//! Sharpe, Sortino, information and Calmar ratios; each takes the annualisation factor
//! (periods per year) as an argument.
//!
//! Reference: Sharpe (1994), J. Portfolio Management 21(1), 49–58, DOI 10.3905/jpm.1994.409501
//! Reference: Sortino & van der Meer (1991), J. Portfolio Management 17(4), 10.3905/jpm.1991.409343
//! Reference: Young (1991), "Calmar Ratio: A Smoother Tool", Futures Magazine

use ndarray::ArrayView1;

use super::drawdown::equity_from_returns;
use super::drawdown::max_drawdown;
use super::input::finite;
use super::input::finite_positive;
use super::input::is_constant;
use super::input::validate_samples;
use crate::QuantError;
use crate::traits::RealExt;

/// Arithmetic mean of a slice.
fn mean<T: RealExt>(xs: ArrayView1<T>) -> T {
  xs.iter().fold(T::zero(), |a, &v| a + v) / T::from_usize_(xs.len())
}

/// Mean and Bessel-corrected deviation of finite, non-constant samples, whose deviation only an
/// under- or overflow keeps from being finite and positive.
fn mean_and_deviation<T: RealExt>(
  xs: ArrayView1<T>,
  what: &'static str,
) -> Result<(T, T), QuantError> {
  let m = mean(xs);
  let var = xs.iter().fold(T::zero(), |a, &v| a + (v - m).powi(2)) / T::from_usize_(xs.len() - 1);
  Ok((m, finite_positive(var.sqrt(), what, "the deviation")?))
}

/// Annualised Sharpe ratio over `k` periods a year, NaN for a constant series; `InsufficientData`
/// below two returns, `InvalidInput` for a non-finite one, `Numerical` if `σ` under- or overflows.
///
/// $$\mathrm{Sharpe}=\frac{\bar r-r_f^{\text{period}}}{\sigma}\sqrt{k}.$$
pub fn sharpe_ratio<T: RealExt>(
  returns: ArrayView1<T>,
  risk_free_per_period: T,
  periods_per_year: T,
) -> Result<T, QuantError> {
  validate_samples(returns, "Sharpe ratio", 2)?;
  if is_constant(returns) {
    return Ok(T::nan());
  }
  let (m, s) = mean_and_deviation(returns, "Sharpe ratio")?;
  Ok((m - risk_free_per_period) / s * periods_per_year.sqrt())
}

/// Annualised Sortino ratio, NaN with no return below the target; the downside deviation divides
/// by all `n` returns (Bawa & Lindenberg 1977), not the count below. Errors as [`sharpe_ratio`]'s.
///
/// $$\mathrm{Sortino}=\frac{\bar r-\mathrm{MAR}}{\sigma_{\text{down}}}\sqrt{k}.$$
pub fn sortino_ratio<T: RealExt>(
  returns: ArrayView1<T>,
  mar_per_period: T,
  periods_per_year: T,
) -> Result<T, QuantError> {
  validate_samples(returns, "Sortino ratio", 2)?;
  let n = returns.len();
  let m = mean(returns);
  let mut sum_sq = T::zero();
  let mut below = false;
  for &r in returns.iter() {
    let d = r - mar_per_period;
    if d < T::zero() {
      sum_sq += d.powi(2);
      below = true;
    }
  }
  if !below {
    return Ok(T::nan());
  }
  let m = finite(m, "Sortino ratio", "the mean")?;
  let downside = (sum_sq / T::from_usize_(n)).sqrt();
  let downside = finite_positive(downside, "Sortino ratio", "the downside deviation")?;
  Ok((m - mar_per_period) / downside * periods_per_year.sqrt())
}

/// Annualised Information Ratio against `benchmark`, NaN for a constant excess series; errors as
/// [`sharpe_ratio`]'s, `Numerical` for an excess that overflows, `DimensionMismatch` for lengths.
///
/// $$\mathrm{IR}=\frac{\overline{r-r_b}}{\sigma_{r-r_b}}\sqrt{k}.$$
pub fn information_ratio<T: RealExt>(
  returns: ArrayView1<T>,
  benchmark: ArrayView1<T>,
  periods_per_year: T,
) -> Result<T, QuantError> {
  if benchmark.len() != returns.len() {
    return Err(QuantError::DimensionMismatch {
      what: "benchmark, one per return",
      expected: returns.len(),
      got: benchmark.len(),
    });
  }
  validate_samples(returns, "information ratio", 2)?;
  validate_samples(benchmark, "information ratio", 2)?;
  let diff = &returns - &benchmark;
  if let Some(i) = diff.iter().position(|excess| !excess.is_finite()) {
    finite(
      diff[i],
      "information ratio",
      &format!("the excess return {i}"),
    )?;
  }
  if is_constant(diff.view()) {
    return Ok(T::nan());
  }
  let (m, s) = mean_and_deviation(diff.view(), "information ratio")?;
  Ok(m / s * periods_per_year.sqrt())
}

/// Annualised Calmar ratio, NaN with no drawdown; `InsufficientData` for no return, `InvalidInput`
/// for a non-finite one, `Numerical` if the mean or the equity curve overflows.
///
/// $$\mathrm{Calmar}=\frac{\bar r\,k}{|\mathrm{MDD}|}.$$
pub fn calmar_ratio<T: RealExt>(
  returns: ArrayView1<T>,
  periods_per_year: T,
) -> Result<T, QuantError> {
  validate_samples(returns, "Calmar ratio", 1)?;
  let annualised_return = finite(mean(returns), "Calmar ratio", "the mean")? * periods_per_year;
  let equity = equity_from_returns(returns, T::one());
  let mdd = finite(
    max_drawdown(equity.view()),
    "Calmar ratio",
    "the maximum drawdown",
  )?;
  let mdd_abs = mdd.abs();
  if mdd_abs <= T::min_positive_val() {
    return Ok(T::nan());
  }
  Ok(annualised_return / mdd_abs)
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use super::*;

  /// Finite returns whose excess over the benchmark overflows are `Numerical`, not an invalid
  /// sample; a non-finite return or benchmark value stays `InvalidInput`.
  #[test]
  fn an_excess_return_that_overflows_is_numerical() {
    let returns = array![1e308, 0.01, -0.02];
    let benchmark = array![-1e308, 0.0, 0.0];
    let err = information_ratio(returns.view(), benchmark.view(), 252.0).unwrap_err();
    assert_eq!(
      err.to_string(),
      "information ratio: the excess return 0 of finite samples overflows to inf"
    );
    for (r, b) in [
      (array![0.01, f64::NAN, 0.0], array![0.0, 0.0, 0.0]),
      (array![0.01, 0.02, 0.0], array![0.0, f64::INFINITY, 0.0]),
    ] {
      let err = information_ratio(r.view(), b.view(), 252.0).unwrap_err();
      assert!(matches!(err, QuantError::InvalidInput { .. }), "{err}");
    }
  }
}

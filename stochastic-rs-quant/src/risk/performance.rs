//! Performance ratios — Sharpe, Sortino, Information Ratio, Calmar.
//!
//! Reference: Sharpe, "The Sharpe Ratio", Journal of Portfolio Management,
//! 21(1), 49–58 (1994). DOI: 10.3905/jpm.1994.409501
//!
//! Reference: Sortino & van der Meer, "Downside Risk", Journal of Portfolio
//! Management, 17(4), 27–31 (1991). DOI: 10.3905/jpm.1991.409343
//!
//! Reference: Young, "Calmar Ratio: A Smoother Tool", Futures Magazine (1991).
//!
//! All ratios accept an explicit annualisation factor (periods per year) so
//! no `252`, `365`, or `12` constants are hard-coded — callers pass whatever
//! matches their sampling frequency.

use ndarray::ArrayView1;

use super::drawdown::equity_from_returns;
use super::drawdown::max_drawdown;
use super::var::validate_samples;
use crate::QuantError;
use crate::traits::RealExt;

/// Arithmetic mean of a slice.
fn mean<T: RealExt>(xs: ArrayView1<T>) -> T {
  xs.iter().fold(T::zero(), |a, &v| a + v) / T::from_usize_(xs.len())
}

/// Sample standard deviation (Bessel-corrected).
fn stdev<T: RealExt>(xs: ArrayView1<T>) -> T {
  let m = mean(xs);
  let var = xs.iter().fold(T::zero(), |a, &v| a + (v - m).powi(2)) / T::from_usize_(xs.len() - 1);
  var.sqrt()
}

/// Annualised Sharpe ratio, NaN for a zero `σ`; `InsufficientData` below two returns,
/// `InvalidInput` for a non-finite one.
///
/// $$\mathrm{Sharpe}=\frac{\bar r-r_f^{\text{period}}}{\sigma}\sqrt{k},$$
/// where $k$ is the number of periods per year.
pub fn sharpe_ratio<T: RealExt>(
  returns: ArrayView1<T>,
  risk_free_per_period: T,
  periods_per_year: T,
) -> Result<T, QuantError> {
  validate_samples(returns, "Sharpe ratio", 2)?;
  let m = mean(returns);
  let s = stdev(returns);
  if s <= T::min_positive_val() {
    return Ok(T::nan());
  }
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
  for &r in returns.iter() {
    let d = r - mar_per_period;
    if d < T::zero() {
      sum_sq += d.powi(2);
    }
  }
  let downside = (sum_sq / T::from_usize_(n)).sqrt();
  if downside <= T::min_positive_val() {
    return Ok(T::nan());
  }
  Ok((m - mar_per_period) / downside * periods_per_year.sqrt())
}

/// Annualised Information Ratio against `benchmark`, NaN for a zero tracking error; errors as
/// [`sharpe_ratio`]'s over the excess returns, and `DimensionMismatch` for unequal lengths.
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
  let diff = &returns.to_owned() - &benchmark.to_owned();
  validate_samples(diff.view(), "information ratio", 2)?;
  let m = mean(diff.view());
  let s = stdev(diff.view());
  if s <= T::min_positive_val() {
    return Ok(T::nan());
  }
  Ok(m / s * periods_per_year.sqrt())
}

/// Annualised Calmar ratio, NaN with no drawdown; `InsufficientData` for no return, `InvalidInput`
/// for a non-finite one.
///
/// $$\mathrm{Calmar}=\frac{\bar r\,k}{|\mathrm{MDD}|}.$$
pub fn calmar_ratio<T: RealExt>(
  returns: ArrayView1<T>,
  periods_per_year: T,
) -> Result<T, QuantError> {
  validate_samples(returns, "Calmar ratio", 1)?;
  let annualised_return = mean(returns) * periods_per_year;
  let equity = equity_from_returns(returns, T::one());
  let mdd = max_drawdown(equity.view());
  let mdd_abs = mdd.abs();
  if mdd_abs <= T::min_positive_val() {
    return Ok(T::nan());
  }
  Ok(annualised_return / mdd_abs)
}

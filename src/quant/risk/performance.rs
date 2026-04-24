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
use crate::traits::FloatExt;

/// Arithmetic mean of a slice.
fn mean<T: FloatExt>(xs: ArrayView1<T>) -> T {
  let n = xs.len();
  assert!(n > 0, "empty series");
  xs.iter().fold(T::zero(), |a, &v| a + v) / T::from_usize_(n)
}

/// Sample standard deviation (Bessel-corrected).
fn stdev<T: FloatExt>(xs: ArrayView1<T>) -> T {
  let n = xs.len();
  assert!(n >= 2, "need at least two observations for stdev");
  let m = mean(xs);
  let var = xs.iter().fold(T::zero(), |a, &v| a + (v - m).powi(2)) / T::from_usize_(n - 1);
  var.sqrt()
}

/// Annualised Sharpe ratio.
///
/// $$\mathrm{Sharpe}=\frac{\bar r-r_f^{\text{period}}}{\sigma}\sqrt{k},$$
/// where $k$ is the number of periods per year.
pub fn sharpe_ratio<T: FloatExt>(
  returns: ArrayView1<T>,
  risk_free_per_period: T,
  periods_per_year: T,
) -> T {
  let m = mean(returns);
  let s = stdev(returns);
  if s <= T::min_positive_val() {
    return T::zero();
  }
  (m - risk_free_per_period) / s * periods_per_year.sqrt()
}

/// Annualised Sortino ratio with target `mar` (minimum acceptable return).
///
/// $$\mathrm{Sortino}=\frac{\bar r-\mathrm{MAR}}{\sigma_{\text{down}}}\sqrt{k},$$
/// where the downside deviation uses only returns below MAR.
pub fn sortino_ratio<T: FloatExt>(
  returns: ArrayView1<T>,
  mar_per_period: T,
  periods_per_year: T,
) -> T {
  let n = returns.len();
  assert!(n >= 2, "need at least two observations for Sortino");
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
    return T::zero();
  }
  (m - mar_per_period) / downside * periods_per_year.sqrt()
}

/// Annualised Information Ratio against a benchmark return series.
///
/// $$\mathrm{IR}=\frac{\overline{r-r_b}}{\sigma_{r-r_b}}\sqrt{k}.$$
pub fn information_ratio<T: FloatExt>(
  returns: ArrayView1<T>,
  benchmark: ArrayView1<T>,
  periods_per_year: T,
) -> T {
  assert_eq!(
    returns.len(),
    benchmark.len(),
    "returns and benchmark must have matching length"
  );
  let diff = &returns.to_owned() - &benchmark.to_owned();
  let m = mean(diff.view());
  let s = stdev(diff.view());
  if s <= T::min_positive_val() {
    return T::zero();
  }
  m / s * periods_per_year.sqrt()
}

/// Annualised Calmar ratio — annualised return divided by max drawdown.
///
/// $$\mathrm{Calmar}=\frac{\bar r\,k}{|\mathrm{MDD}|}.$$
pub fn calmar_ratio<T: FloatExt>(returns: ArrayView1<T>, periods_per_year: T) -> T {
  let annualised_return = mean(returns) * periods_per_year;
  let equity = equity_from_returns(returns, T::one());
  let mdd = max_drawdown(equity.view());
  let mdd_abs = mdd.abs();
  if mdd_abs <= T::min_positive_val() {
    return T::zero();
  }
  annualised_return / mdd_abs
}

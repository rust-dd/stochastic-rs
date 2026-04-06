use crate::traits::FloatExt;

/// Dirty / clean bond price decomposition.
#[derive(Debug, Clone)]
pub struct BondPrice<T: FloatExt> {
  /// Present value including accrued interest.
  pub dirty_price: T,
  /// Accrued interest at settlement / valuation.
  pub accrued_interest: T,
  /// Present value net of accrued interest.
  pub clean_price: T,
}

/// Standard fixed-rate bond analytics.
#[derive(Debug, Clone)]
pub struct BondAnalytics<T: FloatExt> {
  /// Dirty price including accrued interest.
  pub dirty_price: T,
  /// Clean price excluding accrued interest.
  pub clean_price: T,
  /// Accrued interest at settlement / valuation.
  pub accrued_interest: T,
  /// Yield-to-maturity under the supplied compounding convention.
  pub yield_to_maturity: T,
  /// Macaulay duration in years.
  pub macaulay_duration: T,
  /// Modified duration.
  pub modified_duration: T,
  /// Convexity.
  pub convexity: T,
}

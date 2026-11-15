//! Tabulated critical values of a test statistic, and the two-sided normal p-value.

use stochastic_rs_distributions::special::norm_cdf;

/// Critical values at ascending levels; `value_at` takes the smallest tabulated level ≥ `alpha`.
#[derive(Clone, Debug, PartialEq)]
pub struct CriticalValues {
  entries: Vec<(f64, f64)>,
}

impl CriticalValues {
  /// `(level, value)` pairs in any order; panics unless every level is in `(0, 1)`, no value is NaN
  /// and the levels are distinct.
  pub fn new(entries: impl IntoIterator<Item = (f64, f64)>) -> Self {
    let mut entries = entries.into_iter().collect::<Vec<(f64, f64)>>();
    for &(level, value) in &entries {
      assert!(
        level > 0.0 && level < 1.0,
        "level must satisfy `0.0 < level < 1.0`, got level = {level:?}"
      );
      assert!(
        !value.is_nan(),
        "value must satisfy `!value.is_nan()`, got value = {value:?}"
      );
    }
    entries.sort_by(|a, b| a.0.total_cmp(&b.0));
    assert!(
      entries.windows(2).all(|w| w[0].0 != w[1].0),
      "levels must satisfy `distinct`, got levels = {:?}",
      entries.iter().map(|e| e.0).collect::<Vec<f64>>()
    );
    Self { entries }
  }

  /// The `(level, value)` pairs, ascending in level.
  pub fn entries(&self) -> &[(f64, f64)] {
    &self.entries
  }

  /// The value tabulated at exactly `alpha`.
  pub fn at(&self, alpha: f64) -> Option<f64> {
    self.entries.iter().find(|e| e.0 == alpha).map(|e| e.1)
  }

  /// The value at the smallest tabulated level ≥ `alpha`; panics when `alpha` exceeds every
  /// tabulated level.
  pub fn value_at(&self, alpha: f64) -> f64 {
    let largest = self.entries.last().map_or(0.0, |e| e.0);
    assert!(
      alpha <= largest,
      "alpha must satisfy `alpha <= {largest}` (the largest tabulated level), got alpha = {alpha:?}"
    );
    self
      .entries
      .iter()
      .find(|e| e.0 >= alpha)
      .map(|e| e.1)
      .unwrap()
  }
}

/// Two-sided normal p-value `2 Φ(−|z|)`, read off the lower tail so a far-tail statistic keeps its
/// digits.
pub(crate) fn two_sided_p_value(statistic: f64) -> f64 {
  2.0 * norm_cdf(-statistic.abs())
}

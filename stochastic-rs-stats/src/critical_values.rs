//! Tabulated critical values of a test statistic.

use stochastic_rs_distributions::special::norm_cdf;

/// Critical values at ascending levels; `value_at` takes the smallest tabulated level ≥ `alpha`.
#[derive(Clone, Debug, PartialEq)]
pub struct CriticalValues {
  entries: Vec<(f64, f64)>,
}

impl CriticalValues {
  /// `(level, value)` pairs in any order; panics unless the table is non-empty, every level is in
  /// `(0, 1)`, no value is NaN and the levels are distinct.
  pub fn new(entries: impl IntoIterator<Item = (f64, f64)>) -> Self {
    let mut entries = entries.into_iter().collect::<Vec<(f64, f64)>>();
    assert!(
      !entries.is_empty(),
      "entries must satisfy `!entries.is_empty()`, got entries = {entries:?}"
    );
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

#[cfg(test)]
mod tests {
  use super::two_sided_p_value;

  /// `p = erfc(|z|/√2)`, mpmath at 50 digits; the far tail keeps its relative digits.
  #[test]
  fn two_sided_p_value_keeps_its_tail() {
    let pins = [
      (0.0, 1.0),
      (1.0, 0.3173105078629141),
      (3.0, 0.002699796063260189),
      (5.0, 5.733031437583878e-7),
      (6.0, 1.973175290075396e-9),
      (8.0, 1.2441921148543568e-15),
      (9.0, 2.257176811907681e-19),
      (12.0, 3.552964224155358e-33),
      (20.0, 5.5072482372124674e-89),
    ];
    for (z, want) in pins {
      let got = two_sided_p_value(z);
      assert!(
        ((got - want) / want).abs() < 1e-13,
        "z = {z}: p = {got}, want {want}"
      );
      assert_eq!(two_sided_p_value(-z), got);
    }
  }
}

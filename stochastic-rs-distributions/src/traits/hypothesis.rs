//! `HypothesisTest`: the two answers every test result shares, so stats and copulas report alike.

/// A test result: its statistic, and the rejection at the alpha the test was run with —
/// `None` when the result carries no rejection rule (informational statistics, a bootstrap p-value alone).
pub trait HypothesisTest {
  /// The test statistic.
  fn statistic(&self) -> f64;

  /// Rejection at the alpha baked into the result; `None` when the result embeds no rejection decision.
  fn null_rejected(&self) -> Option<bool>;
}

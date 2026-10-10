/// A test result: its statistic, and the rejection at the alpha the test was run with —
/// `None` when the result carries no rejection rule (informational statistics, a bootstrap p-value alone).
pub trait HypothesisTest {
  fn statistic(&self) -> f64;

  /// `Some(true)` when the null hypothesis is rejected.
  fn null_rejected(&self) -> Option<bool>;
}

//! Re-exports of upstream traits so `crate::traits::Foo` continues to resolve
//! inside the stats sub-crate's source files. Plus the local
//! [`HypothesisTest`] trait that unifies hypothesis-test result types.

pub use stochastic_rs_distributions::traits::DistributionExt;
pub use stochastic_rs_distributions::traits::DistributionSampler;
pub use stochastic_rs_distributions::traits::FloatExt;
pub use stochastic_rs_distributions::traits::Fn1D;
pub use stochastic_rs_distributions::traits::Fn2D;
pub use stochastic_rs_distributions::traits::SimdFloatExt;
pub use stochastic_rs_stochastic::traits::Malliavin2DExt;
pub use stochastic_rs_stochastic::traits::MalliavinExt;
pub use stochastic_rs_stochastic::traits::ProcessExt;

/// Unifies `*Result` types produced by hypothesis-test estimators.
///
/// Implemented by stationarity tests (ADF, KPSS, ERS-DFGLS,
/// Phillips-Perron, Leybourne-McCabe) and normality tests (Jarque-Bera,
/// Anderson-Darling, Shapiro-Francia). Each test exposes its statistic;
/// the rejection decision is `Some(true|false)` when the test embeds a
/// rejection rule for a fixed `alpha` chosen at run time, or `None` when
/// the result is informational only (e.g. Phillips-Perron with the
/// `Z(alpha)` output that lacks tabulated critical values in this crate).
///
/// The trait does not specify the *p-value* signature because some tests
/// (ADF, KPSS, ERS, PP-tau) do not currently return a p-value and only
/// expose tabulated critical values. Use the underlying struct's named
/// fields when you need the p-value or critical-values directly.
pub trait HypothesisTest {
  /// The test statistic.
  fn statistic(&self) -> f64;
  /// Whether the null hypothesis is rejected at the test-specific `alpha`
  /// already baked into the result. Returns `None` when the result type
  /// does not embed a rejection decision.
  fn null_rejected(&self) -> Option<bool>;
}

/// Cumulants of the log-price distribution.
#[derive(Debug, Clone)]
pub struct Cumulants {
  /// First cumulant, the mean of $\ln(S_T/F_T)$.
  pub c1: f64,
  /// Second cumulant (variance of log-return).
  pub c2: f64,
  /// Fourth cumulant; `None` for a model that does not compute it (the Heston family, Bates).
  pub c4: Option<f64>,
}

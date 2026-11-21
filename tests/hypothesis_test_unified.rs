use stochastic_rs::copulas::gof::GofResult;
use stochastic_rs::prelude::HypothesisTest;
use stochastic_rs::stats::normality::jarque_bera::JarqueBeraTest;

fn answers<T: HypothesisTest>(result: &T) -> (f64, Option<bool>) {
  (result.statistic(), result.null_rejected())
}

#[test]
fn a_stats_and_a_copula_result_answer_through_one_trait() {
  let jb = JarqueBeraTest {
    statistic: 7.5,
    p_value: Some(0.02),
    critical_values: None,
    rejects_null: true,
    skewness: 0.4,
    excess_kurtosis: 1.1,
  };
  let gof = GofResult {
    statistic: 0.042,
    p_value: 0.31,
    replications: 200,
    alpha: 0.05,
    rejects_null: false,
  };
  assert_eq!(answers(&jb), (7.5, Some(true)));
  assert_eq!(answers(&gof), (0.042, Some(false)));
}

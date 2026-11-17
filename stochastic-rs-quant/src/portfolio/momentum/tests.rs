use super::*;

fn dummy_evals() -> Vec<AssetModelEstimate> {
  vec![
    AssetModelEstimate {
      ticker: "AAA".to_string(),
      annualized_return: 0.12,
      implied_vol: 0.2,
      model_label: "gbm".to_string(),
      calibration_window: 63,
      rolling_error: 0.1,
    },
    AssetModelEstimate {
      ticker: "BBB".to_string(),
      annualized_return: 0.08,
      implied_vol: 0.15,
      model_label: "gbm".to_string(),
      calibration_window: 63,
      rolling_error: 0.1,
    },
    AssetModelEstimate {
      ticker: "CCC".to_string(),
      annualized_return: 0.03,
      implied_vol: 0.2,
      model_label: "gbm".to_string(),
      calibration_window: 63,
      rolling_error: 0.1,
    },
  ]
}

#[test]
fn compute_scores_generates_expected_values() {
  let scores = compute_scores(&dummy_evals(), 0.02);
  assert_eq!(scores.len(), 3);
  let aaa = scores.iter().find(|s| s.ticker == "AAA").unwrap();
  assert!((aaa.momentum_score - 0.5).abs() < 1e-12);
}

#[test]
fn build_portfolio_equal_weights() {
  let scores = compute_scores(&dummy_evals(), 0.0);
  let pf = build_portfolio(&scores, 2, 1, WeightScheme::Equal, None);

  let long_sum: f64 = pf.long_positions.iter().map(|(_, w)| *w).sum();
  let short_sum: f64 = pf.short_positions.iter().map(|(_, w)| *w).sum();

  assert!((long_sum - 1.0).abs() < 1e-12);
  assert!((short_sum - 1.0).abs() < 1e-12);
}

/// A NaN score ranks last whatever its sign, which an x86 and an aarch64 NaN do not share.
#[test]
fn a_nan_score_ranks_last_whatever_its_sign() {
  let mut scores = compute_scores(&dummy_evals(), 0.0);
  let ends = |scores: &[MomentumScore]| {
    let pf = build_portfolio(scores, 1, 1, WeightScheme::Equal, None);
    (
      pf.long_positions[0].0.clone(),
      pf.short_positions[0].0.clone(),
    )
  };
  for nan in [f64::NAN, -f64::NAN] {
    scores[1].momentum_score = nan;
    assert_eq!(ends(&scores), ("AAA".to_string(), "BBB".to_string()));
    let averages = decile_analysis(&scores)
      .iter()
      .map(|bucket| bucket.avg_momentum_score)
      .collect::<Vec<_>>();
    assert!(
      (averages[0] - 0.6).abs() < 1e-12 && averages[2].is_nan(),
      "{averages:?}"
    );
  }
  scores[0].momentum_score = -f64::NAN;
  scores[1].momentum_score = f64::NAN;
  let ranked = decile_analysis(&scores)
    .iter()
    .map(|bucket| bucket.tickers.concat())
    .collect::<Vec<_>>();
  assert_eq!(ranked, ["CCC", "AAA", "BBB"]);
}

#[test]
fn compute_scores_from_custom_model_estimate_type() {
  struct CustomEstimate {
    id: &'static str,
    mu: f64,
    sigma: f64,
  }

  impl ModelEstimate for CustomEstimate {
    fn ticker(&self) -> &str {
      self.id
    }

    fn annualized_return(&self) -> f64 {
      self.mu
    }

    fn implied_vol(&self) -> f64 {
      self.sigma
    }
  }

  let xs = vec![
    CustomEstimate {
      id: "X1",
      mu: 0.10,
      sigma: 0.2,
    },
    CustomEstimate {
      id: "X2",
      mu: 0.07,
      sigma: 0.1,
    },
  ];

  let scores = compute_scores(&xs, 0.02);
  assert_eq!(scores.len(), 2);
  assert_eq!(scores[0].model_label, "unknown");
}

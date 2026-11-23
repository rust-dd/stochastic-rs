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

/// A NaN score of either sign is in neither leg and in no decile.
#[test]
fn a_nan_score_is_in_neither_leg_and_in_no_decile() {
  let mut scores = compute_scores(&dummy_evals(), 0.0);
  for nan in [f64::NAN, -f64::NAN] {
    scores[1].momentum_score = nan;
    let pf = build_portfolio(&scores, 2, 2, WeightScheme::Equal, None);
    assert_eq!(
      pf.long_positions,
      [("AAA".to_string(), 0.5), ("CCC".to_string(), 0.5)]
    );
    assert!(pf.short_positions.is_empty(), "{:?}", pf.short_positions);
    let pf = build_portfolio(&scores, 1, 1, WeightScheme::Equal, None);
    assert_eq!(pf.short_positions, [("CCC".to_string(), 1.0)]);
    let ranked = decile_analysis(&scores)
      .iter()
      .map(|bucket| (bucket.tickers.concat(), bucket.avg_momentum_score))
      .collect::<Vec<_>>();
    assert_eq!(ranked.len(), 2);
    assert_eq!(ranked[0].0, "AAA");
    assert!((ranked[0].1 - 0.6).abs() < 1e-12 && (ranked[1].1 - 0.15).abs() < 1e-12);
  }
  scores[0].momentum_score = f64::NAN;
  scores[2].momentum_score = f64::NAN;
  assert!(decile_analysis(&scores).is_empty());
  let pf = build_portfolio(&scores, 1, 1, WeightScheme::Equal, None);
  assert!(pf.long_positions.is_empty() && pf.short_positions.is_empty());
}

/// The score is `(return − risk_free) / vol`, NaN unless the vol is positive.
#[test]
fn a_score_is_nan_unless_the_vol_is_positive() {
  let mut evals = dummy_evals();
  evals[0].implied_vol = 0.0;
  evals[1].implied_vol = f64::NAN;
  evals[2].implied_vol = 1e-13;
  let scores = compute_scores(&evals, 0.01);
  assert!(scores[0].momentum_score.is_nan() && scores[1].momentum_score.is_nan());
  assert_eq!(scores[2].momentum_score, (0.03 - 0.01) / 1e-13);
}

/// A score that overflows at a subnormal vol is NaN, as one at no vol is.
#[test]
fn a_score_that_overflows_is_nan() {
  let mut evals = dummy_evals();
  evals[0].implied_vol = 1e-310;
  let score = compute_scores(&evals, 0.01)[0].momentum_score;
  assert!(score.is_nan(), "{score}");
}

/// An infinite score of either sign is in neither leg and in no decile, as a NaN one is.
#[test]
fn an_infinite_score_is_in_neither_leg_and_in_no_decile() {
  let mut scores = compute_scores(&dummy_evals(), 0.0);
  for infinite in [f64::INFINITY, f64::NEG_INFINITY] {
    scores[1].momentum_score = infinite;
    let pf = build_portfolio(&scores, 2, 2, WeightScheme::ScoreWeighted, None);
    let legs = pf.long_positions.iter().chain(&pf.short_positions);
    assert!(
      legs
        .clone()
        .all(|(ticker, w)| ticker != "BBB" && w.is_finite()),
      "{pf:?}"
    );
    let buckets = decile_analysis(&scores);
    assert!(
      buckets
        .iter()
        .all(|b| !b.tickers.iter().any(|t| t == "BBB"))
    );
  }
}

/// `ScoreWeighted` is `|sᵢ| / Σ|sⱼ|` at any scale: scores whose sum overflows, or underflows below
/// `1e-15`, still weigh by size; only all-zero scores weigh equally.
#[test]
fn score_weights_are_proportional_at_any_scale() {
  let mut scores = compute_scores(&dummy_evals(), 0.0);
  let weights = |scores: &[MomentumScore]| {
    let pf = build_portfolio(scores, 3, 0, WeightScheme::ScoreWeighted, None);
    pf.long_positions
      .iter()
      .map(|(_, w)| *w)
      .collect::<Vec<_>>()
  };
  for (scale, want) in [
    (1e308, [0.5, 1.0 / 3.0, 1.0 / 6.0]),
    (1e-20, [0.5, 1.0 / 3.0, 1.0 / 6.0]),
    (0.0, [1.0 / 3.0; 3]),
  ] {
    for (score, size) in scores.iter_mut().zip([1.5, 1.0, 0.5]) {
      score.momentum_score = size * scale;
    }
    let got = weights(&scores);
    assert!(
      got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-15),
      "{scale}: {got:?}"
    );
  }
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

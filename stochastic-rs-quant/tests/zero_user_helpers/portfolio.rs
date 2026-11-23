use stochastic_rs_quant::portfolio::data::align_return_series_by_index;
use stochastic_rs_quant::portfolio::data::log_returns_series;
use stochastic_rs_quant::portfolio::engine::PortfolioEngine;
use stochastic_rs_quant::portfolio::engine::PortfolioEngineConfig;
use stochastic_rs_quant::portfolio::momentum::AssetModelEstimate;
use stochastic_rs_quant::portfolio::momentum::MomentumPortfolio;
use stochastic_rs_quant::portfolio::momentum::MomentumScore;
use stochastic_rs_quant::portfolio::momentum::build_portfolio_target;
use stochastic_rs_quant::portfolio::momentum::build_portfolio_target_with_corr;
use stochastic_rs_quant::portfolio::momentum::compute_scores;
use stochastic_rs_quant::portfolio::types::OptimizerMethod;

/// `ln(cᵢ / cᵢ₋₁)` per pair of consecutive closes, NaN in its place for a pair with a close that
/// is not finite and positive, so each return keeps its date.
#[test]
fn log_returns_series_is_the_log_ratio_of_consecutive_closes() {
  let closes = [100.0, 110.0, 99.0];
  let expected = [(110.0_f64 / 100.0).ln(), (99.0_f64 / 110.0).ln()];
  assert_eq!(log_returns_series(&closes).to_vec(), expected);
  for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
    let returns = log_returns_series(&[100.0, bad, 99.0, 108.9]);
    assert_eq!(returns.len(), 3, "{bad}");
    assert!(
      returns[0].is_nan() && returns[1].is_nan(),
      "{bad}: {returns}"
    );
    assert_eq!(returns[2], (108.9_f64 / 99.0).ln());
  }
  assert!(log_returns_series(&[100.0]).is_empty() && log_returns_series(&[]).is_empty());
}

#[test]
fn align_return_series_by_index_takes_each_window_at_its_offset() {
  let series = vec![vec![0.1, 0.2, 0.3, 0.4], vec![1.0, 2.0, 3.0]];
  let aligned = align_return_series_by_index(&series, &[2, 1], 2).unwrap();
  assert_eq!(aligned, ndarray::array![[0.3, 0.4], [2.0, 3.0]]);
  assert!(align_return_series_by_index(&series, &[2, 2], 2).is_err());
  assert!(align_return_series_by_index(&series, &[2], 2).is_err());
}

fn estimate(i: usize) -> AssetModelEstimate {
  AssetModelEstimate {
    ticker: format!("T{i:02}"),
    annualized_return: 0.02 + 0.01 * i as f64,
    implied_vol: 0.2,
    model_label: "gbm".to_string(),
    calibration_window: 63,
    rolling_error: 0.0,
  }
}

/// Twelve assets fall into ten buckets, best score first: the first `12 mod 10` hold two, and
/// each average is the mean of its members' `(return − r) / vol`.
#[test]
fn momentum_deciles_split_the_ranked_scores_into_ten_buckets() {
  let risk_free = 0.01;
  let engine = PortfolioEngine::new(PortfolioEngineConfig {
    r: risk_free,
    ..PortfolioEngineConfig::default()
  });
  let estimates = (0..12).map(estimate).collect::<Vec<_>>();
  let buckets = engine.momentum_deciles(&estimates);
  let sizes = buckets.iter().map(|b| b.tickers.len()).collect::<Vec<_>>();
  assert_eq!(sizes, [2, 2, 1, 1, 1, 1, 1, 1, 1, 1]);
  assert_eq!(buckets[0].tickers, ["T11", "T10"]);
  assert_eq!(buckets[9].tickers, ["T00"]);
  let score = |i: usize| (0.02 + 0.01 * i as f64 - risk_free) / 0.2;
  let top = (score(11) + score(10)) / 2.0;
  assert!((buckets[0].avg_momentum_score - top).abs() < 1e-15);
  assert_eq!(buckets[9].avg_momentum_score, score(0));
  assert!(
    buckets
      .windows(2)
      .all(|pair| pair[0].avg_momentum_score > pair[1].avg_momentum_score)
  );
}

fn scores() -> Vec<MomentumScore> {
  let estimates = [(0.10, 0.1), (0.08, 0.2), (0.06, 0.4)]
    .iter()
    .enumerate()
    .map(
      |(i, &(annualized_return, implied_vol))| AssetModelEstimate {
        annualized_return,
        implied_vol,
        ..estimate(i)
      },
    )
    .collect::<Vec<_>>();
  compute_scores(&estimates, 0.0)
}

fn assert_positions(portfolio: &MomentumPortfolio, weights: &[f64], vol: f64) {
  assert!(portfolio.short_positions.is_empty());
  for ((ticker, weight), (i, expected)) in portfolio
    .long_positions
    .iter()
    .zip(weights.iter().enumerate())
  {
    assert_eq!(ticker, &format!("T{i:02}"));
    assert!((weight - expected).abs() < 1e-15, "{ticker}: {weight}");
  }
  let mean = weights[0] * 0.10 + weights[1] * 0.08 + weights[2] * 0.06;
  assert!((portfolio.expected_return - mean).abs() < 1e-15);
  assert!(
    (portfolio.expected_vol - vol).abs() < 1e-15,
    "{}",
    portfolio.expected_vol
  );
}

/// Inverse-vol weights are `(1/σᵢ) / Σ 1/σⱼ`. Uncorrelated, the portfolio vol is `√n / Σ 1/σⱼ`;
/// perfectly correlated, `n / Σ 1/σⱼ`.
#[test]
fn build_portfolio_target_weights_by_inverse_vol_under_either_correlation_input() {
  let inverse = [1.0 / 0.1, 1.0 / 0.2, 1.0 / 0.4];
  let total = inverse.iter().sum::<f64>();
  let weights = inverse.map(|x| x / total);
  let identity = vec![
    vec![1.0, 0.0, 0.0],
    vec![0.0, 1.0, 0.0],
    vec![0.0, 0.0, 1.0],
  ];
  let method = OptimizerMethod::InverseVol;
  let with_corr = build_portfolio_target_with_corr(&scores(), 0.08, 0.0, &identity, method);
  assert_positions(&with_corr, &weights, 3.0_f64.sqrt() / total);

  let base = [0.01, -0.02, 0.015, 0.003, -0.007];
  let comoving = [1.0, 2.0, 0.5]
    .iter()
    .map(|scale| base.iter().map(|r| scale * r).collect::<Vec<_>>())
    .collect::<Vec<_>>();
  let from_returns = build_portfolio_target(&scores(), 0.08, 0.0, &comoving, method);
  assert_positions(&from_returns, &weights, 3.0 / total);
}

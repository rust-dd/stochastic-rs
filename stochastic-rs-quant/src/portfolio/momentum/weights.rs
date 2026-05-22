//! Weighting schemes for momentum legs and portfolio-vol aggregation.

use super::types::MomentumScore;
use super::types::WeightScheme;

/// Assign within-leg weights for one momentum leg (long-only or short-only).
///
/// **`ScoreWeighted` normalisation:** weights are `|score_i| / sum(|score_j|)`
/// over the supplied `indices`. The resulting weights are **non-negative and
/// sum to 1 within the supplied leg**. When called on a short basket, callers
/// who want the short-side weights to also sum to 1 (so total long+short
/// gross exposure is 2) get that for free; callers who want a dollar-neutral
/// long/short with equal long and short notionals (sum to 1 each) should call
/// `assign_weights` separately for each leg and concatenate. Callers who
/// want long-side `+w_i` and short-side `−w_i` to sum to 0 (net-zero
/// exposure) must apply the sign downstream.
pub(super) fn assign_weights(
  indices: &[usize],
  scores: &[MomentumScore],
  scheme: WeightScheme,
) -> Vec<(usize, f64)> {
  if indices.is_empty() {
    return Vec::new();
  }

  match scheme {
    WeightScheme::Equal => {
      let w = 1.0 / indices.len() as f64;
      indices.iter().map(|&idx| (idx, w)).collect()
    }
    WeightScheme::ScoreWeighted => {
      let raw: Vec<f64> = indices
        .iter()
        .map(|&idx| scores[idx].momentum_score.abs())
        .collect();
      let total: f64 = raw.iter().sum();

      if total < 1e-15 {
        let w = 1.0 / indices.len() as f64;
        return indices.iter().map(|&idx| (idx, w)).collect();
      }

      indices
        .iter()
        .zip(raw.iter())
        .map(|(&idx, &v)| (idx, v / total))
        .collect()
    }
  }
}

pub(super) fn compute_portfolio_vol(
  signed_positions: &[(usize, f64)],
  scores: &[MomentumScore],
  corr: Option<&[Vec<f64>]>,
) -> f64 {
  if signed_positions.is_empty() {
    return 0.0;
  }

  let sigmas: Vec<f64> = signed_positions
    .iter()
    .map(|(idx, _)| scores[*idx].predicted_vol.max(0.0))
    .collect();

  if let Some(corr) = corr {
    let mut var = 0.0;
    for (i, (idx_i, w_i)) in signed_positions.iter().enumerate() {
      for (j, (idx_j, w_j)) in signed_positions.iter().enumerate() {
        let c_ij = corr
          .get(*idx_i)
          .and_then(|row| row.get(*idx_j))
          .copied()
          .unwrap_or(if idx_i == idx_j { 1.0 } else { 0.0 });
        var += w_i * w_j * sigmas[i] * sigmas[j] * c_ij;
      }
    }
    return var.abs().sqrt();
  }

  let var: f64 = signed_positions
    .iter()
    .zip(sigmas.iter())
    .map(|((_, w), s)| (w * s).powi(2))
    .sum();
  var.sqrt()
}

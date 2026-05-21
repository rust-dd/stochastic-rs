/// Empirical $W_1$ distance between two 1D sample sets using quantile coupling.
pub fn empirical_wasserstein_1(x: &[f64], y: &[f64]) -> f64 {
  let mut xs = x
    .iter()
    .copied()
    .filter(|v| v.is_finite())
    .collect::<Vec<f64>>();
  let mut ys = y
    .iter()
    .copied()
    .filter(|v| v.is_finite())
    .collect::<Vec<f64>>();

  if xs.is_empty() || ys.is_empty() {
    return f64::INFINITY;
  }

  xs.sort_by(|a, b| a.total_cmp(b));
  ys.sort_by(|a, b| a.total_cmp(b));

  let m = xs.len().max(ys.len());
  let mut acc = 0.0;
  for i in 0..m {
    let u = (i as f64 + 0.5) / m as f64;
    let qx = quantile_sorted(&xs, u);
    let qy = quantile_sorted(&ys, u);
    acc += (qx - qy).abs();
  }
  acc / m as f64
}

/// Computes bid-ask calibration tolerance:
/// $\varepsilon = \frac1M \sum_j |\mathrm{ask}_j - \mathrm{bid}_j|$.
pub fn bid_ask_tolerance(bid: &[f64], ask: &[f64]) -> f64 {
  if bid.is_empty() || ask.is_empty() {
    return 0.0;
  }
  let m = bid.len().min(ask.len());
  bid
    .iter()
    .zip(ask.iter())
    .take(m)
    .map(|(b, a)| (a - b).abs())
    .sum::<f64>()
    / m as f64
}

fn quantile_sorted(sorted: &[f64], u: f64) -> f64 {
  if sorted.len() == 1 {
    return sorted[0];
  }
  let z = u.clamp(0.0, 1.0);
  let pos = z * (sorted.len() as f64 - 1.0);
  let lo = pos.floor() as usize;
  let hi = pos.ceil() as usize;
  if lo == hi {
    sorted[lo]
  } else {
    let w = pos - lo as f64;
    sorted[lo] * (1.0 - w) + sorted[hi] * w
  }
}

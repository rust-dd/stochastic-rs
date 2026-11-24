use stochastic_rs_distributions::RealExt;

/// $W_1$ between the empirical measures of `x` and `y`, exact; NaN for an empty sample or one that
/// holds a value that is not finite.
///
/// $$
/// W_1=\int_0^1\bigl|F_x^{-1}(u)-F_y^{-1}(u)\bigr|\,du,\qquad
/// W_1=\frac1m\sum_{i=1}^m\bigl|x_{(i)}-y_{(i)}\bigr|\ \text{ for equal sizes } m
/// $$
pub fn empirical_wasserstein_1(x: &[f64], y: &[f64]) -> f64 {
  let finite = |sample: &[f64]| !sample.is_empty() && sample.iter().all(|v| v.is_finite());
  if !(finite(x) && finite(y)) {
    return f64::NAN;
  }
  let (mut xs, mut ys) = (x.to_vec(), y.to_vec());
  xs.sort_by(RealExt::nan_last);
  ys.sort_by(RealExt::nan_last);

  // Both quantile functions are steps: `xs[i]` on `(i/n, (i+1)/n]`, `ys[j]` on `(j/m, (j+1)/m]`.
  let (n, m) = (xs.len(), ys.len());
  let (mut i, mut j, mut done, mut acc) = (0, 0, 0, 0.0);
  while i < n && j < m {
    let next = ((i + 1) * m).min((j + 1) * n);
    acc += (xs[i] - ys[j]).abs() * (next - done) as f64;
    done = next;
    i += usize::from((i + 1) * m == next);
    j += usize::from((j + 1) * n == next);
  }
  acc / (n * m) as f64
}

/// The mean width $\frac1M \sum_j |\mathrm{ask}_j - \mathrm{bid}_j|$, NaN for no quote; panics
/// unless there is one ask per bid.
pub fn bid_ask_tolerance(bid: &[f64], ask: &[f64]) -> f64 {
  assert!(
    ask.len() == bid.len(),
    "ask must satisfy `ask.len() == bid.len()`, got ask.len() = {}, bid.len() = {}",
    ask.len(),
    bid.len()
  );
  if bid.is_empty() {
    return f64::NAN;
  }
  bid.iter().zip(ask).map(|(b, a)| (a - b).abs()).sum::<f64>() / bid.len() as f64
}

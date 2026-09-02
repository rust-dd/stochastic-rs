//! Derivative-free minimisation for the two-parameter copula fits: a plain
//! Nelder–Mead simplex (Nelder & Mead 1965; Lagarias, Reeds, Wright & Wright
//! 1998 parameters) on unconstrained coordinates, which the callers map onto
//! their parameter domains.

/// Minimises `f` from `start` with initial simplex steps `steps`; returns
/// the best vertex after `max_iter` iterations or once the simplex spread
/// falls below `tolerance`.
pub(crate) fn nelder_mead(
  f: impl Fn(&[f64]) -> f64,
  start: &[f64],
  steps: &[f64],
  max_iter: usize,
  tolerance: f64,
) -> Vec<f64> {
  let n = start.len();
  assert_eq!(steps.len(), n, "one initial step per coordinate");
  let mut simplex: Vec<Vec<f64>> = (0..=n)
    .map(|i| {
      let mut v = start.to_vec();
      if i > 0 {
        v[i - 1] += steps[i - 1];
      }
      v
    })
    .collect();
  let mut values: Vec<f64> = simplex.iter().map(|v| f(v)).collect();
  let (alpha, gamma, rho, sigma) = (1.0, 2.0, 0.5, 0.5);
  for _ in 0..max_iter {
    let mut order: Vec<usize> = (0..=n).collect();
    order.sort_by(|&a, &b| {
      values[a]
        .partial_cmp(&values[b])
        .unwrap_or(std::cmp::Ordering::Equal)
    });
    let simplex_sorted: Vec<Vec<f64>> = order.iter().map(|&i| simplex[i].clone()).collect();
    let values_sorted: Vec<f64> = order.iter().map(|&i| values[i]).collect();
    simplex = simplex_sorted;
    values = values_sorted;
    let spread = simplex[1..]
      .iter()
      .map(|v| {
        v.iter()
          .zip(&simplex[0])
          .map(|(a, b)| (a - b).abs())
          .fold(0.0, f64::max)
      })
      .fold(0.0, f64::max);
    if spread < tolerance && (values[n] - values[0]).abs() < tolerance {
      break;
    }
    let centroid: Vec<f64> = (0..n)
      .map(|j| simplex[..n].iter().map(|v| v[j]).sum::<f64>() / n as f64)
      .collect();
    let worst = simplex[n].clone();
    let reflect: Vec<f64> = centroid
      .iter()
      .zip(&worst)
      .map(|(c, w)| c + alpha * (c - w))
      .collect();
    let f_reflect = f(&reflect);
    if f_reflect < values[0] {
      let expand: Vec<f64> = centroid
        .iter()
        .zip(&reflect)
        .map(|(c, r)| c + gamma * (r - c))
        .collect();
      let f_expand = f(&expand);
      if f_expand < f_reflect {
        simplex[n] = expand;
        values[n] = f_expand;
      } else {
        simplex[n] = reflect;
        values[n] = f_reflect;
      }
    } else if f_reflect < values[n - 1] {
      simplex[n] = reflect;
      values[n] = f_reflect;
    } else {
      let contract: Vec<f64> = if f_reflect < values[n] {
        centroid
          .iter()
          .zip(&reflect)
          .map(|(c, r)| c + rho * (r - c))
          .collect()
      } else {
        centroid
          .iter()
          .zip(&worst)
          .map(|(c, w)| c + rho * (w - c))
          .collect()
      };
      let f_contract = f(&contract);
      if f_contract < values[n].min(f_reflect) {
        simplex[n] = contract;
        values[n] = f_contract;
      } else {
        let best = simplex[0].clone();
        for i in 1..=n {
          simplex[i] = simplex[i]
            .iter()
            .zip(&best)
            .map(|(x, b)| b + sigma * (x - b))
            .collect();
          values[i] = f(&simplex[i]);
        }
      }
    }
  }
  let best = (0..=n)
    .min_by(|&a, &b| {
      values[a]
        .partial_cmp(&values[b])
        .unwrap_or(std::cmp::Ordering::Equal)
    })
    .expect("non-empty simplex");
  simplex[best].clone()
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn finds_the_minimum_of_a_rosenbrock_valley() {
    let f = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
    let best = nelder_mead(f, &[-1.2, 1.0], &[0.5, 0.5], 5000, 1e-12);
    assert!(
      (best[0] - 1.0).abs() < 1e-4 && (best[1] - 1.0).abs() < 1e-4,
      "{best:?}"
    );
  }
}

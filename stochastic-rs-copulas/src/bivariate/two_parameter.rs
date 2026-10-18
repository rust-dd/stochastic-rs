//! Shared plumbing of the two-parameter Archimedean families BB1 and BB7: the numerical Kendall's τ and the
//! maximum-likelihood fit on unconstrained coordinates.

use ndarray::ArrayView1;

use crate::optim::nelder_mead;

/// Kendall's τ `1 − 4∫∫ ∂_u C ∂_v C` (Nelsen 2006, §5.1.1) from `h` alone, `∂_u C` being `h` with
/// swapped arguments (exact for exchangeable BB families); bounded, with no node on the boundary.
pub(crate) fn kendall_tau_numeric(h: impl Fn(f64, f64) -> f64, panels: usize) -> f64 {
  let width = 1.0 / panels as f64;
  let offset = 0.5 * width / 3.0_f64.sqrt();
  let nodes: Vec<f64> = (0..panels)
    .flat_map(|i| {
      let centre = (i as f64 + 0.5) * width;
      [centre - offset, centre + offset]
    })
    .collect();
  let mut acc = 0.0;
  for &u in &nodes {
    for &v in &nodes {
      acc += h(u, v) * h(v, u);
    }
  }
  // Every node carries the weight `width / 2` in each dimension.
  1.0 - 4.0 * acc * (0.5 * width).powi(2)
}

/// Maximum-likelihood `(θ, δ)` on the pseudo-observation columns `u`, `v` inside `(0, 1)`: Nelder–Mead over
/// `lower + exp(c)`, so the search is unconstrained, on the negative log-likelihood of the kernel `log_density(θ, δ)`.
pub(crate) fn fit_two_parameters<K: Fn(f64, f64) -> f64>(
  u: ArrayView1<f64>,
  v: ArrayView1<f64>,
  lower: (f64, f64),
  start: (f64, f64),
  log_density: impl Fn(f64, f64) -> K,
) -> (f64, f64) {
  let to_params = |c: &[f64]| (lower.0 + c[0].exp(), lower.1 + c[1].exp());
  let objective = |c: &[f64]| {
    let (theta, delta) = to_params(c);
    if !(theta.is_finite() && delta.is_finite()) {
      return f64::INFINITY;
    }
    let ln_c = log_density(theta, delta);
    let nll: f64 = u.iter().zip(&v).map(|(&a, &b)| -ln_c(a, b)).sum();
    if nll.is_finite() { nll } else { f64::INFINITY }
  };
  let start_coords = [
    (start.0 - lower.0).max(1e-6).ln(),
    (start.1 - lower.1).max(1e-6).ln(),
  ];
  let best = nelder_mead(objective, &start_coords, &[0.3, 0.3], 400, 1e-9);
  to_params(&best)
}

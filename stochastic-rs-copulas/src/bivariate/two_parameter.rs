//! Shared plumbing of the two-parameter Archimedean families BB1 and BB7:
//! the numerical Kendall's τ, the maximum-likelihood fit on unconstrained
//! coordinates, and the marginal clipping used by every evaluation.

use ndarray::Array2;

use crate::optim::nelder_mead;

/// `ln(1 − e^z)` for `z ≤ 0` without cancellation: `ln1p(−e^z)` when `e^z`
/// is small, `ln(−expm1(z))` when it is close to one.
pub(crate) fn ln_one_minus_exp(z: f64) -> f64 {
  if z < -std::f64::consts::LN_2 {
    (-z.exp()).ln_1p()
  } else {
    (-z.exp_m1()).ln()
  }
}

/// Clips a pseudo-observation into the open unit interval so the powers
/// and logarithms of the copula formulas stay finite.
pub(crate) fn clip(u: f64) -> f64 {
  u.clamp(1e-12, 1.0 - 1e-12)
}

/// Kendall's τ from the h-function alone,
/// `τ = 1 − 4 ∫∫ ∂_u C(u, v) ∂_v C(u, v) du dv` (Nelsen 2006, §5.1.1: the
/// integration-by-parts form of Theorem 5.1.3), by a composite two-point
/// Gauss–Legendre product rule on `panels × panels` cells. Both factors are
/// conditional CDFs in `[0, 1]`, so the integrand stays bounded for the
/// tail-dependent families whose density diverges at the corners, and no
/// node lies on the boundary, where the powers inside the h-functions lose
/// all their digits. `h` is `∂C/∂v`; `∂C/∂u` is read off it with the
/// arguments swapped, exact for the exchangeable BB families.
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

/// Solves `h(u) = p` for `u` in the open unit interval by bisection. `h` is
/// a conditional CDF, increasing in `u`, so the bracket always holds the
/// root; 100 halvings reach the resolution of `f64` in `(0, 1)`.
pub(crate) fn invert_h(h: impl Fn(f64) -> f64, p: f64) -> f64 {
  let (mut lo, mut hi) = (1e-12, 1.0 - 1e-12);
  for _ in 0..100 {
    let mid = 0.5 * (lo + hi);
    if h(mid) < p {
      lo = mid;
    } else {
      hi = mid;
    }
    if hi - lo <= 1e-15 {
      break;
    }
  }
  0.5 * (lo + hi)
}

/// Maximum-likelihood fit of `(θ, δ)` on the pseudo-observations `x`:
/// the parameters live at `lower + exp(coordinate)` so the search is
/// unconstrained, and the objective is the negative log-likelihood of
/// `log_density`.
pub(crate) fn fit_two_parameters(
  x: &Array2<f64>,
  lower: (f64, f64),
  start: (f64, f64),
  log_density: impl Fn(f64, f64, f64, f64) -> f64,
) -> (f64, f64) {
  let to_params = |c: &[f64]| (lower.0 + c[0].exp(), lower.1 + c[1].exp());
  let objective = |c: &[f64]| {
    let (theta, delta) = to_params(c);
    if !(theta.is_finite() && delta.is_finite()) {
      return f64::INFINITY;
    }
    let nll: f64 = x
      .rows()
      .into_iter()
      .map(|row| -log_density(clip(row[0]), clip(row[1]), theta, delta))
      .sum();
    if nll.is_finite() { nll } else { f64::INFINITY }
  };
  let start_coords = [
    (start.0 - lower.0).max(1e-6).ln(),
    (start.1 - lower.1).max(1e-6).ln(),
  ];
  let best = nelder_mead(objective, &start_coords, &[0.3, 0.3], 400, 1e-9);
  to_params(&best)
}

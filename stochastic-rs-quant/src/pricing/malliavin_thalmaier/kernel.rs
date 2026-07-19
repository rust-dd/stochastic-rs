//! Poisson kernel and `g^h_{i,j}` kernel functions.
//!
//! See Kohatsu-Higa & Yasuda (2008), §2.2 and §3 for the regularisation.

use ndarray::Array2;
use stochastic_rs_distributions::special::gamma;

use crate::traits::FloatExt;

/// Surface area of the unit sphere in ℝ^d: `a_d = 2π^{d/2} / Γ(d/2)`.
pub fn sphere_area<T: FloatExt>(d: usize) -> T {
  let dh = d as f64 / 2.0;
  T::from_f64_fast(2.0 * std::f64::consts::PI.powf(dh) / gamma(dh))
}

/// Regularised norm `|x|_h = √(Σ xᵢ² + h)`.
#[inline]
pub fn norm_h<T: FloatExt>(x: &[T], h: T) -> T {
  let s = x.iter().map(|xi| *xi * *xi).sum::<T>();
  (s + h).sqrt()
}

/// Gradient of the regularised Poisson kernel `∂Q_d^h/∂x_i`.
///
/// - `d = 2`: `∂Q₂^h/∂xᵢ = (1/a₂) · xᵢ / |x|_h²`
/// - `d ≥ 3`: `∂Q_d^h/∂xᵢ = (1/a_d) · xᵢ / |x|_h^d`
pub fn grad_poisson_reg<T: FloatExt>(x: &[T], h: T) -> Vec<T> {
  let d = x.len();
  assert!(d >= 2, "Poisson kernel requires d >= 2");
  let ad = sphere_area::<T>(d);
  let r = norm_h(x, h);
  let rd = r.powi(d as i32);
  let factor = T::one() / (ad * rd);
  x.iter().map(|&xi| xi * factor).collect()
}

/// Regularised second derivative kernel
/// `K^h_{i,j}(x) = ∂²Q_d^h(x) / (∂x_i ∂x_j)`.
pub fn kernel_k_ij_h<T: FloatExt>(x: &[T], h: T, i: usize, j: usize) -> T {
  let d = x.len();
  assert!(d >= 2, "Poisson kernel requires d >= 2");
  assert!(i < d && j < d, "kernel indices out of bounds");
  let ad = sphere_area::<T>(d);
  let r = norm_h(x, h);
  let rd = r.powi(d as i32);
  let delta = if i == j { T::one() } else { T::zero() };
  (delta / rd - T::from_usize_(d) * x[i] * x[j] / r.powi(d as i32 + 2)) / ad
}

/// Closed-form `g_{i,j}` for the 2-asset digital put
/// `f(x) = 1(0≤x₁≤K₁)·1(0≤x₂≤K₂)`.
///
/// From Kohatsu-Higa & Yasuda (2008), eq. (6.3). Valid as `h → 0`.
/// The second diagonal follows from `g₁₁(y) + g₂₂(y) = f(y)` because
/// the Poisson kernel is normalised by `ΔQ₂ = δ₀`.
/// Values on an edge use the one-sided limit from the rectangle interior. At
/// a corner the diagonal uses the relative interior path
/// `(|y₁-bound₁|/K₁) = (|y₂-bound₂|/K₂)`. The off-diagonal
/// logarithmic singularity is preserved.
pub fn g_digital_put_2d<T: FloatExt>(y: [T; 2], k: [T; 2]) -> [[T; 2]; 2] {
  assert!(
    y.iter().all(|value| value.is_finite()),
    "evaluation point must be finite"
  );
  assert!(
    k.iter()
      .all(|strike| strike.is_finite() && *strike > T::zero()),
    "digital-put strikes must be finite and positive"
  );
  let a2_inv = T::one() / (T::from_f64_fast(2.0) * T::from_f64_fast(std::f64::consts::PI));
  let scale = y
    .into_iter()
    .chain(k)
    .map(<T as num_traits::Float>::abs)
    .fold(T::zero(), |current, value| current.max(value));
  let y1 = y[0] / scale;
  let y2 = y[1] / scale;
  let k1 = k[0] / scale;
  let k2 = k[1] / scale;
  let y1mk1 = y1 - k1;
  let y2mk2 = y2 - k2;

  let g11 = a2_inv
    * (atan_ratio_interior_limit(y2, y1, k2, k1)
      - atan_ratio_interior_limit(y2mk2, y1, -k2, k1)
      - atan_ratio_interior_limit(y2, y1mk1, k2, -k1)
      + atan_ratio_interior_limit(y2mk2, y1mk1, -k2, -k1));

  let r00_squared = y1 * y1 + y2 * y2;
  let r11_squared = y1mk1 * y1mk1 + y2mk2 * y2mk2;
  let r10_squared = y1mk1 * y1mk1 + y2 * y2;
  let r01_squared = y1 * y1 + y2mk2 * y2mk2;
  let log_ratio = r00_squared.ln() + r11_squared.ln() - r10_squared.ln() - r01_squared.ln();
  let g21 = (a2_inv / T::from_f64_fast(2.0)) * log_ratio;
  let payoff = if y[0] >= T::zero() && y[0] <= k[0] && y[1] >= T::zero() && y[1] <= k[1] {
    T::one()
  } else {
    T::zero()
  };

  [[g11, g21], [g21, payoff - g11]]
}

fn atan_ratio_interior_limit<T: FloatExt>(
  numerator: T,
  denominator: T,
  numerator_direction: T,
  denominator_direction: T,
) -> T {
  if denominator != T::zero() {
    return (numerator / denominator).atan();
  }
  if numerator == T::zero() {
    return (numerator_direction / denominator_direction).atan();
  }

  let half_pi = T::from_f64_fast(std::f64::consts::FRAC_PI_2);
  if (numerator > T::zero()) == (denominator_direction > T::zero()) {
    half_pi
  } else {
    -half_pi
  }
}

/// Compute `g^h_{i,j}(y)` numerically for an arbitrary payoff via 2-D quadrature.
///
/// Uses composite midpoint rule on `[lo, hi]²` with `n_quad` points per axis.
pub fn g_kernel_numerical_2d<T, F>(
  y: &[T; 2],
  payoff: &F,
  h: T,
  lo: &[T; 2],
  hi: &[T; 2],
  n_quad: usize,
) -> Array2<T>
where
  T: FloatExt,
  F: Fn(&[T]) -> T,
{
  assert!(n_quad > 0, "n_quad must be positive");
  let dx0 = (hi[0] - lo[0]) / T::from_usize_(n_quad);
  let dx1 = (hi[1] - lo[1]) / T::from_usize_(n_quad);
  let area = dx0 * dx1;
  let half = T::from_f64_fast(0.5);
  let mut g = Array2::<T>::zeros((2, 2));

  for q0 in 0..n_quad {
    let x0 = lo[0] + (T::from_usize_(q0) + half) * dx0;
    for q1 in 0..n_quad {
      let x1 = lo[1] + (T::from_usize_(q1) + half) * dx1;
      let fval = payoff(&[x0, x1]);
      if fval.abs() < T::from_f64_fast(1e-15) {
        continue;
      }
      let dy = [y[0] - x0, y[1] - x1];
      let weighted = fval * area;
      for i in 0..2 {
        for j in 0..2 {
          g[[i, j]] += weighted * kernel_k_ij_h(&dy, h, i, j);
        }
      }
    }
  }
  g
}

/// Compute `g^h_{i,j}(y)` numerically in arbitrary dimension via tensor-product
/// midpoint quadrature on `[lo, hi]^d`.
///
/// This is the direct numerical counterpart of Theorem 6.1 in
/// Kohatsu-Higa--Yasuda when no closed-form `g_{i,j}` is available.
pub fn g_kernel_numerical_nd<T, F>(
  y: &[T],
  payoff: &F,
  h: T,
  lo: &[T],
  hi: &[T],
  n_quad: usize,
) -> Array2<T>
where
  T: FloatExt,
  F: Fn(&[T]) -> T,
{
  let d = y.len();
  assert!(d >= 2, "g_kernel_numerical_nd requires d >= 2");
  assert_eq!(lo.len(), d, "lo must have the same dimension as y");
  assert_eq!(hi.len(), d, "hi must have the same dimension as y");
  assert!(n_quad > 0, "n_quad must be positive");

  let half = T::from_f64_fast(0.5);
  let dx = (0..d)
    .map(|k| (hi[k] - lo[k]) / T::from_usize_(n_quad))
    .collect::<Vec<_>>();
  let cell_volume = dx.iter().copied().fold(T::one(), |acc, step| acc * step);
  let tol = T::from_f64_fast(1e-15);
  let mut g = Array2::<T>::zeros((d, d));
  let mut x = vec![T::zero(); d];
  let mut multi_idx = vec![0usize; d];

  loop {
    for k in 0..d {
      x[k] = lo[k] + (T::from_usize_(multi_idx[k]) + half) * dx[k];
    }

    let fval = payoff(&x);
    if fval.abs() >= tol {
      let weighted = fval * cell_volume;
      let diff = (0..d).map(|k| y[k] - x[k]).collect::<Vec<_>>();
      for i in 0..d {
        for j in 0..d {
          g[[i, j]] += weighted * kernel_k_ij_h(&diff, h, i, j);
        }
      }
    }

    let mut exhausted = true;
    for k in (0..d).rev() {
      if multi_idx[k] + 1 < n_quad {
        multi_idx[k] += 1;
        for reset in (k + 1)..d {
          multi_idx[reset] = 0;
        }
        exhausted = false;
        break;
      }
    }
    if exhausted {
      break;
    }
  }

  g
}

#[cfg(test)]
mod tests;

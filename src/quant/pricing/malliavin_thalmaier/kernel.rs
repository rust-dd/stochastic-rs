//! Poisson kernel and `g^h_{i,j}` kernel functions.
//!
//! See Kohatsu-Higa & Yasuda (2008), §2.2 and §3 for the regularisation.

use ndarray::Array2;
use statrs::function::gamma::gamma;

use crate::traits::FloatExt;

/// Surface area of the unit sphere in ℝ^d: `a_d = 2π^{d/2} / Γ(d/2)`.
pub fn sphere_area<T: FloatExt>(d: usize) -> T {
  let dh = d as f64 / 2.0;
  T::from_f64_fast(2.0 * std::f64::consts::PI.powf(dh) / gamma(dh))
}

/// Regularised norm `|x|_h = √(Σ xᵢ² + h)`.
#[inline]
pub fn norm_h<T: FloatExt>(x: &[T], h: T) -> T {
  let s: T = x.iter().map(|xi| *xi * *xi).fold(T::zero(), |a, b| a + b);
  (s + h).sqrt()
}

/// Gradient of the regularised Poisson kernel `∂Q_d^h/∂x_i`.
///
/// - `d = 2`: `∂Q₂^h/∂xᵢ = (1/a₂) · xᵢ / |x|_h²`
/// - `d ≥ 3`: `∂Q_d^h/∂xᵢ = (d−2)/a_d · xᵢ / |x|_h^d`
pub fn grad_poisson_reg<T: FloatExt>(x: &[T], h: T) -> Vec<T> {
  let d = x.len();
  let ad: T = sphere_area(d);
  let r = norm_h(x, h);
  let rd = r.powi(d as i32);
  let factor = if d >= 3 {
    T::from_usize_(d - 2) / (ad * rd)
  } else {
    T::one() / (ad * rd)
  };
  x.iter().map(|&xi| xi * factor).collect()
}

/// Closed-form `g_{i,j}` for the 2-asset digital put `f(x) = 1(x₁≤K₁)·1(x₂≤K₂)`.
///
/// From Kohatsu-Higa & Yasuda (2008), eq. (6.3). Valid as `h → 0`.
pub fn g_digital_put_2d<T: FloatExt>(y: [T; 2], k: [T; 2]) -> [[T; 2]; 2] {
  let a2_inv = T::one() / (T::from_f64_fast(2.0) * T::from_f64_fast(std::f64::consts::PI));
  let nudge = T::from_f64_fast(1e-10);

  let y1 = y[0];
  let y2 = y[1];
  let k1 = k[0];
  let k2 = k[1];
  let y1mk1 = if (y1 - k1).abs() < nudge {
    nudge
  } else {
    y1 - k1
  };
  let y2mk2 = if (y2 - k2).abs() < nudge {
    nudge
  } else {
    y2 - k2
  };
  let y1s = if y1.abs() < nudge { nudge } else { y1 };
  let y2s = if y2.abs() < nudge { nudge } else { y2 };

  let g11 = a2_inv
    * ((y2s / y1s).atan() - (y2mk2 / y1s).atan() - (y2s / y1mk1).atan() + (y2mk2 / y1mk1).atan());

  let num = (y1 * y1 + y2 * y2 + nudge) * (y1mk1 * y1mk1 + y2mk2 * y2mk2 + nudge);
  let den = (y1mk1 * y1mk1 + y2 * y2 + nudge) * (y1 * y1 + y2mk2 * y2mk2 + nudge);
  let g21 = (a2_inv / T::from_f64_fast(2.0)) * (num / den).ln();

  [[g11, g21], [g21, -g11]]
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
  let dx0 = (hi[0] - lo[0]) / T::from_usize_(n_quad);
  let dx1 = (hi[1] - lo[1]) / T::from_usize_(n_quad);
  let area = dx0 * dx1;
  let ad: T = sphere_area(2);
  let half = T::from_f64_fast(0.5);
  let two = T::from_f64_fast(2.0);

  let eps = T::from_f64_fast(1e-4) * (y[0].abs() + y[1].abs() + T::one());

  let shifts: [[T; 2]; 5] = [
    [T::zero(), T::zero()],
    [eps, T::zero()],
    [-eps, T::zero()],
    [T::zero(), eps],
    [T::zero(), -eps],
  ];
  let mut phi = [[T::zero(); 2]; 5];

  for q0 in 0..n_quad {
    let x0 = lo[0] + (T::from_usize_(q0) + half) * dx0;
    for q1 in 0..n_quad {
      let x1 = lo[1] + (T::from_usize_(q1) + half) * dx1;
      let fval = payoff(&[x0, x1]);
      if fval.abs() < T::from_f64_fast(1e-15) {
        continue;
      }
      for (s, shift) in shifts.iter().enumerate() {
        let dy = [y[0] + shift[0] - x0, y[1] + shift[1] - x1];
        let r = norm_h(&dy, h);
        let rd = r * r; // d=2
        let inv = fval * area / (ad * rd);
        phi[s][0] = phi[s][0] + dy[0] * inv;
        phi[s][1] = phi[s][1] + dy[1] * inv;
      }
    }
  }

  let inv_2eps = T::one() / (two * eps);
  let mut g = Array2::<T>::zeros((2, 2));
  g[[0, 0]] = (phi[1][0] - phi[2][0]) * inv_2eps;
  g[[0, 1]] = (phi[3][0] - phi[4][0]) * inv_2eps;
  g[[1, 0]] = (phi[1][1] - phi[2][1]) * inv_2eps;
  g[[1, 1]] = (phi[3][1] - phi[4][1]) * inv_2eps;
  g
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn sphere_area_known_values() {
    let a2: f64 = sphere_area(2);
    assert!((a2 - 2.0 * std::f64::consts::PI).abs() < 1e-10);
    let a3: f64 = sphere_area(3);
    assert!((a3 - 4.0 * std::f64::consts::PI).abs() < 1e-10);
  }

  #[test]
  fn g_digital_put_2d_finite() {
    let g = g_digital_put_2d([100.0_f64, 100.0], [100.0, 100.0]);
    for i in 0..2 {
      for j in 0..2 {
        assert!(g[i][j].is_finite(), "g[{i}][{j}] = {} not finite", g[i][j]);
      }
    }
  }

  #[test]
  fn poisson_kernel_grad_decays() {
    let g1 = grad_poisson_reg(&[1.0, 1.0], 0.01);
    let g2 = grad_poisson_reg(&[10.0, 10.0], 0.01);
    let n1: f64 = g1.iter().map(|x| x * x).sum::<f64>().sqrt();
    let n2: f64 = g2.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(n1 > n2, "|∇Q(1)| = {n1} should > |∇Q(10)| = {n2}");
  }
}

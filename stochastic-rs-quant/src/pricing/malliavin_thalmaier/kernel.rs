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

/// Regularised second derivative kernel
/// `K^h_{i,j}(x) = ∂²Q_d^h(x) / (∂x_i ∂x_j)`.
pub fn kernel_k_ij_h<T: FloatExt>(x: &[T], h: T, i: usize, j: usize) -> T {
  let d = x.len();
  assert!(i < d && j < d, "kernel indices out of bounds");
  let ad: T = sphere_area(d);
  let r = norm_h(x, h);
  let rd = r.powi(d as i32);
  let factor = if d >= 3 {
    T::from_usize_(d - 2) / ad
  } else {
    T::one() / ad
  };
  let delta = if i == j { T::one() } else { T::zero() };
  factor * (delta / rd - T::from_usize_(d) * x[i] * x[j] / r.powi(d as i32 + 2))
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
        phi[s][0] += dy[0] * inv;
        phi[s][1] += dy[1] * inv;
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
  let dx: Vec<T> = (0..d)
    .map(|k| (hi[k] - lo[k]) / T::from_usize_(n_quad))
    .collect();
  let cell_volume = dx.iter().copied().fold(T::one(), |acc, step| acc * step);
  let eps = T::from_f64_fast(1e-4)
    * (y.iter().map(|yi| yi.abs()).fold(T::zero(), |a, b| a + b) / T::from_usize_(d) + T::one());
  let tol = T::from_f64_fast(1e-15);

  let mut shifts = Vec::with_capacity(1 + 2 * d);
  shifts.push(vec![T::zero(); d]);
  for j in 0..d {
    let mut up = vec![T::zero(); d];
    up[j] = eps;
    shifts.push(up);
    let mut dn = vec![T::zero(); d];
    dn[j] = -eps;
    shifts.push(dn);
  }

  let mut phi = vec![vec![T::zero(); d]; shifts.len()];
  let mut x = vec![T::zero(); d];
  let mut multi_idx = vec![0usize; d];

  loop {
    for k in 0..d {
      x[k] = lo[k] + (T::from_usize_(multi_idx[k]) + half) * dx[k];
    }

    let fval = payoff(&x);
    if fval.abs() >= tol {
      let weighted = fval * cell_volume;
      for (shift_id, shift) in shifts.iter().enumerate() {
        let diff: Vec<T> = (0..d).map(|k| y[k] + shift[k] - x[k]).collect();
        let grad = grad_poisson_reg(&diff, h);
        for i in 0..d {
          phi[shift_id][i] += weighted * grad[i];
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

  let inv_2eps = T::one() / (eps + eps);
  let mut g = Array2::<T>::zeros((d, d));
  for j in 0..d {
    let up = 1 + 2 * j;
    let dn = up + 1;
    for i in 0..d {
      g[[i, j]] = (phi[up][i] - phi[dn][i]) * inv_2eps;
    }
  }
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

  #[test]
  fn poisson_kernel_grad_matches_3d_newton_potential() {
    let x = [1.0_f64, 2.0, 2.0];
    let grad = grad_poisson_reg(&x, 0.0);
    let r = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
    let factor = 1.0 / (4.0 * std::f64::consts::PI * r.powi(3));

    for i in 0..3 {
      let expected = x[i] * factor;
      assert!(
        (grad[i] - expected).abs() < 1e-12,
        "grad[{i}] = {}, expected {expected}",
        grad[i]
      );
    }
  }

  #[test]
  fn g_kernel_numerical_nd_matches_2d_specialisation() {
    let y = [0.7_f64, 0.4];
    let lo = [0.0_f64, 0.0];
    let hi = [1.5_f64, 1.5];
    let h = 0.05_f64;
    let payoff = |x: &[f64]| if x[0] <= 1.0 && x[1] <= 0.8 { 1.0 } else { 0.0 };

    let g_2d = g_kernel_numerical_2d(&y, &payoff, h, &lo, &hi, 32);
    let g_nd = g_kernel_numerical_nd(&y, &payoff, h, &lo, &hi, 32);

    for i in 0..2 {
      for j in 0..2 {
        assert!(
          (g_2d[[i, j]] - g_nd[[i, j]]).abs() < 5e-3,
          "mismatch at ({i},{j}): {} vs {}",
          g_2d[[i, j]],
          g_nd[[i, j]]
        );
      }
    }
  }

  #[test]
  fn kernel_trace_identity_holds() {
    let z = [2.0_f64, 3.0, 1.5];
    let h = 0.01;
    let d = z.len();

    let trace: f64 = (0..d).map(|i| kernel_k_ij_h(&z, h, i, i)).sum();

    let ad: f64 = 2.0 * std::f64::consts::PI.powf(d as f64 / 2.0) / gamma(d as f64 / 2.0);
    let factor = (d - 2) as f64 / ad;
    let r_h = (z.iter().map(|x| x * x).sum::<f64>() + h).sqrt();
    let z_sq: f64 = z.iter().map(|x| x * x).sum();
    let expected =
      factor * (d as f64 / r_h.powi(d as i32) - d as f64 * z_sq / r_h.powi(d as i32 + 2));

    assert!(
      (trace - expected).abs() < 1e-10,
      "trace={trace}, expected={expected}"
    );
  }

  #[test]
  fn kernel_is_symmetric_in_indices() {
    let z = [1.0_f64, 2.0, 3.0, 4.0];
    let h = 0.01;

    for i in 0..z.len() {
      for j in (i + 1)..z.len() {
        let kij = kernel_k_ij_h(&z, h, i, j);
        let kji = kernel_k_ij_h(&z, h, j, i);
        assert!(
          (kij - kji).abs() < 1e-14,
          "K[{i},{j}]={kij} != K[{j},{i}]={kji}"
        );
      }
    }
  }
}

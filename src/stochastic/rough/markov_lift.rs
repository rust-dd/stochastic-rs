//! # Markov-lift Volterra SDE stepper
//!
//! $$
//! \begin{aligned} X_{n+1} &= X_0 + \frac{\delta t^{H+1/2} f(X_n)}{\Gamma(H+3/2)} + \frac{\sum_{l=1}^{N'} w_l\, e^{-x_l \delta t}\,(H_l^{(n)} + J_l^{(n)})}{\Gamma(H+1/2)} + \frac{\delta t^{H-1/2} g(X_n)\,\delta W_n}{\Gamma(H+1/2)} \\ H_l^{(n+1)} &= \tfrac{f(X_n)}{x_l}\bigl(1 - e^{-x_l \delta t}\bigr) + e^{-x_l \delta t}\, H_l^{(n)} \\ J_l^{(n+1)} &= e^{-x_l \delta t}\bigl(g(X_n)\,\delta W_n + J_l^{(n)}\bigr) \end{aligned}
//! $$
//!
//! A single-path simulator for Volterra-type SDEs driven by a Riemann–Liouville
//! kernel. Per step:
//!
//! - $\sum_l w_l e_l (H_l + J_l)$ with $w_l e_l$ pre-merged and accumulated
//!   with `wide::f64x4::mul_add` (FMA when available)
//! - The $H$/$J$ state update uses a pre-computed $(1-e_l)/x_l$ factor and
//!   fuses the multiply-add into a single SIMD line per factor
//!
//! Load/store goes through unsafe pointer casts to avoid the element-wise
//! scatter-gather overhead of `f64x4::from([a, b, c, d])`. Reference:
//! Bilokon & Wong (2026), p. 16 of J. Appl. Probab. 2026.
use ndarray::Array1;
use wide::f32x8;
use wide::f64x4;

use super::kernel::RlKernel;
use crate::traits::FloatExt;

/// Per-scalar SIMD kernel for the inner factor loop. Implemented for `f64`
/// with `f64x4` (4-wide) and `f32` with `f32x8` (8-wide).
pub trait RoughSimd: FloatExt {
  /// $\sum_l (w_l e_l)\,(H_l + J_l)$ using a pre-merged `we[l] = w[l] * e[l]`.
  fn history_sum_fused(we: &[Self], h_state: &[Self], j_state: &[Self]) -> Self;

  /// In-place, fused SIMD update of the $H_l$ and $J_l$ state vectors using
  /// pre-computed $\mathrm{omx}_l = (1 - e_l)/x_l$:
  /// - $H_l \leftarrow f\,\mathrm{omx}_l + e_l\,H_l$
  /// - $J_l \leftarrow e_l\,(g\,\delta W + J_l)$
  fn update_state_fused(
    h_state: &mut [Self],
    j_state: &mut [Self],
    exp_neg: &[Self],
    omx: &[Self],
    f_prev: Self,
    g_dw: Self,
  );
}

impl RoughSimd for f64 {
  #[inline]
  fn history_sum_fused(we: &[f64], h_state: &[f64], j_state: &[f64]) -> f64 {
    let n = we.len();
    let chunks = n / 4;
    let mut acc = f64x4::splat(0.0);
    unsafe {
      for i in 0..chunks {
        let base = 4 * i;
        let we_v = load_f64x4(we, base);
        let h_v = load_f64x4(h_state, base);
        let j_v = load_f64x4(j_state, base);
        acc = we_v.mul_add(h_v + j_v, acc);
      }
    }
    let mut sum = acc.reduce_add();
    for i in (chunks * 4)..n {
      sum += we[i] * (h_state[i] + j_state[i]);
    }
    sum
  }

  #[inline]
  fn update_state_fused(
    h_state: &mut [f64],
    j_state: &mut [f64],
    exp_neg: &[f64],
    omx: &[f64],
    f_prev: f64,
    g_dw: f64,
  ) {
    let n = h_state.len();
    let chunks = n / 4;
    let f_v = f64x4::splat(f_prev);
    let gdw_v = f64x4::splat(g_dw);
    unsafe {
      for i in 0..chunks {
        let base = 4 * i;
        let e = load_f64x4(exp_neg, base);
        let o = load_f64x4(omx, base);
        let h = load_f64x4(h_state, base);
        let j = load_f64x4(j_state, base);
        let h_new = e.mul_add(h, f_v * o);
        let j_new = e * (gdw_v + j);
        store_f64x4(h_state, base, h_new);
        store_f64x4(j_state, base, j_new);
      }
    }
    for i in (chunks * 4)..n {
      h_state[i] = f_prev * omx[i] + exp_neg[i] * h_state[i];
      j_state[i] = exp_neg[i] * (g_dw + j_state[i]);
    }
  }
}

impl RoughSimd for f32 {
  #[inline]
  fn history_sum_fused(we: &[f32], h_state: &[f32], j_state: &[f32]) -> f32 {
    let n = we.len();
    let chunks = n / 8;
    let mut acc = f32x8::splat(0.0);
    unsafe {
      for i in 0..chunks {
        let base = 8 * i;
        let we_v = load_f32x8(we, base);
        let h_v = load_f32x8(h_state, base);
        let j_v = load_f32x8(j_state, base);
        acc = we_v.mul_add(h_v + j_v, acc);
      }
    }
    let mut sum = acc.reduce_add();
    for i in (chunks * 8)..n {
      sum += we[i] * (h_state[i] + j_state[i]);
    }
    sum
  }

  #[inline]
  fn update_state_fused(
    h_state: &mut [f32],
    j_state: &mut [f32],
    exp_neg: &[f32],
    omx: &[f32],
    f_prev: f32,
    g_dw: f32,
  ) {
    let n = h_state.len();
    let chunks = n / 8;
    let f_v = f32x8::splat(f_prev);
    let gdw_v = f32x8::splat(g_dw);
    unsafe {
      for i in 0..chunks {
        let base = 8 * i;
        let e = load_f32x8(exp_neg, base);
        let o = load_f32x8(omx, base);
        let h = load_f32x8(h_state, base);
        let j = load_f32x8(j_state, base);
        let h_new = e.mul_add(h, f_v * o);
        let j_new = e * (gdw_v + j);
        store_f32x8(h_state, base, h_new);
        store_f32x8(j_state, base, j_new);
      }
    }
    for i in (chunks * 8)..n {
      h_state[i] = f_prev * omx[i] + exp_neg[i] * h_state[i];
      j_state[i] = exp_neg[i] * (g_dw + j_state[i]);
    }
  }
}

#[inline(always)]
unsafe fn load_f64x4(src: &[f64], base: usize) -> f64x4 {
  debug_assert!(base + 4 <= src.len());
  let ptr = unsafe { src.as_ptr().add(base) as *const [f64; 4] };
  f64x4::from(unsafe { *ptr })
}

#[inline(always)]
unsafe fn store_f64x4(dst: &mut [f64], base: usize, v: f64x4) {
  debug_assert!(base + 4 <= dst.len());
  let ptr = unsafe { dst.as_mut_ptr().add(base) as *mut [f64; 4] };
  unsafe { *ptr = v.to_array() };
}

#[inline(always)]
unsafe fn load_f32x8(src: &[f32], base: usize) -> f32x8 {
  debug_assert!(base + 8 <= src.len());
  let ptr = unsafe { src.as_ptr().add(base) as *const [f32; 8] };
  f32x8::from(unsafe { *ptr })
}

#[inline(always)]
unsafe fn store_f32x8(dst: &mut [f32], base: usize, v: f32x8) {
  debug_assert!(base + 8 <= dst.len());
  let ptr = unsafe { dst.as_mut_ptr().add(base) as *mut [f32; 8] };
  unsafe { *ptr = v.to_array() };
}

/// Single-path Markov-lift stepper for $f,g$-driven RL-Volterra SDEs.
#[derive(Debug, Clone)]
pub struct MarkovLift<T: FloatExt> {
  /// Kernel approximation (nodes + scaled weights).
  pub kernel: RlKernel<T>,
  /// Time-step size $\delta t$.
  pub dt: T,
  /// $e^{-x_l \delta t}$ for each node.
  exp_neg_x_dt: Array1<T>,
  /// Pre-merged $w_l\,e^{-x_l \delta t}$ (constant across steps).
  we: Array1<T>,
  /// Pre-computed $(1 - e^{-x_l \delta t})/x_l$ for the $H_l$ update.
  one_minus_e_over_x: Array1<T>,
  /// $\delta t^{H+1/2}$ (boundary drift coefficient numerator).
  dt_pow_h_plus_half: T,
  /// $\delta t^{H-1/2}$ (boundary diffusion coefficient numerator).
  dt_pow_h_minus_half: T,
  /// $\Gamma(H + 3/2) = (H+1/2)\,\Gamma(H+1/2)$.
  gamma_h_plus_three_half: T,
}

impl<T: FloatExt + RoughSimd> MarkovLift<T> {
  /// Build a stepper for the given kernel and step size $\delta t > 0$.
  #[must_use]
  pub fn new(kernel: RlKernel<T>, dt: T) -> Self {
    assert!(dt > T::zero(), "dt must be positive");

    let h = kernel.hurst;
    let half = T::from_f64_fast(0.5);
    let h_plus_half = h + half;

    let n_prime = kernel.degree();
    let mut exp_neg_x_dt = Array1::<T>::zeros(n_prime);
    let mut we = Array1::<T>::zeros(n_prime);
    let mut one_minus_e_over_x = Array1::<T>::zeros(n_prime);
    for i in 0..n_prime {
      let e = (-kernel.nodes[i] * dt).exp();
      exp_neg_x_dt[i] = e;
      we[i] = kernel.weights[i] * e;
      one_minus_e_over_x[i] = (T::one() - e) / kernel.nodes[i];
    }

    Self {
      dt_pow_h_plus_half: dt.powf(h_plus_half),
      dt_pow_h_minus_half: dt.powf(h - half),
      gamma_h_plus_three_half: kernel.gamma_h_half * h_plus_half,
      exp_neg_x_dt,
      we,
      one_minus_e_over_x,
      kernel,
      dt,
    }
  }

  /// Integrate a single path. `dw` carries Brownian increments on the same
  /// grid as the output (length $n{-}1$). Returns the full path of length
  /// $n = \text{dw.len()} + 1$.
  pub fn simulate<F, G>(&self, x0: T, f: F, g: G, dw: &[T]) -> Array1<T>
  where
    F: Fn(T) -> T,
    G: Fn(T) -> T,
  {
    let n = dw.len() + 1;
    let n_prime = self.kernel.degree();

    let mut path = Array1::<T>::zeros(n);
    path[0] = x0;

    let mut h_state = vec![T::zero(); n_prime];
    let mut j_state = vec![T::zero(); n_prime];

    let inv_gamma_h_half = T::one() / self.kernel.gamma_h_half;
    let inv_gamma_h_three_half = T::one() / self.gamma_h_plus_three_half;

    let we = self.we.as_slice().expect("we must be contiguous");
    let exp_neg = self
      .exp_neg_x_dt
      .as_slice()
      .expect("exp_neg must be contiguous");
    let omx = self
      .one_minus_e_over_x
      .as_slice()
      .expect("omx must be contiguous");

    for step in 0..n - 1 {
      let x_prev = path[step];
      let f_prev = f(x_prev);
      let g_prev = g(x_prev);
      let dw_n = dw[step];
      let g_dw = g_prev * dw_n;

      let history = T::history_sum_fused(we, &h_state, &j_state);

      path[step + 1] = x0
        + self.dt_pow_h_plus_half * f_prev * inv_gamma_h_three_half
        + history * inv_gamma_h_half
        + self.dt_pow_h_minus_half * g_prev * dw_n * inv_gamma_h_half;

      T::update_state_fused(&mut h_state, &mut j_state, exp_neg, omx, f_prev, g_dw);
    }

    path
  }
}

#[cfg(test)]
mod tests {
  use super::super::kernel::RlKernel;
  use super::MarkovLift;

  #[test]
  fn trivial_drift_zero_diffusion_stays_at_x0() {
    let kernel = RlKernel::<f64>::new(0.15, 30);
    let dt = 0.01_f64;
    let step = MarkovLift::new(kernel, dt);
    let dw = vec![0.0_f64; 50];
    let path = step.simulate(0.42, |_| 0.0, |_| 0.0, &dw);
    for v in path.iter() {
      assert!((*v - 0.42).abs() < 1e-12);
    }
  }

  /// With $g \equiv 0$ and constant drift $f = c$ the RL-integral is
  /// $c\, t^{H+1/2}/\Gamma(H+3/2)$.
  #[test]
  fn constant_drift_matches_mittag_leffler_linear_case() {
    let hurst = 0.3_f64;
    let c = 1.5_f64;
    let n = 201;
    let total_t = 1.0_f64;
    let dt = total_t / (n as f64 - 1.0);
    let kernel = RlKernel::<f64>::new(hurst, 40);
    let step = MarkovLift::new(kernel, dt);
    let dw = vec![0.0_f64; n - 1];

    let path = step.simulate(0.0, |_| c, |_| 0.0, &dw);

    let exponent = hurst + 0.5;
    let gamma_h_three_half = statrs::function::gamma::gamma(hurst + 1.5);
    for i in 1..n {
      let t = dt * i as f64;
      let truth = c * t.powf(exponent) / gamma_h_three_half;
      let rel = (path[i] - truth).abs() / truth.abs().max(1e-9);
      assert!(
        rel < 2e-2,
        "i={i} t={t} got={} truth={truth} rel={rel}",
        path[i]
      );
    }
  }

  #[test]
  fn f32_path_is_finite() {
    let kernel = RlKernel::<f32>::new(0.25_f32, 32);
    let dt = 0.005_f32;
    let step = MarkovLift::new(kernel, dt);
    let dw: Vec<f32> = (0..100).map(|i| ((i as f32 * 0.1).sin()) * 0.05).collect();
    let path = step.simulate(0.3_f32, |x| 0.5 * (1.0 - x), |_| 0.2, &dw);
    assert!(path.iter().all(|v| v.is_finite()));
  }
}

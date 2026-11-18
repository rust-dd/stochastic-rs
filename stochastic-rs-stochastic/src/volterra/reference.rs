//! The exact-kernel `O(n²)` scheme: the permanent oracle of `VolterraSde` and `VolterraLift`.
//!
//! $$
//! X_{t_i} = X_0 + \sum_{j \le i} K(t_i - t_{j-1})
//! \bigl[b(t_{j-1}, X_{j-1})\,\Delta t + \sigma(t_{j-1}, X_{j-1})\,\Delta W_j\bigr]
//! $$
//!
//! Also `Volterra`'s production sampler for a kernel the lift cannot represent, such as `H ≥ 1/2`.
//!
//! Reference: Li, Huang & Hu (2020), arXiv:2004.04916, rate $\min\{1-\alpha,\,\tfrac12-\beta\}$
use ndarray::Array1;

use crate::traits::FloatExt;

/// Direct $O(n^2)$ discretisation of the stochastic Volterra equation on $t_i = i\,\delta t$ from
/// $n-1$ increments `dw`. `kernel` is any callable $K(\tau)$, called at $\tau \ge \delta t$ only.
#[must_use]
pub fn reference_path<T: FloatExt>(
  kernel: impl Fn(T) -> T,
  drift: impl Fn(T, T) -> T,
  diffusion: impl Fn(T, T) -> T,
  x0: T,
  dt: T,
  dw: &[T],
) -> Array1<T> {
  let n = dw.len() + 1;
  let mut path = Array1::<T>::zeros(n);
  path[0] = x0;
  if n == 1 {
    return path;
  }

  // increments[k] = b(t_k, X_k)*dt + sigma(t_k, X_k)*dw[k] is driven by step k alone, so it is
  // computed once, when X_k becomes available, for every later step that reads it.
  let mut increments = vec![T::zero(); n - 1];
  increments[0] = drift(T::zero(), x0) * dt + diffusion(T::zero(), x0) * dw[0];

  for i in 1..n {
    let t_i = T::from_usize_(i) * dt;
    let mut acc = T::zero();
    for (k, &inc) in increments.iter().enumerate().take(i) {
      let tau = t_i - T::from_usize_(k) * dt;
      acc += kernel(tau) * inc;
    }
    path[i] = x0 + acc;

    if i < n - 1 {
      let t_i_val = t_i;
      increments[i] = drift(t_i_val, path[i]) * dt + diffusion(t_i_val, path[i]) * dw[i];
    }
  }

  path
}

#[cfg(test)]
mod tests {
  use super::reference_path;

  fn zero_2d(_t: f64, _x: f64) -> f64 {
    0.0
  }

  fn one_2d(_t: f64, _x: f64) -> f64 {
    1.0
  }

  #[test]
  fn trivial_drift_zero_diffusion_stays_at_x0() {
    let dw = vec![0.0_f64; 10];
    let path = reference_path(|tau: f64| (-tau).exp(), zero_2d, zero_2d, 0.37, 0.1, &dw);
    for v in path.iter() {
      assert!((*v - 0.37).abs() < 1e-12);
    }
  }

  /// $K(\tau)=e^{-\beta\tau}$, constant drift, no diffusion give $X_t = f\,(1-e^{-\beta t})/\beta$.
  /// The left-endpoint sum is off by about $\beta\delta t/2$, 1.75e-3 here; the bound is twice it.
  #[test]
  fn exponential_kernel_constant_drift_matches_closed_form() {
    let beta = 0.7_f64;
    let f_const = 0.9_f64;
    let n = 401;
    let total_t = 2.0_f64;
    let dt = total_t / (n as f64 - 1.0);
    let dw = vec![0.0_f64; n - 1];
    let tol = beta * dt;

    let path = reference_path(
      |tau: f64| (-beta * tau).exp(),
      |_t, _x| f_const,
      zero_2d,
      0.0,
      dt,
      &dw,
    );

    for i in 0..n {
      let t = dt * i as f64;
      let truth = f_const * (1.0 - (-beta * t).exp()) / beta;
      let rel = (path[i] - truth).abs() / truth.abs().max(1e-12);
      assert!(
        rel < tol,
        "i={i} t={t} got={} truth={truth} rel={rel} tol={tol}",
        path[i]
      );
    }
  }

  /// `b` and `sigma` see the previous grid point's state $(t_{j-1}, X_{j-1})$: an asymmetric,
  /// state-dependent diffusion desynchronises from the hand-computed reference otherwise.
  #[test]
  fn coefficients_see_causal_state_not_future_state() {
    let dw = vec![0.05_f64; 5];
    let dt = 0.2_f64;
    let path = reference_path(
      |tau: f64| (-tau).exp(),
      zero_2d,
      |_t, x: f64| 1.0 + x,
      1.0,
      dt,
      &dw,
    );
    // Hand-unrolled for i=1: only k=0 contributes, with sigma evaluated at
    // (t_0, X_0) = (0, 1.0), never at the still-unknown X_1.
    let expected_1 = 1.0 + (-dt).exp() * (1.0 + 1.0) * dw[0];
    assert!((path[1] - expected_1).abs() < 1e-12);
  }

  #[test]
  fn one_2d_is_used_as_unit_diffusion_in_gaussian_case() {
    let dw = vec![0.1_f64, -0.2, 0.05];
    let dt = 0.25_f64;
    let path = reference_path(|tau: f64| tau, zero_2d, one_2d, 0.0, dt, &dw);
    // Pure Wiener-type convolution with K(tau)=tau: path[i] = sum_k (t_i -
    // t_k) * dw[k], hand-checked for i=2.
    let t2 = 2.0 * dt;
    let expected_2 = (t2 - 0.0) * dw[0] + (t2 - dt) * dw[1];
    assert!((path[2] - expected_2).abs() < 1e-12);
  }
}

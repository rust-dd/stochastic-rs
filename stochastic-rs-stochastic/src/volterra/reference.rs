//! # Direct $O(n^2)$ Volterra convolution — the lift's cross-implementation oracle
//!
//! $$
//! X_{t_i} = X_0 + \sum_{j \le i} K(t_i - t_{j-1})\bigl[b(t_{j-1}, X_{j-1})\,\Delta t + \sigma(t_{j-1}, X_{j-1})\,\Delta W_j\bigr]
//! $$
//!
//! [`reference_path`] discretises the stochastic Volterra equation by
//! evaluating the kernel *exactly* (whatever closure the caller supplies —
//! typically a [`VolterraKernel::evaluate`](super::kernel::VolterraKernel::evaluate)
//! call, never the exponential-sum fit) at every $(i,j)$ pair, at $O(n^2)$
//! cost. This is deliberately **not** scaffolding: it is the permanent
//! cross-implementation oracle for [`VolterraSde`](super::sve::VolterraSde) /
//! [`VolterraLift`](super::lift::VolterraLift) — this crate's own validation
//! audit found cross-implementation comparisons to be its highest-yield test
//! class, and a fast $O(nN')$ path with no independent oracle is exactly how
//! a subtle exponential-sum-fit or Markov-state-recursion bug would survive
//! undetected. It also backs [`Volterra`](crate::process::volterra::Volterra)'s
//! fallback for kernels [`VolterraLift`](super::lift::VolterraLift) cannot
//! represent (e.g. Hurst $H \ge 1/2$, where [`RlKernel`](crate::rough::kernel::RlKernel)
//! does not apply) — so it is load-bearing production code, not test-only.
//!
//! The kernel-evaluation loop is genuinely $O(n^2)$: the weight
//! $K(t_i-t_{j-1})$ depends on both $i$ and $j$, so it cannot be hoisted.
//! `b`/`sigma`, however, depend only on $j$ (through $X_{j-1}$, already
//! realised by the time step $i>j$ is reached), so each is evaluated
//! exactly once per grid point — $O(n)$ total — rather than recomputed
//! inside the $O(n^2)$ loop; this is an implementation optimisation, not a
//! different discretisation, since a literal per-$(i,j)$ re-evaluation of
//! `b`/`sigma` would recompute the identical deterministic value every time.
//!
//! **On convergence.** For weakly singular kernels this explicit,
//! exact-kernel scheme sits in the class Li, Huang & Hu (arXiv:2004.04916,
//! 2020) analyse: the strong rate is $\min\{1-\alpha,\,\tfrac12-\beta\}$,
//! not the usual $\tfrac12$ — this module makes no independent claim beyond
//! that citation (no rate-sweep test lives here; the module exists to be an
//! oracle at fixed $n$, not to characterise convergence order).
use ndarray::Array1;

use crate::traits::FloatExt;

/// Direct $O(n^2)$ discretisation of $X_t = X_0 + \int_0^t K(t-s)\,b(s,X_s)\,ds
/// + \int_0^t K(t-s)\,\sigma(s,X_s)\,dW_s$ on the grid $t_i = i\,\delta t$.
///
/// `kernel` is any $K(\tau) \to T$ callable — pass
/// [`VolterraKernel::evaluate`](super::kernel::VolterraKernel::evaluate) to
/// validate a [`VolterraKernel`](super::kernel::VolterraKernel) implementor
/// against its own exponential-sum fit, or a closed-form kernel with no
/// [`VolterraKernel`](super::kernel::VolterraKernel) impl at all (this
/// function has no trait bound on the kernel beyond callability, precisely
/// so both uses share one implementation). `dw` holds $n-1$ Brownian
/// increments, the same grid convention [`VolterraLift::simulate`](super::lift::VolterraLift::simulate)
/// uses.
///
/// # Panics
/// - if `kernel(tau)` is called at `tau <= 0` (never happens here: every
///   lag $t_i - t_{j-1}$ with $j \le i$ satisfies $\tau \ge \delta t > 0$)
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

  // increments[k] = b(t_k, X_k)*dt + sigma(t_k, X_k)*dw[k], the term driven
  // by step k alone — independent of which later step i >= k+1 consumes it,
  // so each is computed exactly once, right when X_k becomes available.
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

  /// With a pure exponential kernel $K(\tau)=e^{-\beta\tau}$, constant drift
  /// `f`, and zero diffusion, $X_t = f\,(1-e^{-\beta t})/\beta$ in closed
  /// form. Unlike [`VolterraLift`](super::lift::VolterraLift) — which
  /// integrates $K$ *exactly* over each sub-interval via
  /// [`VolterraKernel::integral_from_zero`](super::kernel::VolterraKernel::integral_from_zero)
  /// — this function evaluates $K$ once per sub-interval, at its left
  /// endpoint $t_{j-1}$ (the brief's own formula), a first-order (left
  /// Riemann sum) discretisation of the same integral. So agreement here is
  /// bounded by that scheme's own $O(\delta t)$ local truncation, not
  /// machine precision: for this kernel the sum telescopes to
  /// $\delta t\cdot\beta\,e^{-\beta\delta t}/(1-e^{-\beta\delta t})$ relative
  /// to the exact $(1-e^{-\beta t})/\beta$, which a direct expansion in
  /// $\delta t$ gives $1-\tfrac{\beta\delta t}{2}+O(\delta t^2)$ — a
  /// relative error of $\approx\beta\delta t/2$, confirmed numerically flat
  /// across every $i$ (not just small ones) at $\approx 1.75\text{e-}3$ for
  /// $\beta=0.7,\ \delta t=0.005$; `2\times` that headroom is the bound
  /// below.
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

  /// `b`/`sigma` must be evaluated at the *previous* grid point's own state
  /// $(t_{j-1}, X_{j-1})$, not the step being computed — proven by an
  /// asymmetric, state-dependent diffusion that would desynchronise from a
  /// hand-computed reference if the wrong state were fed in.
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

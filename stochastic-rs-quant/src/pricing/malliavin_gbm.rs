//! # Malliavin Gbm
//!
//! $$
//! \Delta=\mathbb E\!\left[e^{-rT}\Phi(S_T)\,\frac{W_T}{S_0\sigma T}\right]
//! $$
//!
use ndarray::Array1;
use ndarray::Array2;
use ndarray::s;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::diffusion::gbm::Gbm;

use crate::traits::ModelPricer;
use crate::traits::ProcessExt;

fn laplace_pdf(x: f64, l: f64) -> f64 {
  if l <= 0.0 {
    return 0.0;
  }

  (-(x.abs()) / l).exp() / (2.0 * l)
}

fn laplace_cdf(x: f64, l: f64) -> f64 {
  if l <= 0.0 {
    return if x < 0.0 { 0.0 } else { 1.0 };
  }

  0.5 * (1.0 + x.signum() * (1.0 - (-(x.abs()) / l).exp()))
}

/// Vanilla call/put pricer using Gbm paths and a Malliavin-based conditional estimator.
///
/// The idea:
/// - Simulate Gbm paths S_t on [0, T] using the existing Gbm module.
/// - Reconstruct the Brownian paths W_t from S_t.
/// - Use the Malliavin weight (coef) to estimate the conditional call price
///   C(t, S_t^{(i)}) for each path i.
/// - Then use the tower property to get the time-0 call price:
///   C(0) = E[ e^{-r t} C(t, S_t) ]
/// - Put price is recovered from put-call parity.
///
/// The struct holds **model and method state only** — the volatility, the
/// Monte Carlo path/step counts, and the intermediate time `t_eval` at
/// which the conditional price is estimated. Spot, strike, rate, dividend
/// yield and maturity are the pricing *query* and travel as arguments to
/// [`ModelPricer::price_call`].
///
/// # Panics
/// Every pricing method asserts `0 < t_eval < tau`. `t_eval` is an
/// *absolute* time, not a fraction of the maturity, so one instance can
/// only price maturities strictly longer than its `t_eval` — a real
/// constraint on the strike/maturity grids this model can cover, and the
/// reason `t_eval` is a construction parameter rather than a query one.
/// The struct is `Clone`, so a shorter maturity means a second instance.
#[derive(Debug, Clone, Copy)]
pub struct GbmMalliavinPricer {
  /// Volatility σ
  pub v: f64,
  /// Number of Monte Carlo paths (M)
  pub n_paths: usize,
  /// Number of time steps (N)
  pub n_steps: usize,
  /// Intermediate time t where the Malliavin conditional price C(t, S_t) is estimated
  /// (0 < t_eval < tau)
  pub t_eval: f64,
}

impl ModelPricer for GbmMalliavinPricer {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.call_put(s, k, r, q, tau).0
  }

  /// Overrides the trait's vanilla-parity default for two reasons. The
  /// arithmetic is the same put-call parity — this is a Gbm model, so the
  /// carry factor really is $e^{-q\tau}$ — but the default would run a
  /// **second, independent** Monte Carlo for its `price_call` term, so the
  /// returned put would be derived from a different sample than any call
  /// the caller had already obtained. It also drops the `max(0)` floor the
  /// pre-query `calculate_call_put` applied. This routes through
  /// [`call_put`](Self::call_put), which does one simulation and floors
  /// both legs.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.call_put(s, k, r, q, tau).1
  }
}

impl GbmMalliavinPricer {
  pub const fn new(v: f64, n_paths: usize, n_steps: usize, t_eval: f64) -> Self {
    Self {
      v,
      n_paths,
      n_steps,
      t_eval,
    }
  }

  /// Call and put price from a single Monte Carlo simulation, using the
  /// plain (unlocalized) Malliavin conditional estimator.
  pub fn call_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> (f64, f64) {
    let (_s_t, c_t) = self.conditional_call_malliavin(s, k, r, q, tau);
    self.call_put_from_conditional(s, k, r, q, tau, &c_t)
  }

  fn call_put_from_conditional(
    &self,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
    c_t: &Array1<f64>,
  ) -> (f64, f64) {
    let t_eval = self.t_eval;
    assert!(t_eval > 0.0 && t_eval < tau, "t_eval must be in (0, T)");

    // Time-0 call price via tower property:
    //   C(0) = E[ e^{-r t} C(t, S_t) ]
    // Here we approximate E[ C(t, S_t) ] with the Monte Carlo average,
    // but ignore non-finite pathwise estimates and enforce non-negativity.
    let disc_0t = (-r * t_eval).exp();

    let mut sum = 0.0_f64;
    let mut count = 0_usize;
    for &v in c_t.iter() {
      if v.is_finite() {
        sum += v;
        count += 1;
      }
    }
    let avg_c_t = if count > 0 { sum / count as f64 } else { 0.0 };

    let mut call_0 = disc_0t * avg_c_t;
    if call_0 < 0.0 {
      call_0 = 0.0;
    }

    // Put price from put–call parity with dividend yield q:
    //   C - P = S_0 e^{-qT} - K e^{-rT}
    let df_rt = (-r * tau).exp();
    let df_qt = (-q * tau).exp();
    let mut put_0 = call_0 + k * df_rt - s * df_qt;
    if put_0 < 0.0 {
      put_0 = 0.0;
    }

    (call_0, put_0)
  }

  /// Call/put prices using the localized Malliavin estimator.
  pub fn call_put_localized(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> (f64, f64) {
    let (_s_t, c_t) = self.conditional_call_malliavin_localized(s, k, r, q, tau);
    self.call_put_from_conditional(s, k, r, q, tau, &c_t)
  }

  /// Simulate Gbm paths S_t using the existing Gbm<f64> module.
  ///
  /// Returns:
  ///   S: shape (M, N), with S[i, k] = S^{(i)}_{t_k}
  fn sample_paths(&self, s: f64, r: f64, q: f64, tau: f64) -> Array2<f64> {
    let mu = r - q;

    // Construct a Gbm process with Euler discretization on [0, T].
    let gbm = Gbm::new(mu, self.v, self.n_steps, Some(s), Some(tau), Unseeded);

    let m = self.n_paths;
    let n = self.n_steps;

    let mut paths = Array2::<f64>::zeros((m, n));
    for i in 0..m {
      let path = gbm.sample();
      paths.slice_mut(s![i, ..]).assign(&path);
    }

    paths
  }
}

impl GbmMalliavinPricer {
  /// Malliavin-based conditional CALL prices C^M(t, S_t^{(i)}) for each path i.
  ///
  /// Returns:
  ///   - S_t: shape (M,)
  ///   - C^M(t, S_t^{(i)}): shape (M,)
  ///
  /// The estimator is:
  ///   C^M(t, x) ≈ e^{-r(T-t)} * [ Σ_j φ(S_T^{(j)}) H(S_t^{(j)} - x) coef^{(j)} ] /
  ///                                   [ Σ_j H(S_t^{(j)} - x) coef^{(j)} ],
  /// where H is the Heaviside step function and coef^{(j)} is the Malliavin weight.
  ///
  /// An entry is `NaN` where that path's denominator collapses to ~0 — no
  /// sampled path lands above `S_t^{(i)}` with appreciable weight, so the
  /// conditional expectation has nothing to average and is undefined at that
  /// point rather than zero. Raise `n_paths` if entries come back `NaN`.
  pub fn conditional_call_malliavin(
    &self,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
  ) -> (Array1<f64>, Array1<f64>) {
    let t_eval = self.t_eval;
    assert!(t_eval > 0.0 && t_eval < tau, "t_eval must be in (0, T)");

    let mu = r - q;
    let dt = tau / (self.n_steps - 1) as f64;

    // Simulate Gbm paths
    let paths = self.sample_paths(s, r, q, tau);
    let m = paths.nrows();
    let n = paths.ncols();

    // Reconstruct Brownian paths W from S:
    //
    // Gbm Euler step:
    //   S_k = S_{k-1} + μ S_{k-1} dt + σ S_{k-1} dW_{k-1}
    // => dW_{k-1} = (S_k - S_{k-1} - μ S_{k-1} dt) / (σ S_{k-1})
    //
    // Then W_k = Σ_{j=0}^{k-1} dW_j.
    let w_paths = self.reconstruct_brownian(&paths, mu, dt);

    // Discrete index corresponding to t_eval
    let k_t = ((t_eval / dt).round() as usize).min(n - 1);

    let s_t = paths.slice(s![.., k_t]).to_owned();
    let s_final = paths.slice(s![.., n - 1]).to_owned();
    let w_t = w_paths.slice(s![.., k_t]).to_owned();
    let w_final = w_paths.slice(s![.., n - 1]).to_owned();

    // Payoff φ(S_T) = (S_T - K)^+
    let payoff: Array1<f64> = s_final.iter().map(|&x_t| (x_t - k).max(0.0)).collect();

    // Malliavin-weight (coef) for Gbm:
    //
    //   coef^{(i)} = ((T W_t^{(i)} - t_eval W_T^{(i)}) / (T - t_eval) + σ t_eval) / S_t^{(i)}
    //
    // This is the weight that appears after the Malliavin integration by parts
    // when rewriting the conditional expectation with a Dirac delta as a ratio of expectations.
    let mut coef = Array1::<f64>::zeros(m);
    for i in 0..m {
      let st = s_t[i];
      if st.abs() < 1e-14 {
        coef[i] = 0.0;
      } else {
        let num = (tau * w_t[i] - t_eval * w_final[i]) / (tau - t_eval) + self.v * t_eval;
        coef[i] = num / st;
      }
    }

    let discount_tt = (-r * (tau - t_eval)).exp();
    let mut c_hat = Array1::<f64>::zeros(m);

    // For each path i, estimate C^M(t, S_t^{(i)}).
    for i in 0..m {
      let x = s_t[i];
      let mut num = 0.0;
      let mut den = 0.0;

      for j in 0..m {
        // Heaviside H(S_t^{(j)} - S_t^{(i)})
        if s_t[j] >= x {
          let w = coef[j];
          num += payoff[j] * w;
          den += w;
        }
      }

      c_hat[i] = if den.abs() > 1e-14 {
        discount_tt * (num / den)
      } else {
        f64::NAN
      };
    }

    (s_t, c_hat)
  }

  /// Malliavin-based conditional CALL prices C^M(t, S_t^{(i)}) with localization
  /// based on a Laplace kernel, following the reference implementation.
  ///
  /// Returns:
  ///   - S_t: shape (M,)
  ///   - Localized C^M(t, S_t^{(i)}): shape (M,)
  ///
  /// An entry is `NaN` where that path's localised denominator collapses to
  /// ~0, for the same reason as
  /// [`conditional_call_malliavin`](Self::conditional_call_malliavin) — the
  /// Laplace kernel put no appreciable mass near `S_t^{(i)}`.
  pub fn conditional_call_malliavin_localized(
    &self,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
  ) -> (Array1<f64>, Array1<f64>) {
    let t_eval = self.t_eval;
    assert!(t_eval > 0.0 && t_eval < tau, "t_eval must be in (0, T)");

    let mu = r - q;
    let dt = tau / (self.n_steps - 1) as f64;

    // Simulate Gbm paths
    let paths = self.sample_paths(s, r, q, tau);
    let m = paths.nrows();
    let n = paths.ncols();

    // Reconstruct Brownian paths W from S
    let w_paths = self.reconstruct_brownian(&paths, mu, dt);

    // Discrete index corresponding to t_eval
    let k_t = ((t_eval / dt).round() as usize).min(n - 1);

    let s_t = paths.slice(s![.., k_t]).to_owned();
    let s_final = paths.slice(s![.., n - 1]).to_owned();
    let w_t = w_paths.slice(s![.., k_t]).to_owned();
    let w_final = w_paths.slice(s![.., n - 1]).to_owned();

    // Payoff φ(S_T) = (S_T - K)^+
    let payoff: Array1<f64> = s_final.iter().map(|&x_t| (x_t - k).max(0.0)).collect();

    // Localized Malliavin quantities
    //
    // DeltaW = (T * W_t - t_eval * W_T) + (T - t_eval) * t_eval * σ
    let mut delta_w = Array1::<f64>::zeros(m);
    for i in 0..m {
      delta_w[i] = tau * w_t[i] - t_eval * w_final[i] + (tau - t_eval) * t_eval * self.v;
    }

    // den_loc = payoff^2
    let den_loc: Array1<f64> = payoff.iter().map(|&po| po * po).collect();

    // t2 = DeltaW / (t_eval * (T - t_eval) * σ * S_t)
    let mut t2 = Array1::<f64>::zeros(m);
    let denom_scalar = t_eval * (tau - t_eval) * self.v;
    for i in 0..m {
      let st = s_t[i];
      if st.abs() > 1e-14 && denom_scalar.abs() > 1e-14 {
        t2[i] = delta_w[i] / (denom_scalar * st);
      } else {
        t2[i] = 0.0;
      }
    }

    // num_loc = den_loc * t2^2
    let mut num_loc = Array1::<f64>::zeros(m);
    for i in 0..m {
      num_loc[i] = den_loc[i] * t2[i] * t2[i];
    }

    let mean_den_loc = den_loc.mean().unwrap_or(0.0);
    let mean_num_loc = num_loc.mean().unwrap_or(0.0);
    let lf = if mean_den_loc > 0.0 && mean_num_loc >= 0.0 {
      (mean_num_loc / mean_den_loc).sqrt()
    } else {
      0.0
    };

    // l1 scale
    let sigma2 = self.v * self.v;
    let h = mu - 0.5 * sigma2;
    let t = t_eval;
    let numer_l = tau + sigma2 * t * (tau - t);
    let denom_l = sigma2 * t * (tau - t);
    let l1 = if denom_l > 0.0 && s > 0.0 {
      (1.0 / s) * (-(h + sigma2) * t).exp() * (numer_l / denom_l).sqrt()
    } else if lf > 0.0 {
      lf
    } else {
      1e-8
    };

    let discount_tt = (-r * (tau - t_eval)).exp();
    let mut c_hat_loc = Array1::<f64>::zeros(m);

    // For each path i, estimate localized C^M(t, S_t^{(i)})
    for i in 0..m {
      let x = s_t[i];
      let mut num_i = 0.0;
      let mut den_i = 0.0;

      for j in 0..m {
        let diff = s_t[j] - x;
        let heav = if diff >= 0.0 { 1.0 } else { 0.0 };

        let lap_df_l1 = laplace_pdf(diff, l1);
        let lap_cdf_l1 = laplace_cdf(diff, l1);
        let pp_loc_1 = lap_df_l1 + (heav - lap_cdf_l1) * t2[j];

        let lap_df_lf = laplace_pdf(diff, lf);
        let lap_cdf_lf = laplace_cdf(diff, lf);
        let pp_loc_f = lap_df_lf + (heav - lap_cdf_lf) * t2[j];

        den_i += pp_loc_1;
        num_i += payoff[j] * pp_loc_f;
      }

      c_hat_loc[i] = if den_i.abs() > 1e-14 {
        discount_tt * (num_i / den_i)
      } else {
        f64::NAN
      };
    }

    (s_t, c_hat_loc)
  }

  /// Reconstruct the driving Brownian paths from simulated Gbm paths by
  /// inverting the Euler step.
  fn reconstruct_brownian(&self, paths: &Array2<f64>, mu: f64, dt: f64) -> Array2<f64> {
    let m = paths.nrows();
    let n = paths.ncols();
    let mut w_paths = Array2::<f64>::zeros((m, n));
    for i in 0..m {
      let mut w = 0.0;
      w_paths[[i, 0]] = w;

      for step in 1..n {
        let s_prev = paths[[i, step - 1]];
        let s_curr = paths[[i, step]];

        let dw = if s_prev.abs() > 1e-14 {
          (s_curr - s_prev - mu * s_prev * dt) / (self.v * s_prev)
        } else {
          0.0
        };

        w += dw;
        w_paths[[i, step]] = w;
      }
    }
    w_paths
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  const S: f64 = 100.0;
  const K: f64 = 99.99;
  const R: f64 = 0.1;
  const TAU: f64 = 1.0;

  fn pricer() -> GbmMalliavinPricer {
    GbmMalliavinPricer::new(0.1, 2_000, 128, 0.5)
  }

  #[test]
  fn malliavin_pricer_returns_finite_non_negative_prices() {
    let p = pricer();
    let (call, put) = p.call_put(S, K, R, 0.0, TAU);

    // Basic sanity checks: finite and non-negative prices
    assert!(call.is_finite(), "Call price should be finite");
    assert!(put.is_finite(), "Put price should be finite");
    assert!(call >= 0.0, "Call price should be non-negative");
    assert!(put >= 0.0, "Put price should be non-negative");

    // Very loose upper bounds to avoid flakiness due to Monte Carlo noise
    assert!(call < S * 2.0, "Call price is unreasonably large");
    assert!(put < K * 2.0, "Put price is unreasonably large");
  }

  #[test]
  fn malliavin_pricer_localized_returns_finite_non_negative_prices() {
    let p = pricer();
    let (call, put) = p.call_put_localized(S, K, R, 0.0, TAU);

    // Basic sanity checks: finite and non-negative prices
    assert!(call.is_finite(), "Localized call price should be finite");
    assert!(put.is_finite(), "Localized put price should be finite");
    assert!(call >= 0.0, "Localized call price should be non-negative");
    assert!(put >= 0.0, "Localized put price should be non-negative");

    // Very loose upper bounds to avoid flakiness due to Monte Carlo noise
    assert!(call < S * 2.0, "Localized call price is unreasonably large");
    assert!(put < K * 2.0, "Localized put price is unreasonably large");
  }

  /// `price_call` and `price_put` are the two legs of `call_put`, so the
  /// put must satisfy put-call parity **against the call from its own
  /// simulation**, not against some other run's. This is what the trait's
  /// `price_put` default would break: it would run a second, independent
  /// Monte Carlo for its `price_call` term.
  #[test]
  fn malliavin_put_is_parity_against_its_own_call() {
    let (call, put) = pricer().call_put(S, K, R, 0.02, TAU);
    let parity = call - S * (-0.02_f64 * TAU).exp() + K * (-R * TAU).exp();
    assert!(
      (put - parity.max(0.0)).abs() < 1e-12,
      "put {put} must be the floored parity value {parity} of its own call"
    );
  }

  /// The `max(0)` floor the pre-query `calculate_call_put` applied — which
  /// the trait's `price_put` default does **not** have — still guards the
  /// output. It cannot be pinned by a single deterministic value: the floor
  /// fires exactly when the Monte Carlo call estimate lands below its
  /// parity-implied lower bound, which is an estimator-noise event on an
  /// `Unseeded` RNG. What *is* deterministic is the guarantee, so that is
  /// what this asserts, across the strike range where a negative
  /// parity value is reachable.
  #[test]
  fn malliavin_put_is_never_negative() {
    let p = pricer();
    for &k in &[1.0, 50.0, 99.99, 150.0] {
      let put = p.price_put(S, k, R, 0.0, TAU);
      assert!(put >= 0.0, "put at K={k} must be floored, got {put}");
    }
  }

  /// `t_eval` is an absolute time, so a maturity shorter than it is not a
  /// query this instance can price — and it says so rather than returning
  /// a number.
  #[test]
  #[should_panic(expected = "t_eval must be in (0, T)")]
  fn malliavin_rejects_a_maturity_shorter_than_t_eval() {
    let _ = pricer().price_call(S, K, R, 0.0, 0.25);
  }

  /// The capability the reshape exists for: one model, many query points.
  /// Monte Carlo noise makes a strict monotonicity assertion flaky, so this
  /// pins the weaker property that every point is priced and finite, plus
  /// the no-arbitrage upper bound.
  #[test]
  fn malliavin_one_model_prices_a_grid() {
    let model = GbmMalliavinPricer::new(0.2, 400, 64, 0.25);
    for &tau in &[0.5, 1.0] {
      for &k in &[90.0, 100.0, 110.0] {
        let c = model.price_call(S, k, 0.03, 0.01, tau);
        assert!(
          c.is_finite() && (0.0..=S).contains(&c),
          "call {c} out of bounds"
        );
      }
    }
  }
}

//! European prices on Gbm paths from the Malliavin conditional expectation, plain or localised.
//!
//! $$
//! \Delta=\mathbb E\!\left[e^{-rT}\Phi(S_T)\,\frac{W_T}{S_0\sigma T}\right]
//! $$

use ndarray::Array1;
use ndarray::Array2;
use ndarray::s;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::diffusion::gbm::Gbm;

use crate::CallPut;
use crate::traits::ModelPricer;
use crate::traits::ProcessExt;
use crate::traits::RealExt;
use crate::traits::VanillaEuropeanCall;

/// Laplace kernel of bandwidth `l`; `l = 0` is the Dirac limit (zero density, right-continuous step
/// cdf), so the localisation weight collapses exactly. Panics on `l < 0`.
fn laplace_pdf(x: f64, l: f64) -> f64 {
  assert!(
    l >= 0.0,
    "laplace bandwidth l must be non-negative (got {l})"
  );
  if l == 0.0 {
    return 0.0;
  }

  (-(x.abs()) / l).exp() / (2.0 * l)
}

/// Cdf of [`laplace_pdf`] — see its doc for the `l == 0` Dirac limit and the
/// negative-bandwidth panic, which this shares.
fn laplace_cdf(x: f64, l: f64) -> f64 {
  assert!(
    l >= 0.0,
    "laplace bandwidth l must be non-negative (got {l})"
  );
  if l == 0.0 {
    return if x < 0.0 { 0.0 } else { 1.0 };
  }

  0.5 * (1.0 + x.signum() * (1.0 - (-(x.abs()) / l).exp()))
}

/// A query no path is simulated for: the payoff at a zero horizon, and no price for a non-finite
/// query, a spot or strike that is not positive, or a horizon at or inside `t_eval`.
enum Degenerate {
  OffDomain,
  ZeroHorizon,
}

impl Degenerate {
  fn of(s: f64, k: f64, r: f64, q: f64, tau: f64, t_eval: f64) -> Option<Self> {
    let finite = [s, k, r, q, tau].iter().all(|x| x.is_finite());
    if !finite || tau < 0.0 {
      Some(Self::OffDomain)
    } else if tau == 0.0 {
      Some(Self::ZeroHorizon)
    } else if s <= 0.0 || k <= 0.0 || tau <= t_eval {
      Some(Self::OffDomain)
    } else {
      None
    }
  }

  fn price_call_put(&self, s: f64, k: f64) -> CallPut {
    let (call, put) = match self {
      Self::OffDomain => (f64::NAN, f64::NAN),
      Self::ZeroHorizon => ((s - k).max_or_nan(0.0), (k - s).max_or_nan(0.0)),
    };
    CallPut { call, put }
  }

  /// The spot at `t_eval` and the conditional call there, on each of `m` paths.
  fn conditional(&self, s: f64, k: f64, m: usize) -> (Array1<f64>, Array1<f64>) {
    match self {
      Self::OffDomain => (
        Array1::from_elem(m, f64::NAN),
        Array1::from_elem(m, f64::NAN),
      ),
      Self::ZeroHorizon => (
        Array1::from_elem(m, s),
        Array1::from_elem(m, (s - k).max_or_nan(0.0)),
      ),
    }
  }
}

/// Malliavin conditional-expectation pricer on [`Gbm`] paths, `C(0) = E[e^{−rt} C(t, S_t)]`; a
/// `Deterministic` seed makes the price a function of the query, NaN for `0 < tau <= t_eval`.
#[derive(Debug, Clone, Copy)]
pub struct GbmMalliavinPricer<S: SeedExt = Unseeded> {
  /// Volatility σ
  sigma: f64,
  /// Number of Monte Carlo paths (M)
  n_paths: usize,
  /// Number of time steps (N)
  n_steps: usize,
  /// Intermediate time t where the Malliavin conditional price C(t, S_t) is estimated
  /// (0 < t_eval < tau)
  t_eval: f64,
  /// Seed strategy handed to the underlying [`Gbm`].
  pub seed: S,
}

impl<S: SeedExt> ModelPricer for GbmMalliavinPricer<S> {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.price_call_put(s, k, r, q, tau).call
  }

  /// The pair's put, floored at zero and from the call's own simulation, which the parity default
  /// would not give.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.price_call_put(s, k, r, q, tau).put
  }
}

/// European vanilla call under Gbm's carry `b = r − q`; a surface inherits the Monte Carlo error,
/// and each maturity of its grid must exceed [`t_eval`](GbmMalliavinPricer::t_eval()).
impl<S: SeedExt> VanillaEuropeanCall for GbmMalliavinPricer<S> {}

impl GbmMalliavinPricer {
  /// Panics unless `sigma` is finite and `>= 0` (the Malliavin weight carries it linearly),
  /// `n_paths >= 1`, `n_steps >= 2` and `t_eval` is finite and `> 0`; `t_eval < tau` is per query.
  pub fn new(sigma: f64, n_paths: usize, n_steps: usize, t_eval: f64) -> Self {
    assert!(
      sigma.is_finite() && sigma >= 0.0,
      "sigma must satisfy `sigma.is_finite() && sigma >= 0.0`, got sigma = {sigma:?}"
    );
    assert!(
      n_paths >= 1,
      "n_paths must satisfy `n_paths >= 1`, got n_paths = {n_paths:?}"
    );
    assert!(
      n_steps >= 2,
      "n_steps must satisfy `n_steps >= 2`, got n_steps = {n_steps:?}"
    );
    assert!(
      t_eval.is_finite() && t_eval > 0.0,
      "t_eval must satisfy `t_eval.is_finite() && t_eval > 0.0`, got t_eval = {t_eval:?}"
    );
    Self {
      sigma,
      n_paths,
      n_steps,
      t_eval,
      seed: Unseeded,
    }
  }
}

impl<S: SeedExt> GbmMalliavinPricer<S> {
  pub fn with_seed<S2: SeedExt>(self, seed: S2) -> GbmMalliavinPricer<S2> {
    GbmMalliavinPricer {
      sigma: self.sigma,
      n_paths: self.n_paths,
      n_steps: self.n_steps,
      t_eval: self.t_eval,
      seed,
    }
  }

  pub fn sigma(&self) -> f64 {
    self.sigma
  }

  /// The time `t` of `C(t, S_t)`, which each queried `tau` must exceed.
  pub fn t_eval(&self) -> f64 {
    self.t_eval
  }

  /// Call and put from one simulation of the plain Malliavin estimator: the payoff at `tau = 0`;
  /// NaN for a non-finite query, a spot or strike that is not positive, or `tau <= t_eval`.
  pub fn price_call_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> CallPut {
    if let Some(limit) = Degenerate::of(s, k, r, q, tau, self.t_eval) {
      return limit.price_call_put(s, k);
    }
    let (_s_t, c_t) = self.conditional_call_malliavin(s, k, r, q, tau);
    self.price_call_put_from_conditional(s, k, r, q, tau, &c_t)
  }

  fn price_call_put_from_conditional(
    &self,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
    c_t: &Array1<f64>,
  ) -> CallPut {
    let t_eval = self.t_eval;

    // The tower property `C(0) = E[e^{−rt} C(t, S_t)]`, over the finite pathwise estimates and
    // floored at zero.
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

    CallPut {
      call: call_0,
      put: put_0,
    }
  }

  /// Call and put from one simulation of the localized estimator, at
  /// [`price_call_put`](Self::price_call_put)'s limits.
  pub fn price_call_put_localized(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> CallPut {
    if let Some(limit) = Degenerate::of(s, k, r, q, tau, self.t_eval) {
      return limit.price_call_put(s, k);
    }
    let (_s_t, c_t) = self.conditional_call_malliavin_localized(s, k, r, q, tau);
    self.price_call_put_from_conditional(s, k, r, q, tau, &c_t)
  }

  /// The `(M, N)` Gbm paths `S[i, k] = S^{(i)}_{t_k}`, each its own stream of a clone of the seed,
  /// so they are independent, replay from the constructor argument and leave the seed alone.
  fn sample_paths(&self, s: f64, r: f64, q: f64, tau: f64) -> Array2<f64> {
    let mu = r - q;

    // Construct a Gbm process with Euler discretization on [0, T].
    let gbm = Gbm::new(
      mu,
      self.sigma,
      self.n_steps,
      s,
      Some(tau),
      self.seed.clone(),
    );

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

impl<S: SeedExt> GbmMalliavinPricer<S> {
  /// Each path's `S_t` and Malliavin conditional call (`H` the Heaviside step, `w` the weight); NaN
  /// where the denominator collapses, as no path lands above `S_t^{(i)}` (raise `n_paths`):
  /// $$
  /// C^M(t,x)\approx e^{-r(T-t)}\,
  /// \frac{\sum_j \varphi(S_T^{(j)})\,H(S_t^{(j)}-x)\,w^{(j)}}{\sum_j H(S_t^{(j)}-x)\,w^{(j)}}
  /// $$
  pub fn conditional_call_malliavin(
    &self,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
  ) -> (Array1<f64>, Array1<f64>) {
    if let Some(limit) = Degenerate::of(s, k, r, q, tau, self.t_eval) {
      return limit.conditional(s, k, self.n_paths);
    }
    let t_eval = self.t_eval;

    let mu = r - q;
    let dt = tau / (self.n_steps - 1) as f64;

    // Simulate Gbm paths
    let paths = self.sample_paths(s, r, q, tau);
    let m = paths.nrows();
    let n = paths.ncols();

    // The Euler step `S_k = S_{k−1}(1 + μ dt + σ dW_{k−1})` inverted for `dW`, then summed to `W`.
    let w_paths = self.reconstruct_brownian(&paths, mu, dt);

    // Discrete index corresponding to t_eval
    let k_t = ((t_eval / dt).round() as usize).min(n - 1);

    let s_t = paths.slice(s![.., k_t]).to_owned();
    let s_final = paths.slice(s![.., n - 1]).to_owned();
    let w_t = w_paths.slice(s![.., k_t]).to_owned();
    let w_final = w_paths.slice(s![.., n - 1]).to_owned();

    // Payoff φ(S_T) = (S_T - K)^+
    let payoff: Array1<f64> = s_final.iter().map(|&x_t| (x_t - k).max(0.0)).collect();

    // The weight left by integrating the conditional's Dirac delta by parts:
    // `w^{(i)} = ((T W_t^{(i)} − t W_T^{(i)})/(T − t) + σt) / S_t^{(i)}`, `t = t_eval`.
    let mut coef = Array1::<f64>::zeros(m);
    for i in 0..m {
      let st = s_t[i];
      if st.abs() < 1e-14 {
        coef[i] = 0.0;
      } else {
        let num = (tau * w_t[i] - t_eval * w_final[i]) / (tau - t_eval) + self.sigma * t_eval;
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

  /// [`Self::conditional_call_malliavin`] localised by a Laplace kernel, as the reference
  /// implementation does; NaN where the kernel puts no mass near `S_t^{(i)}`.
  pub fn conditional_call_malliavin_localized(
    &self,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
  ) -> (Array1<f64>, Array1<f64>) {
    if let Some(limit) = Degenerate::of(s, k, r, q, tau, self.t_eval) {
      return limit.conditional(s, k, self.n_paths);
    }
    let t_eval = self.t_eval;

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
      delta_w[i] = tau * w_t[i] - t_eval * w_final[i] + (tau - t_eval) * t_eval * self.sigma;
    }

    // den_loc = payoff^2
    let den_loc: Array1<f64> = payoff.iter().map(|&po| po * po).collect();

    // t2 = DeltaW / (t_eval * (T - t_eval) * σ * S_t)
    let mut t2 = Array1::<f64>::zeros(m);
    let denom_scalar = t_eval * (tau - t_eval) * self.sigma;
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
    let sigma2 = self.sigma * self.sigma;
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
          (s_curr - s_prev - mu * s_prev * dt) / (self.sigma * s_prev)
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
#[path = "malliavin_gbm/tests.rs"]
mod tests;

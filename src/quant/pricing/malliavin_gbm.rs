//! # Malliavin GBM
//!
//! $$
//! \Delta=\mathbb E\!\left[e^{-rT}\Phi(S_T)\,\frac{W_T}{S_0\sigma T}\right]
//! $$
//!
use ndarray::s;
use ndarray::Array1;
use ndarray::Array2;

use crate::stochastic::diffusion::gbm::GBM;
use crate::traits::PricerExt;
use crate::traits::ProcessExt;
use crate::traits::TimeExt;

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

/// Vanilla call/put pricer using GBM paths and a Malliavin-based conditional estimator.
///
/// The idea:
/// - Simulate GBM paths S_t on [0, T] using the existing GBM module.
/// - Reconstruct the Brownian paths W_t from S_t.
/// - Use the Malliavin weight (coef) to estimate the conditional call price
///   C(t, S_t^{(i)}) for each path i.
/// - Then use the tower property to get the time-0 call price:
///   C(0) = E[ e^{-r t} C(t, S_t) ]
/// - Put price is recovered from put-call parity.
pub struct GbmMalliavinPricer {
  /// Underlying spot S_0
  pub s: f64,
  /// Volatility σ
  pub v: f64,
  /// Strike K
  pub k: f64,
  /// Risk-free rate r
  pub r: f64,
  /// Dividend yield q
  pub q: Option<f64>,
  /// Time to maturity in years
  pub tau: Option<f64>,
  /// Evaluation date
  pub eval: Option<chrono::NaiveDate>,
  /// Expiration date
  pub expiration: Option<chrono::NaiveDate>,

  /// Number of Monte Carlo paths (M)
  pub n_paths: usize,
  /// Number of time steps (N)
  pub n_steps: usize,
  /// Intermediate time t where the Malliavin conditional price C(t, S_t) is estimated
  /// (0 < t_eval < tau)
  pub t_eval: f64,
}

impl TimeExt for GbmMalliavinPricer {
  fn tau(&self) -> Option<f64> {
    self.tau
  }

  fn eval(&self) -> Option<chrono::NaiveDate> {
    self.eval
  }

  fn expiration(&self) -> Option<chrono::NaiveDate> {
    self.expiration
  }
}

impl PricerExt for GbmMalliavinPricer {
  fn calculate_call_put(&self) -> (f64, f64) {
    let t = self.t_eval;
    let (_s_t, c_t) = self.conditional_call_malliavin(t);

    self.call_put_from_conditional(t, &c_t)
  }

  fn calculate_price(&self) -> f64 {
    self.calculate_call_put().0
  }
}

impl GbmMalliavinPricer {
  fn call_put_from_conditional(&self, t_eval: f64, c_t: &Array1<f64>) -> (f64, f64) {
    // Use maturity in years, consistent with the reference implementation and drift/discounting.
    let T = self.tau().unwrap_or_else(|| self.calculate_tau_in_years());
    assert!(t_eval > 0.0 && t_eval < T, "t_eval must be in (0, T)");

    // Time-0 call price via tower property:
    //   C(0) = E[ e^{-r t} C(t, S_t) ]
    // Here we approximate E[ C(t, S_t) ] with the Monte Carlo average,
    // but ignore non-finite pathwise estimates and enforce non-negativity.
    let disc_0t = (-self.r * t_eval).exp();

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
    let q = self.q.unwrap_or(0.0);
    let df_rT = (-self.r * T).exp();
    let df_qT = (-q * T).exp();
    let mut put_0 = call_0 + self.k * df_rT - self.s * df_qT;
    if put_0 < 0.0 {
      put_0 = 0.0;
    }

    (call_0, put_0)
  }

  /// Call/put prices using the localized Malliavin estimator.
  pub fn calculate_call_put_localized(&self) -> (f64, f64) {
    let t = self.t_eval;
    let (_s_t, c_t) = self.conditional_call_malliavin_localized(t);

    self.call_put_from_conditional(t, &c_t)
  }

  /// Simulate GBM paths S_t using the existing GBM<f64> module.
  ///
  /// Returns:
  ///   S: shape (M, N), with S[i, k] = S^{(i)}_{t_k}
  fn sample_paths(&self) -> Array2<f64> {
    // Time horizon in years for the GBM simulation.
    let T = self.tau().unwrap_or_else(|| self.calculate_tau_in_years());
    let mu = self.r - self.q.unwrap_or(0.0);

    // Construct a GBM process with Euler discretization on [0, T].
    let gbm = GBM::new(mu, self.v, self.n_steps, Some(self.s), Some(T));

    let m = self.n_paths;
    let n = self.n_steps;

    let mut S = Array2::<f64>::zeros((m, n));
    for i in 0..m {
      let path = gbm.sample();
      S.slice_mut(s![i, ..]).assign(&path);
    }

    S
  }

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
  pub fn conditional_call_malliavin(&self, t_eval: f64) -> (Array1<f64>, Array1<f64>) {
    // Work with maturity in years to stay consistent with the reference script.
    let T = self.tau().unwrap_or_else(|| self.calculate_tau_in_years());
    assert!(t_eval > 0.0 && t_eval < T, "t_eval must be in (0, T)");

    let q = self.q.unwrap_or(0.0);
    let mu = self.r - q;
    let dt = T / (self.n_steps - 1) as f64;

    // Simulate GBM paths S
    let S = self.sample_paths();
    let m = S.nrows();
    let n = S.ncols();

    // Reconstruct Brownian paths W from S:
    //
    // GBM Euler step:
    //   S_k = S_{k-1} + μ S_{k-1} dt + σ S_{k-1} dW_{k-1}
    // => dW_{k-1} = (S_k - S_{k-1} - μ S_{k-1} dt) / (σ S_{k-1})
    //
    // Then W_k = Σ_{j=0}^{k-1} dW_j.
    let mut W = Array2::<f64>::zeros((m, n));
    for i in 0..m {
      let mut w = 0.0;
      W[[i, 0]] = w;

      for k in 1..n {
        let s_prev = S[[i, k - 1]];
        let s_curr = S[[i, k]];

        let dW = if s_prev.abs() > 1e-14 {
          (s_curr - s_prev - mu * s_prev * dt) / (self.v * s_prev)
        } else {
          0.0
        };

        w += dW;
        W[[i, k]] = w;
      }
    }

    // Discrete index corresponding to t_eval
    let k_t = ((t_eval / dt).round() as usize).min(n - 1);

    let s_t = S.slice(s![.., k_t]).to_owned();
    let s_T = S.slice(s![.., n - 1]).to_owned();
    let w_t = W.slice(s![.., k_t]).to_owned();
    let w_T = W.slice(s![.., n - 1]).to_owned();

    // Payoff φ(S_T) = (S_T - K)^+
    let payoff: Array1<f64> = s_T.iter().map(|&x_T| (x_T - self.k).max(0.0)).collect();

    // Malliavin-weight (coef) for GBM:
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
        let num = (T * w_t[i] - t_eval * w_T[i]) / (T - t_eval) + self.v * t_eval;
        coef[i] = num / st;
      }
    }

    let discount_tT = (-self.r * (T - t_eval)).exp();
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
        discount_tT * (num / den)
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
  pub fn conditional_call_malliavin_localized(&self, t_eval: f64) -> (Array1<f64>, Array1<f64>) {
    // Work with maturity in years to stay consistent with the reference script.
    let T = self.tau().unwrap_or_else(|| self.calculate_tau_in_years());
    assert!(t_eval > 0.0 && t_eval < T, "t_eval must be in (0, T)");

    let q = self.q.unwrap_or(0.0);
    let mu = self.r - q;
    let dt = T / (self.n_steps - 1) as f64;

    // Simulate GBM paths S
    let S = self.sample_paths();
    let m = S.nrows();
    let n = S.ncols();

    // Reconstruct Brownian paths W from S
    let mut W = Array2::<f64>::zeros((m, n));
    for i in 0..m {
      let mut w = 0.0;
      W[[i, 0]] = w;

      for k in 1..n {
        let s_prev = S[[i, k - 1]];
        let s_curr = S[[i, k]];

        let dW = if s_prev.abs() > 1e-14 {
          (s_curr - s_prev - mu * s_prev * dt) / (self.v * s_prev)
        } else {
          0.0
        };

        w += dW;
        W[[i, k]] = w;
      }
    }

    // Discrete index corresponding to t_eval
    let k_t = ((t_eval / dt).round() as usize).min(n - 1);

    let s_t = S.slice(s![.., k_t]).to_owned();
    let s_T = S.slice(s![.., n - 1]).to_owned();
    let w_t = W.slice(s![.., k_t]).to_owned();
    let w_T = W.slice(s![.., n - 1]).to_owned();

    // Payoff φ(S_T) = (S_T - K)^+
    let payoff: Array1<f64> = s_T.iter().map(|&x_T| (x_T - self.k).max(0.0)).collect();

    // Localized Malliavin quantities
    //
    // DeltaW = (T * W_t - t_eval * W_T) + (T - t_eval) * t_eval * σ
    let mut delta_w = Array1::<f64>::zeros(m);
    for i in 0..m {
      delta_w[i] = T * w_t[i] - t_eval * w_T[i] + (T - t_eval) * t_eval * self.v;
    }

    // den_loc = payoff^2
    let den_loc: Array1<f64> = payoff.iter().map(|&po| po * po).collect();

    // t2 = DeltaW / (t_eval * (T - t_eval) * σ * S_t)
    let mut t2 = Array1::<f64>::zeros(m);
    let denom_scalar = t_eval * (T - t_eval) * self.v;
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
    let numer_l = T + sigma2 * t * (T - t);
    let denom_l = sigma2 * t * (T - t);
    let l1 = if denom_l > 0.0 && self.s > 0.0 {
      (1.0 / self.s) * (-(h + sigma2) * t).exp() * (numer_l / denom_l).sqrt()
    } else if lf > 0.0 {
      lf
    } else {
      1e-8
    };

    let discount_tT = (-self.r * (T - t_eval)).exp();
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
        discount_tT * (num_i / den_i)
      } else {
        f64::NAN
      };
    }

    (s_t, c_hat_loc)
  }
}

#[cfg(test)]
mod tests {
  use chrono::NaiveDate;

  use super::*;

  #[test]
  fn malliavin_pricer_returns_finite_non_negative_prices() {
    let eval = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
    let expiration = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();

    let pricer = GbmMalliavinPricer {
      s: 100.0,
      v: 0.1,
      k: 99.99,
      r: 0.1,
      q: Some(0.0),
      tau: Some(1.0),
      eval: Some(eval),
      expiration: Some(expiration),
      n_paths: 2_000,
      n_steps: 128,
      t_eval: 0.5,
    };

    let (call, put) = pricer.calculate_call_put();
    println!("Call price: {}", call);
    println!("Put price: {}", put);

    // Basic sanity checks: finite and non-negative prices
    assert!(call.is_finite(), "Call price should be finite");
    assert!(put.is_finite(), "Put price should be finite");
    assert!(call >= 0.0, "Call price should be non-negative");
    assert!(put >= 0.0, "Put price should be non-negative");

    // Very loose upper bounds to avoid flakiness due to Monte Carlo noise
    assert!(call < pricer.s * 2.0, "Call price is unreasonably large");
    assert!(put < pricer.k * 2.0, "Put price is unreasonably large");
  }

  #[test]
  fn malliavin_pricer_localized_returns_finite_non_negative_prices() {
    let eval = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
    let expiration = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();

    let pricer = GbmMalliavinPricer {
      s: 100.0,
      v: 0.1,
      k: 99.99,
      r: 0.1,
      q: Some(0.0),
      tau: Some(1.0),
      eval: Some(eval),
      expiration: Some(expiration),
      n_paths: 2_000,
      n_steps: 128,
      t_eval: 0.5,
    };

    let (call, put) = pricer.calculate_call_put_localized();
    println!("Localized call price: {}", call);
    println!("Localized put price: {}", put);

    // Basic sanity checks: finite and non-negative prices
    assert!(call.is_finite(), "Localized call price should be finite");
    assert!(put.is_finite(), "Localized put price should be finite");
    assert!(call >= 0.0, "Localized call price should be non-negative");
    assert!(put >= 0.0, "Localized put price should be non-negative");

    // Very loose upper bounds to avoid flakiness due to Monte Carlo noise
    assert!(
      call < pricer.s * 2.0,
      "Localized call price is unreasonably large"
    );
    assert!(
      put < pricer.k * 2.0,
      "Localized put price is unreasonably large"
    );
  }
}
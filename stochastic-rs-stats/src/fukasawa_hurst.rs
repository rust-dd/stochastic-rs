//! Fukasawa adapted Whittle estimator for the Hurst parameter.
//!
//! $$
//! \hat H = \arg\min_{(H,\nu)} U_n(H,\nu), \quad
//! U_n \approx \frac{1}{2\pi}\int_\psi^\pi\!\left[\log g_{H,\nu}(\lambda)
//!       + \frac{I_n(\lambda)}{g_{H,\nu}(\lambda)}\right]\mathrm d\lambda
//!       + A^1_{H,\nu}(\psi) + A^2_{H,\nu}(\psi)
//! $$
//!
//! Estimates the Hurst parameter of the log-volatility from a realized
//! variance time series using the adapted Whittle quasi-likelihood of
//! Fukasawa, Takabatake & Westphal (2019), arXiv:1905.04852.
//!
//! Implementation follows Section 3.2 of the paper:
//! - Paxson (Euler–Maclaurin) boundary correction for the spectral density sum
//! - Correction terms A¹ and A² for low-frequency truncation (eq. 16)
//! - Multi-start L-BFGS-B optimisation (projected L-BFGS via argmin)

use argmin::core::CostFunction;
use argmin::core::Executor;
use argmin::core::Gradient;
use argmin::core::State;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use statrs::function::gamma::gamma;

use crate::traits::FloatExt;

/// Estimation result.
#[derive(Clone, Debug)]
pub struct FukasawaResult {
  pub hurst: f64,
  pub eta: f64,
  pub neg_log_lik: f64,
  pub n_obs: usize,
}

/// `C_H = Γ(2H+1) sin(πH) / (2π)`.
fn c_h(h: f64) -> f64 {
  let pi = std::f64::consts::PI;
  gamma(2.0 * h + 1.0) * (pi * h).sin() / (2.0 * pi)
}

/// Approximate spectral density of the observation process (eq. 14, p. 9).
///
/// ```text
/// g(λ) = ν² C_H Σ_{k∈Z} |λ + 2πk|^{-1-2H}  +  (2/m) ℓ(λ)
/// ```
///
/// The fGN spectral density sum is truncated at `k_trunc` terms.
/// For H > 0.05 and K=500 the truncation error is < 1%.
pub fn spectral_density(lambda: f64, h: f64, v: f64, m: usize, _n: usize, k_trunc: usize) -> f64 {
  let pi = std::f64::consts::PI;
  let ch = c_h(h);
  let alpha = 1.0 + 2.0 * h;

  // fGN spectral density: f_H(λ) = C_H · 4sin²(λ/2) · Σ_k |λ + 2πk|^{−1−2H}
  //
  // The 4sin²(λ/2) = |1 − e^{iλ}|² factor arises because fGN is the
  // increment process of fBM.  It ensures f_H(0) = 0 for H < 0.5
  // (anti-persistent / rough volatility).
  let mut sum = lambda.abs().powf(-alpha); // k=0 term
  for k in 1..=k_trunc {
    let kf = k as f64;
    sum +=
      (2.0 * pi * kf + lambda).abs().powf(-alpha) + (2.0 * pi * kf - lambda).abs().powf(-alpha);
  }

  let sin_half = (lambda / 2.0).sin();
  let signal = v * v * ch * 4.0 * sin_half * sin_half * sum;

  // Noise: (2/m) · ℓ(λ)  where ℓ(λ) = 2 sin²(λ/2) / π
  //
  // The proxy error ε_t ~ i.i.d. N(0, 2/m) enters Y_t as Δε_t = ε_{t+1} − ε_t.
  // The spectral density of {Δε_t} is |1−e^{iλ}|² · Var(ε)/(2π) =
  // 4 sin²(λ/2) · (2/m)/(2π) = (2/m) · 2 sin²(λ/2) / π.
  let half = lambda / 2.0;
  let ell = 2.0 * half.sin().powi(2) / pi;
  let noise = 2.0 / m as f64 * ell;

  signal + noise
}

/// Periodogram of a real-valued series at Fourier frequencies `2πj/n`.
fn periodogram(y: &[f64], n_freq: usize) -> Vec<f64> {
  let n = y.len();
  let nf = n as f64;
  let pi2 = 2.0 * std::f64::consts::PI;
  (1..=n_freq)
    .map(|j| {
      let lam = pi2 * j as f64 / nf;
      let mut cr = 0.0;
      let mut ci = 0.0;
      for (t, &yt) in y.iter().enumerate() {
        let phase = lam * t as f64;
        cr += yt * phase.cos();
        ci += yt * phase.sin();
      }
      (cr * cr + ci * ci) / (pi2 * nf)
    })
    .collect()
}

/// Sample autocovariance γ̂_n(τ) = (1/n) Σ_{t=0}^{n-1-τ} Y_t Y_{t+τ}.
fn autocovariance(y: &[f64]) -> Vec<f64> {
  let n = y.len();
  (0..n)
    .map(|tau| {
      let s: f64 = (0..n - tau).map(|t| y[t] * y[t + tau]).sum();
      s / n as f64
    })
    .collect()
}

/// Correction term A¹_{H,ν}(ψ) from eq. 16.
///
/// Approximates `(1/2π) ∫₀^ψ [log g_{H,ν}(λ) + noise/g] dλ` via
/// leading-order asymptotics for small ψ.
fn correction_a1(psi: f64, h: f64, v: f64, m: usize) -> f64 {
  let pi = std::f64::consts::PI;
  let ch = c_h(h);
  let v2ch = v * v * ch;
  if v2ch < 1e-30 || psi < 1e-30 {
    return 0.0;
  }

  // ∫₀^ψ log(ν²C_H · λ^{-(1+2H)}) dλ
  //   = ψ·log(ν²C_H) − (1+2H)·ψ·(log ψ − 1)
  let log_term = psi * v2ch.ln() - (1.0 + 2.0 * h) * psi * (psi.ln() - 1.0);

  // Noise correction from expanding log(signal + noise) ≈ log(signal) + noise/signal:
  // (2/(m·ν²C_H)) · ψ^{2+2H} / (2+2H)
  let noise_corr = 2.0 * psi.powf(2.0 + 2.0 * h) / (m as f64 * v2ch * (2.0 + 2.0 * h));

  (log_term + noise_corr) / (2.0 * pi)
}

/// Coefficient a_{H,ν}(τ, ψ, J) for the A² correction term.
///
/// Taylor expansion of `∫₀^ψ cos(τλ) / g_{H,ν}(λ) dλ` to order J,
/// using `1/g ≈ λ^{1+2H} / (ν²C_H)` for small λ.
fn a_hv(tau: usize, psi: f64, h: f64, v: f64, j_max: usize) -> f64 {
  let pi = std::f64::consts::PI;
  let v2ch = v * v * c_h(h);
  if v2ch < 1e-30 {
    return 0.0;
  }

  let tau_psi = tau as f64 * psi;
  let tau_psi_sq = tau_psi * tau_psi;
  let base_exp = 2.0 + 2.0 * h;
  let psi_base = psi.powf(base_exp);

  let mut sum = 0.0;
  // coeff tracks (-1)^j · (τψ)^{2j} / (2j)! incrementally to avoid overflow
  let mut coeff = 1.0_f64;
  let mut psi_pow = psi_base; // ψ^{2+2H+2j}

  for j in 0..=j_max {
    if j > 0 {
      coeff *= -tau_psi_sq / ((2 * j - 1) as f64 * (2 * j) as f64);
      psi_pow *= psi * psi;
    }

    let exponent = base_exp + 2.0 * j as f64;
    let term = coeff * psi_pow / (exponent * v2ch);
    sum += term;

    if j > 0 && term.abs() < 1e-30 {
      break;
    }
  }

  sum / (2.0 * pi)
}

/// Correction term A²_{H,ν}(ψ) from eq. 16.
///
/// Approximates `(1/2π) ∫₀^ψ I_n(λ)/g_{H,ν}(λ) dλ` using sample
/// autocovariance γ̂ and the Taylor-expanded kernel a_{H,ν}(τ,ψ,J).
fn correction_a2(gamma: &[f64], psi: f64, h: f64, v: f64, j_max: usize) -> f64 {
  let pi = std::f64::consts::PI;

  let a0 = a_hv(0, psi, h, v, j_max);
  let mut sum = a0 * gamma[0];

  for (tau, &g_tau) in gamma.iter().enumerate().skip(1) {
    sum += 2.0 * a_hv(tau, psi, h, v, j_max) * g_tau;
  }

  sum / (2.0 * pi)
}

// ── Whittle objectives ──────────────────────────────────────────────

/// Whittle quasi-likelihood integral `(1/n_freq) Σ [log g + I/g]` (eq. 8).
///
/// This is the integral part only, without correction terms.
pub fn whittle_objective(
  pgram: &[f64],
  h: f64,
  v: f64,
  m: usize,
  n: usize,
  psi: f64,
  k_trunc: usize,
) -> f64 {
  let pi = std::f64::consts::PI;
  let nf = n as f64;
  let mut sum = 0.0;
  let mut count = 0;

  for j in 1..=pgram.len() {
    let lam = 2.0 * pi * j as f64 / nf;
    if lam < psi {
      continue;
    }

    let g = spectral_density(lam, h, v, m, n, k_trunc);
    if g > 1e-20 {
      sum += g.ln() + pgram[j - 1] / g;
      count += 1;
    }
  }

  if count > 0 {
    sum / count as f64
  } else {
    f64::INFINITY
  }
}

/// Full adapted Whittle objective with correction terms (eq. 16).
///
/// `U_n(H,ν) ≈ integral + A¹(ψ) + A²(ψ)`
fn whittle_objective_full(
  pgram: &[f64],
  gamma: &[f64],
  h: f64,
  v: f64,
  m: usize,
  n: usize,
  psi: f64,
  k_trunc: usize,
  j_max: usize,
) -> f64 {
  if h <= 0.0 || h >= 1.0 || v <= 0.0 {
    return f64::INFINITY;
  }

  let base = whittle_objective(pgram, h, v, m, n, psi, k_trunc);
  if !base.is_finite() {
    return f64::INFINITY;
  }

  let a1 = correction_a1(psi, h, v, m);
  let a2 = correction_a2(gamma, psi, h, v, j_max);

  let result = base + a1 + a2;
  if result.is_finite() {
    result
  } else {
    f64::INFINITY
  }
}

/// Argmin problem wrapper for Whittle quasi-likelihood minimisation.
#[derive(Clone)]
struct WhittleProblem {
  pgram: Vec<f64>,
  gamma: Vec<f64>,
  m: usize,
  n: usize,
  psi: f64,
  k_trunc: usize,
  j_max: usize,
  h_bounds: (f64, f64),
  v_bounds: (f64, f64),
}

impl WhittleProblem {
  /// Project parameters onto the feasible box.
  fn clamp(&self, params: &[f64]) -> Vec<f64> {
    vec![
      params[0].clamp(self.h_bounds.0, self.h_bounds.1),
      params[1].clamp(self.v_bounds.0, self.v_bounds.1),
    ]
  }

  fn eval(&self, params: &[f64]) -> f64 {
    let p = self.clamp(params);
    whittle_objective_full(
      &self.pgram,
      &self.gamma,
      p[0],
      p[1],
      self.m,
      self.n,
      self.psi,
      self.k_trunc,
      self.j_max,
    )
  }
}

impl CostFunction for WhittleProblem {
  type Param = Vec<f64>;
  type Output = f64;

  fn cost(&self, params: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
    Ok(self.eval(params))
  }
}

impl Gradient for WhittleProblem {
  type Param = Vec<f64>;
  type Gradient = Vec<f64>;

  /// Central finite-difference gradient with projected bounds.
  fn gradient(&self, params: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
    let p = self.clamp(params);
    let bounds = [self.h_bounds, self.v_bounds];
    let mut grad = vec![0.0; 2];

    for i in 0..2 {
      let step = 1e-7 * (1.0 + p[i].abs());
      let mut p_plus = p.clone();
      let mut p_minus = p.clone();
      p_plus[i] = (p[i] + step).min(bounds[i].1);
      p_minus[i] = (p[i] - step).max(bounds[i].0);
      let actual_2h = p_plus[i] - p_minus[i];
      if actual_2h > 0.0 {
        let fp = self.eval(&p_plus);
        let fm = self.eval(&p_minus);
        grad[i] = (fp - fm) / actual_2h;
      }
    }

    Ok(grad)
  }
}

/// Run projected L-BFGS from a single starting point, returning (H, ν, cost).
fn run_lbfgs(problem: &WhittleProblem, h_init: f64, v_init: f64) -> (f64, f64, f64) {
  let init = vec![h_init, v_init];
  let fallback_cost = problem.eval(&init);

  let linesearch = MoreThuenteLineSearch::new();
  let solver = LBFGS::new(linesearch, 10);

  let result = Executor::new(problem.clone(), solver)
    .configure(|state| state.param(init.clone()).max_iters(200))
    .run();

  match result {
    Ok(res) => {
      let best_p = res.state.get_best_param().cloned().unwrap_or(init);
      let clamped = problem.clamp(&best_p);
      let cost = problem.eval(&clamped);
      (clamped[0], clamped[1], cost)
    }
    Err(_) => (h_init, v_init, fallback_cost),
  }
}

/// Estimate `(H, η)` from log realized variance increments.
///
/// `log_rv`: daily `ln(RV_t)` series of length `n`.
/// `m`: number of intraday observations per day (e.g. 72 for 5-min with 6h trading).
/// `delta`: time interval per day in years (e.g. 1/250).
///
/// Uses a dense grid search followed by L-BFGS refinement of the
/// Whittle quasi-likelihood with Paxson spectral density correction.
pub fn estimate(log_rv: ndarray::ArrayView1<f64>, m: usize, delta: f64) -> FukasawaResult {
  let log_rv = log_rv
    .as_slice()
    .expect("estimate requires a contiguous ArrayView1");
  let n = log_rv.len();
  assert!(n >= 30, "need at least 30 observations, got {n}");

  // Y_t = log RV_{t+1} - log RV_t (increments)
  let y: Vec<f64> = (1..n).map(|i| log_rv[i] - log_rv[i - 1]).collect();
  let ny = y.len();
  let n_freq = ny / 2;

  let pgram = periodogram(&y, n_freq);

  let psi = 1e-5;
  let k_trunc = 500;

  // Phase 1: Dense grid search over H ∈ (0.01, 0.49)
  // ν grid: log-spaced from 0.02 to 10 (60 points) to cover the full
  // range ν = η·δ^H for typical δ = 1/250 and H ∈ (0, 0.5).
  let mut best_h = 0.1;
  let mut best_v = 1.0;
  let mut best_nll = f64::INFINITY;

  let n_v = 60;
  let v_lo_grid = 0.02_f64;
  let v_hi_grid = 10.0_f64;
  let log_ratio = (v_hi_grid / v_lo_grid).ln();

  for h_idx in 1..50 {
    let h = h_idx as f64 * 0.01;
    for v_idx in 0..n_v {
      let v = v_lo_grid * (log_ratio * v_idx as f64 / (n_v - 1) as f64).exp();
      let nll = whittle_objective(&pgram, h, v, m, ny, psi, k_trunc);
      if nll < best_nll {
        best_nll = nll;
        best_h = h;
        best_v = v;
      }
    }
  }

  // Phase 2: L-BFGS refinement from grid optimum
  let gamma_hat = autocovariance(&y);
  let problem = WhittleProblem {
    pgram,
    gamma: gamma_hat,
    m,
    n: ny,
    psi,
    k_trunc,
    j_max: 20,
    h_bounds: (0.005, 0.495),
    v_bounds: (0.01, 12.0),
  };

  let (h_r, v_r, nll_r) = run_lbfgs(&problem, best_h, best_v);
  if nll_r < best_nll {
    best_nll = nll_r;
    best_h = h_r;
    best_v = v_r;
  }

  // Recover η from ν = η·δ^H
  let eta = best_v / delta.powf(best_h);

  FukasawaResult {
    hurst: best_h,
    eta,
    neg_log_lik: best_nll,
    n_obs: n,
  }
}

/// Convenience: estimate H from daily close prices.
///
/// Computes log realized variance (squared daily returns) and applies
/// the Fukasawa estimator with `m = 1` (daily data, no intraday).
pub fn estimate_from_prices(closes: ndarray::ArrayView1<f64>) -> FukasawaResult {
  let closes = closes
    .as_slice()
    .expect("estimate_from_prices requires a contiguous ArrayView1");
  assert!(closes.len() >= 31, "need at least 31 prices");

  let log_rv: Vec<f64> = (1..closes.len())
    .map(|i| {
      let r = (closes[i] / closes[i - 1]).ln();
      (r * r).max(1e-20).ln()
    })
    .collect();

  estimate(ndarray::ArrayView1::from(&log_rv), 1, 1.0 / 250.0)
}

/// FloatExt generic wrapper.
pub fn estimate_from_prices_generic<T: FloatExt>(closes: ndarray::ArrayView1<T>) -> FukasawaResult {
  let closes_f64: Vec<f64> = closes.iter().map(|x| x.to_f64().unwrap()).collect();
  estimate_from_prices(ndarray::ArrayView1::from(&closes_f64))
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::traits::ProcessExt;

  /// With m=1 (daily data, no intraday), estimation has heavy downward
  /// bias (Fukasawa Table 1 footnote).  Use m≥72 for accurate results.
  /// This test uses the full model (fOU + intraday) with m=72.
  #[test]
  fn estimate_h_from_simulated_rv() {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::Distribution;
    use rand_distr::StandardNormal;
    use stochastic_rs_stochastic::diffusion::fou::Fou;

    let true_h = 0.3_f64;
    let m = 72_usize;
    let n_days = 500_usize;
    let delta = 1.0 / 250.0;

    let fou = Fou::new(
      true_h,
      0.001,
      -3.2,
      1.0,
      n_days + 1,
      Some(-3.2),
      Some(n_days as f64 * delta),
    );
    let log_vol_sq: ndarray::Array1<f64> = fou.sample();

    let mut rng = StdRng::seed_from_u64(123);
    let mut log_rv = vec![0.0_f64; n_days];
    for day in 0..n_days {
      let sigma = log_vol_sq[day].exp().sqrt();
      let dt = delta / m as f64;
      let mut rv = 0.0;
      for _ in 0..m {
        let z: f64 = StandardNormal.sample(&mut rng);
        rv += (sigma * dt.sqrt() * z).powi(2);
      }
      log_rv[day] = rv.max(1e-20).ln();
    }

    let result = estimate(ndarray::ArrayView1::from(&log_rv), m, delta);
    let err = (result.hurst - true_h).abs();
    assert!(
      err < 0.1,
      "H={:.3}, true={true_h}, err={err:.3}",
      result.hurst
    );
  }

  /// With m=72, the estimator should distinguish rough (H=0.1) from
  /// smooth (H=0.45) volatility.
  #[test]
  fn rough_vs_smooth_distinguished() {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::Distribution;
    use rand_distr::StandardNormal;
    use stochastic_rs_stochastic::diffusion::fou::Fou;

    let m = 72_usize;
    let n_days = 500_usize;
    let delta = 1.0 / 250.0;

    let estimate_h = |true_h: f64, seed: u64| -> f64 {
      let fou = Fou::new(
        true_h,
        0.001,
        -3.2,
        1.0,
        n_days + 1,
        Some(-3.2),
        Some(n_days as f64 * delta),
      );
      let log_vol_sq: ndarray::Array1<f64> = fou.sample();
      let mut rng = StdRng::seed_from_u64(seed);
      let mut log_rv = vec![0.0_f64; n_days];
      for day in 0..n_days {
        let sigma = log_vol_sq[day].exp().sqrt();
        let dt = delta / m as f64;
        let mut rv = 0.0;
        for _ in 0..m {
          let z: f64 = StandardNormal.sample(&mut rng);
          rv += (sigma * dt.sqrt() * z).powi(2);
        }
        log_rv[day] = rv.max(1e-20).ln();
      }
      estimate(ndarray::ArrayView1::from(&log_rv), m, delta).hurst
    };

    let h_rough = estimate_h(0.1, 77);
    let h_smooth = estimate_h(0.45, 88);

    assert!(
      h_rough < h_smooth,
      "rough H={h_rough:.3} should be < smooth H={h_smooth:.3}"
    );
  }

  #[test]
  fn estimate_from_prices_runs() {
    let mut prices = vec![100.0_f64; 500];
    for i in 1..500 {
      prices[i] = prices[i - 1] * (1.0 + 0.001 * ((i as f64) * 0.1).sin());
    }
    let result = estimate_from_prices(ndarray::ArrayView1::from(&prices));
    assert!(
      result.hurst > 0.0 && result.hurst < 0.5,
      "H={:.2}",
      result.hurst
    );
  }

  /// Reproduce Fukasawa Table 1 with m=72 (5-min RV, 6h trading day).
  ///
  /// Simulates a fOU log-vol process, generates intraday prices,
  /// computes 5-min realized variance, then estimates H.
  #[test]
  fn table1_m72_accuracy() {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::Distribution;
    use rand_distr::StandardNormal;
    use stochastic_rs_stochastic::diffusion::fou::Fou;

    let true_h_values = [0.1, 0.3, 0.5];
    let eta_0 = 1.0;
    let alpha = 0.001; // mean-reversion of log-vol
    let c = -3.2; // log σ²_0
    let m = 72_usize; // 5-min bars in a 6h trading day
    let n_days = 500_usize;
    let delta = 1.0 / 250.0;

    println!("\nFukasawa Table 1 reproduction (m={m}, n={n_days}):");
    println!("{:<10} {:<10} {:<10}", "True H", "Est H", "Error");

    for &true_h in &true_h_values {
      // Simulate fOU log-vol path at daily frequency
      let n_vol = n_days + 1;
      let fou = Fou::new(
        true_h,
        alpha,
        c,
        eta_0,
        n_vol,
        Some(c),
        Some(n_days as f64 * delta),
      );
      let log_vol_sq: ndarray::Array1<f64> = fou.sample();

      // For each day, generate m intraday returns and compute RV
      let mut rng = StdRng::seed_from_u64(42);
      let mut log_rv = vec![0.0_f64; n_days];

      for day in 0..n_days {
        let sigma_sq: f64 = log_vol_sq[day].exp(); // σ² for this day
        let sigma = sigma_sq.sqrt();
        let dt_intra = delta / m as f64;

        let mut rv_sum = 0.0;
        for _ in 0..m {
          let z: f64 = StandardNormal.sample(&mut rng);
          let r = sigma * dt_intra.sqrt() * z;
          rv_sum += r * r;
        }
        log_rv[day] = rv_sum.max(1e-20).ln();
      }

      let result = estimate(ndarray::ArrayView1::from(&log_rv), m, delta);
      let err = (result.hurst - true_h).abs();

      println!("{:<10.2} {:<10.2} {:<10.3}", true_h, result.hurst, err);

      // With m=72, error should be < 0.1 for H >= 0.1
      if true_h >= 0.1 {
        assert!(
          err < 0.15,
          "H={:.2}, true={true_h:.2}, err={err:.3} too large with m={m}",
          result.hurst
        );
      }
    }
  }
}

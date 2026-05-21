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
//! - Paxson (Euler-Maclaurin) boundary correction for the spectral density sum
//! - Correction terms `A¹` and `A²` for low-frequency truncation (eq. 16)
//! - Multi-start L-BFGS-B optimisation (projected L-BFGS via argmin)
//!
//! Wrapped under the [`super::HurstEstimator`] trait as [`Whittle`];
//! free functions [`estimate`] / [`estimate_from_prices`] are also
//! available for direct use.

use argmin::core::CostFunction;
use argmin::core::Executor;
use argmin::core::Gradient;
use argmin::core::State;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use ndarray::ArrayView1;
use stochastic_rs_distributions::special::gamma;

use super::HurstDiagnostic;
use super::HurstError;
use super::HurstEstimator;
use super::HurstResult;
use crate::traits::FloatExt;

/// Estimation result from the adapted Whittle estimator.
#[derive(Clone, Debug)]
pub struct FukasawaResult {
  pub hurst: f64,
  pub eta: f64,
  pub neg_log_lik: f64,
  pub n_obs: usize,
}

/// Whittle estimator config (intraday bars + day fraction + numerical
/// tuning constants).  [`Whittle::default`] is daily data (`m = 1`,
/// `δ = 1/250`) — note this matches the high-bias setting of Fukasawa
/// Table 1; pass `m ≥ 72` for accurate intraday RV input.
#[derive(Clone, Copy, Debug)]
pub struct Whittle {
  /// Number of intraday observations per day.
  pub m: usize,
  /// Time interval per day in years.
  pub delta: f64,
  /// Low-frequency cutoff used in the spectral-density integral.
  pub psi: f64,
  /// Paxson truncation count `K` for the periodic spectral sum.
  pub k_trunc: usize,
  /// Taylor order `J` for the `A²` correction term.
  pub j_max: usize,
}

impl Default for Whittle {
  fn default() -> Self {
    Self {
      m: 1,
      delta: 1.0 / 250.0,
      psi: 1e-5,
      k_trunc: 500,
      j_max: 20,
    }
  }
}

impl Whittle {
  #[must_use]
  pub fn new(m: usize, delta: f64) -> Self {
    Self {
      m,
      delta,
      ..Self::default()
    }
  }
}

impl<T: FloatExt> HurstEstimator<T> for Whittle {
  fn estimate(&self, x: ArrayView1<T>) -> Result<HurstResult<T>, HurstError> {
    if x.len() < 30 {
      return Err(HurstError::TooFewObservations {
        got: x.len(),
        required: 30,
      });
    }
    let log_rv: Vec<f64> = x.iter().map(|v| v.to_f64().unwrap_or(f64::NAN)).collect();
    let result = run_estimate(&log_rv, self.m, self.delta, self.psi, self.k_trunc, self.j_max);
    Ok(HurstResult {
      hurst: T::from_f64_fast(result.hurst),
      std_err: None,
      n_obs: result.n_obs,
      diagnostic: HurstDiagnostic::Whittle {
        neg_log_lik: T::from_f64_fast(result.neg_log_lik),
        eta: T::from_f64_fast(result.eta),
      },
    })
  }
}

/// `C_H = Γ(2H+1) sin(πH) / (2π)`.
fn c_h(h: f64) -> f64 {
  let pi = std::f64::consts::PI;
  gamma(2.0 * h + 1.0) * (pi * h).sin() / (2.0 * pi)
}

/// Approximate spectral density of the observation process (eq. 14).
pub fn spectral_density(lambda: f64, h: f64, v: f64, m: usize, _n: usize, k_trunc: usize) -> f64 {
  let pi = std::f64::consts::PI;
  let ch = c_h(h);
  let alpha = 1.0 + 2.0 * h;

  let mut sum = lambda.abs().powf(-alpha);
  for k in 1..=k_trunc {
    let kf = k as f64;
    sum += (2.0 * pi * kf + lambda).abs().powf(-alpha)
      + (2.0 * pi * kf - lambda).abs().powf(-alpha);
  }

  let sin_half = (lambda / 2.0).sin();
  let signal = v * v * ch * 4.0 * sin_half * sin_half * sum;

  let half = lambda / 2.0;
  let ell = 2.0 * half.sin().powi(2) / pi;
  let noise = 2.0 / m as f64 * ell;

  signal + noise
}

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

fn autocovariance(y: &[f64]) -> Vec<f64> {
  let n = y.len();
  (0..n)
    .map(|tau| {
      let s: f64 = (0..n - tau).map(|t| y[t] * y[t + tau]).sum();
      s / n as f64
    })
    .collect()
}

fn correction_a1(psi: f64, h: f64, v: f64, m: usize) -> f64 {
  let pi = std::f64::consts::PI;
  let ch = c_h(h);
  let v2ch = v * v * ch;
  if v2ch < 1e-30 || psi < 1e-30 {
    return 0.0;
  }
  let log_term = psi * v2ch.ln() - (1.0 + 2.0 * h) * psi * (psi.ln() - 1.0);
  let noise_corr = 2.0 * psi.powf(2.0 + 2.0 * h) / (m as f64 * v2ch * (2.0 + 2.0 * h));
  (log_term + noise_corr) / (2.0 * pi)
}

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
  let mut coeff = 1.0_f64;
  let mut psi_pow = psi_base;

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

fn correction_a2(gamma_buf: &[f64], psi: f64, h: f64, v: f64, j_max: usize) -> f64 {
  let pi = std::f64::consts::PI;
  let a0 = a_hv(0, psi, h, v, j_max);
  let mut sum = a0 * gamma_buf[0];
  for (tau, &g_tau) in gamma_buf.iter().enumerate().skip(1) {
    sum += 2.0 * a_hv(tau, psi, h, v, j_max) * g_tau;
  }
  sum / (2.0 * pi)
}

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
  if count > 0 { sum / count as f64 } else { f64::INFINITY }
}

fn whittle_objective_full(
  pgram: &[f64],
  gamma_buf: &[f64],
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
  let a2 = correction_a2(gamma_buf, psi, h, v, j_max);
  let result = base + a1 + a2;
  if result.is_finite() { result } else { f64::INFINITY }
}

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

fn run_estimate(
  log_rv: &[f64],
  m: usize,
  delta: f64,
  psi: f64,
  k_trunc: usize,
  j_max: usize,
) -> FukasawaResult {
  let n = log_rv.len();
  let y: Vec<f64> = (1..n).map(|i| log_rv[i] - log_rv[i - 1]).collect();
  let ny = y.len();
  let n_freq = ny / 2;
  let pgram = periodogram(&y, n_freq);

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

  let gamma_hat = autocovariance(&y);
  let problem = WhittleProblem {
    pgram,
    gamma: gamma_hat,
    m,
    n: ny,
    psi,
    k_trunc,
    j_max,
    h_bounds: (0.005, 0.495),
    v_bounds: (0.01, 12.0),
  };
  let (h_r, v_r, nll_r) = run_lbfgs(&problem, best_h, best_v);
  if nll_r < best_nll {
    best_nll = nll_r;
    best_h = h_r;
    best_v = v_r;
  }
  let eta = best_v / delta.powf(best_h);
  FukasawaResult {
    hurst: best_h,
    eta,
    neg_log_lik: best_nll,
    n_obs: n,
  }
}

/// Estimate `(H, η)` from log realized variance increments.
///
/// `m` = intraday observations / day (e.g. 72 for 5-min bars).
/// `delta` = day fraction in years (e.g. `1/250`).  Uses dense grid +
/// L-BFGS refinement with Paxson spectral density correction.
pub fn estimate(log_rv: ArrayView1<f64>, m: usize, delta: f64) -> FukasawaResult {
  let log_rv = log_rv
    .as_slice()
    .expect("estimate requires a contiguous ArrayView1");
  assert!(log_rv.len() >= 30, "need at least 30 observations, got {}", log_rv.len());
  run_estimate(log_rv, m, delta, 1e-5, 500, 20)
}

/// Convenience: estimate `H` from daily close prices (`m = 1`,
/// `δ = 1/250`).  Note the heavy downward bias when `m = 1`
/// (Fukasawa Table 1 footnote).
pub fn estimate_from_prices(closes: ArrayView1<f64>) -> FukasawaResult {
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
  estimate(ArrayView1::from(&log_rv), 1, 1.0 / 250.0)
}

/// FloatExt generic wrapper.
pub fn estimate_from_prices_generic<T: FloatExt>(closes: ArrayView1<T>) -> FukasawaResult {
  let closes_f64: Vec<f64> = closes.iter().map(|x| x.to_f64().unwrap()).collect();
  estimate_from_prices(ArrayView1::from(&closes_f64))
}

#[cfg(test)]
mod tests {
  use super::*;
  use ndarray::Array1;
  use rand::SeedableRng;
  use rand::rngs::StdRng;
  use rand_distr::Distribution;
  use rand_distr::StandardNormal;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_stochastic::diffusion::fou::Fou;

  use crate::traits::ProcessExt;

  fn simulate_log_rv(true_h: f64, m: usize, n_days: usize, delta: f64, seed: u64) -> Vec<f64> {
    let fou = Fou::new(
      true_h,
      0.001,
      -3.2,
      1.0,
      n_days + 1,
      Some(-3.2),
      Some(n_days as f64 * delta),
      Deterministic::new(seed),
    );
    let log_vol_sq: Array1<f64> = fou.sample();
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
    log_rv
  }

  #[test]
  fn estimate_h_from_simulated_rv() {
    let true_h = 0.3_f64;
    let log_rv = simulate_log_rv(true_h, 72, 500, 1.0 / 250.0, 42);
    let result = estimate(ArrayView1::from(&log_rv), 72, 1.0 / 250.0);
    assert!(
      (result.hurst - true_h).abs() < 0.1,
      "H={:.3}, true={true_h}",
      result.hurst
    );
  }

  #[test]
  fn rough_vs_smooth_distinguished() {
    let log_rv_rough = simulate_log_rv(0.1, 72, 500, 1.0 / 250.0, 77);
    let log_rv_smooth = simulate_log_rv(0.45, 72, 500, 1.0 / 250.0, 88);
    let h_rough = estimate(ArrayView1::from(&log_rv_rough), 72, 1.0 / 250.0).hurst;
    let h_smooth = estimate(ArrayView1::from(&log_rv_smooth), 72, 1.0 / 250.0).hurst;
    assert!(
      h_rough < h_smooth,
      "rough H={h_rough:.3} should be < smooth H={h_smooth:.3}"
    );
  }

  #[test]
  fn whittle_trait_smoke() {
    let log_rv = simulate_log_rv(0.3, 72, 500, 1.0 / 250.0, 11);
    let r = Whittle {
      m: 72,
      delta: 1.0 / 250.0,
      ..Default::default()
    }
    .estimate(Array1::from_vec(log_rv).view())
    .expect("whittle trait estimate");
    assert!(r.hurst > 0.0 && r.hurst < 0.5);
    assert!(matches!(r.diagnostic, HurstDiagnostic::Whittle { .. }));
  }
}

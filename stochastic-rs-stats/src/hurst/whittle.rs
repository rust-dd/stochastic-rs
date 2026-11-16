//! Adapted Whittle estimate of the roughness `H` of log-volatility from log realized variance:
//! §3.2's Paxson spectral sum, low-frequency terms `A¹`, `A²` (eq. 16), L-BFGS-B from a grid start.
//!
//! $$
//! \hat H = \arg\min_{(H,\nu)} U_n(H,\nu), \quad
//! U_n \approx \frac{1}{2\pi}\int_\psi^\pi\!\left[\log g_{H,\nu}(\lambda)
//!       + \frac{I_n(\lambda)}{g_{H,\nu}(\lambda)}\right]\mathrm d\lambda
//!       + A^1_{H,\nu}(\psi) + A^2_{H,\nu}(\psi)
//! $$
//!
//! Reference: Fukasawa, M., Takabatake, T. & Westphal, R. (2019), "Is Volatility Rough?", arXiv:1905.04852.

use std::convert::Infallible;

use basin::BoxConstraints;
use basin::CostFunction;
use basin::Executor;
use basin::Gradient;
use basin::LbfgsState;
use basin::Lbfgsb;
use ndarray::ArrayView1;
use stochastic_rs_distributions::special::gamma;

use crate::StatsError;
use crate::input::as_slice_or_copy;
use crate::input::validate;
use crate::optim::more_thuente;

/// Estimation result from the adapted Whittle estimator.
#[derive(Clone, Debug)]
pub struct FukasawaFit {
  pub hurst: f64,
  pub eta: f64,
  pub neg_log_lik: f64,
  pub nobs: usize,
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

  /// The Fukasawa–Takabatake–Westphal fit to log realized variance; `Err` below 30 points or on a
  /// non-finite one.
  pub fn fit(&self, log_rv: ArrayView1<f64>) -> Result<FukasawaFit, StatsError> {
    let log_rv = as_slice_or_copy(log_rv);
    validate(&log_rv, 30)?;
    Ok(run_estimate(
      &log_rv,
      self.m,
      self.delta,
      self.psi,
      self.k_trunc,
      self.j_max,
    ))
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
    sum +=
      (2.0 * pi * kf + lambda).abs().powf(-alpha) + (2.0 * pi * kf - lambda).abs().powf(-alpha);
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
  if count > 0 {
    sum / count as f64
  } else {
    f64::INFINITY
  }
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
  if result.is_finite() {
    result
  } else {
    f64::INFINITY
  }
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
  lower: Vec<f64>,
  upper: Vec<f64>,
}

impl WhittleProblem {
  fn clamp(&self, params: &[f64]) -> Vec<f64> {
    params
      .iter()
      .enumerate()
      .map(|(i, &value)| value.clamp(self.lower[i], self.upper[i]))
      .collect()
  }

  fn eval(&self, params: &[f64]) -> f64 {
    // The spectral parameterization remains restricted during line searches.
    let params = self.clamp(params);
    whittle_objective_full(
      &self.pgram,
      &self.gamma,
      params[0],
      params[1],
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
  type Error = Infallible;

  fn cost(&self, params: &Self::Param) -> Result<Self::Output, Self::Error> {
    Ok(self.eval(params))
  }
}

impl Gradient for WhittleProblem {
  type Gradient = Vec<f64>;

  fn gradient(&self, params: &Self::Param) -> Result<Self::Gradient, Self::Error> {
    let params = self.clamp(params);
    let mut grad = vec![0.0; 2];
    for i in 0..2 {
      let step = 1e-7 * (1.0 + params[i].abs());
      let mut p_plus = params.clone();
      let mut p_minus = params.clone();
      p_plus[i] = (params[i] + step).min(self.upper[i]);
      p_minus[i] = (params[i] - step).max(self.lower[i]);
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

impl BoxConstraints for WhittleProblem {
  fn lower(&self) -> &Self::Param {
    &self.lower
  }

  fn upper(&self) -> &Self::Param {
    &self.upper
  }
}

fn run_lbfgs(problem: &WhittleProblem, h_init: f64, v_init: f64) -> (f64, f64, f64) {
  let init = vec![h_init, v_init];
  let solver = Lbfgsb::with_line_search(more_thuente())
    .with_absolute_projected_gradient_tolerance(f64::EPSILON.sqrt())
    .with_absolute_cost_change_tolerance(f64::EPSILON);
  let state = LbfgsState::new(init, 10);
  let result = Executor::new(problem.clone(), solver, state)
    .max_iter(200)
    .run()
    .expect("Whittle objective is infallible");
  let best = problem.clamp(result.best_param());
  (best[0], best[1], problem.eval(&best))
}

fn run_estimate(
  log_rv: &[f64],
  m: usize,
  delta: f64,
  psi: f64,
  k_trunc: usize,
  j_max: usize,
) -> FukasawaFit {
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
    lower: vec![0.005, 0.01],
    upper: vec![0.495, 12.0],
  };
  let (h_r, v_r, nll_r) = run_lbfgs(&problem, best_h, best_v);
  if nll_r < best_nll {
    best_nll = nll_r;
    best_h = h_r;
    best_v = v_r;
  }
  let eta = best_v / delta.powf(best_h);
  FukasawaFit {
    hurst: best_h,
    eta,
    neg_log_lik: best_nll,
    nobs: n,
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::SimdDistribution;
  use stochastic_rs_distributions::normal::SimdNormal;
  use stochastic_rs_stochastic::diffusion::fou::Fou;

  use super::*;
  use crate::traits::ProcessExt;

  fn bounded_problem() -> WhittleProblem {
    WhittleProblem {
      pgram: vec![0.1; 16],
      gamma: vec![0.1; 32],
      m: 72,
      n: 32,
      psi: 1e-5,
      k_trunc: 20,
      j_max: 5,
      lower: vec![0.005, 0.01],
      upper: vec![0.495, 12.0],
    }
  }

  #[test]
  fn bounded_whittle_cost_and_gradient_use_feasible_parameters() {
    let problem = bounded_problem();
    for (outside, boundary) in [
      (vec![0.6, 1.0], vec![0.495, 1.0]),
      (vec![-0.1, 1.0], vec![0.005, 1.0]),
      (vec![0.3, 15.0], vec![0.3, 12.0]),
    ] {
      assert_eq!(problem.eval(&outside), problem.eval(&boundary));
      let expected = problem.gradient(&boundary).unwrap();
      assert!(expected.iter().any(|value| value.abs() > 1e-6));
      assert_eq!(problem.gradient(&outside).unwrap(), expected);
    }
  }

  #[test]
  fn bounded_whittle_refinement_returns_feasible_parameters_and_cost() {
    let problem = bounded_problem();
    for initial in [[0.2, 1.0], [0.6, 15.0], [-0.1, -1.0]] {
      let (h, v, cost) = run_lbfgs(&problem, initial[0], initial[1]);
      assert!((problem.lower[0]..=problem.upper[0]).contains(&h));
      assert!((problem.lower[1]..=problem.upper[1]).contains(&v));
      assert!(cost.is_finite());
      assert_eq!(cost, problem.eval(&[h, v]));
    }
  }

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
    let mut log_rv = vec![0.0_f64; n_days];
    let mut normals = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(0x7717));
    for day in 0..n_days {
      let sigma = log_vol_sq[day].exp().sqrt();
      let dt = delta / m as f64;
      let mut rv = 0.0;
      for _ in 0..m {
        let z = normals.sample();
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
    let result = Whittle::new(72, 1.0 / 250.0)
      .fit(ArrayView1::from(&log_rv))
      .unwrap();
    assert!(
      (result.hurst - true_h).abs() < 0.1,
      "H={:.3}, true={true_h}",
      result.hurst
    );
  }

  #[test]
  fn rough_vs_smooth_distinguished() {
    let whittle = Whittle::new(72, 1.0 / 250.0);
    let log_rv_rough = simulate_log_rv(0.1, 72, 500, 1.0 / 250.0, 77);
    let log_rv_smooth = simulate_log_rv(0.45, 72, 500, 1.0 / 250.0, 88);
    let h_rough = whittle.fit(ArrayView1::from(&log_rv_rough)).unwrap().hurst;
    let h_smooth = whittle.fit(ArrayView1::from(&log_rv_smooth)).unwrap().hurst;
    assert!(
      h_rough < h_smooth,
      "rough H={h_rough:.3} should be < smooth H={h_smooth:.3}"
    );
  }

  /// `fit` is the estimator's one entry; the `HurstEstimator` trait is for path self-similarity, not latent volatility.
  #[test]
  fn fit_reports_the_fukasawa_estimate_and_rejects_short_or_non_finite_data() {
    let whittle = Whittle::new(72, 1.0 / 250.0);
    let short = Array1::<f64>::linspace(-1.0, 1.0, 10);
    assert_eq!(
      whittle.fit(short.view()).unwrap_err(),
      StatsError::InsufficientData {
        needed: 30,
        got: 10
      }
    );
    let mut with_nan = Array1::<f64>::linspace(-1.0, 1.0, 40);
    with_nan[5] = f64::NAN;
    assert!(matches!(
      whittle.fit(with_nan.view()),
      Err(StatsError::InvalidObservation { index: 5, .. })
    ));
    let log_rv = simulate_log_rv(0.1, 72, 400, 1.0 / 250.0, 3);
    let fit = whittle.fit(ArrayView1::from(&log_rv)).unwrap();
    assert!(
      (fit.hurst - 0.1).abs() < 0.08 && fit.eta > 0.0 && fit.nobs == 400,
      "{fit:?}"
    );
  }
}

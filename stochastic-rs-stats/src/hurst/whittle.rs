//! Adapted Whittle estimate of the roughness `H` of log-volatility from log realized variance.
//!
//! $$
//! \hat H = \arg\min_{(H,\nu)} U_n(H,\nu), \quad
//! U_n \approx \frac{1}{2\pi}\int_\psi^\pi\!\left[\log g_{H,\nu}(\lambda)
//!       + \frac{I_n(\lambda)}{g_{H,\nu}(\lambda)}\right]\mathrm d\lambda
//!       + A^1_{H,\nu}(\psi) + A^2_{H,\nu}(\psi)
//! $$
//!
//! §3.2's Paxson spectral sum, the low-frequency terms `A¹`, `A²` (eq. 16), L-BFGS-B from a grid.
//!
//! Reference: Fukasawa, Takabatake & Westphal (2019), "Is Volatility Rough?", arXiv:1905.04852

use std::convert::Infallible;
use std::f64::consts::PI;

use basin::BoxBounds;
use basin::ConstraintDescription;
use basin::ConstraintSet;
use basin::Constraints;
use basin::CostFunction;
use basin::Executor;
use basin::FirstOrderState;
use basin::Gradient;
use ndarray::ArrayView1;
use stochastic_rs_distributions::special::gamma;

use crate::StatsError;
use crate::input::as_slice_or_copy;
use crate::input::reject_constant;
use crate::input::validate;
use crate::optim::lbfgsb;
use crate::optim::more_thuente;

/// Estimation result from the adapted Whittle estimator.
#[derive(Clone, Debug)]
pub struct FukasawaFit {
  pub hurst: f64,
  pub eta: f64,
  pub neg_log_lik: f64,
  pub nobs: usize,
  /// L-BFGS-B, from the grid's best point, stopped on a tolerance and not on its 200 iterations.
  pub converged: bool,
}

/// Whittle config: intraday bars `m`, day fraction `δ`, numerical constants. The default is daily
/// (`m = 1`, `δ = 1/250`), the high-bias case of Fukasawa Table 1; intraday RV wants `m ≥ 72`.
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
  /// `m` intraday observations a day of `delta` years; panics unless `m >= 1` and `delta` is finite
  /// and positive.
  #[must_use]
  pub fn new(m: usize, delta: f64) -> Self {
    let whittle = Self {
      m,
      delta,
      ..Self::default()
    };
    whittle.assert_config();
    whittle
  }

  /// Fukasawa–Takabatake–Westphal's fit to log RV; `Err` below 30 points, on a non-finite point or
  /// objective, a constant series, or unless `m >= 1`, `0 < delta < ∞` and `0 < psi < π`.
  pub fn fit(&self, log_rv: ArrayView1<f64>) -> Result<FukasawaFit, StatsError> {
    self.check_config()?;
    let log_rv = as_slice_or_copy(log_rv);
    validate(&log_rv, 30)?;
    reject_constant(&log_rv, "constant series")?;
    run_estimate(
      &log_rv,
      self.m,
      self.delta,
      self.psi,
      self.k_trunc,
      self.j_max,
    )
  }

  /// [`Whittle::new`]'s checks and `psi`'s, which `fit` answers for a struct literal.
  fn check_config(&self) -> Result<(), StatsError> {
    let Self { m, delta, psi, .. } = *self;
    let invalid = |name, value, constraint| {
      Err(StatsError::InvalidParameter {
        name,
        value,
        constraint,
      })
    };
    if m < 1 {
      return invalid("m", m as f64, "m >= 1");
    }
    if !(delta.is_finite() && delta > 0.0) {
      return invalid("delta", delta, "0.0 < delta < inf");
    }
    if !(psi > 0.0 && psi < PI) {
      return invalid("psi", psi, "0.0 < psi < pi");
    }
    Ok(())
  }

  /// [`Whittle::check_config`] as the panic of `new` and of the Python constructor.
  #[track_caller]
  pub(crate) fn assert_config(&self) {
    let Self { m, delta, psi, .. } = *self;
    assert!(m >= 1, "m must satisfy `m >= 1`, got m = {m:?}");
    assert!(
      delta.is_finite(),
      "delta must satisfy `delta.is_finite()`, got delta = {delta:?}"
    );
    assert!(
      delta > 0.0,
      "delta must satisfy `delta > 0.0`, got delta = {delta:?}"
    );
    assert!(
      psi > 0.0 && psi < PI,
      "psi must satisfy `0.0 < psi < {PI:?}`, got psi = {psi:?}"
    );
  }
}

/// `C_H = Γ(2H+1) sin(πH) / (2π)`.
fn c_h(h: f64) -> f64 {
  let pi = std::f64::consts::PI;
  gamma(2.0 * h + 1.0) * (pi * h).sin() / (2.0 * pi)
}

/// Approximate spectral density of the observation process (eq. 14).
fn spectral_density(lambda: f64, h: f64, v: f64, m: usize, k_trunc: usize) -> f64 {
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

fn whittle_objective(
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
    let g = spectral_density(lam, h, v, m, k_trunc);
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
    self.objective(params[0], params[1])
  }

  fn objective(&self, h: f64, v: f64) -> f64 {
    if h <= 0.0 || h >= 1.0 || v <= 0.0 {
      return f64::INFINITY;
    }
    let base = whittle_objective(&self.pgram, h, v, self.m, self.n, self.psi, self.k_trunc);
    if !base.is_finite() {
      return f64::INFINITY;
    }
    let a1 = correction_a1(self.psi, h, v, self.m);
    let a2 = correction_a2(&self.gamma, self.psi, h, v, self.j_max);
    let result = base + a1 + a2;
    if result.is_finite() {
      result
    } else {
      f64::INFINITY
    }
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

impl Constraints for WhittleProblem {
  type Set = ConstraintSet<BoxBounds<Vec<f64>>>;

  fn constraints(&self) -> <Self::Set as ConstraintDescription>::Borrowed<'_> {
    ConstraintSet::new().with_bounds(BoxBounds::new(Some(&self.lower), Some(&self.upper)))
  }
}

fn run_lbfgs(problem: &WhittleProblem, h_init: f64, v_init: f64) -> (f64, f64, f64, bool) {
  let init = vec![h_init, v_init];
  let state = FirstOrderState::new(init.clone());
  let result = Executor::new(problem.clone(), lbfgsb(more_thuente()), state)
    .max_iter(200)
    .run()
    .expect("the Whittle box is ordered and its objective infallible");
  let converged = result.report.termination.is_converged();
  // A search that met no finite objective leaves the start, which the caller reports.
  let best = problem.clamp(result.state.best().map_or(&init, |(best, _)| best));
  (best[0], best[1], problem.eval(&best), converged)
}

fn run_estimate(
  log_rv: &[f64],
  m: usize,
  delta: f64,
  psi: f64,
  k_trunc: usize,
  j_max: usize,
) -> Result<FukasawaFit, StatsError> {
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
  let (h_r, v_r, nll_r, converged) = run_lbfgs(&problem, best_h, best_v);
  if nll_r < best_nll {
    best_nll = nll_r;
    best_h = h_r;
    best_v = v_r;
  }
  if !best_nll.is_finite() {
    return Err(StatsError::Numerical {
      stage: "whittle objective",
    });
  }
  let eta = best_v / delta.powf(best_h);
  Ok(FukasawaFit {
    hurst: best_h,
    eta,
    neg_log_lik: best_nll,
    nobs: n,
    converged,
  })
}

#[cfg(test)]
mod tests;

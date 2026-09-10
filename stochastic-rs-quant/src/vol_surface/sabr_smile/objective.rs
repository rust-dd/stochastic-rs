use std::convert::Infallible;

use basin::BoxConstraints;
use basin::CostFunction;
use basin::CostTolerance;
use basin::Gradient;
use basin::InnerExecutor;
use basin::LbfgsState;
use basin::Lbfgsb;
use basin::MoreThuente;
use basin::Problem;
use basin::TerminationReason;
use stochastic_rs_core::simd_rng::SimdRng;

use crate::pricing::sabr::alpha_from_atm_vol;
use crate::pricing::sabr::bs_price_fx;
use crate::pricing::sabr::forward_fx;
use crate::pricing::sabr::fx_delta_from_forward;
use crate::pricing::sabr::hagan_implied_vol;
use crate::pricing::sabr::model_price_hagan_general;

pub(super) fn rr_sigma(
  k_call: f64,
  k_put: f64,
  f: f64,
  tau: f64,
  alpha: f64,
  beta: f64,
  nu: f64,
  rho: f64,
) -> f64 {
  let sc = hagan_implied_vol(k_call, f, tau, alpha, beta, nu, rho);
  let sp = hagan_implied_vol(k_put, f, tau, alpha, beta, nu, rho);
  sc - sp
}

pub(super) fn bf_premium_mismatch(
  s: f64,
  k_call: f64,
  k_put: f64,
  r_d: f64,
  r_f: f64,
  tau: f64,
  alpha: f64,
  beta: f64,
  nu: f64,
  rho: f64,
  sigma_ref: f64,
) -> f64 {
  let (mc, _) = model_price_hagan_general(s, k_call, r_d, r_f, tau, alpha, beta, nu, rho);
  let (_, mp) = model_price_hagan_general(s, k_put, r_d, r_f, tau, alpha, beta, nu, rho);
  let (bc, _) = bs_price_fx(s, k_call, r_d, r_f, tau, sigma_ref);
  let (_, bp) = bs_price_fx(s, k_put, r_d, r_f, tau, sigma_ref);
  (mc + mp) - (bc + bp)
}

/// Optimization variables: `[k_rr_c, k_rr_p, k_bf_c, k_bf_p, nu, rho]`.
///
/// α is derived from σ_ATM via [`alpha_from_atm_vol`] so the ATM vol is
/// matched by construction.
pub(super) const NVARS: usize = 6;

/// Problem definition for Basin optimization.
#[derive(Clone)]
pub(super) struct SabrSmileProblem {
  pub(super) s: f64,
  pub(super) r_d: f64,
  pub(super) r_f: f64,
  pub(super) tau: f64,
  pub(super) beta: f64,
  pub(super) sigma_atm: f64,
  pub(super) sigma_rr: f64,
  pub(super) sigma_bf: f64,
  pub(super) bounds_lo: Vec<f64>,
  pub(super) bounds_hi: Vec<f64>,
}

impl SabrSmileProblem {
  pub(super) fn clamp_params(&self, x: &[f64]) -> [f64; NVARS] {
    std::array::from_fn(|i| x[i].clamp(self.bounds_lo[i], self.bounds_hi[i]))
  }
}

impl CostFunction for SabrSmileProblem {
  type Param = Vec<f64>;
  type Output = f64;
  type Error = Infallible;

  fn cost(&self, x: &Self::Param) -> Result<Self::Output, Self::Error> {
    // Raw starts and line-search probes must satisfy Hagan's domain contract.
    let x = self.clamp_params(x);
    let (k_rr_c, k_rr_p, k_bf_c, k_bf_p, nu, rho) = (x[0], x[1], x[2], x[3], x[4], x[5]);

    let f = forward_fx(self.s, self.tau, self.r_d, self.r_f);
    let alpha = alpha_from_atm_vol(self.sigma_atm, f, self.tau, self.beta, rho, nu);
    if !(alpha.is_finite() && alpha > 0.0) {
      // Outside the Hagan formula's domain: an infeasible point of the search,
      // scored as such rather than letting the formula's contract panic.
      return Ok(f64::MAX);
    }

    let term_rr =
      (rr_sigma(k_rr_c, k_rr_p, f, self.tau, alpha, self.beta, nu, rho) - self.sigma_rr).powi(2);

    let call_sigma_rr = hagan_implied_vol(k_rr_c, f, self.tau, alpha, self.beta, nu, rho);
    let put_sigma_rr = hagan_implied_vol(k_rr_p, f, self.tau, alpha, self.beta, nu, rho);
    let d_call_rr = fx_delta_from_forward(k_rr_c, f, call_sigma_rr, self.tau, self.r_f, 1.0);
    let d_put_rr = fx_delta_from_forward(k_rr_p, f, put_sigma_rr, self.tau, self.r_f, -1.0);
    let term_rr_delta = (d_call_rr - 0.25).powi(2) + (d_put_rr + 0.25).powi(2);

    // Market strangle convention: σ_ref = σ_ATM + σ_BF when computing delta.
    let sigma_ref = self.sigma_atm + self.sigma_bf;
    let term_bf = bf_premium_mismatch(
      self.s, k_bf_c, k_bf_p, self.r_d, self.r_f, self.tau, alpha, self.beta, nu, rho, sigma_ref,
    )
    .powi(2);

    let d_call_bf = fx_delta_from_forward(k_bf_c, f, sigma_ref, self.tau, self.r_f, 1.0);
    let d_put_bf = fx_delta_from_forward(k_bf_p, f, sigma_ref, self.tau, self.r_f, -1.0);
    let term_bf_delta = (d_call_bf - 0.25).powi(2) + (d_put_bf + 0.25).powi(2);

    Ok(term_rr + term_rr_delta + term_bf + term_bf_delta)
  }
}

impl Gradient for SabrSmileProblem {
  type Gradient = Vec<f64>;

  fn gradient(&self, x: &Self::Param) -> Result<Self::Gradient, Self::Error> {
    let x = self.clamp_params(x);
    let eps = 1e-8;
    let mut grad = vec![0.0; NVARS];
    for i in 0..NVARS {
      let mut x_plus = x.to_vec();
      let mut x_minus = x.to_vec();
      x_plus[i] = (x_plus[i] + eps).min(self.bounds_hi[i]);
      x_minus[i] = (x_minus[i] - eps).max(self.bounds_lo[i]);
      let step = x_plus[i] - x_minus[i];
      if step > 0.0 {
        grad[i] = (self.cost(&x_plus)? - self.cost(&x_minus)?) / step;
      }
    }
    Ok(grad)
  }
}

impl BoxConstraints for SabrSmileProblem {
  fn lower(&self) -> &Self::Param {
    &self.bounds_lo
  }

  fn upper(&self) -> &Self::Param {
    &self.bounds_hi
  }
}

/// Basin-hopping with Basin L-BFGS-B.
///
/// Keep the existing SimdRng stream and skip every failed local solve. Basin's
/// BasinHopping owns a ChaCha RNG, optimizes the initial point before hopping,
/// and can accept failed candidates when the incumbent also failed.
pub(super) fn basin_hopping_opt(
  x0: [f64; NVARS],
  niter: usize,
  stepsize: f64,
  problem: &SabrSmileProblem,
) -> ([f64; NVARS], f64) {
  let mut rng = SimdRng::from_seed(3);

  let mut current_x = problem.clamp_params(&x0);
  let mut current_f = problem.cost(&current_x.to_vec()).unwrap_or(f64::INFINITY);

  let mut best_x = current_x;
  let mut best_f = current_f;

  let temp = 1.0_f64;

  let linesearch = MoreThuente::new()
    .ftol(1e-4)
    .gtol(0.9)
    .xtol(1e-10)
    .stpmin(f64::EPSILON.sqrt())
    .stpmax(f64::INFINITY);
  let solver = Lbfgsb::with_line_search(linesearch).with_tol_pg(f64::EPSILON.sqrt());
  let mut local = InnerExecutor::new(solver)
    .max_iter(100)
    .terminate_on(CostTolerance::new(f64::EPSILON));
  let mut local_problem = Problem::new(problem.clone());

  for _ in 0..niter {
    let mut x_trial = current_x;
    for (i, x) in x_trial.iter_mut().enumerate() {
      *x += (rng.next_f64() * 2.0 - 1.0) * stepsize;
      *x = (*x).clamp(problem.bounds_lo[i], problem.bounds_hi[i]);
    }

    let x_init = x_trial.to_vec();
    let state = LbfgsState::new(x_init, 10);
    let result = local
      .run(&mut local_problem, state)
      .expect("Sabr smile objective is infallible");

    if result.reason != TerminationReason::SolverFailed {
      let param = problem.clamp_params(result.param());
      let cost = problem
        .cost(&param.to_vec())
        .expect("Sabr smile objective is infallible");

      let delta = cost - current_f;
      let accept = if delta <= 0.0 {
        true
      } else {
        let u = rng.next_f64();
        u < (-delta / temp).exp()
      };

      if accept {
        current_x = param;
        current_f = cost;

        if cost < best_f {
          best_f = cost;
          best_x = current_x;
        }
      }
    }
  }

  (best_x, best_f)
}

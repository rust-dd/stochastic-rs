use argmin::core::CostFunction;
use argmin::core::Executor;
use argmin::core::Gradient;
use argmin::core::State;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

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

/// Problem definition for argmin optimization
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
  pub(super) bounds_lo: [f64; NVARS],
  pub(super) bounds_hi: [f64; NVARS],
}

impl SabrSmileProblem {
  fn clamp_params(&self, x: &[f64]) -> [f64; NVARS] {
    let mut p = [0.0; NVARS];
    for i in 0..NVARS {
      p[i] = x[i].clamp(self.bounds_lo[i], self.bounds_hi[i]);
    }
    p
  }
}

impl CostFunction for SabrSmileProblem {
  type Param = Vec<f64>;
  type Output = f64;

  fn cost(&self, x: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
    let p = self.clamp_params(x);
    let (k_rr_c, k_rr_p, k_bf_c, k_bf_p, nu, rho) = (p[0], p[1], p[2], p[3], p[4], p[5]);

    let f = forward_fx(self.s, self.tau, self.r_d, self.r_f);
    let alpha = alpha_from_atm_vol(self.sigma_atm, f, self.tau, self.beta, rho, nu);

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
  type Param = Vec<f64>;
  type Gradient = Vec<f64>;

  fn gradient(&self, x: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
    let eps = 1e-8;
    let f0 = self.cost(x)?;
    let mut grad = vec![0.0; NVARS];
    for i in 0..NVARS {
      let mut x_plus = x.clone();
      x_plus[i] += eps;
      grad[i] = (self.cost(&x_plus)? - f0) / eps;
    }
    Ok(grad)
  }
}

/// Basin-hopping with argmin L-BFGS
pub(super) fn basin_hopping_opt(
  x0: [f64; NVARS],
  niter: usize,
  stepsize: f64,
  problem: &SabrSmileProblem,
) -> ([f64; NVARS], f64) {
  let mut rng = StdRng::seed_from_u64(3);

  let mut current_x = x0;
  let mut current_f = problem.cost(&x0.to_vec()).unwrap_or(f64::INFINITY);

  let mut best_x = current_x;
  let mut best_f = current_f;

  let temp = 1.0_f64;

  for _ in 0..niter {
    let mut x_trial = current_x;
    for (i, x) in x_trial.iter_mut().enumerate() {
      *x += rng.random_range(-stepsize..stepsize);
      *x = (*x).clamp(problem.bounds_lo[i], problem.bounds_hi[i]);
    }

    let linesearch = MoreThuenteLineSearch::new()
      .with_c(1e-4, 0.9)
      .expect("Wolfe params (1e-4, 0.9) satisfy 0 < c1 < c2 < 1 by construction");
    let solver = LBFGS::new(linesearch, 10);

    let x_init = x_trial.to_vec();
    let res = Executor::new(problem.clone(), solver)
      .configure(|state| state.param(x_init).max_iters(100))
      .run();

    if let Ok(optimization_result) = res {
      let state = optimization_result.state();
      if let Some(param) = state.get_param() {
        let cost = state.get_cost();

        let delta = cost - current_f;
        let accept = if delta <= 0.0 {
          true
        } else {
          let u: f64 = rng.random();
          u < (-delta / temp).exp()
        };

        if accept {
          current_x.copy_from_slice(&param[..NVARS]);
          current_f = cost;

          if cost < best_f {
            best_f = cost;
            best_x = current_x;
          }
        }
      }
    }
  }

  (best_x, best_f)
}

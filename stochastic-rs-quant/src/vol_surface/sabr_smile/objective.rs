use std::convert::Infallible;

use basin::BoxBounds;
use basin::ConstraintDescription;
use basin::ConstraintSet;
use basin::ConstraintSolveError;
use basin::Constraints;
use basin::CostFunction;
use basin::FirstOrderState;
use basin::Gradient;
use basin::InnerExecutor;
use basin::Lbfgsb;
use basin::MoreThuente;
use basin::Problem;
use basin::Solver;
use stochastic_rs_core::simd_rng::SimdRng;
use stochastic_rs_distributions::RealExt;

use super::types::SmileStrikes;
use crate::QuantError;
use crate::pricing::sabr::SabrParams;
use crate::pricing::sabr::alpha_from_atm_vol;
use crate::pricing::sabr::bs_price_fx;
use crate::pricing::sabr::forward_fx;
use crate::pricing::sabr::fx_delta_from_forward;
use crate::pricing::sabr::hagan_implied_vol;
use crate::pricing::sabr::model_price_hagan;

pub(super) fn rr_sigma(k_call: f64, k_put: f64, f: f64, tau: f64, params: &SabrParams) -> f64 {
  let sc = hagan_implied_vol(k_call, f, tau, params);
  let sp = hagan_implied_vol(k_put, f, tau, params);
  sc - sp
}

// The butterfly's query, its two strikes, the model and the reference vol.
#[allow(clippy::too_many_arguments)]
pub(super) fn bf_premium_mismatch(
  s: f64,
  k_call: f64,
  k_put: f64,
  r_d: f64,
  r_f: f64,
  tau: f64,
  params: &SabrParams,
  sigma_ref: f64,
) -> f64 {
  let (mc, _) = model_price_hagan(s, k_call, r_d, r_f, tau, params);
  let (_, mp) = model_price_hagan(s, k_put, r_d, r_f, tau, params);
  let (bc, _) = bs_price_fx(s, k_call, r_d, r_f, tau, sigma_ref);
  let (_, bp) = bs_price_fx(s, k_put, r_d, r_f, tau, sigma_ref);
  (mc + mp) - (bc + bp)
}

pub(super) const NU_BOUNDS: (f64, f64) = (0.01, 10.0);

pub(super) const RHO_BOUNDS: (f64, f64) = (-0.99, 0.99);

/// Garman–Kohlhagen vega of one option at `sigma`.
fn fx_vega(s: f64, k: f64, f: f64, r_f: f64, tau: f64, sigma: f64) -> f64 {
  let root = tau.sqrt();
  let d1 = ((f / k).ln() + 0.5 * sigma * sigma * tau) / (sigma * root);
  s * (-r_f * tau).exp() * (-0.5 * d1 * d1).exp() / std::f64::consts::TAU.sqrt() * root
}

/// Optimisation variables `[k_rr_c, k_rr_p, k_bf_c, k_bf_p, nu, rho]`; `α` comes from `σ_ATM` via
/// [`alpha_from_atm_vol`], so the ATM vol is matched by construction.
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
  /// Market and model `[ATM, RR, BF]` in vol, the model BF `σ_BF` plus the strangle premium
  /// mismatch over its vega at `σ_ATM + σ_BF`; `Numerical` unless that vega is finite and positive.
  pub(super) fn quotes_in_vol(
    &self,
    p: &SabrParams,
    k: &SmileStrikes,
  ) -> Result<([f64; 3], [f64; 3]), QuantError> {
    let f = k.k_atm;
    let sigma_ref = self.sigma_atm + self.sigma_bf;
    let mismatch = bf_premium_mismatch(
      self.s,
      k.k_bf_call,
      k.k_bf_put,
      self.r_d,
      self.r_f,
      self.tau,
      p,
      sigma_ref,
    );
    let vega = fx_vega(self.s, k.k_bf_call, f, self.r_f, self.tau, sigma_ref)
      + fx_vega(self.s, k.k_bf_put, f, self.r_f, self.tau, sigma_ref);
    if !(vega.is_finite() && vega > 0.0) {
      return Err(QuantError::Numerical {
        what: "strangle vega",
        reason: format!(
          "it is {vega:?} at strikes {:?} and {:?}, sigma_atm + sigma_bf = {sigma_ref:?}",
          k.k_bf_call, k.k_bf_put
        ),
      });
    }
    let market = [self.sigma_atm, self.sigma_rr, self.sigma_bf];
    let model = [
      hagan_implied_vol(f, f, self.tau, p),
      rr_sigma(k.k_rr_call, k.k_rr_put, f, self.tau, p),
      self.sigma_bf + mismatch / vega,
    ];
    Ok((market, model))
  }

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

    // Parameters Hagan's formula refuses, a NaN coordinate's among them, price nothing.
    if !(rho > -1.0 && rho < 1.0) {
      return Ok(f64::NAN);
    }
    let f = forward_fx(self.s, self.tau, self.r_d, self.r_f);
    let alpha = alpha_from_atm_vol(self.sigma_atm, f, self.tau, self.beta, rho, nu);
    let params = SabrParams {
      alpha,
      beta: self.beta,
      rho,
      nu,
    };
    if params.violation().is_some() {
      return Ok(f64::NAN);
    }
    let term_rr = (rr_sigma(k_rr_c, k_rr_p, f, self.tau, &params) - self.sigma_rr).powi(2);

    let call_sigma_rr = hagan_implied_vol(k_rr_c, f, self.tau, &params);
    let put_sigma_rr = hagan_implied_vol(k_rr_p, f, self.tau, &params);
    let d_call_rr = fx_delta_from_forward(k_rr_c, f, call_sigma_rr, self.tau, self.r_f, 1.0);
    let d_put_rr = fx_delta_from_forward(k_rr_p, f, put_sigma_rr, self.tau, self.r_f, -1.0);
    let term_rr_delta = (d_call_rr - 0.25).powi(2) + (d_put_rr + 0.25).powi(2);

    // Market strangle convention: σ_ref = σ_ATM + σ_BF when computing delta.
    let sigma_ref = self.sigma_atm + self.sigma_bf;
    let term_bf = bf_premium_mismatch(
      self.s, k_bf_c, k_bf_p, self.r_d, self.r_f, self.tau, &params, sigma_ref,
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
      x_plus[i] = (x_plus[i] + eps).min_or_nan(self.bounds_hi[i]);
      x_minus[i] = (x_minus[i] - eps).max_or_nan(self.bounds_lo[i]);
      let step = x_plus[i] - x_minus[i];
      if step != 0.0 {
        grad[i] = (self.cost(&x_plus)? - self.cost(&x_minus)?) / step;
      }
    }
    Ok(grad)
  }
}

impl Constraints for SabrSmileProblem {
  type Set = ConstraintSet<BoxBounds<Vec<f64>>>;

  fn constraints(&self) -> <Self::Set as ConstraintDescription>::Borrowed<'_> {
    ConstraintSet::new().with_bounds(BoxBounds::new(Some(&self.bounds_lo), Some(&self.bounds_hi)))
  }
}

/// A hop's local search: L-BFGS-B over ten curvature pairs under Moré–Thuente, stopping on a `√ε`
/// projected gradient or an `ε` cost change alone, its looser relative cost-reduction stop off.
pub(super) fn local_search<P>()
-> impl Solver<P, FirstOrderState<Vec<f64>>, Error = ConstraintSolveError<P::Error>>
where
  P: CostFunction<Param = Vec<f64>, Output = f64>
    + Gradient<Gradient = Vec<f64>>
    + Constraints<Set = ConstraintSet<BoxBounds<Vec<f64>>>>,
{
  let line_search = MoreThuente::new()
    .with_sufficient_decrease_coefficient(1e-4)
    .with_curvature_coefficient(0.9)
    .with_relative_bracket_tolerance(1e-10)
    .stpmin(f64::EPSILON.sqrt())
    .stpmax(f64::INFINITY);
  Lbfgsb::<Vec<f64>, f64, _>::with_line_search(line_search)
    .with_m_capacity(10)
    .with_absolute_projected_gradient_tolerance(f64::EPSILON.sqrt())
    .with_relative_cost_reduction_tolerance(None)
    .with_absolute_cost_change_tolerance(f64::EPSILON)
}

/// Basin-hopping over basin's L-BFGS-B on this crate's `SimdRng` stream, skipping failed local
/// solves; basin's own `BasinHopping` owns a ChaCha RNG and can accept a failed candidate.
pub(super) fn basin_hopping_opt(
  x0: [f64; NVARS],
  niter: usize,
  stepsize: f64,
  problem: &SabrSmileProblem,
) -> ([f64; NVARS], f64) {
  let mut rng = SimdRng::from_seed(3);

  let mut current_x = problem.clamp_params(&x0);
  // A start Hagan's formula refuses costs NaN; ranked `+∞`, any hop that prices replaces it.
  let start = problem.cost(&current_x.to_vec()).ok();
  let mut current_f = start.filter(|f| !f.is_nan()).unwrap_or(f64::INFINITY);

  let mut best_x = current_x;
  let mut best_f = current_f;

  let temp = 1.0_f64;

  let mut local = InnerExecutor::new(local_search()).max_iter(100);
  let mut local_problem = Problem::new(problem.clone());

  for _ in 0..niter {
    let mut x_trial = current_x;
    for (i, x) in x_trial.iter_mut().enumerate() {
      *x += (rng.next_f64() * 2.0 - 1.0) * stepsize;
      *x = (*x).clamp(problem.bounds_lo[i], problem.bounds_hi[i]);
    }

    let x_init = x_trial.to_vec();
    let state = FirstOrderState::new(x_init);
    let result = local
      .run(&mut local_problem, state)
      .expect("Sabr smile bounds are ordered and its objective infallible");

    if !result.report.termination.is_failure() {
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

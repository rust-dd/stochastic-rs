//! # SABR Smile
//!
//! FX volatility smile calibrator using the Hagan (2002) SABR approximation
//! with general β support.
//!
//! **Reference:** P. S. Hagan, D. Kumar, A. S. Lesniewski, D. E. Woodward,
//! *Managing Smile Risk*, Wilmott Magazine, pp. 84–108, 2002.
//!
//! α is derived analytically from the ATM vol (Eq. A.69b) so the
//! optimizer only searches over (ρ, ν) plus the four 25-delta strike levels.
//!
//! $$
//! \sigma_{imp}(K)\approx \sigma_{Hagan}(K;\alpha(\sigma_{ATM},\rho,\nu),\beta,\rho,\nu)
//! $$
//!
use argmin::core::CostFunction;
use argmin::core::Executor;
use argmin::core::Gradient;
use argmin::core::State;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use plotly::Plot;
use plotly::Scatter;
use plotly::common::Mode;
use plotly::common::Title;
use plotly::layout::Axis;
use plotly::layout::Layout;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::quant::calibration::sabr::SabrParams;
use crate::quant::pricing::sabr::alpha_from_atm_vol;
use crate::quant::pricing::sabr::bs_price_fx;
use crate::quant::pricing::sabr::forward_fx;
use crate::quant::pricing::sabr::fx_delta_from_forward;
use crate::quant::pricing::sabr::hagan_implied_vol;
use crate::quant::pricing::sabr::model_price_hagan_general;

#[derive(Clone, Copy, Debug)]
pub struct SabrSmileQuotes {
  /// Time to maturity in years
  pub tau: f64,
  /// ATM vol (decimal)
  pub sigma_atm: f64,
  /// Risk-reversal (decimal): sigma(25d call) - sigma(25d put)
  pub sigma_rr: f64,
  /// Butterfly (decimal): average of call/put away-from-atm vol premium over ATM
  pub sigma_bf: f64,
}

#[derive(Clone, Debug)]
pub struct SabrSmileCalibrator {
  /// Spot FX rate S
  pub s: f64,
  /// Domestic rate r_d
  pub r_d: f64,
  /// Foreign rate r_f
  pub r_f: f64,
  /// CEV exponent (0 = normal, 1 = lognormal)
  pub beta: f64,
  /// Quotes for one tenor
  pub quotes: SabrSmileQuotes,
}

impl SabrSmileCalibrator {
  pub fn new(s: f64, r_d: f64, r_f: f64, beta: f64, quotes: SabrSmileQuotes) -> Self {
    Self {
      s,
      r_d,
      r_f,
      beta,
      quotes,
    }
  }
}

#[derive(Clone, Debug)]
pub struct SabrSmileResult {
  /// ATM strike (= forward).
  pub k_atm: f64,
  /// Call strike corresponding to risk-reversal quote.
  pub k_rr_call: f64,
  /// Put strike corresponding to risk-reversal quote.
  pub k_rr_put: f64,
  /// Call strike corresponding to butterfly quote.
  pub k_bf_call: f64,
  /// Put strike corresponding to butterfly quote.
  pub k_bf_put: f64,
  /// Model parameter set (calibrated output).
  pub params: SabrParams,
  /// Final objective-function value.
  pub objective: f64,
  /// Indicates whether optimization converged successfully.
  pub success: bool,
}

fn rr_sigma(
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

fn bf_premium_mismatch(
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
const NVARS: usize = 6;

/// Problem definition for argmin optimization
#[derive(Clone)]
struct SabrSmileProblem {
  s: f64,
  r_d: f64,
  r_f: f64,
  tau: f64,
  beta: f64,
  sigma_atm: f64,
  sigma_rr: f64,
  sigma_bf: f64,
  bounds_lo: [f64; NVARS],
  bounds_hi: [f64; NVARS],
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

    // RR vol match
    let term_rr =
      (rr_sigma(k_rr_c, k_rr_p, f, self.tau, alpha, self.beta, nu, rho) - self.sigma_rr)
        .powi(2);

    // RR 25-delta constraints (smile vol for delta)
    let call_sigma_rr = hagan_implied_vol(k_rr_c, f, self.tau, alpha, self.beta, nu, rho);
    let put_sigma_rr = hagan_implied_vol(k_rr_p, f, self.tau, alpha, self.beta, nu, rho);
    let d_call_rr = fx_delta_from_forward(k_rr_c, f, call_sigma_rr, self.tau, self.r_f, 1.0);
    let d_put_rr = fx_delta_from_forward(k_rr_p, f, put_sigma_rr, self.tau, self.r_f, -1.0);
    let term_rr_delta = (d_call_rr - 0.25).powi(2) + (d_put_rr + 0.25).powi(2);

    // BF premium match (market strangle: σ_ref = σ_ATM + σ_BF for delta)
    let sigma_ref = self.sigma_atm + self.sigma_bf;
    let term_bf = bf_premium_mismatch(
      self.s, k_bf_c, k_bf_p, self.r_d, self.r_f, self.tau, alpha, self.beta, nu, rho,
      sigma_ref,
    )
    .powi(2);

    // BF 25-delta constraints (σ_ref for delta)
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
fn basin_hopping_opt(
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
    for i in 0..NVARS {
      x_trial[i] += rng.random_range(-stepsize..stepsize);
      x_trial[i] = x_trial[i].clamp(problem.bounds_lo[i], problem.bounds_hi[i]);
    }

    let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9).unwrap();
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

impl SabrSmileCalibrator {
  pub fn calibrate(&self) -> SabrSmileResult {
    // Bounds: [k_rr_c, k_rr_p, k_bf_c, k_bf_p, nu, rho]
    let lo = [1.0, 1.0, 1.0, 1.0, 0.01, -0.99];
    let hi = [10.0, 10.0, 10.0, 10.0, 10.0, 0.99];

    let s = self.s;
    let tau = self.quotes.tau;
    let sigma_atm = self.quotes.sigma_atm;
    let sigma_rr = self.quotes.sigma_rr;
    let sigma_bf = self.quotes.sigma_bf;
    let f = forward_fx(s, tau, self.r_d, self.r_f);

    // Initial guess: [k_rr_c, k_rr_p, k_bf_c, k_bf_p, nu, rho]
    let x0 = [s + 0.1, s - 0.1, s + 0.1, s - 0.1, 0.6, 0.5];

    let niter = if (tau - (1.0 / 365.0)).abs() < 1e-12 {
      1000
    } else {
      100
    };

    let problem = SabrSmileProblem {
      s,
      r_d: self.r_d,
      r_f: self.r_f,
      tau,
      beta: self.beta,
      sigma_atm,
      sigma_rr,
      sigma_bf,
      bounds_lo: lo,
      bounds_hi: hi,
    };

    let (x_best, f_best) = basin_hopping_opt(x0, niter, 0.0005, &problem);

    let rho = x_best[5].clamp(-0.99, 0.99);
    let nu = x_best[4].max(0.01);
    let alpha = alpha_from_atm_vol(sigma_atm, f, tau, self.beta, rho, nu);

    SabrSmileResult {
      k_atm: f,
      k_rr_call: x_best[0],
      k_rr_put: x_best[1],
      k_bf_call: x_best[2],
      k_bf_put: x_best[3],
      params: SabrParams {
        alpha,
        beta: self.beta,
        nu,
        rho,
      },
      objective: f_best,
      success: f_best.is_finite(),
    }
  }

  /// Build and write an HTML plot of the SABR smile using calibrated params and sensible K range.
  pub fn plot(&self, res: &SabrSmileResult) {
    let tau = self.quotes.tau;
    let (k_min, k_max) = (
      (res.k_rr_put.min(res.k_bf_put)).min(res.k_atm) * 0.95,
      (res.k_rr_call.max(res.k_bf_call)).max(res.k_atm) * 1.05,
    );
    let n = 200usize;
    let xs: Vec<f64> = (0..n)
      .map(|i| k_min + (k_max - k_min) * (i as f64) / ((n - 1) as f64))
      .collect();
    let fwd = forward_fx(self.s, tau, self.r_d, self.r_f);
    let ys: Vec<f64> = xs
      .iter()
      .map(|&k| {
        hagan_implied_vol(
          k,
          fwd,
          tau,
          res.params.alpha,
          res.params.beta,
          res.params.nu,
          res.params.rho,
        )
      })
      .collect();

    let trace = Scatter::new(xs, ys)
      .mode(Mode::Lines)
      .name(format!("SABR beta={}", self.beta));
    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(
      Layout::new()
        .title(Title::from("SABR Smile"))
        .x_axis(Axis::new().title("Strike price"))
        .y_axis(Axis::new().title("Implied volatility")),
    );
    plot.show();
  }

  /// Returns the vector of calibration results in the same order as `cases`.
  pub fn calibrate_and_plot_many(
    s: f64,
    r_d: f64,
    r_f: f64,
    beta: f64,
    cases: &[(&str, SabrSmileQuotes)],
  ) -> Vec<SabrSmileResult> {
    let mut results: Vec<SabrSmileResult> = Vec::with_capacity(cases.len());
    for (_, q) in cases.iter() {
      let calib = SabrSmileCalibrator::new(s, r_d, r_f, beta, *q);
      results.push(calib.calibrate());
    }

    let colors = [
      "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ];

    let mut plot = Plot::new();

    // Each tenor gets its own x-range based on its calibrated strikes.
    // This prevents the Hagan approximation from blowing up at strikes
    // far outside the calibration region (especially for short tenors
    // with extreme ρ).
    for (i, (label, q)) in cases.iter().enumerate() {
      let res = &results[i];
      let fwd = forward_fx(s, q.tau, r_d, r_f);

      let lo = res
        .k_rr_put
        .min(res.k_bf_put)
        .min(res.k_atm);
      let hi = res
        .k_rr_call
        .max(res.k_bf_call)
        .max(res.k_atm);
      let pad = (hi - lo) * 0.25;
      let k_lo = (lo - pad).max(1e-6);
      let k_hi = hi + pad;

      let n = 200usize;
      let xs: Vec<f64> = (0..n)
        .map(|j| k_lo + (k_hi - k_lo) * (j as f64) / ((n - 1) as f64))
        .collect();

      // Cap vols at 3× ATM to filter Hagan blow-ups
      let vol_cap = q.sigma_atm * 3.0;
      let ys: Vec<f64> = xs
        .iter()
        .map(|&k| {
          let v = hagan_implied_vol(
            k,
            fwd,
            q.tau,
            res.params.alpha,
            res.params.beta,
            res.params.nu,
            res.params.rho,
          );
          if v > 0.0 && v < vol_cap {
            v
          } else {
            f64::NAN
          }
        })
        .collect();

      let color = colors[i % colors.len()];
      let trace = Scatter::new(xs, ys)
        .mode(Mode::Lines)
        .name(*label)
        .line(plotly::common::Line::new().color(color));
      plot.add_trace(trace);

      // Markers for calibrated strikes (ATM, RR, BF)
      let strikes = vec![
        res.k_atm,
        res.k_rr_call,
        res.k_rr_put,
        res.k_bf_call,
        res.k_bf_put,
      ];
      let marker_vols: Vec<f64> = strikes
        .iter()
        .map(|&k| {
          hagan_implied_vol(
            k,
            fwd,
            q.tau,
            res.params.alpha,
            res.params.beta,
            res.params.nu,
            res.params.rho,
          )
        })
        .collect();
      let marker = Scatter::new(strikes, marker_vols)
        .mode(Mode::Markers)
        .marker(plotly::common::Marker::new().color(color).size(8))
        .show_legend(false);
      plot.add_trace(marker);
    }

    plot.set_layout(
      Layout::new()
        .title(Title::from(&format!(
          "SABR Smile (β={}) — ATM/RR/BF calibrated strikes",
          beta
        )))
        .x_axis(Axis::new().title("Strike"))
        .y_axis(Axis::new().title("Implied vol")),
    );
    plot.show();

    results
  }
}

/// Solve for strike K such that the FX delta equals the desired value under SABR (general β).
pub fn strike_for_delta(
  s: f64,
  r_d: f64,
  r_f: f64,
  tau: f64,
  params: SabrParams,
  target_delta: f64,
  phi: f64,
) -> f64 {
  let fwd = forward_fx(s, tau, r_d, r_f);
  let mut a = s * 0.1;
  let mut b = s * 10.0;
  let fa = |k: f64| -> f64 {
    let sig = hagan_implied_vol(k, fwd, tau, params.alpha, params.beta, params.nu, params.rho);
    let d = fx_delta_from_forward(k, fwd, sig, tau, r_f, phi);
    (d - target_delta).powi(2)
  };

  let gr = 0.5 * (5.0f64.sqrt() - 1.0);
  let mut c = b - gr * (b - a);
  let mut d = a + gr * (b - a);
  let mut fc = fa(c);
  let mut fd = fa(d);
  for _ in 0..200 {
    if fc < fd {
      b = d;
      d = c;
      fd = fc;
      c = b - gr * (b - a);
      fc = fa(c);
    } else {
      a = c;
      c = d;
      fc = fd;
      d = a + gr * (b - a);
      fd = fa(d);
    }
  }
  0.5 * (a + b)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_sabr_smile_calibrate() {
    let r_usd = 0.022_f64;
    let r_brl = 0.065_f64;
    let s = 3.724_f64;
    let beta = 1.0;

    let cases: [(&str, SabrSmileQuotes); 8] = [
      (
        "ON",
        SabrSmileQuotes {
          tau: 1.0 / 365.0,
          sigma_atm: 20.98 / 100.0,
          sigma_rr: 1.2 / 100.0,
          sigma_bf: 0.15 / 100.0,
        },
      ),
      (
        "1W",
        SabrSmileQuotes {
          tau: 7.0 / 365.0,
          sigma_atm: 13.91 / 100.0,
          sigma_rr: 1.3 / 100.0,
          sigma_bf: 0.20 / 100.0,
        },
      ),
      (
        "2W",
        SabrSmileQuotes {
          tau: 14.0 / 365.0,
          sigma_atm: 13.75 / 100.0,
          sigma_rr: 1.4 / 100.0,
          sigma_bf: 0.20 / 100.0,
        },
      ),
      (
        "1M",
        SabrSmileQuotes {
          tau: 30.0 / 365.0,
          sigma_atm: 14.24 / 100.0,
          sigma_rr: 1.5 / 100.0,
          sigma_bf: 0.22 / 100.0,
        },
      ),
      (
        "2M",
        SabrSmileQuotes {
          tau: 60.0 / 365.0,
          sigma_atm: 13.84 / 100.0,
          sigma_rr: 1.75 / 100.0,
          sigma_bf: 0.27 / 100.0,
        },
      ),
      (
        "3M",
        SabrSmileQuotes {
          tau: 90.0 / 365.0,
          sigma_atm: 13.82 / 100.0,
          sigma_rr: 2.0 / 100.0,
          sigma_bf: 0.32 / 100.0,
        },
      ),
      (
        "6M",
        SabrSmileQuotes {
          tau: 180.0 / 365.0,
          sigma_atm: 13.82 / 100.0,
          sigma_rr: 2.4 / 100.0,
          sigma_bf: 0.43 / 100.0,
        },
      ),
      (
        "1Y",
        SabrSmileQuotes {
          tau: 1.0,
          sigma_atm: 13.94 / 100.0,
          sigma_rr: 2.9 / 100.0,
          sigma_bf: 0.55 / 100.0,
        },
      ),
    ];

    // Calibrate and plot all in one figure
    let results =
      SabrSmileCalibrator::calibrate_and_plot_many(s, r_brl, r_usd, beta, &cases);

    for (i, ((label, q), res)) in cases.iter().zip(results.iter()).enumerate() {
      println!("\nTenor {} (T={:.4}):", label, q.tau);
      println!(
        "  K_ATM={:.6}, alpha={:.6}, beta={:.2}, nu={:.6}, rho={:.6}",
        res.k_atm, res.params.alpha, res.params.beta, res.params.nu, res.params.rho
      );
      println!("  Objective: {:.6e}", res.objective);
      assert!(res.success);
      assert!(res.objective < 1e-3, "Objective too large for tenor {}", i);
    }
  }
}

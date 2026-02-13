use argmin::core::CostFunction;
use argmin::core::Executor;
use argmin::core::Gradient;
use argmin::core::State;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use plotly::common::Mode;
use plotly::common::Title;
use plotly::layout::Axis;
use plotly::layout::Layout;
use plotly::Plot;
use plotly::Scatter;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

use crate::quant::calibration::sabr::SabrParams;
use crate::quant::pricing::sabr::bs_price_fx;
use crate::quant::pricing::sabr::forward_fx;
use crate::quant::pricing::sabr::fx_delta_from_forward;
use crate::quant::pricing::sabr::hagan_implied_vol_beta1;
use crate::quant::pricing::sabr::model_price_hagan;

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
  /// Quotes for one tenor
  pub quotes: SabrSmileQuotes,
}

impl SabrSmileCalibrator {
  pub fn new(s: f64, r_d: f64, r_f: f64, quotes: SabrSmileQuotes) -> Self {
    Self {
      s,
      r_d,
      r_f,
      quotes,
    }
  }
}

#[derive(Clone, Debug)]
pub struct SabrSmileResult {
  pub k_atm: f64,
  pub k_rr_call: f64,
  pub k_rr_put: f64,
  pub k_bf_call: f64,
  pub k_bf_put: f64,
  pub params: SabrParams,
  pub objective: f64,
  pub success: bool,
}

fn rr_sigma(k_call: f64, k_put: f64, f: f64, tau: f64, alpha: f64, nu: f64, rho: f64) -> f64 {
  let sc = hagan_implied_vol_beta1(k_call, f, tau, alpha, nu, rho);
  let sp = hagan_implied_vol_beta1(k_put, f, tau, alpha, nu, rho);
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
  nu: f64,
  rho: f64,
  sigma_ref: f64,
) -> f64 {
  let (mc, _) = model_price_hagan(s, k_call, r_d, r_f, tau, alpha, nu, rho);
  let (_, mp) = model_price_hagan(s, k_put, r_d, r_f, tau, alpha, nu, rho);
  let (bc, _) = bs_price_fx(s, k_call, r_d, r_f, tau, sigma_ref);
  let (_, bp) = bs_price_fx(s, k_put, r_d, r_f, tau, sigma_ref);
  (mc + mp) - (bc + bp)
}

fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
  if x < lo {
    lo
  } else if x > hi {
    hi
  } else {
    x
  }
}

/// Problem definition for argmin optimization
#[derive(Clone)]
struct SabrSmileProblem {
  s: f64,
  r_d: f64,
  r_f: f64,
  tau: f64,
  sigma_atm: f64,
  sigma_rr: f64,
  sigma_bf: f64,
  bounds_lo: [f64; 8],
  bounds_hi: [f64; 8],
}

impl CostFunction for SabrSmileProblem {
  // TODO: temp solution until argmin has ndarray@0.17 support
  type Param = Vec<f64>;
  type Output = f64;

  fn cost(&self, x: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
    let mut p = [0.0; 8];
    for i in 0..8 {
      let xi = x[i];
      let lo = self.bounds_lo[i];
      let hi = self.bounds_hi[i];
      p[i] = if xi < lo {
        lo
      } else if xi > hi {
        hi
      } else {
        xi
      };
    }

    let f = forward_fx(self.s, self.tau, self.r_d, self.r_f);
    let (k_atm, k_rr_c, k_rr_p, k_bf_c, k_bf_p, alpha, nu, rho) =
      (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);

    let sigma_atm_model = hagan_implied_vol_beta1(k_atm, f, self.tau, alpha, nu, rho);
    let term_atm = (sigma_atm_model - self.sigma_atm).powi(2);

    let term_rr = (rr_sigma(k_rr_c, k_rr_p, f, self.tau, alpha, nu, rho) - self.sigma_rr).powi(2);
    let call_sigma_rr = hagan_implied_vol_beta1(k_rr_c, f, self.tau, alpha, nu, rho);
    let put_sigma_rr = hagan_implied_vol_beta1(k_rr_p, f, self.tau, alpha, nu, rho);
    let d_call_rr = fx_delta_from_forward(k_rr_c, f, call_sigma_rr, self.tau, self.r_f, 1.0);
    let d_put_rr = fx_delta_from_forward(k_rr_p, f, put_sigma_rr, self.tau, self.r_f, -1.0);
    let term_rr_delta = (d_call_rr - 0.25).powi(2) + (d_put_rr + 0.25).powi(2);

    let sigma_ref = self.sigma_atm + self.sigma_bf;
    let term_bf = bf_premium_mismatch(
      self.s, k_bf_c, k_bf_p, self.r_d, self.r_f, self.tau, alpha, nu, rho, sigma_ref,
    )
    .powi(2);
    let d_call_bf = fx_delta_from_forward(k_bf_c, f, sigma_ref, self.tau, self.r_f, 1.0);
    let d_put_bf = fx_delta_from_forward(k_bf_p, f, sigma_ref, self.tau, self.r_f, -1.0);
    let term_bf_delta = (d_call_bf - 0.25).powi(2) + (d_put_bf + 0.25).powi(2);

    Ok(term_atm + term_rr + term_rr_delta + term_bf + term_bf_delta)
  }
}

impl Gradient for SabrSmileProblem {
  type Param = Vec<f64>;
  type Gradient = Vec<f64>;

  fn gradient(&self, x: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
    let mut grad = vec![0.0; x.len()];
    let eps = 1e-8;
    let f0 = self.cost(x)?;

    for i in 0..x.len() {
      let mut x_plus = x.clone();
      x_plus[i] += eps;
      let f_plus = self.cost(&x_plus)?;
      grad[i] = (f_plus - f0) / eps;
    }

    Ok(grad)
  }
}

// Basin-hopping with argmin L-BFGS
fn basin_hopping_opt(
  x0: [f64; 8],
  niter: usize,
  stepsize: f64,
  s: f64,
  r_d: f64,
  r_f: f64,
  tau: f64,
  sigma_atm: f64,
  sigma_rr: f64,
  sigma_bf: f64,
  bounds_lo: [f64; 8],
  bounds_hi: [f64; 8],
) -> ([f64; 8], f64) {
  let mut rng = StdRng::seed_from_u64(3);

  let problem = SabrSmileProblem {
    s,
    r_d,
    r_f,
    tau,
    sigma_atm,
    sigma_rr,
    sigma_bf,
    bounds_lo,
    bounds_hi,
  };

  let mut current_x = x0;
  let mut current_f = problem.cost(&x0.to_vec()).unwrap_or(f64::INFINITY);

  let mut best_x = current_x;
  let mut best_f = current_f;

  let temp = 1.0_f64;

  for _ in 0..niter {
    let mut x_trial = current_x;
    for i in 0..8 {
      x_trial[i] += rng.random_range(-stepsize..stepsize);
      x_trial[i] = clamp(x_trial[i], bounds_lo[i], bounds_hi[i]);
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
          current_x.copy_from_slice(&param[..8]);
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
    let lo = [1.0, 1.0, 1.0, 1.0, 1.0, 0.01, 0.01, -0.99];
    let hi = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.99];

    let s = self.s;
    let tau = self.quotes.tau;
    let sigma_atm = self.quotes.sigma_atm;
    let sigma_rr = self.quotes.sigma_rr;
    let sigma_bf = self.quotes.sigma_bf;

    let x0 = [s + 0.2, s + 0.1, s - 0.1, s + 0.1, s - 0.1, 0.11, 0.6, 0.5];

    let niter = if (tau - (1.0 / 365.0)).abs() < 1e-12 {
      1000
    } else {
      100
    };

    let (x_best, f_best) = basin_hopping_opt(
      x0, niter, 0.0005, s, self.r_d, self.r_f, tau, sigma_atm, sigma_rr, sigma_bf, lo, hi,
    );

    SabrSmileResult {
      k_atm: x_best[0],
      k_rr_call: x_best[1],
      k_rr_put: x_best[2],
      k_bf_call: x_best[3],
      k_bf_put: x_best[4],
      params: SabrParams {
        alpha: x_best[5],
        nu: x_best[6],
        rho: x_best[7],
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
        hagan_implied_vol_beta1(k, fwd, tau, res.params.alpha, res.params.nu, res.params.rho)
      })
      .collect();

    let trace = Scatter::new(xs, ys).mode(Mode::Lines).name("SABR beta=1");
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
    cases: &[(&str, SabrSmileQuotes)],
  ) -> Vec<SabrSmileResult> {
    // Calibrate each case
    let mut results: Vec<SabrSmileResult> = Vec::with_capacity(cases.len());
    for (_, q) in cases.iter() {
      let calib = SabrSmileCalibrator::new(s, r_d, r_f, *q);
      results.push(calib.calibrate());
    }

    let mut k_call_delta_p10: Vec<f64> = Vec::with_capacity(cases.len());
    let mut k_put_delta_m10: Vec<f64> = Vec::with_capacity(cases.len());

    for (i, (_, q)) in cases.iter().enumerate() {
      let res = &results[i];
      let kc = strike_for_delta(s, r_d, r_f, q.tau, res.params, 0.1, 1.0);
      let kp = strike_for_delta(s, r_d, r_f, q.tau, res.params, -0.1, -1.0);
      k_call_delta_p10.push(kc);
      k_put_delta_m10.push(kp);
    }

    let k_min = k_put_delta_m10
      .into_iter()
      .fold(f64::INFINITY, |a, b| a.min(b));
    let k_max = k_call_delta_p10
      .into_iter()
      .fold(f64::INFINITY, |a, b| a.min(b));

    let k_lo = (k_min - 1.0).max(1e-6);
    let k_hi = k_max + 1.0;
    let step = 0.01;
    let n = (((k_hi - k_lo) / step).ceil() as usize).max(2);
    let xs: Vec<f64> = (0..n).map(|i| k_lo + (i as f64) * step).collect();

    // Build combined plot
    let mut plot = Plot::new();
    for (i, (label, q)) in cases.iter().enumerate() {
      let res = &results[i];
      let fwd = forward_fx(s, q.tau, r_d, r_f);
      let ys: Vec<f64> = xs
        .iter()
        .map(|&k| {
          hagan_implied_vol_beta1(
            k,
            fwd,
            q.tau,
            res.params.alpha,
            res.params.nu,
            res.params.rho,
          )
        })
        .collect();
      let trace = Scatter::new(xs.clone(), ys).mode(Mode::Lines).name(*label);
      plot.add_trace(trace);
    }

    plot.set_layout(
      Layout::new()
        .title(Title::from(
          "Relationships between K and σ for different tenors (SABR, β=1)",
        ))
        .x_axis(Axis::new().title("Strike price"))
        .y_axis(
          Axis::new()
            .title("Implied volatility")
            .range(vec![0.0, 0.3]),
        ),
    );
    plot.show();

    results
  }
}

/// Solve for strike K such that the FX delta equals the desired value under SABR(Hagan beta=1).
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
    let sig = hagan_implied_vol_beta1(k, fwd, tau, params.alpha, params.nu, params.rho);
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
    let results = SabrSmileCalibrator::calibrate_and_plot_many(s, r_brl, r_usd, &cases);

    for (i, ((label, q), res)) in cases.iter().zip(results.iter()).enumerate() {
      println!("\nTenor {} (T={:.4}):", label, q.tau);
      println!(
        "  K_ATM={:.6}, alpha={:.6}, nu={:.6}, rho={:.6}",
        res.k_atm, res.params.alpha, res.params.nu, res.params.rho
      );
      println!("  Objective: {:.6e}", res.objective);
      assert!(res.success);
      assert!(res.objective < 1e-3, "Objective too large for tenor {}", i);
    }
  }
}

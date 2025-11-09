use impl_new_derive::ImplNew;
use rand::{thread_rng, Rng};

use plotly::common::Mode;
use plotly::{Plot, Scatter};

use crate::quant::calibration::sabr::SabrParams;
use crate::quant::pricing::sabr::{
  bs_price_fx, forward_fx, fx_delta_from_forward, hagan_implied_vol_beta1, model_price_hagan,
};

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

#[derive(ImplNew, Clone, Debug)]
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

/// Params layout: [K_ATM, K_RR_Call, K_RR_Put, K_BF_Call, K_BF_Put, alpha, nu, rho]
fn objective_all(
  x: &[f64],
  s: f64,
  r_d: f64,
  r_f: f64,
  tau: f64,
  sigma_atm: f64,
  sigma_rr: f64,
  sigma_bf: f64,
  bounds_lo: &[f64; 8],
  bounds_hi: &[f64; 8],
) -> f64 {
  let mut p = [0.0; 8];
  for i in 0..8 {
    p[i] = clamp(x[i], bounds_lo[i], bounds_hi[i]);
  }

  let f = forward_fx(s, tau, r_d, r_f);
  let (k_atm, k_rr_c, k_rr_p, k_bf_c, k_bf_p, alpha, nu, rho) =
    (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);

  let sigma_atm_model = hagan_implied_vol_beta1(k_atm, f, tau, alpha, nu, rho);
  let term_atm = (sigma_atm_model - sigma_atm).powi(2);

  let term_rr = (rr_sigma(k_rr_c, k_rr_p, f, tau, alpha, nu, rho) - sigma_rr).powi(2);
  let call_sigma_rr = hagan_implied_vol_beta1(k_rr_c, f, tau, alpha, nu, rho);
  let put_sigma_rr = hagan_implied_vol_beta1(k_rr_p, f, tau, alpha, nu, rho);
  let d_call_rr = fx_delta_from_forward(k_rr_c, f, call_sigma_rr, tau, r_f, 1.0);
  let d_put_rr = fx_delta_from_forward(k_rr_p, f, put_sigma_rr, tau, r_f, -1.0);
  let term_rr_delta = (d_call_rr - 0.25).powi(2) + (d_put_rr + 0.25).powi(2);

  let sigma_ref = sigma_atm + sigma_bf;
  let term_bf =
    bf_premium_mismatch(s, k_bf_c, k_bf_p, r_d, r_f, tau, alpha, nu, rho, sigma_ref).powi(2);
  let d_call_bf = fx_delta_from_forward(k_bf_c, f, sigma_ref, tau, r_f, 1.0);
  let d_put_bf = fx_delta_from_forward(k_bf_p, f, sigma_ref, tau, r_f, -1.0);
  let term_bf_delta = (d_call_bf - 0.25).powi(2) + (d_put_bf + 0.25).powi(2);

  let penalty_bounds: f64 = x
    .iter()
    .enumerate()
    .map(|(i, &xi)| {
      let mut pen = 0.0;
      if xi < bounds_lo[i] {
        pen += (bounds_lo[i] - xi).powi(2);
      }
      if xi > bounds_hi[i] {
        pen += (xi - bounds_hi[i]).powi(2);
      }
      pen
    })
    .sum();

  term_atm + term_rr + term_rr_delta + term_bf + term_bf_delta + 1e3 * penalty_bounds
}

fn basin_hopping_opt(
  x0: [f64; 8],
  n_restarts: usize,
  iters_per_start: usize,
  step: f64,
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
  let mut rng = thread_rng();
  let mut best_x = x0;
  let mut best_f = objective_all(
    &x0, s, r_d, r_f, tau, sigma_atm, sigma_rr, sigma_bf, &bounds_lo, &bounds_hi,
  );

  for r in 0..n_restarts {
    let mut x = x0;
    if r > 0 {
      for i in 0..8 {
        let width = (bounds_hi[i] - bounds_lo[i]) * 0.1;
        x[i] = clamp(
          x[i] + rng.gen_range(-width..width),
          bounds_lo[i],
          bounds_hi[i],
        );
      }
    }

    let mut f_cur = objective_all(
      &x, s, r_d, r_f, tau, sigma_atm, sigma_rr, sigma_bf, &bounds_lo, &bounds_hi,
    );
    for _ in 0..iters_per_start {
      let mut x_new = x;
      for i in 0..8 {
        x_new[i] = clamp(
          x_new[i] + rng.gen_range(-step..step),
          bounds_lo[i],
          bounds_hi[i],
        );
      }
      let f_new = objective_all(
        &x_new, s, r_d, r_f, tau, sigma_atm, sigma_rr, sigma_bf, &bounds_lo, &bounds_hi,
      );
      if f_new <= f_cur {
        x = x_new;
        f_cur = f_new;
      }

      if rng.gen_bool(0.2) {
        let mut improved = true;
        let mut tries = 0;
        while improved && tries < 5 {
          improved = false;
          tries += 1;
          for i in 0..8 {
            let mut cand = x;
            cand[i] = clamp(cand[i] + step, bounds_lo[i], bounds_hi[i]);
            let f1 = objective_all(
              &cand, s, r_d, r_f, tau, sigma_atm, sigma_rr, sigma_bf, &bounds_lo, &bounds_hi,
            );
            if f1 < f_cur {
              x = cand;
              f_cur = f1;
              improved = true;
              continue;
            }
            cand[i] = clamp(x[i] - step, bounds_lo[i], bounds_hi[i]);
            let f2 = objective_all(
              &cand, s, r_d, r_f, tau, sigma_atm, sigma_rr, sigma_bf, &bounds_lo, &bounds_hi,
            );
            if f2 < f_cur {
              x = cand;
              f_cur = f2;
              improved = true;
            }
          }
        }
      }
    }

    if f_cur < best_f {
      best_f = f_cur;
      best_x = x;
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

    let (x_best, f_best) = basin_hopping_opt(
      x0,
      if (tau - (1.0 / 365.0)).abs() < 1e-12 {
        25
      } else {
        10
      },
      if (tau - (1.0 / 365.0)).abs() < 1e-12 {
        1500
      } else {
        800
      },
      5e-4,
      s,
      self.r_d,
      self.r_f,
      tau,
      sigma_atm,
      sigma_rr,
      sigma_bf,
      lo,
      hi,
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
    plot.show();
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
    let table = [
      (1.0 / 365.0, 20.98, 1.2, 0.15),
      (7.0 / 365.0, 13.91, 1.3, 0.2),
      (14.0 / 365.0, 13.75, 1.4, 0.2),
      (30.0 / 365.0, 14.24, 1.5, 0.22),
      (60.0 / 365.0, 13.84, 1.75, 0.27),
      (90.0 / 365.0, 13.82, 2.0, 0.32),
      (180.0 / 365.0, 13.82, 2.4, 0.43),
      (1.0, 13.94, 2.9, 0.55),
    ];

    for (tau, atm_bps, rr_bps, bf_bps) in table {
      let quotes = SabrSmileQuotes {
        tau,
        sigma_atm: atm_bps / 100.0,
        sigma_rr: rr_bps / 100.0,
        sigma_bf: bf_bps / 100.0,
      };
      let calib = SabrSmileCalibrator::new(s, r_brl, r_usd, quotes);
      let res = calib.calibrate();
      println!("Calibration result: {:?}", res);
      assert!(res.success);
      calib.plot(&res);
    }
  }
}

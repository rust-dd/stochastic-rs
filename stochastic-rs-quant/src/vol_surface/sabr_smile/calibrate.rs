use plotly::Plot;
use plotly::Scatter;
use plotly::common::Mode;
use plotly::common::Title;
use plotly::layout::Axis;
use plotly::layout::Layout;

use super::objective::NVARS;
use super::objective::SabrSmileProblem;
use super::objective::basin_hopping_opt;
use super::types::SabrSmileCalibrator;
use super::types::SabrSmileQuotes;
use super::types::SabrSmileResult;
use crate::calibration::sabr::SabrParams;
use crate::pricing::sabr::alpha_from_atm_vol;
use crate::pricing::sabr::forward_fx;
use crate::pricing::sabr::fx_delta_from_forward;
use crate::pricing::sabr::hagan_implied_vol;

impl SabrSmileCalibrator {
  pub fn calibrate(&self) -> SabrSmileResult {
    // Bounds: [k_rr_c, k_rr_p, k_bf_c, k_bf_p, nu, rho]
    // Strike bounds default to (s * 0.5, s * 2.0); override via
    // `with_strike_bounds` for non-FX underlyings.
    let lo = [
      self.strike_lo,
      self.strike_lo,
      self.strike_lo,
      self.strike_lo,
      0.01,
      -0.99,
    ];
    let hi = [
      self.strike_hi,
      self.strike_hi,
      self.strike_hi,
      self.strike_hi,
      10.0,
      0.99,
    ];

    let s = self.s;
    let tau = self.quotes.tau;
    let sigma_atm = self.quotes.sigma_atm;
    let sigma_rr = self.quotes.sigma_rr;
    let sigma_bf = self.quotes.sigma_bf;
    let f = forward_fx(s, tau, self.r_d, self.r_f);

    let x0: [f64; NVARS] = [s + 0.1, s - 0.1, s + 0.1, s - 0.1, 0.6, 0.5];

    let niter = if tau < self.short_tenor_threshold {
      self.short_tenor_iters
    } else {
      self.long_tenor_iters
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
      success: f_best.is_finite() && f_best < self.success_tol,
    }
  }

  /// Build and write an HTML plot of the Sabr smile using calibrated params and sensible K range.
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
      .name(format!("Sabr beta={}", self.beta));
    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.set_layout(
      Layout::new()
        .title(Title::from("Sabr Smile"))
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

    // Each tenor gets its own x-range based on its calibrated strikes — the
    // Hagan approximation can blow up at strikes far outside the calibration
    // region (especially short tenors with extreme ρ).
    for (i, (label, q)) in cases.iter().enumerate() {
      let res = &results[i];
      let fwd = forward_fx(s, q.tau, r_d, r_f);

      let lo = res.k_rr_put.min(res.k_bf_put).min(res.k_atm);
      let hi = res.k_rr_call.max(res.k_bf_call).max(res.k_atm);
      let pad = (hi - lo) * 0.25;
      let k_lo = (lo - pad).max(1e-6);
      let k_hi = hi + pad;

      let n = 200usize;
      let xs: Vec<f64> = (0..n)
        .map(|j| k_lo + (k_hi - k_lo) * (j as f64) / ((n - 1) as f64))
        .collect();

      // Cap vols at 3× ATM to filter Hagan blow-ups.
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
          if v > 0.0 && v < vol_cap { v } else { f64::NAN }
        })
        .collect();

      let color = colors[i % colors.len()];
      let trace = Scatter::new(xs, ys)
        .mode(Mode::Lines)
        .name(*label)
        .line(plotly::common::Line::new().color(color));
      plot.add_trace(trace);

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
          "Sabr Smile (β={}) — ATM/RR/BF calibrated strikes",
          beta
        )))
        .x_axis(Axis::new().title("Strike"))
        .y_axis(Axis::new().title("Implied vol")),
    );
    plot.show();

    results
  }
}

/// Solve for strike K such that the FX delta equals the desired value under Sabr (general β).
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
    let sig = hagan_implied_vol(
      k,
      fwd,
      tau,
      params.alpha,
      params.beta,
      params.nu,
      params.rho,
    );
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

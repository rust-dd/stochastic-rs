use std::collections::HashMap;

#[cfg(feature = "viz")]
use plotly::Plot;
#[cfg(feature = "viz")]
use plotly::Scatter;
#[cfg(feature = "viz")]
use plotly::common::Mode;
#[cfg(feature = "viz")]
use plotly::common::Title;
#[cfg(feature = "viz")]
use plotly::layout::Axis;
#[cfg(feature = "viz")]
use plotly::layout::Layout;

use super::objective::NVARS;
use super::objective::SabrSmileProblem;
use super::objective::basin_hopping_opt;
use super::types::SabrSmileCalibrator;
use super::types::SabrSmileQuotes;
use super::types::SmileStrikes;
use crate::CalibrationLossScore;
use crate::LossMetric;
use crate::QuantError;
use crate::calibration::Calibrated;
use crate::calibration::hw_swaption::check_finite_start;
use crate::pricing::sabr::SabrParams;
use crate::pricing::sabr::alpha_from_atm_vol;
use crate::pricing::sabr::forward_fx;
use crate::pricing::sabr::fx_delta_from_forward;
use crate::pricing::sabr::hagan_implied_vol;
use crate::traits::Calibrator;

impl Calibrator for SabrSmileCalibrator {
  type Params = SabrParams;
  type Output = Calibrated<SabrParams, SmileStrikes>;
  type Error = QuantError;

  /// `start` seeds `(ν, ρ)`, `α` follows from the ATM vol and `β` is the calibrator's; `Optimizer`
  /// when no searched point reaches the ATM vol. `loss` is the objective over the three quotes.
  fn calibrate(&self, start: Option<SabrParams>) -> Result<Self::Output, QuantError> {
    self.quotes.validate()?;
    if let Some(p) = start {
      check_finite_start(&[("nu", p.nu), ("rho", p.rho)])?;
    }
    // Bounds of [k_rr_c, k_rr_p, k_bf_c, k_bf_p, nu, rho].
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

    let call_start = (s + 0.1).clamp(self.strike_lo, self.strike_hi);
    let put_start = (s - 0.1).clamp(self.strike_lo, self.strike_hi);
    let [nu_start, rho_start] = start.map_or([0.6, 0.5], |p| [p.nu, p.rho]);
    let x0: [f64; NVARS] = [
      call_start, put_start, call_start, put_start, nu_start, rho_start,
    ];

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
      bounds_lo: lo.to_vec(),
      bounds_hi: hi.to_vec(),
    };

    let (x_best, f_best) = basin_hopping_opt(x0, niter, 0.0005, &problem);

    let rho = x_best[5];
    let nu = x_best[4];
    let alpha = alpha_from_atm_vol(sigma_atm, f, tau, self.beta, rho, nu);
    if !(alpha.is_finite() && alpha > 0.0) {
      return Err(QuantError::Optimizer {
        what: "sabr smile",
        reason: format!("no searched (nu, rho) reaches the ATM vol, objective {f_best:e}"),
      });
    }
    let mse = f_best / 3.0;
    let loss = CalibrationLossScore {
      scores: HashMap::from([(LossMetric::Mse, mse), (LossMetric::Rmse, mse.sqrt())]),
    };
    let params = SabrParams {
      alpha,
      beta: self.beta,
      rho,
      nu,
    };
    let strikes = SmileStrikes {
      k_atm: f,
      k_rr_call: x_best[0],
      k_rr_put: x_best[1],
      k_bf_call: x_best[2],
      k_bf_put: x_best[3],
    };
    let converged = f_best.is_finite() && f_best < self.success_tol;
    Ok(
      Calibrated::new(params, loss, converged)
        .with_message(format!("objective {f_best:e}"))
        .with_diagnostics(strikes),
    )
  }
}

impl SabrSmileCalibrator {
  /// One fit per case, in order; the first `Err` ends the run.
  pub fn calibrate_many(
    s: f64,
    r_d: f64,
    r_f: f64,
    beta: f64,
    cases: &[(&str, SabrSmileQuotes)],
  ) -> Result<Vec<Calibrated<SabrParams, SmileStrikes>>, QuantError> {
    cases
      .iter()
      .map(|(_, quotes)| SabrSmileCalibrator::new(s, r_d, r_f, beta, *quotes).calibrate(None))
      .collect()
  }

  /// Writes the fitted smile across its strikes to `path` as a standalone plotly HTML page.
  #[cfg(feature = "viz")]
  pub fn plot(
    &self,
    path: impl AsRef<std::path::Path>,
    result: &Calibrated<SabrParams, SmileStrikes>,
  ) -> std::io::Result<()> {
    let tau = self.quotes.tau;
    let k = &result.diagnostics;
    let (k_min, k_max) = (
      (k.k_rr_put.min(k.k_bf_put)).min(k.k_atm) * 0.95,
      (k.k_rr_call.max(k.k_bf_call)).max(k.k_atm) * 1.05,
    );
    let n = 200usize;
    let xs: Vec<f64> = (0..n)
      .map(|i| k_min + (k_max - k_min) * (i as f64) / ((n - 1) as f64))
      .collect();
    let fwd = forward_fx(self.s, tau, self.r_d, self.r_f);
    let ys: Vec<f64> = xs
      .iter()
      .map(|&k| hagan_implied_vol(k, fwd, tau, &result.params))
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
    std::fs::write(path, plot.to_html())
  }

  /// Writes every case's smile and its five strikes to `path` as one plotly HTML page; panics
  /// unless `results` holds one fit per case.
  #[cfg(feature = "viz")]
  pub fn plot_many(
    path: impl AsRef<std::path::Path>,
    s: f64,
    r_d: f64,
    r_f: f64,
    beta: f64,
    cases: &[(&str, SabrSmileQuotes)],
    results: &[Calibrated<SabrParams, SmileStrikes>],
  ) -> std::io::Result<()> {
    assert!(
      results.len() == cases.len(),
      "results must satisfy `results.len() == cases.len()`, got results.len() = {} for {} cases",
      results.len(),
      cases.len()
    );
    let colors = [
      "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ];

    let mut plot = Plot::new();

    // Each tenor gets its own x-range: Hagan's expansion can blow up far outside its strikes.
    for (i, ((label, q), res)) in cases.iter().zip(results).enumerate() {
      let k = &res.diagnostics;
      let fwd = forward_fx(s, q.tau, r_d, r_f);

      let lo = k.k_rr_put.min(k.k_bf_put).min(k.k_atm);
      let hi = k.k_rr_call.max(k.k_bf_call).max(k.k_atm);
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
          let v = hagan_implied_vol(k, fwd, q.tau, &res.params);
          if v > 0.0 && v < vol_cap { v } else { f64::NAN }
        })
        .collect();

      let color = colors[i % colors.len()];
      let trace = Scatter::new(xs, ys)
        .mode(Mode::Lines)
        .name(*label)
        .line(plotly::common::Line::new().color(color));
      plot.add_trace(trace);

      let strikes = vec![k.k_atm, k.k_rr_call, k.k_rr_put, k.k_bf_call, k.k_bf_put];
      let marker_vols: Vec<f64> = strikes
        .iter()
        .map(|&k| hagan_implied_vol(k, fwd, q.tau, &res.params))
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
    std::fs::write(path, plot.to_html())
  }
}

/// Solve for strike K such that the FX delta equals the desired value under Sabr (general β); NaN where the delta is not
/// computable (a negative or infinite `tau`, a non-positive spot), not a strike at the edge of the bracket.
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
    let sig = hagan_implied_vol(k, fwd, tau, &params);
    let d = fx_delta_from_forward(k, fwd, sig, tau, r_f, phi);
    (d - target_delta).powi(2)
  };

  let gr = 0.5 * (5.0f64.sqrt() - 1.0);
  let mut c = b - gr * (b - a);
  let mut d = a + gr * (b - a);
  let mut fc = fa(c);
  let mut fd = fa(d);
  for _ in 0..200 {
    if fc.is_nan() || fd.is_nan() {
      return f64::NAN;
    }
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

//! Validates Fgn construction correctness and generates diagnostic plots.
//!
//! ```sh
//! cargo run --release --example fgn_validate
//! ```
//!
//! Produces HTML plots in `target/`:
//! - `fgn_paths.html` — sample fBm paths for H=0.25, 0.5, 0.75
//! - `fgn_covariance.html` — empirical vs theoretical autocovariance
//! - `fgn_eigenvalues.html` — sqrt-eigenvalue spectrum
//! - `fgn_hurst_estimation.html` — variogram Hurst estimation across H values
use plotly::Layout;
use plotly::Plot;
use plotly::Scatter;
use plotly::common::Line;
use plotly::common::Mode;
use plotly::layout::Axis;
use stochastic_rs::simd_rng::{Deterministic, Unseeded};
use stochastic_rs::stochastic::noise::fgn::Fgn;
use stochastic_rs::traits::ProcessExt;

const N: usize = 4096;
const T_HORIZON: f64 = 1.0;
const M_PATHS: usize = 512;
const SEED_BASE: u64 = 42;

fn theoretical_covariance(h: f64, k: usize, dt: f64) -> f64 {
  if k == 0 {
    dt.powf(2.0 * h)
  } else {
    0.5
      * (((k + 1) as f64).powf(2.0 * h) - 2.0 * (k as f64).powf(2.0 * h)
        + ((k - 1) as f64).powf(2.0 * h))
      * dt.powf(2.0 * h)
  }
}

fn empirical_covariance(paths: &[Vec<f64>], mean: f64, lag: usize) -> f64 {
  let mut s = 0.0;
  let mut c = 0usize;
  for p in paths {
    for i in 0..(p.len() - lag) {
      s += (p[i] - mean) * (p[i + lag] - mean);
      c += 1;
    }
  }
  s / c as f64
}

fn generate_paths(h: f64, n: usize, t: f64, m: usize) -> Vec<Vec<f64>> {
  let fgn = Fgn::<f64, _>::new(h, n, Some(t), Unseeded);
  (0..m).map(|_| fgn.sample().to_vec()).collect()
}

fn fgn_to_fbm(increments: &[f64]) -> Vec<f64> {
  let mut fbm = Vec::with_capacity(increments.len() + 1);
  fbm.push(0.0);
  let mut s = 0.0;
  for &inc in increments {
    s += inc;
    fbm.push(s);
  }
  fbm
}

fn variogram_hurst_estimate(paths: &[Vec<f64>], max_lag: usize) -> f64 {
  let n_vals = paths[0].len();
  let mut log_lag = Vec::new();
  let mut log_var = Vec::new();
  for lag in 1..=max_lag {
    let mut s = 0.0;
    let mut c = 0usize;
    for p in paths {
      for i in 0..(n_vals - lag) {
        let d = p[i + lag] - p[i];
        s += d * d;
        c += 1;
      }
    }
    let v = s / c as f64;
    log_lag.push((lag as f64).ln());
    log_var.push(v.ln());
  }
  let n = log_lag.len() as f64;
  let sx: f64 = log_lag.iter().sum();
  let sy: f64 = log_var.iter().sum();
  let sxy: f64 = log_lag.iter().zip(&log_var).map(|(x, y)| x * y).sum();
  let sxx: f64 = log_lag.iter().map(|x| x * x).sum();
  let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  slope / 2.0
}

fn main() {
  let hursts = [0.25, 0.50, 0.75];
  let colors = [
    "#1f77b4".to_string(),
    "#ff7f0e".to_string(),
    "#2ca02c".to_string(),
  ];

  println!("Generating fBm sample paths...");
  plot_sample_paths(&hursts, &colors);

  println!("Validating covariance structure...");
  plot_covariance(&hursts, &colors);

  println!("Plotting eigenvalue spectrum...");
  plot_eigenvalues(&hursts, &colors);

  println!("Estimating Hurst exponents...");
  plot_hurst_estimation();

  println!("Done. Plots written to target/fgn_*.html");
}

fn plot_sample_paths(hursts: &[f64], colors: &[String]) {
  let mut plot = Plot::new();
  let dt = T_HORIZON / N as f64;
  let time: Vec<f64> = (0..=N).map(|i| i as f64 * dt).collect();

  for (i, &h) in hursts.iter().enumerate() {
    let fgn = Fgn::new(h, N, Some(T_HORIZON), Deterministic::new(SEED_BASE + i as u64));
    let inc = fgn.sample();
    let fbm = fgn_to_fbm(inc.as_slice().unwrap());
    let trace = Scatter::new(time.clone(), fbm)
      .name(format!("H = {h}"))
      .mode(Mode::Lines)
      .line(Line::new().color(colors[i].clone()).width(1.2));
    plot.add_trace(trace);
  }

  let layout = Layout::new()
    .title("Fractional Brownian Motion sample paths")
    .x_axis(Axis::new().title("t"))
    .y_axis(Axis::new().title("B_H(t)"));
  plot.set_layout(layout);
  plot.write_html("target/fgn_paths.html");
}

fn plot_covariance(hursts: &[f64], colors: &[String]) {
  let mut plot = Plot::new();
  let max_lag = 30;
  let dt = T_HORIZON / N as f64;
  let lags: Vec<f64> = (0..=max_lag).map(|k| k as f64).collect();

  for (i, &h) in hursts.iter().enumerate() {
    let paths = generate_paths(h, N, T_HORIZON, M_PATHS);
    let all_vals: Vec<f64> = paths.iter().flatten().copied().collect();
    let mean = all_vals.iter().sum::<f64>() / all_vals.len() as f64;

    let emp: Vec<f64> = (0..=max_lag)
      .map(|k| empirical_covariance(&paths, mean, k))
      .collect();
    let theory: Vec<f64> = (0..=max_lag)
      .map(|k| theoretical_covariance(h, k, dt))
      .collect();

    let max_err = emp
      .iter()
      .zip(&theory)
      .map(|(e, t)| ((e - t) / t.abs().max(1e-15)).abs())
      .fold(0.0_f64, f64::max);
    println!("  H={h:.2}: max relative covariance error = {max_err:.4}");

    plot.add_trace(
      Scatter::new(lags.clone(), emp)
        .name(format!("H={h} empirical"))
        .mode(Mode::Markers)
        .line(Line::new().color(colors[i].clone())),
    );
    plot.add_trace(
      Scatter::new(lags.clone(), theory)
        .name(format!("H={h} theory"))
        .mode(Mode::Lines)
        .line(
          Line::new()
            .color(colors[i].clone())
            .dash(plotly::common::DashType::Dash),
        ),
    );
  }

  let layout = Layout::new()
    .title("Fgn autocovariance: empirical vs theory")
    .x_axis(Axis::new().title("lag k"))
    .y_axis(Axis::new().title("Cov(k)"));
  plot.set_layout(layout);
  plot.write_html("target/fgn_covariance.html");
}

fn plot_eigenvalues(hursts: &[f64], colors: &[String]) {
  let mut plot = Plot::new();

  for (i, &h) in hursts.iter().enumerate() {
    let fgn = Fgn::<f64, _>::new(h, N, Some(T_HORIZON), Unseeded);
    let eig_vals: Vec<f64> = fgn.sqrt_eigenvalues.iter().copied().collect();
    let indices: Vec<f64> = (0..eig_vals.len()).map(|j| j as f64).collect();

    plot.add_trace(
      Scatter::new(indices, eig_vals)
        .name(format!("H = {h}"))
        .mode(Mode::Lines)
        .line(Line::new().color(colors[i].clone()).width(1.0)),
    );
  }

  let layout = Layout::new()
    .title("Fgn sqrt-eigenvalue spectrum")
    .x_axis(Axis::new().title("index"))
    .y_axis(Axis::new().title("sqrt(lambda)"));
  plot.set_layout(layout);
  plot.write_html("target/fgn_eigenvalues.html");
}

fn plot_hurst_estimation() {
  let test_hursts: Vec<f64> = (1..=19).map(|i| i as f64 * 0.05).collect();
  let mut estimated = Vec::new();

  for &h in &test_hursts {
    let paths = generate_paths(h, N, T_HORIZON, 256);
    let fbm_paths: Vec<Vec<f64>> = paths.iter().map(|p| fgn_to_fbm(p)).collect();
    let h_est = variogram_hurst_estimate(&fbm_paths, 64);
    let err = (h_est - h).abs();
    println!("  H={h:.2} -> H_est={h_est:.4} (err={err:.4})");
    estimated.push(h_est);
  }

  let mut plot = Plot::new();
  plot.add_trace(
    Scatter::new(test_hursts.clone(), estimated)
      .name("estimated H")
      .mode(Mode::Markers)
      .line(Line::new().color("#1f77b4")),
  );
  plot.add_trace(
    Scatter::new(test_hursts.clone(), test_hursts.clone())
      .name("H = H (ideal)")
      .mode(Mode::Lines)
      .line(
        Line::new()
          .color("#d62728")
          .dash(plotly::common::DashType::Dash),
      ),
  );

  let layout = Layout::new()
    .title("Fgn Hurst estimation: variogram method")
    .x_axis(Axis::new().title("true H"))
    .y_axis(Axis::new().title("estimated H"));
  plot.set_layout(layout);
  plot.write_html("target/fgn_hurst_estimation.html");
}

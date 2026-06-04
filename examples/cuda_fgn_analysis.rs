//! CUDA FGN analysis: covariance vector comparison (CPU vs CUDA) and FBM path plots.
//!
//! Run: cargo run --example cuda_fgn_analysis --features cuda-native --release
use plotly::common::Mode;
use plotly::layout::Axis;
use plotly::{Layout, Plot, Scatter};
use stochastic_rs::simd_rng::Unseeded;
use stochastic_rs::stochastic::device::CudaNative;
use stochastic_rs::stochastic::noise::fgn::Fgn;
use stochastic_rs::traits::ProcessExt;

fn lag_covariance(paths: &[Vec<f64>], mean: f64, lag: usize) -> f64 {
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

fn unit_lag_cov(h: f64, k: usize) -> f64 {
  if k == 0 {
    1.0
  } else {
    0.5
      * (((k + 1) as f64).powf(2.0 * h) - 2.0 * (k as f64).powf(2.0 * h)
        + ((k - 1) as f64).powf(2.0 * h))
  }
}

fn main() {
  let h = 0.72_f64;
  let n = 4096_usize;
  let t = 1.0_f64;
  let m = 2048_usize;
  let max_lag = 20_usize;

  let fgn = Fgn::<f64>::new(h, n, Some(t), Unseeded);
  let fgn_cuda = Fgn::<f64>::new(h, n, Some(t), Unseeded).on::<CudaNative>();

  // ── Covariance vector comparison ──────────────────────────────────────────
  println!("Generating {m} CPU paths (n={n}, H={h})...");
  let cpu_paths: Vec<Vec<f64>> = (0..m).map(|_| fgn.sample().to_vec()).collect();
  let cpu_vals: Vec<f64> = cpu_paths.iter().flatten().copied().collect();
  let cpu_mean = cpu_vals.iter().sum::<f64>() / cpu_vals.len() as f64;

  println!("Generating {m} CUDA paths...");
  let cuda_paths: Vec<Vec<f64>> = fgn_cuda
    .sample_par(m)
    .into_iter()
    .map(|p| p.to_vec())
    .collect();
  let cuda_vals: Vec<f64> = cuda_paths.iter().flatten().copied().collect();
  let cuda_mean = cuda_vals.iter().sum::<f64>() / cuda_vals.len() as f64;

  let dt = t / n as f64;
  let var_theory = dt.powf(2.0 * h);

  println!("\n{:>4} {:>14} {:>14} {:>14} {:>10} {:>10}", "lag", "theory", "CPU", "CUDA", "CPU/th", "CUDA/th");
  println!("{}", "-".repeat(78));

  let mut lags = Vec::new();
  let mut theory_vec = Vec::new();
  let mut cpu_cov_vec = Vec::new();
  let mut cuda_cov_vec = Vec::new();

  for k in 0..=max_lag {
    let cov_theory = var_theory * unit_lag_cov(h, k);
    let cov_cpu = lag_covariance(&cpu_paths, cpu_mean, k);
    let cov_cuda = lag_covariance(&cuda_paths, cuda_mean, k);

    let ratio_cpu = cov_cpu / cov_theory;
    let ratio_cuda = cov_cuda / cov_theory;

    println!(
      "{:>4} {:>14.8e} {:>14.8e} {:>14.8e} {:>10.4} {:>10.4}",
      k, cov_theory, cov_cpu, cov_cuda, ratio_cpu, ratio_cuda
    );

    lags.push(k as f64);
    theory_vec.push(cov_theory);
    cpu_cov_vec.push(cov_cpu);
    cuda_cov_vec.push(cov_cuda);
  }

  // Plot covariance vectors
  let mut cov_plot = Plot::new();
  cov_plot.add_trace(
    Scatter::new(lags.clone(), theory_vec)
      .name("Theory")
      .mode(Mode::LinesMarkers),
  );
  cov_plot.add_trace(
    Scatter::new(lags.clone(), cpu_cov_vec)
      .name("CPU (empirical)")
      .mode(Mode::Markers),
  );
  cov_plot.add_trace(
    Scatter::new(lags, cuda_cov_vec)
      .name("CUDA (empirical)")
      .mode(Mode::Markers),
  );
  cov_plot.set_layout(
    Layout::new()
      .title(format!("FGN Autocovariance: CPU vs CUDA vs Theory (H={h}, n={n}, m={m})"))
      .x_axis(Axis::new().title("Lag k"))
      .y_axis(Axis::new().title("Cov(X_i, X_{i+k})")),
  );
  cov_plot.write_html("fgn_covariance_comparison.html");
  println!("\nCovariance plot saved to fgn_covariance_comparison.html");

  // ── FBM path plots from CUDA ──────────────────────────────────────────────
  let fbm_n = 4096_usize;
  let fbm_paths = 8_usize;
  let hursts = [0.15, 0.25, 0.50, 0.75, 0.90];
  let time: Vec<f64> = (0..=fbm_n).map(|i| i as f64 / fbm_n as f64).collect();

  for &hh in &hursts {
    let mut plot = Plot::new();
    let fgn_h = Fgn::<f64>::new(hh, fbm_n, Some(1.0), Unseeded).on::<CudaNative>();
    let batch = fgn_h.sample_par(fbm_paths);

    for (j, inc) in batch.iter().enumerate() {
      let mut fbm = vec![0.0_f64; fbm_n + 1];
      for i in 0..fbm_n {
        fbm[i + 1] = fbm[i] + inc[i];
      }
      plot.add_trace(
        Scatter::new(time.clone(), fbm)
          .name(format!("path {}", j + 1))
          .opacity(0.8),
      );
    }

    let regime = if hh < 0.5 {
      "rough / anti-persistent"
    } else if hh == 0.5 {
      "standard Brownian motion"
    } else {
      "smooth / persistent"
    };
    plot.set_layout(
      Layout::new()
        .title(format!("CUDA fBM  H={hh:.2}  ({regime})"))
        .x_axis(Axis::new().title("t"))
        .y_axis(Axis::new().title("B_H(t)")),
    );
    let fname = format!("cuda_fbm_H{}.html", format!("{hh:.2}").replace('.', ""));
    plot.write_html(&fname);
    println!("Saved {fname}");
  }
}

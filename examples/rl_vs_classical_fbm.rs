//! Timing + sample-path comparison between the two fBM generators in
//! stochastic-rs:
//!
//! - **Classical** (`Fbm`): Mandelbrot–Van Ness fBM via Fgn with Davies–Harte
//!   circulant-embedding FFT + cumulative sum.
//! - **RL-Volterra** (`RlFBm`): Riemann–Liouville fBM via Bilokon–Wong (2026)
//!   modified fast algorithm (Markov-lift with Gauss–Laguerre kernel).
//!
//! Writes a Plotly HTML chart to `docs/rl_vs_classical_fbm.html` and prints
//! a timing table to stdout. Run with `cargo run --release --example rl_vs_classical_fbm`.
use std::hint::black_box;
use std::time::Instant;

use ndarray::Array1;
use plotly::Layout;
use plotly::Plot;
use plotly::Scatter;
use plotly::common::Line;
use plotly::common::Mode;
use plotly::common::Title;
use plotly::layout::Annotation;
use plotly::layout::Axis;
use plotly::layout::GridPattern;
use plotly::layout::LayoutGrid;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stochastic::process::fbm::Fbm;
use stochastic_rs::stochastic::rough::RlFBm;
use stochastic_rs::traits::ProcessExt;

fn bench_time<F: FnMut() -> R, R>(mut f: F, iters: usize) -> f64 {
  for _ in 0..iters.max(1) / 4 {
    black_box(f());
  }
  let start = Instant::now();
  for _ in 0..iters {
    black_box(f());
  }
  let dt = start.elapsed().as_secs_f64();
  dt / iters as f64
}

fn fmt_us(secs: f64) -> String {
  format!("{:>10.2} µs", secs * 1e6)
}

struct TimingPoint {
  n: usize,
  hurst: f64,
  t_classical_us: f64,
  t_rl_us: f64,
}

fn timing_table() -> Vec<TimingPoint> {
  let sizes = [256_usize, 1024, 4096, 16384];
  let hursts = [0.1_f64, 0.25, 0.45];
  let mut points = Vec::new();
  println!("\n=== Timing (median per sample, release build) ===");
  println!(
    "{:>8} {:>8}  {:>14}  {:>14}  {:>10}",
    "n", "H", "classical (DH)", "RL (Bilokon)", "RL/DH"
  );
  for &h in &hursts {
    for &n in &sizes {
      let classical = Fbm::<f64>::new(h, n, Some(1.0));
      let rl = RlFBm::<f64>::new(h, n, Some(1.0), None);
      let iters = (5_000_000 / n).clamp(40, 2_000);
      let t_classical = bench_time(|| classical.sample(), iters);
      let t_rl = bench_time(|| rl.sample(), iters);
      let ratio = t_rl / t_classical;
      println!(
        "{:>8} {:>8.2}  {}  {}  {:>10.2}x",
        n,
        h,
        fmt_us(t_classical),
        fmt_us(t_rl),
        ratio
      );
      points.push(TimingPoint {
        n,
        hurst: h,
        t_classical_us: t_classical * 1e6,
        t_rl_us: t_rl * 1e6,
      });
    }
  }
  points
}

fn write_timing_plot(points: &[TimingPoint], out_path: &str) {
  let hursts: Vec<f64> = {
    let mut hs: Vec<f64> = points.iter().map(|p| p.hurst).collect();
    hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    hs.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    hs
  };

  let palette = ["#2065c2", "#c23b27", "#1f9e4a"];
  let mut plot = Plot::new();
  for (i, h) in hursts.iter().enumerate() {
    let color = palette[i % palette.len()];
    let xs: Vec<f64> = points
      .iter()
      .filter(|p| (p.hurst - *h).abs() < 1e-9)
      .map(|p| p.n as f64)
      .collect();
    let classical_y: Vec<f64> = points
      .iter()
      .filter(|p| (p.hurst - *h).abs() < 1e-9)
      .map(|p| p.t_classical_us)
      .collect();
    let rl_y: Vec<f64> = points
      .iter()
      .filter(|p| (p.hurst - *h).abs() < 1e-9)
      .map(|p| p.t_rl_us)
      .collect();

    plot.add_trace(
      Scatter::new(xs.clone(), classical_y)
        .name(format!("Classical DH, H={h:.2}"))
        .mode(Mode::LinesMarkers)
        .line(Line::new().color(color).width(1.8)),
    );
    plot.add_trace(
      Scatter::new(xs, rl_y)
        .name(format!("RL Bilokon-Wong, H={h:.2}"))
        .mode(Mode::LinesMarkers)
        .line(
          Line::new()
            .color(color)
            .width(1.8)
            .dash(plotly::common::DashType::Dash),
        ),
    );
  }

  let layout = Layout::new()
    .title(Title::with_text(
      "Timing: fBM generation (log-log, Apple Silicon release build, single path)",
    ))
    .x_axis(
      Axis::new()
        .title(Title::with_text("n (path length)"))
        .type_(plotly::layout::AxisType::Log),
    )
    .y_axis(
      Axis::new()
        .title(Title::with_text("time per sample (µs)"))
        .type_(plotly::layout::AxisType::Log),
    )
    .height(620)
    .width(1100);
  plot.set_layout(layout);
  plot.write_html(out_path);
  println!("Wrote timing chart to {out_path}");
}

fn sample_paths(hurst: f64, n: usize, seed: u64) -> (Array1<f64>, Array1<f64>) {
  let classical = Fbm::<f64, Deterministic>::seeded(hurst, n, Some(1.0), seed);
  let rl = RlFBm::<f64, Deterministic>::seeded(hurst, n, Some(1.0), None, seed);
  (classical.sample(), rl.sample())
}

fn write_plot(out_path: &str) {
  let n = 1024_usize;
  let hursts_seeds: &[(f64, u64)] = &[(0.10, 42), (0.25, 7), (0.45, 123)];

  let mut plot = Plot::new();
  let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect();

  for (row, &(h, seed)) in hursts_seeds.iter().enumerate() {
    let (classical, rl) = sample_paths(h, n, seed);
    let axis_id = row + 1;

    let classical_trace = Scatter::new(xs.clone(), classical.to_vec())
      .name(format!("Classical (DH) H={h:.2}"))
      .mode(Mode::Lines)
      .line(Line::new().color("#2065c2").width(1.4))
      .x_axis(format!("x{axis_id}"))
      .y_axis(format!("y{axis_id}"));
    let rl_trace = Scatter::new(xs.clone(), rl.to_vec())
      .name(format!("RL-Volterra (BW) H={h:.2}"))
      .mode(Mode::Lines)
      .line(
        Line::new()
          .color("#c23b27")
          .width(1.4)
          .dash(plotly::common::DashType::Dash),
      )
      .x_axis(format!("x{axis_id}"))
      .y_axis(format!("y{axis_id}"));

    plot.add_trace(classical_trace);
    plot.add_trace(rl_trace);
  }

  let annotation = |text: &str, y_anchor: f64| -> Annotation {
    Annotation::new()
      .text(text)
      .x_ref("paper")
      .y_ref("paper")
      .x(0.02)
      .y(y_anchor)
      .show_arrow(false)
      .font(plotly::common::Font::new().size(13))
  };

  let layout = Layout::new()
    .title(Title::with_text(
      "fBM paths: Classical (Davies–Harte / MVN) vs RL-Volterra (Bilokon–Wong) — same seed",
    ))
    .grid(
      LayoutGrid::new()
        .rows(3)
        .columns(1)
        .pattern(GridPattern::Independent),
    )
    .x_axis(Axis::new().title(Title::with_text("t")))
    .y_axis(Axis::new().title(Title::with_text("W^H_t, H=0.10")))
    .x_axis2(Axis::new().title(Title::with_text("t")))
    .y_axis2(Axis::new().title(Title::with_text("W^H_t, H=0.25")))
    .x_axis3(Axis::new().title(Title::with_text("t")))
    .y_axis3(Axis::new().title(Title::with_text("W^H_t, H=0.45")))
    .annotations(vec![
      annotation("Top: H = 0.10 (very rough)", 1.02),
      annotation("Middle: H = 0.25", 0.66),
      annotation("Bottom: H = 0.45 (near-Brownian)", 0.30),
    ])
    .height(900)
    .width(1100)
    .show_legend(true);

  plot.set_layout(layout);

  std::fs::create_dir_all(
    std::path::Path::new(out_path)
      .parent()
      .unwrap_or(std::path::Path::new(".")),
  )
  .unwrap();
  plot.write_html(out_path);
  println!("\nWrote interactive HTML chart to {out_path}");
}

fn variance_comparison() {
  println!("\n=== Terminal variance Var[W^H_T], T=1 (MC vs theory) ===");
  println!("     Classical (MVN-fBM): Var = T^{{2H}}");
  println!("     RL-fBM (Bilokon-Wong): Var = T^{{2H}} / [2H · Γ(H+1/2)²]");
  let n = 512;
  let t = 1.0_f64;
  let samples = 500_usize;
  let hursts = [0.1_f64, 0.25, 0.45];

  println!(
    "\n{:>6} {:>14} {:>14} {:>14} {:>14}",
    "H", "MC classical", "theory class", "MC RL", "theory RL"
  );
  for &h in &hursts {
    let mut c_end = Vec::with_capacity(samples);
    let mut r_end = Vec::with_capacity(samples);
    for k in 0..samples {
      let classical = Fbm::<f64, Deterministic>::seeded(h, n, Some(t), 10_000 + k as u64);
      let rl = RlFBm::<f64, Deterministic>::seeded(h, n, Some(t), None, 10_000 + k as u64);
      c_end.push(*classical.sample().last().unwrap());
      r_end.push(*rl.sample().last().unwrap());
    }
    let mean_c = c_end.iter().sum::<f64>() / samples as f64;
    let mean_r = r_end.iter().sum::<f64>() / samples as f64;
    let var_c = c_end.iter().map(|v| (v - mean_c).powi(2)).sum::<f64>() / samples as f64;
    let var_r = r_end.iter().map(|v| (v - mean_r).powi(2)).sum::<f64>() / samples as f64;
    let theory_c = t.powf(2.0 * h);
    let g = stochastic_rs::distributions::special::gamma(h + 0.5);
    let theory_rl = t.powf(2.0 * h) / (2.0 * h * g * g);
    println!(
      "{:>6.2} {:>14.6} {:>14.6} {:>14.6} {:>14.6}",
      h, var_c, theory_c, var_r, theory_rl
    );
  }
  println!("\nNote: RL-fBM and classical MVN-fBM are DIFFERENT processes — their");
  println!("terminal variances diverge at small H. This is expected.");
}

fn main() {
  let timing_points = timing_table();
  variance_comparison();
  write_plot("docs/rl_vs_classical_fbm.html");
  write_timing_plot(&timing_points, "docs/rl_vs_classical_fbm_timing.html");
}

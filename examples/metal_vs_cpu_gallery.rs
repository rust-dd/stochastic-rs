//! The same processes on the CPU and on Metal, side by side.
//!
//! Samples a gallery of processes twice — once through the host sampler, once
//! through the Metal kernels — and writes one Plotly page per process into
//! `target/metal_vs_cpu/`, plus an index that links them. Each page holds
//! three panels: the CPU's paths, Metal's paths on the same scale, and the
//! two terminal laws overlaid.
//!
//! ```text
//! cargo run --release --example metal_vs_cpu_gallery --features metal
//! ```
//!
//! **The paths differ, and must.** The host draws from this crate's SIMD
//! ziggurat and the kernel hashes its own normals from `(path, step, seed)`,
//! so no path is shared. What has to agree is the law, which the third panel
//! shows and the table on stdout measures — in standard errors of the
//! estimate, not in a tolerance anyone picked. The same table prints what
//! each process says about itself first: `device_fallback()` names the reason
//! a configuration stays on the host, so a row that ran on the CPU twice is
//! visible rather than merely slow.
//!
//! Metal computes in single precision, so every process here is `f32`; that
//! is a compile-time property of the backend, not a choice this example
//! makes.

use std::fs;
use std::time::Instant;

use ndarray::Array1;
use plotly::Histogram;
use plotly::Layout;
use plotly::Plot;
use plotly::Scatter;
use plotly::common::Line;
use plotly::common::Mode;
use plotly::layout::Axis;
use plotly::layout::GridPattern;
use plotly::layout::LayoutGrid;
use stochastic_rs::simd_rng::Deterministic;
use stochastic_rs::stochastic::autoregressive::garch::Garch;
use stochastic_rs::stochastic::device::Cpu;
use stochastic_rs::stochastic::device::Metal;
use stochastic_rs::stochastic::diffusion::cir::Cir;
use stochastic_rs::stochastic::diffusion::gbm::Gbm;
use stochastic_rs::stochastic::diffusion::jacobi::Jacobi;
use stochastic_rs::stochastic::diffusion::ou::Ou;
use stochastic_rs::stochastic::diffusion::three_half::ThreeHalf;
use stochastic_rs::stochastic::interest::vasicek::Vasicek;
use stochastic_rs::stochastic::jump::nig::Nig;
use stochastic_rs::stochastic::jump::vg::Vg;
use stochastic_rs::stochastic::process::fbm::Fbm;
use stochastic_rs::stochastic::process::poisson::Poisson;
use stochastic_rs::stochastic::process::subordinator::alpha_stable::AlphaStableSubordinator;
use stochastic_rs::stochastic::volatility::sabr::Sabr;
use stochastic_rs::traits::ProcessExt;

/// Grid points per path.
const N: usize = 512;

/// Paths per side. Enough that a terminal mean and spread are stable, few
/// enough that the whole gallery runs in seconds.
const PATHS: usize = 4_000;

/// Paths drawn in each panel. More than this and the picture is ink.
const DRAWN: usize = 24;

/// The seed both sides are built from. They still draw different streams —
/// that is the point — but each side is reproducible.
const SEED: u64 = 20_260_907;

/// One entry of the gallery: how to sample it on either backend, and what it
/// says about itself before it runs.
struct Case {
  family: &'static str,
  label: &'static str,
  slug: &'static str,
  host: fn() -> Vec<Array1<f32>>,
  device: fn() -> Vec<Array1<f32>>,
  fallback: fn() -> Option<&'static str>,
}

/// Builds the three closures a [`Case`] needs from one process constructor.
///
/// The constructor is written twice, once per backend, because the backend is
/// a type parameter: `Gbm<f32, _, Cpu>` and `Gbm<f32, _, Metal>` are
/// different types, which is what makes an unsupported backend a compile
/// error rather than a run-time surprise.
macro_rules! case {
  ($family:literal, $label:literal, $slug:literal, $build:expr) => {
    Case {
      family: $family,
      label: $label,
      slug: $slug,
      host: || $build.on::<Cpu>().sample_par(PATHS),
      device: || $build.on::<Metal>().sample_par(PATHS),
      fallback: || $build.device_fallback(),
    }
  };
}

fn gallery() -> Vec<Case> {
  vec![
    case!(
      "diffusion",
      "GBM  μ=0.05 σ=0.2",
      "gbm",
      Gbm::<f32, _>::new(
        0.05,
        0.2,
        N,
        Some(100.0),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "diffusion",
      "Ornstein-Uhlenbeck  θ=2 μ=0.04",
      "ou",
      Ou::<f32, _>::new(
        2.0,
        0.04,
        0.25,
        N,
        Some(0.12),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "diffusion",
      "CIR  θ=1.5 μ=0.05",
      "cir",
      Cir::<f32, _>::new(
        1.5,
        0.05,
        0.15,
        N,
        Some(0.05),
        Some(1.0),
        None,
        Deterministic::new(SEED)
      )
    ),
    case!(
      "diffusion",
      "Jacobi  α=0.9 β=3 (unit interval)",
      "jacobi",
      Jacobi::<f32, _>::new(
        0.9,
        3.0,
        0.5,
        N,
        Some(0.3),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "diffusion",
      "3/2 model  κ=4 μ=0.09",
      "three-half",
      ThreeHalf::<f32, _>::new(
        4.0,
        0.09,
        0.8,
        N,
        Some(0.09),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "short rate",
      "Vasicek  θ=1.5 μ=0.04",
      "vasicek",
      Vasicek::<f32, _>::new(
        1.5,
        0.04,
        0.3,
        N,
        Some(0.04),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "jump",
      "variance gamma  θ=-0.2 ν=0.35",
      "variance-gamma",
      Vg::<f32, _>::new(
        -0.2,
        0.4,
        0.35,
        N,
        Some(0.0),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "jump",
      "normal inverse Gaussian  θ=0.15 κ=0.4",
      "nig",
      Nig::<f32, _>::new(
        0.15,
        0.5,
        0.4,
        N,
        Some(0.0),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "point process",
      "Poisson  λ=25 (arrival times)",
      "poisson",
      Poisson::<f32, _>::new(25.0, Some(N), None, Deterministic::new(SEED))
    ),
    case!(
      "fractional",
      "fBm  H=0.7",
      "fbm",
      Fbm::<f32, _>::new(0.7, N, Some(1.0), Deterministic::new(SEED))
    ),
    case!(
      "subordinator",
      "α-stable subordinator  α=0.7",
      "alpha-stable",
      AlphaStableSubordinator::<f32, _>::new(
        0.7,
        0.8,
        N,
        Some(0.0),
        Some(1.0),
        Deterministic::new(SEED)
      )
    ),
    case!(
      "conditional variance",
      "GARCH(1,1)  ω=.15 α=.15 β=.7",
      "garch",
      Garch::<f32, _>::new(
        0.15,
        ndarray::array![0.15],
        ndarray::array![0.7],
        N,
        Deterministic::new(SEED)
      )
    ),
  ]
}

/// The volatility pair is a two-component system, so it reaches the gallery
/// through its first component — the forward — with the same shape as every
/// other case.
fn sabr(backend: bool) -> Vec<Array1<f32>> {
  let build = || {
    Sabr::<f32, _>::new(
      0.4,
      0.5,
      -0.3,
      N,
      Some(100.0),
      Some(0.2),
      Some(1.0),
      Deterministic::new(SEED),
    )
  };
  let pairs = if backend {
    build().on::<Metal>().sample_par(PATHS)
  } else {
    build().on::<Cpu>().sample_par(PATHS)
  };
  pairs.into_iter().map(|[f, _]| f).collect()
}

/// Mean, standard deviation and the standard error of each, from the paths'
/// terminal values. The spread's error uses the sample's own kurtosis
/// (`σ√((κ−1)/4m)`), which is what makes a heavy-tailed process comparable at
/// all.
fn terminal_stats(paths: &[Array1<f32>]) -> (f64, f64, f64, f64) {
  let last: Vec<f64> = paths.iter().map(|p| p[p.len() - 1] as f64).collect();
  let n = last.len() as f64;
  let mean = last.iter().sum::<f64>() / n;
  let m2 = last.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
  let m4 = last.iter().map(|v| (v - mean).powi(4)).sum::<f64>() / n;
  let kurtosis = m4 / (m2 * m2);
  let sd = m2.sqrt();
  (
    mean,
    (m2 / n).sqrt(),
    sd,
    sd * ((kurtosis - 1.0) / (4.0 * n)).sqrt(),
  )
}

/// One page: the host's paths, the device's paths on the same scale, and the
/// two terminal laws.
fn page(label: &str, host: &[Array1<f32>], device: &[Array1<f32>], path: &str) {
  let mut plot = Plot::new();
  let xs: Vec<f64> = (0..host[0].len()).map(|i| i as f64).collect();
  for (paths, axis, colour) in [(host, 1usize, "#1f77b4"), (device, 2, "#d62728")] {
    for series in paths.iter().take(DRAWN) {
      let trace = Scatter::new(
        xs.clone(),
        series.iter().map(|v| *v as f64).collect::<Vec<_>>(),
      )
      .mode(Mode::Lines)
      .line(Line::new().width(0.8).color(colour))
      .show_legend(false);
      plot.add_trace(match axis {
        1 => trace,
        _ => trace.x_axis("x2").y_axis("y2"),
      });
    }
  }
  let terminal =
    |paths: &[Array1<f32>]| -> Vec<f64> { paths.iter().map(|p| p[p.len() - 1] as f64).collect() };
  plot.add_trace(
    Histogram::new(terminal(host))
      .name("CPU")
      .opacity(0.6)
      .n_bins_x(60)
      .x_axis("x3")
      .y_axis("y3"),
  );
  plot.add_trace(
    Histogram::new(terminal(device))
      .name("Metal")
      .opacity(0.6)
      .n_bins_x(60)
      .x_axis("x3")
      .y_axis("y3"),
  );
  plot.set_layout(
    Layout::new()
      .title(format!(
        "{label} — CPU and Metal draw different streams; the law is what agrees"
      ))
      .grid(
        LayoutGrid::new()
          .rows(1)
          .columns(3)
          .pattern(GridPattern::Independent),
      )
      .height(420)
      .x_axis(Axis::new().title("step — CPU"))
      .x_axis2(Axis::new().title("step — Metal"))
      .x_axis3(Axis::new().title("terminal value"))
      .y_axis3(Axis::new().title("count")),
  );
  plot.write_html(path);
}

fn main() {
  let dir = "target/metal_vs_cpu";
  fs::create_dir_all(dir).expect("output directory");

  // What the device is, before anything is asked of it. A handle is a value:
  // `Metal::default()` reads the ordinal from the environment, and `probe`
  // opens it.
  match stochastic_rs::stochastic::device::Backend::probe(&Metal::default()) {
    Ok(info) => println!(
      "device: {} ({}), precisions {:?}\n",
      info.name, info.backend, info.precisions
    ),
    Err(e) => {
      eprintln!("no Metal device: {e}");
      return;
    }
  }

  let mut cases = gallery();
  cases.push(Case {
    family: "volatility",
    label: "SABR  α=0.4 β=0.5 ρ=-0.3 (forward)",
    slug: "sabr",
    host: || sabr(false),
    device: || sabr(true),
    fallback: || None,
  });

  // One launch before the timings: the first call to a Metal device compiles
  // the kernel and builds the pipeline, a few hundred milliseconds that
  // belong to no process in particular.
  let _ = Gbm::<f32, _>::new(0.05, 0.2, 32, Some(1.0), Some(1.0), Deterministic::new(1))
    .on::<Metal>()
    .sample_par(8);

  println!(
    "{:<38} {:>10} {:>10} {:>7} {:>10} {:>10} {:>7} {:>8} {:>8}",
    "process", "mean cpu", "mean metal", "z", "sd cpu", "sd metal", "z", "cpu ms", "metal ms"
  );
  for case in &cases {
    if let Some(why) = (case.fallback)() {
      println!("{:<38} host on both sides: {why}", case.label);
    }
    let start = Instant::now();
    let host = (case.host)();
    let host_ms = start.elapsed().as_secs_f64() * 1e3;
    let start = Instant::now();
    let device = (case.device)();
    let device_ms = start.elapsed().as_secs_f64() * 1e3;

    let (hm, hme, hs, hse) = terminal_stats(&host);
    let (dm, dme, ds, dse) = terminal_stats(&device);
    println!(
      "{:<38} {hm:>10.4} {dm:>10.4} {:>7.1} {hs:>10.4} {ds:>10.4} {:>7.1} {host_ms:>8.1} {device_ms:>8.1}",
      case.label,
      (hm - dm) / (hme * hme + dme * dme).sqrt(),
      (hs - ds) / (hse * hse + dse * dse).sqrt(),
    );
    page(
      case.label,
      &host,
      &device,
      &format!("{dir}/{}.html", case.slug),
    );
  }

  let mut index = String::from(
    "<!doctype html><meta charset=utf-8><title>Metal against the CPU</title>\
     <style>body{font:15px system-ui;margin:3rem auto;max-width:44rem}\
     li{margin:.35rem 0}code{color:#666}</style>\
     <h1>The same processes on the CPU and on Metal</h1>\
     <p>Each page holds the host's paths, the device's paths on the same \
     scale, and the two terminal laws overlaid. The paths differ by \
     construction — the two sides draw different streams — so what the \
     pictures compare is the law.</p><ul>",
  );
  for case in &cases {
    index.push_str(&format!(
      "<li><a href=\"{}.html\">{}</a> <code>{}</code></li>",
      case.slug, case.label, case.family
    ));
  }
  index.push_str("</ul>");
  fs::write(format!("{dir}/index.html"), index).expect("index");
  println!("\nwrote {} pages to {dir}/index.html", cases.len());
  println!(
    "z is in standard errors; |z| under 5 is agreement. The α-stable row's \
     means are not comparable — that law has no mean — which is why the z \
     column, scaled by the sample's own spread, is the honest read."
  );
}

//! Side-by-side comparison of all Fgn backends: CPU, GPU (CubeCL), Metal, Accelerate.
//! Plots autocovariance vectors and trajectories for each.

#[cfg(all(feature = "cubecl-wgpu", feature = "metal", feature = "accelerate"))]
mod all_backends {
  use ndarray::Array1;
  use stochastic_rs::simd_rng::Unseeded;
  use stochastic_rs::stats::fractal_dim::FractalDimEstimator;
  use stochastic_rs::stats::fractal_dim::Higuchi;
  use stochastic_rs::stochastic::device::Accelerate;
  use stochastic_rs::stochastic::device::CubeclWgpu;
  use stochastic_rs::stochastic::device::Metal;
  use stochastic_rs::stochastic::noise::fgn::Fgn;
  use stochastic_rs::stochastic::process::fbm::Fbm;
  use stochastic_rs::traits::ProcessExt;

  /// Minimal grid-of-subplots HTML writer for this file's visual tests only
  /// (`stochastic-rs-viz`'s `GridPlotter` was removed workspace-wide; this
  /// keeps just the rows/cols layout + per-panel title annotation it used to
  /// provide, without pulling a crate back in for two test files).
  ///
  /// `panels` is `(panel title, [(series name, y-values)])`; every series in
  /// a panel is plotted against an implicit `0..1`-normalized index axis.
  fn grid_plot(
    panels: &[(String, Vec<(String, Vec<f64>)>)],
    cols: usize,
    title: &str,
    show_legend: bool,
    line_width: f64,
  ) -> plotly::Plot {
    use plotly::Layout;
    use plotly::Plot;
    use plotly::Scatter;
    use plotly::common::Anchor;
    use plotly::common::Font;
    use plotly::common::Line;
    use plotly::common::Mode;
    use plotly::layout::Annotation;
    use plotly::layout::GridPattern;
    use plotly::layout::LayoutGrid;

    let rows = panels.len().div_ceil(cols);
    let mut plot = Plot::new();
    let mut annotations = Vec::with_capacity(panels.len());

    for (idx, (label, series)) in panels.iter().enumerate() {
      let subplot = idx + 1;
      let (xa, ya) = if subplot == 1 {
        ("x".to_string(), "y".to_string())
      } else {
        (format!("x{subplot}"), format!("y{subplot}"))
      };
      let n_points = series[0].1.len();
      let t = (0..n_points)
        .map(|i| i as f64 / (n_points - 1).max(1) as f64)
        .collect::<Vec<f64>>();
      for (name, values) in series {
        plot.add_trace(
          Scatter::new(t.clone(), values.clone())
            .mode(Mode::Lines)
            .line(Line::new().width(line_width))
            .name(name.as_str())
            .show_legend(show_legend)
            .x_axis(xa.as_str())
            .y_axis(ya.as_str()),
        );
      }
      annotations.push(
        Annotation::new()
          .text(format!("<b>{label}</b>"))
          .x_ref(format!("{xa} domain"))
          .y_ref(format!("{ya} domain"))
          .x(0.5)
          .y(0.985)
          .x_anchor(Anchor::Center)
          .y_anchor(Anchor::Top)
          .font(Font::new().size(11))
          .show_arrow(false),
      );
    }

    plot.set_layout(
      Layout::new()
        .title(title)
        .height((rows * 380 + 120).max(500))
        .width((cols * 420).max(700))
        .annotations(annotations)
        .grid(
          LayoutGrid::new()
            .rows(rows)
            .columns(cols)
            .pattern(GridPattern::Independent),
        ),
    );
    plot
  }

  fn cpu_fgn(h: f64, n: usize, m: usize) -> Vec<Vec<f64>> {
    let fgn = Fgn::<f64>::new(h, n, Some(1.0), Unseeded);
    fgn.sample_par(m).into_iter().map(|p| p.to_vec()).collect()
  }

  fn gpu_fgn(h: f32, n: usize, m: usize) -> Vec<Vec<f64>> {
    let fgn = Fgn::<f32>::new(h, n, Some(1.0), Unseeded).on::<CubeclWgpu>();
    fgn
      .sample_par(m)
      .into_iter()
      .map(|p| p.iter().map(|&x| x as f64).collect())
      .collect()
  }

  fn metal_fgn(h: f32, n: usize, m: usize) -> Vec<Vec<f64>> {
    let fgn = Fgn::<f32>::new(h, n, Some(1.0), Unseeded).on::<Metal>();
    fgn
      .sample_par(m)
      .into_iter()
      .map(|p| p.iter().map(|&x| x as f64).collect())
      .collect()
  }

  fn accel_fgn(h: f32, n: usize, m: usize) -> Vec<Vec<f64>> {
    let fgn = Fgn::<f32>::new(h, n, Some(1.0), Unseeded).on::<Accelerate>();
    fgn
      .sample_par(m)
      .into_iter()
      .map(|p| p.iter().map(|&x| x as f64).collect())
      .collect()
  }

  fn metal_fbm(h: f32, n: usize, m: usize) -> Vec<Vec<f64>> {
    let fbm = Fbm::<f32>::new(h, n, Some(1.0), Unseeded).on::<Metal>();
    fbm
      .sample_par(m)
      .into_iter()
      .map(|p| p.iter().map(|&x| x as f64).collect())
      .collect()
  }

  fn empirical_acov(paths: &[Vec<f64>], max_lag: usize) -> Vec<f64> {
    let all: Vec<f64> = paths.iter().flatten().copied().collect();
    let mean = all.iter().sum::<f64>() / all.len() as f64;
    (0..=max_lag)
      .map(|lag| {
        let (mut s, mut c) = (0.0, 0usize);
        for p in paths {
          for i in 0..(p.len() - lag) {
            s += (p[i] - mean) * (p[i + lag] - mean);
            c += 1;
          }
        }
        s / c as f64
      })
      .collect()
  }

  fn theory_acov(h: f64, n: usize, max_lag: usize) -> Vec<f64> {
    let dt = 1.0 / n as f64;
    let var = dt.powf(2.0 * h);
    (0..=max_lag)
      .map(|k| {
        if k == 0 {
          var
        } else {
          var
            * 0.5
            * (((k + 1) as f64).powf(2.0 * h) - 2.0 * (k as f64).powf(2.0 * h)
              + ((k - 1) as f64).powf(2.0 * h))
        }
      })
      .collect()
  }

  #[test]
  #[ignore = "visual: writes HTML, no assertions; run with --ignored"]
  fn plot_all_backends_autocovariance() {
    let n = 1024;
    let m = 4096;
    let max_lag = 15;
    let hursts = [0.25_f64, 0.5, 0.72, 0.9];

    let mut panels = Vec::with_capacity(hursts.len());
    for &h in &hursts {
      let th = theory_acov(h, n, max_lag);
      let cpu = empirical_acov(&cpu_fgn(h, n, m), max_lag);
      let gpu = empirical_acov(&gpu_fgn(h as f32, n, m), max_lag);
      let mtl = empirical_acov(&metal_fgn(h as f32, n, m), max_lag);
      let acc = empirical_acov(&accel_fgn(h as f32, n, m), max_lag);

      eprintln!(
        "H={h}  lag  theory         CPU            GPU            Metal          Accelerate"
      );
      for k in 0..=max_lag.min(6) {
        eprintln!(
          "       {k:<4} {:<14.8} {:<14.8} {:<14.8} {:<14.8} {:<14.8}",
          th[k], cpu[k], gpu[k], mtl[k], acc[k]
        );
      }

      panels.push((
        format!("H={h}"),
        vec![
          ("theory".to_string(), th),
          ("CPU".to_string(), cpu),
          ("GPU".to_string(), gpu),
          ("Metal".to_string(), mtl),
          ("Accelerate".to_string(), acc),
        ],
      ));
    }

    let plot = grid_plot(
      &panels,
      2,
      "fGN autocovariance: Theory / CPU / GPU / Metal / Accelerate",
      true,
      2.0,
    );
    plot.write_html("target/all_backends_autocovariance.html");
    eprintln!("\nWrote target/all_backends_autocovariance.html");
  }

  #[test]
  #[ignore = "visual: writes HTML, no assertions; run with --ignored"]
  fn plot_all_backends_fgn_trajectories() {
    let n = 1024;
    let traj = 5;
    let hursts = [0.25_f32, 0.5, 0.72, 0.9];

    let to_series = |paths: Vec<Vec<f64>>| {
      paths
        .into_iter()
        .enumerate()
        .map(|(i, p)| (format!("path {}", i + 1), p))
        .collect::<Vec<(String, Vec<f64>)>>()
    };

    let mut panels = Vec::with_capacity(hursts.len() * 4);
    for &h in &hursts {
      panels.push((format!("CPU H={h}"), to_series(cpu_fgn(h as f64, n, traj))));
      panels.push((format!("GPU H={h}"), to_series(gpu_fgn(h, n, traj))));
      panels.push((format!("Metal H={h}"), to_series(metal_fgn(h, n, traj))));
      panels.push((format!("Accel H={h}"), to_series(accel_fgn(h, n, traj))));
    }

    let plot = grid_plot(
      &panels,
      4,
      "fGN trajectories by backend (5 paths each)",
      false,
      1.0,
    );
    plot.write_html("target/all_backends_fgn_trajectories.html");
    eprintln!("Wrote target/all_backends_fgn_trajectories.html");
  }

  #[test]
  #[ignore = "visual: writes HTML, no assertions; run with --ignored"]
  fn plot_all_backends_fbm_trajectories() {
    let n = 1024;
    let traj = 5;
    let hursts = [0.25_f32, 0.5, 0.72, 0.9];

    let mut panels = Vec::with_capacity(hursts.len());
    for &h in &hursts {
      let series = metal_fbm(h, n, traj)
        .into_iter()
        .enumerate()
        .map(|(i, p)| (format!("path {}", i + 1), p))
        .collect();
      panels.push((format!("fBM H={h}"), series));
    }

    let plot = grid_plot(
      &panels,
      2,
      "fBM trajectories — Metal GPU (5 paths each)",
      false,
      1.2,
    );
    plot.write_html("target/all_backends_fbm_trajectories.html");
    eprintln!("Wrote target/all_backends_fbm_trajectories.html");
  }

  #[test]
  fn all_backends_hurst_estimation() {
    let n = 4096;
    let m = 64;
    let hursts = [0.25_f32, 0.5, 0.72];

    eprintln!("\nHurst estimation (Higuchi FD, n={n}, m={m}):");
    eprintln!(
      "{:<8} {:<10} {:<10} {:<10} {:<10}",
      "H_true", "CPU", "GPU", "Metal", "Accel"
    );

    for &h in &hursts {
      let est = |paths: Vec<Vec<f64>>| -> f64 {
        let s: f64 = paths
          .iter()
          .map(|p| {
            let arr = Array1::from_vec(p.clone());
            let d = Higuchi::new(32)
              .estimate(arr.view())
              .expect("Higuchi on fGN path")
              .d;
            2.0 - d
          })
          .sum();
        s / paths.len() as f64
      };

      let hc = est(cpu_fgn(h as f64, n - 1, m));
      let hg = est(gpu_fgn(h, n - 1, m));
      let hm = est(metal_fgn(h, n - 1, m));
      let ha = est(accel_fgn(h, n - 1, m));

      eprintln!(
        "{:<8.2} {:<10.4} {:<10.4} {:<10.4} {:<10.4}",
        h, hc, hg, hm, ha
      );
    }
  }
}

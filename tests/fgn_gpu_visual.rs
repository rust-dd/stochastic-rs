//! Visual validation of GPU-sampled fGN and fBM.
//!
//! Compares empirical autocovariance vectors (CPU vs GPU) and plots
//! trajectories for multiple Hurst parameters. Also estimates H back
//! from the generated fBM paths using fractal dimension.

#[cfg(any(feature = "gpu-cuda", feature = "gpu-wgpu"))]
mod gpu_visual {
  use ndarray::Array1;
  use stochastic_rs::simd_rng::Unseeded;
  use stochastic_rs::stats::fractal_dim::FractalDimEstimator;
  use stochastic_rs::stats::fractal_dim::Higuchi;
  use stochastic_rs::stats::fractal_dim::Variogram;
  use stochastic_rs::stochastic::device::CubeCl;
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

  fn gpu_fgn_paths(h: f32, n: usize, m: usize) -> Vec<Vec<f64>> {
    let fgn = Fgn::<f32>::new(h, n, Some(1.0), Unseeded).on::<CubeCl>();
    fgn
      .sample_par(m)
      .into_iter()
      .map(|p| p.iter().map(|&x| x as f64).collect())
      .collect()
  }

  fn cpu_fgn_paths(h: f64, n: usize, m: usize) -> Vec<Vec<f64>> {
    let fgn = Fgn::<f64>::new(h, n, Some(1.0), Unseeded);
    fgn.sample_par(m).into_iter().map(|p| p.to_vec()).collect()
  }

  fn gpu_fbm_paths(h: f32, n: usize, m: usize) -> Vec<Vec<f64>> {
    let fbm = Fbm::<f32>::new(h, n, Some(1.0), Unseeded).on::<CubeCl>();
    fbm
      .sample_par(m)
      .into_iter()
      .map(|p| p.iter().map(|&x| x as f64).collect())
      .collect()
  }

  fn empirical_autocovariance(paths: &[Vec<f64>], max_lag: usize) -> Vec<f64> {
    let all: Vec<f64> = paths.iter().flatten().copied().collect();
    let mean = all.iter().sum::<f64>() / all.len() as f64;
    (0..=max_lag)
      .map(|lag| {
        let mut s = 0.0;
        let mut c = 0usize;
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

  fn theoretical_autocovariance(h: f64, n: usize, max_lag: usize) -> Vec<f64> {
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
  fn plot_cpu_vs_gpu_autocovariance() {
    let n = 1024;
    let m = 4096;
    let max_lag = 20;
    let hursts = [0.25_f64, 0.5, 0.72, 0.9];

    let mut panels = Vec::with_capacity(hursts.len());
    for &h in &hursts {
      let theory = theoretical_autocovariance(h, n, max_lag);
      let cpu_acov = empirical_autocovariance(&cpu_fgn_paths(h, n, m), max_lag);
      let gpu_acov = empirical_autocovariance(&gpu_fgn_paths(h as f32, n, m), max_lag);

      eprintln!("H={h}: autocovariance (lag 0..{max_lag})");
      eprintln!("  lag  theory          CPU             GPU             CPU/th  GPU/th");
      for k in 0..=max_lag.min(10) {
        eprintln!(
          "  {k:<4} {:<15.8} {:<15.8} {:<15.8} {:<7.4} {:<7.4}",
          theory[k],
          cpu_acov[k],
          gpu_acov[k],
          cpu_acov[k] / theory[k],
          gpu_acov[k] / theory[k],
        );
      }

      panels.push((
        format!("H={h}"),
        vec![
          ("theory".to_string(), theory),
          ("CPU".to_string(), cpu_acov),
          ("GPU".to_string(), gpu_acov),
        ],
      ));
    }

    let plot = grid_plot(
      &panels,
      2,
      "fGN autocovariance: theory vs CPU vs GPU",
      true,
      2.0,
    );
    plot.write_html("target/gpu_autocovariance.html");
    eprintln!("\nWrote target/gpu_autocovariance.html");
  }

  #[test]
  #[ignore = "visual: writes HTML, no assertions; run with --ignored"]
  fn plot_gpu_fgn_trajectories() {
    let n = 1024;
    let traj = 8;
    let hursts = [0.2_f32, 0.35, 0.5, 0.72, 0.85, 0.95];

    let mut panels = Vec::with_capacity(hursts.len());
    for &h in &hursts {
      let series = gpu_fgn_paths(h, n, traj)
        .into_iter()
        .enumerate()
        .map(|(i, p)| (format!("path {}", i + 1), p))
        .collect();
      panels.push((format!("fGN H={h}"), series));
    }

    let plot = grid_plot(&panels, 3, "GPU fGN trajectories (CubeCL)", false, 1.0);
    plot.write_html("target/gpu_fgn_trajectories.html");
    eprintln!("Wrote target/gpu_fgn_trajectories.html");
  }

  #[test]
  #[ignore = "visual: writes HTML, no assertions; run with --ignored"]
  fn plot_gpu_fbm_trajectories() {
    let n = 1024;
    let traj = 8;
    let hursts = [0.2_f32, 0.35, 0.5, 0.72, 0.85, 0.95];

    let mut panels = Vec::with_capacity(hursts.len());
    for &h in &hursts {
      let series = gpu_fbm_paths(h, n, traj)
        .into_iter()
        .enumerate()
        .map(|(i, p)| (format!("path {}", i + 1), p))
        .collect();
      panels.push((format!("fBM H={h}"), series));
    }

    let plot = grid_plot(&panels, 3, "GPU fBM trajectories (CubeCL)", false, 1.0);
    plot.write_html("target/gpu_fbm_trajectories.html");
    eprintln!("Wrote target/gpu_fbm_trajectories.html");
  }

  #[test]
  fn gpu_fbm_hurst_estimation() {
    let n = 4096;
    let m = 64;
    let hursts = [0.2_f32, 0.35, 0.5, 0.72, 0.85];

    eprintln!("\nGPU fBM Hurst estimation (n={n}, m={m}):");
    eprintln!(
      "{:<8} {:<12} {:<12} {:<12}",
      "H_true", "H_vario", "H_higuchi", "avg_err"
    );

    for &h_true in &hursts {
      let paths = gpu_fbm_paths(h_true, n, m);

      let mut h_vario_sum = 0.0;
      let mut h_higuchi_sum = 0.0;
      for path in &paths {
        let arr = Array1::from_vec(path.to_vec());
        let v_d = Variogram { p: 2.0 }
          .estimate(arr.view())
          .expect("variogram on fGN path")
          .d;
        let h_d = Higuchi { kmax: 32 }
          .estimate(arr.view())
          .expect("Higuchi on fGN path")
          .d;
        h_vario_sum += 2.0 - v_d;
        h_higuchi_sum += 2.0 - h_d;
      }
      let h_vario = h_vario_sum / m as f64;
      let h_higuchi = h_higuchi_sum / m as f64;
      let h64 = h_true as f64;
      let avg_err = ((h_vario - h64).abs() + (h_higuchi - h64).abs()) / 2.0;

      eprintln!(
        "{:<8.2} {:<12.4} {:<12.4} {:<12.4}",
        h_true, h_vario, h_higuchi, avg_err
      );

      if h64 <= 0.75 {
        assert!(
          (h_vario - h64).abs() < 0.08,
          "H={h_true}: variogram estimate {h_vario} too far"
        );
      }
      assert!(
        (h_higuchi - h64).abs() < 0.08,
        "H={h_true}: higuchi estimate {h_higuchi} too far"
      );
    }
  }
}

//! Visual validation of GPU-sampled fGN and fBM.
//!
//! Compares empirical autocovariance vectors (CPU vs GPU) and plots
//! trajectories for multiple Hurst parameters. Also estimates H back
//! from the generated fBM paths using fractal dimension.

#[cfg(feature = "gpu-wgpu")]
mod gpu_visual {
  use either::Either;
  use ndarray::Array1;
  use stochastic_rs::stats::fd::FractalDim;
  use stochastic_rs::stochastic::noise::fgn::Fgn;
  use stochastic_rs::stochastic::process::fbm::Fbm;
  use stochastic_rs::traits::ProcessExt;
  use stochastic_rs::visualization::GridPlotter;

  fn gpu_fgn_paths(h: f32, n: usize, m: usize) -> Vec<Vec<f64>> {
    let fgn = Fgn::<f32>::new(h, n, Some(1.0));
    match fgn.sample_gpu(m).expect("GPU Fgn sampling failed") {
      Either::Left(p) => vec![p.iter().map(|&x| x as f64).collect()],
      Either::Right(ps) => ps
        .outer_iter()
        .map(|r| r.iter().map(|&x| x as f64).collect())
        .collect(),
    }
  }

  fn cpu_fgn_paths(h: f64, n: usize, m: usize) -> Vec<Vec<f64>> {
    let fgn = Fgn::<f64>::new(h, n, Some(1.0));
    fgn.sample_par(m).into_iter().map(|p| p.to_vec()).collect()
  }

  fn gpu_fbm_paths(h: f32, n: usize, m: usize) -> Vec<Vec<f64>> {
    let fbm = Fbm::<f32>::new(h, n, Some(1.0));
    match fbm.sample_gpu(m).expect("GPU Fbm sampling failed") {
      Either::Left(p) => vec![p.iter().map(|&x| x as f64).collect()],
      Either::Right(ps) => ps
        .outer_iter()
        .map(|r| r.iter().map(|&x| x as f64).collect())
        .collect(),
    }
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
  fn plot_cpu_vs_gpu_autocovariance() {
    let n = 1024;
    let m = 4096;
    let max_lag = 20;
    let hursts = [0.25_f64, 0.5, 0.72, 0.9];

    let mut grid = GridPlotter::new()
      .title("fGN autocovariance: theory vs CPU vs GPU")
      .cols(2)
      .line_width(2.0)
      .show_legend(true);

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

      grid = grid.register_paths(
        vec![theory, cpu_acov, gpu_acov],
        &format!("H={h} (blue=theory, red=CPU, green=GPU)"),
      );
    }

    let plot = grid.plot();
    plot.write_html("target/gpu_autocovariance.html");
    eprintln!("\nWrote target/gpu_autocovariance.html");
  }

  #[test]
  fn plot_gpu_fgn_trajectories() {
    let n = 1024;
    let traj = 8;
    let hursts = [0.2_f32, 0.35, 0.5, 0.72, 0.85, 0.95];

    let mut grid = GridPlotter::new()
      .title("GPU fGN trajectories (CubeCL)")
      .cols(3)
      .line_width(1.0)
      .show_legend(false);

    for &h in &hursts {
      let paths = gpu_fgn_paths(h, n, traj);
      grid = grid.register_paths(paths, &format!("fGN H={h}"));
    }

    let plot = grid.plot();
    plot.write_html("target/gpu_fgn_trajectories.html");
    eprintln!("Wrote target/gpu_fgn_trajectories.html");
  }

  #[test]
  fn plot_gpu_fbm_trajectories() {
    let n = 1024;
    let traj = 8;
    let hursts = [0.2_f32, 0.35, 0.5, 0.72, 0.85, 0.95];

    let mut grid = GridPlotter::new()
      .title("GPU fBM trajectories (CubeCL)")
      .cols(3)
      .line_width(1.0)
      .show_legend(false);

    for &h in &hursts {
      let paths = gpu_fbm_paths(h, n, traj);
      grid = grid.register_paths(paths, &format!("fBM H={h}"));
    }

    let plot = grid.plot();
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
        let fd = FractalDim::new(arr);
        h_vario_sum += 2.0 - fd.variogram(Some(2.0));
        h_higuchi_sum += 2.0 - fd.higuchi_fd(32);
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

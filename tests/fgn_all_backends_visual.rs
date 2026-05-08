//! Side-by-side comparison of all Fgn backends: CPU, GPU (CubeCL), Metal, Accelerate.
//! Plots autocovariance vectors and trajectories for each.

#[cfg(all(feature = "gpu-wgpu", feature = "metal", feature = "accelerate"))]
mod all_backends {
  use either::Either;
  use ndarray::Array1;
  use stochastic_rs::stats::fd::FractalDim;
  use stochastic_rs::stochastic::noise::fgn::Fgn;
  use stochastic_rs::stochastic::process::fbm::Fbm;
  use stochastic_rs::traits::ProcessExt;
  use stochastic_rs::visualization::GridPlotter;

  fn cpu_fgn(h: f64, n: usize, m: usize) -> Vec<Vec<f64>> {
    let fgn = Fgn::<f64>::new(h, n, Some(1.0));
    fgn.sample_par(m).into_iter().map(|p| p.to_vec()).collect()
  }

  fn gpu_fgn(h: f32, n: usize, m: usize) -> Vec<Vec<f64>> {
    let fgn = Fgn::<f32>::new(h, n, Some(1.0));
    match fgn.sample_gpu(m).unwrap() {
      Either::Left(p) => vec![p.iter().map(|&x| x as f64).collect()],
      Either::Right(ps) => ps
        .outer_iter()
        .map(|r| r.iter().map(|&x| x as f64).collect())
        .collect(),
    }
  }

  fn metal_fgn(h: f32, n: usize, m: usize) -> Vec<Vec<f64>> {
    let fgn = Fgn::<f32>::new(h, n, Some(1.0));
    match fgn.sample_metal(m).unwrap() {
      Either::Left(p) => vec![p.iter().map(|&x| x as f64).collect()],
      Either::Right(ps) => ps
        .outer_iter()
        .map(|r| r.iter().map(|&x| x as f64).collect())
        .collect(),
    }
  }

  fn accel_fgn(h: f32, n: usize, m: usize) -> Vec<Vec<f64>> {
    let fgn = Fgn::<f32>::new(h, n, Some(1.0));
    match fgn.sample_accelerate(m).unwrap() {
      Either::Left(p) => vec![p.iter().map(|&x| x as f64).collect()],
      Either::Right(ps) => ps
        .outer_iter()
        .map(|r| r.iter().map(|&x| x as f64).collect())
        .collect(),
    }
  }

  fn metal_fbm(h: f32, n: usize, m: usize) -> Vec<Vec<f64>> {
    let fbm = Fbm::<f32>::new(h, n, Some(1.0));
    match fbm.sample_metal(m).unwrap() {
      Either::Left(p) => vec![p.iter().map(|&x| x as f64).collect()],
      Either::Right(ps) => ps
        .outer_iter()
        .map(|r| r.iter().map(|&x| x as f64).collect())
        .collect(),
    }
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
  fn plot_all_backends_autocovariance() {
    let n = 1024;
    let m = 4096;
    let max_lag = 15;
    let hursts = [0.25_f64, 0.5, 0.72, 0.9];

    let mut grid = GridPlotter::new()
      .title("fGN autocovariance: Theory / CPU / GPU / Metal / Accelerate")
      .cols(2)
      .line_width(2.0)
      .show_legend(true);

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

      grid = grid.register_paths(
        vec![th, cpu, gpu, mtl, acc],
        &format!("H={h} (theory/CPU/GPU/Metal/Accel)"),
      );
    }

    let plot = grid.plot();
    plot.write_html("target/all_backends_autocovariance.html");
    eprintln!("\nWrote target/all_backends_autocovariance.html");
  }

  #[test]
  fn plot_all_backends_fgn_trajectories() {
    let n = 1024;
    let traj = 5;
    let hursts = [0.25_f32, 0.5, 0.72, 0.9];

    let mut grid = GridPlotter::new()
      .title("fGN trajectories by backend (5 paths each)")
      .cols(4)
      .line_width(1.0)
      .show_legend(false);

    for &h in &hursts {
      grid = grid.register_paths(cpu_fgn(h as f64, n, traj), &format!("CPU H={h}"));
      grid = grid.register_paths(gpu_fgn(h, n, traj), &format!("GPU H={h}"));
      grid = grid.register_paths(metal_fgn(h, n, traj), &format!("Metal H={h}"));
      grid = grid.register_paths(accel_fgn(h, n, traj), &format!("Accel H={h}"));
    }

    let plot = grid.plot();
    plot.write_html("target/all_backends_fgn_trajectories.html");
    eprintln!("Wrote target/all_backends_fgn_trajectories.html");
  }

  #[test]
  fn plot_all_backends_fbm_trajectories() {
    let n = 1024;
    let traj = 5;
    let hursts = [0.25_f32, 0.5, 0.72, 0.9];

    let mut grid = GridPlotter::new()
      .title("fBM trajectories — Metal GPU (5 paths each)")
      .cols(2)
      .line_width(1.2)
      .show_legend(false);

    for &h in &hursts {
      grid = grid.register_paths(metal_fbm(h, n, traj), &format!("fBM H={h}"));
    }

    let plot = grid.plot();
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
            let fd = FractalDim::new(Array1::from_vec(p.clone()));
            2.0 - fd.higuchi_fd(32).expect("Higuchi on fGN path")
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

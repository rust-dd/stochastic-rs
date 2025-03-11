use ndarray::Array1;
use prettytable::{format, row, Table};
use stochastic_rs::plot_1d;
use stochastic_rs::stochastic::noise::fgn::FGN;
use stochastic_rs::stochastic::Sampling;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

use stochastic_rs::stats::fd::FractalDim;
use stochastic_rs::stats::fou_estimator::{
  FOUParameterEstimationV1, FOUParameterEstimationV2, FilterType,
};

// Import your estimator modules and types here
// use your_crate::{FOUParameterEstimationV1, FOUParameterEstimationV2, FilterType};

fn main() -> Result<(), Box<dyn Error>> {
  let fbm = FGN::new(0.7, 10_000, Some(1.0), Some(10000));
  let fgn = fbm.sample_cuda().unwrap();
  let fgn = fgn.row(0);
  plot_1d!(fgn, "Fractional Brownian Motion (H = 0.7)");
  let mut path = Array1::<f64>::zeros(500);
  for i in 1..500 {
    path[i] += path[i-1] + fgn[i];
  }
  plot_1d!(path, "Fractional Brownian Motion (H = 0.7)");

  let start = std::time::Instant::now();
  let _ = fbm.sample_cuda();
  let end = start.elapsed().as_millis();
  println!("20000 fgn generated on cuda in: {end}");

  let start = std::time::Instant::now();
    let _ = fbm.sample_par();
  let end = start.elapsed().as_millis();
  println!("20000 fgn generated on cuda in: {end}");
  // File paths
  // let paths = vec![
  //   "./test/kecskekut_original.txt",
  //   "./test/kecskekut_sim.txt",
  //   "./test/kecskekut_without_jumps.txt",
  //   "./test/komlos_original.txt",
  //   "./test/komlos_sim.txt",
  //   "./test/komlos_without_jumps.txt",
  // ];

  // // Process other datasets
  // for path in paths {
  //   println!("\nProcessing {}", path);
  //   let data = read_vector_from_file(path)?;

  //   // V1 Estimation
  //   let delta = 1.0; // Adjust as necessary
  //   let mut estimator_v1 =
  //     FOUParameterEstimationV1::new(data.clone().into(), FilterType::Daubechies, Some(delta));
  //   let hurst = FractalDim::new(data.clone().into());
  //   let hurst = hurst.higuchi_fd(12);
  //   println!("Higuchi FD: {}", 2.0 - hurst);
  //   // estimator_v1.set_hurst(2. - hurst);
  //   let (hurst_v1, sigma_v1, mu_v1, theta_v1) = estimator_v1.estimate_parameters();

  //   // V2 Estimation
  //   let delta = 1.0; // Adjust as necessary
  //   let n = data.len();
  //   let mut estimator_v2 = FOUParameterEstimationV2::new(data.clone().into(), Some(delta), n);
  //   // estimator_v2.set_hurst(2. - hurst);
  //   let (hurst_v2, sigma_v2, mu_v2, theta_v2) = estimator_v2.estimate_parameters();

  //   let mut table = Table::new();
  //   table.set_format(*format::consts::FORMAT_NO_LINESEP_WITH_TITLE);
  //   table.add_row(row![
  //     "Version",
  //     "Hurst",
  //     "Sigma",
  //     "Mu",
  //     "Theta",
  //     "exp(Theta)"
  //   ]);
  //   table.add_row(row![
  //     "V1",
  //     format!("{:.4}", hurst_v1),
  //     format!("{:.4}", sigma_v1),
  //     format!("{:.4}", mu_v1),
  //     format!("{:.4}", theta_v1),
  //     format!("{:.4}", (-theta_v1).exp())
  //   ]);
  //   table.add_row(row![
  //     "V2",
  //     format!("{:.4}", hurst_v2),
  //     format!("{:.4}", sigma_v2),
  //     format!("{:.4}", mu_v2),
  //     format!("{:.4}", theta_v2),
  //     format!("{:.4}", (-theta_v2).exp())
  //   ]);

  //   // Táblázat kiíratása
  //   println!("\nEstimated Parameters for {}:\n", path);
  //   table.printstd();
  // }

  Ok(())
}

fn read_vector_from_file(filename: &str) -> Result<Vec<f64>, Box<dyn Error>> {
  let file = File::open(filename)?;
  let reader = BufReader::new(file);
  let mut data = Vec::new();

  for line in reader.lines() {
    let line = line?;
    let value: f64 = line.trim().parse()?;
    data.push(value);
  }

  Ok(data)
}

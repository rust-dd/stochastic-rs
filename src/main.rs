use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

use stochastic_rs::stats::fou_estimator::{
  FOUParameterEstimationV1, FOUParameterEstimationV2, FilterType,
};
use stochastic_rs::stats::fd::FractalDim;

// Import your estimator modules and types here
// use your_crate::{FOUParameterEstimationV1, FOUParameterEstimationV2, FilterType};

fn main() -> Result<(), Box<dyn Error>> {
  // File paths
  let paths = vec![
    "./test/kecskekut_original.txt",
    "./test/kecskekut_sim.txt",
    "./test/kecskekut_without_jumps.txt",
    "./test/komlos_original.txt",
    "./test/komlos_sim.txt",
    "./test/komlos_without_jumps.txt",
  ];

  // Process other datasets
  for path in paths {
    println!("\nProcessing {}", path);
    let data = read_vector_from_file(path)?;

    // V1 Estimation
    let mut estimator_v1 =
      FOUParameterEstimationV1::new(data.clone().into(), FilterType::Daubechies);
    let hurst = FractalDim::new(data.clone().into());
    let hurst = hurst.higuchi_fd(10);
    estimator_v1.set_hurst(2. - hurst);
    let (hurst_v1, sigma_v1, mu_v1, theta_v1) = estimator_v1.estimate_parameters();
    println!("V1 - Estimated Parameters for {}:", path);
    println!("  Hurst exponent: {}", hurst_v1);
    println!("  Sigma: {}", sigma_v1);
    println!("  Mu: {}", mu_v1);
    println!("  Theta: {}", theta_v1);

    // V2 Estimation
    let delta = 1.0 / 256.0; // Adjust as necessary
    let n = data.len();
    let mut estimator_v2 = FOUParameterEstimationV2::new(data.clone().into(), delta, n);
    estimator_v2.set_hurst(2. - hurst);
    let (hurst_v2, sigma_v2, mu_v2, theta_v2) = estimator_v2.estimate_parameters();
    println!("V1 - Estimated Parameters for {}:", path);
    println!("  Hurst exponent: {}", hurst_v2);
    println!("  Sigma: {}", sigma_v2);
    println!("  Mu: {}", mu_v2);
    println!("  Theta: {}", theta_v2);
  }

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

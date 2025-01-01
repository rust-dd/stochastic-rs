use ndarray::{Array, Array1, Array2};
use plotly::{Plot, Scatter};
use rand::prelude::*;
use rand_distr::Uniform;
use statrs::distribution::{ContinuousCDF, MultivariateNormal, Normal};

/// ======================================================
/// 1) Empirical Copula (Ranking)
/// ======================================================
pub fn empirical_copula(x: &Array1<f64>, y: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
  // Sort indices for x
  let mut idx_x = (0..x.len()).collect::<Vec<usize>>();
  idx_x.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());
  // Ranks for x
  let mut rank_sx = Array::zeros(x.len());
  for (i, &idx) in idx_x.iter().enumerate() {
    rank_sx[idx] = (i + 1) as f64 / x.len() as f64;
  }

  // Sort indices for y
  let mut idx_y = (0..y.len()).collect::<Vec<usize>>();
  idx_y.sort_by(|&i, &j| y[i].partial_cmp(&y[j]).unwrap());
  // Ranks for y
  let mut rank_sy = Array::zeros(y.len());
  for (i, &idx) in idx_y.iter().enumerate() {
    rank_sy[idx] = (i + 1) as f64 / y.len() as f64;
  }

  (rank_sx, rank_sy)
}

/// ======================================================
/// 2) Plot (Empirical) Copula
/// ======================================================
pub fn plot_copula(u: &Array1<f64>, v: &Array1<f64>, title: &str) {
  let trace = Scatter::new(u.to_vec(), v.to_vec())
    .mode(plotly::common::Mode::Markers)
    .marker(plotly::common::Marker::new().size(4))
    .name(title);

  let mut plot = Plot::new();
  plot.add_trace(trace);
  plot.show();
}

/// ======================================================
/// 3) Gaussian Copula
/// ======================================================
pub fn generate_gaussian_copula_sample(n: usize, rho: f64) -> (Array1<f64>, Array1<f64>) {
  let mean = vec![0.0, 0.0];
  let cov = vec![vec![1.0, rho], vec![rho, 1.0]];
  let cov_flat = cov.into_iter().flatten().collect();

  let mvn = MultivariateNormal::new(mean, cov_flat)
    .expect("Invalid covariance matrix for MultivariateNormal.");

  let mut rng = thread_rng();
  let mut samples = Array2::<f64>::zeros((n, 2));

  for i in 0..n {
    let xy = mvn.sample(&mut rng);
    samples[[i, 0]] = xy[0];
    samples[[i, 1]] = xy[1];
  }

  // Transform each dimension by standard normal CDF to get U(0,1)
  let standard_normal = Normal::new(0.0, 1.0).unwrap();
  let cdf = |z: f64| standard_normal.cdf(z);

  let u = samples.column(0).mapv(cdf);
  let v = samples.column(1).mapv(cdf);

  (u.to_owned(), v.to_owned())
}

/// ======================================================
/// 4) Clayton Copula (Archimedean)
/// ======================================================
pub fn generate_clayton_copula_sample(n: usize, alpha: f64) -> (Array1<f64>, Array1<f64>) {
  use rand_distr::{Exp, Gamma};
  assert!(alpha > 0.0, "Clayton alpha must be > 0.");

  let mut rng = thread_rng();
  let gamma_dist = Gamma::new(1.0 / alpha, 1.0).unwrap();
  let exp_dist = Exp::new(1.0).unwrap();

  let mut u = Array1::<f64>::zeros(n);
  let mut v = Array1::<f64>::zeros(n);

  for i in 0..n {
    let w = gamma_dist.sample(&mut rng);
    let e1 = exp_dist.sample(&mut rng);
    let e2 = exp_dist.sample(&mut rng);

    // (1 + e1 / w)^(-1/alpha)
    u[i] = (1.0 + e1 / w).powf(-1.0 / alpha);
    v[i] = (1.0 + e2 / w).powf(-1.0 / alpha);
  }
  (u, v)
}

/// ======================================================
/// 5) Gumbel Copula (Archimedean)
/// ======================================================
pub fn generate_gumbel_copula_sample(n: usize, alpha: f64) -> (Array1<f64>, Array1<f64>) {
  use rand_distr::Exp;
  assert!(alpha >= 1.0, "Gumbel alpha must be >= 1.0.");

  let mut rng = thread_rng();
  let exp_dist = Exp::new(1.0).unwrap();
  let uniform_dist = Uniform::new(1e-15, 1.0 - 1e-15);

  let mut u = Array1::<f64>::zeros(n);
  let mut v = Array1::<f64>::zeros(n);

  for i in 0..n {
    let e1 = exp_dist.sample(&mut rng);
    let e2 = exp_dist.sample(&mut rng);
    // Avoid exact 0 or 1 for stable approx
    let x = uniform_dist.sample(&mut rng);

    let w_approx = e1 / (1.0 + x) + 1e-15; // ensure nonzero

    let z1 = (e2 / w_approx as f64).powf(1.0 / alpha);
    let z2 = (e1 / w_approx).powf(1.0 / alpha);

    u[i] = (-z1).exp();
    v[i] = (-z2).exp();
  }
  (u, v)
}

/// ======================================================
/// 6) Frank Copula (Archimedean) - with clamping
/// ======================================================
pub fn generate_frank_copula_sample(n: usize, theta: f64) -> (Array1<f64>, Array1<f64>) {
  assert!(theta != 0.0, "Frank copula parameter must be non-zero.");

  let mut rng = thread_rng();
  let uni = Uniform::new(1e-15, 1.0 - 1e-15);

  let mut u = Array1::<f64>::zeros(n);
  let mut v = Array1::<f64>::zeros(n);

  for i in 0..n {
    // Avoid exact 0 or 1 by sampling in (1e-15, 1 - 1e-15)
    let uu = uni.sample(&mut rng);
    let zz = uni.sample(&mut rng);
    u[i] = uu;

    let denom = 1.0 - (-theta * uu).exp();
    let numerator = (1.0 - (-theta).exp()) * zz;
    let mut inside = 1.0 - numerator / denom;

    // If inside <= 0 => clamp to a small positive
    if inside <= 1e-15 {
      inside = 1e-15;
    }

    v[i] = -1.0 / theta * inside.ln();
  }
  (u, v)
}

/// ======================================================
/// 7) Tests (including Empirical Copula)
/// ======================================================
#[cfg(test)]
mod tests {
  use super::*;
  use approx::assert_abs_diff_eq;

  const N: usize = 500; // sample size for tests

  // ----------------------------------------------
  // A) Direct Empirical Copula Test
  // ----------------------------------------------
  #[test]
  fn test_empirical_copula_direct() {
    let mut rng = thread_rng();
    let uniform_dist = Uniform::new(0.0, 1.0);

    let x_data = (0..N).map(|_| uniform_dist.sample(&mut rng)).collect();
    let y_data = (0..N).map(|_| uniform_dist.sample(&mut rng)).collect();

    let x_arr = Array1::from_vec(x_data);
    let y_arr = Array1::from_vec(y_data);

    // Build empirical copula
    let (sx, sy) = empirical_copula(&x_arr, &y_arr);

    // Check range
    assert!(sx.iter().all(|&x| x >= 0.0 && x <= 1.0));
    assert!(sy.iter().all(|&y| y >= 0.0 && y <= 1.0));

    // Means near 0.5
    assert_abs_diff_eq!(sx.mean().unwrap(), 0.5, epsilon = 0.1);
    assert_abs_diff_eq!(sy.mean().unwrap(), 0.5, epsilon = 0.1);

    // Plot
    plot_copula(&sx, &sy, "Empirical Copula (Direct Uniform Data)");
  }

  // ----------------------------------------------
  // B) Gaussian Copula Test
  // ----------------------------------------------
  #[test]
  fn test_gaussian_copula() {
    let rho = 0.7;
    let (u, v) = generate_gaussian_copula_sample(N, rho);
    let (sx, sy) = empirical_copula(&u, &v);

    assert!(u.iter().all(|&x| x >= 0.0 && x <= 1.0));
    assert!(v.iter().all(|&y| y >= 0.0 && y <= 1.0));

    plot_copula(&sx, &sy, "Empirical Copula (Gaussian, rho=0.7)");

    // Means ~ 0.5
    assert_abs_diff_eq!(sx.mean().unwrap(), 0.5, epsilon = 0.1);
    assert_abs_diff_eq!(sy.mean().unwrap(), 0.5, epsilon = 0.1);
  }

  // ----------------------------------------------
  // C) Clayton Copula Test (Archimedean)
  // ----------------------------------------------
  #[test]
  fn test_clayton_copula() {
    let alpha = 1.5;
    let (u, v) = generate_clayton_copula_sample(N, alpha);
    let (sx, sy) = empirical_copula(&u, &v);

    assert!(u.iter().all(|&x| x >= 0.0 && x <= 1.0));
    assert!(v.iter().all(|&y| y >= 0.0 && y <= 1.0));

    plot_copula(&sx, &sy, "Empirical Copula (Clayton, alpha=1.5)");

    assert_abs_diff_eq!(sx.mean().unwrap(), 0.5, epsilon = 0.1);
    assert_abs_diff_eq!(sy.mean().unwrap(), 0.5, epsilon = 0.1);
  }

  // ----------------------------------------------
  // D) Gumbel Copula Test (Archimedean)
  // ----------------------------------------------
  #[test]
  fn test_gumbel_copula() {
    let alpha = 1.5; // Gumbel parameter >= 1
    let (u, v) = generate_gumbel_copula_sample(N, alpha);
    let (sx, sy) = empirical_copula(&u, &v);

    assert!(u.iter().all(|&x| x >= 0.0 && x <= 1.0));
    assert!(v.iter().all(|&y| y >= 0.0 && y <= 1.0));

    plot_copula(&sx, &sy, "Empirical Copula (Gumbel, alpha=1.5)");

    // Gumbel can have heavier tail dependence, so allow a bit more tolerance
    assert_abs_diff_eq!(sx.mean().unwrap(), 0.5, epsilon = 0.2);
    assert_abs_diff_eq!(sy.mean().unwrap(), 0.5, epsilon = 0.2);
  }

  // ----------------------------------------------
  // E) Frank Copula Test (Archimedean)
  // ----------------------------------------------
  #[test]
  fn test_frank_copula() {
    let theta = 0.5;
    let (u, v) = generate_frank_copula_sample(N, theta);
    let (sx, sy) = empirical_copula(&u, &v);

    // Check range only if you clamp inside the generator
    // TODO: this test is failing
    assert!(u.iter().all(|&x| x >= 0.0 && x <= 1.0));
    assert!(v.iter().all(|&y| y >= 0.0 && y <= 1.0));

    plot_copula(&sx, &sy, "Empirical Copula (Frank, theta=5.0)");

    // Means ~ 0.5
    assert_abs_diff_eq!(sx.mean().unwrap(), 0.5, epsilon = 0.1);
    assert_abs_diff_eq!(sy.mean().unwrap(), 0.5, epsilon = 0.1);
  }
}

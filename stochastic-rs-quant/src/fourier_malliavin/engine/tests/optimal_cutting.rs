//! Deterministic noisy-path regression for optimal frequency selection.

use ndarray::Array1;

use super::fixtures::heston_paths;
use crate::fourier_malliavin::FMVol;
use crate::fourier_malliavin::coefficients::fourier_coefficients_dx_uniform;
use crate::fourier_malliavin::default_cutting_freq_noisy;
use crate::fourier_malliavin::optimal_cutting_frequency;

fn uniform_engine(prices: &[f64], n_freq: usize, max_freq: usize) -> FMVol<f64> {
  let n = prices.len() - 1;
  FMVol {
    dx: fourier_coefficients_dx_uniform(prices, 1.0, max_freq),
    period: 1.0,
    n,
    mesh: 1.0 / n as f64,
    origin: 0.0,
    n_freq,
    max_freq,
  }
}

#[test]
fn optimal_frequency_filters_deterministic_microstructure_noise() {
  let (log_prices, variance, times) = heston_paths();
  let dt = 1.0 / (log_prices.len() - 1) as f64;
  let expected = (0..variance.len() - 1)
    .map(|index| (variance[index] + variance[index + 1]) * 0.5 * dt)
    .sum::<f64>();
  let sigma_eta = 0.005;
  let noisy = log_prices
    .iter()
    .enumerate()
    .map(|(index, price)| {
      let noise = sigma_eta * (((index * 7919 + 104729) % 10000) as f64 / 5000.0 - 1.0);
      price + noise
    })
    .collect::<Array1<_>>();
  let result = optimal_cutting_frequency(noisy.as_slice().unwrap(), times.as_slice().unwrap());
  let (n_opt, m_opt, _) = result.cutting_freqs();
  let n = log_prices.len() - 1;
  let (n_heuristic, m_heuristic, _) = default_cutting_freq_noisy(n);
  let optimal = uniform_engine(noisy.as_slice().unwrap(), n_opt, n_opt + m_opt + 10);
  let heuristic = uniform_engine(
    noisy.as_slice().unwrap(),
    n_heuristic,
    n_heuristic + m_heuristic + 10,
  );
  let naive = FMVol::new_uniform(noisy.as_slice().unwrap(), 1.0);
  let optimal_error = (optimal.integrated_variance() - expected).abs() / expected;
  let _heuristic_error = (heuristic.integrated_variance() - expected).abs() / expected;
  let naive_error = (naive.integrated_variance() - expected).abs() / expected;

  assert!(optimal_error < naive_error);
  assert!(n_opt < n / 4);
  assert!(result.noise_variance > 0.0);
}

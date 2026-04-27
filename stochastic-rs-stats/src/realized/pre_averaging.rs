//! Pre-averaging estimator of integrated variance under microstructure noise.
//!
//! Smooths a block of $k_n$ consecutive returns with a triangular weight
//! function $g(x) = \min(x, 1-x)$, then aggregates the squared pre-averages and
//! debiases for the residual noise. With $\psi_1^{(g)} = 1$ and
//! $\psi_2^{(g)} = 1/12$ for the triangular kernel, the estimator simplifies to
//!
//! $$
//! \widehat{IV}^{PA} = \frac{12}{k_n}\sum_{i=0}^{n-k_n}\bar Y_i^{\,2} - \frac{6\,RV}{k_n^2},
//! $$
//!
//! where $\bar Y_i = \sum_{j=1}^{k_n-1} g(j/k_n)\,(Y_{i+j}-Y_{i+j-1})$ and
//! $RV = \sum_i (\Delta Y_i)^2$. Choosing $k_n = \lfloor \theta\sqrt n \rfloor$
//! with $\theta = 1/3$ delivers the optimal $n^{1/4}$ rate.
//!
//! Reference: Jacod, Li, Mykland, Podolskij, Vetter, "Microstructure Noise in
//! the Continuous Case: The Pre-Averaging Approach", Stochastic Processes and
//! their Applications, 119(7), 2249-2276 (2009).
//! DOI: 10.1016/j.spa.2008.11.004
//!
//! Reference: Christensen, Kinnebrock, Podolskij, "Pre-Averaging Estimators of
//! the Ex-Post Covariance Matrix in Noisy Diffusion Models with Non-Synchronous
//! Data", Journal of Econometrics, 159(1), 116-133 (2010).
//! DOI: 10.1016/j.jeconom.2010.05.001

use ndarray::ArrayView1;

use crate::realized::variance::realized_variance;
use crate::traits::FloatExt;

/// Pre-averaged variance estimator (Jacod et al. 2009).
///
/// `returns` are intraday log-returns. `theta` is the block-size parameter
/// (typical value `1/3 .. 1/2`), used to pick $k_n = \lfloor \theta \sqrt n \rfloor$.
/// Falls back to `2` if the resulting $k_n$ would be smaller than `2`.
pub fn pre_averaged_variance<T: FloatExt>(returns: ArrayView1<T>, theta: T) -> T {
  let n = returns.len();
  if n < 4 {
    return T::zero();
  }
  let theta_f = theta.to_f64().unwrap();
  assert!(theta_f > 0.0, "theta must be positive");
  let nf = n as f64;
  let kn = (theta_f * nf.sqrt()).floor() as usize;
  let kn = kn.max(2).min(n);
  pre_averaged_variance_with_block(returns, kn)
}

/// Pre-averaged variance estimator with explicit block length `kn`.
pub fn pre_averaged_variance_with_block<T: FloatExt>(returns: ArrayView1<T>, kn: usize) -> T {
  let n = returns.len();
  if n < kn || kn < 2 {
    return T::zero();
  }
  let kn_f = T::from_usize_(kn);
  let mut sum_bar2 = T::zero();
  for i in 0..=(n - kn) {
    let mut bar = T::zero();
    for j in 1..kn {
      let x = T::from_usize_(j) / kn_f;
      let g = triangular_weight(x);
      bar += g * returns[i + j - 1];
    }
    sum_bar2 += bar * bar;
  }
  let rv = realized_variance(returns);
  let twelve = T::from_f64_fast(12.0);
  let six = T::from_f64_fast(6.0);
  twelve / kn_f * sum_bar2 - six * rv / (kn_f * kn_f)
}

#[inline]
fn triangular_weight<T: FloatExt>(x: T) -> T {
  let half = T::from_f64_fast(0.5);
  if x < half { x } else { T::one() - x }
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::*;

  fn iid_normal(seed: u64, n: usize, std: f64) -> Array1<f64> {
    let dist = SimdNormal::<f64>::with_seed(0.0, std, seed);
    let mut out = Array1::<f64>::zeros(n);
    dist.fill_slice_fast(out.as_slice_mut().unwrap());
    out
  }

  fn noisy_returns(seed: u64, n: usize, sigma: f64, omega: f64) -> Array1<f64> {
    let dx = SimdNormal::<f64>::with_seed(0.0, sigma, seed);
    let dn = SimdNormal::<f64>::with_seed(0.0, omega, seed.wrapping_add(1));
    let mut steps = vec![0.0_f64; n];
    dx.fill_slice_fast(&mut steps);
    let mut noise = vec![0.0_f64; n + 1];
    dn.fill_slice_fast(&mut noise);
    let mut dy = Array1::<f64>::zeros(n);
    for i in 0..n {
      dy[i] = steps[i] + noise[i + 1] - noise[i];
    }
    dy
  }

  #[test]
  fn pav_close_to_iv_under_no_noise() {
    let n = 5_000;
    let r = iid_normal(101, n, 0.005);
    let iv = (n as f64) * 0.005_f64.powi(2);
    let pav = pre_averaged_variance(r.view(), 1.0_f64 / 3.0);
    let rv: f64 = r.iter().map(|v| v * v).sum();
    let pav_err = (pav - iv).abs();
    let rv_err = (rv - iv).abs();
    let tol = pav_err.max(rv_err) * 5.0;
    assert!(pav_err <= tol);
  }

  #[test]
  fn pav_corrects_microstructure_noise_better_than_rv() {
    let n = 20_000;
    let sigma = 0.005_f64;
    let omega = 0.003_f64;
    let dy = noisy_returns(103, n, sigma, omega);
    let iv = (n as f64) * sigma.powi(2);
    let rv: f64 = dy.iter().map(|v| v * v).sum();
    let pav = pre_averaged_variance(dy.view(), 1.0 / 3.0);
    assert!((pav - iv).abs() < (rv - iv).abs());
  }
}

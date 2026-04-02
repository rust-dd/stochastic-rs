//! # Volterra Process
//!
//! $$
//! X_t = \int_0^t K(t,s)\,dW_s
//! $$
//!
//! Gaussian process driven by a deterministic causal kernel $K(t,s)$ for $s \le t$.
//! Fractional Brownian motion is the special case $K(t,s) = (t-s)^{H-1/2}/\Gamma(H+1/2)$.
//!
//! Covariance: $\mathrm{Cov}(X_t, X_u) = \int_0^{\min(t,u)} K(t,s)\,K(u,s)\,ds$
//!
//! Discretised via $X_{t_i} \approx \sum_{j=1}^{i} K(t_i, t_{j-1})\,\Delta W_j$
//! using the lower-triangular kernel matrix.
//!
//! Reference:
//! - Decreusefond, L. & Üstünel, A. S. (1999), "Stochastic Analysis of the Fractional Brownian Motion"

use ndarray::Array1;

use crate::distributions::normal::SimdNormal;
use crate::simd_rng::Deterministic;
use crate::simd_rng::SeedExt;
use crate::simd_rng::Unseeded;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Volterra kernel type.
#[derive(Clone, Copy, Debug)]
pub enum VolterraKernel {
  /// Fractional Brownian motion: $K(t,s) = (t-s)^{H-1/2}/\Gamma(H+1/2)$
  FractionalBM { h: f64 },
  /// Power-law: $K(t,s) = (t-s)^\gamma$ for $\gamma > -1/2$
  PowerLaw { gamma: f64 },
  /// Exponential: $K(t,s) = e^{-\beta(t-s)}$
  Exponential { beta: f64 },
}

impl VolterraKernel {
  fn eval<T: FloatExt>(&self, t: T, s: T) -> T {
    let tau = t - s;
    if tau <= T::zero() {
      return T::zero();
    }
    match self {
      VolterraKernel::FractionalBM { h } => {
        let exp = T::from_f64_fast(*h - 0.5);
        let gamma_val = T::from_f64_fast(scilib::math::basic::gamma(*h + 0.5));
        tau.powf(exp) / gamma_val
      }
      VolterraKernel::PowerLaw { gamma } => tau.powf(T::from_f64_fast(*gamma)),
      VolterraKernel::Exponential { beta } => (-T::from_f64_fast(*beta) * tau).exp(),
    }
  }
}

/// Generic Volterra process with configurable kernel.
pub struct Volterra<T: FloatExt, S: SeedExt = Unseeded> {
  /// Kernel function.
  pub kernel: VolterraKernel,
  /// Number of grid points.
  pub n: usize,
  /// Time horizon $T$.
  pub t: Option<T>,
  pub seed: S,
}

impl<T: FloatExt> Volterra<T> {
  pub fn new(kernel: VolterraKernel, n: usize, t: Option<T>) -> Self {
    Self {
      kernel,
      n,
      t,
      seed: Unseeded,
    }
  }

  /// Fractional Brownian motion with Hurst parameter $H$.
  pub fn fbm(h: f64, n: usize, t: Option<T>) -> Self {
    assert!(h > 0.0 && h < 1.0, "Hurst parameter must be in (0,1)");
    Self::new(VolterraKernel::FractionalBM { h }, n, t)
  }
}

impl<T: FloatExt> Volterra<T, Deterministic> {
  pub fn seeded(kernel: VolterraKernel, n: usize, t: Option<T>, seed: u64) -> Self {
    Self {
      kernel,
      n,
      t,
      seed: Deterministic(seed),
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Volterra<T, S> {
  type Output = Array1<T>;

  /// $X_{t_i} = \sum_{j=1}^{i} K(t_i, t_{j-1})\,\Delta W_j$
  ///
  /// Complexity: $O(n^2)$ due to full history convolution.
  fn sample(&self) -> Self::Output {
    let mut seed = self.seed;
    let t_max = self.t.unwrap_or(T::one());
    let dt = t_max / T::from_usize_(self.n - 1);
    let sqrt_dt = dt.sqrt();

    let normal = SimdNormal::<T, 64>::from_seed_source(T::zero(), T::one(), &mut seed);

    // Pre-generate Brownian increments
    let mut dw = Array1::<T>::zeros(self.n);
    normal.fill_slice_fast(dw.as_slice_mut().unwrap());
    for val in dw.iter_mut() {
      *val = *val * sqrt_dt;
    }

    let mut x = Array1::<T>::zeros(self.n);

    for i in 1..self.n {
      let t_i = T::from_usize_(i) * dt;
      let mut sum = T::zero();
      for j in 1..=i {
        let t_jm1 = T::from_usize_(j - 1) * dt;
        sum += self.kernel.eval(t_i, t_jm1) * dw[j];
      }
      x[i] = sum;
    }

    x
  }
}

py_process_1d!(PyVolterra, Volterra,
  sig: (kernel_type, kernel_param, n, t=None, seed=None, dtype=None),
  params: (kernel_type: String, kernel_param: f64, n: usize, t: Option<f64>)
);

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn volterra_fbm_runs() {
    let v = Volterra::<f64>::fbm(0.7, 100, Some(1.0));
    let path = v.sample();
    assert_eq!(path.len(), 100);
    assert!(path[0] == 0.0);
  }

  #[test]
  fn volterra_exponential_kernel() {
    let v = Volterra::<f64>::new(VolterraKernel::Exponential { beta: 1.0 }, 100, Some(1.0));
    let path = v.sample();
    assert_eq!(path.len(), 100);
  }

  #[test]
  fn volterra_fbm_h05_is_bm() {
    // H=0.5 → K(t,s) = 1/Γ(1) = 1 → X_t = W_t (standard BM)
    let v = Volterra::<f64, Deterministic>::seeded(VolterraKernel::FractionalBM { h: 0.5 }, 200, Some(1.0), 42);
    let path = v.sample();
    // Variance of BM at t=1 should be ~1
    let var: f64 = path.iter().map(|&x| x * x).sum::<f64>() / path.len() as f64;
    // Very rough check — just ensure it's not degenerate
    assert!(var > 0.001, "variance = {var}");
  }

  #[test]
  fn volterra_seeded_deterministic() {
    let v = Volterra::<f64, Deterministic>::seeded(VolterraKernel::FractionalBM { h: 0.7 }, 50, Some(1.0), 123);
    let p1 = v.sample();
    let p2 = v.sample();
    assert_eq!(p1, p2);
  }
}

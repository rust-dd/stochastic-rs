//! # Arch
//!
//! $$
//! \sigma_t^2=\omega+\sum_{i=1}^m\alpha_iX_{t-i}^2,\qquad X_t=\sigma_t z_t,\ z_t\sim\mathcal N(0,1)
//! $$
//!
use ndarray::Array1;

use crate::distributions::normal::SimdNormal;
use crate::simd_rng::Deterministic;
use crate::simd_rng::Seed;
use crate::simd_rng::Unseeded;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Implements an ARCH(m) model:
///
/// \[
///   \sigma_t^2 = \omega + \sum_{i=1}^m \alpha_i X_{t-i}^2,
///   \quad X_t = \sigma_t \cdot z_t, \quad z_t \sim \mathcal{N}(0,1).
/// \]
///
/// # Fields
/// - `omega`: Constant term.
/// - `alpha`: Array of ARCH coefficients.
/// - `n`: Number of observations.
/// - `m`: Optional batch size.
pub struct ARCH<T: FloatExt, S: Seed = Unseeded> {
  /// Omega (constant term in variance)
  pub omega: T,
  /// Coefficients alpha_i
  pub alpha: Array1<T>,
  /// Length of series
  pub n: usize,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt> ARCH<T> {
  /// Create a new ARCH model.
  pub fn new(omega: T, alpha: Array1<T>, n: usize) -> Self {
    assert!(omega > T::zero(), "ARCH requires omega > 0");
    Self {
      omega,
      alpha,
      n,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> ARCH<T, Deterministic> {
  /// Create a new ARCH model with a deterministic seed for reproducible output.
  pub fn seeded(omega: T, alpha: Array1<T>, n: usize, seed: u64) -> Self {
    assert!(omega > T::zero(), "ARCH requires omega > 0");
    Self {
      omega,
      alpha,
      n,
      seed: Deterministic(seed),
    }
  }
}

impl<T: FloatExt, S: Seed> ProcessExt<T> for ARCH<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let m = self.alpha.len();
    let mut z = Array1::<T>::zeros(self.n);
    if self.n > 0 {
      let slice = z.as_slice_mut().expect("contiguous");
      let mut seed = self.seed;
      let normal = SimdNormal::<T>::from_seed_source(T::zero(), T::one(), &mut seed);
      normal.fill_slice_fast(slice);
    }
    let mut x = Array1::<T>::zeros(self.n);
    let var_floor = T::from_f64_fast(1e-12);

    for t in 0..self.n {
      // compute sigma_t^2
      let mut var_t = self.omega;
      for i in 1..=m {
        if t >= i {
          let x_lag = x[t - i];
          var_t += self.alpha[i - 1] * x_lag.powi(2);
        }
      }
      assert!(
        var_t.is_finite() && var_t > T::zero(),
        "ARCH produced non-positive or non-finite conditional variance at t={}",
        t
      );
      let sigma_t = var_t.max(var_floor).sqrt();
      x[t] = sigma_t * z[t];
    }

    x
  }
}

py_process_1d!(PyARCH, ARCH,
  sig: (omega, alpha, n, dtype=None),
  params: (omega: f64, alpha: Vec<f64>, n: usize)
);

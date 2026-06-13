//! # Arch
//!
//! $$
//! \sigma_t^2=\omega+\sum_{i=1}^m\alpha_iX_{t-i}^2,\qquad X_t=\sigma_t z_t,\ z_t\sim\mathcal N(0,1)
//! $$
//!
use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Implements an Arch(m) model:
///
/// \[
///   \sigma_t^2 = \omega + \sum_{i=1}^m \alpha_i X_{t-i}^2,
///   \quad X_t = \sigma_t \cdot z_t, \quad z_t \sim \mathcal{N}(0,1).
/// \]
///
/// # Fields
/// - `omega`: Constant term.
/// - `alpha`: Array of Arch coefficients.
/// - `n`: Number of observations.
/// - `m`: Optional batch size.
#[derive(Debug, Clone)]
pub struct Arch<T: FloatExt, S: SeedExt = Unseeded> {
  /// Omega (constant term in variance)
  pub omega: T,
  /// Coefficients alpha_i
  pub alpha: Array1<T>,
  /// Length of series
  pub n: usize,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> Arch<T, S> {
  /// Create a new Arch model.
  pub fn new(omega: T, alpha: Array1<T>, n: usize, seed: S) -> Self {
    assert!(omega > T::zero(), "Arch requires omega > 0");
    Self {
      omega,
      alpha,
      n,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Arch<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = ArchSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> ArchSampler<T> {
    ArchSampler {
      n: self.n,
      omega: self.omega,
      alpha: self.alpha.clone(),
      normal: SimdNormal::<T>::new(T::zero(), T::one(), &self.seed),
    }
  }
}

/// Reusable [`Arch`] sampling state: owns the standard-normal innovation source
/// and the variance coefficients so a Monte-Carlo loop pays the `SimdNormal`
/// setup once.
#[doc(hidden)]
pub struct ArchSampler<T: FloatExt> {
  n: usize,
  omega: T,
  alpha: Array1<T>,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> ArchSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    let n = out.len();
    let m = self.alpha.len();

    let mut z = Array1::<T>::zeros(n);
    if n > 0 {
      let slice = z.as_slice_mut().expect("contiguous");
      self.normal.fill_slice_fast(slice);
    }
    let var_floor = T::from_f64_fast(1e-12);

    for t in 0..n {
      // compute sigma_t^2
      let mut var_t = self.omega;
      for i in 1..=m {
        if t >= i {
          let x_lag = out[t - i];
          var_t += self.alpha[i - 1] * x_lag.powi(2);
        }
      }
      assert!(
        var_t.is_finite() && var_t > T::zero(),
        "Arch produced non-positive or non-finite conditional variance at t={}",
        t
      );
      let sigma_t = var_t.max(var_floor).sqrt();
      out[t] = sigma_t * z[t];
    }
  }
}

impl<T: FloatExt> PathSampler<T> for ArchSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("Arch output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyArch, Arch,
  sig: (omega, alpha, n, seed=None, dtype=None),
  params: (omega: f64, alpha: Vec<f64>, n: usize)
);

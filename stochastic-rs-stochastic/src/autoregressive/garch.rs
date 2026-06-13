//! # Garch
//!
//! $$
//! \sigma_t^2=\omega+\sum_{i=1}^p\alpha_iX_{t-i}^2+\sum_{j=1}^q\beta_j\sigma_{t-j}^2,\qquad X_t=\sigma_t z_t
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

/// Implements a general Garch(p,q) model.
///
/// \[
///   \sigma_t^2
///     = \omega
///       + \sum_{i=1}^p \alpha_i \, X_{t-i}^2
///       + \sum_{j=1}^q \beta_j \, \sigma_{t-j}^2,
///   \quad X_t = \sigma_t \, z_t, \quad z_t \sim \mathcal{N}(0,1).
/// \]
///
/// # Parameters
/// - `omega`: Constant term (\(\omega\)) in the Garch variance equation.
/// - `alpha`: Array \(\{\alpha_1, \ldots, \alpha_p\}\) for past squared observations.
/// - `beta`:  Array \(\{\beta_1, \ldots, \beta_q\}\) for past variances.
/// - `n`:     Length of the time series.
/// - `m`:     Optional batch size (unused by default).
///
/// # Notes
/// 1. Stationarity typically requires \(\sum \alpha_i + \sum \beta_j < 1\).
/// 2. We initialize with an unconditional variance approximation for \(\sigma_0^2\).
#[derive(Debug, Clone)]
pub struct Garch<T: FloatExt, S: SeedExt = Unseeded> {
  /// Constant term in conditional variance dynamics.
  pub omega: T,
  /// Model shape / loading parameter.
  pub alpha: Array1<T>,
  /// Model slope / loading parameter.
  pub beta: Array1<T>,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> Garch<T, S> {
  pub fn new(omega: T, alpha: Array1<T>, beta: Array1<T>, n: usize, seed: S) -> Self {
    assert!(omega > T::zero(), "Garch requires omega > 0");
    Garch {
      omega,
      alpha,
      beta,
      n,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Garch<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = GarchSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> GarchSampler<T> {
    GarchSampler {
      n: self.n,
      omega: self.omega,
      alpha: self.alpha.clone(),
      beta: self.beta.clone(),
      normal: SimdNormal::<T>::new(T::zero(), T::one(), &self.seed),
    }
  }
}

/// Reusable [`Garch`] sampling state: owns the standard-normal innovation source
/// and the variance coefficients so a Monte-Carlo loop pays the `SimdNormal`
/// setup once.
#[doc(hidden)]
pub struct GarchSampler<T: FloatExt> {
  n: usize,
  omega: T,
  alpha: Array1<T>,
  beta: Array1<T>,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> GarchSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    let n = out.len();
    let p = self.alpha.len();
    let q = self.beta.len();

    // Generate white noise z_t ~ N(0,1)
    let mut z = Array1::<T>::zeros(n);
    if n > 0 {
      let slice = z.as_slice_mut().expect("contiguous");
      self.normal.fill_slice_fast(slice);
    }

    // Array for sigma_t^2 (the output buffer holds X_t)
    let mut sigma2 = Array1::<T>::zeros(n);
    let var_floor = T::from_f64_fast(1e-12);

    // Sum of alpha/beta for unconditional variance initialization
    let sum_alpha = self.alpha.iter().cloned().sum();
    let sum_beta = self.beta.iter().cloned().sum();
    let denom = T::one() - sum_alpha - sum_beta;
    assert!(
      denom > T::zero(),
      "Garch requires sum(alpha) + sum(beta) < 1 for finite unconditional variance"
    );

    for t in 0..n {
      if t == 0 {
        sigma2[t] = self.omega / denom;
      } else {
        let mut var_t = self.omega;

        // Sum alpha_i * X_{t-i}^2
        for i in 1..=p {
          if t >= i {
            var_t += self.alpha[i - 1] * out[t - i].powi(2);
          }
        }
        // Sum beta_j * sigma2[t-j]
        for j in 1..=q {
          if t >= j {
            var_t += self.beta[j - 1] * sigma2[t - j];
          }
        }
        sigma2[t] = var_t;
      }
      assert!(
        sigma2[t].is_finite() && sigma2[t] > T::zero(),
        "Garch produced non-positive or non-finite conditional variance at t={}",
        t
      );
      // X_t = sigma_t * z[t]
      out[t] = sigma2[t].max(var_floor).sqrt() * z[t];
    }
  }
}

impl<T: FloatExt> PathSampler<T> for GarchSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("Garch output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyGarch, Garch,
  sig: (omega, alpha, beta, n, seed=None, dtype=None),
  params: (omega: f64, alpha: Vec<f64>, beta: Vec<f64>, n: usize)
);

//! # Sarima
//!
//! $$
//! \Phi(B^s)\phi(B)(1-B)^d(1-B^s)^DX_t=\Theta(B^s)\theta(B)\varepsilon_t
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

/// Implements a Sarima model, often denoted:
///
/// Sarima(p, d, q) (P, D, Q)_s
///
/// using the "backshift" definition:
/// \[
///   \Phi(B^s)\,\phi(B)\,(1 - B)^d (1 - B^s)^D\,X_t
///     = \Theta(B^s)\,\theta(B)\,\epsilon_t,
/// \]
/// where:
/// - \(\phi(B)\) and \(\theta(B)\) capture the non-seasonal AR/MA parts (orders p,q),
/// - \(\Phi(B^s)\) and \(\Theta(B^s)\) capture the seasonal AR/MA parts (orders P,Q) with season length \(s\),
/// - \(d\) is the non-seasonal differencing order,
/// - \(D\) is the seasonal differencing order,
/// - \(\epsilon_t\) is white noise with std dev \(\sigma\).
///
/// # Fields
/// - `non_seasonal_ar_coefs` (\(\phi\)): array of length p.
/// - `non_seasonal_ma_coefs` (\(\theta\)): array of length q.
/// - `seasonal_ar_coefs` (\(\Phi\)): array of length P.
/// - `seasonal_ma_coefs` (\(\Theta\)): array of length Q.
/// - `d`: Non-seasonal differencing order.
/// - `D`: Seasonal differencing order.
/// - `s`: Season length.
/// - `sigma`: Std dev of the white noise.
/// - `n`: Length of the final time series.
/// - `m`: Optional batch size (unused by default).
///
/// # Implementation Notes
/// 1. We multiply the non-seasonal and seasonal AR polynomials (and likewise the MA polynomials)
///    to produce the combined polynomial with cross-terms.
/// 2. A single-pass SARMA recursion generates the "fully differenced" data.
/// 3. We invert the seasonal differencing D times (lag s) and then invert the non-seasonal differencing d times to recover X_t.
#[derive(Debug, Clone)]
pub struct Sarima<T: FloatExt, S: SeedExt = Unseeded> {
  /// Non-seasonal AR coefficients, length p
  pub non_seasonal_ar_coefs: Array1<T>,
  /// Non-seasonal MA coefficients, length q
  pub non_seasonal_ma_coefs: Array1<T>,
  /// Seasonal AR coefficients, length P
  pub seasonal_ar_coefs: Array1<T>,
  /// Seasonal MA coefficients, length Q
  pub seasonal_ma_coefs: Array1<T>,
  /// Non-seasonal differencing (d)
  pub d: usize,
  /// Seasonal differencing (D)
  pub D: usize,
  /// Season length
  pub s: usize,
  /// Noise std dev
  pub sigma: T,
  /// Final length of the time series
  pub n: usize,
  /// Seed strategy (compile-time: [`Unseeded`] or [`Deterministic`]).
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> Sarima<T, S> {
  /// Create a new Sarima model
  #[allow(non_snake_case)]
  pub fn new(
    non_seasonal_ar_coefs: Array1<T>,
    non_seasonal_ma_coefs: Array1<T>,
    seasonal_ar_coefs: Array1<T>,
    seasonal_ma_coefs: Array1<T>,
    d: usize,
    D: usize,
    s: usize,
    sigma: T,
    n: usize,
    seed: S,
  ) -> Self {
    assert!(sigma > T::zero(), "Sarima requires sigma > 0");
    assert!(s > 0, "Sarima requires season length s > 0");
    Sarima {
      non_seasonal_ar_coefs,
      non_seasonal_ma_coefs,
      seasonal_ar_coefs,
      seasonal_ma_coefs,
      d,
      D,
      s,
      sigma,
      n,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Sarima<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = SarimaSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> SarimaSampler<T> {
    SarimaSampler {
      n: self.n,
      non_seasonal_ar_coefs: self.non_seasonal_ar_coefs.clone(),
      non_seasonal_ma_coefs: self.non_seasonal_ma_coefs.clone(),
      seasonal_ar_coefs: self.seasonal_ar_coefs.clone(),
      seasonal_ma_coefs: self.seasonal_ma_coefs.clone(),
      d: self.d,
      big_d: self.D,
      s: self.s,
      normal: SimdNormal::<T>::new(T::zero(), self.sigma, &self.seed),
    }
  }
}

/// Reusable [`Sarima`] sampling state: owns the Gaussian innovation source and
/// the seasonal/non-seasonal ARMA coefficients so a Monte-Carlo loop pays the
/// `SimdNormal` setup once.
#[doc(hidden)]
pub struct SarimaSampler<T: FloatExt> {
  n: usize,
  non_seasonal_ar_coefs: Array1<T>,
  non_seasonal_ma_coefs: Array1<T>,
  seasonal_ar_coefs: Array1<T>,
  seasonal_ma_coefs: Array1<T>,
  d: usize,
  big_d: usize,
  s: usize,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> SarimaSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    let n = out.len();

    let mut noise = Array1::<T>::zeros(n);
    if n > 0 {
      let slice = noise.as_slice_mut().expect("contiguous");
      self.normal.fill_slice_fast(slice);
    }

    // Multiply φ(B) and Φ(Bˢ) to get the combined AR polynomial.
    // φ(B) = 1 - φ_1 B - ... - φ_p B^p
    // Φ(Bˢ) = 1 - Φ_1 B^s - ... - Φ_P B^{Ps}
    // Product polynomial has terms at lags: i + j*s for all combinations.
    let ar_lags =
      multiply_ar_polynomials(&self.non_seasonal_ar_coefs, &self.seasonal_ar_coefs, self.s);

    // Multiply θ(B) and Θ(Bˢ) to get the combined MA polynomial.
    // θ(B) = 1 + θ_1 B + ... + θ_q B^q
    // Θ(Bˢ) = 1 + Θ_1 B^s + ... + Θ_Q B^{Qs}
    let ma_lags =
      multiply_ma_polynomials(&self.non_seasonal_ma_coefs, &self.seasonal_ma_coefs, self.s);

    // Single-pass SARMA recursion:
    // W_t = sum(ar_coef_k * W_{t-k}) + eps_t + sum(ma_coef_k * eps_{t-k})
    let mut sarma_series = Array1::<T>::zeros(n);

    for t in 0..n {
      let mut val = noise[t];

      for &(lag, coef) in &ar_lags {
        if t >= lag {
          val += coef * sarma_series[t - lag];
        }
      }

      for &(lag, coef) in &ma_lags {
        if t >= lag {
          val += coef * noise[t - lag];
        }
      }

      sarma_series[t] = val;
    }

    // Invert seasonal differencing D times, then non-seasonal differencing d times
    let mut integrated = sarma_series;
    for _ in 0..self.big_d {
      integrated = inverse_seasonal_difference(&integrated, self.s);
    }
    for _ in 0..self.d {
      integrated = inverse_difference(&integrated);
    }

    out.copy_from_slice(integrated.as_slice().expect("contiguous"));
  }
}

impl<T: FloatExt> PathSampler<T> for SarimaSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("Sarima output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

/// Multiply the non-seasonal AR polynomial φ(B) with the seasonal AR polynomial Φ(Bˢ).
///
/// φ(B) = 1 - φ_1 B - ... - φ_p B^p
/// Φ(Bˢ) = 1 - Φ_1 B^s - ... - Φ_P B^{Ps}
///
/// Returns a vector of (lag, coefficient) pairs for the product polynomial
/// (excluding the constant term 1). The coefficients are the positive form
/// used in the recursion: W_t = sum(coef * W_{t-lag}) + ...
fn multiply_ar_polynomials<T: FloatExt>(
  non_seasonal: &Array1<T>,
  seasonal: &Array1<T>,
  s: usize,
) -> Vec<(usize, T)> {
  let p = non_seasonal.len();
  let big_p = seasonal.len();
  let max_lag = p + big_p * s;

  let mut combined = vec![T::zero(); max_lag + 1];

  // φ(B) contributes lags 1..p
  for i in 0..p {
    combined[i + 1] += non_seasonal[i];
  }

  // Φ(Bˢ) contributes lags s, 2s, ..., Ps
  for j in 0..big_p {
    let lag_j = (j + 1) * s;
    combined[lag_j] += seasonal[j];
  }

  // Cross-terms: -(-φ_i)(-Φ_j) = -φ_i*Φ_j at lag i+1+j*s
  // In recursion form (positive): the cross-term subtracts
  for i in 0..p {
    for j in 0..big_p {
      let lag = (i + 1) + (j + 1) * s;
      combined[lag] -= non_seasonal[i] * seasonal[j];
    }
  }

  combined
    .into_iter()
    .enumerate()
    .filter(|&(lag, _)| lag > 0)
    .filter(|&(_, c)| c != T::zero())
    .collect()
}

/// Multiply the non-seasonal MA polynomial θ(B) with the seasonal MA polynomial Θ(Bˢ).
///
/// θ(B) = 1 + θ_1 B + ... + θ_q B^q
/// Θ(Bˢ) = 1 + Θ_1 B^s + ... + Θ_Q B^{Qs}
///
/// Returns a vector of (lag, coefficient) pairs for the product polynomial
/// (excluding the constant term 1).
fn multiply_ma_polynomials<T: FloatExt>(
  non_seasonal: &Array1<T>,
  seasonal: &Array1<T>,
  s: usize,
) -> Vec<(usize, T)> {
  let q = non_seasonal.len();
  let big_q = seasonal.len();
  let max_lag = q + big_q * s;

  let mut combined = vec![T::zero(); max_lag + 1];

  // θ(B) contributes lags 1..q
  for i in 0..q {
    combined[i + 1] += non_seasonal[i];
  }

  // Θ(Bˢ) contributes lags s, 2s, ..., Qs
  for j in 0..big_q {
    let lag_j = (j + 1) * s;
    combined[lag_j] += seasonal[j];
  }

  // Cross-terms: θ_i * Θ_j at lag i+1+j*s
  for i in 0..q {
    for j in 0..big_q {
      let lag = (i + 1) + (j + 1) * s;
      combined[lag] += non_seasonal[i] * seasonal[j];
    }
  }

  combined
    .into_iter()
    .enumerate()
    .filter(|&(lag, _)| lag > 0)
    .filter(|&(_, c)| c != T::zero())
    .collect()
}

fn inverse_difference<T: FloatExt>(y: &Array1<T>) -> Array1<T> {
  let n = y.len();
  if n == 0 {
    return y.clone();
  }
  let mut x = Array1::<T>::zeros(n);
  x[0] = y[0];
  for t in 1..n {
    x[t] = x[t - 1] + y[t];
  }
  x
}

fn inverse_seasonal_difference<T: FloatExt>(y: &Array1<T>, s: usize) -> Array1<T> {
  let n = y.len();
  if n == 0 || s == 0 {
    return y.clone();
  }
  let mut x = Array1::<T>::zeros(n);
  for t in 0..s.min(n) {
    x[t] = y[t];
  }
  for t in s..n {
    x[t] = x[t - s] + y[t];
  }
  x
}

py_process_1d!(PySarima, Sarima,
  sig: (non_seasonal_ar_coefs, non_seasonal_ma_coefs, seasonal_ar_coefs, seasonal_ma_coefs, d, d_seasonal, s, sigma, n, seed=None, dtype=None),
  params: (non_seasonal_ar_coefs: Vec<f64>, non_seasonal_ma_coefs: Vec<f64>, seasonal_ar_coefs: Vec<f64>, seasonal_ma_coefs: Vec<f64>, d: usize, d_seasonal: usize, s: usize, sigma: f64, n: usize)
);

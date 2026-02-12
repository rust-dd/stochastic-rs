use ndarray::Array1;

use crate::stochastic::noise::wn::Wn;
use crate::stochastic::Float;
use crate::stochastic::ProcessExt;

/// Implements a SARIMA model, often denoted:
///
/// SARIMA(p, d, q) (P, D, Q)_s
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
pub struct SARIMA<T: Float> {
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
  wn: Wn<T>,
}

impl<T: Float> SARIMA<T> {
  /// Create a new SARIMA model
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
  ) -> Self {
    SARIMA {
      non_seasonal_ar_coefs,
      non_seasonal_ma_coefs,
      seasonal_ar_coefs,
      seasonal_ma_coefs,
      d,
      D,
      s,
      sigma,
      n,
      wn: Wn::new(n, None, Some(sigma)),
    }
  }
}

impl<T: Float> ProcessExt<T> for SARIMA<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let noise = self.wn.sample();

    // Multiply φ(B) and Φ(Bˢ) to get the combined AR polynomial.
    // φ(B) = 1 - φ_1 B - ... - φ_p B^p
    // Φ(Bˢ) = 1 - Φ_1 B^s - ... - Φ_P B^{Ps}
    // Product polynomial has terms at lags: i + j*s for all combinations.
    let ar_lags =
      Self::multiply_ar_polynomials(&self.non_seasonal_ar_coefs, &self.seasonal_ar_coefs, self.s);

    // Multiply θ(B) and Θ(Bˢ) to get the combined MA polynomial.
    // θ(B) = 1 + θ_1 B + ... + θ_q B^q
    // Θ(Bˢ) = 1 + Θ_1 B^s + ... + Θ_Q B^{Qs}
    let ma_lags =
      Self::multiply_ma_polynomials(&self.non_seasonal_ma_coefs, &self.seasonal_ma_coefs, self.s);

    // Single-pass SARMA recursion:
    // W_t = sum(ar_coef_k * W_{t-k}) + eps_t + sum(ma_coef_k * eps_{t-k})
    let mut sarma_series = Array1::<T>::zeros(self.n);

    for t in 0..self.n {
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
    for _ in 0..self.D {
      integrated = Self::inverse_seasonal_difference(&integrated, self.s);
    }
    for _ in 0..self.d {
      integrated = Self::inverse_difference(&integrated);
    }

    integrated
  }
}

impl<T: Float> SARIMA<T> {
  /// Multiply the non-seasonal AR polynomial φ(B) with the seasonal AR polynomial Φ(Bˢ).
  ///
  /// φ(B) = 1 - φ_1 B - ... - φ_p B^p
  /// Φ(Bˢ) = 1 - Φ_1 B^s - ... - Φ_P B^{Ps}
  ///
  /// Returns a vector of (lag, coefficient) pairs for the product polynomial
  /// (excluding the constant term 1). The coefficients are the positive form
  /// used in the recursion: W_t = sum(coef * W_{t-lag}) + ...
  fn multiply_ar_polynomials(
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
  fn multiply_ma_polynomials(
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

  fn inverse_difference(y: &Array1<T>) -> Array1<T> {
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

  fn inverse_seasonal_difference(y: &Array1<T>, s: usize) -> Array1<T> {
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
}

use impl_new_derive::ImplNew;
use ndarray::Array1;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::stochastic::SamplingExt;

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
/// 1. We generate a naive "SARMA" by summing four components:
///    - Non-seasonal AR(p),
///    - Non-seasonal MA(q),
///    - Seasonal AR(P) at lag multiples of s,
///    - Seasonal MA(Q) at lag multiples of s.
/// 2. That sum is interpreted as the "fully differenced" data, i.e., \(\Delta^d \Delta_s^D X_t\).
/// 3. We invert the seasonal differencing D times (lag s) and then invert the non-seasonal differencing d times to recover X_t.
#[derive(ImplNew)]
pub struct SARIMA<T> {
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
  /// Optional batch size
  pub m: Option<usize>,
}

#[cfg(feature = "f64")]
impl SamplingExt<f64> for SARIMA<f64> {
  fn sample(&self) -> Array1<f64> {
    // Generate white noise for dimension n
    let noise = Array1::random(self.n, Normal::new(0.0, self.sigma).unwrap());

    // 1) Construct naive "SARMA" by combining:
    //    - AR(p) + MA(q)
    //    - Seasonal AR(P) + Seasonal MA(Q)
    //    We'll do this in a single pass to fill an array of length n.
    let mut sarma_series = Array1::<f64>::zeros(self.n);

    // We'll store an array of past noise for referencing in MA calculations
    // (non-seasonal and seasonal). The simplest approach is to just use `noise[t - k]`
    // if t >= k, or skip otherwise.

    // Fill sarma_series[t] by summing:
    //   - Non-seasonal AR from lag 1..p
    //   - Seasonal AR from lag s, 2s, ..., P*s
    //   - Non-seasonal MA from lag 1..q
    //   - Seasonal MA from lag s, 2s, ..., Q*s
    //   + current noise
    for t in 0..self.n {
      let mut val = 0.0;

      // Current noise is always added for MA structure
      val += noise[t];

      // Non-seasonal AR part
      for (lag_idx, &phi) in self.non_seasonal_ar_coefs.iter().enumerate() {
        let k = lag_idx + 1; // backshift exponent
        if t >= k {
          val += phi * sarma_series[t - k];
        }
      }

      // Seasonal AR part
      for (lag_idx, &phi_s) in self.seasonal_ar_coefs.iter().enumerate() {
        let k = (lag_idx + 1) * self.s;
        if t >= k {
          val += phi_s * sarma_series[t - k];
        }
      }

      // Non-seasonal MA part
      for (lag_idx, &theta) in self.non_seasonal_ma_coefs.iter().enumerate() {
        let k = lag_idx + 1;
        if t >= k {
          val += theta * noise[t - k];
        }
      }

      // Seasonal MA part
      for (lag_idx, &theta_s) in self.seasonal_ma_coefs.iter().enumerate() {
        let k = (lag_idx + 1) * self.s;
        if t >= k {
          val += theta_s * noise[t - k];
        }
      }

      sarma_series[t] = val;
    }

    // 2) Interpret sarma_series as (1-B)^d (1-B^s)^D X_t,
    //    so we do inverse differencing:
    //    a) Seasonal differencing (1-B^s)^{-D}
    //    b) Non-seasonal differencing (1-B)^{-d}

    // a) Invert seasonal differencing D times
    let mut integrated = sarma_series;
    for _ in 0..self.D {
      integrated = inverse_seasonal_difference(&integrated, self.s);
    }

    // b) Invert non-seasonal differencing d times
    for _ in 0..self.d {
      integrated = inverse_difference(&integrated);
    }

    // integrated is now X_t = SARIMA(...) of length n
    integrated
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Array1<f64> {
    use crate::stats::distr::normal::SimdNormal;

    // Generate white noise for dimension n
    let noise = Array1::random(self.n, SimdNormal::new(0.0, self.sigma as f32));

    let mut sarma_series = Array1::<f64>::zeros(self.n);

    for t in 0..self.n {
      let mut val = 0.0;

      // Current noise is always added for MA structure
      val += noise[t] as f64;

      // Non-seasonal AR part
      for (lag_idx, &phi) in self.non_seasonal_ar_coefs.iter().enumerate() {
        let k = lag_idx + 1; // backshift exponent
        if t >= k {
          val += phi * sarma_series[t - k];
        }
      }

      // Seasonal AR part
      for (lag_idx, &phi_s) in self.seasonal_ar_coefs.iter().enumerate() {
        let k = (lag_idx + 1) * self.s;
        if t >= k {
          val += phi_s * sarma_series[t - k];
        }
      }

      // Non-seasonal MA part
      for (lag_idx, &theta) in self.non_seasonal_ma_coefs.iter().enumerate() {
        let k = lag_idx + 1;
        if t >= k {
          val += theta * noise[t - k] as f64;
        }
      }

      // Seasonal MA part
      for (lag_idx, &theta_s) in self.seasonal_ma_coefs.iter().enumerate() {
        let k = (lag_idx + 1) * self.s;
        if t >= k {
          val += theta_s * noise[t - k] as f64;
        }
      }

      sarma_series[t] = val;
    }

    let mut integrated = sarma_series;
    for _ in 0..self.D {
      integrated = inverse_seasonal_difference(&integrated, self.s);
    }

    for _ in 0..self.d {
      integrated = inverse_difference(&integrated);
    }

    integrated
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}

#[cfg(feature = "f32")]
impl SamplingExt<f32> for SARIMA<f32> {
  fn sample(&self) -> Array1<f32> {
    let noise =
      Array1::random(self.n, Normal::new(0.0, self.sigma as f64).unwrap()).mapv(|x| x as f32);

    let mut sarma_series = Array1::<f32>::zeros(self.n);

    for t in 0..self.n {
      let mut val = 0.0;
      val += noise[t];

      for (lag_idx, &phi) in self.non_seasonal_ar_coefs.iter().enumerate() {
        let k = lag_idx + 1;
        if t >= k {
          val += phi * sarma_series[t - k];
        }
      }

      for (lag_idx, &phi_s) in self.seasonal_ar_coefs.iter().enumerate() {
        let k = (lag_idx + 1) * self.s;
        if t >= k {
          val += phi_s * sarma_series[t - k];
        }
      }

      for (lag_idx, &theta) in self.non_seasonal_ma_coefs.iter().enumerate() {
        let k = lag_idx + 1;
        if t >= k {
          val += theta * noise[t - k];
        }
      }

      for (lag_idx, &theta_s) in self.seasonal_ma_coefs.iter().enumerate() {
        let k = (lag_idx + 1) * self.s;
        if t >= k {
          val += theta_s * noise[t - k];
        }
      }

      sarma_series[t] = val;
    }

    let mut integrated = sarma_series;
    for _ in 0..self.D {
      integrated = inverse_seasonal_difference_f32(&integrated, self.s);
    }

    for _ in 0..self.d {
      integrated = inverse_difference_f32(&integrated);
    }

    integrated
  }

  #[cfg(feature = "simd")]
  fn sample_simd(&self) -> Array1<f32> {
    use crate::stats::distr::normal::SimdNormal;

    let noise = Array1::random(self.n, SimdNormal::new(0.0, self.sigma));

    let mut sarma_series = Array1::<f32>::zeros(self.n);

    for t in 0..self.n {
      let mut val = 0.0;
      val += noise[t];

      for (lag_idx, &phi) in self.non_seasonal_ar_coefs.iter().enumerate() {
        let k = lag_idx + 1;
        if t >= k {
          val += phi * sarma_series[t - k];
        }
      }

      for (lag_idx, &phi_s) in self.seasonal_ar_coefs.iter().enumerate() {
        let k = (lag_idx + 1) * self.s;
        if t >= k {
          val += phi_s * sarma_series[t - k];
        }
      }

      for (lag_idx, &theta) in self.non_seasonal_ma_coefs.iter().enumerate() {
        let k = lag_idx + 1;
        if t >= k {
          val += theta * noise[t - k];
        }
      }

      for (lag_idx, &theta_s) in self.seasonal_ma_coefs.iter().enumerate() {
        let k = (lag_idx + 1) * self.s;
        if t >= k {
          val += theta_s * noise[t - k];
        }
      }

      sarma_series[t] = val;
    }

    let mut integrated = sarma_series;
    for _ in 0..self.D {
      integrated = inverse_seasonal_difference_f32(&integrated, self.s);
    }

    for _ in 0..self.d {
      integrated = inverse_difference_f32(&integrated);
    }

    integrated
  }

  fn n(&self) -> usize {
    self.n
  }

  fn m(&self) -> Option<usize> {
    self.m
  }
}

/// Inverse *non-seasonal* differencing
///
/// If Y = (1-B)X, then
///   X[0] = Y[0],
///   X[t] = X[t-1] + Y[t].
fn inverse_difference(y: &Array1<f64>) -> Array1<f64> {
  let n = y.len();
  if n == 0 {
    return y.clone();
  }
  let mut x = Array1::<f64>::zeros(n);
  x[0] = y[0];
  for t in 1..n {
    x[t] = x[t - 1] + y[t];
  }
  x
}

/// Inverse *seasonal* differencing (1 - B^s)^{-1} once.
///
/// If Y = (1 - B^s)X, then
///   X[t] = X[t - s] + Y[t],   for t >= s,
///   with X[t] = Y[t] if t < s.
///
fn inverse_seasonal_difference(y: &Array1<f64>, s: usize) -> Array1<f64> {
  let n = y.len();
  if n == 0 || s == 0 {
    return y.clone();
  }
  let mut x = Array1::<f64>::zeros(n);
  // For t < s, X[t] = Y[t] since X[t-s] doesn't exist
  for t in 0..s.min(n) {
    x[t] = y[t];
  }
  // For t >= s, X[t] = X[t-s] + Y[t]
  for t in s..n {
    x[t] = x[t - s] + y[t];
  }
  x
}

#[cfg(feature = "f32")]
fn inverse_difference_f32(y: &Array1<f32>) -> Array1<f32> {
  let n = y.len();
  if n == 0 {
    return y.clone();
  }
  let mut x = Array1::<f32>::zeros(n);
  x[0] = y[0];
  for t in 1..n {
    x[t] = x[t - 1] + y[t];
  }
  x
}

#[cfg(feature = "f32")]
fn inverse_seasonal_difference_f32(y: &Array1<f32>, s: usize) -> Array1<f32> {
  let n = y.len();
  if n == 0 || s == 0 {
    return y.clone();
  }
  let mut x = Array1::<f32>::zeros(n);
  for t in 0..s.min(n) {
    x[t] = y[t];
  }
  for t in s..n {
    x[t] = x[t - s] + y[t];
  }
  x
}

#[cfg(test)]
mod tests {
  use ndarray::arr1;

  use crate::plot_1d;
  use crate::stochastic::autoregressive::sarima::SARIMA;
  use crate::stochastic::SamplingExt;

  #[test]
  fn sarima_plot() {
    let non_seasonal_ar = arr1(&[-0.4, 0.2]);
    let non_seasonal_ma = arr1(&[0.3]);
    let seasonal_ar = arr1(&[0.2]);
    let seasonal_ma = arr1(&[0.1]);
    let s = 12;

    let sarima = SARIMA::new(
      non_seasonal_ar,
      non_seasonal_ma,
      seasonal_ar,
      seasonal_ma,
      2,   // d
      2,   // D
      s,   // season length
      1.0, // sigma
      120, // n
      None,
    );
    plot_1d!(sarima.sample(), "SARIMA(p,d,q)(P,D,Q)_s process");
  }
}

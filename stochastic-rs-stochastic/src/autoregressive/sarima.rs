//! # Sarima
//!
//! $$
//! \Phi(B^s)\phi(B)(1-B)^d(1-B^s)^DX_t=\Theta(B^s)\theta(B)\varepsilon_t
//! $$
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::precondition::assert_entries;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// SARIMA(p, d, q)(P, D, Q)ₛ. The seasonal and plain polynomials are multiplied with their
/// cross-terms, run as one SARMA pass, then the seasonal and the plain differences are inverted.
#[derive(Debug, Clone)]
pub struct Sarima<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
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
  pub seasonal_d: usize,
  /// Season length
  pub s: usize,
  /// Noise std dev
  pub sigma: T,
  /// Final length of the time series
  pub n: usize,
  /// Seed: [`Unseeded`] or a [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Sarima<T, S> {
  /// Panics unless `n >= 1`, `s > 0`, `sigma >= 0`, and `sigma` and every coefficient are finite.
  pub fn new(
    non_seasonal_ar_coefs: Array1<T>,
    non_seasonal_ma_coefs: Array1<T>,
    seasonal_ar_coefs: Array1<T>,
    seasonal_ma_coefs: Array1<T>,
    d: usize,
    seasonal_d: usize,
    s: usize,
    sigma: T,
    n: usize,
    seed: S,
  ) -> Self {
    assert!(
      sigma.is_finite(),
      "sigma must satisfy `sigma.is_finite()`, got sigma = {sigma:?}"
    );
    assert_entries(
      "non_seasonal_ar_coefs",
      "non_seasonal_ar_coefs[i].is_finite()",
      &non_seasonal_ar_coefs,
      |v| v.is_finite(),
    );
    assert_entries(
      "non_seasonal_ma_coefs",
      "non_seasonal_ma_coefs[i].is_finite()",
      &non_seasonal_ma_coefs,
      |v| v.is_finite(),
    );
    assert_entries(
      "seasonal_ar_coefs",
      "seasonal_ar_coefs[i].is_finite()",
      &seasonal_ar_coefs,
      |v| v.is_finite(),
    );
    assert_entries(
      "seasonal_ma_coefs",
      "seasonal_ma_coefs[i].is_finite()",
      &seasonal_ma_coefs,
      |v| v.is_finite(),
    );
    assert!(n >= 1, "n must satisfy `n >= 1`, got n = {n:?}");
    assert!(
      sigma >= T::zero(),
      "sigma must satisfy `sigma >= T::zero()`, got sigma = {sigma:?}"
    );
    assert!(s > 0, "s must satisfy `s > 0`, got s = {s:?}");
    Sarima {
      backend: Cpu,
      non_seasonal_ar_coefs,
      non_seasonal_ma_coefs,
      seasonal_ar_coefs,
      seasonal_ma_coefs,
      d,
      seasonal_d,
      s,
      sigma,
      n,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Sarima<T, S, B> {
  /// `sigma` as host and device read it; a write past `new` that breaks its rule fails here.
  fn checked_sigma(&self) -> T {
    let sigma = self.sigma;
    assert!(
      sigma.is_finite() && sigma >= T::zero(),
      "sigma must satisfy `sigma.is_finite() && sigma >= T::zero()`, got sigma = {sigma:?}"
    );
    sigma
  }
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for Sarima<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::MovingAverageFilter {
      sigma: self.checked_sigma(),
    }
  }

  fn initial_value(&self) -> T {
    T::zero()
  }

  /// The first point is itself a draw, as on the host.
  fn step_first(&self) -> bool {
    true
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  /// A unit step per point: the series has no time of its own.
  fn horizon(&self) -> T {
    T::from_usize_(self.n)
  }

  fn time_step(&self) -> T {
    T::one()
  }

  /// The impulse response of this SARIMA's own recursion, so the device's convolution of the
  /// innovations is that recursion exactly, seasonal terms and differencing included.
  fn curves(&self) -> Option<Vec<Vec<T>>> {
    Some(vec![super::arima::impulse_response(self.n, |unit| {
      sarima_filter(
        unit,
        &self.non_seasonal_ar_coefs,
        &self.non_seasonal_ma_coefs,
        &self.seasonal_ar_coefs,
        &self.seasonal_ma_coefs,
        self.d,
        self.seasonal_d,
        self.s,
      )
    })])
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Sarima<T, S> { non_seasonal_ar_coefs, non_seasonal_ma_coefs, seasonal_ar_coefs, seasonal_ma_coefs, d, seasonal_d, s, sigma, n, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Sarima<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Sarima<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = SarimaSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> SarimaSampler<T> {
    let sigma = self.checked_sigma();
    SarimaSampler {
      n: self.n,
      non_seasonal_ar_coefs: self.non_seasonal_ar_coefs.clone(),
      non_seasonal_ma_coefs: self.non_seasonal_ma_coefs.clone(),
      seasonal_ar_coefs: self.seasonal_ar_coefs.clone(),
      seasonal_ma_coefs: self.seasonal_ma_coefs.clone(),
      d: self.d,
      big_d: self.seasonal_d,
      s: self.s,
      normal: (sigma != T::zero())
        .then(|| SimdNormal::<T>::new(T::zero(), sigma).seeded(&self.seed)),
    }
  }

  /// Through the Euler engine when the series fits the kernels' per-path history; a longer
  /// series stays on the host, chunked as [`ProcessExt`] chunks.
  fn sample(&self) -> Array1<T> {
    if self.device_ready() {
      self.backend.euler_sample(self)
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    if self.device_ready() {
      self.backend.euler_paths_map(self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, f)
    }
  }

  fn sample_map_view<R: Send>(
    &self,
    m: usize,
    f: impl Fn(ndarray::ArrayView1<T>) -> R + Sync,
  ) -> Vec<R> {
    if self.device_ready() {
      self.backend.euler_paths_map_view(self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, |path| f(path.view()))
    }
  }

  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    if self.device_ready() {
      self.backend.euler_paths(self, m)
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<Array1<T>, crate::device::DeviceError> {
    if self.device_ready() {
      self.backend.try_sample(self)
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, crate::device::DeviceError> {
    if self.device_ready() {
      self.backend.try_euler_paths(self, m)
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }

  /// What a device runs here: the series fits the kernels'
  /// per-path history. A longer series samples on the host.
  fn device_fallback(&self) -> Option<&'static str> {
    (!(self.n <= crate::euler::HISTORY_SLOTS))
      .then_some("a series longer than the kernels' per-path history")
  }
}

/// Reusable [`Sarima`] sampling state: it owns the innovation source and the coefficients, so a
/// Monte-Carlo loop pays the `SimdNormal` setup once.
#[doc(hidden)]
#[derive(Debug)]
pub struct SarimaSampler<T: FloatExt> {
  n: usize,
  non_seasonal_ar_coefs: Array1<T>,
  non_seasonal_ma_coefs: Array1<T>,
  seasonal_ar_coefs: Array1<T>,
  seasonal_ma_coefs: Array1<T>,
  d: usize,
  big_d: usize,
  s: usize,
  /// `None` at `sigma = 0`, where every innovation is zero.
  normal: Option<Seeded<SimdNormal<T>>>,
}

impl<T: FloatExt> SarimaSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    let n = out.len();
    let mut noise = Array1::<T>::zeros(n);
    if let Some(normal) = self.normal.as_mut().filter(|_| n > 0) {
      normal.fill_slice(noise.as_slice_mut().expect("contiguous"));
    }
    let integrated = sarima_filter(
      &noise,
      &self.non_seasonal_ar_coefs,
      &self.non_seasonal_ma_coefs,
      &self.seasonal_ar_coefs,
      &self.seasonal_ma_coefs,
      self.d,
      self.big_d,
      self.s,
    );
    out.copy_from_slice(integrated.as_slice().expect("contiguous"));
  }
}

/// The SARIMA recursion as a linear filter: one SARMA pass on the combined polynomials, then the
/// inverse differences. Run on a unit impulse it gives the response a device convolves with.
pub(crate) fn sarima_filter<T: FloatExt>(
  noise: &Array1<T>,
  non_seasonal_ar_coefs: &Array1<T>,
  non_seasonal_ma_coefs: &Array1<T>,
  seasonal_ar_coefs: &Array1<T>,
  seasonal_ma_coefs: &Array1<T>,
  d: usize,
  big_d: usize,
  s: usize,
) -> Array1<T> {
  let n = noise.len();

  // φ(B)Φ(Bˢ) with φ(B) = 1 − Σ φᵢ Bⁱ and Φ(Bˢ) = 1 − Σ Φⱼ Bʲˢ: a term at every lag i + j·s.
  let ar_lags = multiply_ar_polynomials(non_seasonal_ar_coefs, seasonal_ar_coefs, s);

  // θ(B)Θ(Bˢ) with θ(B) = 1 + Σ θᵢ Bⁱ and Θ(Bˢ) = 1 + Σ Θⱼ Bʲˢ.
  let ma_lags = multiply_ma_polynomials(non_seasonal_ma_coefs, seasonal_ma_coefs, s);

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
  for _ in 0..big_d {
    integrated = inverse_seasonal_difference(&integrated, s);
  }
  for _ in 0..d {
    integrated = inverse_difference(&integrated);
  }
  integrated
}

impl<T: FloatExt> crate::traits::Sealed for SarimaSampler<T> {}

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

/// (lag, coefficient) pairs of φ(B)Φ(Bˢ) without the constant 1, signed as the recursion reads
/// them: `W_t = Σ coef · W_{t−lag} + …`.
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

/// (lag, coefficient) pairs of θ(B)Θ(Bˢ) without the constant 1.
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
  params: (non_seasonal_ar_coefs: Vec<f64>, non_seasonal_ma_coefs: Vec<f64>, seasonal_ar_coefs: Vec<f64>, seasonal_ma_coefs: Vec<f64>, d: usize, d_seasonal: usize, s: usize, sigma: f64, n: usize),
  device
);

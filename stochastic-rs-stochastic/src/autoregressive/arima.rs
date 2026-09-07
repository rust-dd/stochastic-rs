//! # Arima
//!
//! $$
//! \phi(B)(1-B)^dX_t=\theta(B)\varepsilon_t,\qquad \varepsilon_t\sim\mathcal N(0,\sigma^2)
//! $$
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Implements an Arima(p, d, q) process using explicit backshift notation:
///
/// \[
///   \phi(B)\,(1 - B)^d X_t = \theta(B)\,\epsilon_t,
/// \]
/// where \(\phi(B)\) and \(\theta(B)\) are polynomials of orders p and q, respectively,
/// and \(B\) is the backshift (lag) operator (\(B X_t = X_{t-1}\)).
#[derive(Debug, Clone)]
pub struct Arima<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// AR coefficients (\(\phi_1,\dots,\phi_p\)) as an Array1
  pub ar_coefs: Array1<T>,
  /// MA coefficients (\(\theta_1,\dots,\theta_q\)) as an Array1
  pub ma_coefs: Array1<T>,
  /// Differencing order (d)
  pub d: usize,
  /// Noise std dev (\(\sigma\)) for the innovations
  pub sigma: T,
  /// Final length of time series
  pub n: usize,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Arima<T, S> {
  /// Create a new Arima model with the given parameters.
  pub fn new(
    ar_coefs: Array1<T>,
    ma_coefs: Array1<T>,
    d: usize,
    sigma: T,
    n: usize,
    seed: S,
  ) -> Self {
    assert!(sigma > T::zero(), "Arima requires sigma > 0");
    Self {
      backend: Cpu,
      ar_coefs,
      ma_coefs,
      d,
      sigma,
      n,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Arima<T, S, B> {}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for Arima<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::MovingAverageFilter { sigma: self.sigma }
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

  /// The impulse response of this ARIMA's own recursion, so the device's
  /// convolution of the innovations is that recursion exactly, at any order.
  fn curves(&self) -> Option<Vec<Vec<T>>> {
    Some(vec![impulse_response(self.n, |unit| {
      arima_filter(unit, &self.ar_coefs, &self.ma_coefs, self.d)
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

backend_switch!([T: FloatExt, S: SeedExt] Arima<T, S> { ar_coefs, ma_coefs, d, sigma, n, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Arima<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = ArimaSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> ArimaSampler<T> {
    ArimaSampler {
      n: self.n,
      ar_coefs: self.ar_coefs.clone(),
      ma_coefs: self.ma_coefs.clone(),
      d: self.d,
      normal: SimdNormal::<T>::new(T::zero(), self.sigma, &self.seed),
    }
  }

  /// Through the Euler engine when the series fits the kernels'
  /// per-path history; a longer series keeps the process on the host,
  /// chunked exactly as [`ProcessExt`] chunks.
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

/// Reusable [`Arima`] sampling state: owns the Gaussian innovation source and
/// the ARMA coefficients so a Monte-Carlo loop pays the `SimdNormal` setup once.
#[doc(hidden)]
pub struct ArimaSampler<T: FloatExt> {
  n: usize,
  ar_coefs: Array1<T>,
  ma_coefs: Array1<T>,
  d: usize,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> ArimaSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    let n = out.len();
    let mut noise = Array1::<T>::zeros(n);
    if n > 0 {
      let slice = noise.as_slice_mut().expect("contiguous");
      self.normal.fill_slice(slice);
    }
    let result = arima_filter(&noise, &self.ar_coefs, &self.ma_coefs, self.d);
    out.copy_from_slice(result.as_slice().expect("contiguous"));
  }
}

/// The ARIMA(p, d, q) recursion as a linear filter of its innovations: the
/// single-pass ARMA(p, q) recursion with shared noise, then `d` inverse
/// differences. Linear in `noise`, so run on a unit impulse it yields the
/// impulse response a device convolves the innovations with.
pub(crate) fn arima_filter<T: FloatExt>(
  noise: &Array1<T>,
  ar_coefs: &Array1<T>,
  ma_coefs: &Array1<T>,
  d: usize,
) -> Array1<T> {
  let n = noise.len();
  let p = ar_coefs.len();
  let q = ma_coefs.len();
  let mut arma_series = Array1::<T>::zeros(n);

  // Single-pass ARMA(p,q) recursion with shared noise:
  // X_t = sum_k(phi_k * X_{t-k}) + eps_t + sum_k(theta_k * eps_{t-k})
  for t in 0..n {
    let mut val = noise[t];

    for k in 1..=p {
      if t >= k {
        val += ar_coefs[k - 1] * arma_series[t - k];
      }
    }

    for k in 1..=q {
      if t >= k {
        val += ma_coefs[k - 1] * noise[t - k];
      }
    }

    arma_series[t] = val;
  }

  // Inverse difference d times -> Arima(p,d,q)
  let mut result = arma_series;
  for _ in 0..d {
    result = inverse_difference(&result);
  }
  result
}

/// The impulse response of a linear filter over `n` points: the filter run on
/// a unit innovation at the origin and silence after it.
pub(crate) fn impulse_response<T: FloatExt>(
  n: usize,
  filter: impl Fn(&Array1<T>) -> Array1<T>,
) -> Vec<T> {
  let mut unit = Array1::<T>::zeros(n);
  if n > 0 {
    unit[0] = T::one();
  }
  filter(&unit).to_vec()
}

impl<T: FloatExt> PathSampler<T> for ArimaSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("Arima output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

/// Inverse differencing once, converting Y into X:
/// `X[0] = Y[0]`, `X[t] = X[t-1] + Y[t]`, for `t = 1..(n-1)`.
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

py_process_1d!(PyArima, Arima,
  sig: (ar_coefs, ma_coefs, d, sigma, n, seed=None, dtype=None),
  params: (ar_coefs: Vec<f64>, ma_coefs: Vec<f64>, d: usize, sigma: f64, n: usize),
  device
);

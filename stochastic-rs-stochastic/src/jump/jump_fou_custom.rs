//! # Jump fOU Custom
//!
//! $$
//! dX_t=\kappa(\theta-X_t)dt+\sigma dB_t^H+dJ_t
//! $$
//!
mod params;

use std::any::Any;

use ndarray::Array1;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRng;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::device::FgnBackend;
use crate::noise::fgn::Fgn;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// A fractional OU plus jumps whose waiting times follow `DT` and whose sizes follow `DS`; fields
/// are private, since `fgn` caches a spectrum of `hurst`, `n` and `t` that the setters rebuild.
///
/// ```compile_fail,E0616
/// use stochastic_rs_core::simd_rng::Unseeded;
/// use stochastic_rs_distributions::normal::SimdNormal;
/// use stochastic_rs_stochastic::jump::jump_fou_custom::JumpFouCustom;
/// let times = SimdNormal::<f64>::new(0.5, 0.01);
/// let sizes = SimdNormal::<f64>::new(0.0, 0.1);
/// let mut p = JumpFouCustom::new(0.7, 1.0, 0.0, 0.2, 10, None, None, times, sizes, Unseeded);
/// p.n = 1000;
/// ```
pub struct JumpFouCustom<T, DT, DS, S: SeedExt = Unseeded, B = Cpu>
where
  T: FloatExt,
  DT: Distribution<T> + Send + Sync,
  DS: Distribution<T> + Send + Sync,
{
  hurst: T,
  theta: T,
  mu: T,
  sigma: T,
  n: usize,
  x0: Option<T>,
  t: Option<T>,
  jump_times: DT,
  jump_sizes: DS,
  fgn: Fgn<T, Unseeded, B>,
  seed: S,
}

impl<T, DT, DS, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>> crate::traits::Sealed
  for JumpFouCustom<T, DT, DS, S, B>
where
  T: FloatExt,
  DT: Distribution<T> + Send + Sync + Any,
  DS: Distribution<T> + Send + Sync + Any,
{
}

impl<T, DT, DS, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>> ProcessExt<T>
  for JumpFouCustom<T, DT, DS, S, B>
where
  T: FloatExt,
  DT: Distribution<T> + Send + Sync + Any,
  DS: Distribution<T> + Send + Sync + Any,
{
  type Output = Array1<T>;
  type Sampler<'s>
    = JumpFouCustomSampler<'s, T, DT, DS, B>
  where
    Self: 's;

  fn sampler(&self) -> JumpFouCustomSampler<'_, T, DT, DS, B> {
    // Owns a Gaussian source derived from `self.seed` (not `fgn.sampler()`,
    // which would build it from `fgn`'s own permanently-`Unseeded` field —
    // see the type doc) and an owned jump RNG also derived from `self.seed`,
    // and borrows `fgn` (for its `Arc`-shared FFT plan/eigenvalues only) and
    // the user-supplied inter-arrival / jump-size distributions. The two
    // owned sources are independent derives off the same counter, so they
    // stay mutually uncorrelated; both advance on reuse.
    JumpFouCustomSampler {
      n: self.n,
      theta: self.theta,
      mu: self.mu,
      sigma: self.sigma,
      x0: self.x0.unwrap_or(T::zero()),
      dt: self.fgn.dt(),
      fgn: &self.fgn,
      normal: SimdNormal::<T>::new(T::zero(), T::one()).seeded(&self.seed.derive()),
      jump_times: &self.jump_times,
      jump_sizes: &self.jump_sizes,
      rng: self.seed.rng(),
    }
  }

  /// Through the Euler engine when the inter-arrivals are exponential and
  /// the jump-size law is one the device kernels draw; anything else keeps the process on the host,
  /// chunked exactly as [`ProcessExt`] chunks.
  fn sample(&self) -> Array1<T> {
    if self.device_ready() {
      crate::euler::EulerBackend::euler_sample(&self.fgn.backend, self)
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    if self.device_ready() {
      crate::euler::EulerBackend::euler_paths_map(&self.fgn.backend, self, m, f)
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
      crate::euler::EulerBackend::euler_paths_map_view(&self.fgn.backend, self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, |path| f(path.view()))
    }
  }

  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    if self.device_ready() {
      crate::euler::EulerBackend::euler_paths(&self.fgn.backend, self, m)
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<Array1<T>, crate::device::DeviceError> {
    if self.device_ready() {
      crate::euler::EulerBackend::try_sample(&self.fgn.backend, self)
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, crate::device::DeviceError> {
    if self.device_ready() {
      crate::euler::EulerBackend::try_euler_paths(&self.fgn.backend, self, m)
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }

  /// What a device runs here: exponential inter-arrivals and a
  /// size law the kernels draw. Any other pair samples on the host.
  fn device_fallback(&self) -> Option<&'static str> {
    (!(self.device_intensity().is_some() && self.device_jump_sizes().is_some()))
      .then_some("an inter-arrival or size law the kernels do not draw")
  }
}

/// Reusable [`JumpFouCustom`] sampling state: borrows `fgn` for its
/// `Arc`-shared FFT plan/eigenvalues and the inter-arrival / jump-size
/// distributions, and owns the Gaussian source and jump RNG, so a
/// Monte-Carlo loop pays the fGn `SimdNormal` setup once.
#[doc(hidden)]
pub struct JumpFouCustomSampler<'a, T, DT, DS, B>
where
  T: FloatExt,
  DT: Distribution<T> + Send + Sync,
  DS: Distribution<T> + Send + Sync,
  B: FgnBackend<T>,
{
  n: usize,
  theta: T,
  mu: T,
  sigma: T,
  x0: T,
  dt: T,
  fgn: &'a Fgn<T, Unseeded, B>,
  normal: Seeded<SimdNormal<T>>,
  jump_times: &'a DT,
  jump_sizes: &'a DS,
  rng: SimdRng,
}

impl<T, DT, DS, B> JumpFouCustomSampler<'_, T, DT, DS, B>
where
  T: FloatExt,
  DT: Distribution<T> + Send + Sync,
  DS: Distribution<T> + Send + Sync,
  B: FgnBackend<T>,
{
  fn fill_path(&mut self, out: &mut [T]) {
    // Match the legacy `n <= 1` path: return an all-zeros buffer (x0 is only
    // applied once there are at least two points). `array1_from_fill` does not
    // zero, so write it explicitly.
    if out.len() <= 1 {
      if let Some(first) = out.first_mut() {
        *first = T::zero();
      }
      return;
    }

    let mut fgn = Array1::<T>::zeros(self.fgn.n());
    self
      .fgn
      .fill_cpu(&mut self.normal, fgn.as_slice_mut().unwrap());

    out[0] = self.x0;
    let mut next_jump_time = self.jump_times.sample(&mut self.rng);
    assert!(
      next_jump_time > T::zero(),
      "JumpFouCustom: jump_times closure must return strictly positive inter-arrival times \
       (this is a runtime contract on the user-supplied distribution; if the distribution \
       can return ≤0 values, wrap it in `.max(eps)` or use a strictly-positive distribution)"
    );

    for i in 1..out.len() {
      let current_time = T::from_usize_(i) * self.dt;
      let mut jump_sum = T::zero();
      while next_jump_time <= current_time {
        jump_sum += self.jump_sizes.sample(&mut self.rng);
        let delta = self.jump_times.sample(&mut self.rng);
        assert!(
          delta > T::zero(),
          "JumpFouCustom: jump_times closure must return strictly positive inter-arrival times"
        );
        next_jump_time += delta;
      }

      out[i] = out[i - 1]
        + self.theta * (self.mu - out[i - 1]) * self.dt
        + self.sigma * fgn[i - 1]
        + jump_sum;
    }
  }
}

impl<T, DT, DS, B> crate::traits::Sealed for JumpFouCustomSampler<'_, T, DT, DS, B>
where
  T: FloatExt,
  DT: Distribution<T> + Send + Sync,
  DS: Distribution<T> + Send + Sync,
  B: FgnBackend<T>,
{
}

impl<T, DT, DS, B> PathSampler<T> for JumpFouCustomSampler<'_, T, DT, DS, B>
where
  T: FloatExt,
  DT: Distribution<T> + Send + Sync,
  DS: Distribution<T> + Send + Sync,
  B: FgnBackend<T>,
{
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    self.fill_path(
      out
        .as_slice_mut()
        .expect("JumpFouCustom output must be contiguous"),
    );
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

impl<T, DT, DS, S: SeedExt, B> JumpFouCustom<T, DT, DS, S, B>
where
  T: FloatExt,
  DT: Distribution<T> + Send + Sync + Any,
  DS: Distribution<T> + Send + Sync + Any,
{
  /// The arrival intensity when the inter-arrival law is exponential, which
  /// is when the arrivals are the Poisson stream the kernels count.
  fn device_intensity(&self) -> Option<T> {
    crate::process::cpoisson::device_arrival_rate(&self.jump_times)
  }

  /// The jump-size law as the Euler engine's kernels draw it, `None` for a
  /// distribution they do not carry.
  fn device_jump_sizes(&self) -> Option<crate::euler::JumpSizes<T>> {
    crate::process::cpoisson::device_jump_sizes(&self.jump_sizes)
  }
}

impl<T, DT, DS, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>>
  crate::euler::EulerCoefficients<T> for JumpFouCustom<T, DT, DS, S, B>
where
  T: FloatExt,
  DT: Distribution<T> + Send + Sync + Any,
  DS: Distribution<T> + Send + Sync + Any,
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::JumpFractionalOu {
      theta: self.theta,
      mu: self.mu,
      sigma: self.sigma,
    }
  }

  fn initial_value(&self) -> T {
    self.x0.unwrap_or(T::zero())
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  fn time_step(&self) -> T {
    self.fgn.dt()
  }

  fn fgn_spec(&self) -> Option<crate::euler::FgnSpec<'_, T>> {
    Some(self.fgn.fgn_spec(1))
  }

  /// The Poisson intensity the exponential inter-arrival law amounts to. A
  /// law that is not exponential never reaches a launch — [`ProcessExt::sample`]
  /// keeps such a process on the host — so asking here is a caller bypassing
  /// that guard.
  fn jump_intensity(&self) -> Option<T> {
    Some(self.device_intensity().expect(
      "JumpFouCustom: the inter-arrival distribution is not exponential, which is the only \
       arrival law the Euler engine draws on a device; sample through `ProcessExt`, which keeps \
       it on the host",
    ))
  }

  fn jump_sizes(&self) -> Option<crate::euler::JumpSizes<T>> {
    Some(self.device_jump_sizes().expect(
      "JumpFouCustom: the jump-size distribution is not a law the Euler engine draws on a \
       device; sample through `ProcessExt`, which keeps it on the host",
    ))
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

backend_switch!([T, DT, DS, S: SeedExt] JumpFouCustom<T, DT, DS, S> { hurst, theta, mu, sigma, n, x0, t, jump_times, jump_sizes, seed } via fgn euler
  where T: FloatExt, DT: Distribution<T> + Send + Sync, DS: Distribution<T> + Send + Sync);

#[cfg(test)]
mod tests {
  use rand::distr::Distribution;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::exp::SimdExp;

  use super::*;
  use crate::traits::ProcessExt;

  #[derive(Clone, Copy)]
  struct ConstDist<T>(T);

  impl<T: Copy> Distribution<T> for ConstDist<T> {
    fn sample<R: rand::Rng + ?Sized>(&self, _rng: &mut R) -> T {
      self.0
    }
  }

  #[test]
  fn exponential_times_and_normal_sizes_are_two_laws() {
    let p = JumpFouCustom::<f64, SimdExp<f64>, SimdNormal<f64>, Deterministic>::new(
      0.7,
      1.0,
      0.0,
      0.2,
      65,
      None,
      Some(1.0),
      SimdExp::new(3.0),
      SimdNormal::new(0.0, 0.1),
      Deterministic::new(1),
    );
    assert!(
      p.device_ready(),
      "an exponential arrival law and a normal size law are both kernel laws"
    );
    assert_eq!(
      (p.jump_time_law().lambda(), p.jump_size_law().sigma()),
      (3.0, 0.1)
    );
    assert!(p.sample().iter().all(|x| x.is_finite()));
  }

  #[test]
  fn allows_multiple_jumps_in_single_dt() {
    let p = JumpFouCustom::new(
      0.7_f64,
      0.0,
      0.0,
      0.0,
      3,
      Some(0.0),
      Some(1.0),
      ConstDist(0.2), // inter-arrival
      ConstDist(0.2), // jump size
      Unseeded,
    );

    let x = p.sample();
    assert_eq!(x.len(), 3);
    assert!((x[1] - 0.4).abs() < 1e-12);
    assert!((x[2] - 1.0).abs() < 1e-12);
  }
}

#[cfg(feature = "python")]
#[doc(hidden)]
#[pyo3::prelude::pyclass]
pub struct PyJumpFOUCustom {
  inner_f32:
    Option<JumpFouCustom<f32, crate::traits::CallableDist<f32>, crate::traits::CallableDist<f32>>>,
  inner_f64:
    Option<JumpFouCustom<f64, crate::traits::CallableDist<f64>, crate::traits::CallableDist<f64>>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyJumpFOUCustom {
  #[new]
  #[pyo3(signature = (hurst, theta, mu, sigma, jump_times, jump_sizes, n, x0=None, t=None, dtype=None))]
  fn new(
    hurst: f64,
    theta: f64,
    mu: f64,
    sigma: f64,
    jump_times: pyo3::Py<pyo3::PyAny>,
    jump_sizes: pyo3::Py<pyo3::PyAny>,
    n: usize,
    x0: Option<f64>,
    t: Option<f64>,
    dtype: Option<&str>,
  ) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| match dtype.unwrap_or("f64") {
      "f32" => Self {
        inner_f32: Some(JumpFouCustom::new(
          hurst as f32,
          theta as f32,
          mu as f32,
          sigma as f32,
          n,
          x0.map(|v| v as f32),
          t.map(|v| v as f32),
          crate::traits::CallableDist::new(jump_times),
          crate::traits::CallableDist::new(jump_sizes),
          Unseeded,
        )),
        inner_f64: None,
      },
      _ => Self {
        inner_f32: None,
        inner_f64: Some(JumpFouCustom::new(
          hurst,
          theta,
          mu,
          sigma,
          n,
          x0,
          t,
          crate::traits::CallableDist::new(jump_times),
          crate::traits::CallableDist::new(jump_sizes),
          Unseeded,
        )),
      },
    })
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use numpy::IntoPyArray;
      use pyo3::IntoPyObjectExt;

      use crate::traits::ProcessExt;
      if let Some(ref inner) = self.inner_f64 {
        inner.sample().into_pyarray(py).into_py_any(py).unwrap()
      } else if let Some(ref inner) = self.inner_f32 {
        inner.sample().into_pyarray(py).into_py_any(py).unwrap()
      } else {
        unreachable!()
      }
    })
  }

  /// `m` paths as an `(m, n)` array. The GIL is released while they are generated; the Python
  /// laws run on rayon workers, one call at a time.
  fn sample_par<'py>(
    &self,
    py: pyo3::Python<'py>,
    m: usize,
  ) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use numpy::IntoPyArray;
      use numpy::ndarray::Array2;
      use pyo3::IntoPyObjectExt;

      use crate::traits::ProcessExt;
      if let Some(ref inner) = self.inner_f64 {
        let paths = py.detach(|| inner.sample_par(m));
        let n = paths.first().map_or(0, |p| p.len());
        let mut result = Array2::<f64>::zeros((m, n));
        for (i, path) in paths.iter().enumerate() {
          result.row_mut(i).assign(path);
        }
        result.into_pyarray(py).into_py_any(py).unwrap()
      } else if let Some(ref inner) = self.inner_f32 {
        let paths = py.detach(|| inner.sample_par(m));
        let n = paths.first().map_or(0, |p| p.len());
        let mut result = Array2::<f32>::zeros((m, n));
        for (i, path) in paths.iter().enumerate() {
          result.row_mut(i).assign(path);
        }
        result.into_pyarray(py).into_py_any(py).unwrap()
      } else {
        unreachable!()
      }
    })
  }
}

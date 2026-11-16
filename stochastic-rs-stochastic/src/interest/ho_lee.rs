//! # Ho Lee
//!
//! $$
//! dr_t=\theta(t)dt+\sigma dW_t
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
use crate::params::HoLeeDrift;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Clone)]
pub struct HoLee<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// The module header's θ(t): a constant, or read off the forward curve.
  pub drift: HoLeeDrift<T>,
  /// Diffusion scale σ multiplying `dW_t`.
  pub sigma: T,
  /// Number of points sampled along the Ho-Lee path.
  pub n: usize,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> HoLee<T, S> {
  pub fn new(drift: HoLeeDrift<T>, sigma: T, n: usize, t: Option<T>, seed: S) -> Self {
    if let HoLeeDrift::Constant(theta) = drift {
      assert!(
        theta.is_finite(),
        "theta must satisfy `theta.is_finite()`, got theta = {theta:?}"
      );
    }
    assert!(
      sigma.is_finite(),
      "sigma must satisfy `sigma.is_finite()`, got sigma = {sigma:?}"
    );
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );

    Self {
      backend: Cpu,
      drift,
      sigma,
      n,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> HoLee<T, S, B> {}

/// The Euler engine's view of Ho-Lee. Both drift forms — a constant `θ` or
/// the forward curve's slope plus `σ²t` — are one value per grid point, so
/// both reach the device the same way.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for HoLee<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::CurveDrift { sigma: self.sigma }
  }

  fn initial_value(&self) -> T {
    T::zero()
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  /// The drift at each grid point, from whichever form the process carries.
  fn curve(&self) -> Option<Vec<T>> {
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(self.n.saturating_sub(1).max(1));
    let sigma_sq = self.sigma * self.sigma;
    Some(
      (0..self.n)
        .map(|i| {
          let t = T::from_usize_(i) * dt;
          match &self.drift {
            HoLeeDrift::ForwardCurve(f) => {
              let eps = dt.max(T::from_f64_fast(1e-8));
              let t_minus = (t - eps).max(T::zero());
              let t_plus = t + eps;
              (f.call(t_plus) - f.call(t_minus)) / (t_plus - t_minus) + sigma_sq * t
            }
            HoLeeDrift::Constant(theta) => *theta,
          }
        })
        .collect(),
    )
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

backend_switch!([T: FloatExt, S: SeedExt] HoLee<T, S> { drift, sigma, n, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for HoLee<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for HoLee<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = HoLeeSampler<'s, T>
  where
    Self: 's;

  fn sampler(&self) -> HoLeeSampler<'_, T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    HoLeeSampler {
      n: self.n,
      dt,
      sigma: self.sigma,
      diff_scale: self.sigma,
      drift: &self.drift,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt()).seeded(&self.seed),
    }
  }

  /// Through the Euler engine: on a device the recursion runs in the kernel
  /// with its time-varying coefficient bound per step, on the host devices it
  /// is this process's own sampler, chunked exactly as `ProcessExt` chunks.
  fn sample(&self) -> Array1<T> {
    self.backend.euler_sample(self)
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    self.backend.euler_paths_map(self, m, f)
  }

  fn sample_map_view<R: Send>(
    &self,
    m: usize,
    f: impl Fn(ndarray::ArrayView1<T>) -> R + Sync,
  ) -> Vec<R> {
    self.backend.euler_paths_map_view(self, m, f)
  }

  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    self.backend.euler_paths(self, m)
  }

  fn try_sample(&self) -> Result<Array1<T>, crate::device::DeviceError> {
    self.backend.try_sample(self)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, crate::device::DeviceError> {
    self.backend.try_euler_paths(self, m)
  }
}

/// Reusable [`HoLee`] sampling state. Borrows the process for its optional
/// forward-curve function and owns the Gaussian source so a Monte-Carlo loop
/// pays the `SimdNormal` setup once.
#[doc(hidden)]
pub struct HoLeeSampler<'a, T: FloatExt> {
  n: usize,
  dt: T,
  sigma: T,
  diff_scale: T,
  drift: &'a HoLeeDrift<T>,
  normal: Seeded<SimdNormal<T>>,
}

impl<T: FloatExt> HoLeeSampler<'_, T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.len() <= 1 {
      if let Some(first) = out.first_mut() {
        *first = T::zero();
      }
      return;
    }

    let dt = self.dt;
    let diff_scale = self.diff_scale;
    out[0] = T::zero();
    let mut prev = out[0];
    let tail = &mut out[1..];
    self.normal.fill_slice(tail);

    for (k, z) in tail.iter_mut().enumerate() {
      let i = k + 1;
      let t = T::from_usize_(i) * dt;
      let drift = match self.drift {
        HoLeeDrift::ForwardCurve(f) => {
          let eps = dt.max(T::from_f64_fast(1e-8));
          let t_minus = (t - eps).max(T::zero());
          let t_plus = t + eps;
          let df_dt = (f.call(t_plus) - f.call(t_minus)) / (t_plus - t_minus);
          df_dt + self.sigma.powf(T::from_usize_(2)) * t
        }
        HoLeeDrift::Constant(theta) => *theta,
      };

      let next = prev + drift * dt + diff_scale * *z;
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> crate::traits::Sealed for HoLeeSampler<'_, T> {}

impl<T: FloatExt> PathSampler<T> for HoLeeSampler<'_, T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("HoLee output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::traits::Fn1D;
  use crate::traits::ProcessExt;

  fn f_curve(t: f64) -> f64 {
    t * t
  }

  #[test]
  fn uses_forward_curve_derivative_when_provided() {
    let p = HoLee::new(
      HoLeeDrift::ForwardCurve(Fn1D::Native(f_curve as fn(f64) -> f64)),
      0.0_f64,
      3,
      Some(1.0),
      Unseeded,
    );
    let r = p.sample();
    assert!((r[1] - 0.5).abs() < 1e-12);
    assert!((r[2] - 1.5).abs() < 1e-12);
  }
}

#[cfg(feature = "python")]
#[doc(hidden)]
#[pyo3::prelude::pyclass]
pub struct PyHoLee {
  inner: Option<HoLee<f64>>,
  seeded: Option<HoLee<f64, crate::python_device::SharedSeed>>,
  /// The device the class samples on, chosen at construction.
  device: crate::python_device::Device,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyHoLee {
  #[new]
  #[pyo3(signature = (sigma, n, forward_curve=None, theta=None, t=None, seed=None, device=None))]
  fn new(
    sigma: f64,
    n: usize,
    forward_curve: Option<pyo3::Py<pyo3::PyAny>>,
    theta: Option<f64>,
    t: Option<f64>,
    seed: Option<u64>,
    device: Option<&str>,
  ) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> pyo3::PyResult<Self> {
      let device = crate::python_device::Device::parse(device, "f64")?;
      let drift = crate::python_params::ho_lee_drift_from_kwargs(theta, forward_curve)?;
      Ok(match seed {
        Some(s) => Self {
          device,
          inner: None,
          seeded: Some(HoLee::new(
            drift,
            sigma,
            n,
            t,
            crate::python_device::SharedSeed::new(s),
          )),
        },
        None => Self {
          device,
          inner: Some(HoLee::new(drift, sigma, n, t, Unseeded)),
          seeded: None,
        },
      })
    })?
  }

  /// The reason a device kernel cannot carry this configuration, if there is
  /// one: the engine states what it cannot do rather than doing it quietly on
  /// the host. The question is about the configuration, not the handle, so
  /// the answer is the same whatever `device=` was passed.
  fn device_fallback(&self) -> Option<&'static str> {
    use crate::traits::ProcessExt;
    crate::py_dispatch_f64!(self, |inner| inner.device_fallback())
  }

  /// Whether this configuration runs on a device kernel: the absence of a
  /// [`device_fallback`](Self::device_fallback) reason.
  fn device_ready(&self) -> bool {
    self.device_fallback().is_none()
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use numpy::IntoPyArray;
      use pyo3::IntoPyObjectExt;

      use crate::traits::ProcessExt;
      py_device_dispatch_f64!(self, |inner| inner
        .sample()
        .into_pyarray(py)
        .into_py_any(py)
        .unwrap())
    })
  }

  /// `m` independent paths stacked into an `(m, n)` array. The GIL is
  /// released while the paths are generated; every callable coefficient
  /// re-acquires it, so a Python function is called from rayon workers one
  /// at a time.
  fn sample_par<'py>(
    &self,
    py: pyo3::Python<'py>,
    m: usize,
  ) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use ndarray::Array2;
      use numpy::IntoPyArray;
      use pyo3::IntoPyObjectExt;

      use crate::traits::ProcessExt;
      py_device_dispatch_f64!(self, |inner| {
        let paths = py.detach(|| inner.sample_par(m));
        let n = paths.first().map_or(0, |p| p.len());
        let mut result = Array2::zeros((m, n));
        for (i, path) in paths.iter().enumerate() {
          result.row_mut(i).assign(path);
        }
        result.into_pyarray(py).into_py_any(py).unwrap()
      })
    })
  }
}

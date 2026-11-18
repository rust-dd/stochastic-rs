//! # Hull White
//!
//! $$
//! dr_t=\left(\theta(t)-a r_t\right)dt+\sigma dW_t
//! $$
//!
//! Reference: Hull & White (1990), Rev. Financ. Stud. 3(4), 573–592, DOI 10.1093/rfs/3.4.573
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
use crate::precondition::check_n;
use crate::precondition::check_t;
use crate::traits::FloatExt;
use crate::traits::Fn1D;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Debug, Clone)]
pub struct HullWhite<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Time-dependent drift target function θ(t), fitted to the initial
  /// term structure — added directly into the drift alongside `-alpha*r`.
  pub theta: Fn1D<T>,
  /// Mean-reversion speed α (multiplies `-r_t` in the drift) — not a
  /// shape/loading parameter.
  pub alpha: T,
  /// Diffusion scale σ multiplying `dW_t`.
  pub sigma: T,
  /// Number of points sampled along the Hull-White path.
  pub n: usize,
  /// Initial value; `None` starts at zero.
  pub x0: Option<T>,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// The seed: [`Unseeded`] or [`Deterministic`](stochastic_rs_core::simd_rng::Deterministic).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

/// The constant θ(t) ≡ 0.04 of [`HullWhite`]'s `Default`: the time-homogeneous case, and this
/// file's `const_theta` test fixture.
fn default_theta<T: FloatExt>(_t: T) -> T {
  T::from_f64_fast(0.04)
}

/// Every field has a `with_*` setter (`HullWhite::default().with_alpha(0.6)`); nothing is
/// cached, since `sampler()` builds its Gaussian source from `self` on every call.
impl<T: FloatExt, S: SeedExt> HullWhite<T, S> {
  /// Panics unless `n >= 2`, a given `t` is positive and finite, and every parameter is finite.
  pub fn new(
    theta: impl Into<Fn1D<T>>,
    alpha: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    Self::check_alpha(alpha);
    Self::check_sigma(sigma);
    check_n(n);
    Self::check_x0(x0);
    check_t(t);
    Self {
      backend: Cpu,
      theta: theta.into(),
      alpha,
      sigma,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> HullWhite<T, S, B> {
  fn check_alpha(alpha: T) {
    require!(alpha.is_finite(), alpha);
  }

  fn check_sigma(sigma: T) {
    require!(sigma.is_finite(), sigma);
  }

  fn check_x0(x0: Option<T>) {
    require!(x0.is_none_or(|v| v.is_finite()), x0);
  }

  /// Replace `theta`, all else unchanged.
  pub fn with_theta(mut self, theta: impl Into<Fn1D<T>>) -> Self {
    self.theta = theta.into();
    self
  }

  /// Replace `alpha`; panics unless `alpha` is finite.
  pub fn with_alpha(mut self, alpha: T) -> Self {
    Self::check_alpha(alpha);
    self.alpha = alpha;
    self
  }

  /// Replace `sigma`; panics unless `sigma` is finite.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    Self::check_sigma(sigma);
    self.sigma = sigma;
    self
  }

  /// Replace `x0`; panics unless `x0` is finite.
  pub fn with_x0(mut self, x0: Option<T>) -> Self {
    Self::check_x0(x0);
    self.x0 = x0;
    self
  }

  /// Replace `n`, the grid points along the path; panics unless `n >= 2`.
  pub fn with_points(mut self, n: usize) -> Self {
    check_n(n);
    self.n = n;
    self
  }

  /// Replace the simulation horizon `t`; panics unless `t`, when given, is positive and finite.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    check_t(t);
    self.t = t;
    self
  }

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

/// θ(t)≡0.04, α=0.4, σ=0.02, r₀=0.02, a textbook Hull-White (this file's `const_theta` fixture);
/// t=1, 252 points, a trading year of daily closes.
impl<T: FloatExt> Default for HullWhite<T, Unseeded> {
  fn default() -> Self {
    Self::new(
      default_theta::<T> as fn(T) -> T,
      T::from_f64_fast(0.4),
      T::from_f64_fast(0.02),
      252,
      Some(T::from_f64_fast(0.02)),
      Some(T::one()),
      Unseeded,
    )
  }
}

/// The Euler engine's view of Hull-White.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for HullWhite<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::HullWhite {
      alpha: self.alpha,
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

  /// The mean-reversion level at each grid point.
  fn curve(&self) -> Option<Vec<T>> {
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(self.n.saturating_sub(1).max(1));
    Some(
      (0..self.n)
        .map(|i| self.theta.call(T::from_usize_(i) * dt))
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

backend_switch!([T: FloatExt, S: SeedExt] HullWhite<T, S> { theta, alpha, sigma, n, x0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for HullWhite<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for HullWhite<T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = HullWhiteSampler<'s, T>
  where
    Self: 's;

  fn sampler(&self) -> HullWhiteSampler<'_, T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    HullWhiteSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      alpha: self.alpha,
      diff_scale: self.sigma,
      theta: &self.theta,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt()).seeded(&self.seed),
    }
  }

  /// Through the Euler engine: a device binds the time-varying coefficient per step in its
  /// kernel, the host runs this process's own sampler, chunked as `ProcessExt` chunks.
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

/// Reusable [`HullWhite`] sampling state: it borrows the process for `θ(t)` and owns the
/// Gaussian source, so a Monte-Carlo loop pays the `SimdNormal` setup once.
#[doc(hidden)]
#[derive(Debug)]
pub struct HullWhiteSampler<'a, T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  alpha: T,
  diff_scale: T,
  theta: &'a Fn1D<T>,
  normal: Seeded<SimdNormal<T>>,
}

impl<T: FloatExt> HullWhiteSampler<'_, T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }
    let dt = self.dt;
    let diff_scale = self.diff_scale;
    let mut prev = out[0];
    let tail = &mut out[1..];
    self.normal.fill_slice(tail);

    for (k, z) in tail.iter_mut().enumerate() {
      let i = k + 1;
      let next =
        prev + (self.theta.call(T::from_usize_(i) * dt) - self.alpha * prev) * dt + diff_scale * *z;
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> crate::traits::Sealed for HullWhiteSampler<'_, T> {}

impl<T: FloatExt> PathSampler<T> for HullWhiteSampler<'_, T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("HullWhite output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

#[cfg(feature = "python")]
#[doc(hidden)]
#[pyo3::prelude::pyclass]
pub struct PyHullWhite {
  inner: Option<HullWhite<f64>>,
  seeded: Option<HullWhite<f64, crate::python_device::SharedSeed>>,
  /// The device the class samples on, chosen at construction.
  device: crate::python_device::Device,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyHullWhite {
  #[new]
  #[pyo3(signature = (theta, alpha, sigma, n, x0=None, t=None, seed=None, device=None))]
  fn new(
    theta: pyo3::Py<pyo3::PyAny>,
    alpha: f64,
    sigma: f64,
    n: usize,
    x0: Option<f64>,
    t: Option<f64>,
    seed: Option<u64>,
    device: Option<&str>,
  ) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> pyo3::PyResult<Self> {
      let device = crate::python_device::Device::parse(device, "f64")?;
      Ok(match seed {
        Some(s) => Self {
          device,
          inner: None,
          seeded: Some(HullWhite::new(
            Fn1D::Py(theta),
            alpha,
            sigma,
            n,
            x0,
            t,
            crate::python_device::SharedSeed::new(s),
          )),
        },
        None => Self {
          device,
          inner: Some(HullWhite::new(
            Fn1D::Py(theta),
            alpha,
            sigma,
            n,
            x0,
            t,
            Unseeded,
          )),
          seeded: None,
        },
      })
    })?
  }

  /// Why a device kernel cannot carry this configuration, if it cannot; the answer depends on
  /// the configuration alone, whatever `device=` was passed.
  fn device_fallback(&self) -> Option<&'static str> {
    use crate::traits::ProcessExt;
    py_dispatch_f64!(self, |inner| inner.device_fallback())
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

  /// `m` independent paths stacked into an `(m, n)` array. The GIL is released meanwhile; each
  /// `theta(t)` call re-acquires it, so rayon workers call the Python callable one at a time.
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

#[cfg(test)]
mod tests {
  use super::*;

  fn const_theta(_t: f64) -> f64 {
    0.04
  }

  #[test]
  fn sample_length_matches_n() {
    let hw = HullWhite::<f64>::new(
      const_theta as fn(f64) -> f64,
      0.5,
      0.01,
      64,
      Some(0.04),
      Some(1.0),
      Unseeded,
    );
    let path = hw.sample();
    assert_eq!(path.len(), 64);
  }
}

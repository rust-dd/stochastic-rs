//! Two-factor Hull-White in the state-space form of Brigo–Mercurio §4.2: `θ(t)` and the auxiliary
//! factor `u_t` enter `r`'s drift directly, where G2++ sums two factors. The output is `[r, u]`.
//!
//! $$
//! dr_t=\bigl[\theta(t)+u_t-a\,r_t\bigr]dt+\sigma_1\,dW_t^1,\qquad
//! du_t=-b\,u_t\,dt+\sigma_2\,dW_t^2
//! $$
//!
//! Reference: Hull & White (1994), J. Derivatives 2(2), 37–48, DOI 10.3905/jod.1994.407908
//! Reference: Brigo & Mercurio (2006), 2nd ed., Springer, §4.2, DOI 10.1007/978-3-540-34604-3
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::device::Cpu;
use crate::noise::cgns::Cgns;
use crate::traits::FloatExt;
use crate::traits::Fn1D;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Debug, Clone)]
pub struct HullWhite2F<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Time-dependent drift target $\theta(t)$, fitted to the initial term structure, as
  /// `HullWhite::theta` in the single-factor model.
  pub theta: Fn1D<T>,
  /// Mean-reversion speed $a$ of the primary state variable.
  pub a: T,
  /// Diffusion/noise scale for factor 1.
  pub sigma1: T,
  /// Diffusion/noise scale for factor 2.
  pub sigma2: T,
  /// Instantaneous correlation ρ between the two driving Brownian motions
  /// `dW1`/`dW2`.
  pub rho: T,
  /// Mean-reversion speed b of the auxiliary factor u_t, multiplying `-u_{t-1}` in its drift;
  /// the diffusion terms are `sigma1` and `sigma2`.
  pub b: T,
  /// Initial value; `None` starts at zero.
  pub x0: Option<T>,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Number of points sampled along each of the `r`/`u` paths.
  pub n: usize,
  /// The seed: [`Unseeded`] or [`Deterministic`](stochastic_rs_core::simd_rng::Deterministic).
  pub seed: S,
  cgns: Cgns<T>,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> HullWhite2F<T, S> {
  /// Panics unless a given `t` is positive and finite, `n >= 2`, and every parameter is finite.
  pub fn new(
    theta: impl Into<Fn1D<T>>,
    a: T,
    sigma1: T,
    sigma2: T,
    rho: T,
    b: T,
    x0: Option<T>,
    t: Option<T>,
    n: usize,
    seed: S,
  ) -> Self {
    assert!(
      a.is_finite(),
      "a must satisfy `a.is_finite()`, got a = {a:?}"
    );
    assert!(
      sigma1.is_finite(),
      "sigma1 must satisfy `sigma1.is_finite()`, got sigma1 = {sigma1:?}"
    );
    assert!(
      sigma2.is_finite(),
      "sigma2 must satisfy `sigma2.is_finite()`, got sigma2 = {sigma2:?}"
    );
    assert!(
      rho.is_finite(),
      "rho must satisfy `rho.is_finite()`, got rho = {rho:?}"
    );
    assert!(
      b.is_finite(),
      "b must satisfy `b.is_finite()`, got b = {b:?}"
    );
    assert!(
      x0.is_none_or(|v| v.is_finite()),
      "x0 must satisfy `x0.is_none_or(|v| v.is_finite())`, got x0 = {x0:?}"
    );
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    Self {
      backend: Cpu,
      theta: theta.into(),
      a,
      sigma1,
      sigma2,
      rho,
      b,
      x0,
      t,
      n,
      seed,
      cgns: Cgns::new(rho, n - 1, t, Unseeded),
    }
  }
}

/// The Euler engine's view of the two-factor Hull-White model.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 2>
  for HullWhite2F<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::TwoFactorHullWhite {
      a: self.a,
      b: self.b,
      sigma1: self.sigma1,
      sigma2: self.sigma2,
      rho: self.rho,
    }
  }

  fn initial_state(&self) -> [T; 4] {
    [
      self.x0.unwrap_or(T::zero()),
      T::zero(),
      T::zero(),
      T::zero(),
    ]
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  /// The correlated-noise source divides the horizon by the number of points,
  /// not by the number of steps, so the device steps by that same amount.
  fn time_step(&self) -> T {
    self.cgns.dt()
  }

  /// The mean-reversion level at each grid point.
  fn curve(&self) -> Option<Vec<T>> {
    let dt = self.cgns.dt();
    Some(
      (0..self.n)
        .map(|i| self.theta.call(T::from_usize_(i) * dt))
        .collect(),
    )
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> [Array1<T>; 2] {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] HullWhite2F<T, S> { theta, a, sigma1, sigma2, rho, b, x0, t, n, seed, cgns } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for HullWhite2F<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for HullWhite2F<T, S, B>
{
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = HullWhite2FSampler<'s, T, S>
  where
    Self: 's;

  /// Derives `self.seed` into the sampler, never clones it: the mixed next tick hash-scrambles
  /// chunk `i`'s basis against chunk `i+1`'s, where a raw snapshot leaves them one stride apart.
  fn sampler(&self) -> HullWhite2FSampler<'_, T, S> {
    HullWhite2FSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      a: self.a,
      sigma1: self.sigma1,
      sigma2: self.sigma2,
      b: self.b,
      theta: &self.theta,
      dt: self.cgns.dt(),
      cgns: self.cgns,
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine: a device steps every component in its kernel, the host runs this
  /// process's own sampler, chunked as `ProcessExt` chunks.
  fn sample(&self) -> [Array1<T>; 2] {
    self.backend.system_sample(self)
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&[Array1<T>; 2]) -> R + Sync) -> Vec<R> {
    self.backend.system_paths_map(self, m, f)
  }

  fn sample_par(&self, m: usize) -> Vec<[Array1<T>; 2]> {
    self.backend.system_paths(self, m)
  }

  fn try_sample(&self) -> Result<[Array1<T>; 2], crate::device::DeviceError> {
    self.backend.try_system_sample(self)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<[Array1<T>; 2]>, crate::device::DeviceError> {
    self.backend.try_system_paths(self, m)
  }
}

/// Reusable [`HullWhite2F`] sampling state: it borrows the process for `theta(t)` and owns the
/// correlated-Gaussian generator and the seed source, so a loop reuses both output buffers.
#[doc(hidden)]
#[derive(Debug)]
pub struct HullWhite2FSampler<'a, T: FloatExt, S: SeedExt> {
  n: usize,
  x0: T,
  a: T,
  sigma1: T,
  sigma2: T,
  b: T,
  theta: &'a Fn1D<T>,
  dt: T,
  cgns: Cgns<T>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> HullWhite2FSampler<'_, T, S> {
  fn fill_paths(&mut self, x: &mut [T], u: &mut [T]) {
    if self.n == 0 {
      return;
    }
    let dt = self.dt;
    let [cgn1, cgn2] = &self.cgns.sample_impl(&self.seed);

    x[0] = self.x0;
    u[0] = T::zero();

    for i in 1..self.n {
      x[i] = x[i - 1]
        + (self.theta.call(T::from_usize_(i) * dt) + u[i - 1] - self.a * x[i - 1]) * dt
        + self.sigma1 * cgn1[i - 1];

      u[i] = u[i - 1] - self.b * u[i - 1] * dt + self.sigma2 * cgn2[i - 1];
    }
  }
}

impl<T: FloatExt, S: SeedExt> crate::traits::Sealed for HullWhite2FSampler<'_, T, S> {}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for HullWhite2FSampler<'_, T, S> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [x_arr, u_arr] = out;
    let x = x_arr
      .as_slice_mut()
      .expect("HullWhite2F output must be contiguous");
    let u = u_arr
      .as_slice_mut()
      .expect("HullWhite2F output must be contiguous");
    self.fill_paths(x, u);
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    let mut x = Array1::<T>::zeros(self.n);
    let mut u = Array1::<T>::zeros(self.n);
    self.fill_paths(
      x.as_slice_mut().expect("contiguous"),
      u.as_slice_mut().expect("contiguous"),
    );
    [x, u]
  }
}

#[cfg(feature = "python")]
#[doc(hidden)]
#[pyo3::prelude::pyclass]
pub struct PyHullWhite2F {
  inner: Option<HullWhite2F<f64>>,
  seeded: Option<HullWhite2F<f64, crate::python_device::SharedSeed>>,
  /// The device the class samples on, chosen at construction.
  device: crate::python_device::Device,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyHullWhite2F {
  // `theta` is the drift-target curve and `a` the mean-reversion speed, the names every other
  // short-rate class here gives them.
  #[new]
  #[pyo3(signature = (theta, a, sigma1, sigma2, rho, b, n, x0=None, t=None, seed=None, device=None))]
  fn new(
    theta: pyo3::Py<pyo3::PyAny>,
    a: f64,
    sigma1: f64,
    sigma2: f64,
    rho: f64,
    b: f64,
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
          seeded: Some(HullWhite2F::new(
            Fn1D::Py(theta),
            a,
            sigma1,
            sigma2,
            rho,
            b,
            x0,
            t,
            n,
            crate::python_device::SharedSeed::new(s),
          )),
        },
        None => Self {
          device,
          inner: Some(HullWhite2F::new(
            Fn1D::Py(theta),
            a,
            sigma1,
            sigma2,
            rho,
            b,
            x0,
            t,
            n,
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

  fn sample<'py>(
    &self,
    py: pyo3::Python<'py>,
  ) -> pyo3::PyResult<(pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>)> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use numpy::IntoPyArray;
      use pyo3::IntoPyObjectExt;

      use crate::traits::ProcessExt;
      py_device_dispatch_f64!(self, |inner| {
        let [a, b] = inner.sample();
        (
          a.into_pyarray(py).into_py_any(py).unwrap(),
          b.into_pyarray(py).into_py_any(py).unwrap(),
        )
      })
    })
  }

  /// `m` independent paths as a pair of `(m, n)` arrays, one a component. The GIL is released
  /// meanwhile; a callable coefficient re-acquires it, so rayon workers call Python one at a time.
  fn sample_par<'py>(
    &self,
    py: pyo3::Python<'py>,
    m: usize,
  ) -> pyo3::PyResult<(pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>)> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use ndarray::Array2;
      use numpy::IntoPyArray;
      use pyo3::IntoPyObjectExt;

      use crate::traits::ProcessExt;
      py_device_dispatch_f64!(self, |inner| {
        let paths = py.detach(|| inner.sample_par(m));
        let n = paths.first().map_or(0, |p| p[0].len());
        let mut first = Array2::zeros((m, n));
        let mut second = Array2::zeros((m, n));
        for (i, path) in paths.iter().enumerate() {
          first.row_mut(i).assign(&path[0]);
          second.row_mut(i).assign(&path[1]);
        }
        (
          first.into_pyarray(py).into_py_any(py).unwrap(),
          second.into_pyarray(py).into_py_any(py).unwrap(),
        )
      })
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn const_theta(_t: f64) -> f64 {
    0.5
  }

  #[test]
  fn sample_returns_two_paths() {
    let hw2 = HullWhite2F::<f64>::new(
      const_theta as fn(f64) -> f64,
      0.04,
      0.01,
      0.005,
      -0.3,
      0.4,
      Some(0.04),
      Some(1.0),
      64,
      Unseeded,
    );
    let [x, u] = hw2.sample();
    assert_eq!(x.len(), 64);
    assert_eq!(u.len(), 64);
  }

  /// `theta` is the additive target θ(t) and `a` the mean-reversion speed, as in the module's SDE
  /// and `HullWhite`'s `theta`/`alpha`; with the diffusion zeroed the path is deterministic Euler.
  #[test]
  fn hw2f_a_and_theta_zero_diffusion_matches_deterministic_euler() {
    let a = 0.6_f64;
    let b = 0.3_f64;
    let n = 33;
    let t = 1.0_f64;
    let x0 = 0.02_f64;

    let hw2 = HullWhite2F::<f64>::new(
      const_theta as fn(f64) -> f64,
      a,
      0.0,
      0.0,
      -0.3,
      b,
      Some(x0),
      Some(t),
      n,
      Unseeded,
    );
    assert_eq!(hw2.a, a);
    assert_eq!(hw2.theta.call(0.25), const_theta(0.25));

    let [x, u] = hw2.sample();

    let dt = t / (n as f64 - 1.0);
    let mut expected_x = x0;
    let mut expected_u = 0.0_f64;
    for i in 1..n {
      let next_x = expected_x + (const_theta(i as f64 * dt) + expected_u - a * expected_x) * dt;
      let next_u = expected_u - b * expected_u * dt;
      expected_x = next_x;
      expected_u = next_u;
      assert!((x[i] - expected_x).abs() < 1e-12, "x mismatch at {i}");
      assert!((u[i] - expected_u).abs() < 1e-12, "u mismatch at {i}");
    }
  }
}

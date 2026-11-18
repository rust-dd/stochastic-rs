//! $$
//! dS_t=S_t\sqrt{v_t}\,dW_t^1,\quad v_t = v_0^2\,\exp\!\bigl(\nu\,W_t^2 - \tfrac12\nu^2 t\bigr)
//! $$
//!
//! Single-factor log-normal Bergomi, not Bergomi's (2009) one-factor model: `H = ½`, no mean
//! reversion (`κ = 0`), flat `ξ_0(t) ≡ v_0²`; `rough_bergomi::RoughBergomi` is the rough model.
//!
//! ```text
//! v(t_i) = v_0² · exp(ν · Σ_{j<i} cgn2_j  −  ½ ν² t_i)
//! ```
//!
//! Reference: Bergomi (2005), "Smile Dynamics II", Risk 18(10), 67-73
//! Reference: Bergomi (2016), "Stochastic Volatility Modeling", §7

use ndarray::Array1;
use ndarray::s;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::device::Cpu;
use crate::noise::cgns::Cgns;
use crate::precondition::check_n;
use crate::precondition::check_t;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Debug, Clone)]
pub struct Bergomi<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Vol-of-vol ν scaling the log-variance driver's dispersion.
  pub nu: T,
  /// Initial volatility; the variance starts at `v0²`.
  pub v0: T,
  /// Initial spot.
  pub s0: T,
  /// Constant proportional drift rate of the asset (a GBM-style drift, not
  /// a correlation — that role belongs to `rho`).
  pub r: T,
  /// Correlation ρ between the asset's Brownian shock and the
  /// log-variance driver's innovations.
  pub rho: T,
  /// Number of points sampled along the Bergomi path.
  pub n: usize,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// The seed: [`Unseeded`] or [`Deterministic`](stochastic_rs_core::simd_rng::Deterministic).
  pub seed: S,
  cgns: Cgns<T>,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

/// Every field has a matching `with_*` builder setter, e.g.
/// `Bergomi::default().with_nu(0.6).with_rho(-0.2)`.
impl<T: FloatExt, S: SeedExt> Bergomi<T, S> {
  /// Panics unless `n >= 2`, a given `t` is positive and finite, and every parameter is finite.
  pub fn new(nu: T, v0: T, s0: T, r: T, rho: T, n: usize, t: Option<T>, seed: S) -> Self {
    Self::check_nu(nu);
    Self::check_v0(v0);
    Self::check_s0(s0);
    Self::check_r(r);
    Self::check_rho(rho);
    check_n(n);
    check_t(t);
    Self {
      backend: Cpu,
      nu,
      v0,
      s0,
      r,
      rho,
      n,
      t,
      seed,
      cgns: Cgns::new(rho, n - 1, t, Unseeded),
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Bergomi<T, S, B> {
  fn check_nu(nu: T) {
    require!(nu.is_finite(), nu);
  }

  fn check_v0(v0: T) {
    require!(v0.is_finite(), v0);
  }

  fn check_s0(s0: T) {
    require!(s0.is_finite(), s0);
  }

  fn check_r(r: T) {
    require!(r.is_finite(), r);
  }

  fn check_rho(rho: T) {
    require!(rho.is_finite(), rho);
  }

  /// Replace `nu`; panics unless `nu` is finite.
  pub fn with_nu(mut self, nu: T) -> Self {
    Self::check_nu(nu);
    self.nu = nu;
    self
  }

  /// Replace `v0`; panics unless `v0` is finite.
  pub fn with_v0(mut self, v0: T) -> Self {
    Self::check_v0(v0);
    self.v0 = v0;
    self
  }

  /// Replace `s0`; panics unless `s0` is finite.
  pub fn with_s0(mut self, s0: T) -> Self {
    Self::check_s0(s0);
    self.s0 = s0;
    self
  }

  /// Replace `r`; panics unless `r` is finite.
  pub fn with_r(mut self, r: T) -> Self {
    Self::check_r(r);
    self.r = r;
    self
  }

  /// Replace `rho` and rebuild the cached correlated-Gaussian generator `cgns`, which the sampler
  /// reads; panics unless `rho` is finite and in `[-1, 1]`.
  pub fn with_rho(mut self, rho: T) -> Self {
    Self::check_rho(rho);
    self.rho = rho;
    self.cgns = Cgns::new(rho, self.n - 1, self.t, Unseeded);
    self
  }

  /// Replace `n`, the grid points along the path; rebuilds the cached correlated-Gaussian
  /// generator, whose length and step size derive from `n`; panics unless `n >= 2`.
  pub fn with_points(mut self, n: usize) -> Self {
    check_n(n);
    self.n = n;
    self.cgns = Cgns::new(self.rho, n - 1, self.t, Unseeded);
    self
  }

  /// Replace the simulation horizon `t`; rebuilds the cached correlated-Gaussian generator's step
  /// size, which derives from `t`; panics unless `t`, when given, is positive and finite.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    check_t(t);
    self.t = t;
    self.cgns = Cgns::new(self.rho, self.n - 1, t, Unseeded);
    self
  }

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

/// ν=0.4, v₀=0.2, s₀=100, r=0.01, ρ=-0.6, a textbook Bergomi; t=1, 252 points, a trading year of
/// daily closes.
impl<T: FloatExt> Default for Bergomi<T, Unseeded> {
  fn default() -> Self {
    Self::new(
      T::from_f64_fast(0.4),
      T::from_f64_fast(0.2),
      T::from_f64_fast(100.0),
      T::from_f64_fast(0.01),
      T::from_f64_fast(-0.6),
      252,
      Some(T::one()),
      Unseeded,
    )
  }
}

/// The Euler engine's view: the variance is a function of the running sum of the correlated
/// increments, so that sum and the elapsed time are two further state components, not reported.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 2>
  for Bergomi<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::Bergomi {
      r: self.r,
      nu: self.nu,
      half_nu_sq: T::from_f64_fast(0.5) * self.nu * self.nu,
      v0_sq: self.v0 * self.v0,
      rho: self.rho,
    }
  }

  fn initial_state(&self) -> [T; 4] {
    [self.s0, self.v0 * self.v0, T::zero(), T::zero()]
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

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> [Array1<T>; 2] {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Bergomi<T, S> { nu, v0, s0, r, rho, n, t, seed, cgns } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Bergomi<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Bergomi<T, S, B> {
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = BergomiSampler<T, S>
  where
    Self: 's;

  /// Derives `self.seed` into the sampler, never clones it: the mixed next tick hash-scrambles
  /// chunk `i`'s basis against chunk `i+1`'s, where a raw snapshot leaves them one stride apart.
  fn sampler(&self) -> BergomiSampler<T, S> {
    BergomiSampler {
      n: self.n,
      nu: self.nu,
      v0_sq: self.v0.powi(2),
      s0: self.s0,
      r: self.r,
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

/// Reusable [`Bergomi`] sampling state: the correlated-Gaussian generator and the seed source.
#[doc(hidden)]
#[derive(Debug)]
pub struct BergomiSampler<T: FloatExt, S: SeedExt> {
  n: usize,
  nu: T,
  v0_sq: T,
  s0: T,
  r: T,
  dt: T,
  cgns: Cgns<T>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> BergomiSampler<T, S> {
  fn fill_paths(&mut self, s: &mut [T], v2: &mut [T]) {
    if self.n == 0 {
      return;
    }
    let dt = self.dt;
    let [cgn1, cgn2] = &self.cgns.sample_impl(&self.seed);

    s[0] = self.s0;
    v2[0] = self.v0_sq;

    for i in 1..self.n {
      s[i] = s[i - 1] + self.r * s[i - 1] * dt + v2[i - 1].sqrt() * s[i - 1] * cgn1[i - 1];

      let sum_z = cgn2.slice(s![..i]).sum();
      let t = T::from_usize_(i) * dt;
      v2[i] = self.v0_sq * (self.nu * sum_z - T::from_f64_fast(0.5) * self.nu.powi(2) * t).exp()
    }
  }
}

impl<T: FloatExt, S: SeedExt> crate::traits::Sealed for BergomiSampler<T, S> {}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for BergomiSampler<T, S> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [s, v2] = out;
    self.fill_paths(
      s.as_slice_mut().expect("Bergomi output must be contiguous"),
      v2.as_slice_mut()
        .expect("Bergomi output must be contiguous"),
    );
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    let mut s = Array1::<T>::zeros(self.n);
    let mut v2 = Array1::<T>::zeros(self.n);
    self.fill_paths(
      s.as_slice_mut().expect("contiguous"),
      v2.as_slice_mut().expect("contiguous"),
    );
    [s, v2]
  }
}

#[cfg(feature = "python")]
#[doc(hidden)]
#[pyo3::prelude::pyclass]
pub struct PyBergomi {
  inner_f32: Option<Bergomi<f32>>,
  inner_f64: Option<Bergomi<f64>>,
  seeded_f32: Option<Bergomi<f32, crate::python_device::SharedSeed>>,
  seeded_f64: Option<Bergomi<f64, crate::python_device::SharedSeed>>,
  /// The device the class samples on, chosen at construction.
  device: crate::python_device::Device,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyBergomi {
  #[new]
  #[pyo3(signature = (nu, r, rho, n, v0, s0, t=None, seed=None, dtype=None, device=None))]
  fn new(
    nu: f64,
    r: f64,
    rho: f64,
    n: usize,
    v0: f64,
    s0: f64,
    t: Option<f64>,
    seed: Option<u64>,
    dtype: Option<&str>,
    device: Option<&str>,
  ) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> pyo3::PyResult<Self> {
      let device = crate::python_device::Device::parse(device, dtype.unwrap_or("f64"))?;
      let mut s = Self {
        inner_f32: None,
        inner_f64: None,
        seeded_f32: None,
        seeded_f64: None,
        device,
      };
      match (seed, dtype.unwrap_or("f64")) {
        (Some(sd), "f32") => {
          s.seeded_f32 = Some(Bergomi::new(
            nu as f32,
            v0 as f32,
            s0 as f32,
            r as f32,
            rho as f32,
            n,
            t.map(|v| v as f32),
            crate::python_device::SharedSeed::new(sd),
          ));
        }
        (Some(sd), _) => {
          s.seeded_f64 = Some(Bergomi::new(
            nu,
            v0,
            s0,
            r,
            rho,
            n,
            t,
            crate::python_device::SharedSeed::new(sd),
          ));
        }
        (None, "f32") => {
          s.inner_f32 = Some(Bergomi::new(
            nu as f32,
            v0 as f32,
            s0 as f32,
            r as f32,
            rho as f32,
            n,
            t.map(|v| v as f32),
            Unseeded,
          ));
        }
        (None, _) => {
          s.inner_f64 = Some(Bergomi::new(nu, v0, s0, r, rho, n, t, Unseeded));
        }
      }
      Ok(s)
    })?
  }

  /// Why a device kernel cannot carry this configuration, if it cannot; the answer depends on
  /// the configuration alone, whatever `device=` was passed.
  fn device_fallback(&self) -> Option<&'static str> {
    use crate::traits::ProcessExt;
    py_dispatch!(self, |inner| inner.device_fallback())
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
      py_device_dispatch!(self, |inner| {
        let [a, b] = inner.sample();
        (
          a.into_pyarray(py).into_py_any(py).unwrap(),
          b.into_pyarray(py).into_py_any(py).unwrap(),
        )
      })
    })
  }

  fn sample_par<'py>(
    &self,
    py: pyo3::Python<'py>,
    m: usize,
  ) -> pyo3::PyResult<(pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>)> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use numpy::IntoPyArray;
      use numpy::ndarray::Array2;
      use pyo3::IntoPyObjectExt;

      use crate::traits::ProcessExt;
      py_device_dispatch!(self, |inner| {
        let samples = inner.sample_par(m);
        let n = samples[0][0].len();
        let mut r0 = Array2::zeros((m, n));
        let mut r1 = Array2::zeros((m, n));
        for (i, [a, b]) in samples.iter().enumerate() {
          r0.row_mut(i).assign(a);
          r1.row_mut(i).assign(b);
        }
        (
          r0.into_pyarray(py).into_py_any(py).unwrap(),
          r1.into_pyarray(py).into_py_any(py).unwrap(),
        )
      })
    })
  }
}

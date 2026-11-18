//! $$
//! dS_t=\mu S_tdt+\sqrt{v_t}S_tdW_t,\quad dv_t=\kappa(\theta-v_t)dt+\xi\sqrt{v_t}dB_t^H
//! $$
//!
//! A Heston model whose variance is driven by fractional Gaussian noise through a one-factor
//! carrier plus an `O(n²)` memory sum, this crate's own scheme; rough Heston is `rough::RlHeston`.
//!
//! Reference: Gatheral et al. (2018), Quant. Fin. 18(6), 933–949, DOI 10.1080/14697688.2017.1393551
//! Reference: El Euch & Rosenbaum (2019), Math. Finance 29(1), 3–38, DOI 10.1111/mafi.12173
//! Reference: Abi Jaber & El Euch (2019), SIAM J. Fin. Math. 10(2), 309–349, DOI 10.1137/18M1170236
//!

use ndarray::Array1;
#[cfg(feature = "python")]
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::special::gamma;

use crate::device::Cpu;
use crate::noise::cgns::Cgns;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Debug, Clone)]
pub struct Fheston<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Hurst exponent controlling roughness and long-memory.
  pub hurst: T,
  /// Initial volatility; the variance starts at `v0²`.
  pub v0: T,
  /// Long-run variance level θ the Markovian lifting factor `yt` reverts
  /// toward (θ in the module header).
  pub theta: T,
  /// Mean-reversion speed κ of the Markovian lifting factor `yt`.
  pub kappa: T,
  /// Vol-of-vol ξ scaling the local (`zt`) and memory (Volterra integral) correction terms of
  /// the bounded-state approximation of the fractional kernel.
  pub nu: T,
  /// Coefficient of the local correction term `zt` in the two-term approximation of the
  /// fractional kernel; `None` is 1, the untruncated coefficient.
  pub c1: Option<T>,
  /// Calibration coefficient scaling the memory (Volterra-integral)
  /// correction term in the same approximation; `None` defaults to 1.
  pub c2: Option<T>,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Number of points sampled along the rough-Heston path.
  pub n: usize,
  /// Drift of the log-price.
  pub mu: T,
  /// Initial spot.
  pub s0: T,
  /// Correlation of the price and volatility innovations.
  pub rho: T,
  /// The seed: [`Unseeded`] or [`Deterministic`](stochastic_rs_core::simd_rng::Deterministic).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Fheston<T, S> {
  /// Starts at spot 1 with zero drift and correlation, which `with_s0`, `with_mu`, `with_rho` set.
  /// Panics unless a given `t` is positive and finite, `n >= 2`, and every parameter is finite.
  pub fn new(
    hurst: T,
    v0: T,
    theta: T,
    kappa: T,
    nu: T,
    c1: Option<T>,
    c2: Option<T>,
    t: Option<T>,
    n: usize,
    seed: S,
  ) -> Self {
    assert!(
      hurst.is_finite(),
      "hurst must satisfy `hurst.is_finite()`, got hurst = {hurst:?}"
    );
    assert!(
      v0.is_finite(),
      "v0 must satisfy `v0.is_finite()`, got v0 = {v0:?}"
    );
    assert!(
      theta.is_finite(),
      "theta must satisfy `theta.is_finite()`, got theta = {theta:?}"
    );
    assert!(
      kappa.is_finite(),
      "kappa must satisfy `kappa.is_finite()`, got kappa = {kappa:?}"
    );
    assert!(
      nu.is_finite(),
      "nu must satisfy `nu.is_finite()`, got nu = {nu:?}"
    );
    assert!(
      c1.is_none_or(|v| v.is_finite()),
      "c1 must satisfy `c1.is_none_or(|v| v.is_finite())`, got c1 = {c1:?}"
    );
    assert!(
      c2.is_none_or(|v| v.is_finite()),
      "c2 must satisfy `c2.is_none_or(|v| v.is_finite())`, got c2 = {c2:?}"
    );
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    Fheston {
      backend: Cpu,
      hurst,
      v0,
      theta,
      kappa,
      nu,
      c1,
      c2,
      t,
      n,
      mu: T::zero(),
      s0: T::one(),
      rho: T::zero(),
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Fheston<T, S, B> {
  fn check_s0(s0: T) {
    require!(s0.is_finite(), s0);
    require!(s0 > T::zero(), s0);
  }

  fn check_mu(mu: T) {
    require!(mu.is_finite(), mu);
  }

  fn check_rho(rho: T) {
    require!((-T::one()..=T::one()).contains(&rho), rho);
  }

  /// Replace `s0`, which the multiplicative price step needs positive; panics unless `s0 > 0`
  /// and `s0` is finite.
  pub fn with_s0(mut self, s0: T) -> Self {
    Self::check_s0(s0);
    self.s0 = s0;
    self
  }

  /// Replace `mu`; panics unless `mu` is finite.
  pub fn with_mu(mut self, mu: T) -> Self {
    Self::check_mu(mu);
    self.mu = mu;
    self
  }

  /// Replace `rho`; panics unless `-1 <= rho <= 1`.
  pub fn with_rho(mut self, rho: T) -> Self {
    Self::check_rho(rho);
    self.rho = rho;
    self
  }

  /// The grid spacing, zero for a single point.
  fn dt(&self) -> T {
    if self.n > 1 {
      self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
    } else {
      T::zero()
    }
  }

  /// The memory kernel on the grid, `((m + 1) dt)^{H - 1/2} dt` at lag `m`: the host sums the
  /// local factor against it, and the device reads it as its first curve.
  fn memory_weights(&self) -> Vec<T> {
    let dt = self.dt();
    let half = T::from_f64_fast(0.5);
    (0..self.n)
      .map(|m| (T::from_usize_(m + 1) * dt).powf(self.hurst - half) * dt)
      .collect()
  }
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 2>
  for Fheston<T, S, B>
{
  /// The host's constants folded once: the factor's decay over a step and the
  /// reciprocal of `Γ(H - 1/2)` the memory term is divided by.
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::RoughHestonMemory {
      mu: self.mu,
      theta: self.theta,
      ek: (-self.kappa * self.dt()).exp(),
      nu: self.nu,
      c1: self.c1.unwrap_or(T::one()),
      c2: self.c2.unwrap_or(T::one()),
      inv_g: T::from_f64_fast(1.0 / gamma(self.hurst.to_f64().unwrap() - 0.5)),
      rho: self.rho,
    }
  }

  /// Spot, variance, and the two factors behind it: `y` starts at the
  /// variance and `z` at zero, as on the host.
  fn initial_state(&self) -> [T; 4] {
    let v0_sq = self.v0.powi(2);
    [self.s0, v0_sq, v0_sq, T::zero()]
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  fn time_step(&self) -> T {
    self.dt()
  }

  fn curves(&self) -> Option<Vec<Vec<T>>> {
    Some(vec![self.memory_weights()])
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

backend_switch!([T: FloatExt, S: SeedExt] Fheston<T, S> { hurst, v0, theta, kappa, nu, c1, c2, t, n, mu, s0, rho, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Fheston<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Fheston<T, S, B> {
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = FhestonSampler<T, S>
  where
    Self: 's;

  /// Derives `self.seed` into the sampler, which drives `cgns` through `sample_impl`, so every
  /// draw of the path follows the seed.
  fn sampler(&self) -> FhestonSampler<T, S> {
    let n_steps = self.n.saturating_sub(1);
    let dt = if n_steps > 0 {
      self.t.unwrap_or(T::one()) / T::from_usize_(n_steps)
    } else {
      T::zero()
    };
    // `cgns` keeps its unread `Unseeded` seed: the sampler drives it through
    // `sample_impl(&self.seed)`, as `Bates` drives its own.
    FhestonSampler {
      n: self.n,
      hurst: self.hurst,
      theta: self.theta,
      kappa: self.kappa,
      nu: self.nu,
      c1: self.c1.unwrap_or(T::one()),
      c2: self.c2.unwrap_or(T::one()),
      mu: self.mu,
      s0: self.s0,
      v0_sq: self.v0.powi(2),
      dt,
      g: gamma(self.hurst.to_f64().unwrap() - 0.5),
      cgns: Cgns::new(self.rho, n_steps, self.t, Unseeded),
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine when the grid fits the kernels' per-path history; a longer grid
  /// stays on the host, chunked as [`ProcessExt`] chunks.
  fn sample(&self) -> [Array1<T>; 2] {
    if self.device_ready() {
      self.backend.system_sample(self)
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&[Array1<T>; 2]) -> R + Sync) -> Vec<R> {
    if self.device_ready() {
      self.backend.system_paths_map(self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, f)
    }
  }

  fn sample_par(&self, m: usize) -> Vec<[Array1<T>; 2]> {
    if self.device_ready() {
      self.backend.system_paths(self, m)
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<[Array1<T>; 2], crate::device::DeviceError> {
    if self.device_ready() {
      self.backend.try_system_sample(self)
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<[Array1<T>; 2]>, crate::device::DeviceError> {
    if self.device_ready() {
      self.backend.try_system_paths(self, m)
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }

  /// What a device runs here: the grid fits the kernels'
  /// per-path history. A longer grid samples on the host.
  fn device_fallback(&self) -> Option<&'static str> {
    (!(self.n <= crate::euler::HISTORY_SLOTS))
      .then_some("a grid longer than the kernels' per-path history")
  }
}

/// Reusable [`Fheston`] sampling state: the correlated-Gaussian generator, an already-derived
/// seed to drive it, and the precomputed Volterra-kernel constants.
#[doc(hidden)]
#[derive(Debug)]
pub struct FhestonSampler<T: FloatExt, S: SeedExt> {
  n: usize,
  hurst: T,
  theta: T,
  kappa: T,
  nu: T,
  c1: T,
  c2: T,
  mu: T,
  s0: T,
  v0_sq: T,
  dt: T,
  g: f64,
  cgns: Cgns<T>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> FhestonSampler<T, S> {
  fn fill_paths(&mut self, s: &mut [T], v2: &mut [T]) {
    if self.n == 0 {
      return;
    }
    let dt = self.dt;

    let [gn_vol, gn_price] = self.cgns.sample_impl(&self.seed);

    let mut yt = Array1::<T>::zeros(self.n);
    let mut zt = Array1::<T>::zeros(self.n);
    let mut sigma_tilde2 = Array1::<T>::zeros(self.n);

    let v0_sq = self.v0_sq;
    let mu = self.mu;

    yt[0] = v0_sq;
    zt[0] = T::zero();
    sigma_tilde2[0] = v0_sq;
    v2[0] = v0_sq;
    s[0] = self.s0;
    let g = self.g;
    let half = T::from_f64_fast(0.5);

    for i in 1..self.n {
      let t_i = dt * T::from_usize_(i);
      yt[i] = self.theta + (yt[i - 1] - self.theta) * (-self.kappa * dt).exp();
      zt[i] = zt[i - 1] * (-self.kappa * dt).exp()
        + sigma_tilde2[i - 1].max(T::zero()).sqrt() * gn_vol[i - 1];

      sigma_tilde2[i] = yt[i] + self.nu * zt[i];

      let integral = (0..i)
        .map(|j| {
          let tj = T::from_usize_(j) * dt;
          ((t_i - tj).powf(self.hurst - half) * zt[j]) * dt
        })
        .sum::<T>();

      v2[i] =
        yt[i] + self.c1 * self.nu * zt[i] + self.c2 * self.nu * integral / T::from_f64_fast(g);

      // Price path: gn_price is already rho-correlated with gn_vol via Cgns
      let vi = v2[i - 1].max(T::zero());
      let log_inc = (mu - half * vi) * dt + vi.sqrt() * gn_price[i - 1];
      s[i] = s[i - 1] * log_inc.exp();
    }
  }
}

impl<T: FloatExt, S: SeedExt> crate::traits::Sealed for FhestonSampler<T, S> {}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for FhestonSampler<T, S> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [s, v2] = out;
    self.fill_paths(
      s.as_slice_mut().expect("Fheston output must be contiguous"),
      v2.as_slice_mut()
        .expect("Fheston output must be contiguous"),
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
pub struct PyRoughHeston {
  inner_f32: Option<Fheston<f32>>,
  inner_f64: Option<Fheston<f64>>,
  seeded_f32: Option<Fheston<f32, stochastic_rs_core::simd_rng::Deterministic>>,
  seeded_f64: Option<Fheston<f64, stochastic_rs_core::simd_rng::Deterministic>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyRoughHeston {
  #[new]
  #[pyo3(signature = (hurst, theta, kappa, nu, n, v0, c1=None, c2=None, t=None, mu=0.0, s0=1.0, rho=0.0, seed=None, dtype=None))]
  fn new(
    hurst: f64,
    theta: f64,
    kappa: f64,
    nu: f64,
    n: usize,
    v0: f64,
    c1: Option<f64>,
    c2: Option<f64>,
    t: Option<f64>,
    mu: f64,
    s0: f64,
    rho: f64,
    seed: Option<u64>,
    dtype: Option<&str>,
  ) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| {
      let mut obj = Self {
        inner_f32: None,
        inner_f64: None,
        seeded_f32: None,
        seeded_f64: None,
      };
      match (seed, dtype.unwrap_or("f64")) {
        (Some(sd), "f32") => {
          let m = Fheston::new(
            hurst as f32,
            v0 as f32,
            theta as f32,
            kappa as f32,
            nu as f32,
            c1.map(|v| v as f32),
            c2.map(|v| v as f32),
            t.map(|v| v as f32),
            n,
            Deterministic::new(sd),
          );
          let m = m.with_mu(mu as f32).with_s0(s0 as f32).with_rho(rho as f32);
          obj.seeded_f32 = Some(m);
        }
        (Some(sd), _) => {
          let m = Fheston::new(
            hurst,
            v0,
            theta,
            kappa,
            nu,
            c1,
            c2,
            t,
            n,
            Deterministic::new(sd),
          );
          let m = m.with_mu(mu).with_s0(s0).with_rho(rho);
          obj.seeded_f64 = Some(m);
        }
        (None, "f32") => {
          let m = Fheston::new(
            hurst as f32,
            v0 as f32,
            theta as f32,
            kappa as f32,
            nu as f32,
            c1.map(|v| v as f32),
            c2.map(|v| v as f32),
            t.map(|v| v as f32),
            n,
            Unseeded,
          );
          let m = m.with_mu(mu as f32).with_s0(s0 as f32).with_rho(rho as f32);
          obj.inner_f32 = Some(m);
        }
        (None, _) => {
          let m = Fheston::new(hurst, v0, theta, kappa, nu, c1, c2, t, n, Unseeded);
          let m = m.with_mu(mu).with_s0(s0).with_rho(rho);
          obj.inner_f64 = Some(m);
        }
      }
      obj
    })
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use numpy::IntoPyArray;
      use pyo3::IntoPyObjectExt;

      use crate::traits::ProcessExt;
      py_dispatch!(self, |inner| {
        let [s, v] = inner.sample();
        (s.into_pyarray(py), v.into_pyarray(py))
          .into_py_any(py)
          .unwrap()
      })
    })
  }

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
      py_dispatch!(self, |inner| {
        let paths = inner.sample_par(m);
        let n = paths[0][0].len();
        let mut s_result = Array2::zeros((m, n));
        let mut v_result = Array2::zeros((m, n));
        for (i, [s, v]) in paths.iter().enumerate() {
          s_result.row_mut(i).assign(s);
          v_result.row_mut(i).assign(v);
        }
        (s_result.into_pyarray(py), v_result.into_pyarray(py))
          .into_py_any(py)
          .unwrap()
      })
    })
  }
}

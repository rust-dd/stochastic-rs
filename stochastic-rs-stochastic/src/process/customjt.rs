//! # Customjt
//!
//! $$
//! T_k=\sum_{j=1}^{k}\tau_j,\quad \tau_j\sim D
//! $$
//!
//! Generic renewal-process jump-time generator: cumulative sum of
//! user-supplied inter-arrival draws `τ_j ~ D`, replacing the fixed-rate
//! exponential clock of a standard Poisson process. Produces jump *times*
//! only — no jump sizes and no continuous diffusion; see
//! [`CompoundCustom`](crate::process::ccustom::CompoundCustom) for the
//! type that attaches jump sizes on top.
//!

use std::any::Any;

use ndarray::Array1;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::SimdRng;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

pub struct CustomJt<T, D, S: SeedExt = Unseeded, B = Cpu>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  /// Optional fixed number of generated events.
  pub n: Option<usize>,
  /// Optional horizon for time-based generation.
  /// Used when `n` is `None`.
  pub t_max: Option<T>,
  /// Distribution used for generated increments / inter-arrival draws.
  pub distribution: D,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

#[inline]
fn validate_n_or_tmax<T: FloatExt>(n: Option<usize>, t_max: Option<T>, type_name: &'static str) {
  if n.is_none() && t_max.is_none() {
    panic!("{type_name}: n or t_max must be provided");
  }
}

impl<T, D, S: SeedExt> CustomJt<T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub fn new(n: Option<usize>, t_max: Option<T>, distribution: D, seed: S) -> Self {
    validate_n_or_tmax(n, t_max, "CustomJt");
    CustomJt {
      backend: Cpu,
      n,
      t_max,
      distribution,
      seed,
    }
  }
}

impl<T, D, S: SeedExt, B> CustomJt<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyCustomJt {
  inner_f32: Option<CustomJt<f32, crate::traits::CallableDist<f32>>>,
  inner_f64: Option<CustomJt<f64, crate::traits::CallableDist<f64>>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyCustomJt {
  #[new]
  #[pyo3(signature = (distribution, n=None, t_max=None, dtype=None))]
  fn new(
    distribution: pyo3::Py<pyo3::PyAny>,
    n: Option<usize>,
    t_max: Option<f64>,
    dtype: Option<&str>,
  ) -> Self {
    match dtype.unwrap_or("f64") {
      "f32" => Self {
        inner_f32: Some(CustomJt::new(
          n,
          t_max.map(|v| v as f32),
          crate::traits::CallableDist::new(distribution),
          Unseeded,
        )),
        inner_f64: None,
      },
      _ => Self {
        inner_f32: None,
        inner_f64: Some(CustomJt::new(
          n,
          t_max,
          crate::traits::CallableDist::new(distribution),
          Unseeded,
        )),
      },
    }
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
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
  }

  fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use numpy::ndarray::Array2;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    if let Some(ref inner) = self.inner_f64 {
      let paths = inner.sample_par(m);
      let n = paths[0].len();
      let mut result = Array2::<f64>::zeros((m, n));
      for (i, path) in paths.iter().enumerate() {
        result.row_mut(i).assign(path);
      }
      result.into_pyarray(py).into_py_any(py).unwrap()
    } else if let Some(ref inner) = self.inner_f32 {
      let paths = inner.sample_par(m);
      let n = paths[0].len();
      let mut result = Array2::<f32>::zeros((m, n));
      for (i, path) in paths.iter().enumerate() {
        result.row_mut(i).assign(path);
      }
      result.into_pyarray(py).into_py_any(py).unwrap()
    } else {
      unreachable!()
    }
  }
}

impl<T, D, S: SeedExt, B> CustomJt<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  /// One-shot sample from any seed source. Retained for callers such as
  /// [`crate::process::ccustom`] that need the event times directly.
  pub(crate) fn sample_impl<S2: SeedExt>(&self, seed: &S2) -> Array1<T> {
    self.sampler_impl(seed).sample()
  }

  /// Build the reusable sampling state from any seed source, deriving the RNG
  /// exactly as the legacy `sample_impl` body did per mode.
  pub(crate) fn sampler_impl<S2: SeedExt>(&self, seed: &S2) -> CustomJtSampler<'_, T, D> {
    let mode = if let Some(n) = self.n {
      CustomJtMode::Count { n }
    } else if let Some(t_max) = self.t_max {
      // Horizon mode advanced the seed once more before drawing.
      seed.derive();
      CustomJtMode::Horizon { t_max }
    } else {
      unreachable!("validate_n_or_tmax ensures at least one of n, t_max is set")
    };
    CustomJtSampler {
      distribution: &self.distribution,
      rng: seed.rng(),
      mode,
    }
  }
}

impl<T, D, S: SeedExt, B> CustomJt<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
  /// The arrival intensity when the inter-arrival law is exponential, which
  /// is when the arrivals are the Poisson stream the kernels draw.
  fn device_intensity(&self) -> Option<T> {
    crate::process::cpoisson::device_arrival_rate(&self.distribution)
  }

  /// Whether a device can run this process: a fixed number of arrivals — the
  /// horizon mode has no grid — under an exponential inter-arrival law.
  fn device_ready(&self) -> bool {
    self.n.is_some() && self.device_intensity().is_some()
  }
}

impl<T, D, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for CustomJt<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
  /// Poisson arrivals at the exponential law's rate. Any other law never
  /// reaches a launch — [`ProcessExt::sample`] keeps it on the host — so
  /// asking here is a caller bypassing that guard.
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::PoissonArrivals {
      lambda: self.device_intensity().expect(
        "CustomJt: the inter-arrival distribution is not exponential, which is the only arrival \
         law the Euler engine draws on a device; sample through `ProcessExt`, which keeps it on \
         the host",
      ),
    }
  }

  fn initial_value(&self) -> T {
    T::zero()
  }

  fn grid_points(&self) -> usize {
    self
      .n
      .expect("the Euler engine describes CustomJt's count mode; horizon mode has no grid")
  }

  fn horizon(&self) -> T {
    T::one()
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

backend_switch!([T, D, S: SeedExt] CustomJt<T, D, S> { n, t_max, distribution, seed } via euler where  T: FloatExt,  D: Distribution<T> + Send + Sync);

impl<T, D, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for CustomJt<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
  type Output = Array1<T>;
  type Sampler<'s>
    = CustomJtSampler<'s, T, D>
  where
    Self: 's;

  fn sampler(&self) -> CustomJtSampler<'_, T, D> {
    self.sampler_impl(&self.seed)
  }

  /// Through the Euler engine when the arrival count is fixed and the
  /// inter-arrivals are exponential; anything else keeps the process on the host,
  /// chunked exactly as [`ProcessExt`] chunks.
  fn sample(&self) -> Array1<T> {
    if self.device_ready() {
      crate::euler::EulerBackend::euler_sample(&self.backend, self)
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    if self.device_ready() {
      crate::euler::EulerBackend::euler_paths_map(&self.backend, self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, f)
    }
  }

  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    if self.device_ready() {
      crate::euler::EulerBackend::euler_paths(&self.backend, self, m)
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<Array1<T>, crate::device::DeviceError> {
    if self.device_ready() {
      crate::euler::EulerBackend::try_sample(&self.backend, self)
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, crate::device::DeviceError> {
    if self.device_ready() {
      crate::euler::EulerBackend::try_euler_paths(&self.backend, self, m)
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }
}

/// Sampling regime: a fixed event count or a time horizon (variable-length).
enum CustomJtMode<T: FloatExt> {
  Count { n: usize },
  Horizon { t_max: T },
}

/// Reusable [`CustomJt`] sampling state: an owned RNG plus a borrow of the
/// process's increment distribution (shared, `Sync`), and the sampling regime.
#[doc(hidden)]
pub struct CustomJtSampler<'a, T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  distribution: &'a D,
  rng: SimdRng,
  mode: CustomJtMode<T>,
}

impl<T, D> CustomJtSampler<'_, T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  /// Count-mode fill: `out[0] = 0`, then the running sum of `n - 1` draws.
  fn fill_count(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = T::zero();
    let mut acc = T::zero();
    for x in out[1..].iter_mut() {
      acc += self.distribution.sample(&mut self.rng);
      *x = acc;
    }
  }

  /// Horizon-mode: accumulate event times until the horizon is crossed.
  fn sample_horizon(&mut self, t_max: T) -> Array1<T> {
    let mut x = Vec::with_capacity(16);
    x.push(T::zero());
    let mut t = T::zero();
    while t < t_max {
      t += self.distribution.sample(&mut self.rng);
      x.push(t);
    }
    Array1::from(x)
  }
}

impl<T, D> PathSampler<T> for CustomJtSampler<'_, T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    match self.mode {
      CustomJtMode::Count { .. } => {
        let slice = out
          .as_slice_mut()
          .expect("CustomJt output must be contiguous");
        self.fill_count(slice);
      }
      CustomJtMode::Horizon { t_max } => {
        *out = self.sample_horizon(t_max);
      }
    }
  }

  fn sample(&mut self) -> Array1<T> {
    match self.mode {
      CustomJtMode::Count { n } => array1_from_fill(n, |out| self.fill_count(out)),
      CustomJtMode::Horizon { t_max } => self.sample_horizon(t_max),
    }
  }
}

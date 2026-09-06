//! # Ccustom
//!
//! $$
//! X_t=\sum_{k=1}^{N_t}J_k
//! $$
//!
//! Compound jump process with a user-supplied inter-arrival-time
//! distribution (via [`CustomJt`]) instead of a fixed-rate Poisson clock —
//! no continuous diffusion component; this generates the pure jump sum
//! `X_t`, meant to be added on top of a diffusion elsewhere.
//!

use std::any::Any;

use ndarray::Array1;
use ndarray::Axis;
use rand_distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use super::customjt::CustomJt;
use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

pub struct CompoundCustom<T, D1, D2, S: SeedExt = Unseeded, B = Cpu>
where
  T: FloatExt,
  D1: Distribution<T> + Send + Sync,
  D2: Distribution<T> + Send + Sync,
{
  /// Optional fixed number of generated events.
  pub n: Option<usize>,
  /// Optional horizon for time-based generation.
  /// Used when `n` is `None`.
  pub t_max: Option<T>,
  /// Distribution of jump magnitudes.
  pub jumps_distribution: D1,
  /// Distribution of jump waiting times / event times.
  pub jump_times_distribution: D2,
  /// Underlying jump-time generator used internally.
  pub customjt: CustomJt<T, D2>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T, D1, D2, S: SeedExt> CompoundCustom<T, D1, D2, S>
where
  T: FloatExt,
  D1: Distribution<T> + Send + Sync,
  D2: Distribution<T> + Send + Sync,
{
  pub fn new(
    n: Option<usize>,
    t_max: Option<T>,
    jumps_distribution: D1,
    jump_times_distribution: D2,
    customjt: CustomJt<T, D2>,
    seed: S,
  ) -> Self {
    if n.is_none() && t_max.is_none() {
      panic!("CompoundCustom: n or t_max must be provided");
    }

    Self {
      backend: Cpu,
      n,
      t_max,
      jumps_distribution,
      jump_times_distribution,
      customjt,
      seed,
    }
  }
}

impl<T, D1, D2, S: SeedExt, B> CompoundCustom<T, D1, D2, S, B>
where
  T: FloatExt,
  D1: Distribution<T> + Send + Sync,
  D2: Distribution<T> + Send + Sync,
{
}

impl<T, D1, D2, S: SeedExt, B> CompoundCustom<T, D1, D2, S, B>
where
  T: FloatExt,
  D1: Distribution<T> + Send + Sync + Any,
  D2: Distribution<T> + Send + Sync + Any,
{
  /// The arrival intensity when the inter-arrival law is exponential, which
  /// is when the arrivals are the Poisson stream the kernels draw.
  fn device_intensity(&self) -> Option<T> {
    crate::process::cpoisson::device_arrival_rate(&self.jump_times_distribution)
  }

  /// The jump-size law as the kernels draw it, one size per arrival; `None`
  /// for a distribution they do not carry.
  fn device_jump_sizes(&self) -> Option<crate::euler::JumpSizes<T>> {
    crate::process::cpoisson::device_jump_sizes(&self.jumps_distribution)
      .and_then(crate::euler::JumpSizes::single)
  }

  /// Whether a device can run this process: a fixed number of arrivals — the
  /// horizon mode has no grid — exponential inter-arrivals, and sizes under a
  /// law the kernels draw.
  fn device_ready(&self) -> bool {
    self.n.is_some() && self.device_intensity().is_some() && self.device_jump_sizes().is_some()
  }
}

impl<T, D1, D2, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 3>
  for CompoundCustom<T, D1, D2, S, B>
where
  T: FloatExt,
  D1: Distribution<T> + Send + Sync + Any,
  D2: Distribution<T> + Send + Sync + Any,
{
  /// Poisson arrivals at the exponential law's rate. Any other law never
  /// reaches a launch — [`ProcessExt::sample`] keeps it on the host — so
  /// asking here is a caller bypassing that guard.
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::CompoundPoissonEvents {
      lambda: self.device_intensity().expect(
        "CompoundCustom: the inter-arrival distribution is not exponential, which is the only \
         arrival law the Euler engine draws on a device; sample through `ProcessExt`, which \
         keeps it on the host",
      ),
    }
  }

  fn initial_state(&self) -> [T; 4] {
    [T::zero(); 4]
  }

  fn grid_points(&self) -> usize {
    self
      .n
      .expect("the Euler engine describes CompoundCustom's count mode; horizon mode has no grid")
  }

  fn horizon(&self) -> T {
    T::one()
  }

  fn jump_intensity(&self) -> Option<T> {
    self.device_intensity()
  }

  fn jump_sizes(&self) -> Option<crate::euler::JumpSizes<T>> {
    Some(self.device_jump_sizes().expect(
      "CompoundCustom: the jump-size distribution is not a law the Euler engine draws on a \
       device; sample through `ProcessExt`, which keeps it on the host",
    ))
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> [Array1<T>; 3] {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T, D1, D2, S: SeedExt] CompoundCustom<T, D1, D2, S> { n, t_max, jumps_distribution, jump_times_distribution, customjt, seed } via euler where  T: FloatExt,  D1: Distribution<T> + Send + Sync,  D2: Distribution<T> + Send + Sync);

impl<T, D1, D2, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for CompoundCustom<T, D1, D2, S, B>
where
  T: FloatExt,
  D1: Distribution<T> + Send + Sync + Any,
  D2: Distribution<T> + Send + Sync + Any,
{
  type Output = [Array1<T>; 3];
  type Sampler<'s>
    = CompoundCustomSampler<'s, T, D1, D2, S>
  where
    Self: 's;

  /// Derives (not clones) `self.seed` into the returned sampler: the
  /// derived value is `self.seed`'s *mixed* next tick, not a raw snapshot,
  /// so chunk `i`'s basis and chunk `i+1`'s basis are hash-scrambled
  /// relative to each other rather than one raw stride apart.
  fn sampler(&self) -> CompoundCustomSampler<'_, T, D1, D2, S> {
    CompoundCustomSampler {
      n: self.n,
      jumps_distribution: &self.jumps_distribution,
      customjt: &self.customjt,
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine when the arrival count is fixed, the
  /// inter-arrivals are exponential and the jump-size law is one the device
  /// kernels draw; anything else keeps the process on the host,
  /// chunked exactly as [`ProcessExt`] chunks.
  fn sample(&self) -> [Array1<T>; 3] {
    if self.device_ready() {
      crate::euler::EulerBackend::system_sample(&self.backend, self)
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&[Array1<T>; 3]) -> R + Sync) -> Vec<R> {
    if self.device_ready() {
      crate::euler::EulerBackend::system_paths_map(&self.backend, self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, f)
    }
  }

  fn sample_par(&self, m: usize) -> Vec<[Array1<T>; 3]> {
    if self.device_ready() {
      crate::euler::EulerBackend::system_paths(&self.backend, self, m)
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<[Array1<T>; 3], crate::device::DeviceError> {
    if self.device_ready() {
      crate::euler::EulerBackend::try_system_sample(&self.backend, self)
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<[Array1<T>; 3]>, crate::device::DeviceError> {
    if self.device_ready() {
      crate::euler::EulerBackend::try_system_paths(&self.backend, self, m)
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }
}

/// Reusable [`CompoundCustom`] sampling state: borrows the (non-`Clone`) jump and
/// jump-time distributions and owns the seed source. The per-call output length
/// is event-count-dependent, so the three buffers are reallocated each call; the
/// seed advances exactly as the legacy `sample` body did.
#[doc(hidden)]
pub struct CompoundCustomSampler<'a, T, D1, D2, S: SeedExt>
where
  T: FloatExt,
  D1: Distribution<T> + Send + Sync,
  D2: Distribution<T> + Send + Sync,
{
  n: Option<usize>,
  jumps_distribution: &'a D1,
  customjt: &'a CustomJt<T, D2>,
  seed: S,
}

impl<T, D1, D2, S: SeedExt> CompoundCustomSampler<'_, T, D1, D2, S>
where
  T: FloatExt,
  D1: Distribution<T> + Send + Sync,
  D2: Distribution<T> + Send + Sync,
{
  fn sample_inner(&mut self) -> [Array1<T>; 3] {
    let p = self.customjt.sample_impl(&self.seed);
    let mut jumps = Array1::<T>::zeros(self.n.unwrap_or(p.len()));
    let mut rng = self.seed.rng();
    for i in 1..p.len() {
      jumps[i] = self.jumps_distribution.sample(&mut rng);
    }

    let mut cum_jupms = jumps.clone();
    cum_jupms.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);

    [p, cum_jupms, jumps]
  }
}

impl<T, D1, D2, S: SeedExt> PathSampler<T> for CompoundCustomSampler<'_, T, D1, D2, S>
where
  T: FloatExt,
  D1: Distribution<T> + Send + Sync,
  D2: Distribution<T> + Send + Sync,
{
  type Output = [Array1<T>; 3];

  fn sample_into(&mut self, out: &mut [Array1<T>; 3]) {
    *out = self.sample_inner();
  }

  fn sample(&mut self) -> [Array1<T>; 3] {
    self.sample_inner()
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyCompoundCustom {
  inner_f32:
    Option<CompoundCustom<f32, crate::traits::CallableDist<f32>, crate::traits::CallableDist<f32>>>,
  inner_f64:
    Option<CompoundCustom<f64, crate::traits::CallableDist<f64>, crate::traits::CallableDist<f64>>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyCompoundCustom {
  #[new]
  #[pyo3(signature = (jumps_distribution, jump_times_distribution, n=None, t_max=None, dtype=None))]
  fn new(
    jumps_distribution: pyo3::Py<pyo3::PyAny>,
    jump_times_distribution: pyo3::Py<pyo3::PyAny>,
    n: Option<usize>,
    t_max: Option<f64>,
    dtype: Option<&str>,
  ) -> Self {
    match dtype.unwrap_or("f64") {
      "f32" => {
        let (jt_dist, customjt_dist) = pyo3::Python::attach(|py| {
          let a = jump_times_distribution.clone_ref(py);
          let b = jump_times_distribution;
          (
            crate::traits::CallableDist::<f32>::new(a),
            crate::traits::CallableDist::<f32>::new(b),
          )
        });
        let customjt = CustomJt::new(n, t_max.map(|v| v as f32), customjt_dist, Unseeded);
        Self {
          inner_f32: Some(CompoundCustom::new(
            n,
            t_max.map(|v| v as f32),
            crate::traits::CallableDist::new(jumps_distribution),
            jt_dist,
            customjt,
            Unseeded,
          )),
          inner_f64: None,
        }
      }
      _ => {
        let (jt_dist, customjt_dist) = pyo3::Python::attach(|py| {
          let a = jump_times_distribution.clone_ref(py);
          let b = jump_times_distribution;
          (
            crate::traits::CallableDist::<f64>::new(a),
            crate::traits::CallableDist::<f64>::new(b),
          )
        });
        let customjt = CustomJt::new(n, t_max, customjt_dist, Unseeded);
        Self {
          inner_f32: None,
          inner_f64: Some(CompoundCustom::new(
            n,
            t_max,
            crate::traits::CallableDist::new(jumps_distribution),
            jt_dist,
            customjt,
            Unseeded,
          )),
        }
      }
    }
  }

  fn sample<'py>(
    &self,
    py: pyo3::Python<'py>,
  ) -> (
    pyo3::Py<pyo3::PyAny>,
    pyo3::Py<pyo3::PyAny>,
    pyo3::Py<pyo3::PyAny>,
  ) {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    if let Some(ref inner) = self.inner_f64 {
      let [p, cum, j] = inner.sample();
      (
        p.into_pyarray(py).into_py_any(py).unwrap(),
        cum.into_pyarray(py).into_py_any(py).unwrap(),
        j.into_pyarray(py).into_py_any(py).unwrap(),
      )
    } else if let Some(ref inner) = self.inner_f32 {
      let [p, cum, j] = inner.sample();
      (
        p.into_pyarray(py).into_py_any(py).unwrap(),
        cum.into_pyarray(py).into_py_any(py).unwrap(),
        j.into_pyarray(py).into_py_any(py).unwrap(),
      )
    } else {
      unreachable!()
    }
  }
}

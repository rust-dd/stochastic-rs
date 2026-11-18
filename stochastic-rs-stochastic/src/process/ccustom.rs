//! A pure compound jump sum whose inter-arrival times come from a [`CustomJt`] law, not a
//! Poisson clock; it has no diffusion, and is meant to be added to one.
//!
//! $$
//! X_t=\sum_{k=1}^{N_t}J_k
//! $$
//!

use std::any::Any;

use ndarray::Array1;
use ndarray::Axis;
use rand::distr::Distribution;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use super::CompoundPaths;
use super::customjt::CustomJt;
use crate::device::Cpu;
use crate::params::Horizon;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Debug, Clone)]
pub struct CompoundCustom<T, D1, D2, S: SeedExt = Unseeded, B = Cpu>
where
  T: FloatExt,
  D1: Distribution<T> + Send + Sync,
  D2: Distribution<T> + Send + Sync,
{
  /// Distribution of jump magnitudes.
  pub jumps_distribution: D1,
  /// The arrival times, whose horizon and inter-arrival law are the process's: under `Events(n)`
  /// the rows hold the origin and the `n` jumps after it.
  pub customjt: CustomJt<T, D2>,
  /// Seed: [`Unseeded`] or a [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic).
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
  pub fn new(jumps_distribution: D1, customjt: CustomJt<T, D2>, seed: S) -> Self {
    Self {
      backend: Cpu,
      jumps_distribution,
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
  /// Replace the arrivals' horizon, all else unchanged; panics on an `Until` that is not positive
  /// and finite.
  pub fn with_horizon(mut self, horizon: Horizon<T>) -> Self {
    self.customjt = self.customjt.with_horizon(horizon);
    self
  }
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
    crate::process::cpoisson::device_arrival_rate(&self.customjt.distribution)
  }

  /// The jump-size law as the kernels draw it, one size per arrival; `None`
  /// for a distribution they do not carry.
  fn device_jump_sizes(&self) -> Option<crate::euler::JumpSizes<T>> {
    crate::process::cpoisson::device_jump_sizes(&self.jumps_distribution)
      .and_then(crate::euler::JumpSizes::single)
  }
}

impl<T, D1, D2, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 3>
  for CompoundCustom<T, D1, D2, S, B>
where
  T: FloatExt,
  D1: Distribution<T> + Send + Sync + Any,
  D2: Distribution<T> + Send + Sync + Any,
{
  /// Poisson arrivals at the exponential law's rate. [`ProcessExt::sample`] keeps any other law
  /// on the host, so a launch that asks here bypassed that guard.
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
    let Horizon::Events(n) = self.customjt.horizon else {
      panic!("the Euler engine describes CompoundCustom's count mode; horizon mode has no grid");
    };
    n + 1
  }

  fn horizon(&self) -> T {
    T::one()
  }

  /// One size per step under the kernels' law; the count the intensity would
  /// draw is never read, so no intensity is declared and no count is drawn.
  fn jump_sizes(&self) -> Option<crate::euler::JumpSizes<T>> {
    Some(self.device_jump_sizes().expect(
      "CompoundCustom: the jump-size distribution is not a law the Euler engine draws on a \
       device; sample through `ProcessExt`, which keeps it on the host",
    ))
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> CompoundPaths<T> {
    let _ = <Self as crate::euler::EulerSystem<T, 3>>::grid_points(self);
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T, D1, D2, S: SeedExt] CompoundCustom<T, D1, D2, S> { jumps_distribution, customjt, seed } via euler where  T: FloatExt,  D1: Distribution<T> + Send + Sync,  D2: Distribution<T> + Send + Sync);

impl<T, D1, D2, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for CompoundCustom<T, D1, D2, S, B>
where
  T: FloatExt,
  D1: Distribution<T> + Send + Sync + Any,
  D2: Distribution<T> + Send + Sync + Any,
{
}

impl<T, D1, D2, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for CompoundCustom<T, D1, D2, S, B>
where
  T: FloatExt,
  D1: Distribution<T> + Send + Sync + Any,
  D2: Distribution<T> + Send + Sync + Any,
{
  type Output = CompoundPaths<T>;
  type Sampler<'s>
    = CompoundCustomSampler<'s, T, D1, D2, S>
  where
    Self: 's;

  /// Derives `self.seed` into the sampler, never clones it: the mixed next tick hash-scrambles
  /// chunk `i`'s basis against chunk `i+1`'s, where a raw snapshot leaves them one stride apart.
  fn sampler(&self) -> CompoundCustomSampler<'_, T, D1, D2, S> {
    CompoundCustomSampler {
      jumps_distribution: &self.jumps_distribution,
      customjt: &self.customjt,
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine when the arrival count is fixed, the inter-arrivals exponential
  /// and the jump-size law one the kernels draw; anything else stays on the host.
  fn sample(&self) -> CompoundPaths<T> {
    if self.device_ready() {
      crate::euler::EulerBackend::system_sample(&self.backend, self)
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&CompoundPaths<T>) -> R + Sync) -> Vec<R> {
    if self.device_ready() {
      crate::euler::EulerBackend::system_paths_map(&self.backend, self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, f)
    }
  }

  fn sample_par(&self, m: usize) -> Vec<CompoundPaths<T>> {
    if self.device_ready() {
      crate::euler::EulerBackend::system_paths(&self.backend, self, m)
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<CompoundPaths<T>, crate::device::DeviceError> {
    if self.device_ready() {
      crate::euler::EulerBackend::try_system_sample(&self.backend, self)
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<CompoundPaths<T>>, crate::device::DeviceError> {
    if self.device_ready() {
      crate::euler::EulerBackend::try_system_paths(&self.backend, self, m)
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }

  /// What a device runs here: a fixed number of arrivals — the horizon mode has no grid —
  /// exponential inter-arrivals, and sizes under a law the kernels draw.
  fn device_fallback(&self) -> Option<&'static str> {
    let ready = {
      matches!(self.customjt.horizon, Horizon::Events(_))
        && self.device_intensity().is_some()
        && self.device_jump_sizes().is_some()
    };
    (!ready).then_some(
      "the horizon mode's missing grid, or an arrival or size law the kernels do not draw",
    )
  }
}

/// Reusable [`CompoundCustom`] sampling state: it borrows the two laws and owns the seed source.
/// The output length depends on the event count, so the three buffers are reallocated per call.
#[doc(hidden)]
#[derive(Debug)]
pub struct CompoundCustomSampler<'a, T, D1, D2, S: SeedExt>
where
  T: FloatExt,
  D1: Distribution<T> + Send + Sync,
  D2: Distribution<T> + Send + Sync,
{
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
  fn sample_inner(&mut self) -> CompoundPaths<T> {
    let p = self.customjt.sample_impl(&self.seed);
    let mut jumps = Array1::<T>::zeros(p.len());
    let mut rng = self.seed.rng();
    for i in 1..p.len() {
      jumps[i] = self.jumps_distribution.sample(&mut rng);
    }

    let mut cum_jupms = jumps.clone();
    cum_jupms.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);

    CompoundPaths {
      arrivals: p,
      cumulative: cum_jupms,
      jumps,
    }
  }
}

impl<T, D1, D2, S: SeedExt> crate::traits::Sealed for CompoundCustomSampler<'_, T, D1, D2, S>
where
  T: FloatExt,
  D1: Distribution<T> + Send + Sync,
  D2: Distribution<T> + Send + Sync,
{
}

impl<T, D1, D2, S: SeedExt> PathSampler<T> for CompoundCustomSampler<'_, T, D1, D2, S>
where
  T: FloatExt,
  D1: Distribution<T> + Send + Sync,
  D2: Distribution<T> + Send + Sync,
{
  type Output = CompoundPaths<T>;

  fn sample_into(&mut self, out: &mut CompoundPaths<T>) {
    *out = self.sample_inner();
  }

  fn sample(&mut self) -> CompoundPaths<T> {
    self.sample_inner()
  }
}

#[cfg(feature = "python")]
#[doc(hidden)]
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
  ) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| -> pyo3::PyResult<Self> {
      let horizon = crate::python_params::horizon_from_kwargs(n, t_max)?;
      Ok(match dtype.unwrap_or("f64") {
        "f32" => {
          let customjt = CustomJt::new(
            crate::python_params::horizon_f32(horizon),
            crate::traits::CallableDist::new(jump_times_distribution),
            Unseeded,
          );
          Self {
            inner_f32: Some(CompoundCustom::new(
              crate::traits::CallableDist::new(jumps_distribution),
              customjt,
              Unseeded,
            )),
            inner_f64: None,
          }
        }
        _ => {
          let customjt = CustomJt::new(
            horizon,
            crate::traits::CallableDist::new(jump_times_distribution),
            Unseeded,
          );
          Self {
            inner_f32: None,
            inner_f64: Some(CompoundCustom::new(
              crate::traits::CallableDist::new(jumps_distribution),
              customjt,
              Unseeded,
            )),
          }
        }
      })
    })?
  }

  fn sample(&self, py: pyo3::Python<'_>) -> pyo3::PyResult<super::PyTriple> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use crate::traits::ProcessExt;
      if let Some(ref inner) = self.inner_f64 {
        inner.sample().into_py_tuple(py)
      } else if let Some(ref inner) = self.inner_f32 {
        inner.sample().into_py_tuple(py)
      } else {
        unreachable!()
      }
    })
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_distributions::exp::SimdExp;
  use stochastic_rs_distributions::normal::SimdNormal;

  use super::*;

  #[test]
  fn the_output_names_its_slots() {
    let CompoundPaths {
      arrivals,
      cumulative,
      jumps,
      ..
    } = CompoundCustom::new(
      SimdNormal::new(0.0, 0.1),
      CustomJt::new(Horizon::Events(8), SimdExp::new(2.0), Unseeded),
      Deterministic::new(3),
    )
    .sample();
    assert_eq!((arrivals.len(), cumulative.len(), jumps.len()), (9, 9, 9));
  }

  #[test]
  fn with_horizon_moves_the_arrivals_too() {
    let p = CompoundCustom::new(
      SimdNormal::new(0.0, 0.1),
      CustomJt::new(Horizon::Events(8), SimdExp::new(2.0), Unseeded),
      Deterministic::new(3),
    )
    .with_horizon(Horizon::Events(4));
    let CompoundPaths {
      arrivals,
      cumulative,
      jumps,
      ..
    } = p.sample();
    assert_eq!((arrivals.len(), cumulative.len(), jumps.len()), (5, 5, 5));
  }
}

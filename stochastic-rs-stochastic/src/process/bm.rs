//! # Bm
//!
//! $$
//! B_t=\int_0^t dW_s,\quad B_t-B_s\sim\mathcal N(0,t-s)
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

#[derive(Clone)]
pub struct Bm<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Number of discrete time points in the generated path.
  pub n: usize,
  /// Simulation horizon [0, t] for the path (defaults to `1` if `None`).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

/// Every field has a matching `with_*` builder setter, e.g.
/// `Bm::default().with_steps(500)`. No persisted cache: `sampler()`
/// builds its Gaussian source fresh from `self` every call.
impl<T: FloatExt, S: SeedExt> Bm<T, S> {
  pub fn new(n: usize, t: Option<T>, seed: S) -> Self {
    Self {
      backend: Cpu,
      n,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Bm<T, S, B> {
  /// Replace the number of simulation steps `n`, all else unchanged.
  pub fn with_steps(mut self, n: usize) -> Self {
    self.n = n;
    self
  }

  /// Replace the simulation horizon `t`, all else unchanged.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    self.t = t;
    self
  }

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

/// t=1, n=252 — one trading year of daily steps (this crate's `Default`
/// convention).
impl<T: FloatExt> Default for Bm<T, Unseeded> {
  fn default() -> Self {
    Self::new(252, Some(T::one()), Unseeded)
  }
}

/// The Euler engine's view of Brownian motion: the additive family, whose
/// step is the increment itself, so a device accumulates the increments in
/// the same kernel that draws them.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for Bm<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::Additive
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

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Bm<T, S> { n, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Bm<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = BmSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> BmSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let std_dev = (self.t.unwrap_or(T::one()) / T::from_usize_(n_increments)).sqrt();
    BmSampler {
      n: self.n,
      normal: SimdNormal::<T>::new(T::zero(), std_dev, &self.seed),
    }
  }

  /// Through the Euler engine: on a device the recursion runs in the kernel,
  /// on the host devices it is this process's own sampler, chunked exactly as
  /// `ProcessExt` chunks.
  fn sample(&self) -> Array1<T> {
    self.backend.euler_sample(self)
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    self.backend.euler_paths_map(self, m, f)
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

/// Reusable [`Bm`] sampling state: the owned Gaussian increment source. The
/// path is `B_0 = 0` followed by the running sum of the increments.
#[doc(hidden)]
pub struct BmSampler<T: FloatExt> {
  n: usize,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> BmSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.len() <= 1 {
      if let Some(first) = out.first_mut() {
        *first = T::zero();
      }
      return;
    }
    out[0] = T::zero();
    let tail = &mut out[1..];
    self.normal.fill_slice(tail);
    let mut acc = T::zero();
    for x in tail.iter_mut() {
      acc += *x;
      *x = acc;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for BmSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("Bm output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyBm, Bm,
  sig: (n, t=None, seed=None, dtype=None),
  params: (n: usize, t: Option<f64>),
  device
);

#[cfg(test)]
mod tests {
  use std::time::Instant;

  use super::*;

  #[test]
  fn test_bm() {
    let start = Instant::now();
    let bm = Bm::new(10000, Some(1.0), Unseeded);
    for _ in 0..10000 {
      let m = bm.sample();
      assert_eq!(m.len(), 10000);
    }
    println!("Time elapsed: {:?} ms", start.elapsed().as_millis());

    let start = Instant::now();
    let bm = Bm::new(10000, Some(1.0), Unseeded);
    for _ in 0..10000 {
      let m = bm.sample();
      assert_eq!(m.len(), 10000);
    }
    println!("Time elapsed: {:?} ms", start.elapsed().as_millis());
  }

  #[test]
  fn test_bm_movement_1000_iterations() {
    let bm = Bm::new(1000, Some(1.0), Unseeded);

    let mut max_abs_value: f64 = 0.0;
    let mut min_abs_value: f64 = f64::MAX;
    let mut last_value_sum: f64 = 0.0;

    for _ in 0..1000 {
      let path = bm.sample();
      assert_eq!(path.len(), 1000);

      let last_value = path[999];
      last_value_sum += last_value;

      let abs_last = last_value.abs();
      max_abs_value = max_abs_value.max(abs_last);
      min_abs_value = min_abs_value.min(abs_last);
    }

    let avg_last_value = last_value_sum / 1000.0;
    println!("Bm Movement Test (1000 iterations):");
    println!("  Average last value: {}", avg_last_value);
    println!("  Maximum absolute last value: {}", max_abs_value);
    println!("  Minimum absolute last value: {}", min_abs_value);

    assert!(max_abs_value > 0.0, "Bm should have non-zero movement");
    assert!(
      avg_last_value.abs() < 2.0,
      "Average position should stay relatively close to zero"
    );
  }
}

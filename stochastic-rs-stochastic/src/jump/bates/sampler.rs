use super::*;

impl<T, D, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed for Bates<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
}

impl<T, D, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Bates<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = BatesSampler<'s, T, D, S>
  where
    Self: 's;

  /// Chunk-local children of the diffusion's and the jumps' seeds; the shared ones would let
  /// concurrent chunks race, and `cgns`'s own `Unseeded` seed is never read.
  fn sampler(&self) -> BatesSampler<'_, T, D, S> {
    BatesSampler {
      n: self.n,
      s0: self.s0,
      v0: self.v0.max(T::zero()),
      lambda: self.lambda,
      alpha: self.alpha,
      beta: self.beta,
      sigma: self.sigma,
      drift_c: self.drift_c(),
      boundary: self.boundary,
      dt: self.cgns.dt(),
      cgns: self.cgns,
      jump_distribution: &self.jumps.law,
      log_jumps: self.log_jumps(),
      jump_seed: self.jumps.seed.derive(),
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine when the jump-size law is one the device
  /// kernels draw; any other law keeps the process on the host, chunked
  /// exactly as [`ProcessExt`] chunks.
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

  /// What a device runs here: no jumps, or sizes under a law the kernels draw at no more than 25
  /// expected a step. Anything else samples on the host.
  fn device_fallback(&self) -> Option<&'static str> {
    (!(self.lambda <= T::zero() || self.device_jump_sizes().is_some()))
      .then_some("a jump-size law the kernels do not draw")
      .or_else(|| {
        crate::euler::count_fallback(
          <Self as crate::euler::EulerSystem<T, 2>>::jump_intensity(self),
          <Self as crate::euler::EulerSystem<T, 2>>::time_step(self),
        )
      })
  }
}

/// Reusable [`Bates`] sampling state: the correlated-Gaussian generator, the borrowed jump law and
/// two derived seeds, one for the diffusion and one for the jump arrivals.
#[doc(hidden)]
#[derive(Debug)]
pub struct BatesSampler<'a, T, D, S: SeedExt>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  n: usize,
  s0: T,
  v0: T,
  lambda: T,
  alpha: T,
  beta: T,
  sigma: T,
  drift_c: T,
  boundary: VarianceBoundary,
  dt: T,
  cgns: Cgns<T>,
  jump_distribution: &'a D,
  log_jumps: Option<&'a LogNormalJump<T>>,
  jump_seed: S,
  seed: S,
}

impl<T, D, S: SeedExt> BatesSampler<'_, T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  fn fill_paths(&mut self, s: &mut [T], v: &mut [T]) {
    if self.n == 0 {
      return;
    }
    let dt = self.dt;
    let [cgn1, cgn2] = &self.cgns.sample_impl(&self.seed);
    s[0] = self.s0;
    v[0] = self.v0;
    match self.log_jumps {
      Some(law) => {
        let mut jumps = law.jump_sums(self.lambda, &self.jump_seed, self.n, dt);
        for i in 1..self.n {
          let v_prev = v[i - 1].max(T::zero());
          let half_v = T::from_f64_fast(0.5) * v_prev;
          let log_step =
            (self.drift_c - half_v) * dt + v_prev.sqrt() * cgn1[i - 1] + jumps.next_sum();
          s[i] = s[i - 1] * log_step.exp();
          v[i] = self.variance_step(v[i - 1], v_prev, cgn2[i - 1]);
        }
      }
      None => {
        let jumps = crate::process::cpoisson::grid_relative_increments(
          self.jump_distribution,
          self.lambda,
          &self.jump_seed,
          self.n,
          dt,
        );
        for i in 1..self.n {
          let v_prev = v[i - 1].max(T::zero());
          s[i] = s[i - 1]
            + self.drift_c * s[i - 1] * dt
            + s[i - 1] * v_prev.sqrt() * cgn1[i - 1]
            + s[i - 1] * jumps[i];
          v[i] = self.variance_step(v[i - 1], v_prev, cgn2[i - 1]);
        }
      }
    }
  }

  /// The variance one step on from `v`, whose floored value is `v_prev`, under the noise `z`.
  #[inline]
  fn variance_step(&self, v: T, v_prev: T, z: T) -> T {
    let dv = (self.alpha - self.beta * v_prev) * self.dt + self.sigma * v_prev.sqrt() * z;
    match self.boundary {
      VarianceBoundary::Floor => (v + dv).max(T::zero()),
      VarianceBoundary::Reflect => (v + dv).abs(),
    }
  }
}

impl<T, D, S: SeedExt> crate::traits::Sealed for BatesSampler<'_, T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
}

impl<T, D, S: SeedExt> PathSampler<T> for BatesSampler<'_, T, D, S>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [s, v] = out;
    self.fill_paths(
      s.as_slice_mut().expect("Bates output must be contiguous"),
      v.as_slice_mut().expect("Bates output must be contiguous"),
    );
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    let mut s = Array1::<T>::zeros(self.n);
    let mut v = Array1::<T>::zeros(self.n);
    self.fill_paths(
      s.as_slice_mut().expect("contiguous"),
      v.as_slice_mut().expect("contiguous"),
    );
    [s, v]
  }
}

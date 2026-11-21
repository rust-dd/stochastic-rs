use super::*;

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Wishart<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Wishart<T, S, B> {
  type Output = Array3<T>;
  type Sampler<'s>
    = WishartSampler<T, S>
  where
    Self: 's;

  fn sampler(&self) -> WishartSampler<T, S> {
    WishartSampler {
      process: Wishart {
        backend: Cpu,
        alpha: self.alpha,
        b: self.b.clone(),
        a: self.a.clone(),
        x0: self.x0.clone(),
        n: self.n,
        t: self.t,
        seed: self.seed.derive(),
        step: self.step.clone(),
      },
    }
  }

  /// Through the Euler engine in two dimensions at a degree of at least three; anything else
  /// stays on the host, chunked as [`ProcessExt`] chunks.
  fn sample(&self) -> Array3<T> {
    if self.device_ready() {
      slots_to_matrices(self.backend.system_sample(&WishartLaunch(self)))
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array3<T>) -> R + Sync) -> Vec<R> {
    if self.device_ready() {
      self
        .backend
        .system_paths_map(&WishartLaunch(self), m, |slots| {
          f(&slots_to_matrices(slots.clone()))
        })
    } else {
      crate::traits::process::sample_map_chunked(self, m, f)
    }
  }

  fn sample_par(&self, m: usize) -> Vec<Array3<T>> {
    if self.device_ready() {
      self
        .backend
        .system_paths(&WishartLaunch(self), m)
        .into_iter()
        .map(slots_to_matrices)
        .collect()
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<Array3<T>, crate::device::DeviceError> {
    if self.device_ready() {
      Ok(slots_to_matrices(
        self.backend.try_system_sample(&WishartLaunch(self))?,
      ))
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array3<T>>, crate::device::DeviceError> {
    if self.device_ready() {
      Ok(
        self
          .backend
          .try_system_paths(&WishartLaunch(self), m)?
          .into_iter()
          .map(slots_to_matrices)
          .collect(),
      )
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }

  /// What a device runs: `d = 2`, a noise of rank one or two, and a degree of at least three,
  /// which keeps the path inside the cone and the squared Bessel draw at the kernels' χ² degree.
  fn device_fallback(&self) -> Option<&'static str> {
    (!(self.dim() == 2 && self.step.rank >= 1 && self.alpha >= T::from_usize_(3)))
      .then_some("a Wishart outside the two-dimensional cone the kernel steps")
  }
}

/// Reusable [`Wishart`] sampling state with a derived seed, so parallel
/// chunks draw independent paths.
#[doc(hidden)]
#[derive(Debug)]
pub struct WishartSampler<T: FloatExt, S: SeedExt> {
  process: Wishart<T, S>,
}

impl<T: FloatExt, S: SeedExt> crate::traits::Sealed for WishartSampler<T, S> {}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for WishartSampler<T, S> {
  type Output = Array3<T>;

  fn sample_into(&mut self, out: &mut Array3<T>) {
    // The derived seed travels by reference: every constructor call advances
    // it, so consecutive paths of one chunk are independent.
    self.process.fill_path(&self.process.seed, out);
  }

  fn sample(&mut self) -> Array3<T> {
    let d = self.process.dim();
    let mut out = Array3::<T>::zeros((self.process.n, d, d));
    self.sample_into(&mut out);
    out
  }
}

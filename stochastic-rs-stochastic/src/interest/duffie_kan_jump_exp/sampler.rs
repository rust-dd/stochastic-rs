use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::exp::SimdExp;
use stochastic_rs_distributions::normal::SimdNormal;

use super::*;

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for DuffieKanJumpExp<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for DuffieKanJumpExp<T, S, B>
{
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = DuffieKanJumpExpSampler<T, S>
  where
    Self: 's;

  /// Derives `self.seed` into the sampler, never clones it: the mixed next tick hash-scrambles
  /// chunk `i`'s basis against chunk `i+1`'s, where a raw snapshot leaves them one stride apart.
  fn sampler(&self) -> DuffieKanJumpExpSampler<T, S> {
    DuffieKanJumpExpSampler {
      n: self.n,
      r0: self.r0.unwrap_or(T::zero()),
      x0: self.x0.unwrap_or(T::zero()),
      alpha: self.alpha,
      beta: self.beta,
      gamma: self.gamma,
      a1: self.a1,
      b1: self.b1,
      c1: self.c1,
      sigma1: self.sigma1,
      a2: self.a2,
      b2: self.b2,
      c2: self.c2,
      sigma2: self.sigma2,
      lambda: self.lambda,
      jump_scale: self.jump_scale,
      dt: self.cgns.dt(),
      cgns: self.cgns,
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine: a device steps both factors and the jump sum in its kernel, the
  /// host runs this process's own sampler, chunked as `ProcessExt` chunks.
  fn sample(&self) -> [Array1<T>; 2] {
    if self.device_ready() {
      self.backend.system_sample(self)
    } else {
      <Self as crate::euler::EulerSystem<T, 2>>::host_sample(self)
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

  /// A step expecting more than 25 jumps samples on the host, past the kernels' 64-event count.
  fn device_fallback(&self) -> Option<&'static str> {
    crate::euler::count_fallback(
      <Self as crate::euler::EulerSystem<T, 2>>::jump_intensity(self),
      <Self as crate::euler::EulerSystem<T, 2>>::time_step(self),
    )
  }
}

/// Reusable [`DuffieKanJumpExp`] sampling state: the correlated-Gaussian generator and the seed
/// source. Each call rebuilds the inter-arrival and jump-size drivers from the owned seed.
#[doc(hidden)]
#[derive(Debug)]
pub struct DuffieKanJumpExpSampler<T: FloatExt, S: SeedExt> {
  n: usize,
  r0: T,
  x0: T,
  alpha: T,
  beta: T,
  gamma: T,
  a1: T,
  b1: T,
  c1: T,
  sigma1: T,
  a2: T,
  b2: T,
  c2: T,
  sigma2: T,
  lambda: T,
  jump_scale: T,
  dt: T,
  cgns: Cgns<T>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> DuffieKanJumpExpSampler<T, S> {
  fn fill_paths(&mut self, r: &mut [T], x: &mut [T]) {
    if self.n == 0 {
      return;
    }
    let dt = self.dt;
    let [cgn1, cgn2] = &self.cgns.sample_impl(&self.seed);

    r[0] = self.r0;
    x[0] = self.x0;

    let mut exp_dist = SimdExp::<T>::new(self.lambda).seeded(&self.seed);
    let mut jump_dist = SimdNormal::<T>::new(T::zero(), self.jump_scale).seeded(&self.seed);

    // A draw no engine reads; skipping it would move every stream seeded after this one.
    self.seed.next_seed();
    let mut next_jump_time = exp_dist.sample();

    for i in 1..self.n {
      let current_time = T::from_usize_(i) * dt;
      let r_old = r[i - 1];
      let x_old = x[i - 1];

      let dr = (self.a1 * r_old + self.b1 * x_old + self.c1) * dt
        + self.sigma1 * (self.alpha * r_old + self.beta * x_old + self.gamma) * cgn1[i - 1];
      let dx = (self.a2 * r_old + self.b2 * x_old + self.c2) * dt
        + self.sigma2 * (self.alpha * r_old + self.beta * x_old + self.gamma) * cgn2[i - 1];

      let mut jump_sum_x = T::zero();
      while next_jump_time <= current_time {
        let jump_x = jump_dist.sample();
        jump_sum_x += jump_x;
        next_jump_time += exp_dist.sample();
      }

      r[i] = r_old + dr;
      x[i] = x_old + dx + jump_sum_x;
    }
  }
}

impl<T: FloatExt, S: SeedExt> crate::traits::Sealed for DuffieKanJumpExpSampler<T, S> {}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for DuffieKanJumpExpSampler<T, S> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [r_arr, x_arr] = out;
    let r = r_arr
      .as_slice_mut()
      .expect("DuffieKanJumpExp output must be contiguous");
    let x = x_arr
      .as_slice_mut()
      .expect("DuffieKanJumpExp output must be contiguous");
    self.fill_paths(r, x);
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    let mut r = Array1::<T>::zeros(self.n);
    let mut x = Array1::<T>::zeros(self.n);
    self.fill_paths(
      r.as_slice_mut().expect("contiguous"),
      x.as_slice_mut().expect("contiguous"),
    );
    [r, x]
  }
}

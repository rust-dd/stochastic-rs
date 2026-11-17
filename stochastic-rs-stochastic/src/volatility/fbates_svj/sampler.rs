use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::poisson::SimdPoisson;

use super::*;
use crate::noise::cgns::Cgns;

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for FbatesSvj<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for FbatesSvj<T, S, B>
{
  type Output = [Array1<T>; 2]; // [S, v]
  type Sampler<'s>
    = FbatesSvjSampler<T, S>
  where
    Self: 's;

  /// Derives (not clones) `self.seed` into the returned sampler: the
  /// derived value is `self.seed`'s *mixed* next tick, not a raw snapshot,
  /// so chunk `i`'s basis and chunk `i+1`'s basis are hash-scrambled
  /// relative to each other rather than one raw stride apart.
  fn sampler(&self) -> FbatesSvjSampler<T, S> {
    let n_steps = self.n.saturating_sub(1);
    let dt = if n_steps > 0 {
      self.t.unwrap_or(T::one()) / T::from_usize_(n_steps)
    } else {
      T::zero()
    };
    FbatesSvjSampler {
      n: self.n,
      hurst: self.hurst,
      mu: self.mu,
      s0: self.s0,
      v0: self.v0,
      theta: self.theta,
      kappa: self.kappa,
      xi: self.xi,
      lambda: self.lambda,
      nu: self.nu,
      omega: self.omega,
      dt,
      g: T::from_f64_fast(gamma(self.hurst.to_f64().unwrap() - 0.5)),
      // `Unseeded` baked into the Cgns exactly as the legacy `sample` did; the
      // noise is drawn via `sample_impl(&self.seed)`, so the Cgns's own seed
      // is irrelevant.
      cgns: Cgns::new(self.rho, n_steps, self.t, Unseeded),
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine when the grid fits the kernels'
  /// per-path history; a longer grid keeps the process on the host,
  /// chunked exactly as [`ProcessExt`] chunks.
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

  /// What a device runs here: a grid within the kernels' per-path history, and no more than 25
  /// expected jumps a step. Anything else samples on the host.
  fn device_fallback(&self) -> Option<&'static str> {
    (!(self.n <= crate::euler::HISTORY_SLOTS))
      .then_some("a grid longer than the kernels' per-path history")
      .or_else(|| {
        crate::euler::count_fallback(
          <Self as crate::euler::EulerSystem<T, 2>>::jump_intensity(self),
          <Self as crate::euler::EulerSystem<T, 2>>::time_step(self),
        )
      })
  }
}

/// Reusable [`FbatesSvj`] sampling state: owns the correlated-Gaussian generator
/// and the seed source. The jump (Poisson + Normal) generators are rebuilt per
/// fill in the legacy seed-consumption order, so the first call reproduces the
/// original stream bit-for-bit.
#[doc(hidden)]
#[derive(Debug)]
pub struct FbatesSvjSampler<T: FloatExt, S: SeedExt> {
  n: usize,
  hurst: T,
  mu: T,
  s0: T,
  v0: T,
  theta: T,
  kappa: T,
  xi: T,
  lambda: T,
  nu: T,
  omega: T,
  dt: T,
  g: T,
  cgns: Cgns<T>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> FbatesSvjSampler<T, S> {
  fn fill_paths(&mut self, s: &mut [T], v2: &mut [T]) {
    if self.n == 0 {
      return;
    }
    let dt = self.dt;

    // Use Cgns for rho-correlated noise: [gn_vol, gn_price]
    let [gn_vol, gn_price] = self.cgns.sample_impl(&self.seed);

    let mut yt = Array1::<T>::zeros(self.n);
    let mut zt = Array1::<T>::zeros(self.n);
    let mut sigma_tilde2 = Array1::<T>::zeros(self.n);

    yt[0] = self.v0;
    zt[0] = T::zero();
    sigma_tilde2[0] = self.v0;
    v2[0] = self.v0;
    s[0] = self.s0;

    let g = self.g;
    let half = T::from_f64_fast(0.5);

    // Jump compensation: κ_J = exp(ν + ½ω²) - 1
    let kappa_j = (self.nu + half * self.omega * self.omega).exp() - T::one();

    // Jump RNG
    let mut z_std = SimdNormal::<T>::new(T::zero(), T::one()).seeded(&self.seed);
    // One seed is skipped: the pinned streams that follow take theirs after it.
    self.seed.next_seed();
    let lambda_dt = self.lambda.to_f64().unwrap() * dt.to_f64().unwrap();
    let mut pois = if lambda_dt > 0.0 {
      Some(SimdPoisson::<u32>::new(lambda_dt).seeded(&self.seed))
    } else {
      None
    };

    for i in 1..self.n {
      let t_i = dt * T::from_usize_(i);

      // Rough variance dynamics (same as Fheston/fheston.rs)
      yt[i] = self.theta + (yt[i - 1] - self.theta) * (-self.kappa * dt).exp();
      zt[i] = zt[i - 1] * (-self.kappa * dt).exp()
        + sigma_tilde2[i - 1].max(T::zero()).sqrt() * gn_vol[i - 1];

      sigma_tilde2[i] = yt[i] + self.xi * zt[i];

      let integral = (0..i)
        .map(|j| {
          let tj = T::from_usize_(j) * dt;
          ((t_i - tj).powf(self.hurst - half) * zt[j]) * dt
        })
        .sum::<T>();

      v2[i] = yt[i] + self.xi * zt[i] + self.xi * integral / g;

      // Price dynamics with jumps
      let vi = v2[i - 1].max(T::zero());

      // Jump component
      let mut jump_sum = T::zero();
      if let Some(pois) = &mut pois {
        let n_jumps = pois.sample();
        if n_jumps > 0 {
          let kf = T::from_f64_fast(n_jumps as f64);
          let z0 = z_std.sample();
          jump_sum = self.nu * kf + self.omega * kf.sqrt() * z0;
        }
      }

      let log_inc =
        (self.mu - self.lambda * kappa_j - half * vi) * dt + vi.sqrt() * gn_price[i - 1] + jump_sum;
      s[i] = s[i - 1] * log_inc.exp();
    }
  }
}

impl<T: FloatExt, S: SeedExt> crate::traits::Sealed for FbatesSvjSampler<T, S> {}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for FbatesSvjSampler<T, S> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [s, v2] = out;
    self.fill_paths(
      s.as_slice_mut()
        .expect("FbatesSvj output must be contiguous"),
      v2.as_slice_mut()
        .expect("FbatesSvj output must be contiguous"),
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

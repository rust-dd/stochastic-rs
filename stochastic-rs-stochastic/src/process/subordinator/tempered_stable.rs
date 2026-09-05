use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::poisson::SimdPoisson;
use stochastic_rs_distributions::uniform::SimdUniform;

use super::clamp_open01;
use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Classical tempered-stable subordinator approximation.
///
/// Uses truncated-stable large jumps with exponential thinning and
/// deterministic small-jump drift:
/// `nu(dx) = c * exp(-mu x) * x^{-1-alpha} dx`, `x > 0`, `alpha in (0,1)`.
pub struct TemperedStableSubordinator<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Stable index in `(0,1)`.
  pub alpha: T,
  /// Levy density scale.
  pub c: T,
  /// Tempering rate.
  pub mu: T,
  /// Truncation threshold for the large-jump approximation.
  pub epsilon: T,
  /// Number of grid points.
  pub n: usize,
  /// Initial level.
  pub x0: Option<T>,
  /// Horizon.
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> TemperedStableSubordinator<T, S> {
  pub fn new(
    alpha: T,
    c: T,
    mu: T,
    epsilon: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert!(
      alpha > T::zero() && alpha < T::one(),
      "alpha must be in (0,1)"
    );
    assert!(c > T::zero(), "c must be positive");
    assert!(mu > T::zero(), "mu must be positive");
    assert!(epsilon > T::zero(), "epsilon must be positive");
    Self {
      backend: Cpu,
      alpha,
      c,
      mu,
      epsilon,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> TemperedStableSubordinator<T, S, B> {}

/// The Euler engine's view of the tempered-stable subordinator. Every jump
/// below the truncation contributes a deterministic drift, which is folded
/// here; above it, the step reads the kernel's own sum over the candidates it
/// kept.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for TemperedStableSubordinator<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let one = T::one();
    let dt = self.time_step();
    crate::euler::EulerSpec::TemperedStableSubordinator {
      drift: dt * self.c * self.epsilon.powf(one - self.alpha) / (one - self.alpha),
    }
  }

  fn initial_value(&self) -> T {
    self.x0.unwrap_or(T::zero())
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  /// The candidate rate above the truncation.
  fn jump_intensity(&self) -> Option<T> {
    Some(self.c / self.alpha * self.epsilon.powf(-self.alpha))
  }

  /// A candidate `ε u^{−1/α}`, kept with probability `exp(−μ·candidate)`.
  fn jump_sizes(&self) -> Option<crate::euler::JumpSizes<T>> {
    Some(crate::euler::JumpSizes::TemperedStable {
      eps: self.epsilon,
      neg_inv_alpha: -T::one() / self.alpha,
      mu: self.mu,
    })
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

backend_switch!([T: FloatExt, S: SeedExt] TemperedStableSubordinator<T, S> { alpha, c, mu, epsilon, n, x0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for TemperedStableSubordinator<T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = TemperedStableSubordinatorSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> TemperedStableSubordinatorSampler<T> {
    let x0 = self.x0.unwrap_or(T::zero());
    let alpha = self.alpha.to_f64().unwrap();
    let c = self.c.to_f64().unwrap();
    let mu = self.mu.to_f64().unwrap();
    let eps = self.epsilon.to_f64().unwrap();
    let n_increments = self.n.saturating_sub(1).max(1);
    let t_max = self.t.unwrap_or(T::one()).to_f64().unwrap();
    let dt = t_max / n_increments as f64;

    let lambda0 = (c / alpha) * eps.powf(-alpha);
    let small_jump_drift = dt * c * eps.powf(1.0 - alpha) / (1.0 - alpha);

    TemperedStableSubordinatorSampler {
      n: self.n,
      x0,
      alpha,
      mu,
      eps,
      small_jump_drift,
      poisson: SimdPoisson::<u32>::new(lambda0 * dt, &self.seed),
      uniform: SimdUniform::<f64>::new(0.0, 1.0, &self.seed),
    }
  }

  /// Through the Euler engine: on a device the candidates are drawn and
  /// thinned in the kernel, on the host devices it is this process's own
  /// sampler, chunked exactly as `ProcessExt` chunks.
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

/// Reusable [`TemperedStableSubordinator`] sampling state: the owned Poisson
/// driver for candidate counts, the uniform source for jump sizes / thinning,
/// and precomputed scalars.
#[doc(hidden)]
pub struct TemperedStableSubordinatorSampler<T: FloatExt> {
  n: usize,
  x0: T,
  alpha: f64,
  mu: f64,
  eps: f64,
  small_jump_drift: f64,
  poisson: SimdPoisson<u32>,
  uniform: SimdUniform<f64>,
}

impl<T: FloatExt> TemperedStableSubordinatorSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }
    let mut level = self.x0.to_f64().unwrap();
    for x in out[1..].iter_mut() {
      let n_candidates = self.poisson.sample_fast() as usize;
      let mut jump_sum = 0.0f64;
      for _ in 0..n_candidates {
        let u = clamp_open01(self.uniform.sample_fast());
        let xj = self.eps * u.powf(-1.0 / self.alpha);
        let accept = self.uniform.sample_fast() <= (-self.mu * xj).exp();
        if accept {
          jump_sum += xj;
        }
      }
      level += self.small_jump_drift + jump_sum;
      *x = T::from_f64_fast(level);
    }
  }
}

impl<T: FloatExt> PathSampler<T> for TemperedStableSubordinatorSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("TemperedStableSubordinator output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyTemperedStableSubordinator, TemperedStableSubordinator,
  sig: (alpha, c, mu, epsilon, n, x0=None, t=None, seed=None, dtype=None),
  params: (alpha: f64, c: f64, mu: f64, epsilon: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);

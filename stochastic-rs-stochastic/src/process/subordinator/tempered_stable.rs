use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::poisson::SimdPoisson;
use stochastic_rs_distributions::uniform::SimdUniform;

use super::clamp_open01;
use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Tempered-stable subordinator with Lévy density `nu(dx) = c exp(-mu x) x^{-1-alpha} dx`,
/// `alpha in (0,1)`: truncated-stable large jumps thinned exponentially, small jumps as a drift.
#[derive(Debug, Clone)]
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
  /// Initial value; `None` starts at zero.
  pub x0: Option<T>,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Seed: [`Unseeded`] or a [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> TemperedStableSubordinator<T, S> {
  /// Panics unless `n >= 2`, a given `t` is positive and finite, `0 < alpha < 1`, `c > 0`,
  /// `mu > 0`, `epsilon > 0`, and every parameter is finite.
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
      alpha.is_finite(),
      "alpha must satisfy `alpha.is_finite()`, got alpha = {alpha:?}"
    );
    assert!(
      c.is_finite(),
      "c must satisfy `c.is_finite()`, got c = {c:?}"
    );
    assert!(
      mu.is_finite(),
      "mu must satisfy `mu.is_finite()`, got mu = {mu:?}"
    );
    assert!(
      epsilon.is_finite(),
      "epsilon must satisfy `epsilon.is_finite()`, got epsilon = {epsilon:?}"
    );
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    assert!(
      x0.is_none_or(|v| v.is_finite()),
      "x0 must satisfy `x0.is_none_or(|v| v.is_finite())`, got x0 = {x0:?}"
    );
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    assert!(
      alpha > T::zero() && alpha < T::one(),
      "alpha must satisfy `alpha > T::zero() && alpha < T::one()`, got alpha = {alpha:?}"
    );
    assert!(
      c > T::zero(),
      "c must satisfy `c > T::zero()`, got c = {c:?}"
    );
    assert!(
      mu > T::zero(),
      "mu must satisfy `mu > T::zero()`, got mu = {mu:?}"
    );
    assert!(
      epsilon > T::zero(),
      "epsilon must satisfy `epsilon > T::zero()`, got epsilon = {epsilon:?}"
    );
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

/// The Euler engine's view. Every jump below the truncation adds a deterministic drift, folded
/// here; above it, the step reads the kernel's sum over the candidates it kept.
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

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for TemperedStableSubordinator<T, S, B>
{
}

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
      poisson: SimdPoisson::<u32>::new(lambda0 * dt).seeded(&self.seed),
      uniform: SimdUniform::<f64>::new(0.0, 1.0).seeded(&self.seed),
    }
  }

  /// Through the Euler engine: a device draws and thins the candidates in its kernel, the host
  /// runs this process's own sampler, chunked as `ProcessExt` chunks.
  fn sample(&self) -> Array1<T> {
    if self.device_ready() {
      self.backend.euler_sample(self)
    } else {
      <Self as crate::euler::EulerCoefficients<T>>::host_sample(self)
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    if self.device_ready() {
      self.backend.euler_paths_map(self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, f)
    }
  }

  fn sample_map_view<R: Send>(
    &self,
    m: usize,
    f: impl Fn(ndarray::ArrayView1<T>) -> R + Sync,
  ) -> Vec<R> {
    if self.device_ready() {
      self.backend.euler_paths_map_view(self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, |path| f(path.view()))
    }
  }

  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    if self.device_ready() {
      self.backend.euler_paths(self, m)
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<Array1<T>, crate::device::DeviceError> {
    if self.device_ready() {
      self.backend.try_sample(self)
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, crate::device::DeviceError> {
    if self.device_ready() {
      self.backend.try_euler_paths(self, m)
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }

  /// A step expecting more than 25 jumps samples on the host, past the kernels' 64-event count.
  fn device_fallback(&self) -> Option<&'static str> {
    crate::euler::count_fallback(
      <Self as crate::euler::EulerCoefficients<T>>::jump_intensity(self),
      <Self as crate::euler::EulerCoefficients<T>>::time_step(self),
    )
  }
}

/// Reusable [`TemperedStableSubordinator`] sampling state: the Poisson driver of the candidate
/// counts, the uniform source of the sizes and the thinning, and the precomputed scalars.
#[doc(hidden)]
#[derive(Debug)]
pub struct TemperedStableSubordinatorSampler<T: FloatExt> {
  n: usize,
  x0: T,
  alpha: f64,
  mu: f64,
  eps: f64,
  small_jump_drift: f64,
  poisson: Seeded<SimdPoisson<u32>>,
  uniform: Seeded<SimdUniform<f64>>,
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
      let n_candidates = self.poisson.sample() as usize;
      let mut jump_sum = 0.0f64;
      for _ in 0..n_candidates {
        let u = clamp_open01(self.uniform.sample());
        let xj = self.eps * u.powf(-1.0 / self.alpha);
        let accept = self.uniform.sample() <= (-self.mu * xj).exp();
        if accept {
          jump_sum += xj;
        }
      }
      level += self.small_jump_drift + jump_sum;
      *x = T::from_f64_fast(level);
    }
  }
}

impl<T: FloatExt> crate::traits::Sealed for TemperedStableSubordinatorSampler<T> {}

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
  params: (alpha: f64, c: f64, mu: f64, epsilon: f64, n: usize, x0: Option<f64>, t: Option<f64>),
  device
);

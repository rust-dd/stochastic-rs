//! $$
//! \ln\!\bigl(\tfrac{S_{t+dt}}{S_t}\bigr) = (\mu - \lambda\kappa_J - \tfrac12\sigma^2)\,dt
//!   + \sigma\sqrt{dt}\,Z_1 + \sum_{j=1}^{K} Z_j
//!
//! $$
//!
//! where $K\sim\mathrm{Poisson}(\lambda\,dt)$, $Z_j\sim\mathcal{N}(\nu,\omega^2)$ and
//! $\kappa_J = e^{\nu+\frac12\omega^2}-1$. The log-spot scheme guarantees $S_t > 0$.
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::poisson::SimdPoisson;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::params::Drift;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Debug, Clone)]
pub struct MjdLog<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// The price's drift: `μ`, a cost of carry, or the rate pair.
  pub drift: Drift<T>,
  /// Diffusion volatility
  pub sigma: T,
  /// Jump intensity (Poisson arrival rate)
  pub lambda: T,
  /// Mean of jump log-size Z ~ N(nu, omega^2)
  pub nu: T,
  /// Standard deviation of jump log-size Z
  pub omega: T,
  /// Number of discrete time steps
  pub n: usize,
  /// Initial spot.
  pub s0: T,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Seed strategy (compile-time: `Unseeded` or `Deterministic`).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> MjdLog<T, S> {
  /// Panics unless a given `t` is positive and finite, `n >= 2`, `sigma >= 0`, `lambda >= 0`,
  /// `omega >= 0`, `s0 > 0`, and every parameter is finite.
  pub fn new(
    drift: Drift<T>,
    sigma: T,
    lambda: T,
    nu: T,
    omega: T,
    n: usize,
    s0: T,
    t: Option<T>,
    seed: S,
  ) -> Self {
    drift.assert_finite();
    assert!(
      sigma.is_finite(),
      "sigma must satisfy `sigma.is_finite()`, got sigma = {sigma:?}"
    );
    assert!(
      lambda.is_finite(),
      "lambda must satisfy `lambda.is_finite()`, got lambda = {lambda:?}"
    );
    assert!(
      nu.is_finite(),
      "nu must satisfy `nu.is_finite()`, got nu = {nu:?}"
    );
    assert!(
      omega.is_finite(),
      "omega must satisfy `omega.is_finite()`, got omega = {omega:?}"
    );
    assert!(
      s0.is_finite(),
      "s0 must satisfy `s0.is_finite()`, got s0 = {s0:?}"
    );
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    assert!(
      sigma >= T::zero(),
      "sigma must satisfy `sigma >= T::zero()`, got sigma = {sigma:?}"
    );
    assert!(
      lambda >= T::zero(),
      "lambda must satisfy `lambda >= T::zero()`, got lambda = {lambda:?}"
    );
    assert!(
      omega >= T::zero(),
      "omega must satisfy `omega >= T::zero()`, got omega = {omega:?}"
    );
    assert!(
      s0 > T::zero(),
      "s0 must satisfy `s0 > T::zero()`, got s0 = {s0:?}"
    );
    Self {
      backend: Cpu,
      drift,
      sigma,
      lambda,
      nu,
      omega,
      n,
      s0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> MjdLog<T, S, B> {
  /// Replace `drift`, all else unchanged; panics on a non-finite rate.
  pub fn with_drift(mut self, drift: Drift<T>) -> Self {
    drift.assert_finite();
    self.drift = drift;
    self
  }

  #[inline]
  fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
  }

  #[inline]
  fn kappa_j(&self) -> T {
    (self.nu + T::from_f64_fast(0.5) * self.omega * self.omega).exp() - T::one()
  }
}

/// The Euler engine's view: the whole log-drift is folded here, as the host sampler folds it, and
/// the jump intensity is a launch argument, since the kernel draws one count per step.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for MjdLog<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let half = T::from_f64_fast(0.5);
    let drift_ln =
      (self.drift.rate() - self.lambda * self.kappa_j() - half * self.sigma * self.sigma)
        * self.dt();
    crate::euler::EulerSpec::MertonJumpLog {
      drift_ln,
      sigma: self.sigma,
    }
  }

  fn initial_value(&self) -> T {
    self.s0
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  fn time_step(&self) -> T {
    self.dt()
  }

  /// The jump intensity per unit time; `None` when the process carries none,
  /// which is what makes a zero-intensity build skip the draw entirely.
  fn jump_intensity(&self) -> Option<T> {
    (self.lambda > T::zero()).then_some(self.lambda)
  }

  /// Lognormal jump sizes, which are normal in the log-price the family
  /// steps, so the kernel aggregates however many the step saw into one draw.
  fn jump_sizes(&self) -> Option<crate::euler::JumpSizes<T>> {
    Some(crate::euler::JumpSizes::Normal {
      mean: self.nu,
      sd: self.omega,
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

backend_switch!([T: FloatExt, S: SeedExt] MjdLog<T, S> { drift, sigma, lambda, nu, omega, n, s0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for MjdLog<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for MjdLog<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = MjdLogSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> MjdLogSampler<T> {
    // The sources take their seeds from `self.seed` in a fixed order the pinned streams depend on;
    // reuse advances them for independent paths.
    let dt = self.dt();
    let sqrt_dt = dt.sqrt();
    let drift = self.drift.rate();
    let kappa_j = self.kappa_j();
    let half = T::from_f64_fast(0.5);
    let drift_ln = (drift - self.lambda * kappa_j - half * self.sigma * self.sigma) * dt;

    // One seed is skipped: the pinned streams that follow take theirs after it.
    self.seed.next_seed();

    let pois = if self.lambda > T::zero() {
      Some(SimdPoisson::<u32>::new((self.lambda * dt).to_f64().unwrap()).seeded(&self.seed))
    } else {
      None
    };

    let normal = SimdNormal::<T>::new(T::zero(), sqrt_dt).seeded(&self.seed);
    let jump_normal = SimdNormal::<T>::new(T::zero(), T::one()).seeded(&self.seed);

    MjdLogSampler {
      n: self.n,
      sigma: self.sigma,
      nu: self.nu,
      omega: self.omega,
      s0: self.s0,
      drift_ln,
      pois,
      normal,
      jump_normal,
    }
  }

  /// Through the Euler engine: a device runs the diffusion and the jump in its kernel, the host
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

/// Reusable [`MjdLog`] sampling state: the Poisson driver and the Gaussian sources of the
/// diffusion and of the jump sizes.
#[doc(hidden)]
#[derive(Debug)]
pub struct MjdLogSampler<T: FloatExt> {
  n: usize,
  sigma: T,
  nu: T,
  omega: T,
  s0: T,
  drift_ln: T,
  pois: Option<Seeded<SimdPoisson<u32>>>,
  normal: Seeded<SimdNormal<T>>,
  jump_normal: Seeded<SimdNormal<T>>,
}

impl<T: FloatExt> MjdLogSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }

    assert!(
      self.s0 > T::zero(),
      "s0 must satisfy `s0 > T::zero()`, got s0 = {:?}",
      self.s0
    );
    let s0 = self.s0;
    out[0] = s0;
    if out.len() == 1 {
      return;
    }

    let drift_ln = self.drift_ln;

    let mut prev = s0;
    let tail = &mut out[1..];
    self.normal.fill_slice(tail);

    for z in tail.iter_mut() {
      let diff = self.sigma * *z;

      let mut jump_sum = T::zero();
      if let Some(pois) = &mut self.pois {
        let k = pois.sample();
        if k > 0 {
          let kf = T::from_usize_(k as usize);
          let mut z0 = [T::zero(); 1];
          self.jump_normal.fill_slice(&mut z0);
          jump_sum = self.nu * kf + self.omega * kf.sqrt() * z0[0];
        }
      }

      let log_inc = drift_ln + diff + jump_sum;
      let next = prev * log_inc.exp();
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> crate::traits::Sealed for MjdLogSampler<T> {}

impl<T: FloatExt> PathSampler<T> for MjdLogSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    self.fill_path(
      out
        .as_slice_mut()
        .expect("MjdLog output must be contiguous"),
    );
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn price_stays_positive() {
    let p = MjdLog::new(
      Drift::Mu(0.05_f64),
      0.2,
      0.5,
      -0.1,
      0.15,
      256,
      100.0,
      Some(1.0),
      Unseeded,
    );
    let s = p.sample();
    assert!(s.iter().all(|x| *x > 0.0));
  }
}

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
#[doc(hidden)]
pub use python::PyMjdLog;

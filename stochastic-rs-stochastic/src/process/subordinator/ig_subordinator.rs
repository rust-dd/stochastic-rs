use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::inverse_gauss::SimdInverseGauss;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Inverse-Gaussian subordinator with BNS parameterization:
/// `phi(lambda) = delta (sqrt(gamma^2 + 2 lambda) - gamma)`.
#[derive(Debug, Clone)]
pub struct IgSubordinator<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Scale `delta`.
  pub delta: T,
  /// Shape `gamma`.
  pub gamma: T,
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

impl<T: FloatExt, S: SeedExt> IgSubordinator<T, S> {
  /// Panics unless `n >= 2`, a given `t` is positive and finite, `delta > 0`, `gamma > 0`, and
  /// every parameter is finite.
  pub fn new(delta: T, gamma: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    assert!(
      delta.is_finite(),
      "delta must satisfy `delta.is_finite()`, got delta = {delta:?}"
    );
    assert!(
      gamma.is_finite(),
      "gamma must satisfy `gamma.is_finite()`, got gamma = {gamma:?}"
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
      delta > T::zero(),
      "delta must satisfy `delta > T::zero()`, got delta = {delta:?}"
    );
    assert!(
      gamma > T::zero(),
      "gamma must satisfy `gamma > T::zero()`, got gamma = {gamma:?}"
    );
    Self {
      backend: Cpu,
      delta,
      gamma,
      n,
      x0,
      t,
      seed,
    }
  }
}

/// The Euler engine's view of the inverse-Gaussian subordinator.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for IgSubordinator<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let two = T::from_f64_fast(2.0);
    let four = T::from_f64_fast(4.0);
    crate::euler::EulerSpec::InverseGaussianSubordinator {
      mu_ig: self.delta * self.time_step() / self.gamma,
      two_lam: two * ((self.delta * self.time_step()) * (self.delta * self.time_step())),
      four_mu_lam: four
        * (self.delta * self.time_step() / self.gamma)
        * ((self.delta * self.time_step()) * (self.delta * self.time_step())),
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

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] IgSubordinator<T, S> { delta, gamma, n, x0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for IgSubordinator<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for IgSubordinator<T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = IgSubordinatorSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> IgSubordinatorSampler<T> {
    let x0 = self.x0.unwrap_or(T::zero());
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let mu = (self.delta * dt) / self.gamma;
    let lambda = (self.delta * dt).powi(2);
    IgSubordinatorSampler {
      n: self.n,
      x0,
      ig: SimdInverseGauss::<T>::new(mu, lambda).seeded(&self.seed),
    }
  }

  /// Through the Euler engine: a device draws the increment in its kernel, the host runs this
  /// process's own sampler, chunked as `ProcessExt` chunks.
  fn sample(&self) -> Array1<T> {
    self.backend.euler_sample(self)
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    self.backend.euler_paths_map(self, m, f)
  }

  fn sample_map_view<R: Send>(
    &self,
    m: usize,
    f: impl Fn(ndarray::ArrayView1<T>) -> R + Sync,
  ) -> Vec<R> {
    self.backend.euler_paths_map_view(self, m, f)
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

/// Reusable [`IgSubordinator`] sampling state: the owned inverse-Gaussian
/// increment source. The path is `x0` followed by the running sum.
#[doc(hidden)]
#[derive(Debug)]
pub struct IgSubordinatorSampler<T: FloatExt> {
  n: usize,
  x0: T,
  ig: Seeded<SimdInverseGauss<T>>,
}

impl<T: FloatExt> IgSubordinatorSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }
    let tail = &mut out[1..];
    self.ig.fill_slice(tail);
    let mut acc = self.x0;
    for x in tail.iter_mut() {
      acc += *x;
      *x = acc;
    }
  }
}

impl<T: FloatExt> crate::traits::Sealed for IgSubordinatorSampler<T> {}

impl<T: FloatExt> PathSampler<T> for IgSubordinatorSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("IgSubordinator output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

/// The inverse-Gaussian process of mean `δt` and shape `(δt)²` is this subordinator at `gamma = 1`.
///
/// ```compile_fail,E0432
/// use stochastic_rs_stochastic::jump::ig::Ig;
/// ```
#[cfg(doctest)]
pub struct IgIsTheUnitSubordinator;

py_process_1d!(PyIGSubordinator, IgSubordinator,
  sig: (delta, gamma_, n, x0=None, t=None, seed=None, dtype=None),
  params: (delta: f64, gamma_: f64, n: usize, x0: Option<f64>, t: Option<f64>),
  device
);

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  /// The first eight points `Ig::new(2.0, 65, None, None, Deterministic::new(3))` sampled.
  const IG_GAMMA_TWO: [u64; 8] = [
    0,
    4587770068631316187,
    4588072030184553316,
    4588144219662092651,
    4588253019380183630,
    4590203735888847721,
    4590223691370515422,
    4590258516375784657,
  ];

  #[test]
  fn delta_gamma_with_unit_gamma_is_the_inverse_gaussian_process_bit_for_bit() {
    let path = IgSubordinator::new(2.0_f64, 1.0, 65, None, None, Deterministic::new(3)).sample();
    let bits = path
      .iter()
      .take(8)
      .map(|x| x.to_bits())
      .collect::<Vec<u64>>();
    assert_eq!(bits, IG_GAMMA_TWO);
  }
}

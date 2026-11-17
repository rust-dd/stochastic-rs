//! # Cev
//!
//! $$
//! dS_t=\mu S_t\,dt+\sigma S_t^{\gamma}\,dW_t
//! $$
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::malliavin::Malliavin;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Debug, Clone)]
pub struct Cev<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Constant proportional drift rate μ — CEV has no mean reversion.
  pub mu: T,
  /// Diffusion scale σ multiplying `S_t^γ dW_t`.
  pub sigma: T,
  /// CEV elasticity exponent γ (γ=1 recovers GBM; γ<1 fattens the
  /// left tail, the usual equity-market calibration). At non-integer γ the
  /// sampler raises `|S_t|` rather than `S_t` to this power: a discretized
  /// path can cross zero for a step even at valid parameters, and
  /// `f64::powf` of a negative base at a non-integer exponent is `NaN`,
  /// which would otherwise poison every subsequent step.
  pub gamma: T,
  /// Number of points sampled along the CEV path.
  pub n: usize,
  /// Initial value; `None` starts at zero.
  pub x0: Option<T>,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

/// Every field has a matching `with_*` builder setter, e.g.
/// `Cev::default().with_gamma(0.5).with_sigma(0.3)`. No persisted cache:
/// `sampler()` builds its Gaussian stream fresh from `self` every call.
impl<T: FloatExt, S: SeedExt> Cev<T, S> {
  pub fn new(mu: T, sigma: T, gamma: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
    Self::check_mu(mu);
    Self::check_sigma(sigma);
    Self::check_gamma(gamma);
    Self::check_n(n);
    Self::check_x0(x0);
    Self::check_t(t);
    Self {
      backend: Cpu,
      mu,
      sigma,
      gamma,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Cev<T, S, B> {
  fn check_mu(mu: T) {
    require!(mu.is_finite(), mu);
  }

  fn check_sigma(sigma: T) {
    require!(sigma.is_finite(), sigma);
  }

  fn check_gamma(gamma: T) {
    require!(gamma.is_finite(), gamma);
  }

  fn check_n(n: usize) {
    require!(n >= 2, n);
  }

  fn check_x0(x0: Option<T>) {
    require!(x0.is_none_or(|v| v.is_finite()), x0);
  }

  fn check_t(t: Option<T>) {
    require!(t.is_none_or(|t| t > T::zero() && t.is_finite()), t);
  }

  /// Replace `mu`, all else unchanged.
  pub fn with_mu(mut self, mu: T) -> Self {
    Self::check_mu(mu);
    self.mu = mu;
    self
  }

  /// Replace `sigma`, all else unchanged.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    Self::check_sigma(sigma);
    self.sigma = sigma;
    self
  }

  /// Replace `gamma`, all else unchanged.
  pub fn with_gamma(mut self, gamma: T) -> Self {
    Self::check_gamma(gamma);
    self.gamma = gamma;
    self
  }

  /// Replace `x0`, all else unchanged.
  pub fn with_x0(mut self, x0: Option<T>) -> Self {
    Self::check_x0(x0);
    self.x0 = x0;
    self
  }

  /// Replace `n`, the grid points along the path, all else unchanged.
  pub fn with_points(mut self, n: usize) -> Self {
    Self::check_n(n);
    self.n = n;
    self
  }

  /// Replace the simulation horizon `t`, all else unchanged.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    Self::check_t(t);
    self.t = t;
    self
  }

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

/// μ=0.04, σ=0.2, γ=0.8, x₀=1, a textbook Cev (γ<1 is the usual equity calibration); t=1, 252
/// points, a trading year of daily closes.
impl<T: FloatExt> Default for Cev<T, Unseeded> {
  fn default() -> Self {
    Self::new(
      T::from_f64_fast(0.04),
      T::from_f64_fast(0.2),
      T::from_f64_fast(0.8),
      252,
      Some(T::one()),
      Some(T::one()),
      Unseeded,
    )
  }
}

/// The Euler engine's view of CEV.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for Cev<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::ConstantElasticity {
      mu: self.mu,
      sigma: self.sigma,
      gamma: self.gamma,
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

backend_switch!([T: FloatExt, S: SeedExt] Cev<T, S> { mu, sigma, gamma, n, x0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Cev<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Cev<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = CevSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> CevSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    CevSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      mu: self.mu,
      diff_scale: self.sigma,
      gamma: self.gamma,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt()).seeded(&self.seed),
      dw: Vec::new(),
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

/// Reusable [`Cev`] sampling state.
#[doc(hidden)]
#[derive(Debug)]
pub struct CevSampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  mu: T,
  diff_scale: T,
  gamma: T,
  normal: Seeded<SimdNormal<T>>,
  dw: Vec<T>,
}

impl<T: FloatExt> CevSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    let mut dw = std::mem::take(&mut self.dw);
    dw.resize(out.len().saturating_sub(1), T::zero());
    self.fill_path_with(out, &mut dw);
    self.dw = dw;
  }

  /// Steps `out` from the `√dt`-scaled increments it first draws into `dw`, one per step.
  pub(crate) fn fill_path_with(&mut self, out: &mut [T], dw: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }
    let dw = &mut dw[..out.len() - 1];
    self.normal.fill_slice(dw);
    let mut prev = self.x0;
    for (next, z) in out[1..].iter_mut().zip(dw.iter()) {
      // `|prev|`, as `gamma`'s doc says: a step can cross zero, and a negative base is NaN here.
      *next = prev + self.mu * prev * self.dt + self.diff_scale * prev.abs().powf(self.gamma) * *z;
      prev = *next;
    }
  }
}

impl<T: FloatExt> crate::traits::Sealed for CevSampler<T> {}

impl<T: FloatExt> PathSampler<T> for CevSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("Cev output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> Cev<T, S, B> {
  /// A host-sampled path and Nualart's `D_{t_i} X_T`, `σ′ = γσ sgn(X)|X|^{γ−1}`, summed backward
  /// over the increments that drove the path; undefined at an exact zero for γ < 1:
  /// $$
  /// D_{t_i} X_T = \sigma |X_{t_i}|^{\gamma} \exp\Bigl(\sum_{k=i}^{n-2} \bigl(\mu - \tfrac12 \gamma^2 \sigma^2 |X_{t_k}|^{2\gamma - 2}\bigr) \Delta t + \gamma \sigma \operatorname{sgn}(X_{t_k}) |X_{t_k}|^{\gamma - 1} \Delta W_k\Bigr)
  /// $$
  pub fn malliavin(&self) -> Malliavin<T, Array1<T>> {
    let mut paths = Array1::<T>::zeros(self.n);
    let mut dw = vec![T::zero(); self.n.saturating_sub(1)];
    self
      .sampler()
      .fill_path_with(paths.as_slice_mut().expect("contiguous"), &mut dw);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(self.n.saturating_sub(1).max(1));
    let (half, two) = (T::from_f64_fast(0.5), T::from_usize_(2));
    let mut derivative = Array1::<T>::zeros(self.n);
    let mut tail = T::zero();
    for i in (0..self.n).rev() {
      let x = paths[i].abs();
      if i + 1 < self.n {
        tail += (self.mu
          - half * (self.gamma * self.sigma).powi(2) * x.powf(two * self.gamma - two))
          * dt
          + self.gamma * self.sigma * x.powf(self.gamma - T::one()) * paths[i].signum() * dw[i];
      }
      derivative[i] = self.sigma * x.powf(self.gamma) * tail.exp();
    }
    Malliavin { paths, derivative }
  }
}

py_process_1d!(PyCev, Cev,
  sig: (mu, sigma, gamma, n, x0=None, t=None, seed=None, dtype=None),
  params: (mu: f64, sigma: f64, gamma: f64, n: usize, x0: Option<f64>, t: Option<f64>),
  device
);

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  #[test]
  fn the_derivative_is_driven_by_the_increments_of_the_path_it_returns() {
    for x0 in [1.0_f64, -1.0] {
      let build = || {
        Cev::<f64, _>::new(
          0.05,
          0.3,
          0.7,
          65,
          Some(x0),
          Some(1.0),
          Deterministic::new(7),
        )
      };
      let Malliavin {
        paths, derivative, ..
      } = build().malliavin();
      assert_eq!(paths, build().sample());
      let mut path = vec![0.0_f64; 65];
      let mut dw = vec![0.0; 64];
      build().sampler().fill_path_with(&mut path, &mut dw);
      assert_eq!(paths.as_slice().unwrap(), path.as_slice());
      let (mu, sigma, gamma, dt) = (0.05_f64, 0.3_f64, 0.7_f64, 1.0 / 64.0);
      let mut tail = 0.0;
      for i in (0..65).rev() {
        let x = path[i].abs();
        if i < 64 {
          tail += (mu - 0.5 * (gamma * sigma).powi(2) * x.powf(2.0 * gamma - 2.0)) * dt
            + gamma * sigma * x.powf(gamma - 1.0) * path[i].signum() * dw[i];
        }
        let expected = sigma * x.powf(gamma) * tail.exp();
        assert_eq!(
          derivative[i].to_bits(),
          expected.to_bits(),
          "x0 = {x0}, t_{i}"
        );
      }
    }
  }

  /// Regression: at these valid, realistic parameters (`gamma < 1`, this
  /// file's own "usual equity-market calibration"), Euler discretization
  /// crosses zero routinely — before the `.abs()` fix in `fill_path`, an
  /// equivalent 2000-path run crossed zero in 681 paths (34%), and every
  /// one of those went `NaN` for the rest of its path via
  /// `f64::powf(negative, non-integer)`. Every point of every path must
  /// stay finite regardless.
  #[test]
  fn cev_stays_finite_through_a_zero_crossing() {
    let cev = Cev::<f64, _>::new(
      0.0,
      0.6,
      0.5,
      24,
      Some(0.5),
      Some(1.0),
      Deterministic::new(2718),
    );
    for path in cev.sample_par(500) {
      assert!(
        path.iter().all(|x| x.is_finite()),
        "path went non-finite: {path:?}"
      );
    }
  }

  /// Same regression for `malliavin()`, which independently re-raises the
  /// realized path to fractional powers of its own and needs its own
  /// `.abs()` for the same reason.
  #[test]
  fn cev_malliavin_stays_finite_through_a_zero_crossing() {
    let cev = Cev::<f64, _>::new(
      0.0,
      0.6,
      0.5,
      24,
      Some(0.5),
      Some(1.0),
      Deterministic::new(2718),
    );
    let Malliavin {
      paths: path,
      derivative: m,
      ..
    } = cev.malliavin();
    assert!(
      path.iter().all(|x| x.is_finite()),
      "path went non-finite: {path:?}"
    );
    assert!(
      m.iter().all(|x| x.is_finite()),
      "malliavin derivative went non-finite: {m:?}"
    );
  }
}

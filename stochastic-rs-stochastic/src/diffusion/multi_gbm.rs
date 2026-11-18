//! `k` GBMs on the correlated Brownian motion of [`crate::noise::mcgns::Mcgns`], stepped by the
//! log-Euler scheme of the second display, exact in law on the grid; the output is `k × n`.
//!
//! $$
//! dS^{(i)}_t = \mu_i S^{(i)}_t\,dt + \sigma_i S^{(i)}_t\,dW^{(i)}_t,\qquad
//! d\langle W^{(i)}, W^{(j)}\rangle_t = \rho_{ij}\,dt,
//! $$
//!
//! $$
//! S^{(i)}_{t+\Delta t} = S^{(i)}_t\exp\bigl((\mu_i - \tfrac12\sigma_i^2)\Delta t +
//! \sigma_i\,\Delta W^{(i)}_t\bigr)
//! $$
//!
//! Reference: Glasserman (2003), Springer, §3.2.3, DOI 10.1007/978-0-387-21617-1

use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::device::Cpu;
use crate::noise::mcgns::Mcgns;
use crate::precondition::assert_entries;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Debug, Clone)]
pub struct MultiGbm<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Drifts $\mu_i$, one per asset.
  pub mu: Array1<T>,
  /// Volatilities $\sigma_i$, one per asset.
  pub sigma: Array1<T>,
  /// Instantaneous correlation matrix ρ of the driving Brownian motions.
  pub rho: Array2<T>,
  /// Number of grid points per path, the first being `x0`.
  pub n: usize,
  /// Initial levels $S^{(i)}_0$, one per asset.
  pub x0: Array1<T>,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// The seed: [`Unseeded`] or [`Deterministic`](stochastic_rs_core::simd_rng::Deterministic).
  pub seed: S,
  driver: Mcgns<T, S>,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> MultiGbm<T, S> {
  /// `n >= 2` counts the initial point. Panics unless `rho` is a `k × k` correlation matrix,
  /// `mu`, `sigma >= 0` and `x0` are finite of length `k`, and a given `t` is positive and finite.
  pub fn new(
    mu: Array1<T>,
    sigma: Array1<T>,
    rho: Array2<T>,
    n: usize,
    x0: Array1<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    assert_entries("mu", "mu[i].is_finite()", &mu, |v| v.is_finite());
    assert_entries("sigma", "sigma[i].is_finite()", &sigma, |v| v.is_finite());
    assert_entries("rho", "rho[(i, j)].is_finite()", &rho, |v| v.is_finite());
    assert_entries("x0", "x0[i].is_finite()", &x0, |v| v.is_finite());
    assert!(
      rho.nrows() >= 1,
      "rho.nrows() must satisfy `rho.nrows() >= 1`, got rho.nrows() = {:?}",
      rho.nrows()
    );
    assert!(
      mu.len() == rho.nrows(),
      "mu.len() must satisfy `mu.len() == rho.nrows()`, got mu.len() = {:?}",
      mu.len()
    );
    assert!(
      sigma.len() == rho.nrows(),
      "sigma.len() must satisfy `sigma.len() == rho.nrows()`, got sigma.len() = {:?}",
      sigma.len()
    );
    assert!(
      x0.len() == rho.nrows(),
      "x0.len() must satisfy `x0.len() == rho.nrows()`, got x0.len() = {:?}",
      x0.len()
    );
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    assert_entries("sigma", "sigma[i] >= T::zero()", &sigma, |s| s >= T::zero());
    let driver = Mcgns::new(rho.clone(), n.saturating_sub(1), t, seed.clone());
    Self {
      backend: Cpu,
      mu,
      sigma,
      rho,
      n,
      x0,
      t,
      seed,
      driver,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> MultiGbm<T, S, B> {
  /// Number of assets `k`.
  pub fn assets(&self) -> usize {
    self.rho.nrows()
  }

  fn fill_paths<S2: SeedExt>(&self, seed: &S2, out: &mut Array2<T>) {
    let k = self.assets();
    assert_eq!(out.dim(), (k, self.n), "output must be k × n");
    for i in 0..k {
      out[(i, 0)] = self.x0[i];
    }
    if self.n == 1 {
      return;
    }
    let steps = self.n - 1;
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(steps);
    let mut dw = Array2::<T>::zeros((k, steps));
    self.driver.fill_increments(seed, &mut dw);
    for i in 0..k {
      let drift = (self.mu[i] - self.sigma[i] * self.sigma[i] / T::from_f64_fast(2.0)) * dt;
      for j in 1..self.n {
        out[(i, j)] = out[(i, j - 1)] * (drift + self.sigma[i] * dw[(i, j - 1)]).exp();
      }
    }
  }
}

/// The basket in the engine's four slots, which carries a launch when `k <= 4`: the assets fill
/// the first `k`, the rest are constant zero. It borrows, so it advances the process's seed.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct MultiGbmLaunch<'a, T: FloatExt, S: SeedExt, B>(&'a MultiGbm<T, S, B>);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for MultiGbmLaunch<'_, T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for MultiGbmLaunch<'_, T, S, B>
{
  type Output = [Array1<T>; 4];
  type Sampler<'s>
    = MultiGbmLaunchSampler<T, S>
  where
    Self: 's;

  fn sampler(&self) -> MultiGbmLaunchSampler<T, S> {
    MultiGbmLaunchSampler {
      inner: <MultiGbm<T, S, B> as ProcessExt<T>>::sampler(self.0),
    }
  }
}

/// [`MultiGbmLaunch`]'s sampler: the matrix sampler, its rows lifted into
/// the four slots.
#[doc(hidden)]
#[derive(Debug)]
pub struct MultiGbmLaunchSampler<T: FloatExt, S: SeedExt> {
  inner: MultiGbmSampler<T, S>,
}

impl<T: FloatExt, S: SeedExt> crate::traits::Sealed for MultiGbmLaunchSampler<T, S> {}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for MultiGbmLaunchSampler<T, S> {
  type Output = [Array1<T>; 4];

  fn sample_into(&mut self, out: &mut [Array1<T>; 4]) {
    *out = self.sample();
  }

  fn sample(&mut self) -> [Array1<T>; 4] {
    let matrix = self.inner.sample();
    let n = matrix.ncols();
    std::array::from_fn(|i| {
      if i < matrix.nrows() {
        matrix.row(i).to_owned()
      } else {
        Array1::zeros(n)
      }
    })
  }
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 4>
  for MultiGbmLaunch<'_, T, S, B>
{
  /// Padded into the four slots. [`ProcessExt::sample`] keeps a wider basket on the host, so a
  /// launch that asks here bypassed that guard.
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let p = self.0;
    let k = p.assets();
    assert!(
      k <= crate::euler::CORRELATED_STREAMS,
      "MultiGbm: a launch carries at most {} assets; sample through `ProcessExt`, which keeps a \
       wider basket on the host",
      crate::euler::CORRELATED_STREAMS
    );
    let padded = |v: &Array1<T>| std::array::from_fn(|i| if i < k { v[i] } else { T::zero() });
    crate::euler::EulerSpec::CorrelatedGeometric4 {
      mu: padded(&p.mu),
      sigma: padded(&p.sigma),
      l: crate::euler::pack_cholesky(p.driver.cholesky()),
    }
  }

  fn initial_state(&self) -> [T; 4] {
    let p = self.0;
    std::array::from_fn(|i| if i < p.assets() { p.x0[i] } else { T::zero() })
  }

  fn grid_points(&self) -> usize {
    self.0.n
  }

  fn horizon(&self) -> T {
    self.0.t.unwrap_or(T::one())
  }

  fn time_step(&self) -> T {
    let p = self.0;
    if p.n > 1 {
      p.t.unwrap_or(T::one()) / T::from_usize_(p.n - 1)
    } else {
      T::zero()
    }
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.0.seed)
  }

  fn host_sample(&self) -> [Array1<T>; 4] {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] MultiGbm<T, S> { mu, sigma, rho, n, x0, t, seed, driver } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for MultiGbm<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for MultiGbm<T, S, B>
{
  type Output = Array2<T>;
  type Sampler<'s>
    = MultiGbmSampler<T, S>
  where
    Self: 's;

  fn sampler(&self) -> MultiGbmSampler<T, S> {
    let seed = self.seed.derive();
    MultiGbmSampler {
      process: MultiGbm::new(
        self.mu.clone(),
        self.sigma.clone(),
        self.rho.clone(),
        self.n,
        self.x0.clone(),
        self.t,
        seed,
      ),
    }
  }

  /// Through the Euler engine for up to four assets, stepped in one kernel under their Cholesky
  /// factor; a larger basket needs more shocks than a launch has and stays on the host sampler.
  fn sample(&self) -> Array2<T> {
    if self.device_ready() {
      slots_to_matrix(
        self.backend.system_sample(&MultiGbmLaunch(self)),
        self.assets(),
      )
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array2<T>) -> R + Sync) -> Vec<R> {
    if self.device_ready() {
      let k = self.assets();
      self
        .backend
        .system_paths_map(&MultiGbmLaunch(self), m, |slots| {
          f(&slots_to_matrix(slots.clone(), k))
        })
    } else {
      crate::traits::process::sample_map_chunked(self, m, f)
    }
  }

  fn sample_par(&self, m: usize) -> Vec<Array2<T>> {
    if self.device_ready() {
      let k = self.assets();
      self
        .backend
        .system_paths(&MultiGbmLaunch(self), m)
        .into_iter()
        .map(|slots| slots_to_matrix(slots, k))
        .collect()
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<Array2<T>, crate::device::DeviceError> {
    if self.device_ready() {
      let slots = self.backend.try_system_sample(&MultiGbmLaunch(self))?;
      Ok(slots_to_matrix(slots, self.assets()))
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array2<T>>, crate::device::DeviceError> {
    if self.device_ready() {
      let k = self.assets();
      Ok(
        self
          .backend
          .try_system_paths(&MultiGbmLaunch(self), m)?
          .into_iter()
          .map(|slots| slots_to_matrix(slots, k))
          .collect(),
      )
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }

  /// What a device runs here: at most as many assets as a
  /// launch has state slots and shocks; a wider basket stays on the host.
  fn device_fallback(&self) -> Option<&'static str> {
    (!(self.assets() <= crate::euler::CORRELATED_STREAMS))
      .then_some("more assets than a launch has state slots and shocks")
  }
}

/// The first `k` of the engine's four slots as the `k × n` matrix the process
/// reports; the padded slots are dropped.
fn slots_to_matrix<T: FloatExt>(slots: [Array1<T>; 4], k: usize) -> Array2<T> {
  let n = slots[0].len();
  let mut out = Array2::<T>::zeros((k, n));
  for (i, row) in slots.iter().take(k).enumerate() {
    out.row_mut(i).assign(row);
  }
  out
}

/// Reusable [`MultiGbm`] sampling state with a derived seed, so parallel
/// chunks draw independent paths.
#[doc(hidden)]
#[derive(Debug)]
pub struct MultiGbmSampler<T: FloatExt, S: SeedExt> {
  process: MultiGbm<T, S>,
}

impl<T: FloatExt, S: SeedExt> crate::traits::Sealed for MultiGbmSampler<T, S> {}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for MultiGbmSampler<T, S> {
  type Output = Array2<T>;

  fn sample_into(&mut self, out: &mut Array2<T>) {
    // Same by-reference seed as `McgnsSampler`: the derived seed advances
    // on every call, so the paths of one chunk are independent.
    self.process.fill_paths(&self.process.seed, out);
  }

  fn sample(&mut self) -> Array2<T> {
    let mut out = Array2::<T>::zeros((self.process.assets(), self.process.n));
    self.sample_into(&mut out);
    out
  }
}

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
#[doc(hidden)]
pub use python::PyMultiGbm;

#[cfg(test)]
mod tests;

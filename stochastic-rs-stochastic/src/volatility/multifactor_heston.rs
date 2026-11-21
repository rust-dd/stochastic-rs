//! Multifactor Heston: one CIR factor cannot fit both skews and the ATM term structure (CHJ, 2009).
//!
//! $$
//! \begin{aligned}
//! \frac{dS_t}{S_t} &= \mu\,dt + \sum_{k=1}^{K} \sqrt{V_{k,t}}\,dW^{S}_{k,t}, \\
//! dV_{k,t} &= \kappa_k(\theta_k - V_{k,t})\,dt + \sigma_k\,\sqrt{V_{k,t}}\,dW^{V}_{k,t},\\
//! d\langle W^S_k, W^V_k\rangle_t &= \rho_k\,dt,\\
//! d\langle W^S_j, W^S_k\rangle_t &= 0,\ j \ne k,\\
//! d\langle W^V_j, W^V_k\rangle_t &= 0,\ j \ne k.
//! \end{aligned}
//! $$
//!
//! The asset steps by Euler, each variance factor by full truncation (Lord-Koekkoek-van Dijk 2010).
//!
//! Reference: Christoffersen et al. (2009), Manag. Sci. 55, 1914-1932, DOI 10.1287/mnsc.1090.1065

use ndarray::Array1;
use ndarray::ArrayView1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::device::Cpu;
use crate::noise::cgns::Cgns;
use crate::precondition::assert_entries;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Multifactor Heston model with `K` independent variance factors driving
/// a single stock. See module docs for the SDE form and reference.
#[derive(Debug, Clone)]
pub struct MultifactorHeston<T: FloatExt, const K: usize, S: SeedExt = Unseeded, B = Cpu> {
  /// Initial spot.
  pub s0: T,
  /// Initial variance per factor.
  pub v0: [T; K],
  /// Mean-reversion rate per factor.
  pub kappa: [T; K],
  /// Long-run variance per factor.
  pub theta: [T; K],
  /// Vol-of-vol per factor.
  pub sigma: [T; K],
  /// Asset-variance correlation per factor, $\rho_k \in (-1, 1)$.
  pub rho: [T; K],
  /// Drift of the stock (risk-neutral $r - q$ when pricing).
  pub mu: T,
  /// Grid points sampled along the path; `dt = t / (n − 1)`.
  pub n: usize,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Seed source.
  pub seed: S,
  /// One `Cgns` per factor for $(\Delta W^S_k, \Delta W^V_k)$ with
  /// correlation $\rho_k$.
  cgns: Vec<Cgns<T>>,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

/// Spot and the `K` variance factors.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct MultifactorHestonPaths<T, const K: usize> {
  pub s: Array1<T>,
  pub v: [Array1<T>; K],
}

impl<T, const K: usize> From<(Array1<T>, [Array1<T>; K])> for MultifactorHestonPaths<T, K> {
  fn from((s, v): (Array1<T>, [Array1<T>; K])) -> Self {
    Self { s, v }
  }
}

impl<T: FloatExt, const K: usize, S: SeedExt> MultifactorHeston<T, K, S> {
  /// Panics unless every parameter is finite, `K >= 1`, each `v0`, `kappa`, `theta`, `sigma` entry
  /// is non-negative, each `|rho| < 1`, `n >= 2` and a given `t` is positive and finite.
  pub fn new(
    s0: T,
    v0: [T; K],
    kappa: [T; K],
    theta: [T; K],
    sigma: [T; K],
    rho: [T; K],
    mu: T,
    n: usize,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert!(
      s0.is_finite(),
      "s0 must satisfy `s0.is_finite()`, got s0 = {s0:?}"
    );
    assert!(
      mu.is_finite(),
      "mu must satisfy `mu.is_finite()`, got mu = {mu:?}"
    );
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    assert!(K >= 1, "K must satisfy `K >= 1`, got K = {K:?}");
    assert_entries("v0", "v0[k].is_finite()", &ArrayView1::from(&v0), |v| {
      v.is_finite()
    });
    assert_entries(
      "kappa",
      "kappa[k].is_finite()",
      &ArrayView1::from(&kappa),
      |v| v.is_finite(),
    );
    assert_entries(
      "theta",
      "theta[k].is_finite()",
      &ArrayView1::from(&theta),
      |v| v.is_finite(),
    );
    assert_entries(
      "sigma",
      "sigma[k].is_finite()",
      &ArrayView1::from(&sigma),
      |v| v.is_finite(),
    );
    assert_entries("rho", "rho[k].is_finite()", &ArrayView1::from(&rho), |v| {
      v.is_finite()
    });
    for k in 0..K {
      assert!(
        kappa[k] >= T::zero(),
        "kappa[{k}] must satisfy `kappa[k] >= T::zero()`, got kappa[{k}] = {:?}",
        kappa[k]
      );
      assert!(
        theta[k] >= T::zero(),
        "theta[{k}] must satisfy `theta[k] >= T::zero()`, got theta[{k}] = {:?}",
        theta[k]
      );
      assert!(
        sigma[k] >= T::zero(),
        "sigma[{k}] must satisfy `sigma[k] >= T::zero()`, got sigma[{k}] = {:?}",
        sigma[k]
      );
      assert!(
        v0[k] >= T::zero(),
        "v0[{k}] must satisfy `v0[k] >= T::zero()`, got v0[{k}] = {:?}",
        v0[k]
      );
      let r = rho[k].to_f64().unwrap();
      assert!(
        r.abs() < 1.0,
        "rho[{k}] must satisfy `rho[k].abs() < 1.0`, got rho[{k}] = {r:?}"
      );
    }
    let cgns = (0..K)
      .map(|k| Cgns::new(rho[k], n - 1, t, Unseeded))
      .collect::<Vec<_>>();
    Self {
      backend: Cpu,
      s0,
      v0,
      kappa,
      theta,
      sigma,
      rho,
      mu,
      n,
      t,
      seed,
      cgns,
    }
  }
}

/// How many variance factors a launch carries: the double-Heston family has two, and a
/// one-factor model rides it with the second zeroed; more factors stay on the host sampler.
const DEVICE_FACTORS: usize = 2;

/// The model in the double-Heston family's three slots (spot and two variances), a launch for
/// `K <= 2`; one factor pads the second variance with zeros. It borrows the process's own seed.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct MultifactorHestonLaunch<'a, T: FloatExt, const K: usize, S: SeedExt, B>(
  &'a MultifactorHeston<T, K, S, B>,
);

impl<T: FloatExt, const K: usize, S: SeedExt, B: crate::euler::EulerBackend<T>>
  crate::traits::Sealed for MultifactorHestonLaunch<'_, T, K, S, B>
{
}

impl<T: FloatExt, const K: usize, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for MultifactorHestonLaunch<'_, T, K, S, B>
{
  type Output = [Array1<T>; 3];
  type Sampler<'s>
    = MultifactorHestonLaunchSampler<T, K, S>
  where
    Self: 's;

  fn sampler(&self) -> MultifactorHestonLaunchSampler<T, K, S> {
    MultifactorHestonLaunchSampler {
      inner: <MultifactorHeston<T, K, S, B> as ProcessExt<T>>::sampler(self.0),
    }
  }
}

/// [`MultifactorHestonLaunch`]'s sampler: the model's own sampler, its spot
/// and factors lifted into the three slots.
#[doc(hidden)]
#[derive(Debug)]
pub struct MultifactorHestonLaunchSampler<T: FloatExt, const K: usize, S: SeedExt> {
  inner: MultifactorHestonSampler<T, K, S>,
}

impl<T: FloatExt, const K: usize, S: SeedExt> crate::traits::Sealed
  for MultifactorHestonLaunchSampler<T, K, S>
{
}

impl<T: FloatExt, const K: usize, S: SeedExt> PathSampler<T>
  for MultifactorHestonLaunchSampler<T, K, S>
{
  type Output = [Array1<T>; 3];

  fn sample_into(&mut self, out: &mut [Array1<T>; 3]) {
    *out = self.sample();
  }

  fn sample(&mut self) -> [Array1<T>; 3] {
    let MultifactorHestonPaths { s, v } = self.inner.sample();
    let n = s.len();
    let factor = |k: usize| v.get(k).cloned().unwrap_or_else(|| Array1::zeros(n));
    [s, factor(0), factor(1)]
  }
}

impl<T: FloatExt, const K: usize, S: SeedExt, B: crate::euler::EulerBackend<T>>
  crate::euler::EulerSystem<T, 3> for MultifactorHestonLaunch<'_, T, K, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let p = self.0;
    let at = |arr: &[T; K], k: usize| if k < K { arr[k] } else { T::zero() };
    crate::euler::EulerSpec::DoubleHeston {
      mu: p.mu,
      kappa1: at(&p.kappa, 0),
      theta1: at(&p.theta, 0),
      sigma1: at(&p.sigma, 0),
      rho1: at(&p.rho, 0),
      kappa2: at(&p.kappa, 1),
      theta2: at(&p.theta, 1),
      sigma2: at(&p.sigma, 1),
      rho2: at(&p.rho, 1),
    }
  }

  /// The spot and the factors, the latter floored at zero exactly as the
  /// host sampler floors its own starting variances.
  fn initial_state(&self) -> [T; 4] {
    let p = self.0;
    let v0 = |k: usize| {
      if k < K {
        p.v0[k].max(T::zero())
      } else {
        T::zero()
      }
    };
    [p.s0, v0(0), v0(1), T::zero()]
  }

  fn grid_points(&self) -> usize {
    self.0.n
  }

  fn horizon(&self) -> T {
    self.0.t.unwrap_or(T::one())
  }

  fn time_step(&self) -> T {
    self.0.cgns[0].dt()
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.0.seed)
  }

  fn host_sample(&self) -> [Array1<T>; 3] {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

/// The three slots as the spot and `K` variances the model reports; a padded slot is dropped.
fn slots_to_output<T: FloatExt, const K: usize>(
  slots: [Array1<T>; 3],
) -> MultifactorHestonPaths<T, K> {
  let [s, v1, v2] = slots;
  let factors = [v1, v2];
  MultifactorHestonPaths {
    s,
    v: std::array::from_fn(|k| factors[k].clone()),
  }
}

backend_switch!([T: FloatExt, const K: usize, S: SeedExt] MultifactorHeston<T, K, S> { s0, v0, kappa, theta, sigma, rho, mu, n, t, seed, cgns } via euler);

impl<T: FloatExt, const K: usize, S: SeedExt, B: crate::euler::EulerBackend<T>>
  crate::traits::Sealed for MultifactorHeston<T, K, S, B>
{
}

impl<T: FloatExt, const K: usize, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for MultifactorHeston<T, K, S, B>
{
  type Output = MultifactorHestonPaths<T, K>;
  type Sampler<'s>
    = MultifactorHestonSampler<T, K, S>
  where
    Self: 's;

  /// Derives `self.seed` into the sampler, never clones it: the mixed next tick hash-scrambles
  /// chunk `i`'s basis against chunk `i+1`'s, where a raw snapshot leaves them one stride apart.
  fn sampler(&self) -> MultifactorHestonSampler<T, K, S> {
    MultifactorHestonSampler {
      n: self.n,
      s0: self.s0,
      v0: std::array::from_fn(|k| self.v0[k].max(T::zero())),
      kappa: self.kappa,
      theta: self.theta,
      sigma: self.sigma,
      mu: self.mu,
      dt: self.cgns[0].dt(),
      cgns: self.cgns.clone(),
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine for one or two factors, stepped with the spot in one kernel; more
  /// factors stay on the host sampler, since a launch has four shocks and each factor takes two.
  fn sample(&self) -> MultifactorHestonPaths<T, K> {
    if K <= DEVICE_FACTORS {
      slots_to_output(self.backend.system_sample(&MultifactorHestonLaunch(self)))
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_map<R: Send>(
    &self,
    m: usize,
    f: impl Fn(&MultifactorHestonPaths<T, K>) -> R + Sync,
  ) -> Vec<R> {
    if K <= DEVICE_FACTORS {
      self
        .backend
        .system_paths_map(&MultifactorHestonLaunch(self), m, |slots| {
          f(&slots_to_output(slots.clone()))
        })
    } else {
      crate::traits::process::sample_map_chunked(self, m, f)
    }
  }

  fn sample_par(&self, m: usize) -> Vec<MultifactorHestonPaths<T, K>> {
    if K <= DEVICE_FACTORS {
      self
        .backend
        .system_paths(&MultifactorHestonLaunch(self), m)
        .into_iter()
        .map(slots_to_output)
        .collect()
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<MultifactorHestonPaths<T, K>, crate::device::DeviceError> {
    if K <= DEVICE_FACTORS {
      Ok(slots_to_output(
        self
          .backend
          .try_system_sample(&MultifactorHestonLaunch(self))?,
      ))
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(
    &self,
    m: usize,
  ) -> Result<Vec<MultifactorHestonPaths<T, K>>, crate::device::DeviceError> {
    if K <= DEVICE_FACTORS {
      Ok(
        self
          .backend
          .try_system_paths(&MultifactorHestonLaunch(self), m)?
          .into_iter()
          .map(slots_to_output)
          .collect(),
      )
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }
}

/// Reusable [`MultifactorHeston`] sampling state: one correlated-Gaussian generator per factor
/// and the seed source; per-factor noise is drawn off the owned `seed` in factor order.
#[doc(hidden)]
#[derive(Debug)]
pub struct MultifactorHestonSampler<T: FloatExt, const K: usize, S: SeedExt> {
  n: usize,
  s0: T,
  v0: [T; K],
  kappa: [T; K],
  theta: [T; K],
  sigma: [T; K],
  mu: T,
  dt: T,
  cgns: Vec<Cgns<T>>,
  seed: S,
}

impl<T: FloatExt, const K: usize, S: SeedExt> MultifactorHestonSampler<T, K, S> {
  fn fill_paths(&mut self, s: &mut [T], v: &mut [&mut [T]; K]) {
    if self.n == 0 {
      return;
    }
    let dt = self.dt;

    s[0] = self.s0;
    for k in 0..K {
      v[k][0] = self.v0[k];
    }

    // Pre-sample all factor noises so the inner loop is a tight scalar pass.
    let factor_noises = (0..K)
      .map(|k| self.cgns[k].sample_impl(&self.seed))
      .collect::<Vec<_>>();

    for i in 1..self.n {
      // 1) Compute the total asset increment from all factor draws.
      let mut sum_sqrt_v_dw = T::zero();
      for k in 0..K {
        let v_prev = v[k][i - 1].max(T::zero());
        sum_sqrt_v_dw += v_prev.sqrt() * factor_noises[k][0][i - 1];
      }
      s[i] = s[i - 1] + self.mu * s[i - 1] * dt + s[i - 1] * sum_sqrt_v_dw;

      // 2) Full-truncation Euler step for each variance factor independently.
      for k in 0..K {
        let v_prev = v[k][i - 1].max(T::zero());
        let dv = self.kappa[k] * (self.theta[k] - v_prev) * dt
          + self.sigma[k] * v_prev.sqrt() * factor_noises[k][1][i - 1];
        v[k][i] = (v[k][i - 1] + dv).max(T::zero());
      }
    }
  }
}

impl<T: FloatExt, const K: usize, S: SeedExt> crate::traits::Sealed
  for MultifactorHestonSampler<T, K, S>
{
}

impl<T: FloatExt, const K: usize, S: SeedExt> PathSampler<T> for MultifactorHestonSampler<T, K, S> {
  type Output = MultifactorHestonPaths<T, K>;

  fn sample_into(&mut self, out: &mut MultifactorHestonPaths<T, K>) {
    let MultifactorHestonPaths { s: s_arr, v: v_arr } = out;
    let s = s_arr
      .as_slice_mut()
      .expect("MultifactorHeston stock output must be contiguous");
    let mut v: [&mut [T]; K] = v_arr.each_mut().map(|arr| {
      arr
        .as_slice_mut()
        .expect("MultifactorHeston variance output must be contiguous")
    });
    self.fill_paths(s, &mut v);
  }

  fn sample(&mut self) -> MultifactorHestonPaths<T, K> {
    let mut s_arr = Array1::<T>::zeros(self.n);
    let mut v_arr: [Array1<T>; K] = std::array::from_fn(|_| Array1::<T>::zeros(self.n));
    {
      let s = s_arr.as_slice_mut().expect("contiguous");
      let mut v: [&mut [T]; K] = v_arr
        .each_mut()
        .map(|arr| arr.as_slice_mut().expect("contiguous"));
      self.fill_paths(s, &mut v);
    }
    MultifactorHestonPaths { s: s_arr, v: v_arr }
  }
}

#[cfg(test)]
mod tests;

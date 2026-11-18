//! SABR with piecewise-constant `beta`, `nu` and `rho` over the buckets that `knots` cut.
//!
//! $$
//! \begin{aligned}
//! dF_t &= \alpha_t\,F_t^{\beta(t)}\,dW^1_t,\\
//! d\alpha_t &= \nu(t)\,\alpha_t\,dW^2_t,\\
//! d\langle W^1, W^2\rangle_t &= \rho(t)\,dt,
//! \end{aligned}
//! $$
//!
//! The active bucket is the number of knots $\le t_i$; the forward steps by Euler, absorbed at 0.
//!
//! $$
//! \Delta W^1_i = z^1_i, \qquad
//! \Delta W^2_i = \rho(t_i)\,z^1_i + \sqrt{1 - \rho(t_i)^2}\,z^2_i,\qquad
//! \alpha_{i} = \alpha_{i-1}\exp(\nu\,\Delta W^2 - \tfrac12\nu^2\Delta t)
//! $$
//!
//! Reference: Hagan, Kumar, Lesniewski & Woodward (2002), Wilmott Magazine, 84-108
//! Reference: Fernández, Ferreiro, García et al. (2024), arXiv:2407.20713, the dynamic SABR
//! Reference: Osajima (2007), SSRN 965265

use ndarray::Array1;
use ndarray::ArrayView1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use crate::device::Cpu;
use crate::noise::gn::Gn;
use crate::precondition::assert_entries;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Dynamic / multi-factor SABR with piecewise-constant `beta`, `rho` and
/// `nu` term structures. See module docs for the SDE and discretisation.
#[derive(Debug, Clone)]
pub struct MultifactorSabr<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Initial forward-rate level.
  pub f0: Option<T>,
  /// Initial volatility level $\alpha_0$.
  pub alpha0: Option<T>,
  /// `K` sorted time breakpoints in $(0, T)$, making `K + 1` buckets; empty is the static SABR
  /// with the single bucket's coefficients.
  pub knots: Vec<T>,
  /// Backbone exponent $\beta(t) \in [0, 1]$ per bucket. Length `K + 1`.
  pub beta: Vec<T>,
  /// Correlation $\rho(t) \in (-1, 1)$ per bucket. Length `K + 1`.
  pub rho: Vec<T>,
  /// Vol-of-vol $\nu(t) \ge 0$ per bucket. Length `K + 1`.
  pub nu: Vec<T>,
  /// Number of points sampled along the dynamic-SABR path.
  pub n: usize,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Seed strategy (compile-time: `Unseeded` or `Deterministic`).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> MultifactorSabr<T, S> {
  /// Panics unless all are finite, `beta`, `rho`, `nu` are `knots.len() + 1` long, a given `t > 0`,
  /// `alpha0 >= 0`, `0 <= beta <= 1`, `nu >= 0`, `|rho| < 1`, `knots[i - 1] < knots[i]`, `n >= 2`.
  pub fn new(
    f0: Option<T>,
    alpha0: Option<T>,
    knots: Vec<T>,
    beta: Vec<T>,
    rho: Vec<T>,
    nu: Vec<T>,
    n: usize,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert!(
      f0.is_none_or(|v| v.is_finite()),
      "f0 must satisfy `f0.is_none_or(|v| v.is_finite())`, got f0 = {f0:?}"
    );
    assert!(
      alpha0.is_none_or(|v| v.is_finite()),
      "alpha0 must satisfy `alpha0.is_none_or(|v| v.is_finite())`, got alpha0 = {alpha0:?}"
    );
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    assert_entries(
      "knots",
      "knots[k].is_finite()",
      &ArrayView1::from(&knots),
      |v| v.is_finite(),
    );
    assert_entries(
      "beta",
      "beta[k].is_finite()",
      &ArrayView1::from(&beta),
      |v| v.is_finite(),
    );
    assert_entries("rho", "rho[k].is_finite()", &ArrayView1::from(&rho), |v| {
      v.is_finite()
    });
    assert_entries("nu", "nu[k].is_finite()", &ArrayView1::from(&nu), |v| {
      v.is_finite()
    });
    let buckets = knots.len() + 1;
    assert!(
      beta.len() == knots.len() + 1,
      "beta.len() must satisfy `beta.len() == knots.len() + 1`, got beta.len() = {:?}",
      beta.len()
    );
    assert!(
      rho.len() == knots.len() + 1,
      "rho.len() must satisfy `rho.len() == knots.len() + 1`, got rho.len() = {:?}",
      rho.len()
    );
    assert!(
      nu.len() == knots.len() + 1,
      "nu.len() must satisfy `nu.len() == knots.len() + 1`, got nu.len() = {:?}",
      nu.len()
    );
    for k in 0..buckets {
      assert!(
        beta[k] >= T::zero() && beta[k] <= T::one(),
        "beta[{k}] must satisfy `beta[k] >= T::zero() && beta[k] <= T::one()`, got beta[{k}] = {:?}",
        beta[k]
      );
      assert!(
        nu[k] >= T::zero(),
        "nu[{k}] must satisfy `nu[k] >= T::zero()`, got nu[{k}] = {:?}",
        nu[k]
      );
      let r = rho[k].to_f64().unwrap();
      assert!(
        r.abs() < 1.0,
        "rho[{k}] must satisfy `rho[k].abs() < 1.0`, got rho[{k}] = {r:?}"
      );
    }
    for i in 1..knots.len() {
      assert!(
        knots[i - 1] < knots[i],
        "knots[{i}] must satisfy `knots[i - 1] < knots[i]`, got knots[{i}] = {:?}, knots[{}] = {:?}",
        knots[i],
        i - 1,
        knots[i - 1]
      );
    }
    assert!(
      alpha0.is_none_or(|a| a >= T::zero()),
      "alpha0 must satisfy `alpha0.is_none_or(|a| a >= T::zero())`, got alpha0 = {alpha0:?}"
    );
    Self {
      backend: Cpu,
      f0,
      alpha0,
      knots,
      beta,
      rho,
      nu,
      n,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> MultifactorSabr<T, S, B> {
  /// Index of the active bucket at time `time`: the number of knots
  /// less than or equal to `time`, clamped to the last bucket.
  #[inline]
  pub fn bucket_at(&self, time: T) -> usize {
    let mut idx = 0usize;
    for &knot in &self.knots {
      if time >= knot {
        idx += 1;
      } else {
        break;
      }
    }
    idx.min(self.beta.len() - 1)
  }

  fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n.saturating_sub(1).max(1))
  }
}

/// The Euler engine's view: the term structure travels as three curves, so the host's per-step
/// bucket search becomes a buffer read; the knots never reach the device.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 2>
  for MultifactorSabr<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::DynamicSabr
  }

  fn initial_state(&self) -> [T; 4] {
    [
      self.f0.unwrap_or(T::zero()),
      self.alpha0.unwrap_or(T::zero()).max(T::zero()),
      T::zero(),
      T::zero(),
    ]
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

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  /// `beta`, `rho` and `nu` at each grid point, the step's `ct`, `ct1` and `ct2`. Step `i` uses
  /// the bucket active at `(i - 1) dt`, the time the host reads, so index 0 carries index 1's.
  fn curves(&self) -> Option<Vec<Vec<T>>> {
    let dt = self.dt();
    let at = |i: usize| self.bucket_at(T::from_usize_(i.saturating_sub(1)) * dt);
    Some(vec![
      (0..self.n).map(|i| self.beta[at(i)]).collect(),
      (0..self.n).map(|i| self.rho[at(i)]).collect(),
      (0..self.n).map(|i| self.nu[at(i)]).collect(),
    ])
  }

  fn host_sample(&self) -> [Array1<T>; 2] {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] MultifactorSabr<T, S> { f0, alpha0, knots, beta, rho, nu, n, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for MultifactorSabr<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for MultifactorSabr<T, S, B>
{
  /// `[F path, α path]`: index 0 is the forward `F`, index 1 is the
  /// stochastic-volatility state `α` (see module docs for the SDE).
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = MultifactorSabrSampler<T, S>
  where
    Self: 's;

  /// Derives `self.seed` into the sampler, never clones it: the mixed next tick hash-scrambles
  /// chunk `i`'s basis against chunk `i+1`'s, where a raw snapshot leaves them one stride apart.
  fn sampler(&self) -> MultifactorSabrSampler<T, S> {
    MultifactorSabrSampler {
      n: self.n,
      t: self.t,
      f0: self.f0.unwrap_or(T::zero()),
      alpha0: self.alpha0.unwrap_or(T::zero()).max(T::zero()),
      knots: self.knots.clone(),
      beta: self.beta.clone(),
      rho: self.rho.clone(),
      nu: self.nu.clone(),
      dt: self.dt(),
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine: a device steps both rows in its kernel under the tabulated term
  /// structure, the host runs this process's own sampler, chunked as [`ProcessExt`] chunks.
  fn sample(&self) -> [Array1<T>; 2] {
    self.backend.system_sample(self)
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&[Array1<T>; 2]) -> R + Sync) -> Vec<R> {
    self.backend.system_paths_map(self, m, f)
  }

  fn sample_par(&self, m: usize) -> Vec<[Array1<T>; 2]> {
    self.backend.system_paths(self, m)
  }

  fn try_sample(&self) -> Result<[Array1<T>; 2], crate::device::DeviceError> {
    self.backend.try_system_sample(self)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<[Array1<T>; 2]>, crate::device::DeviceError> {
    self.backend.try_system_paths(self, m)
  }
}

/// Reusable dynamic-[`MultifactorSabr`] sampling state: the coefficient vectors and the seed
/// source, the sampler's own chunk basis, which the two [`Gn`] streams tick twice more per fill.
#[doc(hidden)]
#[derive(Debug)]
pub struct MultifactorSabrSampler<T: FloatExt, S: SeedExt> {
  n: usize,
  t: Option<T>,
  f0: T,
  alpha0: T,
  knots: Vec<T>,
  beta: Vec<T>,
  rho: Vec<T>,
  nu: Vec<T>,
  dt: T,
  seed: S,
}

impl<T: FloatExt, S: SeedExt> MultifactorSabrSampler<T, S> {
  /// Index of the active bucket at time `time`: the number of knots
  /// less than or equal to `time`, clamped to the last bucket.
  #[inline]
  fn bucket_at(&self, time: T) -> usize {
    let mut idx = 0usize;
    for &knot in &self.knots {
      if time >= knot {
        idx += 1;
      } else {
        break;
      }
    }
    idx.min(self.beta.len() - 1)
  }

  fn fill_paths(&mut self, f_: &mut [T], v: &mut [T]) {
    if self.n == 0 {
      return;
    }
    let dt = self.dt;

    // Two independent standard-normal increment streams (std-dev √dt),
    // combined per step under the time-varying correlation ρ(t_i).
    let gn1 = Gn::<T, S>::new(self.n - 1, self.t, self.seed.derive());
    let gn2 = Gn::<T, S>::new(self.n - 1, self.t, self.seed.derive());
    let z1 = gn1.sample();
    let z2 = gn2.sample();

    f_[0] = self.f0;
    v[0] = self.alpha0;

    for i in 1..self.n {
      let time = T::from_usize_(i - 1) * dt;
      let bucket = self.bucket_at(time);
      let beta = self.beta[bucket];
      let rho = self.rho[bucket];
      let nu = self.nu[bucket];

      let dw1 = z1[i - 1];
      // Correlate the second stream against the first at ρ(t_i). z1, z2
      // already carry the √dt scaling from `Gn`.
      let dw2 = rho * dw1 + (T::one() - rho * rho).sqrt() * z2[i - 1];

      let f_prev = f_[i - 1].max(T::zero());
      let v_prev = v[i - 1].max(T::zero());
      f_[i] = (f_[i - 1] + v_prev * f_prev.powf(beta) * dw1).max(T::zero());
      // Exact step for dα = ν α dW: α_i = α_{i-1} exp(ν ΔW₂ − ½ν²Δt).
      v[i] = v_prev * (nu * dw2 - T::from_f64_fast(0.5) * nu * nu * dt).exp();
    }
  }
}

impl<T: FloatExt, S: SeedExt> crate::traits::Sealed for MultifactorSabrSampler<T, S> {}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for MultifactorSabrSampler<T, S> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [f_, v] = out;
    self.fill_paths(
      f_.as_slice_mut()
        .expect("MultifactorSabr output must be contiguous"),
      v.as_slice_mut()
        .expect("MultifactorSabr output must be contiguous"),
    );
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    let mut f_ = Array1::<T>::zeros(self.n);
    let mut v = Array1::<T>::zeros(self.n);
    self.fill_paths(
      f_.as_slice_mut().expect("contiguous"),
      v.as_slice_mut().expect("contiguous"),
    );
    [f_, v]
  }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tabulation {
  use stochastic_rs_core::simd_rng::Unseeded;

  use super::MultifactorSabr;

  /// Curve entry `i` is the coefficient active at `(i - 1) dt`, the time the host's loop reads
  /// for step `i`. A statistical comparison cannot see a one-step phase error, hence the pin.
  #[test]
  fn the_curves_read_the_bucket_the_host_reads() {
    let p = MultifactorSabr::<f64>::new(
      Some(0.04),
      Some(0.3),
      vec![0.3, 0.7],
      vec![0.5, 0.7, 0.9],
      vec![-0.6, -0.2, 0.3],
      vec![0.8, 0.4, 0.2],
      64,
      Some(1.0),
      Unseeded,
    );
    let curves = <MultifactorSabr<f64> as crate::euler::EulerSystem<f64, 2>>::curves(&p)
      .expect("three curves");
    assert_eq!(curves.len(), 3);
    let dt = 1.0 / 63.0;
    for i in 1..64 {
      let bucket = p.bucket_at((i - 1) as f64 * dt);
      assert_eq!(curves[0][i], p.beta[bucket], "beta at step {i}");
      assert_eq!(curves[1][i], p.rho[bucket], "rho at step {i}");
      assert_eq!(curves[2][i], p.nu[bucket], "nu at step {i}");
    }
    // A grid point that lands exactly on a knot resolves to the same bucket
    // on both sides, since both evaluate the same expression.
    let on_knot = MultifactorSabr::<f64>::new(
      Some(0.04),
      Some(0.3),
      vec![0.5],
      vec![0.5, 0.9],
      vec![0.0, 0.0],
      vec![0.1, 0.1],
      11,
      Some(1.0),
      Unseeded,
    );
    let curves = <MultifactorSabr<f64> as crate::euler::EulerSystem<f64, 2>>::curves(&on_knot)
      .expect("curves");
    for i in 1..11 {
      assert_eq!(
        curves[0][i],
        on_knot.beta[on_knot.bucket_at((i - 1) as f64 * 0.1)]
      );
    }
  }
}

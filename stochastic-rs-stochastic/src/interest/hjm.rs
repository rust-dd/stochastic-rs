//! # Hjm
//!
//! $$
//! df(t,T)=\alpha(t,T)dt+\sigma(t,T)\,dW_t
//! $$
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::Fn1D;
use crate::traits::Fn2D;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// HJM-style Euler simulator: `r`, `p` and `f` are user-driven SDE components, and the
/// no-arbitrage drift restriction between `alpha` and `sigma` is not enforced.
#[derive(Debug, Clone)]
pub struct Hjm<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Time-dependent coefficient a(t) in the short rate's drift term
  /// `a(t)·dt`.
  pub a: Fn1D<T>,
  /// Time-dependent diffusion scale b(t) of the short rate: at step `i` it multiplies the
  /// N(0, dt) residual pre-filled into `r[i]`, just before the path value overwrites it.
  pub b: Fn1D<T>,
  /// Scale p(t,T) multiplying the whole drift-plus-diffusion bracket of the bond-price
  /// component; the model's `p`, not the `p` output array.
  pub p: Fn2D<T>,
  /// Time-dependent drift-rate function q(t,T) for the bond-price
  /// component (multiplied by `dt` inside `p`'s bracket).
  pub q: Fn2D<T>,
  /// Time-dependent diffusion scale v(t,T) of the bond price: it multiplies the residual
  /// pre-filled into `p[i]`, the role `b` plays for `r` and `sigma` for `f`.
  pub v: Fn2D<T>,
  /// Time-dependent forward-rate drift function α(t,T) (matches the
  /// module header's own α(t,T) — HJM drift of `f(t,T)`).
  pub alpha: Fn2D<T>,
  /// Time-dependent forward-rate volatility function σ(t,T) (matches the
  /// module header's own σ(t,T)).
  pub sigma: Fn2D<T>,
  /// Grid points sampled along the path; `dt = t / (n − 1)`.
  pub n: usize,
  /// Initial short-rate / interest-rate level.
  pub r0: Option<T>,
  /// Initial bond-price / auxiliary level.
  pub p0: Option<T>,
  /// Initial forward-rate level.
  pub f0: Option<T>,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Seed: [`Unseeded`] or a [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

/// Short rate, bond price and forward rate.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct HjmPaths<T> {
  pub r: Array1<T>,
  pub p: Array1<T>,
  pub f: Array1<T>,
}

impl<T> From<[Array1<T>; 3]> for HjmPaths<T> {
  fn from([r, p, f]: [Array1<T>; 3]) -> Self {
    Self { r, p, f }
  }
}

impl<T: FloatExt, S: SeedExt> Hjm<T, S> {
  /// Panics unless `n >= 2`, a given `t` is positive and finite, and every parameter is finite.
  pub fn new(
    a: impl Into<Fn1D<T>>,
    b: impl Into<Fn1D<T>>,
    p: impl Into<Fn2D<T>>,
    q: impl Into<Fn2D<T>>,
    v: impl Into<Fn2D<T>>,
    alpha: impl Into<Fn2D<T>>,
    sigma: impl Into<Fn2D<T>>,
    n: usize,
    r0: Option<T>,
    p0: Option<T>,
    f0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    assert!(
      r0.is_none_or(|v| v.is_finite()),
      "r0 must satisfy `r0.is_none_or(|v| v.is_finite())`, got r0 = {r0:?}"
    );
    assert!(
      p0.is_none_or(|v| v.is_finite()),
      "p0 must satisfy `p0.is_none_or(|v| v.is_finite())`, got p0 = {p0:?}"
    );
    assert!(
      f0.is_none_or(|v| v.is_finite()),
      "f0 must satisfy `f0.is_none_or(|v| v.is_finite())`, got f0 = {f0:?}"
    );
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    Self {
      backend: Cpu,
      a: a.into(),
      b: b.into(),
      p: p.into(),
      q: q.into(),
      v: v.into(),
      alpha: alpha.into(),
      sigma: sigma.into(),
      n,
      r0,
      p0,
      f0,
      t,
      seed,
    }
  }
}

/// The Euler engine's view: every `Fn2D` is called at the fixed horizon, so the host tabulates
/// six curves per grid point, the bond price's scale folded in; no callable reaches the device.
impl<T: FloatExt, S: SeedExt, B> Hjm<T, S, B> {
  /// The grid spacing the three rows share: `n - 1` increments over the
  /// horizon, and zero when the grid has no increment to take.
  fn dt(&self) -> T {
    if self.n > 1 {
      self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
    } else {
      T::zero()
    }
  }
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerSystem<T, 3>
  for Hjm<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::HeathJarrowMorton
  }

  fn initial_state(&self) -> [T; 4] {
    [
      self.r0.unwrap_or(T::zero()),
      self.p0.unwrap_or(T::zero()),
      self.f0.unwrap_or(T::zero()),
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

  /// The six coefficients at each grid point, in the step's order: `a`, `b`, `p·q`, `p·v`,
  /// `alpha`, `sigma`. Step `i` reads the value at `i · dt`, as the host's loop does.
  fn curves(&self) -> Option<Vec<Vec<T>>> {
    let dt = self.dt();
    let t_max = self.horizon();
    let grid = |f: &dyn Fn(T) -> T| (0..self.n).map(|i| f(T::from_usize_(i) * dt)).collect();
    Some(vec![
      grid(&|t| self.a.call(t)),
      grid(&|t| self.b.call(t)),
      grid(&|t| self.p.call(t, t_max) * self.q.call(t, t_max)),
      grid(&|t| self.p.call(t, t_max) * self.v.call(t, t_max)),
      grid(&|t| self.alpha.call(t, t_max)),
      grid(&|t| self.sigma.call(t, t_max)),
    ])
  }

  fn host_sample(&self) -> HjmPaths<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Hjm<T, S> { a, b, p, q, v, alpha, sigma, n, r0, p0, f0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Hjm<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Hjm<T, S, B> {
  type Output = HjmPaths<T>;
  type Sampler<'s>
    = HjmSampler<'s, T, S, B>
  where
    Self: 's;

  /// Derives one seed for [`HjmSampler`] to own; deriving, not cloning, decorrelates the chunks.
  /// `sample_inner`'s three `SimdNormal` streams take three ticks of it, in that order.
  fn sampler(&self) -> HjmSampler<'_, T, S, B> {
    HjmSampler {
      hjm: self,
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine: a device steps the three rows in its kernel under the tabulated
  /// coefficients, the host runs this process's own sampler, chunked as [`ProcessExt`] chunks.
  fn sample(&self) -> HjmPaths<T> {
    self.backend.system_sample(self)
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&HjmPaths<T>) -> R + Sync) -> Vec<R> {
    self.backend.system_paths_map(self, m, f)
  }

  fn sample_par(&self, m: usize) -> Vec<HjmPaths<T>> {
    self.backend.system_paths(self, m)
  }

  fn try_sample(&self) -> Result<HjmPaths<T>, crate::device::DeviceError> {
    self.backend.try_system_sample(self)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<HjmPaths<T>>, crate::device::DeviceError> {
    self.backend.try_system_paths(self, m)
  }
}

/// Reusable [`Hjm`] sampler: it borrows the process and owns a seed derived once, from which the
/// step body draws each component's Gaussian increments.
#[doc(hidden)]
#[derive(Debug)]
pub struct HjmSampler<'a, T: FloatExt, S: SeedExt, B> {
  hjm: &'a Hjm<T, S, B>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for HjmSampler<'_, T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> PathSampler<T>
  for HjmSampler<'_, T, S, B>
{
  type Output = HjmPaths<T>;

  fn sample_into(&mut self, out: &mut HjmPaths<T>) {
    *out = self.hjm.sample_inner(&self.seed);
  }

  fn sample(&mut self) -> HjmPaths<T> {
    self.hjm.sample_inner(&self.seed)
  }
}

impl<T: FloatExt, S: SeedExt, B> Hjm<T, S, B> {
  fn sample_inner(&self, seed: &S) -> HjmPaths<T> {
    let mut r = Array1::<T>::zeros(self.n);
    let mut p = Array1::<T>::zeros(self.n);
    let mut f_ = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return HjmPaths { r, p, f: f_ };
    }

    r[0] = self.r0.unwrap_or(T::zero());
    p[0] = self.p0.unwrap_or(T::zero());
    f_[0] = self.f0.unwrap_or(T::zero());
    if self.n == 1 {
      return HjmPaths { r, p, f: f_ };
    }

    let n_increments = self.n - 1;
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let sqrt_dt = dt.sqrt();
    {
      let r_slice = r
        .as_slice_mut()
        .expect("Hjm short-rate path must be contiguous in memory");
      let r_tail = &mut r_slice[1..];
      let mut normal_r = SimdNormal::<T>::new(T::zero(), sqrt_dt).seeded(seed);
      normal_r.fill_slice(r_tail);
    }
    {
      let p_slice = p
        .as_slice_mut()
        .expect("Hjm bond-price path must be contiguous in memory");
      let p_tail = &mut p_slice[1..];
      let mut normal_p = SimdNormal::<T>::new(T::zero(), sqrt_dt).seeded(seed);
      normal_p.fill_slice(p_tail);
    }
    {
      let f_slice = f_
        .as_slice_mut()
        .expect("Hjm forward-rate path must be contiguous in memory");
      let f_tail = &mut f_slice[1..];
      let mut normal_f = SimdNormal::<T>::new(T::zero(), sqrt_dt).seeded(seed);
      normal_f.fill_slice(f_tail);
    }

    let t_max = self.t.unwrap_or(T::one());

    for i in 1..self.n {
      let t = T::from_usize_(i) * dt;

      r[i] = r[i - 1] + self.a.call(t) * dt + self.b.call(t) * r[i];
      p[i] = p[i - 1]
        + self.p.call(t, t_max) * (self.q.call(t, t_max) * dt + self.v.call(t, t_max) * p[i]);
      f_[i] = f_[i - 1] + self.alpha.call(t, t_max) * dt + self.sigma.call(t, t_max) * f_[i];
    }

    HjmPaths { r, p, f: f_ }
  }
}

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
#[doc(hidden)]
pub use python::PyHjm;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tabulation {
  use stochastic_rs_core::simd_rng::Unseeded;

  use super::Hjm;

  /// Every coefficient is tabulated at `i · dt`, the host loop's time at step `i`, with the bond
  /// price's scale folded in. A statistical test cannot see a one-step shift, so this is exact.
  #[test]
  fn the_curves_are_the_host_coefficients_at_the_host_times() {
    fn a(t: f64) -> f64 {
      0.02 + 0.03 * t
    }
    fn b(t: f64) -> f64 {
      0.01 + 0.01 * t
    }
    fn p(t: f64, _m: f64) -> f64 {
      1.0 - 0.2 * t
    }
    fn q(t: f64, _m: f64) -> f64 {
      -0.05 + 0.02 * t
    }
    fn v(t: f64, _m: f64) -> f64 {
      0.03 + 0.02 * t
    }
    fn alpha(t: f64, _m: f64) -> f64 {
      0.01 * t
    }
    fn sigma(t: f64, _m: f64) -> f64 {
      0.02 + 0.01 * t
    }
    let h = Hjm::<f64>::new(
      a as fn(f64) -> f64,
      b as fn(f64) -> f64,
      p as fn(f64, f64) -> f64,
      q as fn(f64, f64) -> f64,
      v as fn(f64, f64) -> f64,
      alpha as fn(f64, f64) -> f64,
      sigma as fn(f64, f64) -> f64,
      41,
      Some(0.03),
      Some(1.0),
      Some(0.04),
      Some(2.0),
      Unseeded,
    );
    let curves = <Hjm<f64> as crate::euler::EulerSystem<f64, 3>>::curves(&h).expect("six curves");
    assert_eq!(curves.len(), 6);
    let (dt, t_max) = (2.0 / 40.0, 2.0);
    for i in 0..41 {
      let t = i as f64 * dt;
      assert_eq!(curves[0][i], a(t), "a at step {i}");
      assert_eq!(curves[1][i], b(t), "b at step {i}");
      assert_eq!(curves[2][i], p(t, t_max) * q(t, t_max), "p·q at step {i}");
      assert_eq!(curves[3][i], p(t, t_max) * v(t, t_max), "p·v at step {i}");
      assert_eq!(curves[4][i], alpha(t, t_max), "alpha at step {i}");
      assert_eq!(curves[5][i], sigma(t, t_max), "sigma at step {i}");
    }
  }
}

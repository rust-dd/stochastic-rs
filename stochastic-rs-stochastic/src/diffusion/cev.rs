//! # Cev
//!
//! $$
//! dS_t=\mu S_t\,dt+\sigma S_t^{\gamma}\,dW_t
//! $$
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

#[derive(Clone)]
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
  /// Initial value S₀ of the CEV path.
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on) or [`on_device`](Self::on_device).
  pub backend: B,
}

/// Every field has a matching `with_*` builder setter, e.g.
/// `Cev::default().with_gamma(0.5).with_sigma(0.3)`. No persisted cache:
/// `sampler()` builds its Gaussian stream fresh from `self` every call.
impl<T: FloatExt, S: SeedExt> Cev<T, S> {
  pub fn new(mu: T, sigma: T, gamma: T, n: usize, x0: Option<T>, t: Option<T>, seed: S) -> Self {
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
  /// Replace `mu`, all else unchanged.
  pub fn with_mu(mut self, mu: T) -> Self {
    self.mu = mu;
    self
  }

  /// Replace `sigma`, all else unchanged.
  pub fn with_sigma(mut self, sigma: T) -> Self {
    self.sigma = sigma;
    self
  }

  /// Replace `gamma`, all else unchanged.
  pub fn with_gamma(mut self, gamma: T) -> Self {
    self.gamma = gamma;
    self
  }

  /// Replace `x0`, all else unchanged.
  pub fn with_x0(mut self, x0: Option<T>) -> Self {
    self.x0 = x0;
    self
  }

  /// Replace the number of simulation steps `n`, all else unchanged.
  pub fn with_steps(mut self, n: usize) -> Self {
    self.n = n;
    self
  }

  /// Replace the simulation horizon `t`, all else unchanged.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    self.t = t;
    self
  }

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

/// μ=0.04, σ=0.2, γ=0.8, x₀=1 — a textbook Cev parameterization; γ<1 is
/// the usual equity-market calibration (see the `gamma` field doc). t=1,
/// n=252 — one trading year of daily steps (this crate's `Default`
/// convention).
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
    rand::Rng::random(&mut self.seed.rng())
  }

  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Cev<T, S> { mu, sigma, gamma, n, x0, t, seed } via euler);

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
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
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
pub struct CevSampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  mu: T,
  diff_scale: T,
  gamma: T,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> CevSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }
    let tail = &mut out[1..];
    self.normal.fill_slice(tail);
    let mut prev = self.x0;
    for z in tail.iter_mut() {
      // `prev.abs()` before the fractional power, matching every sibling
      // `sigma * |X|^p`-shaped diffusion in this module
      // ([`Ckls`](crate::diffusion::ckls::Ckls),
      // [`ThreeHalf`](crate::diffusion::three_half::ThreeHalf),
      // [`FellerRoot`](crate::diffusion::feller_root::FellerRoot)): Euler
      // discretization of `dS = mu*S*dt + sigma*S^gamma*dW` routinely pushes
      // `S` below zero for one step even at valid, realistic parameters
      // (`gamma < 1` is this field's own documented "usual equity-market
      // calibration"), and `f64::powf` of a negative base at a non-integer
      // exponent is `NaN` — which would then poison every subsequent step
      // through `prev`. Taking the elasticity off `|S|` instead keeps the
      // path finite through a boundary crossing rather than NaN-poisoning
      // it; it does not floor `S` itself at zero (unlike the CIR/Bessel
      // family), matching how the sibling files above handle the same
      // shape.
      let next =
        prev + self.mu * prev * self.dt + self.diff_scale * prev.abs().powf(self.gamma) * *z;
      *z = next;
      prev = next;
    }
  }
}

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
  /// Calculate the Malliavin derivative of the Cev process
  ///
  /// The Malliavin derivative of the Cev process is given by
  /// D_r S_t = \sigma S_t^{\gamma} * 1_{[0, r]}(r) exp(\int_0^r (\mu - \frac{\gamma^2 \sigma^2 S_u^{2\gamma - 2}}{2}) du + \int_0^r \gamma \sigma S_u^{\gamma - 1} dW_u)
  ///
  /// The Malliavin derivative of the Cev process shows the sensitivity of the stock price with respect to the Wiener process.
  pub fn malliavin(&self) -> [Array1<T>; 2] {
    let dt = if self.n > 1 {
      self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1)
    } else {
      T::zero()
    };
    let mut gn = Array1::<T>::zeros(self.n.saturating_sub(1));
    if let Some(gn_slice) = gn.as_slice_mut() {
      let sqrt_dt = dt.sqrt();
      let normal = SimdNormal::<T>::new(T::zero(), sqrt_dt, &self.seed);
      normal.fill_slice(gn_slice);
    }
    let cev = self.sample();

    let mut det_term = Array1::zeros(self.n);
    let mut stochastic_term = Array1::zeros(self.n);
    let mut m = Array1::zeros(self.n);

    for i in 0..self.n {
      // `cev[i].abs()` before every fractional power, for the same reason
      // as `fill_path`'s own `.abs()`: `sample()` no longer produces NaN
      // through a zero-crossing, but the realized path can still contain a
      // negative point, and raising that to a non-integer power here would
      // reintroduce the same `NaN` this method's own output should not have.
      det_term[i] = (self.mu
        - (self.gamma.powi(2)
          * self.sigma.powi(2)
          * cev[i]
            .abs()
            .powf(T::from_usize_(2) * self.gamma - T::from_usize_(2))
          / T::from_usize_(2)))
        * dt;
      if i > 0 {
        stochastic_term[i] =
          self.sigma * self.gamma * cev[i].abs().powf(self.gamma - T::one()) * gn[i - 1];
      }
      m[i] = self.sigma * cev[i].abs().powf(self.gamma) * (det_term[i] + stochastic_term[i]).exp();
    }

    [cev, m]
  }
}

py_process_1d!(PyCev, Cev,
  sig: (mu, sigma, gamma, n, x0=None, t=None, seed=None, dtype=None),
  params: (mu: f64, sigma: f64, gamma: f64, n: usize, x0: Option<f64>, t: Option<f64>)
);

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

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
    let [path, m] = cev.malliavin();
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

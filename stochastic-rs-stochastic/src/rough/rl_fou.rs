//! # Fractional Ou driven by Riemann–Liouville fBM (RFSV log-volatility)
//!
//! $$
//! dX_t = \kappa(\mu - X_t)\,dt + \nu\,dW^H_t,\qquad H \in (0, 1/2)
//! $$
//!
//! The RFSV log-volatility dynamics of Gatheral, Jaisson & Rosenbaum (2018).
//! The noise is RL-fBM generated non-cumulatively via the Bilokon–Wong
//! modified fast algorithm; the Ou drift is integrated with the usual Euler
//! rule on the resulting increments $\delta W^H_n = W^H_{t_{n+1}} - W^H_{t_n}$.
//!
//! Reference: Bayer C., Friz P., Gatheral J. *Pricing under rough volatility*.
//! Quantitative Finance 16 (2016), 887–904.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use super::markov_lift::MarkovLift;
use super::rl_fbm::RlFbm;
use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::noise::gn::Gn;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Fractional Ornstein–Uhlenbeck driven by RL-fBM noise.
#[derive(Clone)]
pub struct RlFou<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Hurst exponent of the driving fBM.
  pub hurst: T,
  /// Mean-reversion speed $\kappa$.
  pub kappa: T,
  /// Long-run mean $\mu$.
  pub mu: T,
  /// Diffusion scale $\nu$.
  pub nu: T,
  /// Number of simulation points.
  pub n: usize,
  /// Initial value; `None` starts at zero.
  pub x0: Option<T>,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Quadrature degree passed through to the underlying [`RlFbm`].
  pub degree: Option<usize>,
  /// Seed strategy.
  pub seed: S,
  fbm: RlFbm<T>,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> RlFou<T, S> {
  #[must_use]
  pub fn new(
    hurst: T,
    kappa: T,
    mu: T,
    nu: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    degree: Option<usize>,
    seed: S,
  ) -> Self {
    assert!(
      hurst.is_finite(),
      "hurst must satisfy `hurst.is_finite()`, got hurst = {hurst:?}"
    );
    assert!(
      kappa.is_finite(),
      "kappa must satisfy `kappa.is_finite()`, got kappa = {kappa:?}"
    );
    assert!(
      mu.is_finite(),
      "mu must satisfy `mu.is_finite()`, got mu = {mu:?}"
    );
    assert!(
      nu.is_finite(),
      "nu must satisfy `nu.is_finite()`, got nu = {nu:?}"
    );
    assert!(
      x0.is_none_or(|v| v.is_finite()),
      "x0 must satisfy `x0.is_none_or(|v| v.is_finite())`, got x0 = {x0:?}"
    );
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    Self {
      backend: Cpu,
      hurst,
      kappa,
      mu,
      nu,
      n,
      x0,
      t,
      degree,
      seed,
      fbm: RlFbm::new(hurst, n, t, degree, Unseeded),
    }
  }
}

/// The Euler engine's view: the lift the inner fBm carries goes to the device
/// as its node constants and boundary terms; the family steps the lifted fBm
/// in its second slot and this process's own recursion in the first.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for RlFou<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::RiemannLiouvilleOu {
      kappa: self.kappa,
      mu: self.mu,
      nu: self.nu,
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

  fn lift_spec(&self) -> Option<crate::euler::LiftSpec<'_, T>> {
    let lift = self.fbm.markov().lift();
    Some(crate::euler::LiftSpec {
      decay: lift.exp_neg_x_dt.as_slice().expect("contiguous"),
      weight: lift.we.as_slice().expect("contiguous"),
      drift_scale: lift.one_minus_e_over_x.as_slice().expect("contiguous"),
      drift_boundary: lift.drift_boundary,
      diffusion_boundary: lift.diffusion_boundary,
      x0: T::zero(),
    })
  }

  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] RlFou<T, S> { hurst, kappa, mu, nu, n, x0, t, degree, seed, fbm } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for RlFou<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for RlFou<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = RlFouSampler<T, S>
  where
    Self: 's;

  fn sampler(&self) -> RlFouSampler<T, S> {
    RlFouSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      kappa: self.kappa,
      mu: self.mu,
      nu: self.nu,
      dt: self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1),
      gn: Gn::<T, S> {
        backend: Cpu,
        n: self.n - 1,
        t: self.t,
        seed: self.seed.derive(),
      },
      markov: self.fbm.markov().clone(),
    }
  }

  /// Through the Euler engine: on a device the lift and the recursion run in
  /// the kernel, on the host devices it is this process's own sampler,
  /// chunked exactly as [`ProcessExt`] chunks.
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

/// Reusable [`RlFou`] sampling state: owns the cloned RL-fBM [`MarkovLift`]
/// stepper and the Gaussian-increment source, plus the precomputed Euler
/// scalars. `fill_path` regenerates the RL-fBM driver and Euler-integrates the
/// Ou drift in place; the owned `Gn` stream advances each call for independent
/// paths.
#[doc(hidden)]
pub struct RlFouSampler<T: FloatExt, S: SeedExt> {
  n: usize,
  x0: T,
  kappa: T,
  mu: T,
  nu: T,
  dt: T,
  gn: Gn<T, S>,
  markov: MarkovLift<T>,
}

impl<T: FloatExt, S: SeedExt> RlFouSampler<T, S> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    let dw = self.gn.sample();
    let fbm = self.markov.simulate(
      T::zero(),
      |_| T::zero(),
      |_| T::one(),
      dw.as_slice().expect("dw contiguous"),
    );

    out[0] = self.x0;
    for i in 1..out.len() {
      let dfbm = fbm[i] - fbm[i - 1];
      out[i] = out[i - 1] + self.kappa * (self.mu - out[i - 1]) * self.dt + self.nu * dfbm;
    }
  }
}

impl<T: FloatExt, S: SeedExt> crate::traits::Sealed for RlFouSampler<T, S> {}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for RlFouSampler<T, S> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("RlFou output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

#[cfg(test)]
mod tests {
  use ndarray::Array1;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_core::simd_rng::Unseeded;

  use super::RlFou;
  use crate::traits::ProcessExt;

  #[test]
  fn fou_nu_zero_matches_deterministic_euler() {
    let kappa = 1.3_f64;
    let mu = 0.8_f64;
    let n = 129;
    let x0 = 0.2_f64;
    let t = 1.0_f64;

    let p = RlFou::<f64>::new(0.3, kappa, mu, 0.0, n, Some(x0), Some(t), None, Unseeded);
    let x = p.sample();

    let dt = t / (n as f64 - 1.0);
    let mut expected = x0;
    for i in 1..n {
      expected = expected + kappa * (mu - expected) * dt;
      assert!((x[i] - expected).abs() < 1e-12, "mismatch at {i}");
    }
  }

  #[test]
  fn finite_output_at_typical_rfsv_parameters() {
    let p = RlFou::new(
      0.1_f64,
      2.0,
      0.15_f64.ln(),
      0.25,
      512,
      Some(0.15_f64.ln()),
      Some(1.0),
      None,
      Deterministic::new(7),
    );
    let x = p.sample();
    assert_eq!(x.len(), 512);
    assert!(x.iter().all(|v| v.is_finite()));
  }

  /// Guards the field-vs-doc contradiction fixed in A1-b: the module doc
  /// calls this quantity ν, so the field is `nu`. Larger ν must widen the
  /// simulated dispersion.
  #[test]
  fn rl_fou_nu_drives_dispersion() {
    let hurst = 0.3_f64;
    let kappa = 1.0_f64;
    let mu = 0.0_f64;
    let n = 256;
    let t = 1.0_f64;

    let small = RlFou::new(
      hurst,
      kappa,
      mu,
      0.05,
      n,
      Some(0.0),
      Some(t),
      None,
      Deterministic::new(42),
    );
    let large = RlFou::new(
      hurst,
      kappa,
      mu,
      0.5,
      n,
      Some(0.0),
      Some(t),
      None,
      Deterministic::new(42),
    );
    assert_eq!(small.nu, 0.05);
    assert_eq!(large.nu, 0.5);

    let variance = |x: &Array1<f64>| {
      let mean = x.iter().sum::<f64>() / x.len() as f64;
      x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64
    };

    assert!(
      variance(&large.sample()) > variance(&small.sample()),
      "larger nu must widen dispersion"
    );
  }
}

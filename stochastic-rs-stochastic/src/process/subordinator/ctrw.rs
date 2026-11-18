use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::alpha_stable::SimdAlphaStable;
use stochastic_rs_distributions::exp::SimdExp;
use stochastic_rs_distributions::gamma::SimdGamma;
use stochastic_rs_distributions::inverse_gauss::SimdInverseGauss;
use stochastic_rs_distributions::normal::SimdNormal;
use stochastic_rs_distributions::uniform::SimdUniform;

use super::sample_positive_stable;
use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Waiting-time distribution for Ctrw.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum CtrwWaitingLaw<T: FloatExt> {
  Exponential { rate: T },
  Gamma { shape: T, rate: T },
  InverseGaussian { mu: T, lambda: T },
  PositiveStable { alpha: T, scale: T },
}

/// Jump-size distribution for Ctrw.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum CtrwJumpLaw<T: FloatExt> {
  Normal { mean: T, std: T },
  SymmetricStable { alpha: T, scale: T },
  Rademacher { scale: T },
}

impl<T: FloatExt> CtrwWaitingLaw<T> {
  #[track_caller]
  fn assert_valid(&self) {
    match *self {
      CtrwWaitingLaw::Exponential { rate } => assert_positive("rate", rate),
      CtrwWaitingLaw::Gamma { shape, rate } => {
        assert_positive("shape", shape);
        assert_positive("rate", rate);
      }
      CtrwWaitingLaw::InverseGaussian { mu, lambda } => {
        assert_positive("mu", mu);
        assert_positive("lambda", lambda);
      }
      CtrwWaitingLaw::PositiveStable { alpha, scale } => {
        assert!(
          alpha > T::zero() && alpha < T::one(),
          "alpha must satisfy `alpha > T::zero() && alpha < T::one()`, got alpha = {alpha:?}"
        );
        assert_positive("scale", scale);
      }
    }
  }
}

impl<T: FloatExt> CtrwJumpLaw<T> {
  #[track_caller]
  fn assert_valid(&self) {
    match *self {
      CtrwJumpLaw::Normal { mean, std } => {
        assert!(
          mean.is_finite(),
          "mean must satisfy `mean.is_finite()`, got mean = {mean:?}"
        );
        assert_positive("std", std);
      }
      CtrwJumpLaw::SymmetricStable { alpha, scale } => {
        assert!(
          alpha > T::zero() && alpha <= T::from_usize_(2),
          "alpha must satisfy `alpha > T::zero() && alpha <= T::from_usize_(2)`, got alpha = {alpha:?}"
        );
        assert_positive("scale", scale);
      }
      CtrwJumpLaw::Rademacher { scale } => assert_positive("scale", scale),
    }
  }
}

#[track_caller]
fn assert_positive<T: FloatExt>(name: &str, value: T) {
  assert!(
    value > T::zero() && value.is_finite(),
    "{name} must satisfy `{name} > T::zero() && {name}.is_finite()`, got {name} = {value:?}"
  );
}

/// Continuous-time random walk sampled on a fixed output grid.
#[derive(Debug, Clone)]
pub struct Ctrw<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Inter-arrival-time (waiting-time) distribution between jumps.
  pub waiting: CtrwWaitingLaw<T>,
  /// Jump-size distribution applied at each arrival.
  pub jumps: CtrwJumpLaw<T>,
  /// Number of points sampled along the fixed output grid (the CTRW's own
  /// jump times are continuous and generally fall between grid points).
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

impl<T: FloatExt, S: SeedExt> Ctrw<T, S> {
  pub fn new(
    waiting: CtrwWaitingLaw<T>,
    jumps: CtrwJumpLaw<T>,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    assert!(
      x0.is_none_or(|v| v.is_finite()),
      "x0 must satisfy `x0.is_none_or(|v| v.is_finite())`, got x0 = {x0:?}"
    );
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    waiting.assert_valid();
    jumps.assert_valid();
    Self {
      backend: Cpu,
      waiting,
      jumps,
      n,
      x0,
      t,
      seed,
    }
  }
}

#[derive(Debug)]
enum WaitingSampler<T: FloatExt> {
  Exp(Seeded<SimdExp<T>>),
  Gamma(Seeded<SimdGamma<T>>),
  Ig(Seeded<SimdInverseGauss<T>>),
  PosStable { alpha: f64, log_scale: f64 },
}

#[derive(Debug)]
enum JumpSampler<T: FloatExt> {
  Normal(Seeded<SimdNormal<T>>),
  Stable(Seeded<SimdAlphaStable<T>>),
  Rademacher(T),
}

impl<T: FloatExt, S: SeedExt, B> Ctrw<T, S, B> {
  /// The grid spacing the host steps: the horizon over the increments, one
  /// increment at least.
  fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n.saturating_sub(1).max(1))
  }
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for Ctrw<T, S, B>
{
  /// A pure jump path: the additive jump-diffusion family with no drift and
  /// no diffusion, the step being the cell's jump sum alone.
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::AdditiveJumpDiffusion {
      drift_dt: T::zero(),
      sigma: T::zero(),
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

  fn time_step(&self) -> T {
    self.dt()
  }

  /// The exponential waits' rate: the arrivals in a cell of width `dt` are
  /// Poisson with mean `rate · dt`. Any other waiting law never reaches a
  /// launch — [`ProcessExt::sample`] keeps it on the host — so asking here is
  /// a caller bypassing that guard.
  fn jump_intensity(&self) -> Option<T> {
    match self.waiting {
      CtrwWaitingLaw::Exponential { rate } => Some(rate),
      _ => panic!(
        "Ctrw: only exponential waiting times run on a device; sample through `ProcessExt`, \
         which keeps the other laws on the host"
      ),
    }
  }

  fn jump_sizes(&self) -> Option<crate::euler::JumpSizes<T>> {
    Some(match self.jumps {
      CtrwJumpLaw::Normal { mean, std } => crate::euler::JumpSizes::Normal { mean, sd: std },
      CtrwJumpLaw::SymmetricStable { alpha, scale } => {
        crate::euler::JumpSizes::SymmetricStable { alpha, scale }
      }
      CtrwJumpLaw::Rademacher { scale } => crate::euler::JumpSizes::Rademacher { scale },
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

backend_switch!([T: FloatExt, S: SeedExt] Ctrw<T, S> { waiting, jumps, n, x0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Ctrw<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Ctrw<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = CtrwSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> CtrwSampler<T> {
    let x0 = self.x0.unwrap_or(T::zero());
    let n_increments = self.n.saturating_sub(1).max(1);
    let t_max = self.t.unwrap_or(T::one()).to_f64().unwrap();
    let dt = t_max / n_increments as f64;

    self.waiting.assert_valid();
    self.jumps.assert_valid();
    let waiting = match self.waiting {
      CtrwWaitingLaw::Exponential { rate } => {
        WaitingSampler::Exp(SimdExp::new(rate).seeded(&self.seed))
      }
      CtrwWaitingLaw::Gamma { shape, rate } => {
        WaitingSampler::Gamma(SimdGamma::<T>::new(shape, T::one() / rate).seeded(&self.seed))
      }
      CtrwWaitingLaw::InverseGaussian { mu, lambda } => {
        WaitingSampler::Ig(SimdInverseGauss::<T>::new(mu, lambda).seeded(&self.seed))
      }
      CtrwWaitingLaw::PositiveStable { alpha, scale } => WaitingSampler::PosStable {
        alpha: alpha.to_f64().unwrap(),
        log_scale: scale.to_f64().unwrap().ln(),
      },
    };

    let jumps = match self.jumps {
      CtrwJumpLaw::Normal { mean, std } => {
        JumpSampler::Normal(SimdNormal::new(mean, std).seeded(&self.seed))
      }
      CtrwJumpLaw::SymmetricStable { alpha, scale } => JumpSampler::Stable(
        SimdAlphaStable::<T>::new(alpha, T::zero(), scale, T::zero()).seeded(&self.seed),
      ),
      CtrwJumpLaw::Rademacher { scale } => JumpSampler::Rademacher(scale),
    };

    let uniform = SimdUniform::<f64>::new(0.0, 1.0).seeded(&self.seed);

    CtrwSampler {
      n: self.n,
      x0,
      dt,
      waiting,
      jumps,
      uniform,
    }
  }

  /// Through the Euler engine under exponential waits; any other waiting law
  /// keeps the process on the host, chunked exactly as [`ProcessExt`] chunks.
  fn sample(&self) -> Array1<T> {
    if self.device_ready() {
      self.backend.euler_sample(self)
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
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

  /// Exponential waits run on a device while a step expects at most 25 events, below the kernels'
  /// 64-event count; the other waiting laws are not memoryless and stay on the host.
  fn device_fallback(&self) -> Option<&'static str> {
    match self.waiting {
      CtrwWaitingLaw::Exponential { rate } => crate::euler::count_fallback(Some(rate), self.dt()),
      _ => Some("a waiting law that is not memoryless"),
    }
  }
}

/// Reusable [`Ctrw`] sampling state: the owned waiting-time and jump-size
/// distribution drivers plus the shared uniform source (used for positive-
/// stable waiting times and Rademacher jumps).
#[doc(hidden)]
#[derive(Debug)]
pub struct CtrwSampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: f64,
  waiting: WaitingSampler<T>,
  jumps: JumpSampler<T>,
  uniform: Seeded<SimdUniform<f64>>,
}

impl<T: FloatExt> CtrwSampler<T> {
  fn draw_wait(&mut self) -> f64 {
    match &mut self.waiting {
      WaitingSampler::Exp(d) => d.sample().to_f64().unwrap(),
      WaitingSampler::Gamma(d) => d.sample().to_f64().unwrap(),
      WaitingSampler::Ig(d) => d.sample().to_f64().unwrap(),
      WaitingSampler::PosStable { alpha, log_scale } => {
        sample_positive_stable(*alpha, *log_scale, &mut self.uniform)
      }
    }
  }

  fn draw_jump(&mut self) -> f64 {
    match &mut self.jumps {
      JumpSampler::Normal(d) => d.sample().to_f64().unwrap(),
      JumpSampler::Stable(d) => d.sample().to_f64().unwrap(),
      JumpSampler::Rademacher(scale) => {
        if self.uniform.sample() < 0.5 {
          scale.to_f64().unwrap()
        } else {
          -scale.to_f64().unwrap()
        }
      }
    }
  }

  /// Every event up to each grid time is added, with no cap: a law with very many events per step
  /// takes that long rather than returning a truncated path.
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = self.x0;
    if out.len() == 1 {
      return;
    }

    let mut x = self.x0.to_f64().unwrap();
    let mut next_event = self.draw_wait();

    for i in 1..out.len() {
      let t_i = i as f64 * self.dt;
      while next_event <= t_i {
        x += self.draw_jump();
        next_event += self.draw_wait();
      }
      out[i] = T::from_f64_fast(x);
    }
  }
}

impl<T: FloatExt> crate::traits::Sealed for CtrwSampler<T> {}

impl<T: FloatExt> PathSampler<T> for CtrwSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("Ctrw output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
#[doc(hidden)]
pub use python::PyCtrw;

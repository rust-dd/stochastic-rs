//! A Gaussian process under a causal kernel; fBM is `K(t,s) = (t-s)^(H-1/2) / Γ(H+1/2)`.
//!
//! $$
//! X_t = \int_0^t K(t,s)\,dW_s,\qquad
//! \mathrm{Cov}(X_t, X_u) = \int_0^{\min(t,u)} K(t,s)\,K(u,s)\,ds
//! $$
//!
//! With `H < 1/2` it takes [`VolterraSde`]'s lift, any other [`reference_path`]'s `O(n²)` sum.
//!
//! Reference: Decreusefond, Üstünel (1999), Stochastic Analysis of the Fractional Brownian Motion

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::rough::kernel::RlKernel;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;
use crate::volterra::reference::reference_path;
use crate::volterra::volterra_sde::VolterraSde;
use crate::volterra::volterra_sde::VolterraSdeSampler;

/// The kernel shapes [`Volterra`] takes, each parameter in the process's scalar; a shape, not a
/// [`VolterraKernel`](crate::volterra::VolterraKernel), which [`RlKernel`] is for $H < 1/2$.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum VolterraKernelSpec<T> {
  /// Fractional Brownian motion: $K(t,s) = (t-s)^{H-1/2}/\Gamma(H+1/2)$
  FractionalBM { h: T },
  /// Power-law: $K(t,s) = (t-s)^\gamma$ for $\gamma > -1/2$
  PowerLaw { gamma: T },
  /// Exponential: $K(t,s) = e^{-\beta(t-s)}$
  Exponential { beta: T },
}

impl<T: FloatExt> VolterraKernelSpec<T> {
  /// The kernel's parameter terms, evaluated once in `f64` ahead of the $O(n^2)$ convolution.
  fn prepare(&self) -> PreparedVolterraKernel<T> {
    match *self {
      VolterraKernelSpec::FractionalBM { h } => {
        let h = h.to_f64().unwrap();
        PreparedVolterraKernel::FractionalBM {
          exp: T::from_f64_fast(h - 0.5),
          gamma_val: T::from_f64_fast(stochastic_rs_distributions::special::gamma(h + 0.5)),
        }
      }
      VolterraKernelSpec::PowerLaw { gamma } => PreparedVolterraKernel::PowerLaw {
        gamma: T::from_f64_fast(gamma.to_f64().unwrap()),
      },
      VolterraKernelSpec::Exponential { beta } => PreparedVolterraKernel::Exponential {
        beta: T::from_f64_fast(beta.to_f64().unwrap()),
      },
    }
  }
}

/// [`VolterraKernelSpec`] with its parameter-derived terms evaluated once by
/// [`VolterraKernelSpec::prepare`] for every `(i, j)` pair of the sampling loop.
#[derive(Clone, Copy, Debug)]
enum PreparedVolterraKernel<T: FloatExt> {
  FractionalBM { exp: T, gamma_val: T },
  PowerLaw { gamma: T },
  Exponential { beta: T },
}

impl<T: FloatExt> PreparedVolterraKernel<T> {
  fn eval(&self, t: T, s: T) -> T {
    let tau = t - s;
    if tau <= T::zero() {
      return T::zero();
    }
    match self {
      PreparedVolterraKernel::FractionalBM { exp, gamma_val } => tau.powf(*exp) / *gamma_val,
      PreparedVolterraKernel::PowerLaw { gamma } => tau.powf(*gamma),
      PreparedVolterraKernel::Exponential { beta } => (-*beta * tau).exp(),
    }
  }
}

/// The engine a [`Volterra`] samples through, decided once from its [`VolterraKernelSpec`] at
/// construction.
#[derive(Clone, Debug)]
enum VolterraEngine<T: FloatExt> {
  /// $H \in (0, 1/2)$: an [`RlKernel`], driving
  /// [`VolterraSde`](crate::volterra::volterra_sde::VolterraSde) at $O(nN')$.
  Lift(RlKernel<T>),
  /// Everything else: the prepared closed-form kernel, driving the direct
  /// $O(n^2)$ convolution.
  Reference(PreparedVolterraKernel<T>),
}

/// Generic Volterra process with configurable kernel.
#[derive(Debug, Clone)]
pub struct Volterra<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Kernel specification.
  pub kernel: VolterraKernelSpec<T>,
  /// Number of grid points.
  pub n: usize,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Seed strategy (compile-time: `Unseeded` or `Deterministic`).
  pub seed: S,
  engine: VolterraEngine<T>,
  /// The Markov lift a device steps in the lift branch, built once here;
  /// `None` in the reference branch, which stays on the host.
  lift: Option<crate::rough::MarkovLift<T>>,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Volterra<T, S> {
  /// Builds the kernel's engine once. Panics unless `n >= 2`, a given `t` is positive and finite,
  /// and the kernel's parameter is finite with `0 < h < 1`, `gamma > -0.5` or `beta > 0`.
  pub fn new(kernel: VolterraKernelSpec<T>, n: usize, t: Option<T>, seed: S) -> Self {
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    match kernel {
      VolterraKernelSpec::FractionalBM { h } => {
        assert!(
          h > T::zero() && h < T::one(),
          "h must satisfy `h > T::zero() && h < T::one()`, got h = {h:?}"
        )
      }
      VolterraKernelSpec::PowerLaw { gamma } => {
        assert!(
          gamma.is_finite() && gamma > T::from_f64_fast(-0.5),
          "gamma must satisfy `gamma.is_finite() && gamma > T::from_f64_fast(-0.5)`, got gamma = {gamma:?}"
        )
      }
      VolterraKernelSpec::Exponential { beta } => {
        assert!(
          beta.is_finite() && beta > T::zero(),
          "beta must satisfy `beta.is_finite() && beta > T::zero()`, got beta = {beta:?}"
        )
      }
    }
    let engine = match kernel {
      VolterraKernelSpec::FractionalBM { h } if h < T::from_f64_fast(0.5) => {
        let degree = RlKernel::<T>::default_degree(n);
        VolterraEngine::Lift(RlKernel::<T>::new(h, degree))
      }
      _ => VolterraEngine::Reference(kernel.prepare()),
    };
    let lift = match &engine {
      VolterraEngine::Lift(rl) if n > 1 => Some(crate::rough::MarkovLift::new(
        rl.clone(),
        t.unwrap_or(T::one()) / T::from_usize_(n - 1),
      )),
      _ => None,
    };
    Self {
      backend: Cpu,
      kernel,
      n,
      t,
      seed,
      engine,
      lift,
    }
  }
}

impl<T: FloatExt> Volterra<T, Unseeded> {
  /// Fractional Brownian motion with Hurst parameter $H$.
  pub fn fbm(h: T, n: usize, t: Option<T>) -> Self {
    Self::new(VolterraKernelSpec::FractionalBM { h }, n, t, Unseeded)
  }
}

fn lift_zero<T: FloatExt>(_t: T, _x: T) -> T {
  T::zero()
}

fn lift_one<T: FloatExt>(_t: T, _x: T) -> T {
  T::one()
}

impl<T: FloatExt, S: SeedExt, B> Volterra<T, S, B> {
  /// The grid spacing, the horizon itself for a single point.
  fn dt(&self) -> T {
    let t = self.t.unwrap_or(T::one());
    if self.n > 1 {
      t / T::from_usize_(self.n - 1)
    } else {
      t
    }
  }
}

/// The Euler engine's view of both branches: fBm under the Markov lift, the family `RlFbm` rides,
/// and the reference convolution, run by the history block on the kernel tabulated on the grid.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for Volterra<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    match self.engine {
      VolterraEngine::Lift(_) => crate::euler::EulerSpec::RiemannLiouville,
      VolterraEngine::Reference(_) => crate::euler::EulerSpec::VolterraReference,
    }
  }

  /// The reference branch's kernel by lag, `K((m + 1) dt)`, exactly the
  /// weights the host's quadrature applies to the increments.
  fn curves(&self) -> Option<Vec<Vec<T>>> {
    match &self.engine {
      VolterraEngine::Lift(_) => None,
      VolterraEngine::Reference(prepared) => {
        let dt = self.dt();
        Some(vec![
          (0..self.n)
            .map(|m| prepared.eval(T::from_usize_(m + 1) * dt, T::zero()))
            .collect(),
        ])
      }
    }
  }

  fn initial_value(&self) -> T {
    T::zero()
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
    let lift = self.lift.as_ref()?.lift();
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

backend_switch!([T: FloatExt, S: SeedExt] Volterra<T, S> { kernel, n, t, seed, engine, lift } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Volterra<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for Volterra<T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = VolterraSampler<T, S>
  where
    Self: 's;

  fn sampler(&self) -> VolterraSampler<T, S> {
    match &self.engine {
      VolterraEngine::Lift(rl) => {
        let sde = VolterraSde::new(
          rl.clone(),
          lift_zero::<T> as fn(T, T) -> T,
          lift_one::<T> as fn(T, T) -> T,
          self.n,
          Some(T::zero()),
          self.t,
          self.seed.derive(),
        );
        VolterraSampler::Lift(sde.sampler())
      }
      VolterraEngine::Reference(prepared) => {
        let t_max = self.t.unwrap_or(T::one());
        let dt = t_max / T::from_usize_(self.n - 1);
        VolterraSampler::Reference(ReferenceVolterraSampler {
          kernel: *prepared,
          n: self.n,
          dt,
          sqrt_dt: dt.sqrt(),
          normal: SimdNormal::<T>::new(T::zero(), T::one()).seeded(&self.seed),
        })
      }
    }
  }

  /// Through the Euler engine: the lift or the reference convolution runs in the kernel; a
  /// reference grid longer than the kernels' history stays on the host's chunked sampler.
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

  /// What a device runs: the lift branch when it has a lift to replay (none on a single-point
  /// grid) and the reference branch when the grid fits the kernels' per-path history.
  fn device_fallback(&self) -> Option<&'static str> {
    let ready = match self.engine {
      VolterraEngine::Lift(_) => self.lift.is_some(),
      VolterraEngine::Reference(_) => self.n <= crate::euler::HISTORY_SLOTS,
    };
    (!ready)
      .then_some("a grid past the kernels' per-path history, or a lift with nothing to replay")
  }
}

/// Reusable [`Volterra`] sampling state, one variant per [`VolterraEngine`]
/// branch.
#[doc(hidden)]
#[non_exhaustive]
#[expect(
  clippy::large_enum_variant,
  reason = "a stream holds its 64-wide buffer inline; boxing it would allocate per sampler build"
)]
#[derive(Debug)]
pub enum VolterraSampler<T: FloatExt, S: SeedExt> {
  Lift(VolterraSdeSampler<T, RlKernel<T>, S>),
  Reference(ReferenceVolterraSampler<T>),
}

impl<T: FloatExt, S: SeedExt> crate::traits::Sealed for VolterraSampler<T, S> {}

impl<T: FloatExt, S: SeedExt> PathSampler<T> for VolterraSampler<T, S> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    match self {
      VolterraSampler::Lift(s) => s.sample_into(out),
      VolterraSampler::Reference(s) => s.sample_into(out),
    }
  }

  fn sample(&mut self) -> Array1<T> {
    match self {
      VolterraSampler::Lift(s) => s.sample(),
      VolterraSampler::Reference(s) => s.sample(),
    }
  }
}

/// The $O(n^2)$ direct-convolution sampler, $X_{t_i} = \sum_{j=1}^{i} K(t_i, t_{j-1})\,\Delta W_j$
/// through [`reference_path`]: the Gaussian source, the time step and the prepared kernel.
#[doc(hidden)]
#[derive(Debug)]
pub struct ReferenceVolterraSampler<T: FloatExt> {
  kernel: PreparedVolterraKernel<T>,
  n: usize,
  dt: T,
  sqrt_dt: T,
  normal: Seeded<SimdNormal<T>>,
}

impl<T: FloatExt> ReferenceVolterraSampler<T> {
  /// Draws `self.n` scaled Gaussian increments, of which `dw[0]` is unused, and hands
  /// `reference_path` the tail `&dw[1..]`: the draw count and order the pinned streams depend on.
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    out[0] = T::zero();
    if out.len() == 1 {
      return;
    }

    let mut dw = Array1::<T>::zeros(self.n);
    self.normal.fill_slice(dw.as_slice_mut().unwrap());
    for val in dw.iter_mut() {
      *val = *val * self.sqrt_dt;
    }

    let kernel = self.kernel;
    let path = reference_path(
      move |tau: T| kernel.eval(tau, T::zero()),
      lift_zero::<T>,
      lift_one::<T>,
      T::zero(),
      self.dt,
      &dw.as_slice().expect("dw must be contiguous")[1..],
    );
    out.copy_from_slice(path.as_slice().expect("reference path must be contiguous"));
  }
}

impl<T: FloatExt> crate::traits::Sealed for ReferenceVolterraSampler<T> {}

impl<T: FloatExt> PathSampler<T> for ReferenceVolterraSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("Volterra output must be contiguous");
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
pub use python::PyVolterra;

#[cfg(test)]
mod tests;

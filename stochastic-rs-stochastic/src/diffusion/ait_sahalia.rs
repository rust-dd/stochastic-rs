//! # AitSahalia
//!
//! $$
//! dX_t=\left(\frac{a_{-1}}{X_t}+a_0+a_1 X_t+a_2 X_t^2\right)dt+\sqrt{b_0+b_1 X_t+b_2 X_t^{b_3}}\,dW_t
//! $$
//!
//! Reference: Aït-Sahalia Y. (1996) — *Testing Continuous-Time Models of
//! the Spot Interest Rate*, Review of Financial Studies 9(2), 385–426,
//! DOI: 10.1093/rfs/9.2.385. This file's Euler-Maruyama discretisation
//! is a simulation scheme for the paper's nonlinear drift/CEV-type
//! diffusion family, not the paper's own (non-simulation) estimator.
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

#[derive(Clone, Copy)]
pub struct AitSahalia<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Inverse-state drift coefficient a₋₁ in `a₋₁/X_t + a_0 + a_1 X_t + a_2 X_t²`.
  pub am1: T,
  /// Constant drift coefficient a₀.
  pub a0: T,
  /// Linear drift coefficient a₁.
  pub a1: T,
  /// Quadratic drift coefficient a₂.
  pub a2: T,
  /// Constant term b₀ inside the square-rooted diffusion bracket
  /// `√(b_0 + b_1 X_t + b_2 X_t^{b_3})` (unlike
  /// [`NonLinearSDE`](super::nonlinear_sde::NonLinearSDE), this bracket
  /// **is** square-rooted before scaling `dW_t`).
  pub b0: T,
  /// Linear term b₁ inside the square-rooted diffusion bracket.
  pub b1: T,
  /// Power-law term b₂ scaling `X_t^{b_3}` inside the square-rooted
  /// diffusion bracket.
  pub b2: T,
  /// Diffusion exponent b₃ applied to `X_t` inside the square-rooted
  /// diffusion bracket.
  pub b3: T,
  /// Number of points sampled along the Aït-Sahalia path.
  pub n: usize,
  /// Initial value X₀ of the Aït-Sahalia path.
  pub x0: Option<T>,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: `Unseeded` or `Deterministic`).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> AitSahalia<T, S> {
  pub fn new(
    am1: T,
    a0: T,
    a1: T,
    a2: T,
    b0: T,
    b1: T,
    b2: T,
    b3: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    Self {
      backend: Cpu,
      am1,
      a0,
      a1,
      a2,
      b0,
      b1,
      b2,
      b3,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> AitSahalia<T, S, B> {}

/// The Euler engine's view of the Aït-Sahalia short-rate model.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for AitSahalia<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::AitSahalia {
      am1: self.am1,
      a0: self.a0,
      a1: self.a1,
      a2: self.a2,
      b0: self.b0,
      b1: self.b1,
      b2: self.b2,
      b3: self.b3,
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

backend_switch!([T: FloatExt, S: SeedExt] AitSahalia<T, S> { am1, a0, a1, a2, b0, b1, b2, b3, n, x0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for AitSahalia<T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = AitSahaliaSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> AitSahaliaSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    AitSahaliaSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      dt,
      am1: self.am1,
      a0: self.a0,
      a1: self.a1,
      a2: self.a2,
      b0: self.b0,
      b1: self.b1,
      b2: self.b2,
      b3: self.b3,
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

/// Reusable [`AitSahalia`] sampling state.
#[doc(hidden)]
pub struct AitSahaliaSampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  am1: T,
  a0: T,
  a1: T,
  a2: T,
  b0: T,
  b1: T,
  b2: T,
  b3: T,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> AitSahaliaSampler<T> {
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
      let safe_prev = if prev.abs() < T::from_f64_fast(1e-12) {
        T::from_f64_fast(1e-12)
      } else {
        prev
      };
      let drift = self.am1 / safe_prev + self.a0 + self.a1 * prev + self.a2 * prev * prev;
      let diff = (self.b0 + self.b1 * prev + self.b2 * prev.abs().powf(self.b3))
        .abs()
        .sqrt();
      let next = prev + drift * self.dt + diff * *z;
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for AitSahaliaSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("AitSahalia output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyAitSahalia, AitSahalia,
  sig: (am1, a0, a1, a2, b0, b1, b2, b3, n, x0=None, t=None, seed=None, dtype=None),
  params: (am1: f64, a0: f64, a1: f64, a2: f64, b0: f64, b1: f64, b2: f64, b3: f64, n: usize, x0: Option<f64>, t: Option<f64>),
  device
);

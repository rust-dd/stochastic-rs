//! BESQ(δ), then BES(δ), its square root in law. The third display is
//! [`Cir`](crate::diffusion::cir::Cir) (`theta` = κ, `mu` = θ, `sigma` = σ) as a scaled BESQ $Z$.
//!
//! $$
//! dX_t=\delta\,dt+2\sqrt{|X_t|}\,dW_t
//! $$
//!
//! $$
//! dX_t=\frac{\delta-1}{2X_t}\,dt+dW_t
//! $$
//!
//! $$
//! X^{\mathrm{CIR}}_t = e^{-\kappa t} Z(\tau(t)),\quad \delta = \frac{4\kappa\theta}{\sigma^2},
//! \quad \tau(t) = \frac{\sigma^2}{4\kappa}\bigl(e^{\kappa t}-1\bigr)
//! $$
//!
//! Reference: Revuz & Yor, "Continuous Martingales and Brownian Motion", Ch. XI §1

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::params::VarianceBoundary;
use crate::precondition::check_n;
use crate::precondition::check_t;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// BES(δ), sampled as the square root of the Euler BESQ(δ) recursion, since the drift
/// `(δ−1)/(2X)` is singular at zero; the law identity is exact, the scheme has BESQ's O(dt) bias.
#[derive(Debug, Clone)]
pub struct Bessel<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Dimension δ (see [`SquaredBessel::delta`] — the same δ ≥ 2 threshold
  /// keeps the process strictly positive).
  pub delta: T,
  /// Number of points sampled along the BES path.
  pub n: usize,
  /// Initial value; `None` starts at zero.
  pub x0: Option<T>,
  /// Horizon of the path; `None` is a unit horizon.
  pub t: Option<T>,
  /// Treatment of a negative square-root state after a step; the Feller condition is not required.
  pub boundary: VarianceBoundary,
  /// Seed strategy (compile-time: [`Unseeded`] or
  /// the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

/// Every field has a `with_*` setter (`Bessel::default().with_delta(4.0)`); nothing is cached,
/// since `sampler()` builds its Gaussian source from `self` on every call.
impl<T: FloatExt, S: SeedExt> Bessel<T, S> {
  /// Logs [`SquaredBessel::new`]'s δ < 2 warning under [`VarianceBoundary::Floor`]; panics unless
  /// `n >= 2`, `t` is positive when given, and every parameter is finite.
  pub fn new(
    delta: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    boundary: VarianceBoundary,
    seed: S,
  ) -> Self {
    Self::check_delta(delta);
    check_n(n);
    Self::check_x0(x0);
    check_t(t);
    if delta < T::from_usize_(2) && boundary != VarianceBoundary::Reflect {
      log::warn!(
        "Bessel::new: dimension below the strict-positivity \
         threshold (delta < 2) under VarianceBoundary::Floor; the path floors \
         at zero on every boundary hit instead of reflecting — pass \
         VarianceBoundary::Reflect for the standard sub-boundary mitigation"
      );
    }

    Self {
      backend: Cpu,
      delta,
      n,
      x0,
      t,
      boundary,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Bessel<T, S, B> {
  fn check_delta(delta: T) {
    require!(delta.is_finite(), delta);
  }

  fn check_x0(x0: Option<T>) {
    require!(x0.is_none_or(|v| v.is_finite()), x0);
  }

  /// Replace `delta`; panics unless `delta` is finite.
  pub fn with_delta(mut self, delta: T) -> Self {
    Self::check_delta(delta);
    self.delta = delta;
    self
  }

  /// Replace `x0`; panics unless `x0` is finite.
  pub fn with_x0(mut self, x0: Option<T>) -> Self {
    Self::check_x0(x0);
    self.x0 = x0;
    self
  }

  /// Replace `boundary`, all else unchanged.
  pub fn with_boundary(mut self, boundary: VarianceBoundary) -> Self {
    self.boundary = boundary;
    self
  }

  /// Replace `n`, the grid points along the path; panics unless `n >= 2`.
  pub fn with_points(mut self, n: usize) -> Self {
    check_n(n);
    self.n = n;
    self
  }

  /// Replace the simulation horizon `t`; panics unless `t`, when given, is positive and finite.
  pub fn with_horizon(mut self, t: Option<T>) -> Self {
    check_t(t);
    self.t = t;
    self
  }

  /// Replace the seed strategy's value, all else unchanged.
  pub fn with_seed(mut self, seed: S) -> Self {
    self.seed = seed;
    self
  }
}

/// δ=3, x₀=1, t=1, as [`SquaredBessel`]'s `Default` (δ≥2 keeps the process positive); 252 points, a
/// trading year of daily closes.
impl<T: FloatExt> Default for Bessel<T, Unseeded> {
  fn default() -> Self {
    Self::new(
      T::from_f64_fast(3.0),
      252,
      Some(T::one()),
      Some(T::one()),
      VarianceBoundary::Floor,
      Unseeded,
    )
  }
}

/// The Euler engine's view: the BESQ recursion on `X²`, reported as `√X²`, which keeps the
/// `(δ−1)/2X` singularity out. `X₀` is a radius, so the first reported point is `|X₀|`.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for Bessel<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let two = T::from_usize_(2);
    match self.boundary {
      VarianceBoundary::Floor => crate::euler::EulerSpec::BesselFromSquared {
        delta: self.delta,
        two,
      },
      VarianceBoundary::Reflect => crate::euler::EulerSpec::BesselFromSquaredReflected {
        delta: self.delta,
        two,
      },
    }
  }

  fn initial_value(&self) -> T {
    let x0 = self.x0.unwrap_or(T::zero());
    x0 * x0
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

backend_switch!([T: FloatExt, S: SeedExt] Bessel<T, S> { delta, n, x0, t, boundary, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Bessel<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Bessel<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = BesselSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> BesselSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let x0 = self.x0.unwrap_or(T::zero());
    BesselSampler {
      n: self.n,
      x0,
      dt,
      delta: self.delta,
      diff_scale: T::from_usize_(2),
      boundary: self.boundary,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt()).seeded(&self.seed),
    }
  }

  /// Through the Euler engine: a device runs the recursion in its kernel, the host runs this
  /// process's own sampler, chunked as `ProcessExt` chunks.
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

/// Reusable [`Bessel`] sampling state: the BESQ-recursion scale and the owned Gaussian source;
/// [`Bessel`]'s doc says why the step runs in squared space.
#[doc(hidden)]
#[derive(Debug)]
pub struct BesselSampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  delta: T,
  diff_scale: T,
  boundary: VarianceBoundary,
  normal: Seeded<SimdNormal<T>>,
}

impl<T: FloatExt> BesselSampler<T> {
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
    // `prev_sq` tracks X_t², never X_t, so the `(delta-1)/(2X)` singularity stays out of the
    // recursion; the path reports its square root.
    let mut prev_sq = self.x0 * self.x0;
    for z in tail.iter_mut() {
      let dbesq = self.delta * self.dt + self.diff_scale * prev_sq.abs().sqrt() * *z;
      let next_sq = match self.boundary {
        VarianceBoundary::Floor => (prev_sq + dbesq).max(T::zero()),
        VarianceBoundary::Reflect => (prev_sq + dbesq).abs(),
      };
      *z = next_sq.sqrt();
      prev_sq = next_sq;
    }
  }
}

impl<T: FloatExt> crate::traits::Sealed for BesselSampler<T> {}

impl<T: FloatExt> PathSampler<T> for BesselSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("Bessel output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyBessel, Bessel,
  sig: (delta, n, x0=None, t=None, boundary=None, seed=None, dtype=None),
  params: (delta: f64, n: usize, x0: Option<f64>, t: Option<f64>, boundary: Option<crate::python_params::PyVarianceBoundary> => crate::python_params::boundary),
  device
);

mod squared;

#[cfg(feature = "python")]
#[doc(hidden)]
pub use squared::PySquaredBessel;
pub use squared::SquaredBessel;
#[doc(hidden)]
pub use squared::SquaredBesselSampler;

#[cfg(test)]
mod tests;

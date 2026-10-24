//! # Cir 2f
//!
//! $$
//! r_t=x_t+y_t+\varphi(t),\ dx_t=\kappa_1(\theta_1-x_t)dt+\sigma_1\sqrt{x_t}dW_t^1,\ dy_t=\kappa_2(\theta_2-y_t)dt+\sigma_2\sqrt{y_t}dW_t^2
//! $$
//!
//! Two-additive-factor shift-extended CIR short-rate model (CIR2++):
//! two independent CIR factors (Cox, Ingersoll, Ross (1985),
//! DOI: 10.2307/1911242) summed and shifted by a deterministic `φ(t)`
//! fitted to today's term structure — the two-factor counterpart to
//! [`CirPlusPlus`](crate::interest::cir_pp::CirPlusPlus)'s single-factor
//! shift extension.
//!
//! Reference: Brigo D. & Mercurio F. (2006) — *Interest Rate Models —
//! Theory and Practice*, 2nd ed., Springer Finance,
//! DOI: 10.1007/978-3-540-34604-3.
//!

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use super::cir::Cir;
use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::diffusion::cir::CirSampler;
use crate::traits::FloatExt;
use crate::traits::Fn1D;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Two-additive-factor shift-extended CIR short-rate model `r = x + y + φ(t)`.
///
/// Fields are private so the shared-grid assertion of [`new`](Self::new) cannot be bypassed: the
/// getters hand the factors out read-only and `with_x` / `with_y` assert it again.
///
/// ```compile_fail,E0616
/// use stochastic_rs_core::simd_rng::Unseeded;
/// use stochastic_rs_stochastic::diffusion::cir::Cir;
/// use stochastic_rs_stochastic::interest::cir_2f::Cir2F;
/// let factor = |n| Cir::new(1.0_f64, 0.04, 0.2, n, Some(0.04), Some(1.0), Some(false), Unseeded);
/// let mut p = Cir2F::new(factor(10), factor(10), (|t: f64| t) as fn(f64) -> f64, Unseeded);
/// p.x.n = 1000;
/// ```
pub struct Cir2F<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  x: Cir<T, S>,
  y: Cir<T, S>,
  phi: Fn1D<T>,
  seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  backend: B,
}

impl<T: FloatExt, S: SeedExt> Cir2F<T, S> {
  /// `x` and `y` are taken pre-built (rather than absorbing their
  /// constructor parameters, the way `Merton`/`Kou`/`LevyDiffusion` absorbed
  /// `CompoundPoisson`'s) because each factor's own parameter set — κ, θ,
  /// σ, `x0`, `use_sym` — is independently meaningful and worth keeping
  /// addressable as a standalone [`Cir`], unlike a jump driver that is pure
  /// plumbing; flattening both factors' fields into one constructor would
  /// roughly double `Cir::new`'s own arity for no benefit. What `new` does
  /// take over is seeding: `x.seed` and `y.seed` are overwritten with two
  /// independent children derived from `seed` (`derive()`, never `clone()`,
  /// so the factors run mutually uncorrelated streams — see
  /// [`x`](Self::x) and [`y`](Self::y)), so the outer `seed` is the only
  /// seed that matters.
  pub fn new(mut x: Cir<T, S>, mut y: Cir<T, S>, phi: impl Into<Fn1D<T>>, seed: S) -> Self {
    Self::assert_same_grid(&x, &y);
    Self::seed_factors(&mut x, &mut y, &seed);
    Self {
      backend: Cpu,
      x,
      y,
      phi: phi.into(),
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Cir2F<T, S, B> {
  /// The factors are summed point by point, so they must share the grid.
  fn assert_same_grid(x: &Cir<T, S>, y: &Cir<T, S>) {
    assert_eq!(x.n, y.n, "x and y Cir factors must use the same n");
    if let (Some(tx), Some(ty)) = (x.t, y.t) {
      assert!(
        (tx - ty).abs() <= T::from_f64_fast(1e-12),
        "x and y Cir factors must use the same time horizon"
      );
    }
  }

  /// Gives each factor an independent child of `seed`, `x` first, so the
  /// outer seed is the only source of both factors' randomness.
  fn seed_factors(x: &mut Cir<T, S>, y: &mut Cir<T, S>, seed: &S) {
    x.seed = seed.derive();
    y.seed = seed.derive();
  }

  /// Replace the first factor; it takes over the outgoing factor's seed. Panics unless its grid
  /// matches `y`'s, as in `new()`; a new grid needs a new `Cir2F`.
  pub fn with_x(mut self, mut x: Cir<T, S>) -> Self {
    Self::assert_same_grid(&x, &self.y);
    x.seed = self.x.seed;
    self.x = x;
    self
  }

  /// Replace the second factor; it takes over the outgoing factor's seed. Panics unless its grid
  /// matches `x`'s, as in `new()`; a new grid needs a new `Cir2F`.
  pub fn with_y(mut self, mut y: Cir<T, S>) -> Self {
    Self::assert_same_grid(&self.x, &y);
    y.seed = self.y.seed;
    self.y = y;
    self
  }

  /// Replace the deterministic shift φ(t), all else unchanged.
  pub fn with_phi(mut self, phi: impl Into<Fn1D<T>>) -> Self {
    self.phi = phi.into();
    self
  }

  /// Replace the seed and re-derive both factors' seeds from it as `new()` does, so the result
  /// matches a fresh construction.
  pub fn with_seed(mut self, seed: S) -> Self {
    Self::seed_factors(&mut self.x, &mut self.y, &seed);
    self.seed = seed;
    self
  }

  /// First CIR factor `x_t` (κ₁, θ₁, σ₁). Its seed is a child of the outer `seed` drawn by
  /// [`new`](Self::new); the seed the [`Cir`] was built with is discarded.
  pub fn x(&self) -> &Cir<T, S> {
    &self.x
  }

  /// Second CIR factor `y_t` (κ₂, θ₂, σ₂). Its seed is the outer `seed`'s child drawn right after
  /// [`x`](Self::x)'s, so the two factors never share a stream.
  pub fn y(&self) -> &Cir<T, S> {
    &self.y
  }

  /// Deterministic shift φ(t) in `r_t = x_t + y_t + φ(t)`, fitted to the initial term structure
  /// (shift extension, as in CIR++).
  pub fn phi(&self) -> &Fn1D<T> {
    &self.phi
  }

  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  /// [`new`](Self::new) derives both factors' seeds from it, `x` first.
  pub fn seed(&self) -> &S {
    &self.seed
  }
}

/// The Euler engine's view of the two-factor CIR model: both factors step in
/// the kernel and the reported short rate is their shifted sum, which is the
/// first component the launch writes.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for Cir2F<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let flag = |sym: Option<bool>| {
      if sym.unwrap_or(false) {
        T::one()
      } else {
        T::zero()
      }
    };
    crate::euler::EulerSpec::TwoFactorSquareRoot {
      theta1: self.x.theta,
      mu1: self.x.mu,
      sigma1: self.x.sigma,
      theta2: self.y.theta,
      mu2: self.y.mu,
      sigma2: self.y.sigma,
      sym1: flag(self.x.use_sym),
      sym2: flag(self.y.use_sym),
    }
  }

  fn initial_value(&self) -> T {
    self.x.x0.unwrap_or(T::zero())
  }

  fn initial_state(&self) -> [T; 4] {
    [
      self.x.x0.unwrap_or(T::zero()),
      self.y.x0.unwrap_or(T::zero()),
      T::zero(),
      T::zero(),
    ]
  }

  fn grid_points(&self) -> usize {
    self.x.n
  }

  fn horizon(&self) -> T {
    self.x.t.unwrap_or(T::one())
  }

  /// The deterministic shift at each grid point.
  fn curve(&self) -> Option<Vec<T>> {
    let dt = self.horizon() / T::from_usize_(self.grid_points().saturating_sub(1).max(1));
    Some(
      (0..self.grid_points())
        .map(|i| self.phi.call(T::from_usize_(i) * dt))
        .collect(),
    )
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

backend_switch!([T: FloatExt, S: SeedExt] Cir2F<T, S> { x, y, phi, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Cir2F<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Cir2F<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = Cir2FSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> Cir2FSampler<T> {
    let n = self.x.n;
    let dt = self.x.t.unwrap_or(T::one()) / T::from_usize_(n - 1);
    let phi = Array1::<T>::from_shape_fn(n, |i| self.phi.call(T::from_usize_(i) * dt));
    Cir2FSampler {
      n,
      x: self.x.sampler(),
      y: self.y.sampler(),
      phi,
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

/// Reusable [`Cir2F`] sampling state — owns the two inner [`CirSampler`]s and
/// the precomputed deterministic `φ(t)` curve, so each call resamples both
/// factors and sums `x + y + φ`.
#[doc(hidden)]
pub struct Cir2FSampler<T: FloatExt> {
  n: usize,
  x: CirSampler<T>,
  y: CirSampler<T>,
  phi: Array1<T>,
}

impl<T: FloatExt> Cir2FSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    let x = self.x.sample();
    let y = self.y.sample();
    for ((dst, (&xi, &yi)), &p) in out
      .iter_mut()
      .zip(x.iter().zip(y.iter()))
      .zip(self.phi.iter())
    {
      *dst = xi + yi + p;
    }
  }
}

impl<T: FloatExt> crate::traits::Sealed for Cir2FSampler<T> {}

impl<T: FloatExt> PathSampler<T> for Cir2FSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("Cir2F output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  fn phi_fn(t: f64) -> f64 {
    t
  }

  #[test]
  fn default_time_horizon_is_one() {
    let x = Cir::new(0.0_f64, 0.0, 0.0, 3, Some(0.0), None, Some(false), Unseeded);
    let y = Cir::new(0.0_f64, 0.0, 0.0, 3, Some(0.0), None, Some(false), Unseeded);
    let model = Cir2F::new(x, y, phi_fn as fn(f64) -> f64, Unseeded);

    let out = model.sample();
    assert!((out[0] - 0.0).abs() < 1e-12);
    assert!((out[1] - 0.5).abs() < 1e-12);
    assert!((out[2] - 1.0).abs() < 1e-12);
  }

  #[test]
  #[should_panic(expected = "x and y Cir factors must use the same n")]
  fn mismatched_lengths_panic() {
    let x = Cir::new(
      0.0_f64,
      0.0,
      0.0,
      3,
      Some(0.0),
      Some(1.0),
      Some(false),
      Unseeded,
    );
    let y = Cir::new(
      0.0_f64,
      0.0,
      0.0,
      4,
      Some(0.0),
      Some(1.0),
      Some(false),
      Unseeded,
    );
    let _ = Cir2F::new(x, y, phi_fn as fn(f64) -> f64, Unseeded);
  }

  fn cir_pair(seed_x: u64, seed_y: u64) -> (Cir<f64, Deterministic>, Cir<f64, Deterministic>) {
    (
      Cir::new(
        1.0,
        0.03,
        0.1,
        32,
        Some(0.03),
        Some(1.0),
        Some(false),
        Deterministic::new(seed_x),
      ),
      Cir::new(
        1.2,
        0.02,
        0.1,
        32,
        Some(0.02),
        Some(1.0),
        Some(false),
        Deterministic::new(seed_y),
      ),
    )
  }

  /// Would fail if the fix were reverted: pre-fix, `Cir2F::sampler()` called
  /// `self.x.sampler()`/`self.y.sampler()` directly and never read
  /// `self.seed`, so the *sub*-`Cir`s' own seeds drove all of the output and
  /// the outer `Cir2F::new` seed argument was dead.
  #[test]
  fn outer_seed_is_authoritative_over_sub_seeds() {
    let (x1, y1) = cir_pair(7, 8);
    let a = Cir2F::new(x1, y1, phi_fn as fn(f64) -> f64, Deterministic::new(42)).sample();

    let (x2, y2) = cir_pair(7, 8);
    let b = Cir2F::new(
      x2,
      y2,
      phi_fn as fn(f64) -> f64,
      Deterministic::new(999_999),
    )
    .sample();
    assert_ne!(a, b, "changing only the outer seed must change the output");

    let (x3, y3) = cir_pair(12_345, 12_346);
    let c = Cir2F::new(x3, y3, phi_fn as fn(f64) -> f64, Deterministic::new(42)).sample();
    assert_eq!(
      a, c,
      "changing only the sub-Cir seeds must not change the output"
    );
  }

  /// Would fail if `new` derived one child seed and reused it for both `x`
  /// and `y` instead of deriving two: with identical Cir parameters on both
  /// factors, a shared stream would make the two factors' paths identical.
  #[test]
  fn factors_are_independent_streams() {
    let same_params = || {
      Cir::new(
        1.0_f64,
        0.04,
        0.2,
        64,
        Some(0.05),
        Some(1.0),
        Some(false),
        Deterministic::new(0),
      )
    };
    let model = Cir2F::new(
      same_params(),
      same_params(),
      phi_fn as fn(f64) -> f64,
      Deterministic::new(42),
    );

    assert_ne!(
      model.x.seed.current(),
      model.y.seed.current(),
      "x and y must receive independently derived seeds, not the same one twice"
    );

    let mut x_only = same_params();
    x_only.seed = model.x.seed.clone();
    let mut y_only = same_params();
    y_only.seed = model.y.seed.clone();
    assert_ne!(
      x_only.sample(),
      y_only.sample(),
      "x and y must be independent streams, not one stream reused twice"
    );
  }
}

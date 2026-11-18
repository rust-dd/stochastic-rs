use super::*;

/// BESQ(δ), `dX_t = δ dt + 2√|X_t| dW_t`: [`Bessel`] is its square root in law, and
/// [`Cir`](crate::diffusion::cir::Cir) a time-changed, scaled instance.
#[derive(Debug, Clone)]
pub struct SquaredBessel<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Dimension δ ≥ 0. δ ≥ 2 keeps the continuous-time process positive once away from 0
  /// (Going-Jaeschke & Yor, 2003), the analogue of `Cir`'s Feller condition.
  pub delta: T,
  /// Number of points sampled along the BESQ path.
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

/// Every field has a `with_*` setter (`SquaredBessel::default().with_delta(4.0)`); nothing is
/// cached, since `sampler()` builds its Gaussian source from `self` on every call.
impl<T: FloatExt, S: SeedExt> SquaredBessel<T, S> {
  /// δ < 2 is accepted and, under [`VarianceBoundary::Floor`], logged with `log::warn!`. Panics
  /// unless `n >= 2`, a given `t` is positive and finite, and `delta` and `x0` are finite.
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
        "SquaredBessel::new: dimension below the strict-positivity \
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

impl<T: FloatExt, S: SeedExt, B> SquaredBessel<T, S, B> {
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

/// δ=3, x₀=1, t=1, the closed-form mean tests' values (δ≥2 keeps the process positive); 252 points,
/// a trading year of daily closes.
impl<T: FloatExt> Default for SquaredBessel<T, Unseeded> {
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

/// The Euler engine's view: the BESQ recursion, reported as it stands. Reflection and truncation
/// are separate families, as for the square-root diffusions.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for SquaredBessel<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let two = T::from_usize_(2);
    match self.boundary {
      VarianceBoundary::Floor => crate::euler::EulerSpec::SquaredBesselState {
        delta: self.delta,
        two,
      },
      VarianceBoundary::Reflect => crate::euler::EulerSpec::SquaredBesselStateReflected {
        delta: self.delta,
        two,
      },
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

  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] SquaredBessel<T, S> { delta, n, x0, t, boundary, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for SquaredBessel<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for SquaredBessel<T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = SquaredBesselSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> SquaredBesselSampler<T> {
    let n_increments = self.n.saturating_sub(1).max(1);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    SquaredBesselSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
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

/// Reusable [`SquaredBessel`] sampling state: precomputed Euler scale and the
/// owned Gaussian source.
#[doc(hidden)]
#[derive(Debug)]
pub struct SquaredBesselSampler<T: FloatExt> {
  n: usize,
  x0: T,
  dt: T,
  delta: T,
  diff_scale: T,
  boundary: VarianceBoundary,
  normal: Seeded<SimdNormal<T>>,
}

impl<T: FloatExt> SquaredBesselSampler<T> {
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
      let dbesq = self.delta * self.dt + self.diff_scale * prev.abs().sqrt() * *z;
      let next = match self.boundary {
        VarianceBoundary::Floor => (prev + dbesq).max(T::zero()),
        VarianceBoundary::Reflect => (prev + dbesq).abs(),
      };
      *z = next;
      prev = next;
    }
  }
}

impl<T: FloatExt> crate::traits::Sealed for SquaredBesselSampler<T> {}

impl<T: FloatExt> PathSampler<T> for SquaredBesselSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("SquaredBessel output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PySquaredBessel, SquaredBessel,
  sig: (delta, n, x0=None, t=None, boundary=None, seed=None, dtype=None),
  params: (delta: f64, n: usize, x0: Option<f64>, t: Option<f64>, boundary: Option<crate::python_params::PyVarianceBoundary> => crate::python_params::boundary),
  device
);

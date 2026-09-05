//! # Hjm
//!
//! $$
//! df(t,T)=\alpha(t,T)dt+\sigma(t,T)\,dW_t
//! $$
//!

use ndarray::Array1;
#[cfg(feature = "python")]
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::Fn1D;
use crate::traits::Fn2D;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Hjm-style Euler simulator.
///
/// This implementation treats `r`, `p`, and `f` as user-driven SDE components and
/// does not enforce the no-arbitrage Hjm drift restriction between `alpha` and `sigma`.
pub struct Hjm<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Time-dependent coefficient a(t) in the short rate's drift term
  /// `a(t)·dt`.
  pub a: Fn1D<T>,
  /// Time-dependent diffusion-scale function b(t) for the short-rate
  /// component: `sample_inner` pre-fills `r[1..]` with fresh N(0, dt)
  /// residuals before the update loop runs, and at step `i` this
  /// multiplies the residual still sitting in `r[i]` just before that
  /// slot is overwritten in place with the real path value — the same
  /// pre-fill-then-consume pattern as e.g. `HullWhite`'s `diff_scale *
  /// *z`. A diffusion coefficient, not a level-multiplying term.
  pub b: Fn1D<T>,
  /// Time-dependent scaling function p(t,T) multiplying the entire
  /// bracketed drift-plus-diffusion term of the bond-price component
  /// (this is the model's own `p`, distinct from the `p` output array it
  /// helps compute).
  pub p: Fn2D<T>,
  /// Time-dependent drift-rate function q(t,T) for the bond-price
  /// component (multiplied by `dt` inside `p`'s bracket).
  pub q: Fn2D<T>,
  /// Time-dependent diffusion-scale function v(t,T) for the bond-price
  /// component: `sample_inner` pre-fills `p[1..]` with fresh N(0, dt)
  /// residuals before the update loop runs, and at step `i` this
  /// multiplies the residual still sitting in `p[i]` just before that
  /// slot is overwritten in place — the diffusion term inside `p`'s own
  /// bracket, the same role `b` plays for `r` and `sigma` plays for `f`.
  /// Not a level-multiplying term.
  pub v: Fn2D<T>,
  /// Time-dependent forward-rate drift function α(t,T) (matches the
  /// module header's own α(t,T) — HJM drift of `f(t,T)`).
  pub alpha: Fn2D<T>,
  /// Time-dependent forward-rate volatility function σ(t,T) (matches the
  /// module header's own σ(t,T)).
  pub sigma: Fn2D<T>,
  /// Number of time steps shared by all three output components
  /// (`r`, `p`, `f`).
  pub n: usize,
  /// Initial short-rate / interest-rate level.
  pub r0: Option<T>,
  /// Initial bond-price / auxiliary level.
  pub p0: Option<T>,
  /// Initial forward-rate level.
  pub f0: Option<T>,
  /// Horizon shared by the short-rate, bond-price and forward-rate paths
  /// (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Hjm<T, S> {
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

impl<T: FloatExt, S: SeedExt, B> Hjm<T, S, B> {}

/// The Euler engine's view: the seven coefficient functions are functions of
/// time alone — every `Fn2D` is called at the fixed horizon — so the host
/// tabulates them per grid point and the kernel steps three rows under six
/// curves, the bond price's outer scale folded into its drift and diffusion.
/// The callables themselves never reach the device.
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

  /// The six coefficients at each grid point, in the order the step names
  /// them: `a`, `b`, `p·q`, `p·v`, `alpha`, `sigma`. Step `i` reads the
  /// value at `i · dt`, which is the time the host's own loop evaluates.
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

  fn host_sample(&self) -> [Array1<T>; 3] {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Hjm<T, S> { a, b, p, q, v, alpha, sigma, n, r0, p0, f0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Hjm<T, S, B> {
  type Output = [Array1<T>; 3];
  type Sampler<'s>
    = HjmSampler<'s, T, S, B>
  where
    Self: 's;

  /// Derives a seed once, at construction, for [`HjmSampler`] to own.
  /// Deriving (not cloning) is what decorrelates chunks: the derived value
  /// is `self.seed`'s *mixed* next tick, not a raw snapshot, so chunk `i`'s
  /// basis and chunk `i+1`'s basis are hash-scrambled relative to each
  /// other rather than one raw stride apart. The three SDE components are
  /// driven by user-supplied [`Fn1D`] / [`Fn2D`] callables (not clonable,
  /// since the Python variant holds a `pyo3::Py`) so there is nothing else
  /// reusable to hoist across calls beyond the borrowed process itself;
  /// `sample_inner`'s three `SimdNormal::new(..., seed)` calls consume this
  /// owned seed directly — the same three ticks the legacy code consumed
  /// from `self.seed` per call, so the first path reproduces the legacy
  /// stream bit-for-bit.
  fn sampler(&self) -> HjmSampler<'_, T, S, B> {
    HjmSampler {
      hjm: self,
      seed: self.seed.derive(),
    }
  }

  /// Through the Euler engine: on a device the three rows are stepped in the
  /// kernel under the tabulated coefficients, on the host devices it is this
  /// process's own sampler, chunked exactly as [`ProcessExt`] chunks.
  fn sample(&self) -> [Array1<T>; 3] {
    self.backend.system_sample(self)
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&[Array1<T>; 3]) -> R + Sync) -> Vec<R> {
    self.backend.system_paths_map(self, m, f)
  }

  fn sample_par(&self, m: usize) -> Vec<[Array1<T>; 3]> {
    self.backend.system_paths(self, m)
  }

  fn try_sample(&self) -> Result<[Array1<T>; 3], crate::device::DeviceError> {
    self.backend.try_system_sample(self)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<[Array1<T>; 3]>, crate::device::DeviceError> {
    self.backend.try_system_paths(self, m)
  }
}

/// Reusable [`Hjm`] sampler: borrows the process and owns a seed derived
/// once at construction. Each SDE component's Gaussian increments are
/// generated inside the step body from that owned seed.
#[doc(hidden)]
pub struct HjmSampler<'a, T: FloatExt, S: SeedExt, B> {
  hjm: &'a Hjm<T, S, B>,
  seed: S,
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> PathSampler<T>
  for HjmSampler<'_, T, S, B>
{
  type Output = [Array1<T>; 3];

  fn sample_into(&mut self, out: &mut [Array1<T>; 3]) {
    *out = self.hjm.sample_inner(&self.seed);
  }

  fn sample(&mut self) -> [Array1<T>; 3] {
    self.hjm.sample_inner(&self.seed)
  }
}

impl<T: FloatExt, S: SeedExt, B> Hjm<T, S, B> {
  fn sample_inner(&self, seed: &S) -> [Array1<T>; 3] {
    let mut r = Array1::<T>::zeros(self.n);
    let mut p = Array1::<T>::zeros(self.n);
    let mut f_ = Array1::<T>::zeros(self.n);
    if self.n == 0 {
      return [r, p, f_];
    }

    r[0] = self.r0.unwrap_or(T::zero());
    p[0] = self.p0.unwrap_or(T::zero());
    f_[0] = self.f0.unwrap_or(T::zero());
    if self.n == 1 {
      return [r, p, f_];
    }

    let n_increments = self.n - 1;
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(n_increments);
    let sqrt_dt = dt.sqrt();
    {
      let r_slice = r
        .as_slice_mut()
        .expect("Hjm short-rate path must be contiguous in memory");
      let r_tail = &mut r_slice[1..];
      let normal_r = SimdNormal::<T>::new(T::zero(), sqrt_dt, seed);
      normal_r.fill_slice(r_tail);
    }
    {
      let p_slice = p
        .as_slice_mut()
        .expect("Hjm bond-price path must be contiguous in memory");
      let p_tail = &mut p_slice[1..];
      let normal_p = SimdNormal::<T>::new(T::zero(), sqrt_dt, seed);
      normal_p.fill_slice(p_tail);
    }
    {
      let f_slice = f_
        .as_slice_mut()
        .expect("Hjm forward-rate path must be contiguous in memory");
      let f_tail = &mut f_slice[1..];
      let normal_f = SimdNormal::<T>::new(T::zero(), sqrt_dt, seed);
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

    [r, p, f_]
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyHjm {
  inner: Option<Hjm<f64>>,
  seeded: Option<Hjm<f64, crate::simd_rng::Deterministic>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyHjm {
  #[new]
  #[pyo3(signature = (a, b, p, q, v, alpha, sigma, n, r0=None, p0=None, f0=None, t=None, seed=None))]
  fn new(
    a: pyo3::Py<pyo3::PyAny>,
    b: pyo3::Py<pyo3::PyAny>,
    p: pyo3::Py<pyo3::PyAny>,
    q: pyo3::Py<pyo3::PyAny>,
    v: pyo3::Py<pyo3::PyAny>,
    alpha: pyo3::Py<pyo3::PyAny>,
    sigma: pyo3::Py<pyo3::PyAny>,
    n: usize,
    r0: Option<f64>,
    p0: Option<f64>,
    f0: Option<f64>,
    t: Option<f64>,
    seed: Option<u64>,
  ) -> Self {
    use crate::traits::Fn2D;
    match seed {
      Some(s) => Self {
        inner: None,
        seeded: Some(Hjm::new(
          Fn1D::Py(a),
          Fn1D::Py(b),
          Fn2D::Py(p),
          Fn2D::Py(q),
          Fn2D::Py(v),
          Fn2D::Py(alpha),
          Fn2D::Py(sigma),
          n,
          r0,
          p0,
          f0,
          t,
          Deterministic::new(s),
        )),
      },
      None => Self {
        inner: Some(Hjm::new(
          Fn1D::Py(a),
          Fn1D::Py(b),
          Fn2D::Py(p),
          Fn2D::Py(q),
          Fn2D::Py(v),
          Fn2D::Py(alpha),
          Fn2D::Py(sigma),
          n,
          r0,
          p0,
          f0,
          t,
          Unseeded,
        )),
        seeded: None,
      },
    }
  }

  fn sample<'py>(
    &self,
    py: pyo3::Python<'py>,
  ) -> (
    pyo3::Py<pyo3::PyAny>,
    pyo3::Py<pyo3::PyAny>,
    pyo3::Py<pyo3::PyAny>,
  ) {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_dispatch_f64!(self, |inner| {
      let [a, b, c] = inner.sample();
      (
        a.into_pyarray(py).into_py_any(py).unwrap(),
        b.into_pyarray(py).into_py_any(py).unwrap(),
        c.into_pyarray(py).into_py_any(py).unwrap(),
      )
    })
  }

  /// `m` independent paths as three `(m, n)` arrays — short rate, bond
  /// price and forward rate. The GIL is released while the paths are
  /// generated; the Python callables re-acquire it per evaluation.
  fn sample_par<'py>(
    &self,
    py: pyo3::Python<'py>,
    m: usize,
  ) -> (
    pyo3::Py<pyo3::PyAny>,
    pyo3::Py<pyo3::PyAny>,
    pyo3::Py<pyo3::PyAny>,
  ) {
    use ndarray::Array2;
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    py_dispatch_f64!(self, |inner| {
      let paths = py.detach(|| inner.sample_par(m));
      let n = paths.first().map_or(0, |p| p[0].len());
      let mut stacked = [
        Array2::zeros((m, n)),
        Array2::zeros((m, n)),
        Array2::zeros((m, n)),
      ];
      for (i, [r, p, f]) in paths.iter().enumerate() {
        stacked[0].row_mut(i).assign(r);
        stacked[1].row_mut(i).assign(p);
        stacked[2].row_mut(i).assign(f);
      }
      let [r, p, f] = stacked;
      (
        r.into_pyarray(py).into_py_any(py).unwrap(),
        p.into_pyarray(py).into_py_any(py).unwrap(),
        f.into_pyarray(py).into_py_any(py).unwrap(),
      )
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn zero_1d(_: f64) -> f64 {
    0.0
  }

  fn zero_2d(_: f64, _: f64) -> f64 {
    0.0
  }

  fn one_2d(_: f64, _: f64) -> f64 {
    1.0
  }

  fn tmax_2d(_: f64, t_max: f64) -> f64 {
    t_max
  }

  #[test]
  fn default_t_max_is_one() {
    let model = Hjm::new(
      zero_1d as fn(f64) -> f64,
      zero_1d as fn(f64) -> f64,
      tmax_2d as fn(f64, f64) -> f64,
      one_2d as fn(f64, f64) -> f64,
      zero_2d as fn(f64, f64) -> f64,
      zero_2d as fn(f64, f64) -> f64,
      zero_2d as fn(f64, f64) -> f64,
      3,
      Some(0.0),
      Some(0.0),
      Some(0.0),
      None,
      Unseeded,
    );

    let [_r, p, _f] = model.sample();
    assert!((p[1] - 0.5).abs() < 1e-12);
    assert!((p[2] - 1.0).abs() < 1e-12);
  }
}

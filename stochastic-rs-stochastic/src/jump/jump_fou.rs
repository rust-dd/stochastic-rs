//! # Jump fOU
//!
//! $$
//! dX_t=\kappa(\theta-X_t)dt+\sigma dB_t^H+dJ_t
//! $$
//!
//! Composition of a fractional Ornstein-Uhlenbeck diffusion (Cheridito,
//! Kawaguchi, Maejima (2003), *Fractional Ornstein-Uhlenbeck Processes*,
//! Electronic Journal of Probability 8, paper 3, 1–14,
//! DOI: 10.1214/EJP.v8-125) with an additive, independent
//! compound-Poisson jump term `dJ_t` in the style of Merton (1976) —
//! *Option Pricing When Underlying Stock Returns Are Discontinuous*,
//! Journal of Financial Economics 3(1-2), 125–144,
//! DOI: 10.1016/0304-405X(76)90022-2. This exact combination (fOU base
//! plus an independent jump driver) is this crate's own composition
//! rather than a single named model from one paper.
//!
mod params;

use std::any::Any;

use ndarray::Array1;
use rand_distr::Distribution;
#[cfg(feature = "python")]
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::device::FgnBackend;
use crate::noise::fgn::Fgn;
use crate::process::cpoisson::CompoundPoisson;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Fully seed-reproducible: no exception to [`ProcessExt`]'s reproducibility
/// guarantee. Both halves consult `self.seed`, independently:
///
/// - The private `fgn: Fgn<T, Unseeded, B>` field is never consulted for its
///   own seed — `sampler()` builds a plain [`SimdNormal`] from
///   `self.seed.derive()` and borrows `fgn` only for its `Arc`-shared FFT
///   plan and eigenvalues, the same pattern
///   [`JumpFOUCustom`](crate::jump::jump_fou_custom::JumpFOUCustom) and
///   [`Fbm`](crate::process::fbm::Fbm) use for their own embedded,
///   permanently-`Unseeded` `fgn`. (This was fixable non-breakingly because
///   the field is private — rewiring how it's used carries no signature
///   change.)
/// - `cpoisson` is built internally by [`new`](Self::new) from `seed`,
///   exactly like [`Merton`](crate::jump::merton::Merton)'s field of the
///   same name — see [`cpoisson`](Self::cpoisson).
///
/// (This type was previously documented as a *full* exception — "no
/// randomness derives from `self.seed` at all" — on the grounds that both
/// halves were hard-wired away from it. That was correct about the values at
/// the time, but not about what was fixable: `fgn` needed only a private,
/// non-breaking rewire (done first); `cpoisson` needed the same breaking
/// widening `Merton`/`Kou`/`LevyDiffusion`/`Bates1996` needed, applied here
/// last.)
///
/// The fields are private because the embedded fGN driver caches a spectrum
/// derived from `hurst`, `n` and `t`, and the jump driver is derived from
/// `lambda`, `n`, `t` and `seed`: read the parameters through the getters and
/// change them through the `with_*` setters, which rebuild both. Assigning to
/// a field does not compile:
///
/// ```compile_fail,E0616
/// use stochastic_rs_core::simd_rng::Unseeded;
/// use stochastic_rs_distributions::scalar::ScalarNormal;
/// use stochastic_rs_stochastic::jump::jump_fou::JumpFou;
/// let law = ScalarNormal::<f64>::new(0.0, 0.1);
/// let mut p = JumpFou::new(0.7, 1.0, 0.0, 0.2, 2.0, law, 10, None, None, Unseeded);
/// p.n = 1000;
/// ```
pub struct JumpFou<T, D, S: SeedExt = Unseeded, B = Cpu>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  hurst: T,
  theta: T,
  mu: T,
  sigma: T,
  n: usize,
  x0: Option<T>,
  t: Option<T>,
  lambda: T,
  cpoisson: CompoundPoisson<T, D, S>,
  fgn: Fgn<T, Unseeded, B>,
  seed: S,
}

impl<T, D, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>> ProcessExt<T>
  for JumpFou<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
  type Output = Array1<T>;
  type Sampler<'s>
    = JumpFouSampler<'s, T, D, S, B>
  where
    Self: 's;

  /// Owns a Gaussian source derived from `self.seed` (not `fgn.sampler()`,
  /// which would build it from `fgn`'s own permanently-`Unseeded` field —
  /// see the type doc) and borrows `fgn` (for its `Arc`-shared FFT
  /// plan/eigenvalues only). Also owns a separate, independently-derived
  /// jump seed and borrows only the jump-size distribution — never a
  /// borrowed `&self.cpoisson` shared across chunks, which would let
  /// concurrent chunks race on the same shared atomic during the parallel
  /// region (see `ProcessExt`'s trait-level reproducibility requirement).
  fn sampler(&self) -> JumpFouSampler<'_, T, D, S, B> {
    JumpFouSampler {
      n: self.n,
      theta: self.theta,
      mu: self.mu,
      sigma: self.sigma,
      x0: self.x0.unwrap_or(T::zero()),
      dt: self.fgn.dt(),
      fgn: &self.fgn,
      normal: SimdNormal::<T>::new(T::zero(), T::one(), &self.seed.derive()),
      jump_distribution: &self.cpoisson.distribution,
      lambda: self.lambda,
      jump_seed: self.cpoisson.seed.derive(),
    }
  }

  /// Through the Euler engine when the jump-size law is one the device
  /// kernels draw; any other law keeps the process on the host, chunked
  /// exactly as [`ProcessExt`] chunks.
  fn sample(&self) -> Array1<T> {
    if self.device_ready() {
      crate::euler::EulerBackend::euler_sample(&self.fgn.backend, self)
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    if self.device_ready() {
      crate::euler::EulerBackend::euler_paths_map(&self.fgn.backend, self, m, f)
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
      crate::euler::EulerBackend::euler_paths_map_view(&self.fgn.backend, self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, |path| f(path.view()))
    }
  }

  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    if self.device_ready() {
      crate::euler::EulerBackend::euler_paths(&self.fgn.backend, self, m)
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<Array1<T>, crate::device::DeviceError> {
    if self.device_ready() {
      crate::euler::EulerBackend::try_sample(&self.fgn.backend, self)
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, crate::device::DeviceError> {
    if self.device_ready() {
      crate::euler::EulerBackend::try_euler_paths(&self.fgn.backend, self, m)
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }

  /// What a device runs here: it has no jumps, or its sizes
  /// follow a law the kernels draw. Anything else samples on the host.
  fn device_fallback(&self) -> Option<&'static str> {
    (!(self.lambda <= T::zero() || self.device_jump_sizes().is_some()))
      .then_some("a jump-size law the kernels do not draw")
  }
}

/// Reusable [`JumpFou`] sampling state: borrows `fgn` for its `Arc`-shared
/// FFT plan/eigenvalues and the jump-size distribution, and owns the
/// Gaussian source and a separately-derived jump seed, so a Monte-Carlo loop
/// pays the fGn `SimdNormal` setup once and reuses the FFT plan.
#[doc(hidden)]
pub struct JumpFouSampler<'a, T, D, S: SeedExt, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
  B: FgnBackend<T>,
{
  n: usize,
  theta: T,
  mu: T,
  sigma: T,
  x0: T,
  dt: T,
  fgn: &'a Fgn<T, Unseeded, B>,
  normal: SimdNormal<T>,
  jump_distribution: &'a D,
  lambda: T,
  jump_seed: S,
}

impl<T, D, S: SeedExt, B> JumpFouSampler<'_, T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
  B: FgnBackend<T>,
{
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }

    let mut fgn = Array1::<T>::zeros(self.fgn.out_len);
    self
      .fgn
      .fill_cpu(&mut self.normal, fgn.as_slice_mut().unwrap());
    let jump_increments = crate::process::cpoisson::grid_increments(
      self.jump_distribution,
      self.lambda,
      &self.jump_seed,
      out.len(),
      self.dt,
    );

    out[0] = self.x0;

    for i in 1..out.len() {
      out[i] = out[i - 1]
        + self.theta * (self.mu - out[i - 1]) * self.dt
        + self.sigma * fgn[i - 1]
        + jump_increments[i];
    }
  }
}

impl<T, D, S: SeedExt, B> PathSampler<T> for JumpFouSampler<'_, T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
  B: FgnBackend<T>,
{
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    self.fill_path(
      out
        .as_slice_mut()
        .expect("JumpFou output must be contiguous"),
    );
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

impl<T, D, S: SeedExt, B> JumpFou<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
  /// The jump-size law as the Euler engine's kernels draw it, `None` for a
  /// distribution they do not carry.
  fn device_jump_sizes(&self) -> Option<crate::euler::JumpSizes<T>> {
    crate::process::cpoisson::device_jump_sizes(&self.cpoisson.distribution)
  }
}

impl<T, D, S: SeedExt, B: FgnBackend<T> + crate::euler::EulerBackend<T>>
  crate::euler::EulerCoefficients<T> for JumpFou<T, D, S, B>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync + Any,
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::JumpFractionalOu {
      theta: self.theta,
      mu: self.mu,
      sigma: self.sigma,
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
    self.fgn.dt()
  }

  fn fgn_spec(&self) -> Option<crate::euler::FgnSpec<'_, T>> {
    Some(crate::euler::FgnSpec {
      sqrt_eigenvalues: self.fgn.sqrt_eigenvalues.as_slice().expect("contiguous"),
      n: self.fgn.n,
      offset: self.fgn.offset,
      hurst: self.fgn.hurst.to_f64().unwrap_or(0.5),
      t: self.fgn.t.unwrap_or(T::one()).to_f64().unwrap_or(1.0),
      streams: 1,
    })
  }

  /// The jump intensity per unit time; `None` when the process carries none,
  /// which is what makes a zero-intensity build skip the draw entirely.
  fn jump_intensity(&self) -> Option<T> {
    (self.lambda > T::zero()).then_some(self.lambda)
  }

  /// The size law as the kernels draw it. A law they do not carry never
  /// reaches a launch — [`ProcessExt::sample`] keeps such a process on the
  /// host — so asking for one here is a caller bypassing that guard.
  fn jump_sizes(&self) -> Option<crate::euler::JumpSizes<T>> {
    if self.lambda <= T::zero() {
      return None;
    }
    Some(self.device_jump_sizes().expect(
      "JumpFou: the jump-size distribution is not a law the Euler engine draws on a device; \
       sample through `ProcessExt`, which keeps it on the host",
    ))
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

backend_switch!([T, D, S: SeedExt] JumpFou<T, D, S> { hurst, theta, mu, sigma, n, x0, t, lambda, cpoisson, seed } via fgn euler
  where T: FloatExt, D: Distribution<T> + Send + Sync);

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyJumpFou {
  inner_f32: Option<JumpFou<f32, crate::traits::CallableDist<f32>>>,
  inner_f64: Option<JumpFou<f64, crate::traits::CallableDist<f64>>>,
  seeded_f32:
    Option<JumpFou<f32, crate::traits::CallableDist<f32>, crate::simd_rng::Deterministic>>,
  seeded_f64:
    Option<JumpFou<f64, crate::traits::CallableDist<f64>, crate::simd_rng::Deterministic>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyJumpFou {
  #[new]
  #[pyo3(signature = (hurst, theta, mu, sigma, distribution, lambda_, n, x0=None, t=None, seed=None, dtype=None))]
  fn new(
    hurst: f64,
    theta: f64,
    mu: f64,
    sigma: f64,
    distribution: pyo3::Py<pyo3::PyAny>,
    lambda_: f64,
    n: usize,
    x0: Option<f64>,
    t: Option<f64>,
    seed: Option<u64>,
    dtype: Option<&str>,
  ) -> pyo3::PyResult<Self> {
    stochastic_rs_distributions::python::value_error_on_panic(|| {
      let mut s = Self {
        inner_f32: None,
        inner_f64: None,
        seeded_f32: None,
        seeded_f64: None,
      };
      match dtype.unwrap_or("f64") {
        "f32" => {
          let jump_dist = crate::traits::CallableDist::new(distribution);
          match seed {
            Some(sd) => {
              s.seeded_f32 = Some(JumpFou::new(
                hurst as f32,
                theta as f32,
                mu as f32,
                sigma as f32,
                lambda_ as f32,
                jump_dist,
                n,
                x0.map(|v| v as f32),
                t.map(|v| v as f32),
                Deterministic::new(sd),
              ));
            }
            None => {
              s.inner_f32 = Some(JumpFou::new(
                hurst as f32,
                theta as f32,
                mu as f32,
                sigma as f32,
                lambda_ as f32,
                jump_dist,
                n,
                x0.map(|v| v as f32),
                t.map(|v| v as f32),
                Unseeded,
              ));
            }
          }
        }
        _ => {
          let jump_dist = crate::traits::CallableDist::new(distribution);
          match seed {
            Some(sd) => {
              s.seeded_f64 = Some(JumpFou::new(
                hurst,
                theta,
                mu,
                sigma,
                lambda_,
                jump_dist,
                n,
                x0,
                t,
                Deterministic::new(sd),
              ));
            }
            None => {
              s.inner_f64 = Some(JumpFou::new(
                hurst, theta, mu, sigma, lambda_, jump_dist, n, x0, t, Unseeded,
              ));
            }
          }
        }
      }
      s
    })
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use numpy::IntoPyArray;
      use pyo3::IntoPyObjectExt;

      use crate::traits::ProcessExt;
      py_dispatch!(self, |inner| inner
        .sample()
        .into_pyarray(py)
        .into_py_any(py)
        .unwrap())
    })
  }

  fn sample_par<'py>(
    &self,
    py: pyo3::Python<'py>,
    m: usize,
  ) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use numpy::IntoPyArray;
      use numpy::ndarray::Array2;
      use pyo3::IntoPyObjectExt;

      use crate::traits::ProcessExt;
      py_dispatch!(self, |inner| {
        let paths = inner.sample_par(m);
        let n = paths[0].len();
        let mut result = Array2::zeros((m, n));
        for (i, path) in paths.iter().enumerate() {
          result.row_mut(i).assign(path);
        }
        result.into_pyarray(py).into_py_any(py).unwrap()
      })
    })
  }
}

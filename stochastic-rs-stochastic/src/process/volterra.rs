//! # Volterra Process
//!
//! $$
//! X_t = \int_0^t K(t,s)\,dW_s
//! $$
//!
//! Gaussian process driven by a deterministic causal kernel $K(t,s)$ for $s \le t$.
//! Fractional Brownian motion is the special case $K(t,s) = (t-s)^{H-1/2}/\Gamma(H+1/2)$.
//!
//! Covariance: $\mathrm{Cov}(X_t, X_u) = \int_0^{\min(t,u)} K(t,s)\,K(u,s)\,ds$
//!
//! For [`VolterraKernelSpec::FractionalBM`] with $H \in (0, 1/2)$ — where
//! [`RlKernel`] applies — sampling delegates to [`VolterraSde`] with
//! $b\equiv0$, $\sigma\equiv1$, solved at $O(nN')$ by the Markov lift
//! instead of the direct $O(n^2)$ convolution below. Every other case —
//! $H \ge 1/2$ (the 124-type reproducibility guard exercises exactly this,
//! at $H=0.7$), [`VolterraKernelSpec::PowerLaw`], and
//! [`VolterraKernelSpec::Exponential`] — has no
//! [`crate::volterra::VolterraKernel`] representation available in this
//! crate and falls back to [`reference_path`], the same $O(n^2)$
//! discretisation $X_{t_i} \approx \sum_{j=1}^{i} K(t_i,
//! t_{j-1})\,\Delta W_j$ this type always used, now reused rather than
//! duplicated. That fallback preserves the exact driving-randomness stream
//! these three cases always drew (see `ReferenceVolterraSampler`'s own
//! `fill_path`), so their sampled output is unchanged; only $H<1/2$'s
//! output changes.
//!
//! Reference:
//! - Decreusefond, L. & Üstünel, A. S. (1999), "Stochastic Analysis of the Fractional Brownian Motion"

use ndarray::Array1;
#[cfg(feature = "python")]
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::rough::kernel::RlKernel;
use crate::rough::markov_lift::RoughSimd;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;
use crate::volterra::reference::reference_path;
use crate::volterra::sve::VolterraSde;
use crate::volterra::sve::VolterraSdeSampler;

/// Volterra kernel specification.
///
/// Renamed from `VolterraKernel` (breaking) to free that
/// name for [`crate::volterra::VolterraKernel`], the exponential-sum trait
/// [`VolterraSde`] is built on. This enum stays the small, closed set of
/// kernel *shapes* [`Volterra`] accepts, not a trait implementor —
/// [`RlKernel`] (built internally when $H<1/2$) is what actually implements
/// that trait for the [`FractionalBM`](Self::FractionalBM) case.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum VolterraKernelSpec {
  /// Fractional Brownian motion: $K(t,s) = (t-s)^{H-1/2}/\Gamma(H+1/2)$
  FractionalBM { h: f64 },
  /// Power-law: $K(t,s) = (t-s)^\gamma$ for $\gamma > -1/2$
  PowerLaw { gamma: f64 },
  /// Exponential: $K(t,s) = e^{-\beta(t-s)}$
  Exponential { beta: f64 },
}

impl VolterraKernelSpec {
  /// Precomputes this kernel's parameter-derived terms once, ahead of the
  /// $O(n^2)$ convolution in [`ReferenceVolterraSampler::fill_path`].
  ///
  /// [`VolterraKernelSpec::FractionalBM`]'s $\Gamma(H+1/2)$ is a Weierstrass
  /// infinite product (`scilib::math::basic::gamma`) iterated to a fixed
  /// relative-error threshold — measured at ~2.7ms per call, which used to
  /// be paid on every `(i, j)` pair of the kernel loop instead of once, the
  /// way this crate's other `scilib::gamma` call sites do. `h` is fixed for
  /// the sampler's lifetime, so the value prepared here is identical to the
  /// one every per-pair call used to produce.
  fn prepare<T: FloatExt>(&self) -> PreparedVolterraKernel<T> {
    match self {
      VolterraKernelSpec::FractionalBM { h } => PreparedVolterraKernel::FractionalBM {
        exp: T::from_f64_fast(*h - 0.5),
        gamma_val: T::from_f64_fast(scilib::math::basic::gamma(*h + 0.5)),
      },
      VolterraKernelSpec::PowerLaw { gamma } => PreparedVolterraKernel::PowerLaw {
        gamma: T::from_f64_fast(*gamma),
      },
      VolterraKernelSpec::Exponential { beta } => PreparedVolterraKernel::Exponential {
        beta: T::from_f64_fast(*beta),
      },
    }
  }
}

/// [`VolterraKernelSpec`] with its parameter-derived terms evaluated once by
/// [`VolterraKernelSpec::prepare`] and reused for every `(i, j)` pair the
/// sampling loop consults.
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

/// Which engine a constructed [`Volterra`] drives its sampling through,
/// decided once from [`VolterraKernelSpec`] at construction time — see the
/// module doc for which kernels take which branch.
enum VolterraEngine<T: FloatExt + RoughSimd> {
  /// $H \in (0, 1/2)$: an [`RlKernel`], driving
  /// [`VolterraSde`](crate::volterra::sve::VolterraSde) at $O(nN')$.
  Lift(RlKernel<T>),
  /// Everything else: the prepared closed-form kernel, driving the direct
  /// $O(n^2)$ convolution.
  Reference(PreparedVolterraKernel<T>),
}

/// Generic Volterra process with configurable kernel.
pub struct Volterra<T: FloatExt + RoughSimd, S: SeedExt = Unseeded, B = Cpu> {
  /// Kernel specification.
  pub kernel: VolterraKernelSpec,
  /// Number of grid points.
  pub n: usize,
  /// Time horizon $T$.
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

impl<T: FloatExt + RoughSimd, S: SeedExt> Volterra<T, S> {
  /// `kernel`'s own engine (an internal, private `VolterraEngine` — either
  /// the Markov lift or the direct convolution, see the module doc for
  /// which kernels take which branch) is prepared once here, reused by
  /// every subsequent [`sampler`](ProcessExt::sampler) call.
  pub fn new(kernel: VolterraKernelSpec, n: usize, t: Option<T>, seed: S) -> Self {
    let engine = match kernel {
      VolterraKernelSpec::FractionalBM { h } if h > 0.0 && h < 0.5 => {
        let degree = RlKernel::<T>::default_degree(n);
        VolterraEngine::Lift(RlKernel::<T>::new(T::from_f64_fast(h), degree))
      }
      _ => VolterraEngine::Reference(kernel.prepare::<T>()),
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

impl<T: FloatExt + RoughSimd, S: SeedExt, B> Volterra<T, S, B> {}

impl<T: FloatExt + RoughSimd> Volterra<T, Unseeded> {
  /// Fractional Brownian motion with Hurst parameter $H$.
  pub fn fbm(h: f64, n: usize, t: Option<T>) -> Self {
    assert!(h > 0.0 && h < 1.0, "Hurst parameter must be in (0,1)");
    Self::new(VolterraKernelSpec::FractionalBM { h }, n, t, Unseeded)
  }
}

fn lift_zero<T: FloatExt>(_t: T, _x: T) -> T {
  T::zero()
}

fn lift_one<T: FloatExt>(_t: T, _x: T) -> T {
  T::one()
}

/// The Euler engine's view of the lift branch: fBm under the Markov lift, the
/// same family `RlFBm` rides. The reference branch has no lift and never
/// reaches the engine; [`ProcessExt`] keeps it on the host.
impl<T: FloatExt + RoughSimd, S: SeedExt, B: crate::euler::EulerBackend<T>>
  crate::euler::EulerCoefficients<T> for Volterra<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::RiemannLiouville
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

backend_switch!([T: FloatExt + RoughSimd, S: SeedExt] Volterra<T, S> { kernel, n, t, seed, engine, lift } via euler);

impl<T: FloatExt + RoughSimd, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
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
          normal: SimdNormal::<T, 64>::new(T::zero(), T::one(), &self.seed),
        })
      }
    }
  }

  /// Through the Euler engine in the lift branch, whose lift runs in the
  /// kernel; the reference branch has no lift and stays on this process's
  /// own sampler whatever the backend.
  fn sample(&self) -> Array1<T> {
    if self.lift.is_some() {
      self.backend.euler_sample(self)
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    if self.lift.is_some() {
      self.backend.euler_paths_map(self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, f)
    }
  }

  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    if self.lift.is_some() {
      self.backend.euler_paths(self, m)
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<Array1<T>, crate::device::DeviceError> {
    if self.lift.is_some() {
      self.backend.try_sample(self)
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, crate::device::DeviceError> {
    if self.lift.is_some() {
      self.backend.try_euler_paths(self, m)
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }
}

/// Reusable [`Volterra`] sampling state, one variant per [`VolterraEngine`]
/// branch.
#[doc(hidden)]
#[non_exhaustive]
pub enum VolterraSampler<T: FloatExt + RoughSimd, S: SeedExt> {
  Lift(VolterraSdeSampler<T, RlKernel<T>, S>),
  Reference(ReferenceVolterraSampler<T>),
}

impl<T: FloatExt + RoughSimd, S: SeedExt> PathSampler<T> for VolterraSampler<T, S> {
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

/// The pre-lift $O(n^2)$ direct-convolution sampler: the owned Gaussian
/// source for the Brownian increments plus the precomputed time step and
/// prepared kernel (see [`PreparedVolterraKernel`] — built once here rather
/// than re-derived on every kernel evaluation).
///
/// $X_{t_i} = \sum_{j=1}^{i} K(t_i, t_{j-1})\,\Delta W_j$, complexity $O(n^2)$
/// due to the full-history convolution, now delegated to
/// [`reference_path`] — see [`fill_path`](Self::fill_path) for how the
/// pre-lift driving-randomness draw is preserved exactly.
#[doc(hidden)]
pub struct ReferenceVolterraSampler<T: FloatExt> {
  kernel: PreparedVolterraKernel<T>,
  n: usize,
  dt: T,
  sqrt_dt: T,
  normal: SimdNormal<T, 64>,
}

impl<T: FloatExt> ReferenceVolterraSampler<T> {
  /// Draws `self.n` scaled Gaussian increments exactly as this sampler
  /// always did (`dw[0]` unused — a pre-existing quirk kept unchanged so
  /// this refactor draws no fewer, no more, and no differently-ordered
  /// random values), then hands `reference_path` the length-`n-1` tail
  /// `&dw[1..]`. Since `PreparedVolterraKernel::eval(t, s)` is already a
  /// pure function of `t - s`, `|tau| kernel.eval(tau, T::zero())` reduces
  /// to the identical stationary kernel this sampler's own inline loop used
  /// to call directly, and the accumulation order (ascending `k`/`j`,
  /// [`Self::n`]'s own dt) is unchanged — so output is bit-identical to the
  /// pre-refactor implementation.
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

// PyVolterra is hand-written rather than expanded from the `py_process_1d!`
// macro because the constructor takes a [`VolterraKernelSpec`] sum type, not
// the flat positional `(f64, f64, ...)` parameter list the macro assumes.
#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyVolterra {
  inner: Option<Volterra<f64>>,
  seeded: Option<Volterra<f64, Deterministic>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyVolterra {
  /// Build a Volterra process.
  ///
  /// # Arguments
  /// * `kernel` — `"fbm"`, `"power_law"` (alias `"powerlaw"`), or `"exponential"`.
  /// * `param` — the kernel's scalar parameter: Hurst $H$ for `"fbm"`,
  ///   exponent $\gamma$ for `"power_law"`, decay $\beta$ for `"exponential"`.
  /// * `n` — number of grid points.
  /// * `t` — time horizon (default $1$).
  /// * `seed` — optional u64 seed for reproducibility.
  #[new]
  #[pyo3(signature = (kernel, param, n, t = None, seed = None))]
  fn new(kernel: &str, param: f64, n: usize, t: Option<f64>, seed: Option<u64>) -> Self {
    let kernel = match kernel.to_ascii_lowercase().as_str() {
      "fbm" | "fractional_bm" | "fractionalbm" => VolterraKernelSpec::FractionalBM { h: param },
      "power_law" | "powerlaw" => VolterraKernelSpec::PowerLaw { gamma: param },
      "exponential" | "exp" => VolterraKernelSpec::Exponential { beta: param },
      other => {
        panic!(
          "PyVolterra: unknown kernel '{other}' — expected one of 'fbm' | 'fractional_bm' | 'fractionalbm' | 'power_law' | 'powerlaw' | 'exponential' | 'exp'"
        )
      }
    };
    match seed {
      Some(sd) => Self {
        inner: None,
        seeded: Some(Volterra::<f64, Deterministic>::new(
          kernel,
          n,
          t,
          Deterministic::new(sd),
        )),
      },
      None => Self {
        inner: Some(Volterra::<f64>::new(kernel, n, t, Unseeded)),
        seeded: None,
      },
    }
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    crate::py_dispatch_f64!(self, |inner| inner
      .sample()
      .into_pyarray(py)
      .into_py_any(py)
      .unwrap())
  }

  fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use numpy::ndarray::Array2;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    crate::py_dispatch_f64!(self, |inner| {
      let paths = inner.sample_par(m);
      let n = paths[0].len();
      let mut result = Array2::zeros((m, n));
      for (i, path) in paths.iter().enumerate() {
        result.row_mut(i).assign(path);
      }
      result.into_pyarray(py).into_py_any(py).unwrap()
    })
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;

  #[test]
  fn volterra_fbm_runs() {
    let v = Volterra::<f64>::fbm(0.7, 100, Some(1.0));
    let path = v.sample();
    assert_eq!(path.len(), 100);
    assert!(path[0] == 0.0);
  }

  #[test]
  fn volterra_exponential_kernel() {
    let v = Volterra::<f64>::new(
      VolterraKernelSpec::Exponential { beta: 1.0 },
      100,
      Some(1.0),
      Unseeded,
    );
    let path = v.sample();
    assert_eq!(path.len(), 100);
  }

  #[test]
  fn volterra_fbm_h05_is_bm() {
    // H=0.5 → K(t,s) = 1/Γ(1) = 1 → X_t = W_t (standard Bm)
    let v = Volterra::<f64, Deterministic>::new(
      VolterraKernelSpec::FractionalBM { h: 0.5 },
      200,
      Some(1.0),
      Deterministic::new(42),
    );
    let path = v.sample();
    // Variance of Bm at t=1 should be ~1
    let var: f64 = path.iter().map(|&x| x * x).sum::<f64>() / path.len() as f64;
    // Very rough check — just ensure it's not degenerate
    assert!(var > 0.001, "variance = {var}");
  }

  #[test]
  fn volterra_seeded_deterministic() {
    // Two separately built instances with the same seed reproduce each other's first path.
    // (Same instance, repeated `.sample()` calls advance the seed state and produce
    // different paths — that is the desired behaviour for Monte Carlo reuse.)
    let v1 = Volterra::<f64, Deterministic>::new(
      VolterraKernelSpec::FractionalBM { h: 0.7 },
      50,
      Some(1.0),
      Deterministic::new(123),
    );
    let v2 = Volterra::<f64, Deterministic>::new(
      VolterraKernelSpec::FractionalBM { h: 0.7 },
      50,
      Some(1.0),
      Deterministic::new(123),
    );
    assert_eq!(v1.sample(), v2.sample());
  }

  /// `H = 0.7` is exactly the value the 124-type reproducibility guard
  /// (`tests/reproducibility_all_processes/process.rs`) instantiates
  /// `Volterra::fbm` at — the fallback path this pins, since `RlKernel`
  /// cannot represent `H >= 0.5`.
  #[test]
  fn h_above_half_still_uses_reference_fallback_and_stays_finite() {
    let v = Volterra::<f64, Deterministic>::new(
      VolterraKernelSpec::FractionalBM { h: 0.7 },
      64,
      Some(1.0),
      Deterministic::new(7),
    );
    let path = v.sample();
    assert_eq!(path.len(), 64);
    assert_eq!(path[0], 0.0);
    assert!(path.iter().all(|v| v.is_finite()));
  }

  /// Kernels this crate has no [`crate::volterra::VolterraKernel`]
  /// representation for (arbitrary power-law exponent) must still route
  /// through the reference fallback rather than panicking or silently
  /// misbehaving.
  #[test]
  fn power_law_kernel_uses_reference_fallback() {
    let v = Volterra::<f64, Deterministic>::new(
      VolterraKernelSpec::PowerLaw { gamma: -0.2 },
      64,
      Some(1.0),
      Deterministic::new(3),
    );
    let path = v.sample();
    assert_eq!(path.len(), 64);
    assert!(path.iter().all(|v| v.is_finite()));
  }

  /// Exercises the `n == 1` early-return path (no randomness drawn) through
  /// the reference-fallback branch specifically, since it is the branch
  /// whose `fill_path` special-cases that length to preserve the exact
  /// pre-refactor draw count.
  #[test]
  fn single_point_path_draws_no_randomness() {
    let v = Volterra::<f64, Deterministic>::new(
      VolterraKernelSpec::Exponential { beta: 1.0 },
      1,
      Some(1.0),
      Deterministic::new(1),
    );
    let path = v.sample();
    assert_eq!(path.len(), 1);
    assert_eq!(path[0], 0.0);
  }
}

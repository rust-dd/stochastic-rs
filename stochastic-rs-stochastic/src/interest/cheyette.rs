//! # Cheyette
//!
//! $$
//! dx_t = (y_t - \kappa x_t)\,dt + \sigma(t, x_t)\,dW_t,\qquad dy_t = \left(\sigma^2(t, x_t) - 2\kappa y_t\right)dt,\qquad r_t = f_0(t) + x_t
//! $$
//!
//! One-factor quasi-Gaussian HJM model with separable volatility (Cheyette
//! 1992; Andersen & Piterbarg 2010, Ch. 13). The whole forward curve is a
//! function of the two-dimensional Markov state `(x, y)`:
//! `f_t(T) = f_0(T) + e^{−κ(T−t)} (x_t + G(t, T) y_t)` with
//! `G(t, T) = (1 − e^{−κ(T−t)}) / κ`, and zero-coupon bonds reconstruct as
//! `P_t(T) = P_0(T) / P_0(t) · exp(−G(t, T) x_t − ½ G(t, T)² y_t)`. A constant
//! `σ` recovers Hull–White, where `y_t = σ² (1 − e^{−2κt}) / (2κ)` is the
//! deterministic variance of `x_t`; a state-dependent `σ(t, x)` (displaced or
//! CEV) adds the rate skew. The state is Euler–Maruyama-discretised from
//! `x_0 = y_0 = 0` with `σ` evaluated at the left grid point; the path output
//! is the pair `[x, y]`, from which [`Cheyette::short_rate`],
//! [`Cheyette::forward_rate`] and [`Cheyette::zero_bond`] rebuild the curve.
//!
//! References: Cheyette, O. (1992), *Markov Representation of the
//! Heath–Jarrow–Morton Model*, BARRA working paper; Andersen, L. B. G. &
//! Piterbarg, V. V. (2010), *Interest Rate Modeling*, Vol. II, Atlantic
//! Financial Press, Ch. 13; Gairat, A., Gorovoy, V. & Shcherbakov, V. (2025),
//! *Explicit local volatility formula for Cheyette-type interest rate models*,
//! arXiv:2506.23876, §2, eqs. (1)–(3).

use ndarray::Array1;
#[cfg(feature = "python")]
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::device::Cpu;
use crate::device::HostBackend;
use crate::traits::FloatExt;
use crate::traits::Fn1D;
use crate::traits::Fn2D;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Simpson panels used to integrate the initial forward curve for
/// [`Cheyette::initial_discount`].
const FORWARD_INTEGRATION_PANELS: usize = 256;

#[derive(Clone)]
pub struct Cheyette<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Initial instantaneous forward curve `f₀(t)`.
  pub f0: Fn1D<T>,
  /// Mean reversion κ of the state `x`.
  pub kappa: T,
  /// Local volatility `σ(t, x)` of the state `x`.
  pub sigma: Fn2D<T>,
  /// Number of grid points including the initial one.
  pub n: usize,
  /// Simulation horizon [0, t] (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on) or [`on_device`](Self::on_device).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Cheyette<T, S> {
  pub fn new(
    f0: impl Into<Fn1D<T>>,
    kappa: T,
    sigma: impl Into<Fn2D<T>>,
    n: usize,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert!(kappa > T::zero(), "kappa must be positive");
    Self {
      backend: Cpu,
      f0: f0.into(),
      kappa,
      sigma: sigma.into(),
      n,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Cheyette<T, S, B> {
  /// Time step `Δt = t / (n − 1)`.
  pub fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n.saturating_sub(1).max(1))
  }

  /// `G(t, T) = (1 − e^{−κ(T−t)}) / κ`.
  pub fn g(&self, t: T, maturity: T) -> T {
    (T::one() - (-self.kappa * (maturity - t)).exp()) / self.kappa
  }

  /// Short rate `r_t = f₀(t) + x_t`.
  pub fn short_rate(&self, t: T, x: T) -> T {
    self.f0.call(t) + x
  }

  /// Instantaneous forward `f_t(T) = f₀(T) + e^{−κ(T−t)} (x_t + G(t, T) y_t)`.
  pub fn forward_rate(&self, t: T, maturity: T, x: T, y: T) -> T {
    self.f0.call(maturity) + (-self.kappa * (maturity - t)).exp() * (x + self.g(t, maturity) * y)
  }

  /// Initial discount ratio `P₀(T) / P₀(t) = exp(−∫ₜᵀ f₀(u) du)` by Simpson
  /// quadrature of the forward curve.
  pub fn initial_discount(&self, t: T, maturity: T) -> T {
    if maturity <= t {
      return T::one();
    }
    let panels = FORWARD_INTEGRATION_PANELS;
    let h = (maturity - t) / T::from_usize_(panels);
    let mut sum = self.f0.call(t) + self.f0.call(maturity);
    for i in 1..panels {
      let weight = if i % 2 == 1 { 4.0 } else { 2.0 };
      sum += T::from_f64_fast(weight) * self.f0.call(t + T::from_usize_(i) * h);
    }
    (-(sum * h / T::from_f64_fast(3.0))).exp()
  }

  /// Zero-coupon bond `P_t(T) = P₀(T) / P₀(t) · exp(−G x_t − ½ G² y_t)` at
  /// state `(x, y)`.
  pub fn zero_bond(&self, t: T, maturity: T, x: T, y: T) -> T {
    let g = self.g(t, maturity);
    self.initial_discount(t, maturity) * (-g * x - T::from_f64_fast(0.5) * g * g * y).exp()
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Cheyette<T, S> { f0, kappa, sigma, n, t, seed } via host);

impl<T: FloatExt, S: SeedExt, B: HostBackend> ProcessExt<T> for Cheyette<T, S, B> {
  type Output = [Array1<T>; 2];
  type Sampler<'s>
    = CheyetteSampler<'s, T>
  where
    Self: 's;

  fn sampler(&self) -> CheyetteSampler<'_, T> {
    let dt = self.dt();
    CheyetteSampler {
      n: self.n,
      dt,
      kappa: self.kappa,
      sigma: &self.sigma,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }
}

/// Reusable [`Cheyette`] sampling state: borrows the local volatility and
/// owns the Gaussian source, so a Monte Carlo loop pays the setup once.
#[doc(hidden)]
pub struct CheyetteSampler<'a, T: FloatExt> {
  n: usize,
  dt: T,
  kappa: T,
  sigma: &'a Fn2D<T>,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> CheyetteSampler<'_, T> {
  /// Euler–Maruyama on `(x, y)` from the origin; `x` receives `N(0, Δt)`
  /// increments in place before the sweep.
  fn fill_state(&mut self, x: &mut [T], y: &mut [T]) {
    if x.is_empty() {
      return;
    }
    x[0] = T::zero();
    y[0] = T::zero();
    if x.len() == 1 {
      return;
    }
    let dt = self.dt;
    let two = T::from_f64_fast(2.0);
    let (mut xs, mut ys) = (T::zero(), T::zero());
    let tail = &mut x[1..];
    self.normal.fill_slice(tail);
    for (k, slot) in tail.iter_mut().enumerate() {
      let t_prev = T::from_usize_(k) * dt;
      let s = self.sigma.call(t_prev, xs);
      let x_next = xs + (ys - self.kappa * xs) * dt + s * *slot;
      let y_next = ys + (s * s - two * self.kappa * ys) * dt;
      *slot = x_next;
      y[k + 1] = y_next;
      xs = x_next;
      ys = y_next;
    }
  }
}

impl<T: FloatExt> PathSampler<T> for CheyetteSampler<'_, T> {
  type Output = [Array1<T>; 2];

  fn sample_into(&mut self, out: &mut [Array1<T>; 2]) {
    let [x, y] = out;
    let xs = x
      .as_slice_mut()
      .expect("Cheyette x output must be contiguous");
    let ys = y
      .as_slice_mut()
      .expect("Cheyette y output must be contiguous");
    assert_eq!(xs.len(), ys.len(), "x and y outputs must share a length");
    self.fill_state(xs, ys);
  }

  fn sample(&mut self) -> [Array1<T>; 2] {
    let mut out = [Array1::<T>::zeros(self.n), Array1::<T>::zeros(self.n)];
    self.sample_into(&mut out);
    out
  }
}

#[cfg(test)]
mod tests;

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyCheyette {
  inner: Option<Cheyette<f64>>,
  seeded: Option<Cheyette<f64, Deterministic>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyCheyette {
  /// `f0(t)` is the initial forward curve, `sigma(t, x)` the local
  /// volatility of the state; both are Python callables.
  #[new]
  #[pyo3(signature = (f0, kappa, sigma, n, t=None, seed=None))]
  fn new(
    f0: pyo3::Py<pyo3::PyAny>,
    kappa: f64,
    sigma: pyo3::Py<pyo3::PyAny>,
    n: usize,
    t: Option<f64>,
    seed: Option<u64>,
  ) -> Self {
    match seed {
      Some(s) => Self {
        inner: None,
        seeded: Some(Cheyette::new(
          Fn1D::Py(f0),
          kappa,
          Fn2D::Py(sigma),
          n,
          t,
          Deterministic::new(s),
        )),
      },
      None => Self {
        inner: Some(Cheyette::new(
          Fn1D::Py(f0),
          kappa,
          Fn2D::Py(sigma),
          n,
          t,
          Unseeded,
        )),
        seeded: None,
      },
    }
  }

  /// One path of the state as the pair `(x, y)` of arrays.
  fn sample<'py>(&self, py: pyo3::Python<'py>) -> (pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>) {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;
    crate::py_dispatch_f64!(self, |inner| {
      let [x, y] = inner.sample();
      (
        x.into_pyarray(py).into_py_any(py).unwrap(),
        y.into_pyarray(py).into_py_any(py).unwrap(),
      )
    })
  }

  /// `m` paths as a pair of `(m, n)` arrays. Each step calls back into the
  /// Python `f0` / `sigma` callables under the GIL, so the parallel speed-up
  /// is bounded by those calls.
  fn sample_par<'py>(
    &self,
    py: pyo3::Python<'py>,
    m: usize,
  ) -> (pyo3::Py<pyo3::PyAny>, pyo3::Py<pyo3::PyAny>) {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;
    crate::py_dispatch_f64!(self, |inner| {
      // The callbacks re-attach to the interpreter from the worker threads, so
      // the GIL must be released here or the parallel sampler deadlocks.
      let samples = py.detach(|| inner.sample_par(m));
      let mut xs = ndarray::Array2::<f64>::zeros((m, inner.n));
      let mut ys = ndarray::Array2::<f64>::zeros((m, inner.n));
      for (i, [x, y]) in samples.iter().enumerate() {
        xs.row_mut(i).assign(x);
        ys.row_mut(i).assign(y);
      }
      (
        xs.into_pyarray(py).into_py_any(py).unwrap(),
        ys.into_pyarray(py).into_py_any(py).unwrap(),
      )
    })
  }

  /// Short rate `f0(t) + x`.
  fn short_rate(&self, t: f64, x: f64) -> f64 {
    crate::py_dispatch_f64!(self, |inner| inner.short_rate(t, x))
  }

  /// Zero-coupon bond `P_t(T)` at state `(x, y)`.
  fn zero_bond(&self, t: f64, maturity: f64, x: f64, y: f64) -> f64 {
    crate::py_dispatch_f64!(self, |inner| inner.zero_bond(t, maturity, x, y))
  }
}

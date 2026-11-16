use super::*;

/// `theta_r` is a Python callable of time; `rho` a `4 × 4` array in the order `(x, v, r, u)`; `scheme`
/// is `"euler"` (default) or `"qe"`, Andersen's (2008) quadratic-exponential variance step.
#[doc(hidden)]
#[pyo3::prelude::pyclass]
pub struct PyHestonHullWhite2F {
  inner_f32: Option<HestonHullWhite2F<f32>>,
  inner_f64: Option<HestonHullWhite2F<f64>>,
  seeded_f32: Option<HestonHullWhite2F<f32, Euler, crate::python_device::SharedSeed>>,
  seeded_f64: Option<HestonHullWhite2F<f64, Euler, crate::python_device::SharedSeed>>,
  qe: Option<PyHestonHullWhite2FQe>,
}

/// The [`AndersenQe`] twins of [`PyHestonHullWhite2F`]'s slots, named for the dispatch macro.
struct PyHestonHullWhite2FQe {
  inner_f32: Option<HestonHullWhite2F<f32, AndersenQe>>,
  inner_f64: Option<HestonHullWhite2F<f64, AndersenQe>>,
  seeded_f32: Option<HestonHullWhite2F<f32, AndersenQe, crate::python_device::SharedSeed>>,
  seeded_f64: Option<HestonHullWhite2F<f64, AndersenQe, crate::python_device::SharedSeed>>,
}

macro_rules! py_hybrid_dispatch {
  ($self:expr, |$p:ident| $body:expr) => {
    match &$self.qe {
      Some(qe) => $crate::py_dispatch!(qe, |$p| $body),
      None => $crate::py_dispatch!($self, |$p| $body),
    }
  };
}

fn build<T: FloatExt, S: SeedExt>(
  args: &Args,
  theta_r: pyo3::Py<pyo3::PyAny>,
  seed: S,
) -> HestonHullWhite2F<T, Euler, S> {
  let f = |x: f64| T::from_f64_fast(x);
  HestonHullWhite2F::new(
    f(args.kappa),
    f(args.theta),
    f(args.sigma),
    Fn1D::Py(theta_r),
    f(args.a),
    f(args.sigma1),
    f(args.sigma2),
    f(args.b),
    args.rho.mapv(f),
    args.n,
    args.s0.map(f),
    args.v0.map(f),
    args.r0.map(f),
    args.t.map(f),
    seed,
  )
}

/// The scalar arguments of the constructor, carried once for both widths.
struct Args {
  kappa: f64,
  theta: f64,
  sigma: f64,
  a: f64,
  sigma1: f64,
  sigma2: f64,
  b: f64,
  rho: Array2<f64>,
  n: usize,
  s0: Option<f64>,
  v0: Option<f64>,
  r0: Option<f64>,
  t: Option<f64>,
}

#[pyo3::prelude::pymethods]
impl PyHestonHullWhite2F {
  #[new]
  #[pyo3(signature = (kappa, theta, sigma, theta_r, a, sigma1, sigma2, b, rho, n, s0=None, v0=None, r0=None, t=None, seed=None, dtype=None, scheme="euler"))]
  fn new(
    kappa: f64,
    theta: f64,
    sigma: f64,
    theta_r: pyo3::Py<pyo3::PyAny>,
    a: f64,
    sigma1: f64,
    sigma2: f64,
    b: f64,
    rho: numpy::PyReadonlyArray2<'_, f64>,
    n: usize,
    s0: Option<f64>,
    v0: Option<f64>,
    r0: Option<f64>,
    t: Option<f64>,
    seed: Option<u64>,
    dtype: Option<&str>,
    scheme: &str,
  ) -> pyo3::PyResult<Self> {
    let args = Args {
      kappa,
      theta,
      sigma,
      a,
      sigma1,
      sigma2,
      b,
      rho: rho.as_array().to_owned(),
      n,
      s0,
      v0,
      r0,
      t,
    };
    stochastic_rs_distributions::python::value_error_on_panic(|| -> pyo3::PyResult<Self> {
      let qe = match scheme.to_ascii_lowercase().as_str() {
        "euler" => false,
        "qe" => true,
        other => {
          return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "scheme must be 'euler' or 'qe', got '{other}'"
          )));
        }
      };
      let mut s = Self {
        inner_f32: None,
        inner_f64: None,
        seeded_f32: None,
        seeded_f64: None,
        qe: None,
      };
      match (seed, dtype.unwrap_or("f64")) {
        (Some(sd), "f32") => {
          s.seeded_f32 = Some(build(
            &args,
            theta_r,
            crate::python_device::SharedSeed::new(sd),
          ));
        }
        (Some(sd), "f64") => {
          s.seeded_f64 = Some(build(
            &args,
            theta_r,
            crate::python_device::SharedSeed::new(sd),
          ));
        }
        (None, "f32") => s.inner_f32 = Some(build(&args, theta_r, Unseeded)),
        (None, "f64") => s.inner_f64 = Some(build(&args, theta_r, Unseeded)),
        (_, other) => {
          return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "dtype must be 'f32' or 'f64', got '{other}'"
          )));
        }
      }
      if qe {
        s.qe = Some(PyHestonHullWhite2FQe {
          inner_f32: s.inner_f32.take().map(HestonHullWhite2F::qe),
          inner_f64: s.inner_f64.take().map(HestonHullWhite2F::qe),
          seeded_f32: s.seeded_f32.take().map(HestonHullWhite2F::qe),
          seeded_f64: s.seeded_f64.take().map(HestonHullWhite2F::qe),
        });
      }
      Ok(s)
    })?
  }

  /// One path as a `(5, n)` array: spot, variance, short rate, auxiliary factor, integrated short rate.
  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use numpy::IntoPyArray;
      use pyo3::IntoPyObjectExt;

      use crate::traits::ProcessExt;
      py_hybrid_dispatch!(self, |inner| inner
        .sample()
        .into_pyarray(py)
        .into_py_any(py)
        .unwrap())
    })
  }

  /// `m` independent paths as an `(m, 5, n)` array. The GIL is released while the paths are
  /// generated; `theta_r` re-acquires it, so it is called from the rayon workers one at a time.
  fn sample_par<'py>(
    &self,
    py: pyo3::Python<'py>,
    m: usize,
  ) -> pyo3::PyResult<pyo3::Py<pyo3::PyAny>> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use ndarray::Array3;
      use numpy::IntoPyArray;
      use pyo3::IntoPyObjectExt;

      use crate::traits::ProcessExt;
      py_hybrid_dispatch!(self, |inner| {
        let paths = py.detach(|| inner.sample_par(m));
        let n = paths.first().map_or(0, |p| p.ncols());
        let mut out = Array3::zeros((m, ROWS, n));
        for (i, path) in paths.iter().enumerate() {
          out.index_axis_mut(ndarray::Axis(0), i).assign(path);
        }
        out.into_pyarray(py).into_py_any(py).unwrap()
      })
    })
  }
}

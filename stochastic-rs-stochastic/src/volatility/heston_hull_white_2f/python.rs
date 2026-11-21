use super::*;

/// `theta_r` is a Python callable of time, `rho` a `4 × 4` array in the order `(x, v, r, u)`,
/// `scheme` `"euler"` (default) or `"qe"`, Andersen's (2008) quadratic-exponential variance step.
#[doc(hidden)]
#[pyo3::prelude::pyclass]
pub struct PyHestonHullWhite2F {
  inner_f32: Option<HestonHullWhite2F<f32>>,
  inner_f64: Option<HestonHullWhite2F<f64>>,
  seeded_f32: Option<HestonHullWhite2F<f32, Euler, crate::python_device::SharedSeed>>,
  seeded_f64: Option<HestonHullWhite2F<f64, Euler, crate::python_device::SharedSeed>>,
  /// The device the class samples on, chosen at construction.
  device: crate::python_device::Device,
  qe: Option<PyHestonHullWhite2FQe>,
}

/// The [`AndersenQe`] twins of [`PyHestonHullWhite2F`]'s slots, named for the dispatch macro.
struct PyHestonHullWhite2FQe {
  inner_f32: Option<HestonHullWhite2F<f32, AndersenQe>>,
  inner_f64: Option<HestonHullWhite2F<f64, AndersenQe>>,
  seeded_f32: Option<HestonHullWhite2F<f32, AndersenQe, crate::python_device::SharedSeed>>,
  seeded_f64: Option<HestonHullWhite2F<f64, AndersenQe, crate::python_device::SharedSeed>>,
  device: crate::python_device::Device,
}

macro_rules! py_hybrid_dispatch {
  ($dispatch:ident, $self:expr, |$p:ident| $body:expr) => {
    match &$self.qe {
      Some(qe) => $dispatch!(qe, |$p| $body),
      None => $dispatch!($self, |$p| $body),
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
    f(args.s0),
    f(args.v0),
    f(args.kappa),
    f(args.theta),
    f(args.sigma),
    Fn1D::Py(theta_r),
    f(args.a),
    f(args.sigma1),
    f(args.sigma2),
    f(args.b),
    args.rho.mapv(f),
    args.r0.map(f),
    args.n,
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
  s0: f64,
  v0: f64,
  r0: Option<f64>,
  t: Option<f64>,
}

#[pyo3::prelude::pymethods]
impl PyHestonHullWhite2F {
  #[new]
  #[pyo3(signature = (kappa, theta, sigma, theta_r, a, sigma1, sigma2, b, rho, n, s0, v0, r0=None, t=None, seed=None, dtype=None, device=None, scheme="euler"))]
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
    s0: f64,
    v0: f64,
    r0: Option<f64>,
    t: Option<f64>,
    seed: Option<u64>,
    dtype: Option<&str>,
    device: Option<&str>,
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
      let dtype = dtype.unwrap_or("f64");
      if !matches!(dtype, "f32" | "f64") {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
          "dtype must be 'f32' or 'f64', got '{dtype}'"
        )));
      }
      let device = crate::python_device::Device::parse(device, dtype)?;
      let mut s = Self {
        inner_f32: None,
        inner_f64: None,
        seeded_f32: None,
        seeded_f64: None,
        device,
        qe: None,
      };
      match (seed, dtype) {
        (Some(sd), "f32") => {
          s.seeded_f32 = Some(build(
            &args,
            theta_r,
            crate::python_device::SharedSeed::new(sd),
          ));
        }
        (Some(sd), _) => {
          s.seeded_f64 = Some(build(
            &args,
            theta_r,
            crate::python_device::SharedSeed::new(sd),
          ));
        }
        (None, "f32") => s.inner_f32 = Some(build(&args, theta_r, Unseeded)),
        (None, _) => s.inner_f64 = Some(build(&args, theta_r, Unseeded)),
      }
      if qe {
        s.qe = Some(PyHestonHullWhite2FQe {
          inner_f32: s.inner_f32.take().map(HestonHullWhite2F::qe),
          inner_f64: s.inner_f64.take().map(HestonHullWhite2F::qe),
          seeded_f32: s.seeded_f32.take().map(HestonHullWhite2F::qe),
          seeded_f64: s.seeded_f64.take().map(HestonHullWhite2F::qe),
          device: s.device,
        });
      }
      Ok(s)
    })?
  }

  /// Why a device kernel cannot carry this configuration, if it cannot; the answer depends on
  /// the configuration alone, whatever `device=` was passed.
  fn device_fallback(&self) -> Option<&'static str> {
    use crate::traits::ProcessExt;
    py_hybrid_dispatch!(py_dispatch, self, |inner| inner.device_fallback())
  }

  /// Whether this configuration runs on a device kernel: the absence of a
  /// [`device_fallback`](Self::device_fallback) reason.
  fn device_ready(&self) -> bool {
    self.device_fallback().is_none()
  }

  /// One path as five arrays: spot, variance, short rate, auxiliary factor, integrated short rate.
  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::PyResult<[pyo3::Py<pyo3::PyAny>; 5]> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use numpy::IntoPyArray;
      use pyo3::IntoPyObjectExt;

      use crate::traits::ProcessExt;
      py_hybrid_dispatch!(py_device_dispatch, self, |inner| {
        let path = inner.sample();
        [path.s, path.v, path.r, path.u, path.integrated_rate]
          .map(|row| row.into_pyarray(py).into_py_any(py).unwrap())
      })
    })
  }

  /// `m` independent paths as five `(m, n)` arrays, in `sample`'s order. The GIL is released
  /// meanwhile; `theta_r` re-acquires it, so the rayon workers call it one at a time.
  fn sample_par<'py>(
    &self,
    py: pyo3::Python<'py>,
    m: usize,
  ) -> pyo3::PyResult<[pyo3::Py<pyo3::PyAny>; 5]> {
    stochastic_rs_distributions::python::runtime_error_on_panic(|| {
      use numpy::IntoPyArray;
      use pyo3::IntoPyObjectExt;

      use crate::traits::ProcessExt;
      py_hybrid_dispatch!(py_device_dispatch, self, |inner| {
        let paths = py.detach(|| inner.sample_par(m));
        let n = paths.first().map_or(0, |p| p.s.len());
        let mut out = std::array::from_fn::<_, 5, _>(|_| Array2::zeros((m, n)));
        for (i, path) in paths.iter().enumerate() {
          let rows = [&path.s, &path.v, &path.r, &path.u, &path.integrated_rate];
          for (matrix, row) in out.iter_mut().zip(rows) {
            matrix.row_mut(i).assign(row);
          }
        }
        out.map(|matrix| matrix.into_pyarray(py).into_py_any(py).unwrap())
      })
    })
  }
}

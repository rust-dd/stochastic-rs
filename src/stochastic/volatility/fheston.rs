//! # Fheston
//!
//! $$
//! dS_t=\mu S_tdt+\sqrt{v_t}S_tdW_t,\quad dv_t=\kappa(\theta-v_t)dt+\xi\sqrt{v_t}dB_t^H
//! $$
//!
use ndarray::Array1;
use statrs::function::gamma::gamma;

use crate::stochastic::noise::gn::Gn;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct RoughHeston<T: FloatExt> {
  /// Hurst exponent controlling roughness and long-memory.
  pub hurst: T,
  /// Initial variance/volatility level.
  pub v0: Option<T>,
  /// Long-run target level / model location parameter.
  pub theta: T,
  /// Mean-reversion speed parameter.
  pub kappa: T,
  /// Volatility-of-volatility / tail-thickness parameter.
  pub nu: T,
  /// Model coefficient for factor 1.
  pub c1: Option<T>,
  /// Model coefficient for factor 2.
  pub c2: Option<T>,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  gn: Gn<T>,
}

impl<T: FloatExt> RoughHeston<T> {
  pub fn new(
    hurst: T,
    v0: Option<T>,
    theta: T,
    kappa: T,
    nu: T,
    c1: Option<T>,
    c2: Option<T>,
    t: Option<T>,
    n: usize,
  ) -> Self {
    RoughHeston {
      hurst,
      v0,
      theta,
      kappa,
      nu,
      c1,
      c2,
      t,
      n,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for RoughHeston<T> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();
    let gn = self.gn.sample();
    let mut yt = Array1::<T>::zeros(self.n);
    let mut zt = Array1::<T>::zeros(self.n);
    let mut sigma_tilde2 = Array1::<T>::zeros(self.n);
    let mut v2 = Array1::zeros(self.n);

    let v0_sq = self.v0.unwrap_or(T::one()).powi(2);
    yt[0] = v0_sq;
    zt[0] = T::zero();
    sigma_tilde2[0] = v0_sq;
    v2[0] = v0_sq;
    let g = gamma(self.hurst.to_f64().unwrap() - 0.5);

    for i in 1..self.n {
      let t = dt * T::from_usize_(i);
      yt[i] = self.theta + (yt[i - 1] - self.theta) * (-self.kappa * dt).exp();
      zt[i] = zt[i - 1] * (-self.kappa * dt).exp()
        + sigma_tilde2[i - 1].max(T::zero()).sqrt() * gn[i - 1];

      // CIR process: sigma_tilde^2 = Y_t + nu * Z_t
      sigma_tilde2[i] = yt[i] + self.nu * zt[i];

      let integral = (0..i)
        .map(|j| {
          let tj = T::from_usize_(j) * dt;
          ((t - tj).powf(self.hurst - T::from_f64_fast(0.5)) * zt[j]) * dt
        })
        .sum::<T>();

      v2[i] = yt[i]
        + self.c1.unwrap_or(T::one()) * self.nu * zt[i]
        + self.c2.unwrap_or(T::one()) * self.nu * integral / T::from_f64_fast(g);
    }

    v2
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyRoughHeston {
  inner_f32: Option<RoughHeston<f32>>,
  inner_f64: Option<RoughHeston<f64>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyRoughHeston {
  #[new]
  #[pyo3(signature = (hurst, theta, kappa, nu, n, v0=None, c1=None, c2=None, t=None, dtype=None))]
  fn new(
    hurst: f64,
    theta: f64,
    kappa: f64,
    nu: f64,
    n: usize,
    v0: Option<f64>,
    c1: Option<f64>,
    c2: Option<f64>,
    t: Option<f64>,
    dtype: Option<&str>,
  ) -> Self {
    match dtype.unwrap_or("f64") {
      "f32" => Self {
        inner_f32: Some(RoughHeston::new(
          hurst as f32,
          v0.map(|v| v as f32),
          theta as f32,
          kappa as f32,
          nu as f32,
          c1.map(|v| v as f32),
          c2.map(|v| v as f32),
          t.map(|v| v as f32),
          n,
        )),
        inner_f64: None,
      },
      _ => Self {
        inner_f32: None,
        inner_f64: Some(RoughHeston::new(hurst, v0, theta, kappa, nu, c1, c2, t, n)),
      },
    }
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    if let Some(ref inner) = self.inner_f64 {
      inner.sample().into_pyarray(py).into_py_any(py).unwrap()
    } else if let Some(ref inner) = self.inner_f32 {
      inner.sample().into_pyarray(py).into_py_any(py).unwrap()
    } else {
      unreachable!()
    }
  }

  fn sample_par<'py>(&self, py: pyo3::Python<'py>, m: usize) -> pyo3::Py<pyo3::PyAny> {
    use numpy::ndarray::Array2;
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    if let Some(ref inner) = self.inner_f64 {
      let paths = inner.sample_par(m);
      let n = paths[0].len();
      let mut result = Array2::<f64>::zeros((m, n));
      for (i, path) in paths.iter().enumerate() {
        result.row_mut(i).assign(path);
      }
      result.into_pyarray(py).into_py_any(py).unwrap()
    } else if let Some(ref inner) = self.inner_f32 {
      let paths = inner.sample_par(m);
      let n = paths[0].len();
      let mut result = Array2::<f32>::zeros((m, n));
      for (i, path) in paths.iter().enumerate() {
        result.row_mut(i).assign(path);
      }
      result.into_pyarray(py).into_py_any(py).unwrap()
    } else {
      unreachable!()
    }
  }
}
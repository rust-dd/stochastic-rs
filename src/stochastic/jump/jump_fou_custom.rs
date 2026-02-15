use ndarray::Array1;
use rand_distr::Distribution;

use crate::stochastic::noise::fgn::FGN;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct JumpFOUCustom<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub hurst: T,
  pub theta: T,
  pub mu: T,
  pub sigma: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub jump_times: D,
  pub jump_sizes: D,
  fgn: FGN<T>,
}

impl<T, D> JumpFOUCustom<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub fn new(
    hurst: T,
    theta: T,
    mu: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    jump_times: D,
    jump_sizes: D,
  ) -> Self {
    Self {
      hurst,
      mu,
      sigma,
      theta,
      n,
      x0,
      t,
      jump_times,
      jump_sizes,
      fgn: FGN::new(hurst, n - 1, t),
    }
  }
}

impl<T, D> ProcessExt<T> for JumpFOUCustom<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.fgn.dt();
    let fgn = &self.fgn.sample();

    let mut jump_fou = Array1::<T>::zeros(self.n);
    jump_fou[0] = self.x0.unwrap_or(T::zero());
    let mut jump_times = Array1::<T>::zeros(self.n);
    jump_times.mapv_inplace(|_| self.jump_times.sample(&mut rand::rng()));

    for i in 1..self.n {
      let t = T::from_usize_(i) * dt;
      // check if t is a jump time
      let mut jump = T::zero();
      if jump_times[i] < t && t - dt <= jump_times[i] {
        jump = self.jump_sizes.sample(&mut rand::rng());
      }

      jump_fou[i] = jump_fou[i - 1]
        + self.theta * (self.mu - jump_fou[i - 1]) * dt
        + self.sigma * fgn[i - 1]
        + jump;
    }

    jump_fou
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyJumpFOUCustom {
  inner_f32: Option<JumpFOUCustom<f32, crate::traits::CallableDist<f32>>>,
  inner_f64: Option<JumpFOUCustom<f64, crate::traits::CallableDist<f64>>>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyJumpFOUCustom {
  #[new]
  #[pyo3(signature = (hurst, theta, mu, sigma, jump_times, jump_sizes, n, x0=None, t=None, dtype=None))]
  fn new(
    hurst: f64,
    theta: f64,
    mu: f64,
    sigma: f64,
    jump_times: pyo3::Py<pyo3::PyAny>,
    jump_sizes: pyo3::Py<pyo3::PyAny>,
    n: usize,
    x0: Option<f64>,
    t: Option<f64>,
    dtype: Option<&str>,
  ) -> Self {
    match dtype.unwrap_or("f64") {
      "f32" => Self {
        inner_f32: Some(JumpFOUCustom::new(
          hurst as f32, theta as f32, mu as f32, sigma as f32, n,
          x0.map(|v| v as f32), t.map(|v| v as f32),
          crate::traits::CallableDist::new(jump_times),
          crate::traits::CallableDist::new(jump_sizes),
        )),
        inner_f64: None,
      },
      _ => Self {
        inner_f32: None,
        inner_f64: Some(JumpFOUCustom::new(
          hurst, theta, mu, sigma, n, x0, t,
          crate::traits::CallableDist::new(jump_times),
          crate::traits::CallableDist::new(jump_sizes),
        )),
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
    use numpy::IntoPyArray;
    use numpy::ndarray::Array2;
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

use ndarray::Array1;
use ndarray::Axis;
use rand::rng;
use rand_distr::Distribution;

use super::customjt::CustomJt;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct CompoundCustom<T, D1, D2>
where
  T: FloatExt,
  D1: Distribution<T> + Send + Sync,
  D2: Distribution<T> + Send + Sync,
{
  pub n: Option<usize>,
  pub t_max: Option<T>,
  pub m: Option<usize>,
  pub jumps_distribution: D1,
  pub jump_times_distribution: D2,
  pub customjt: CustomJt<T, D2>,
}

impl<T, D1, D2> CompoundCustom<T, D1, D2>
where
  T: FloatExt,
  D1: Distribution<T> + Send + Sync,
  D2: Distribution<T> + Send + Sync,
{
  pub fn new(
    n: Option<usize>,
    t_max: Option<T>,
    m: Option<usize>,
    jumps_distribution: D1,
    jump_times_distribution: D2,
    customjt: CustomJt<T, D2>,
  ) -> Self {
    if n.is_none() && t_max.is_none() {
      panic!("n or t_max must be provided");
    }

    Self {
      n,
      t_max,
      m,
      jumps_distribution,
      jump_times_distribution,
      customjt,
    }
  }
}

impl<T, D1, D2> ProcessExt<T> for CompoundCustom<T, D1, D2>
where
  T: FloatExt,
  D1: Distribution<T> + Send + Sync,
  D2: Distribution<T> + Send + Sync,
{
  type Output = [Array1<T>; 3];

  fn sample(&self) -> Self::Output {
    let p = self.customjt.sample();
    let mut jumps = Array1::<T>::zeros(self.n.unwrap_or(p.len()));
    for i in 1..p.len() {
      jumps[i] = self.jumps_distribution.sample(&mut rng());
    }

    let mut cum_jupms = jumps.clone();
    cum_jupms.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);

    [p, cum_jupms, jumps]
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyCompoundCustom {
  inner: CompoundCustom<f64, crate::traits::CallableDist, crate::traits::CallableDist>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyCompoundCustom {
  #[new]
  #[pyo3(signature = (jumps_distribution, jump_times_distribution, n=None, t_max=None, m=None))]
  fn new(
    jumps_distribution: pyo3::Py<pyo3::PyAny>,
    jump_times_distribution: pyo3::Py<pyo3::PyAny>,
    n: Option<usize>,
    t_max: Option<f64>,
    m: Option<usize>,
  ) -> Self {
    let (jt_dist, customjt_dist) = pyo3::Python::attach(|py| {
      let a = jump_times_distribution.clone_ref(py);
      let b = jump_times_distribution;
      (
        crate::traits::CallableDist::new(a),
        crate::traits::CallableDist::new(b),
      )
    });
    let customjt = CustomJt::new(n, t_max, customjt_dist);
    Self {
      inner: CompoundCustom::new(
        n,
        t_max,
        m,
        crate::traits::CallableDist::new(jumps_distribution),
        jt_dist,
        customjt,
      ),
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
    let [p, cum, j] = self.inner.sample();
    (
      p.into_pyarray(py).into_py_any(py).unwrap(),
      cum.into_pyarray(py).into_py_any(py).unwrap(),
      j.into_pyarray(py).into_py_any(py).unwrap(),
    )
  }
}

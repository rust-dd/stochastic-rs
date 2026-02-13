use ndarray::Array1;
use rand_distr::Distribution;

use crate::stochastic::noise::gn::Gn;
use crate::stochastic::process::cpoisson::CompoundPoisson;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

pub struct Merton<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub alpha: T,
  pub sigma: T,
  pub lambda: T,
  pub theta: T,
  pub n: usize,
  pub x0: Option<T>,
  pub t: Option<T>,
  pub cpoisson: CompoundPoisson<T, D>,
  gn: Gn<T>,
}

impl<T, D> Merton<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  pub fn new(
    alpha: T,
    sigma: T,
    lambda: T,
    theta: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    cpoisson: CompoundPoisson<T, D>,
  ) -> Self {
    Self {
      alpha,
      sigma,
      lambda,
      theta,
      n,
      x0,
      t,
      cpoisson,
      gn: Gn::new(n - 1, t),
    }
  }
}

impl<T, D> ProcessExt<T> for Merton<T, D>
where
  T: FloatExt,
  D: Distribution<T> + Send + Sync,
{
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    let dt = self.gn.dt();
    let gn = &self.gn.sample();

    let mut merton = Array1::<T>::zeros(self.n);
    merton[0] = self.x0.unwrap_or(T::zero());

    for i in 1..self.n {
      let [.., jumps] = self.cpoisson.sample();

      merton[i] = merton[i - 1]
        + (self.alpha
          - self.sigma.powf(T::from_usize(2).unwrap()) / T::from_usize(2).unwrap()
          - self.lambda * self.theta)
          * dt
        + self.sigma * gn[i - 1]
        + jumps.sum();
    }

    merton
  }
}

#[cfg(feature = "python")]
#[pyo3::prelude::pyclass]
pub struct PyMerton {
  inner: Merton<f64, crate::traits::CallableDist>,
}

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl PyMerton {
  #[new]
  #[pyo3(signature = (alpha, sigma, lambda_, theta, distribution, n, x0=None, t=None))]
  fn new(
    alpha: f64,
    sigma: f64,
    lambda_: f64,
    theta: f64,
    distribution: pyo3::Py<pyo3::PyAny>,
    n: usize,
    x0: Option<f64>,
    t: Option<f64>,
  ) -> Self {
    use crate::stochastic::process::poisson::Poisson;
    let cpoisson = CompoundPoisson::new(
      crate::traits::CallableDist::new(distribution),
      Poisson::new(lambda_, Some(n), t),
    );
    Self {
      inner: Merton::new(alpha, sigma, lambda_, theta, n, x0, t, cpoisson),
    }
  }

  fn sample<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Py<pyo3::PyAny> {
    use numpy::IntoPyArray;
    use pyo3::IntoPyObjectExt;

    use crate::traits::ProcessExt;
    self
      .inner
      .sample()
      .into_pyarray(py)
      .into_py_any(py)
      .unwrap()
  }
}
